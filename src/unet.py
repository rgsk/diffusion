"""U-Net eps-predictor: down path, bottleneck, up path with skip concatenation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from resblock import ResBlock
from timestep_embedding import timestep_embedding


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Nearest x2 then a conv -- transposed conv here gives checkerboard artifacts."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Predicts eps from (x_t, t). Output conv is zero-init, so a fresh net
    predicts exactly zero noise and the MSE loss starts at E[eps²] = 1."""

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        mults: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 1,
        time_dim: int = 256,
        groups: int = 8,
    ):
        super().__init__()
        self.base, self.mults = base, mults
        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        ch, skip_chs = base, [base]  # channel counts the up path will concat back
        self.down = nn.ModuleList()
        for i, m in enumerate(mults):
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, base * m, time_dim, groups))
                ch = base * m
                skip_chs.append(ch)
            if i < len(mults) - 1:
                self.down.append(Downsample(ch))
                skip_chs.append(ch)

        self.mid1 = ResBlock(ch, ch, time_dim, groups)
        self.mid2 = ResBlock(ch, ch, time_dim, groups)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(num_res_blocks + 1):  # +1 consumes the downsample's skip
                self.up.append(
                    ResBlock(ch + skip_chs.pop(), base * m, time_dim, groups)
                )
                ch = base * m
            if i > 0:
                self.up.append(Upsample(ch))
        assert not skip_chs, skip_chs

        self.out_norm = nn.GroupNorm(groups, ch)
        self.act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        d = 2 ** (len(self.mults) - 1)
        assert x.shape[-1] % d == 0 and x.shape[-2] % d == 0, f"H,W must divide {d}"
        temb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.conv_in(x)
        hs = [h]
        for m in self.down:
            h = m(h, temb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        h = self.mid2(self.mid1(h, temb), temb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], dim=1), temb)
            else:
                h = m(h)
        assert not hs, len(hs)
        return self.out_conv(self.act(self.out_norm(h)))


if __name__ == "__main__":
    torch.manual_seed(0)
    net = UNet()
    B = 4
    x = torch.randn(B, 1, 28, 28)
    t = torch.randint(0, 1000, (B,))
    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    # 1. eps has the shape of x, and every skip is consumed (asserts in forward)
    assert net(x, t).shape == x.shape
    assert net(torch.randn(1, 1, 28, 28), t[:1]).shape == (1, 1, 28, 28)

    # 2. zero-init output: a fresh net predicts no noise at all, so the MSE
    #    against eps starts at exactly E[eps²] = 1 -- the number to expect on
    #    step 0 of any training run
    assert torch.equal(net(x, t), torch.zeros_like(x))
    eps = torch.randn(B, 1, 28, 28)
    print(f"init loss: {F.mse_loss(net(x, t), eps):.4f}")
    assert abs(F.mse_loss(net(x, t), eps).item() - eps.square().mean().item()) < 1e-6

    # 3. resolutions must line up or the skip concat is wrong. 28 -> 14 -> 7 works;
    #    an odd size divides to a resolution the up path can't rebuild
    try:
        net(torch.randn(B, 1, 30, 30), t)
        raise SystemExit("guard missing")
    except AssertionError:
        pass

    # 4. it learns: overfit 8 MNIST digits, loss must fall well below the 1.0 it
    #    starts at. Every check above passes on a net that cannot train at all
    from torchvision import datasets, transforms

    from forward_process import ForwardProcess
    from utils import repo_root

    ds = datasets.MNIST(
        root=repo_root() / "data",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x0 = torch.stack([ds[i][0] for i in range(8)]).to(dev)
    fp = ForwardProcess().to(dev)
    net = UNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    for step in range(301):
        ti = fp.sample_t(8, dev)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(net(fp.q_sample(x0, ti, noise), ti), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    assert loss.item() < 0.5

    # 5. t reaches the output, per sample -- on the trained net, because at init
    #    every ResBlock's conv2 is zero and no t can reach anything
    x, t = x0, fp.sample_t(8, dev)
    a = net(x, t)
    t2 = t.clone()
    t2[0] = (t[0] + 500) % 1000
    b = net(x, t2)
    print(f"row 0 moved by {(a[0] - b[0]).abs().max():.4f}")
    assert (a[0] - b[0]).abs().max() > 1e-4
    assert torch.equal(a[1:], b[1:])  # rows you didn't touch are untouched, exactly

    print("ok")
