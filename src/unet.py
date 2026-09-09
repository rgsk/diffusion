"""U-Net eps-predictor: down path, bottleneck, up path with skip concatenation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from attention import Attention
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
    """Predicts eps from (x_t, t), and from a class label when num_classes is set.
    Output conv is zero-init, so a fresh net predicts exactly zero noise and the
    MSE loss starts at E[eps²] = 1."""

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        mults: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 1,
        time_dim: int = 256,
        groups: int = 8,
        num_classes: int | None = None,
        attention: bool = False,
    ):
        super().__init__()
        self.base, self.mults = base, mults
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        if num_classes is not None:
            # one extra row: index num_classes is the null label. Unused here, but
            # CFG needs the same weights to have seen it, so reserve it before
            # training rather than retrofitting it after.
            self.null_label = num_classes
            self.label_emb = nn.Embedding(num_classes + 1, time_dim)
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
        # bottleneck only: at 28x28 self-attention is 784² pairs per head, and the
        # 7x7 bottleneck already carries what the whole image contributed
        self.mid_attn = Attention(ch, groups=groups) if attention else None
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

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        d = 2 ** (len(self.mults) - 1)
        assert x.shape[-1] % d == 0 and x.shape[-2] % d == 0, f"H,W must divide {d}"
        temb = self.time_mlp(timestep_embedding(t, self.base))
        if self.num_classes is None:
            assert y is None, "unconditional net was handed a label"
        else:
            # silently dropping y would train an unconditional model that looks fine
            assert y is not None, "conditional net needs y (net.null_label for none)"
            assert y.shape == t.shape, (y.shape, t.shape)
            assert 0 <= int(y.min()) and int(y.max()) <= self.num_classes, y
            temb = temb + self.label_emb(y)  # same footing as t, one vector per sample
        h = self.conv_in(x)
        hs = [h]
        for m in self.down:
            h = m(h, temb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        h = self.mid1(h, temb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid2(h, temb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], dim=1), temb)
            else:
                h = m(h)
        assert not hs, len(hs)
        return self.out_conv(self.act(self.out_norm(h)))


class Conditioned(nn.Module):
    """Freezes y so a conditional net still presents the (x, t) -> eps interface
    the samplers call. CFG wraps the same way, with two calls instead of one."""

    def __init__(self, net: UNet, y: Tensor):
        super().__init__()
        self.net, self.y = net, y

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(x, t, self.y)


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

    # 5. t reaches the output, per sample. Show the vacuous case first: on a
    #    fresh net the output conv is zero, so both t give exactly zero and the
    #    check below would pass on a net that never reads t at all. That is why
    #    this test has to run after training, not before
    x, t = x0, fp.sample_t(8, dev)
    t2 = t.clone()
    t2[0] = (t[0] + 500) % 1000
    fresh = UNet().to(dev)
    moved_fresh = (fresh(x, t) - fresh(x, t2)).abs().max().item()
    print(f"row 0 moved by {moved_fresh:.4f} (fresh net)")
    assert moved_fresh == 0.0

    a, b = net(x, t), net(x, t2)
    print(f"row 0 moved by {(a[0] - b[0]).abs().max():.4f} (trained)")
    assert (a[0] - b[0]).abs().max() > 1e-4
    assert torch.equal(a[1:], b[1:])  # rows you didn't touch are untouched, exactly

    # 6. the label guards. Each of these is a bug that trains to a plausible loss
    #    and only shows up as samples that ignore what you asked for
    cnet = UNet(num_classes=10)
    xc, tc = torch.randn(B, 1, 28, 28), torch.randint(0, 1000, (B,))  # cpu; t is cuda
    yc4 = torch.randint(0, 10, (B,))
    assert cnet.null_label == 10 and cnet.label_emb.num_embeddings == 11
    for bad in (
        lambda: UNet()(xc, tc, yc4),  # label handed to an unconditional net
        lambda: cnet(xc, tc),  # no label handed to a conditional one
        lambda: cnet(xc, tc, yc4[:2]),  # y not batched alongside t
        lambda: cnet(xc, tc, yc4 + 11),  # past the null row
    ):
        try:
            bad()
            raise SystemExit("guard missing")
        except AssertionError:
            pass

    # 7. Conditioned is exactly the frozen-y call, so a sampler driving it sees
    #    what a direct call would give
    assert torch.equal(Conditioned(cnet, yc4)(xc, tc), cnet(xc, tc, yc4))

    # 8. THE property: y carries information. Overfit one digit per class, then
    #    score the same x_t with the right label and with a wrong one. A label
    #    that is wired up but ignored -- added to temb and then averaged away,
    #    say -- passes every check above and fails only here.
    per_class = {}
    for img, lab in ds:
        per_class.setdefault(lab, img)
        if len(per_class) == 10:
            break
    x0 = torch.stack([per_class[c] for c in range(10)]).to(dev)
    yc = torch.arange(10, device=dev)
    net = UNet(num_classes=10).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    for step in range(801):
        ti = fp.sample_t(10, dev)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(net(fp.q_sample(x0, ti, noise), ti, yc), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    right = wrong = 0.0
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(200):  # average over t and noise; one draw is far too noisy
            ti = fp.sample_t(10, dev)
            noise = torch.randn_like(x0)
            xt = fp.q_sample(x0, ti, noise)
            right += F.mse_loss(net(xt, ti, yc), noise).item()
            wrong += F.mse_loss(net(xt, ti, yc.roll(1)), noise).item()
    right, wrong = right / 200, wrong / 200
    print(f"eps-MSE  right label {right:.4f}   wrong label {wrong:.4f}")
    assert wrong > right * 2  # a net ignoring y scores these identically (obs. 4x)

    print("ok")
