"""Time-conditioned residual block: the unit a DDPM U-Net is stacked from."""

import torch
from torch import Tensor, nn


class ResBlock(nn.Module):
    """norm-act-conv twice, with temb added in the middle and a skip around it.
    conv2 is zero-init, so a fresh block is exactly its skip connection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        assert in_ch % groups == 0 and out_ch % groups == 0, (in_ch, out_ch, groups)
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        # 1x1 only when the channel count changes -- otherwise the skip is free
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(temb))[:, :, None, None]  # broadcast over H, W
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, TD = 4, 64, 128
    x = torch.randn(B, C, 28, 28)
    temb = torch.randn(B, TD)

    # 1. shape survives, including the odd sizes the U-Net hits (28 -> 14 -> 7)
    same = ResBlock(C, C, TD)
    assert same(x, temb).shape == x.shape
    assert isinstance(same.skip, nn.Identity)
    wide = ResBlock(C, 2 * C, TD)
    assert wide(x, temb).shape == (B, 2 * C, 28, 28)
    assert isinstance(wide.skip, nn.Conv2d) and wide.skip.kernel_size == (1, 1)
    for hw in (7, 13, 28):
        assert same(torch.randn(B, C, hw, hw), temb).shape == (B, C, hw, hw)

    # 2. zero-init conv2 makes a fresh block exactly its skip: stack 20 of these
    #    and the untrained net is still the identity, not 20 layers of noise
    assert torch.equal(same(x, temb), x)
    assert torch.equal(wide(x, temb), wide.skip(x))
    deep = nn.ModuleList([ResBlock(C, C, TD) for _ in range(20)])
    h = x
    for blk in deep:
        h = blk(h, temb)
    assert torch.equal(h, x)

    # 3. at init only conv2 and skip see gradient -- everything upstream is
    #    multiplied by conv2's zero weight. one step lifts conv2 off zero and
    #    the rest wakes up. (this is why zero-init doesn't strand the block)
    blk = ResBlock(C, C, TD)
    opt = torch.optim.SGD(blk.parameters(), lr=0.1)
    blk(x, temb).square().mean().backward()
    live = {n for n, p in blk.named_parameters() if p.grad.abs().max() > 0}
    print("live at init:", sorted(live))
    assert live == {"conv2.weight", "conv2.bias"}  # skip here is Identity, no params
    opt.step()
    blk.zero_grad()
    blk(x, temb).square().mean().backward()
    dead = {n for n, p in blk.named_parameters() if p.grad.abs().max() == 0}
    print("dead after one step:", sorted(dead) or "none")
    assert not dead

    # 4. temb reaches the output, and per sample -- one t leaking across the
    #    batch is silent, every shape still matches
    a = blk(x, temb)
    b = blk(x, temb.roll(1, dims=0))
    assert (a - b).abs().max() > 1e-4
    rows = torch.cat([blk(x[i : i + 1], temb[i : i + 1]) for i in range(B)])
    print("batched vs row-by-row:", (a - rows).abs().max().item())
    assert (a - rows).abs().max() < 1e-5

    # 5. groups must divide both channel counts
    try:
        ResBlock(C, 60, TD)
        raise SystemExit("guard missing")
    except AssertionError:
        pass

    print("ok")
