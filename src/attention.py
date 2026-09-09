"""Attention: every position reads every other position, in one layer.

A conv sees 3x3. Information crosses the image only by travelling down to the
bottleneck and back, so a plain U-Net has no cheap way to make one corner agree
with another. Attention is that path.

Self-attention takes Q, K and V from the image. Cross-attention takes Q from the
image and K, V from conditioning tokens -- the same block, a different source
for KV, and the mechanism that binds an attribute to an object ("a red 3 in the
top left"), which adding one pooled vector to `temb` cannot do.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    """[B,C,H,W] -> [B,C,H,W]. `context_dim=None` is self-attention; set it and
    forward() requires context [B, L, context_dim].

    Output projection is zero-init, like `ResBlock.conv2` and `UNet.out_conv`, so
    a fresh block is exactly its skip and adding attention cannot make a net
    worse at step 0."""

    def __init__(
        self,
        ch: int,
        heads: int = 4,
        groups: int = 8,
        context_dim: int | None = None,
    ):
        super().__init__()
        assert ch % heads == 0, (ch, heads)
        self.heads, self.context_dim = heads, context_dim
        self.norm = nn.GroupNorm(groups, ch)
        self.to_q = nn.Conv2d(ch, ch, 1)
        # 1x1 conv over the image for self, Linear over the token sequence for cross
        self.to_kv = (
            nn.Conv2d(ch, 2 * ch, 1)
            if context_dim is None
            else nn.Linear(context_dim, 2 * ch)
        )
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _split(self, t: Tensor) -> Tensor:
        """[B, N, C] -> [B, heads, N, C/heads]"""
        B, N, C = t.shape
        return t.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.to_q(h).reshape(B, C, H * W).transpose(1, 2)  # positions = sequence
        if self.context_dim is None:
            assert context is None, "self-attention block was handed a context"
            kv = self.to_kv(h).reshape(B, 2 * C, H * W).transpose(1, 2)
        else:
            assert context is not None, "cross-attention block needs a context"
            assert context.shape[-1] == self.context_dim, context.shape
            kv = self.to_kv(context)  # [B, L, 2C] -- L is free, unlike H*W
        k, v = kv.chunk(2, dim=-1)
        o = F.scaled_dot_product_attention(*(self._split(z) for z in (q, k, v)))
        o = o.transpose(1, 2).reshape(B, H * W, C).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(o)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, S = 2, 64, 7
    x = torch.randn(B, C, S, S)

    # 1. shape, and the zero-init: a fresh block is exactly its input
    attn = Attention(C)
    assert attn(x).shape == x.shape
    assert torch.equal(attn(x), x)

    # everything below needs the block to actually do something
    def live(**kw):
        a = Attention(C, **kw)
        nn.init.normal_(a.proj.weight, std=0.05)
        return a.eval()

    attn = live()

    # 2. THE property a conv does not have: one pixel reaches every position.
    #    A 3x3 conv would leave the far corner untouched; this must not.
    x2 = x.clone()
    x2[0, :, 0, 0] += 5.0
    d = (attn(x2) - attn(x))[0, :, -1, -1].abs().max()
    print(f"far-corner response to a single-pixel change: {d:.4f}")
    assert d > 1e-3

    # 3. positions are a SET: no positional encoding here, so permuting the
    #    pixels permutes the output identically. This is why the convs stay --
    #    attention alone cannot tell top-left from bottom-right.
    perm = torch.randperm(S * S)
    flat = x.reshape(B, C, S * S)
    permuted = flat[:, :, perm].reshape(B, C, S, S)
    got = attn(permuted).reshape(B, C, S * S)
    want = attn(x).reshape(B, C, S * S)[:, :, perm]
    assert torch.allclose(got, want, atol=1e-5)

    # 4. heads must divide the channels, or the reshape silently mixes them
    try:
        Attention(C, heads=7)
        raise SystemExit("guard missing")
    except AssertionError:
        pass

    # 5. cross-attention: KV comes from tokens, so L is free and the output
    #    depends on what the tokens say
    D, L = 32, 5
    cross = live(context_dim=D)
    ctx = torch.randn(B, L, D)
    assert cross(x, ctx).shape == x.shape
    assert cross(x, torch.randn(B, 11, D)).shape == x.shape  # any sequence length
    assert not torch.allclose(cross(x, ctx), cross(x, torch.randn(B, L, D)))
    # and the two modes are not interchangeable -- a missing context is a bug,
    # not a default
    for bad in (lambda: cross(x), lambda: attn(x, ctx)):
        try:
            bad()
            raise SystemExit("guard missing")
        except AssertionError:
            pass

    print(f"self {sum(p.numel() for p in Attention(C).parameters()) / 1e3:.1f}k params")
    print("ok")
