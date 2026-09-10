"""Attention: every position reads every other position, in one layer.

A conv sees 3x3. Information crosses the image only by travelling down to the
bottleneck and back, so a plain U-Net has no cheap way to make one corner agree
with another. Attention is that path.

Self-attention takes Q, K and V from the image. Cross-attention takes Q from the
image and K, V from conditioning tokens -- the same block, a different source
for KV, and the mechanism that binds an attribute to an object ("a red 3 in the
top left"), which adding one pooled vector to `temb` cannot do.

`coords=True` adds a fixed position code to Q only (`coords.py`), so a query can
say which position is asking. Q only, and not the residual stream: the features
flowing through the U-Net stay translation-equivariant, and it is the question
being asked that becomes position-dependent, not the image.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from coords import coord_embedding


class Attention(nn.Module):
    """[B,C,H,W] -> [B,C,H,W]. `context_dim=None` is self-attention; set it and
    forward() requires context [B, L, context_dim].

    Output projection is zero-init, like `ResBlock.conv2` and `UNet.out_conv`, so
    a fresh block is exactly its skip and adding attention cannot make a net
    worse at step 0.

    `coords=True` adds no parameters, so a run with it and its control have
    identical state dicts and the comparison is the coordinates alone."""

    def __init__(
        self,
        ch: int,
        heads: int = 4,
        groups: int = 8,
        context_dim: int | None = None,
        coords: bool = False,
    ):
        super().__init__()
        assert ch % heads == 0, (ch, heads)
        self.heads, self.context_dim, self.coords = heads, context_dim, coords
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
        # the query alone learns where it is asking from; K, V and the residual
        # are untouched
        q_in = h
        if self.coords:
            q_in = h + coord_embedding(H, W, C, device=h.device).to(h.dtype)
        q = self.to_q(q_in).reshape(B, C, H * W).transpose(1, 2)  # positions = sequence
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

    # 6. coords: the query knows where it is asking from. Test 3 is the control
    #    -- the same block without coords permutes with its input, and this one
    #    must not, or nothing has been added.
    coord = live(context_dim=D, coords=True)
    got = coord(permuted, ctx).reshape(B, C, S * S)
    want = coord(x, ctx).reshape(B, C, S * S)[:, :, perm]
    moved = (got - want).abs().max()
    print(f"coords break permutation equivariance by {moved:.4f}")
    assert moved > 1e-3

    # 7. it costs nothing: no parameters and no state-dict keys, so a run with
    #    coords and its control differ in the flag and in nothing else
    assert [k for k, _ in coord.named_parameters()] == [
        k for k, _ in cross.named_parameters()
    ]
    assert sum(p.numel() for p in coord.parameters()) == sum(
        p.numel() for p in cross.parameters()
    )
    coord.load_state_dict(cross.state_dict())  # and the control's weights load
    # ...which makes this the sharpest statement of what coords do: identical
    # weights, identical input, different answer, entirely because of position
    assert not torch.allclose(coord(x, ctx), cross(x, ctx))

    print(f"self {sum(p.numel() for p in Attention(C).parameters()) / 1e3:.1f}k params")
    print("ok")
