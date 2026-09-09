"""Let the words look at each other, before the image looks at the words.

`binding.py` found cross-attention assigning colours to objects at chance, and
the reason was not the cross-attention: it was what cross-attention had to read.
The context was `token_emb(y) + token_pos`, and `token_pos` trained to 2% of the
embedding norm (`results.md`). So the vector at slot 1 said "red" and the vector
at slot 9 said "blue", with almost nothing saying which clause each came from.
A per-position lookup is only as good as what is there to look up.

This is the missing stage. Self-attention over the token sequence, so each word's
vector is rewritten in terms of the words around it: "red" becomes *red,
modifying the 3, in the first clause* before the U-Net ever attends to it. In
Stable Diffusion this stage is CLIP's text transformer, frozen. Here it is two
blocks, trained end to end with everything else, because the vocabulary is 27
words and there is nothing to pretrain on.

Nothing else changes. The block is the standard pre-norm transformer encoder
layer, and it is the *only* thing standing between the two runs it is meant to
separate.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TextBlock(nn.Module):
    """Pre-norm self-attention + MLP over [B, L, D].

    Both residual branches end in a zero-init projection, as `ResBlock.conv2` and
    `Attention.proj` do, so a fresh block is exactly its input and adding an
    encoder cannot make a net worse at step 0. Training moves it off zero."""

    def __init__(self, dim: int, heads: int = 4, mlp_mult: int = 4):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_mult * dim), nn.GELU(), nn.Linear(mlp_mult * dim, dim)
        )
        for layer in (self.proj, self.mlp[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        q, k, v = (
            t.reshape(B, L, self.heads, D // self.heads).transpose(1, 2)
            for t in self.qkv(self.norm1(x)).chunk(3, dim=-1)
        )
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(o.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.norm2(x))


class TextEncoder(nn.Module):
    """[B, L, D] -> [B, L, D], same length, contextualised.

    Deliberately does not own the embedding table: `UNet` keeps `token_emb` and
    `token_pos` where they were, so a checkpoint trained without an encoder still
    loads against the same state-dict keys and stays available as the control.

    No padding mask. `<pad>` is a learned embedding the encoder can learn to
    attend away from, and the two-object captions have no padding at all -- worth
    knowing, not worth the complexity here."""

    def __init__(self, dim: int, layers: int = 2, heads: int = 4):
        super().__init__()
        assert layers > 0, layers
        self.blocks = nn.ModuleList(TextBlock(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, x.shape
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D = 4, 15, 128
    x = torch.randn(B, L, D)

    # 1. shape survives, any length (the sequence axis is free, as in cross-attn)
    enc = TextEncoder(D)
    assert enc(x).shape == x.shape
    for n in (1, 7, 32):
        assert enc(torch.randn(B, n, D)).shape == (B, n, D)

    # 2. zero-init: a fresh encoder is exactly its input, up to the final
    #    LayerNorm. Check the blocks alone, which is where the residuals are.
    blk = TextBlock(D)
    assert torch.equal(blk(x), x)
    deep = nn.ModuleList(TextBlock(D) for _ in range(6))
    h = x
    for b in deep:
        h = b(h)
    assert torch.equal(h, x)

    # everything below needs the blocks to actually do something
    def live(layers=2):
        e = TextEncoder(D, layers)
        for m in e.modules():
            if isinstance(m, TextBlock):
                nn.init.normal_(m.proj.weight, std=0.05)
                nn.init.normal_(m.mlp[-1].weight, std=0.05)
        return e.eval()

    enc = live()

    # 3. THE property, and the whole reason this file exists: a token's output
    #    depends on the OTHER tokens. Change the word at slot 9 and the vector at
    #    slot 1 must move -- that is "red" learning it modifies the 3 rather than
    #    the 7. A per-token MLP, a positional embedding, and the old
    #    embeddings-only context all fail exactly here.
    x2 = x.clone()
    x2[:, 9] = torch.randn(B, D)
    moved = (enc(x2) - enc(x))[:, 1].abs().max()
    print(f"slot 1 moves by {moved:.4f} when slot 9 changes")
    assert moved > 1e-3
    # and it is not global mush: with one layer, a token still reads every other
    # token, so this is about contextualisation, not depth
    assert (live(1)(x2) - live(1)(x))[:, 1].abs().max() > 1e-3

    # 4. order matters. Self-attention alone is permutation-equivariant -- feed it
    #    a permuted sequence and you get the permuted output, which is a bag with
    #    extra steps. That is fine *here* only because UNet adds token_pos before
    #    calling this; assert the equivariance so the reason positions must be
    #    added upstream is written down rather than assumed.
    perm = torch.randperm(L)
    assert torch.allclose(enc(x[:, perm]), enc(x)[:, perm], atol=1e-5)

    # 5. per sample: one caption must not leak into another in the same batch
    rows = torch.cat([enc(x[i : i + 1]) for i in range(B)])
    print(f"batched vs row-by-row: {(enc(x) - rows).abs().max().item():.2e}")
    assert (enc(x) - rows).abs().max() < 1e-5

    # 6. heads must divide the dim, or the reshape silently mixes them
    try:
        TextBlock(D, heads=7)
        raise SystemExit("guard missing")
    except AssertionError:
        pass

    n = sum(p.numel() for p in TextEncoder(D).parameters())
    print(f"2 layers at dim {D}: {n / 1e3:.0f}k params")
    print("ok")
