"""Classifier-free guidance: one net, two predictions, an amplified difference.

Training drops the label to the null token with probability p, so the same
weights learn eps(x,t,y) and eps(x,t,∅). Sampling scales the gap between them:

    eps = eps_∅ + w · (eps_y - eps_∅)

y is a label or a caption -- `null_label` is whichever "nothing" the net was
built with, and the arithmetic below does not care which.

w=1 is the plain conditional model, w=0 the unconditional one, w>1 extrapolates
past what the model itself believes. What that samples is not p(x|y) but a
sharpened p(x)·p(y|x)^w -- diversity falls as w rises, by construction.
"""

import torch
from torch import Tensor, nn

from unet import UNet


def drop_labels(y: Tensor, p: float, null_label: int) -> Tensor:
    """A fresh y with a fraction p of its rows replaced by the null token. Per
    sample, redrawn every step -- every image is seen both ways over training.

    y is [B] labels or [B, L] tokens; the draw is over the batch dimension either
    way. Drawing over `y.shape` would drop individual *words* from a caption, which
    trains a net to inpaint missing words, not an unconditional one."""
    assert 0.0 <= p <= 1.0, p
    if p == 0.0:
        return y
    mask = torch.rand(y.shape[0], device=y.device) < p
    return torch.where(
        mask.reshape(-1, *([1] * (y.ndim - 1))), torch.full_like(y, null_label), y
    )


class Guided(nn.Module):
    """Freezes (y, w) into the (x, t) -> eps interface the samplers call, the way
    `Conditioned` freezes y. Both branches go through in one batched forward:
    two calls would halve the batch efficiency for identical arithmetic."""

    def __init__(self, net: UNet, y: Tensor, w: float = 1.0):
        super().__init__()
        assert net.null_label is not None, "guidance needs a conditional net"
        self.net, self.y, self.w = net, y, w

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if self.w == 1.0:
            return self.net(x, t, self.y)  # the gap cancels; don't pay for it
        null = torch.full_like(self.y, self.net.null_label)
        eps = self.net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([self.y, null]))
        cond, uncond = eps.chunk(2)
        return uncond + self.w * (cond - uncond)


if __name__ == "__main__":
    from resblock import ResBlock
    from unet import Conditioned

    torch.manual_seed(0)
    C, B = 10, 64
    net = UNet(num_classes=C).eval()
    with torch.no_grad():
        # Two zero-inits stand between the label and the output: out_conv, and
        # every ResBlock's conv2 (which is what temb -- and so y -- feeds). Left
        # alone, a fresh net answers 0 for every label and the checks below would
        # all pass on 0 == 0.
        for m in net.modules():
            if isinstance(m, ResBlock):
                nn.init.normal_(m.conv2.weight, std=0.05)
        nn.init.normal_(net.out_conv.weight, std=0.05)
        net.label_emb.weight[C] = 3.0  # a distinct null row, so ignoring it fails
    x = torch.randn(B, 1, 28, 28)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, C, (B,))

    # 1. drop_labels: the rate is what it says, y is untouched, nothing else moves
    y0 = y.clone()
    assert torch.equal(drop_labels(y, 0.0, C), y)
    assert (drop_labels(y, 1.0, C) == C).all()
    big = torch.randint(0, C, (100_000,))
    rate = (drop_labels(big, 0.1, C) == C).double().mean().item()
    print(f"dropout rate at p=0.1: {rate:.4f}")
    assert abs(rate - 0.1) < 0.005
    assert torch.equal(y, y0), "drop_labels must not edit its input in place"
    kept = drop_labels(y, 0.5, C)
    assert ((kept == y) | (kept == C)).all()  # only ever y or null, never a third

    # 1b. a caption is dropped whole. Per-element masking gives a net trained to
    #     fill in missing words, and the same measured 10% rate, and grids that
    #     look fine -- it fails only as guidance that does nothing.
    seq = torch.randint(1, C, (100_000, 7))
    got = drop_labels(seq, 0.1, 0)
    rows = (got == 0).all(1)
    assert (rows == (got == 0).any(1)).all(), "a caption was dropped word by word"
    print(f"caption dropout rate at p=0.1: {rows.double().mean().item():.4f}")
    assert abs(rows.double().mean().item() - 0.1) < 0.005
    assert torch.equal(got[~rows], seq[~rows])  # kept captions are untouched

    # 2. the endpoints are the two models the trick is built from
    with torch.no_grad():
        cond = Conditioned(net, y)(x, t)
        uncond = net(x, t, torch.full_like(y, C))
        assert torch.equal(Guided(net, y, 1.0)(x, t), cond)  # the w=1 shortcut
        # w!=1 goes through the doubled batch, and a conv over 2B rows does not
        # sum in the same order as over B -- float noise at 1e-6 on eps~0.5, not
        # a difference in what is computed
        assert torch.allclose(Guided(net, y, 0.0)(x, t), uncond, atol=1e-5)

        # 3. THE arithmetic: the batched pass must not scramble which half is
        #    which. A swapped chunk gives cond + w(uncond - cond) and still looks
        #    plausible, so check against the two-call form at w != 1.
        for w in (0.5, 3.0, 15.0):
            want = uncond + w * (cond - uncond)
            got = Guided(net, y, w)(x, t)
            assert torch.allclose(got, want, atol=1e-4), w  # 1e-6 x w, w up to 15

        # 4. guidance moves the prediction *away* from unconditional, further as w
        #    grows -- the direction the label points, extended
        d = [(Guided(net, y, w)(x, t) - uncond).norm().item() for w in (1.0, 3.0, 9.0)]
        print(f"||eps_w - eps_∅||: w=1 {d[0]:.3f}  w=3 {d[1]:.3f}  w=9 {d[2]:.3f}")
        assert d[0] < d[1] < d[2]
        assert abs(d[1] / d[0] - 3.0) < 1e-3  # exactly linear in w

    print("ok")
