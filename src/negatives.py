"""A loss term that charges for getting the assignment wrong.

`README.md` records four runs that build every mechanism binding is supposed to
need -- cross-attention, a text encoder, spatial coordinates -- and assign
colours to corners at chance anyway. The measurement that explains all four:
scoring a caption against its colour-swapped twin gives an eps-MSE ratio of
1.000, in every noise bucket, in every run. The objective is indifferent between
the right answer and the swap, and a model indifferent between two answers picks
one by coin flip. No amount of mechanism fixes an objective that never asks.

So ask. Run the same `x_t` under the swapped caption and require it to be worse.

The hard part is that "make the wrong caption score worse" has a second
solution, and it is much cheaper than binding: *be worse at everything*. A model
that cannot tell the two captions apart raises both errors together, so any term
reading the swap's error alone is satisfied by predicting nothing at all. The
first version of this file did exactly that -- `mse_neg / mse_pos.detach()`, a
ratio whose gradient only ever pushed `mse_neg` up -- and it collapsed training
on contact: epoch 1 loss 1.0094, which is E[eps^2], the score for predicting
zero, with the hinge pinned at the full margin. It never came back.

The fix is to score the swap's *share* of the pair rather than its size:

    mse_neg / (mse_pos + mse_neg)   >=   0.5 + margin

Degrading both errors together leaves that fraction at exactly 0.5 and buys
nothing, so the escape route is closed by construction rather than by tuning.
Being right more often and being wrong on the swap more often both raise it, and
both are what we wanted. It is bounded in [0, 1], it is scale-free -- which the
50x span of eps-MSE across `t` demands -- and it needs no detach, because
lowering `mse_pos` now helps rather than cheats.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def per_sample_mse(pred: Tensor, target: Tensor) -> Tensor:
    """[B, ...] -> [B]. The pooled loss is a mean of these; the hinge needs them
    one at a time, so a sample that already answers correctly stops paying."""
    return (pred - target).square().flatten(1).mean(1)


def swap_hinge(
    mse_pos: Tensor, mse_neg: Tensor, margin: float = 0.1, eps: float = 1e-12
) -> Tensor:
    """[B], [B] -> [B]. Zero once the swapped caption owns `0.5 + margin` of the
    pair's error; the shortfall otherwise. `margin=0.1` asks for 1.5x."""
    share = mse_neg / (mse_pos + mse_neg).clamp_min(eps)
    return F.relu(0.5 + margin - share)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, M = 8, 0.1
    pos = torch.full((B,), 0.02)

    # 1. an indifferent model -- the state every run in README.md is in -- pays
    #    the full margin, and a model that separates them pays nothing
    indifferent = swap_hinge(pos, pos.clone(), M)
    print(f"an indifferent model pays {indifferent[0]:.4f} of a {M} margin")
    assert torch.allclose(indifferent, torch.full((B,), M), atol=1e-6)
    assert swap_hinge(pos, pos * 1.5, M).max() < 1e-6  # exactly at the margin
    assert torch.equal(swap_hinge(pos, pos * 3, M), torch.zeros(B))
    assert (swap_hinge(pos, pos * 0.5, M) > indifferent).all()  # prefers the swap

    # 2. THE property the first version of this file lacked: being worse at
    #    everything must not help. A model that cannot bind raises both errors
    #    together, and that is the cheap escape route this term has to close.
    for factor in (2.0, 10.0, 50.0):
        assert torch.allclose(
            swap_hinge(pos * factor, pos * factor, M), indifferent, atol=1e-6
        )

    # 3. and it is scale-free the other way too: eps-MSE spans 50x across t
    #    (README.md), so the same separation must cost the same at both ends
    for scale in (0.0006, 0.03, 0.2):  # the observed t=900 .. t=0 range
        s = torch.full((B,), scale)
        assert torch.allclose(swap_hinge(s, s, M), torch.full((B,), M), atol=1e-6)
        assert swap_hinge(s, s * 1.5, M).max() < 1e-6

    # 4. per-sample: the ones already right drop out, so the gradient goes to
    #    the ones that are not
    h = swap_hinge(pos[:2], torch.stack([pos[0] * 3, pos[0]]), M)
    assert h[0] == 0.0 and h[1] > 0.0

    # 5. both gradients point the way we want, and neither rewards damage:
    #    down on the true caption's error, up on the swap's
    p = torch.tensor([0.02], requires_grad=True)
    n = torch.tensor([0.02], requires_grad=True)
    swap_hinge(p, n, M).backward()
    print(
        f"d(hinge)/d(mse_pos) = {p.grad.item():+.3f}   "
        f"d(hinge)/d(mse_neg) = {n.grad.item():+.3f}"
    )
    assert p.grad.item() > 0, "must pay for being worse at the true caption"
    assert n.grad.item() < 0, "and must pay for being good at the swapped one"
    # equal and opposite, which is what makes rescaling both a no-op
    assert abs(p.grad.item() + n.grad.item()) < 1e-4

    print("ok")
