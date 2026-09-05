"""Sinusoidal timestep embedding: the model's only view of how noisy its input is."""

import math

import torch
from torch import Tensor


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """[B] -> [B, dim]. Block layout (first half sin, second half cos), not the
    interleaved layout of the transformer paper -- a Linear follows either way."""
    assert dim % 2 == 0, "dim must be even"
    assert t.ndim == 1, f"want [B], got {tuple(t.shape)}"  # [B,1] would give [B,1,dim]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None, :]  # [B, half]
    return torch.cat([args.sin(), args.cos()], dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    T, D = 1000, 128
    emb = timestep_embedding(torch.arange(T), D)

    # 1. shape, range, and the two guards
    assert emb.shape == (T, D) and emb.dtype == torch.float32
    assert emb.abs().max() <= 1.0
    for bad in (
        lambda: timestep_embedding(torch.arange(4), 63),
        lambda: timestep_embedding(torch.arange(4)[:, None], 64),
    ):
        try:
            bad()
            raise SystemExit("guard missing")
        except AssertionError:
            pass

    # 2. oracle: the formula written as the loop it describes
    ref = torch.empty(8, D, dtype=torch.float64)
    for ti in range(8):
        for i in range(D // 2):
            ang = ti * math.exp(-math.log(10000) * i / (D // 2))
            ref[ti, i], ref[ti, i + D // 2] = math.sin(ang), math.cos(ang)
    assert (emb[:8] - ref).abs().max() < 1e-6

    # 3. t=0 is all sin(0), cos(0)
    assert torch.equal(emb[0], torch.cat([torch.zeros(D // 2), torch.ones(D // 2)]))

    # 4. THE property: distance depends only on |Δt|, never on where you are.
    #    Every consecutive pair is equally far apart, so no region of the
    #    schedule is harder for the model to tell apart than any other.
    step = (emb[1:] - emb[:-1]).norm(dim=1)
    print(f"adjacent step: [{step.min():.5f}, {step.max():.5f}]")
    assert (step.max() - step.min()) < 1e-3
    g = emb @ emb.T
    # every t sits on one sphere: sin²+cos² per pair, D/2 pairs, no t excepted
    assert (torch.diagonal(g) - D / 2).abs().max() < 1e-4
    for d in (1, 7, 100):
        diag = torch.diagonal(g, offset=d)
        assert (diag - diag[0]).abs().max() < 1e-2, d
    print(
        f"cos(t, t+1) = {torch.cosine_similarity(emb[0], emb[1], dim=0):.4f} "
        f"at t=0, {torch.cosine_similarity(emb[500], emb[501], dim=0):.4f} at t=500"
    )

    # 5. the frequency budget is mostly wasted: max_period=10000 is inherited from
    #    NLP, where positions run to ~10k. Over t<1000 the slow hands turn 0.12 rad
    #    and their channels are near-constant -- dead width the model pays for.
    rng = emb.max(0).values - emb.min(0).values
    print(f"channels with range < 0.5: {(rng < 0.5).sum().item()}/{D}")
    assert (rng < 0.5).sum() > D // 8
    tight = timestep_embedding(torch.arange(T), D, max_period=1000.0)
    tight_rng = tight.max(0).values - tight.min(0).values
    print(f"channels with range (tight) < 0.5: {(tight_rng < 0.5).sum().item()}/{D}")
    assert (tight_rng < 0.5).sum() < (
        rng < 0.5
    ).sum()  # a shorter period uses more of D

    print("ok")
