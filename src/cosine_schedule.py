"""Cosine beta schedule (Nichol & Dhariwal, Improved DDPM).

A schedule is a budget: T steps spent across the range of noise levels. Linear
betas spend that budget badly on small images -- ᾱ crosses 1/2 by t≈259 and half
of all steps sit at SNR < 0.1, where x_t is nearly pure noise and there is
little left to learn. Cosine holds ᾱ near a straight line down to 0, pushing the
crossing to t≈496, while still ending *more* fully noised than linear does.
"""

import math

import torch
from torch import Tensor


def cosine_betas(T: int = 1000, s: float = 0.008, max_beta: float = 0.999) -> Tensor:
    """ᾱ_t = cos²(((t/T)+s)/(1+s) · π/2), normalised to ᾱ_0 = 1; betas are what
    that implies, 1 - ᾱ_t/ᾱ_{t-1}.

    `s` offsets the cosine off its flat peak, so β near t=0 is a real step
    rather than ~0. `max_beta` caps the singularity at t=T, where ᾱ→0 sends the
    ratio to 1 and the last step would erase the state in one go.
    """
    assert T > 0 and 0.0 <= s < 1.0 and 0.0 < max_beta < 1.0, (T, s, max_beta)
    t = torch.arange(T + 1, dtype=torch.float64) / T  # float64: the ratio is 1-ε
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    return (1.0 - ac[1:] / ac[:-1]).clamp(max=max_beta).float()


if __name__ == "__main__":
    from forward_process import ForwardProcess

    T = 1000
    betas = cosine_betas(T)
    ac = torch.cumprod(1.0 - betas, dim=0)
    lin = ForwardProcess(T).betas.double()  # the schedule this replaces
    lac = torch.cumprod(1.0 - lin, dim=0)

    # 1. it is a usable schedule at all: same invariants forward_process.py asserts
    assert betas.shape == (T,) and betas.dtype == torch.float32
    assert (betas > 0).all() and (betas < 1).all()
    assert (ac.diff() < 0).all()
    snr = ac / (1 - ac)
    assert (snr.diff() < 0).all()

    # 2. the betas really do imply the ᾱ they were derived from -- an off-by-one
    #    in the ratio breaks this and nothing else
    arg = (torch.arange(1, T + 1, dtype=torch.float64) / T + 0.008) / 1.008
    want = (
        torch.cos(arg * math.pi / 2) ** 2 / math.cos(0.008 / 1.008 * math.pi / 2) ** 2
    )
    assert (ac.double() - want).abs().max() < 1e-5

    # 3. THE point: where the budget goes. Linear is half-destroyed by t≈259 and
    #    spends half its steps in noise; cosine reaches the same place at t≈496.
    half_c = int((ac < 0.5).nonzero()[0])
    half_l = int((lac < 0.5).nonzero()[0])
    dead_c = ((ac / (1 - ac)) < 0.1).float().mean().item()
    dead_l = ((lac / (1 - lac)) < 0.1).float().mean().item()
    print(f"ᾱ < 0.5 at t: cosine {half_c}, linear {half_l}")
    print(f"steps at SNR < 0.1: cosine {dead_c:.0%}, linear {dead_l:.0%}")
    assert half_c > 1.8 * half_l
    assert dead_c < dead_l / 2

    # 4. and it is not simply "less noise": the end is more thoroughly noised
    #    than linear's, which still leaks ᾱ=4e-5 of the image into x_T
    print(f"ᾱ_T: cosine {ac[-1]:.2e}, linear {lac[-1]:.2e}")
    assert ac[-1] < lac[-1]
    assert ac[-1] > 0  # the max_beta clip is what keeps this off exactly zero

    # 5. the two knobs do what the docstring claims
    assert cosine_betas(T, s=0.0)[0] < betas[0] / 10  # s lifts the first step
    assert (cosine_betas(T)[-1] - 0.999).abs() < 1e-6  # clip binds at the end
    assert (cosine_betas(T, max_beta=0.5) <= 0.5).all()

    # 6. it drops into ForwardProcess, which is the only place it is used
    fp = ForwardProcess(T, schedule="cosine")
    assert torch.equal(fp.betas, betas)
    assert torch.allclose(fp.alphas_cumprod, ac.float(), atol=1e-6)
    assert torch.allclose(
        fp.sqrt_alphas_cumprod**2 + fp.sqrt_one_minus_ac**2, torch.ones(T), atol=1e-6
    )

    print("ok")
