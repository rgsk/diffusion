"""Forward process q(x_t | x_0): fixed schedule, no learning, closed form."""

import torch
from torch import Tensor, nn

from cosine_schedule import cosine_betas


def extract(a: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
    """a[t] with trailing 1s so it broadcasts against x of `shape`."""
    out = a[t].reshape(-1, *([1] * (len(shape) - 1)))
    assert out.ndim == len(shape), (out.shape, shape)  # wrong rank broadcasts silently
    return out


class ForwardProcess(nn.Module):
    """Linear beta schedule. nn.Module so .to() moves all five buffers together and
    the schedule rides along in the checkpoint."""

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        super().__init__()
        assert schedule in ("linear", "cosine"), schedule
        self.T = T
        # beta_start/beta_end parameterise the linear schedule only; cosine is
        # defined by its ᾱ curve and ignores them
        betas = (
            torch.linspace(beta_start, beta_end, T)
            if schedule == "linear"
            else cosine_betas(T)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_ac", (1.0 - alphas_cumprod).sqrt())

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε. Pass noise in when the loss needs it back."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_ac, t, x0.shape) * noise
        )

    def target(self, x0: Tensor, noise: Tensor) -> Tensor:
        """What the net is asked to predict at x_t: the noise itself. Trivial
        here, and it exists so the training loop reads the objective off the
        process rather than hard-coding eps -- `flow.py` returns a velocity from
        the same call."""
        return noise

    def sample_t(self, n: int, device=None) -> Tensor:
        return torch.randint(0, self.T, (n,), device=device)


if __name__ == "__main__":
    torch.manual_seed(0)
    fp = ForwardProcess()
    T = fp.T

    # 1. schedule invariants
    assert (fp.betas.diff() > 0).all()
    assert (fp.alphas_cumprod.diff() < 0).all()
    assert abs(fp.alphas_cumprod[0].item() - (1 - 1e-4)) < 1e-7
    print(f"ᾱ: {fp.alphas_cumprod[0]:.5f} -> {fp.alphas_cumprod[-1]:.2e}")
    assert fp.alphas_cumprod[-1] < 1e-4
    # sqrt pair is a sin/cos: signal² + noise² = 1 at every t
    assert torch.allclose(
        fp.sqrt_alphas_cumprod**2 + fp.sqrt_one_minus_ac**2, torch.ones(T)
    )
    # SNR falls monotonically -- this is the only thing t actually means
    snr = fp.alphas_cumprod / (1 - fp.alphas_cumprod)
    assert (snr.diff() < 0).all()

    # 2. extract: rank must follow x, not a
    t = torch.tensor([0, 5, 999])
    assert extract(fp.betas, t, (3, 1, 28, 28)).shape == (3, 1, 1, 1)
    assert extract(fp.betas, t, (3, 784)).shape == (3, 1)
    assert torch.equal(extract(fp.betas, t, (3, 1, 1, 1)).flatten(), fp.betas[t])
    assert extract(fp.betas, torch.tensor(7), (2, 1, 28, 28)).shape == (
        1,
        1,
        1,
        1,
    )  # int t broadcasts

    # 3. endpoints. t=0 is barely touched, t=T-1 has lost x0
    x0 = torch.randn(4096, 1)
    assert (
        fp.q_sample(x0, torch.zeros(4096, dtype=torch.long)) - x0
    ).abs().max() < 0.06
    xT = fp.q_sample(x0, torch.full((4096,), T - 1))
    cos = torch.cosine_similarity(x0.flatten(), xT.flatten(), dim=0).item()
    print(f"cos(x0, x_T) = {cos:.4f}  (sqrt(ᾱ_T) = {fp.sqrt_alphas_cumprod[-1]:.4f})")
    assert abs(cos) < 0.05

    # 4. variance preserving: unit-variance x0 keeps std 1 at every t.
    #    MNIST in [-1,1] has std ~0.5, so it climbs 0.5 -> 1 instead.
    for ti in (0, 250, 500, 999):
        s = fp.q_sample(torch.randn(20000, 1), torch.full((20000,), ti)).std().item()
        assert abs(s - 1.0) < 0.03, (ti, s)

    # 5. closed form == the step-by-step chain, in distribution.
    #    tol = 4·σ/√N ≈ 0.03 at N=20000; asserting tighter would just be flaky.
    def iterative(x0, t):
        x = x0.clone()
        for i in range(t + 1):
            x = fp.alphas[i].sqrt() * x + fp.betas[i].sqrt() * torch.randn_like(x)
        return x

    for ti in (0, 10, 100, 500):
        x = iterative(torch.ones(20000), ti)
        m, s = x.mean().item(), x.std().item()
        em, es = fp.sqrt_alphas_cumprod[ti].item(), fp.sqrt_one_minus_ac[ti].item()
        print(f"t={ti:3d}  mean {m:.4f} vs {em:.4f}   std {s:.4f} vs {es:.4f}")
        assert abs(m - em) < 0.03 and abs(s - es) < 0.03

    # 6. one t per batch element, not one t per batch
    x0 = torch.randn(8, 1, 28, 28)
    noise = torch.randn_like(x0)
    t = torch.randint(0, T, (8,))
    batched = fp.q_sample(x0, t, noise)
    rows = torch.cat(
        [fp.q_sample(x0[i : i + 1], t[i : i + 1], noise[i : i + 1]) for i in range(8)]
    )
    assert torch.equal(batched, rows)
    assert not torch.allclose(batched[0], fp.q_sample(x0, t.roll(1), noise)[0])

    # 7. alphas where alphas_cumprod belongs: per-step noise is tiny, so x_t stays
    #    ~x0 forever and the model never sees a hard example
    bad = (
        extract(fp.alphas, t, x0.shape).sqrt() * x0
        + extract(1 - fp.alphas, t, x0.shape).sqrt() * noise
    )
    ti = t[0].item()
    print(
        f"signal weight at t={ti}: alphas {fp.alphas[ti].sqrt():.3f} vs ᾱ {fp.sqrt_alphas_cumprod[ti]:.3f}"
    )
    cos_bad = torch.cosine_similarity(bad[0].flatten(), x0[0].flatten(), dim=0).item()
    cos_ok = torch.cosine_similarity(
        batched[0].flatten(), x0[0].flatten(), dim=0
    ).item()
    print(f"cos with x0: wrong {cos_bad:.3f}, right {cos_ok:.3f}")
    assert cos_bad > 0.9 and cos_ok < 0.5  # wrong version never really destroys x0

    # 8. buffers, not attributes: one .to() moves the lot, and it survives a round-trip
    assert set(fp.state_dict()) == {
        "betas",
        "alphas",
        "alphas_cumprod",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_ac",
    }
    assert list(fp.parameters()) == []  # nothing here learns
    # checkpoint wins over the constructor -- load a run and you get its schedule back
    fp2 = ForwardProcess(beta_end=0.05)
    assert not torch.equal(fp2.betas, fp.betas)
    fp2.load_state_dict(fp.state_dict())
    assert torch.equal(fp2.betas, fp.betas)
    # and a T mismatch fails loudly here instead of silently sampling wrong
    try:
        ForwardProcess(T=10).load_state_dict(fp.state_dict())
        raise AssertionError("expected a size mismatch")
    except RuntimeError as e:
        assert "size mismatch" in str(e)
    if torch.cuda.is_available():
        cu = ForwardProcess().cuda()
        assert all(b.is_cuda for b in cu.buffers())
        assert cu.q_sample(x0.cuda(), t.cuda()).is_cuda

    print("ok")
