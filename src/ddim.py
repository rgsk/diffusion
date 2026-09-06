"""DDIM: same trained eps, sampled along a short subsequence of the chain."""

import torch
from torch import Tensor, nn

from forward_process import ForwardProcess


class DDIMSampler(nn.Module):
    """Jumps t -> t_prev in one move by re-noising the implied x0, so any strictly
    decreasing subsequence of timesteps is a valid chain. eta=0 is deterministic
    DDIM, eta=1 reproduces DDPM's posterior variance."""

    def __init__(
        self,
        fp: ForwardProcess,
        steps: int = 50,
        eta: float = 0.0,
        clip: bool = True,
    ):
        super().__init__()
        assert 1 <= steps <= fp.T, steps
        assert 0.0 <= eta <= 1.0, eta
        self.fp, self.eta, self.steps, self.clip = fp, eta, steps, clip
        # descending, always ending at 0 so the final jump lands on x_0 itself
        ts = torch.linspace(fp.T - 1, 0, steps).round().long()
        ac = fp.alphas_cumprod[ts]
        ts = ts.to(ac.device)  # fp may already be on cuda; keep the buffers together
        # ᾱ of the *next* timestep visited; 1.0 past the end, i.e. fully denoised
        ac_prev = torch.cat([fp.alphas_cumprod[ts[1:]], ac.new_ones(1)])
        # sigma=0 at eta=0; at eta=1 and steps=T this is beta_tilde
        var = (1 - ac_prev) / (1 - ac) * (1 - ac / ac_prev)
        self.register_buffer("ts", ts)
        self.register_buffer("ac", ac)
        self.register_buffer("ac_prev", ac_prev)
        self.register_buffer("sigma", eta * var.clamp_min(0).sqrt())

    @torch.no_grad()
    def step(self, model, x: Tensor, i: int) -> Tensor:
        """One jump, ts[i] -> ts[i+1]. `i` indexes the subsequence, not the schedule."""
        ac, ac_prev, sigma = self.ac[i], self.ac_prev[i], self.sigma[i]
        t = torch.full((x.shape[0],), self.ts[i], device=x.device, dtype=torch.long)
        eps = model(x, t)
        x0 = (x - (1 - ac).sqrt() * eps) / ac.sqrt()
        if self.clip:
            # 1/sqrt(ᾱ) is 157 at t=999, so a weak eps throws x0 far outside the data
            # range and never recovers. Assumes images in [-1,1].
            x0 = x0.clamp(-1, 1)
            eps = (x - ac.sqrt() * x0) / (1 - ac).sqrt()  # keep eps consistent with x0
        # re-noise x0 to ac_prev with the *predicted* eps -- this term is what keeps
        # a deterministic chain from contracting to a point
        x = ac_prev.sqrt() * x0 + (1 - ac_prev - sigma**2).clamp_min(0).sqrt() * eps
        if sigma == 0:
            return x
        return x + sigma * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, ...], device=None) -> Tensor:
        x = torch.randn(shape, device=device or self.ac.device)
        for i in range(self.steps):
            x = self.step(model, x, i)
        return x


if __name__ == "__main__":
    from sampler import DDPMSampler

    torch.manual_seed(0)
    fp = ForwardProcess()
    MU, SD = 2.0, 0.5

    def oracle(x, t):
        """Exact E[eps | x_t] when x0 ~ N(MU, SD²) -- a perfect denoiser, no training.
        If the reverse process is right, sampling with this must return N(MU, SD²)."""
        from forward_process import extract

        ac = fp.alphas_cumprod
        v = extract(ac * SD**2 + (1 - ac), t, x.shape)
        return (
            extract(fp.sqrt_one_minus_ac, t, x.shape)
            / v
            * (x - extract(fp.sqrt_alphas_cumprod, t, x.shape) * MU)
        )

    # 1. with a perfect denoiser the full-length chain must reproduce the data
    #    distribution. The std at eta=0 is the point of the file: deterministic, yet
    #    nothing contracts -- DDPM's mean without its noise term collapses this same
    #    setup to std 0.0016. tol = 4·σ/√N ≈ 0.015 at N=20000.
    for eta in (0.0, 1.0):
        s = DDIMSampler(fp, steps=fp.T, eta=eta, clip=False).sample(oracle, (20000, 1))
        print(
            f"steps={fp.T} eta={eta}  mean {s.mean():.4f} (want {MU})  std {s.std():.4f} (want {SD})"
        )
        assert abs(s.mean().item() - MU) < 0.02
        assert abs(s.std().item() - SD) < 0.02

    # 2. shortening the chain costs spread, never the mean: discretization error, so
    #    the deficit has to close monotonically as steps rise. This is the whole
    #    tradeoff -- 50 steps runs 20x faster and lands ~5% low on std.
    stds = []
    for steps in (10, 50, 200, 1000):
        s = DDIMSampler(fp, steps=steps, clip=False).sample(oracle, (20000, 1))
        print(f"steps={steps:4d} eta=0.0  mean {s.mean():.4f}  std {s.std():.4f}")
        assert abs(s.mean().item() - MU) < 0.02  # a bad coefficient biases this first
        stds.append(s.std().item())
    assert (torch.tensor(stds).diff() > 0).all(), stds
    assert stds[1] > 0.9 * SD, stds  # 50 steps stays within 10%

    # 3. the subsequence spans the whole chain. Missing t=0 leaves x_0 noisy; missing
    #    T-1 means x_T never gets denoised from where randn actually starts.
    for steps in (2, 7, 50, 1000):
        ts = DDIMSampler(fp, steps=steps).ts
        assert ts[0] == fp.T - 1 and ts[-1] == 0
        assert (ts.diff() < 0).all()
        assert len(ts) == steps

    # 4. eta=0 is a function of x_T alone -- fix the start, and the rest of the RNG
    #    stream cannot change the answer. eta>0 can. This is the property DDIM is for.
    def run(s, x):  # sample() draws its own x_T, so drive the chain by hand
        for i in range(s.steps):
            x = s.step(oracle, x, i)
        return x

    xT = torch.randn(64, 1)
    det = DDIMSampler(fp, steps=20, clip=False)
    stoch = DDIMSampler(fp, steps=20, eta=1.0, clip=False)
    torch.manual_seed(1)
    a = run(det, xT)
    torch.manual_seed(2)
    assert torch.equal(a, run(det, xT))
    torch.manual_seed(1)
    b = run(stoch, xT)
    torch.manual_seed(2)
    assert not torch.equal(b, run(stoch, xT))
    # and different starts still give different images -- determinism, not collapse
    assert not torch.allclose(a, run(det, torch.randn(64, 1)))

    # 5. one model call per step -- the reason the file exists
    calls = 0

    def counted(x, t):
        global calls
        calls += 1
        return oracle(x, t)

    DDIMSampler(fp, steps=50).sample(counted, (8, 1))
    assert calls == 50, calls
    print(f"{calls} model calls vs {fp.T} for DDPM")

    # 6. eta=1 at full length IS DDPM: same per-step variance, reached by a different
    #    expression. tol is float32 slack on variances up to 0.02 -- a real mistake in
    #    the formula would be off by orders of magnitude, not by 1e-7.
    full = DDIMSampler(fp, steps=fp.T, eta=1.0)
    post = DDPMSampler(fp, sigma="posterior").var
    assert (full.sigma**2 - post[full.ts]).abs().max() < 1e-6

    # 7. clipping bounds x0 to the data range every step, which is what stops an
    #    undertrained eps from blowing the chain up -- 1/sqrt(ᾱ_999) = 157.
    def bad(x, t):  # what an untrained net looks like: eps uncorrelated with x
        return torch.randn_like(x) * 3

    loose = DDIMSampler(fp, steps=50, clip=False).sample(bad, (64, 1, 8, 8))
    tight = DDIMSampler(fp, steps=50, clip=True).sample(bad, (64, 1, 8, 8))
    print(
        f"bad eps: unclipped [{loose.min():.1f}, {loose.max():.1f}]  clipped "
        f"[{tight.min():.2f}, {tight.max():.2f}]"
    )
    assert tight.abs().max() <= 1.0  # last step returns x0 itself, so this is exact
    assert loose.abs().max() > 5

    # 8. built from a schedule that is already on the gpu
    if torch.cuda.is_available():
        cu = DDIMSampler(ForwardProcess().cuda())
        assert cu.ac.is_cuda and cu.ts.is_cuda

    print("ok")
