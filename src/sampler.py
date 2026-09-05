"""Reverse process: ancestral DDPM sampling, x_T ~ N(0,I) walked back to x_0."""

import torch
from torch import Tensor, nn

from forward_process import ForwardProcess, extract


class DDPMSampler(nn.Module):
    """sigma="beta" is the paper's released default, "posterior" is the true
    variance of q(x_{t-1} | x_t, x_0). Both work; beta is slightly noisier."""

    def __init__(self, fp: ForwardProcess, sigma: str = "beta"):
        super().__init__()
        assert sigma in ("beta", "posterior"), sigma
        self.fp = fp
        ac = fp.alphas_cumprod
        ac_prev = torch.cat(
            [ac.new_ones(1), ac[:-1]]
        )  # new_ones: fp may already be on cuda
        # beta_tilde = beta * (1 - ᾱ_{t-1}) / (1 - ᾱ_t); it is 0 at t=0 by construction
        post = fp.betas * (1 - ac_prev) / (1 - ac)
        self.register_buffer("var", fp.betas.clone() if sigma == "beta" else post)

    @torch.no_grad()
    def step(self, model, x: Tensor, i: int) -> Tensor:
        """One denoising step, t = i -> i-1."""
        fp = self.fp
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        eps = model(x, t)
        # mean = (x - beta/sqrt(1-ᾱ) · eps) / sqrt(alpha)
        mean = (
            x
            - extract(fp.betas, t, x.shape)
            / extract(fp.sqrt_one_minus_ac, t, x.shape)
            * eps
        ) / extract(fp.alphas, t, x.shape).sqrt()
        if i == 0:
            return mean  # no noise on the last step, or you hand back a noisy digit
        return mean + extract(self.var, t, x.shape).sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, ...], device=None) -> Tensor:
        x = torch.randn(shape, device=device or self.var.device)
        for i in reversed(range(self.fp.T)):
            x = self.step(model, x, i)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    fp = ForwardProcess()
    MU, SD = 2.0, 0.5

    def oracle(x, t):
        """Exact E[eps | x_t] when x0 ~ N(MU, SD²) -- a perfect denoiser, no training.
        If the reverse process is right, sampling with this must return N(MU, SD²)."""
        ac = fp.alphas_cumprod
        v = extract(ac * SD**2 + (1 - ac), t, x.shape)
        return (
            extract(fp.sqrt_one_minus_ac, t, x.shape)
            / v
            * (x - extract(fp.sqrt_alphas_cumprod, t, x.shape) * MU)
        )

    # 1. with a perfect denoiser the sampler must reproduce the data distribution.
    #    This is the only check that pins the sign and every coefficient at once.
    for kind in ("beta", "posterior"):
        s = DDPMSampler(fp, sigma=kind).sample(oracle, (20000, 1))
        print(
            f"sigma={kind:9s}  mean {s.mean():.4f} (want {MU})  std {s.std():.4f} (want {SD})"
        )
        assert abs(s.mean().item() - MU) < 0.02
        assert abs(s.std().item() - SD) < 0.02

    # 2. the two variance choices differ per step but agree in the end -- except
    #    at t=0, where the posterior is exactly 0 and beta is not
    sp = DDPMSampler(fp, sigma="posterior")
    assert sp.var[0] == 0 and fp.betas[0] > 0
    assert (sp.var[1:] < fp.betas[1:]).all()  # posterior is the tighter one everywhere

    # 3. drop the 1/sqrt(alpha) rescale: it does NOT blow up. The denoiser's own
    #    feedback drags the chain most of the way back, leaving mean and std ~18%
    #    low. On images that reads as washed-out samples, not as a bug.
    class NoRescale(DDPMSampler):
        @torch.no_grad()
        def step(self, model, x, i):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            mean = x - extract(self.fp.betas, t, x.shape) / extract(
                self.fp.sqrt_one_minus_ac, t, x.shape
            ) * model(x, t)
            if i == 0:
                return mean
            return mean + extract(self.var, t, x.shape).sqrt() * torch.randn_like(x)

    bad = NoRescale(fp).sample(oracle, (4000, 1))
    print(f"without 1/sqrt(alpha): mean {bad.mean():.4f}  std {bad.std():.4f}")
    assert abs(bad.mean().item() - MU) > 0.1  # 5x the tolerance test 1 passes at
    assert abs(bad.std().item() - SD) > 0.05

    # 4. the last step is deterministic, every earlier one is not
    x = torch.randn(64, 1)
    s = DDPMSampler(fp)
    torch.manual_seed(1)
    a = s.step(oracle, x, 0)
    torch.manual_seed(2)
    assert torch.equal(a, s.step(oracle, x, 0))
    torch.manual_seed(1)
    b = s.step(oracle, x, 500)
    torch.manual_seed(2)
    assert not torch.equal(b, s.step(oracle, x, 500))

    # 5. built from a schedule that is already on the gpu
    if torch.cuda.is_available():
        cu = DDPMSampler(ForwardProcess().cuda())
        assert cu.var.is_cuda

    print("ok")
