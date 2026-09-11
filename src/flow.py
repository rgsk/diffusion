"""Flow matching: a straight line from noise to data, and the ODE that walks it back.

DDPM defines a stochastic forward process, derives a posterior, and trains a net
to predict the noise that was added. Flow matching throws all of that out and
draws a straight line:

    x_s = (1-s)·x_0 + s·eps        s in [0,1], s=0 is data, s=1 is noise

Differentiate: dx_s/ds = eps - x_0, which does not depend on s at all. The target
is a *constant* along each path -- the velocity -- and training is regression
onto it. No betas, no alphas, no posterior variance: `forward_process.py` and
`sampler.py` together collapse into two lines of arithmetic, and this module's
state_dict is empty. That emptiness is the claim, and it is asserted below.

Sampling is then an ODE, integrated from s=1 down to s=0 by Euler:

    x <- x + ds · v(x_s, s)        ds < 0

Deterministic like DDIM at eta=0, and for the same reason it does not collapse to
a point: the velocity it steps along carries the spread, unlike DDPM's posterior
mean used without its variance.

`t` is quoted in the same 0..T units as the diffusion schedule, continuous rather
than integer. Two practical reasons: `timestep_embedding`'s frequencies were
chosen for arguments of order 1000, and s in [0,1] puts every channel into one
tiny arc (test 7); and the `loss by t` buckets then line up with the diffusion
runs, so the two objectives can be read side by side. T is the *unit* here, not a
step count -- nothing is discretized until the sampler picks its grid.
"""

import torch
from torch import Tensor, nn


class FlowPath(nn.Module):
    """The peer of `ForwardProcess`, and the same interface: `sample_t`,
    `q_sample`, `target`. nn.Module only so `.to()` and `state_dict()` work at
    the same call sites -- there is nothing to move and nothing to save."""

    def __init__(self, T: int = 1000):
        super().__init__()
        self.T = T

    def s(self, t: Tensor, shape: torch.Size) -> Tensor:
        """t -> s = t/T with trailing 1s so it broadcasts against x of `shape`.
        The analogue of `extract`, except there is no table to index."""
        out = t.reshape(-1, *([1] * (len(shape) - 1))).float() / self.T
        assert out.ndim == len(shape), (out.shape, shape)  # wrong rank broadcasts
        return out

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """x_s = (1-s)·x_0 + s·eps. Named for the `ForwardProcess` method it stands
        in for, so the training loop has one path through it."""
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.s(t, x0.shape)
        return (1 - s) * x0 + s * noise

    def target(self, x0: Tensor, noise: Tensor) -> Tensor:
        """v = dx_s/ds = eps - x_0. Takes no t: that is the whole point of a
        straight path, and the reason this is called rectified flow."""
        return noise - x0

    def sample_t(self, n: int, device=None) -> Tensor:
        """Uniform on [0, T), continuous. SD3 samples this logit-normally instead,
        to spend fewer draws on the two easy ends; uniform is the control."""
        return torch.rand(n, device=device) * self.T


class FlowSampler(nn.Module):
    """Euler integration of dx/ds = v from s=1 to s=0.

    `steps` model calls, one per interval, evaluated at the interval's start. The
    grid lands exactly on s=0, so the final state *is* x_0 -- there is no special
    last step, no clipping, and no variance term to choose."""

    def __init__(self, path: FlowPath, steps: int = 50):
        super().__init__()
        assert steps >= 1, steps
        self.path, self.steps = path, steps
        # steps+1 knots -> steps intervals. In the schedule's units, so what the
        # net is handed here is what it was trained on.
        self.register_buffer("ts", torch.linspace(float(path.T), 0.0, steps + 1))

    @torch.no_grad()
    def step(self, model, x: Tensor, i: int) -> Tensor:
        """One Euler step, ts[i] -> ts[i+1]. `i` indexes the grid, not the units."""
        t = self.ts[i].expand(x.shape[0])
        ds = (self.ts[i + 1] - self.ts[i]) / self.path.T  # negative: walking back
        return x + ds * model(x, t)

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, ...], device=None) -> Tensor:
        x = torch.randn(shape, device=device or self.ts.device)
        for i in range(self.steps):
            x = self.step(model, x, i)
        return x


if __name__ == "__main__":
    from ddim import DDIMSampler
    from forward_process import ForwardProcess, extract
    from timestep_embedding import timestep_embedding

    torch.manual_seed(0)
    path = FlowPath()
    T, MU, SD = path.T, 2.0, 0.5

    # 1. endpoints are exact, not approximate. VP diffusion only ever gets
    #    x_T ~= eps (ᾱ_T = 2e-9); here s=1 IS eps, so there is no train/sample
    #    mismatch left to argue about.
    x0 = torch.randn(512, 1, 8, 8)
    eps = torch.randn_like(x0)
    assert torch.equal(path.q_sample(x0, torch.zeros(512), eps), x0)
    assert torch.equal(path.q_sample(x0, torch.full((512,), float(T)), eps), eps)

    # 2. THE property: the path is straight, so the target is one vector per
    #    sample and every point on the line agrees about it. A non-constant
    #    target would still train and still sample -- it would just not be flow
    #    matching, and nothing else here would notice.
    v = path.target(x0, eps)
    for a, b in ((0.0, 250.0), (250.0, 900.0), (100.0, 1000.0)):
        ta, tb = torch.full((512,), a), torch.full((512,), b)
        moved = path.q_sample(x0, tb, eps) - path.q_sample(x0, ta, eps)
        assert torch.allclose(moved, (b - a) / T * v, atol=1e-6), (a, b)

    # 3. not variance preserving, unlike every schedule in forward_process.py.
    #    Var(x_s) = (1-s)²·Var(x0) + s², which dips below both ends mid-path for
    #    unit-variance data. Worth knowing before reading a loss curve: the input
    #    the net sees at s=0.5 is smaller than the data and smaller than noise.
    unit = torch.randn(200000, 1)
    stds = [
        path.q_sample(unit, torch.full((200000,), si * T), torch.randn_like(unit))
        .std()
        .item()
        for si in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    print("std(x_s) at s=0,.25,.5,.75,1: " + " ".join(f"{s:.3f}" for s in stds))
    assert min(stds) == stds[2] and stds[2] < 0.72  # sqrt(0.5) = 0.707
    assert abs(stds[0] - 1) < 0.01 and abs(stds[-1] - 1) < 0.01

    def oracle(x, t):
        """Exact E[v | x_s] when x0 ~ N(MU, SD²) -- a perfect velocity predictor,
        no training. x_s and v are jointly Gaussian, so the conditional mean is
        Cov(v, x_s)/Var(x_s) · (x_s - E[x_s]) + E[v]:

            Var(x_s) = (1-s)²SD² + s²
            Cov(v, x_s) = Cov(eps - x0, (1-s)x0 + s·eps) = s - (1-s)SD²

        If the ODE and its direction are right, integrating this must return
        N(MU, SD²)."""
        s = path.s(t, x.shape)
        var = (1 - s) ** 2 * SD**2 + s**2
        return -MU + (s - (1 - s) * SD**2) / var * (x - (1 - s) * MU)

    # 3b. the oracle's own endpoints, which are forced and need no algebra:
    #     at s=1 the input is eps so E[v] = eps - MU; at s=0 it is x0 so E[v] = -x0
    xs = torch.randn(1000, 1)
    assert torch.allclose(oracle(xs, torch.full((1000,), float(T))), xs - MU, atol=1e-5)
    assert torch.allclose(oracle(xs, torch.zeros(1000)), -xs, atol=1e-5)

    # 4. with a perfect velocity the ODE must reproduce the data distribution.
    #    The single check that pins the sign of ds, the direction of v, and every
    #    coefficient at once. tol = 4·sigma/sqrt(N) ~ 0.015 at N=20000.
    for steps in (200, 1000):
        s = FlowSampler(path, steps=steps).sample(oracle, (20000, 1))
        print(
            f"steps={steps:4d}  mean {s.mean():.4f} (want {MU})  "
            f"std {s.std():.4f} (want {SD})"
        )
        assert abs(s.mean().item() - MU) < 0.02
        assert abs(s.std().item() - SD) < 0.02

    # 5. shortening the grid costs spread, never the mean -- Euler truncation
    #    error, so the deficit has to close monotonically. Same shape of tradeoff
    #    as DDIM's, and the reason a step count is a knob rather than a constant.
    got = []
    for steps in (2, 5, 10, 50, 200):
        s = FlowSampler(path, steps=steps).sample(oracle, (20000, 1))
        print(f"steps={steps:4d}  mean {s.mean():.4f}  std {s.std():.4f}")
        assert abs(s.mean().item() - MU) < 0.02  # a bad coefficient biases this first
        got.append(s.std().item())
    assert (torch.tensor(got).diff() > 0).all(), got
    assert got[3] > 0.9 * SD, got  # 50 steps within 10%, as DDIM's test asserts

    # 5b. head to head with DDIM on the same task, same oracle setup, same number
    #     of model calls. Both are deterministic integrators of a probability
    #     flow; the difference is the path they integrate along, and a straight
    #     one is cheaper to follow with big steps. This is the claim that makes
    #     flow matching worth the rewrite, so measure it rather than repeat it.
    fp = ForwardProcess()

    def eps_oracle(x, t):
        """Exact E[eps | x_t] for the same N(MU, SD²) -- ddim.py's oracle."""
        ac = fp.alphas_cumprod
        var = extract(ac * SD**2 + (1 - ac), t, x.shape)
        return (
            extract(fp.sqrt_one_minus_ac, t, x.shape)
            / var
            * (x - extract(fp.sqrt_alphas_cumprod, t, x.shape) * MU)
        )

    print(f"{'steps':>6} {'ddim std':>9} {'flow std':>9}   (want 0.5)")
    for steps, flow_std in zip((2, 5, 10, 50, 200), got):
        d = DDIMSampler(fp, steps=steps, clip=False).sample(eps_oracle, (20000, 1))
        print(f"{steps:6d} {d.std().item():9.4f} {flow_std:9.4f}")
        # the gap closes as the grid refines; at 200 steps both are essentially
        # exact and which one wins by 0.001 is noise, so only claim the low end
        if steps <= 10:
            assert flow_std > d.std().item(), (steps, flow_std, d.std().item())

    # 6. deterministic in x_1 alone, and *not* collapsed -- the pair of claims
    #    that separates this from DDPM's mean without its noise term, which has
    #    the first property and fails the second at std 0.0016.
    def run(s, x):  # sample() draws its own x_1, so drive the grid by hand
        for i in range(s.steps):
            x = s.step(oracle, x, i)
        return x

    smp = FlowSampler(path, steps=20)
    x1 = torch.randn(4096, 1)
    torch.manual_seed(1)
    a = run(smp, x1)
    torch.manual_seed(2)
    assert torch.equal(a, run(smp, x1))  # no RNG is consulted at all
    assert not torch.allclose(a, run(smp, torch.randn(4096, 1)))
    assert a.std().item() > 0.9 * SD, a.std().item()  # spread survives determinism

    # 7. the units. s in [0,1] would be handed straight to timestep_embedding,
    #    whose slowest frequency is 1e-4: every channel would sit in one tiny arc
    #    and t would stop being legible to the net. Nothing would crash.
    D = 64
    tt = path.sample_t(4096)
    dead = lambda e: int(((e.max(0).values - e.min(0).values) < 0.5).sum())
    scaled, raw = dead(timestep_embedding(tt, D)), dead(timestep_embedding(tt / T, D))
    print(f"near-constant channels of {D}: t in [0,T) {scaled}, t in [0,1) {raw}")
    assert raw > 2 * scaled and raw > D * 0.8

    # 8. bounded by construction, with no clip anywhere. DDIM had to clamp x0
    #    because 1/sqrt(ᾱ_999) = 157 turns a weak eps into a [-20, 22] image;
    #    Euler's coefficients here sum to 1, so a garbage velocity perturbs the
    #    result instead of exploding it. One less thing to get wrong.
    def bad(x, t):  # what an untrained net looks like
        return torch.randn_like(x) * 3

    loose = FlowSampler(path, steps=50).sample(bad, (64, 1, 8, 8))
    print(f"garbage v, no clip: [{loose.min():.2f}, {loose.max():.2f}]")
    assert loose.abs().max() < 8  # ddim.py's unclipped equivalent exceeds 20

    # 9. one model call per step, and the last state is x_0 itself
    calls = 0

    def counted(x, t):
        global calls
        calls += 1
        return oracle(x, t)

    FlowSampler(path, steps=50).sample(counted, (8, 1))
    assert calls == 50, calls
    assert FlowSampler(path, steps=7).ts[-1] == 0.0

    # 10. nothing to checkpoint and nothing to learn. ForwardProcess carries five
    #     buffers and fails loudly on a T mismatch; there is no such failure mode
    #     here because there is no table.
    assert path.state_dict() == {} and list(path.parameters()) == []
    if torch.cuda.is_available():
        cu = FlowSampler(FlowPath().cuda(), steps=10).cuda()
        assert cu.ts.is_cuda
        assert cu.sample(oracle, (64, 1), "cuda").is_cuda

    # 11. the two objectives' losses are not comparable, and this is why: a net
    #     predicting zero scores E[eps²]=1 under eps-MSE and E[(eps-x0)²]=1+Var
    #     under v-MSE. Same trap as linear-vs-cosine in README.md.
    d = torch.randn(100000, 1) * SD + MU
    n = torch.randn_like(d)
    print(
        f"E[eps²] {n.square().mean():.4f} vs E[v²] "
        f"{path.target(d, n).square().mean():.4f} (data std {SD}, mean {MU})"
    )
    assert path.target(d, n).square().mean() > 1.2 * n.square().mean()

    print("ok")
