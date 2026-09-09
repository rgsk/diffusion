"""Exponential moving average of the weights themselves.

Adam already keeps an EMA of the *gradient*, which chooses the step direction;
this averages the *positions* the steps land on, and never feeds back into
training. At a fixed lr the iterates cannot sit at the minimum -- they rattle in
a ball whose radius is set by lr times the minibatch noise -- and momentum does
not shrink that ball. Averaging over it does. Sampling runs the net 50-1000
times in sequence, each step's output the next step's input, so weight noise
compounds down the chain; every DDPM paper samples from these weights.
"""

from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    """A shadow copy of `model`'s state, trailing it. A pure observer: `update`
    touches nothing the optimizer reads, so the training path is unchanged."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        assert 0.0 <= decay < 1.0, decay
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def rate(self) -> float:
        """Ramp: min(decay, (1+n)/(10+n)). At 0.999 the average has a ~1000-step
        memory, so a flat decay would leave the shadow mostly *random init* for
        the first epochs -- EMA would look like it hurt. The ramp starts near 0,
        i.e. "just copy the weights", and tightens as there is history to keep."""
        return min(self.decay, (1 + self.n) / (10 + self.n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.rate()
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.is_floating_point():
                s.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                s.copy_(v)  # step counters and the like: an averaged int is nonsense
        self.n += 1

    @contextmanager
    def as_weights(self, model: nn.Module):
        """Run `model` under the averaged weights, then restore the live ones
        exactly. Sampling borrows the net mid-training, so the restore has to
        survive an exception in the body."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(backup)


if __name__ == "__main__":
    torch.manual_seed(0)

    class Tiny(nn.Module):
        """A float param and an int buffer -- the two branches of update()."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(64))
            self.register_buffer("seen", torch.zeros((), dtype=torch.long))

    m = Tiny()

    # 1. the ramp: 20 steps at a constant target must arrive, from a zero init.
    #    A flat 0.999 would still be 98% init here -- the failure the ramp fixes.
    ema = EMA(m, 0.999)
    with torch.no_grad():
        m.w.fill_(5.0)
    for _ in range(20):
        ema.update(m)
    print(f"after 20 steps: shadow {ema.shadow['w'][0]:.4f} vs weights 5.0")
    assert (ema.shadow["w"] - 5.0).abs().max() < 0.1
    assert abs(ema.rate() - 21 / 30) < 1e-9  # still ramping, decay not yet reached
    for _ in range(50_000):
        ema.n += 1
    assert ema.rate() == 0.999  # and it does reach it

    # 2. THE property: on a noisy trajectory the average is closer to the truth
    #    than the iterate is. This is the whole reason for the file.
    truth = torch.randn(64)
    ema = EMA(m, 0.999)
    for _ in range(4000):
        with torch.no_grad():
            m.w.copy_(truth + 0.5 * torch.randn(64))  # stand-in for the noise ball
        ema.update(m)
    iterate_err = (m.w.detach() - truth).norm().item()
    ema_err = (ema.shadow["w"] - truth).norm().item()
    print(f"iterate err {iterate_err:.4f}  ema err {ema_err:.4f}")
    assert ema_err < iterate_err / 5

    # 3. observer: update() must not move the live weights or carry a graph
    before = m.w.detach().clone()
    ema.update(m)
    assert torch.equal(m.w.detach(), before)
    assert not ema.shadow["w"].requires_grad

    # 4. int buffer is copied, not averaged -- 0.999*3 + 0.001*3 would round to 2
    with torch.no_grad():
        m.seen.fill_(3)
    ema.update(m)
    assert ema.shadow["seen"].dtype == torch.long and ema.shadow["seen"].item() == 3

    # 5. the swap is exact both ways, and survives a raise in the body
    live = {k: v.clone() for k, v in m.state_dict().items()}
    with ema.as_weights(m) as swapped:
        assert torch.equal(swapped.w.detach(), ema.shadow["w"])
        assert not torch.equal(swapped.w.detach(), live["w"])
    assert all(torch.equal(m.state_dict()[k], v) for k, v in live.items())
    try:
        with ema.as_weights(m):
            raise RuntimeError("sampler blew up")
    except RuntimeError:
        pass
    assert all(torch.equal(m.state_dict()[k], v) for k, v in live.items())

    # 6. shadow is a copy, not a view: training on after construction can't edit it
    ema2 = EMA(m, 0.999)
    with torch.no_grad():
        m.w.add_(1.0)
    assert not torch.equal(ema2.shadow["w"], m.w.detach())

    print("ok")
