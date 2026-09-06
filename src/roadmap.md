# Roadmap

Updated 2026-09-06. One concept per file in `src/`, each with a `__main__` that
tests it. Run any file directly.

## Done

- `forward_process.py` — linear beta schedule + `q_sample`, buffers on an
  `nn.Module`. `extract` lives here.
- `timestep_embedding.py` — sinusoidal `t -> [B, dim]`, block layout.
- `resblock.py` — GroupNorm/SiLU/conv ×2, temb added in the middle, zero-init
  `conv2` so a fresh block is exactly its skip.
- `unet.py` — 28→14→7, mults (1,2,2), 4.17M params. Zero-init output conv.
- `sampler.py` — ancestral DDPM, `sigma="beta"|"posterior"`.
- `ddim.py` — DDIM, `steps`/`eta`/`clip`. `eta=0` is deterministic; `eta=1` at
  full length reproduces DDPM's posterior variance, asserted against it.
- `main.py` — train on MNIST, sample a grid per epoch. `--sampler ddim|ddpm`,
  `--ddim-steps`. Runs land in `artifacts/scratch` (wiped each run) or
  `artifacts/{name}_{timestamp}`, with `train.log` written beside the grids.

## Results

15 epochs on MNIST, RTX 4060: 48s/epoch train. Loss 0.0740 → 0.0267 (epoch 2) →
0.0221 (epoch 15). Samples are clean, well-formed digits by epoch 15; epoch 5 is
visibly worse.

**The loss lies.** It looks flat from epoch 2 on, while sample quality keeps
improving a lot. eps-MSE is dominated by high `t`, where nobody beats chance, so
the average mostly measures an irreducible floor. Don't early-stop on it and
don't use it to compare models.

### Sampler cost — epoch-15 weights, same `x_T`

| sampler | time | range | across-sample std |
| --- | --- | --- | --- |
| DDIM 10 | 0.30s | `[-1.08, 1.12]` | 0.395 |
| DDIM 50 | 0.74s | `[-1.05, 1.09]` | 0.417 |
| DDIM 100 | 1.48s | `[-1.04, 1.09]` | 0.420 |
| DDPM 1000 | 14.92s | `[-1.05, 1.11]` | 0.388 |

DDIM at 50 steps is the default; even 10 steps is legible. Against the Gaussian
oracle the mean is exact at every step count while the std runs short — 0.3616
at 10 steps, 0.4728 at 50, 0.4986 at 1000, against 0.5. That deficit is
discretization error, not bias, and it closes monotonically with steps.

**`clip_denoised` is not optional for DDIM.** DDIM forms the implied x0
explicitly, and `1/sqrt(ᾱ)` is 157 at t=999, so an epoch-1 eps put the grid at
`[-20.72, 22.11]`. Clamping x0 to `[-1,1]` each step, and recomputing eps to
match, gives `[-1.00, 1.00]` and legible digits. DDPM never forms x0 and takes
small steps, so it never needed this — its range settles near `[-1.05, 1.09]`,
slightly outside the data, because it does not clip.

### Deleting the noise term collapses the chain

Dropping `sigma * randn` from DDPM's step — keeping only the posterior mean —
doesn't merely reduce diversity, it destroys the sampler. Against the Gaussian
oracle the mean stays exact at 2.0000 while the std falls from 0.5 to **0.0016**:
a point mass. On MNIST all 64 samples land on one image (across-sample std
0.0027, smaller than the variation *within* a single image), and the 1000 steps
forget `x_T` entirely.

Which point they land on is a property of the weights, not of training progress:
epoch 4 gave 64 identical strokes, epoch 5 a blank field. Better ε-prediction
makes the contraction *tighter*, not safer.

**Deterministic is not the problem — dropping the variance is.** DDIM is fully
deterministic and fine, because its re-noising term puts the spread back using
the predicted eps. DDPM's mean without its variance is half a distribution used
as if it were a whole one.

**Training cannot see any of this.** Loss improved every epoch (0.0746 → 0.0238)
while the grids went to garbage. The objective scores one-step eps prediction
against known noise; nothing in the loop exercises the sampling chain. Only the
per-epoch grid catches it — or `sampler.py`'s oracle test, in about a second.

## Next

1. **Loss bucketed by `t`** — the metric that tracks what the eye sees.
2. **Self-attention** at 7×7, wired into the U-Net.
3. **EMA of weights** — standard in DDPM, usually a visible quality win.
4. **Cosine schedule** — linear betas destroy MNIST's signal early.

## Housekeeping

Done: `data/` untracked and gitignored, `ruff` added (config in `pyproject.toml`,
notebooks excluded), `src/` formatted. `artifacts/` and `scratchpad/` gitignored.

Open: the MNIST blobs are still in git history, so `.git` stays ~25MB. Purging
them means rewriting history and force-pushing to `origin`.
