# Results

What was measured, and what it showed. `roadmap.md` holds the direction; nothing
here is a plan. Runs live in `artifacts/{name}_{timestamp}/` with their
`train.log`.

## Unconditional DDPM/DDIM — `first_2026-09-06_15-43-23`

15 epochs on MNIST, RTX 4060: 48s/epoch train. Loss 0.0740 → 0.0267 (epoch 2) →
0.0221 (epoch 15). Samples are clean, well-formed digits by epoch 15; epoch 5 is
visibly worse.

**The loss lies.** It looks flat from epoch 2 on while sample quality keeps
improving a lot. Don't early-stop on it and don't use it to compare models.

> The reason first recorded here was wrong, and the correction is worth keeping.
> The claim was "eps-MSE is dominated by high `t`, where nobody beats chance".
> Bucketing it later (`class_conditioning.ipynb`, probe 4) shows the opposite:
> the average is dominated by **low** `t`. High `t` is the *easy* region, because
> `x_t ≈ eps` there and predicting eps is close to copying the input. The advice
> survives; the mechanism was backwards.

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

### `clip_denoised` is not optional for DDIM

DDIM forms the implied x0 explicitly, and `1/sqrt(ᾱ)` is 157 at t=999, so an
epoch-1 eps put the grid at `[-20.72, 22.11]`. Clamping x0 to `[-1,1]` each step,
and recomputing eps to match, gives `[-1.00, 1.00]` and legible digits. DDPM never
forms x0 and takes small steps, so it never needed this — its range settles near
`[-1.05, 1.09]`, slightly outside the data, because it does not clip.

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

## Class conditioning — `cond_2026-09-09_11-42-46`

15 epochs, same architecture and schedule as the unconditional run, 4.17M params.
Loss 0.0751 → 0.0214. Every grid row is cleanly its own digit by epoch 15; at
epoch 4 the rows are already legible with a few strays.

**The label is load-bearing, proved in `unet.py`.** Overfit one digit per class,
then score the same `x_t` under the right label and a wrong one: eps-MSE 0.0218
vs 0.0867. A label that is wired up but ignored — added to `temb` and then
normalised away, say — passes every other check in the file and fails only that
one.

The probes below are `class_conditioning.ipynb`.

### `x_T` is a style code, `y` is an identity code

Hold `x_T` fixed down a row and vary the label across it. With `eta=0` the sample
is a deterministic function of `(x_T, y)`, so the label is the only variable.
Stroke weight and slant persist across all ten digits of a row, while identity
tracks the column. Nothing in the loss asked for that split.

### The null row is grey mush

Index `num_classes` was reserved and never received a gradient, and generating
with it produces noise. That is the concrete reason CFG's label dropout is not
optional: guidance extrapolates along `eps_cond - eps_uncond`, so an untrained
`eps_uncond` is noise that `w` then amplifies. After adding dropout, that column
should become plausible digits of no particular class.

### A wrong label hurts most mid-chain

Right vs wrong label on one model, 256 training images: 1.03x at t<200, **2.04x**
at t 400-600, back to 1.21x at t>800. Low `t` has the digit legible in `x_t`, so
the label is redundant. High `t` has `sqrt(ᾱ) = 0.016`, so the best prediction is
nearly *copy the input* and no label can help or spoil it much.

### The label pays off at high `t`, not low

Against the unconditional run above — same architecture, schedule and epoch
count — eps-MSE by `t` on **held-out** images, paired `t` and noise draws:

| t | uncond | cond | gain |
| --- | --- | --- | --- |
| 0-50 | 0.1127 | 0.1105 | 1.02x |
| 400-450 | 0.0190 | 0.0165 | 1.15x |
| 600-650 | 0.0053 | 0.0038 | 1.41x |
| 800-850 | 0.0010 | 0.0004 | 2.54x |
| mean | 0.0222 | 0.0210 | |

Both curves have the same steep shape — that is the eps objective, not the label;
an unconditional U-Net shows it too. The label lives in the gap, which widens
monotonically with `t`. At low `t` an unconditional net reads identity off `x_t`
directly and `y` adds nothing; at high `t` there is nothing left to read and `y`
is the only source.

Caveat: two independent runs, so part of the gap is seed, and the high-`t`
absolutes are tiny. CFG fixes this for free — label dropout yields both
predictions from one set of weights, dropping the second run and its seed.

### The eps-MSE profile across `t` — not conditioning-specific

0.116 at t<50 against 0.0002 at t>950, a ~550x span, with held-out tracking
train. It inverts the usual "high `t` is the irreducible floor" story: at high
`t`, `x_t ≈ eps`, so predicting eps is nearly copying the input; at low `t`, eps
is the small residue after a near-clean image and recovering it divides by
`sqrt(1-ᾱ) ≈ 0`, amplifying any error in the implied `x0`. So the per-epoch
average is dominated by low `t` and hides gains at high `t`. Roadmap item 1,
arrived early.

## Loss by `t`, in the training loop — `loss_by_t.py`

`main.py` now prints the split every epoch, accumulated from the training draws
themselves. Two epochs, conditional, 10 buckets:

```
loss by t:    0-99 100-199 200-299 300-399 400-499 500-599 600-699 700-799 800-899 900-999
epoch 1  loss 0.0775
  by t:     0.2014  0.1127  0.0894  0.0758  0.0627  0.0533  0.0472  0.0448  0.0447  0.0435
epoch 2  loss 0.0264
  by t:     0.1122  0.0528  0.0370  0.0271  0.0181  0.0095  0.0041  0.0019  0.0012  0.0011
```

The offline probe's finding now shows up live, and one epoch of training makes
it dramatically worse: epoch 1 spans 5x across `t`, epoch 2 spans 100x. The
model learns high `t` almost immediately and then spends every later epoch on
low `t` — which is exactly the region the pooled scalar already reflects, so the
pooled number keeps moving while the buckets show *where*. Free: the per-sample
errors were being computed and thrown away.

Printed epoch loss is unchanged in meaning — it is the count-weighted pooled
mean, identical to the old `F.mse_loss` average.
