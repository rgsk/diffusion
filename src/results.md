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

## EMA of weights — `ema_2026-09-09_17-21-32`

5 epochs, conditional, decay 0.999 with the `(1+n)/(10+n)` warm-up ramp. The
shadow copy costs one extra 4.17M-param buffer and a multiply-add per step;
epoch time is unchanged at 48s.

Held-out eps-MSE, 16k paired draws (same `t` and same noise for both nets), the
trained weights against their own EMA:

| t | trained | ema | ratio |
| --- | --- | --- | --- |
| 0-99 | 0.09357 | 0.09150 | 1.02x |
| 300-399 | 0.02376 | 0.02249 | 1.06x |
| 600-699 | 0.00313 | 0.00292 | 1.07x |
| 800-899 | 0.00039 | 0.00032 | 1.23x |
| 900-999 | 0.00025 | 0.00020 | 1.28x |
| pooled | 0.02277 | 0.02201 | 1.03x |

EMA wins in every bucket, and **the win grows with `t`**: 2% at the bottom, 28%
at the top. Same asymmetry as the label's, and for a related reason — high `t`
is where the target is nearly the input, so what remains to lose there is
largely gradient noise, which is exactly what averaging removes.

Note what the pooled number does to this: 1.03x. A 28% improvement in the region
the sampler starts from is reported as 3%, because low `t` dominates the
average. Without the bucketed log this would have read as noise.

Visually, at 5 epochs on MNIST the two grids (`compare_trained.png`,
`compare_ema.png`, same `x_T`) are both clean and the difference is not
convincing by eye — a few digits differ in identity, neither set is obviously
better. Across-sample std 0.3877 trained vs 0.3707 EMA. The measurable claim
here is the held-out loss; the visible-quality claim from the literature is for
longer runs and harder data than this.

## Cosine vs linear schedule — closed form, no training

`cosine_schedule.ipynb`. Both schedules are arithmetic, so this holds before any
model exists.

| | ᾱ < 0.5 at | steps at SNR < 0.1 | ᾱ_T |
| --- | --- | --- | --- |
| linear | t=259 | 52% | 4.04e-05 |
| cosine | t=496 | 20% | 2.43e-09 |

Linear spends **half its steps** where SNR < 0.1 — x_t is essentially noise and
there is little left to predict — and reaches half-destroyed by t=259. Cosine
pushes that to t=496 and cuts the near-noise region to a fifth of the steps.

It is not "less noise overall": cosine ends four orders of magnitude *more*
destroyed (ᾱ_T = 2e-9 vs 4e-5). Linear still leaks a trace of the image into
x_T, which is a train/sample mismatch, since sampling starts from pure noise.
Cosine is a better *allocation* of the same T steps at both ends.

The digit strip in the notebook shows the same thing directly: same x₀, same ε,
and linear's 3 is unreadable by t=500 while cosine's survives to t≈600.

Whether it helps a trained model is the next section.

## Cosine vs linear, trained — `sched_lin_2026-09-09_17-49-05` vs `sched_cos_2026-09-09_17-53-11`

5 epochs each, `--seed 0`, identical in every flag but `--schedule`. Same init,
shuffle, `t` draws and noise, so the schedule is the only difference. (Seeding
arrived with this comparison; every earlier run in this file is unseeded.)

**The printed loss is meaningless across schedules.** Linear ends at 0.0232 and
cosine at 0.0400, and this says nothing — the same `t` is a different amount of
noise under each, so the two numbers are averages over different tasks. Cosine's
number is larger because cosine *spends more of its steps* at low noise, where
eps-MSE is high. Reading these two scalars against each other would say linear
won by 70%.

Matched on ᾱ instead — at equal ᾱ the input `x_t` is literally the same tensor,
and only the `t` index each net is told differs — on the held-out set, EMA
weights:

| ᾱ | t_lin | t_cos | linear | cosine | lin/cos |
| --- | --- | --- | --- | --- | --- |
| 0.99 | 27 | 56 | 0.10597 | 0.10480 | 1.011x |
| 0.90 | 97 | 198 | 0.05892 | 0.05755 | 1.024x |
| 0.70 | 184 | 363 | 0.03784 | 0.03718 | 1.018x |
| 0.50 | 258 | 495 | 0.02928 | 0.02882 | 1.016x |
| 0.30 | 342 | 627 | 0.02269 | 0.02253 | 1.007x |
| 0.10 | 475 | 793 | 0.01308 | 0.01314 | 0.995x |
| 0.03 | 587 | 887 | 0.00547 | 0.00571 | 0.959x |
| 0.01 | 673 | 935 | 0.00213 | 0.00232 | 0.919x |
| 0.001 | 825 | 979 | 0.00036 | 0.00057 | 0.629x |

The trade is exactly the one the schedule was designed to make, and nothing
more. Cosine is 1-2% better across the low-noise half, where it now spends its
steps; linear is better in the high-noise tail, by up to 58% at ᾱ=0.001, because
it spends half its budget there. Capacity moved; it was not created.

No visual winner at this scale (`grid.png` in each run, same `x_T`, same labels):
both are clean digits, across-sample std 0.3675 linear vs 0.3790 cosine.

**Verdict: cosine is not a win on MNIST at 5 epochs.** The Improved-DDPM result
is on 64x64 ImageNet, where the model is capacity-bound and the wasted
high-noise steps are a real cost; here a 4.17M-param net on 28x28 has capacity
to spare, so buying low-noise accuracy with high-noise accuracy nets out flat.
The cheap 1-2% gain sits in the region that matters most for perceptual detail,
which is the argument for keeping it — not the loss numbers. Kept: `--schedule`
defaults to cosine from 2026-09-09.

## Classifier-free guidance — `cfg_2026-09-09_18-17-27`

8 epochs, cosine, label dropout 0.1, EMA weights, DDIM 50. Obedience is judged
by a small CNN trained on MNIST for the purpose (98.3% test accuracy); diversity
is across-sample pixel std within each class, at a fixed `x_T` shared by every w.

| w | label accuracy | diversity | pixels at ±1 |
| --- | --- | --- | --- |
| 0.0 | 0.065 | 0.3631 | 0.499 |
| 1.0 | 0.945 | 0.3091 | 0.510 |
| 1.5 | 0.990 | 0.2948 | 0.519 |
| 2.0 | 0.995 | 0.2876 | 0.513 |
| 3.0 | 1.000 | 0.2798 | 0.480 |
| 5.0 | 1.000 | 0.2768 | 0.381 |
| 8.0 | 1.000 | 0.2789 | 0.239 |
| 15.0 | 1.000 | 0.2966 | 0.115 |

The trade is exactly as advertised, and it is cheap: **w=1.5 buys 4.5 points of
obedience for 5% of the diversity**, and by w=3 the judge is never wrong. w=0 is
the sanity check — 6.5%, below the 10% a coin would get, because unconditional
samples are not reliably any digit.

Two MNIST-specific surprises:

**Diversity is not monotonic.** It bottoms out at w=5 (0.2768) and *rises* again
by w=15 (0.2966). That is not returning variety — it is damage. Past w≈8 the
strokes erode into hollow, speckled outlines, and the speckle is pixel variance
the metric cannot tell apart from genuine variation. Any diversity number needs
the grid beside it.

**Nothing oversaturates.** The literature's "oversaturated garbage" at high w is
an RGB failure; here the fraction of pixels pinned at ±1 *falls* by more than
half, 0.51 → 0.115. Overshooting eps thins the strokes rather than blowing out
colour, and DDIM's x0 clamp absorbs what is left. Same mechanism, different
symptom — the failure mode is legibility, not saturation.

Cost is one extra forward per step at w≠1, batched into one pass. `Guided`
short-circuits to a single call at w=1 exactly, where the two branches cancel.

Reproduced in `cfg.ipynb`. Re-running moves the top of the accuracy curve by
about half a point (w=2 read 0.995 once and 1.000 the next time) — cuDNN is
nondeterministic, and `--seed` pairs runs without making them bitwise equal.
Differences that small are not findings.

## Self-attention in the bottleneck — `attn_off_2026-09-09_18-51-37` vs `attn_on_2026-09-09_18-58-11`

8 epochs each, `--seed 0`, identical but for `--attention`. One `Attention` block
between the two bottleneck ResBlocks: +66k params (4.173M → 4.239M), +1s/epoch.

Held-out eps-MSE by `t` — same schedule and seed, so the buckets line up:

| t | off | on | ratio |
| --- | --- | --- | --- |
| 0-99 | 0.10915 | 0.10898 | 1.002x |
| 300-399 | 0.03746 | 0.03755 | 0.998x |
| 600-699 | 0.02072 | 0.02079 | 0.997x |
| 900-999 | 0.00171 | 0.00171 | 0.996x |
| pooled | 0.03680 | 0.03680 | 1.000x |

**No effect.** Every bucket is within 0.4%, and the pooled numbers are identical
to five decimals. Label accuracy 0.920 off vs 0.935 on at w=1, which is inside
the ±1 point this setup wobbles by; both are 1.000 at w=3.

Expected, and worth stating plainly: at 28x28 with a 7x7 bottleneck, a stack of
3x3 convs already has a receptive field covering the whole image by the time it
reaches the middle. Attention's advantage is reach, and there is no reach left to
buy. It pays on 64x64+ where the bottleneck is still large relative to the
features that must agree.

So this block earns its place as **infrastructure, not quality**: the same class
with `context_dim` set is cross-attention, which is the only mechanism that can
bind "red" to "3" and "top left" to a position. That is the next item, and it is
where this will be measured properly.
