# Roadmap

Updated 2026-09-09. One concept per file in `src/`, each with a `__main__` that
tests it. Run any file directly. Direction only — measurements and what they
showed live in `results.md`.

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
  `--ddim-steps`, `--num-classes`. Runs land in `artifacts/scratch` (wiped each
  run) or `artifacts/{name}_{timestamp}`, with `train.log` written beside the
  grids.
- **class conditioning** — `UNet(num_classes=...)` embeds the label into `temb`,
  with a reserved untrained null row at index `num_classes` so CFG can reuse
  these weights. `Conditioned(net, y)` freezes `y` back into the `(x, t) -> eps`
  interface the samplers call, so neither sampler changed. `main.py` conditions
  by default and draws one class per grid row.
- `loss_by_t.py` — per-sample eps-MSE accumulated into `t` buckets from the
  training draws themselves, no extra forward passes. `main.py` prints a row
  under a header of ranges each epoch (`--loss-buckets`, default 10); the epoch
  scalar is now the accumulator's count-weighted pooled mean.
- `ema.py` — `EMA`, a shadow copy of the weights trailing the trained ones
  (`min(decay, (1+n)/(10+n))` ramp so early epochs are not an average with the
  random init), plus an `as_weights` context manager that swaps them in for
  sampling and restores exactly. A pure observer: the training path is
  unchanged. `main.py` updates it per step, draws the epoch grid under it, and
  saves it beside `net` in the checkpoint. `--ema-decay 0` disables.
- `cosine_schedule.py` — `cosine_betas`, the Nichol & Dhariwal ᾱ curve turned
  back into betas, with `s` off the flat peak and a `max_beta` cap at the t=T
  singularity. `ForwardProcess(schedule="cosine")` and `main.py --schedule`.
  Compared in closed form (`cosine_schedule.ipynb`) and trained head-to-head
  against linear (`results.md`): the trade is real but nets out flat on MNIST.
  Kept as the default anyway — the 1-2% it buys sits at low `t`, where detail is.
- `main.py --seed` — init, shuffling and noise, so two runs differing in one
  flag differ in that flag alone. Pairs runs; does not make them bitwise
  reproducible (cuDNN autotuning and atomics), which would need
  `use_deterministic_algorithms` and a throughput cost.
- `cfg.py` — classifier-free guidance. `drop_labels` swaps a fraction of labels
  for the reserved null token during training; `Guided(net, y, w)` freezes
  `(y, w)` into the samplers' `(x, t) -> eps` interface and returns
  `eps_∅ + w·(eps_y - eps_∅)`, both branches in one doubled batch. `main.py
  --label-dropout` (0.1 by default) and `--guidance`. Swept in `results.md`:
  w=1.5 costs 5% diversity for 99% label accuracy. `cfg.ipynb` sweeps it: the
  two curves, and the grids that show what the diversity number stops meaning
  past w≈8.
- `attention.py` — `Attention(ch, heads, groups, context_dim=None)`: self when
  `context_dim` is None, cross when it is set, zero-init output projection so a
  fresh block is its own skip. `UNet(attention=True)` puts one in the 7x7
  bottleneck (`main.py --attention`). Measured in `results.md`: **no effect on
  MNIST** — the convs already reach the whole image by the bottleneck. Kept for
  the cross-attention path it provides.
- `sample.py` — ask a trained run for digits: `python sample.py <run> --label 3
  --n 16 --w 3`. Rebuilds the net from the checkpoint alone (num_classes,
  attention, schedule), EMA weights by default, `--label -1` for one class per
  row. Checkpoints now record `attention` too; older ones are read off their own
  weights.
- `class_conditioning.ipynb` — probes against a trained checkpoint: label sweep
  at fixed `x_T`, the null row, right-vs-wrong label by `t`, conditional vs
  unconditional by `t`. Cell 1 is all imports and helpers; every probe below
  runs on its own.

## Next

Ordered. (1) is the last piece of the mechanism by which a model is told what to
make. (2) is the formulation today's models use instead of the one implemented
above.

1. **Colored digits on 32×32 with synthetic captions** — "a red 3 in the top
   left". Adding a pooled vector to `temb` is enough for 10 classes and fails
   for sentences, because binding an attribute to an object needs
   cross-attention. `attention.py` has the block; threading a context through
   `UNet` is part of this item. Hold out some colour×digit combinations from
   training and check whether they can be generated: a real compositionality
   test at MNIST cost. Real text-to-image adds a captioned dataset, a frozen text encoder,
   and a VAE for latent diffusion — four new systems and a training run too
   expensive to iterate on. This teaches the same lesson in minutes.

2. **Flow matching / rectified flow** — DDPM/DDIM is the SD1/SD2-era
   formulation; SD3 and Flux use a straight-line path from noise to data,
   predicting velocity instead of eps. Simpler than what is already written —
   no beta schedule, no posterior-variance algebra — and it slots in beside
   `sampler.py` as a peer, sharing the same U-Net. The item that makes this
   project about image generation as practised now rather than as it was.
