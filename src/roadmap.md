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
- `class_conditioning.ipynb` — probes against a trained checkpoint: label sweep
  at fixed `x_T`, the null row, right-vs-wrong label by `t`, conditional vs
  unconditional by `t`. Cell 1 is all imports and helpers; every probe below
  runs on its own.

## Next

Ordered. (1)-(2) are knobs, an afternoon each. (3)-(5) are the rest of the
mechanism by which a model is told what to make. (6) is the formulation today's
models use instead of the one implemented above.

1. **EMA of weights** — standard in DDPM, usually a visible quality win.
2. **Cosine schedule** — linear betas destroy MNIST's signal early.

3. **Classifier-free guidance** — the engine of every conditional model. Drop
   the label to a null token with p≈0.1 during training, so one net learns both
   `eps(x_t, t, y)` and `eps(x_t, t, ∅)`; at sample time
   `eps = eps_uncond + w * (eps_cond - eps_uncond)`. Sweep `w` and watch: 0 is
   unconditional, ~3 is sharp and obedient, ~15 is oversaturated garbage.
   Guidance does not sample the conditional distribution — it samples a
   sharpened `p(x)·p(y|x)^w`, so the diversity loss at high `w` is the
   mechanism, not a bug. `sampler.py`'s Gaussian oracle can measure exactly
   that: std shrinking as `w` rises.
4. **Attention** — self-attention at 7×7, and cross-attention (image positions
   as queries, conditioning tokens as keys/values). Same block, different KV
   source. Not a polish item: it is the prerequisite for (5).
5. **Colored digits on 32×32 with synthetic captions** — "a red 3 in the top
   left". Adding a pooled vector to `temb` is enough for 10 classes and fails
   for sentences, because binding an attribute to an object needs
   cross-attention. Hold out some colour×digit combinations from training and
   check whether they can be generated: a real compositionality test at MNIST
   cost. Real text-to-image adds a captioned dataset, a frozen text encoder,
   and a VAE for latent diffusion — four new systems and a training run too
   expensive to iterate on. This teaches the same lesson in minutes.

6. **Flow matching / rectified flow** — DDPM/DDIM is the SD1/SD2-era
   formulation; SD3 and Flux use a straight-line path from noise to data,
   predicting velocity instead of eps. Simpler than what is already written —
   no beta schedule, no posterior-variance algebra — and it slots in beside
   `sampler.py` as a peer, sharing the same U-Net. The item that makes this
   project about image generation as practised now rather than as it was.
