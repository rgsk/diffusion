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
- `unet.py` — 28→14→7, mults (1,2,2), 4.17M params (4.72M with cross-attention).
  Zero-init output conv.
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
  the cross-attention path it provides, which is where it earned its place.
- `sample.py` — ask a trained run for digits: `python sample.py <run> --label 3
  --n 16 --w 3`, or `--prompt "a red 3 in the top left"` for a captioned run.
  Rebuilds the net from the checkpoint alone (num_classes, vocab_size, channels,
  image size, attention, schedule), EMA weights by default, `--label -1` for one
  class per row. Older checkpoints are read off their own weights.
- `class_conditioning.ipynb` — probes against a trained checkpoint: label sweep
  at fixed `x_T`, the null row, right-vs-wrong label by `t`, conditional vs
  unconditional by `t`. Cell 1 is all imports and helpers; every probe below
  runs on its own.
- `colored_mnist.py` — 32x32 RGB MNIST with a caption per image: colour, digit
  and position drawn independently, and "a red 3 in the top left" written from
  the attributes that drew it. A 26-word closed vocabulary with a reserved
  `<null>` token (index 0, the caption analogue of the null label row) and a
  separate `<pad>`, so a short caption is not accidentally a partly-unconditional
  one. Attributes are a function of the MNIST index, so the set is identical
  every epoch and the held-out filter can run without loading an image.
  `HELDOUT` drops six colour x digit pairs — one per colour, six distinct digits,
  10% of the data — leaving every word trained and six combinations unseen.
  `read_color`/`read_position` recover the attributes from pixels, and are
  checked against the dataset that wrote them.
- **cross-attention through the U-Net** — `UNet(vocab_size=...)` embeds tokens
  into a context sequence and reads it with an `Attention(context_dim=...)` after
  every ResBlock and in the bottleneck. Deliberately *not* added to `temb`: the
  claim under test is that a pooled vector cannot bind an attribute to an
  object, so the caption's only route to a pixel is attention — asserted by
  zeroing the attention output projections on a trained net and watching the
  caption go inert. (That claim did not survive its control run on
  single-object images; see `results.md` and item 1 below.) `y` is `[B]` labels or `[B, L]` tokens, so `Conditioned`, `Guided` and
  both samplers are unchanged. `drop_labels` now drops whole captions rather than
  individual words. State dict keys are untouched when `vocab_size` is None, so
  every earlier checkpoint still loads.
- `main.py --dataset colored` — trains the captioned net; the per-epoch grid is
  every colour x every digit, so the held-out cells are in the picture from
  epoch 1.
- `two_objects.py` — two digits on one canvas, distinct colours, digits and
  corners, one caption naming both. The setting where the assignment is not
  recoverable from the multiset of words, which single-object images never were.
  `swap_colors` exchanges the two colour words (the probe the eval rests on) and
  `isolate` blanks all but one corner, so the single-object judge scores a
  two-object image without retraining.
- `binding.py` — scores `colours present` (what a bag of words can get right)
  against `colours bound` (which colour went where). Measured in `results.md`:
  **both mechanisms assign at chance**, and cross-attention is no better than
  pooling, because `token_pos` trained to 2% of the token embedding norm and the
  context is therefore still a bag. The failure mode is the clean swap, counted
  directly.
- `text_encoder.py` — pre-norm self-attention + MLP over `[B, L, D]`, so each
  word's vector is rewritten in terms of the words around it before any pixel
  attends to it: the stage a real model gets from CLIP. Zero-init residual
  projections, as everywhere else here, and it does not own the embedding table,
  so every earlier captioned checkpoint still loads as the control.
  `UNet(text_layers=n)`, `main.py --text-layers`. Measured in `results.md`: it
  trains, it contextualises, and it **does not fix binding** — the failure is on
  the image side of the attention, not the text side.
- `compositional.py` — the held-out eval. Generates every colour x digit pair,
  scores colour and position off the pixels and the digit with a small CNN judge
  trained on the *full* dataset (a judge that never saw a red 3 cannot grade
  one), and reports seen-vs-held-out. The seen column is the control for judge
  error. When a held-out pair fails it also reports which word was dropped —
  kept the colour and got the digit wrong, or the reverse — which is what
  separates recombining factors from recalling pairs.
- `UNet(pooled=True)` / `main.py --pooled-context` — the control the item above
  needed: the same tokens meaned into one vector and added to `temb`, the way a
  class label is, with no attention anywhere. Measured in `results.md`:
  **compositional generalization is complete (held-out 0.990 vs seen 0.993) and
  cross-attention is not what produced it** — the pooled net matches it while
  being provably a bag of words (shuffling the caption changes its output by
  exactly 0.00000). One object per image means there is nothing to mis-bind, so
  the dataset never creates the ambiguity cross-attention exists to resolve. The
  eps-MSE and the per-epoch grids are both blind to the distinction; only the
  control run separated the mechanisms, by showing they are the same here.

## Next

Ordered. (1) is unfinished business: the binding item built the mechanism, found
the task that needs it, and then found that the mechanism alone is not enough.

1. **Spatial coordinates in the image stream** — a 2D positional embedding
   added to the feature maps the cross-attention queries are built from.
   `results.md` has three runs failing binding at chance (pooled, cross-attention,
   cross-attention + text encoder) and a diagnosis that rules out the text side:
   the encoder demonstrably contextualises, and the U-Net's eps is *identical*
   for a caption and its colour-swapped twin at every noise level, so nothing
   about the assignment is ever encoded. Cross-attention binds by matching a
   query to the words about it, and the query is a feature vector at a location
   that, after translation-equivariant convolutions, does not know where it is.
   Add coordinates and the query can say *I am the top-left one*. `binding.py` is
   the eval unchanged and three runs are the control. If this also fails at
   chance, the next suspects are capacity at 128-dim context for two full object
   descriptions, and a loss that pays ~1.6x for the caption and 1.0x for its
   internal structure.

2. **Flow matching / rectified flow** — DDPM/DDIM is the SD1/SD2-era
   formulation; SD3 and Flux use a straight-line path from noise to data,
   predicting velocity instead of eps. Simpler than what is already written —
   no beta schedule, no posterior-variance algebra — and it slots in beside
   `sampler.py` as a peer, sharing the same U-Net. The item that makes this
   project about image generation as practised now rather than as it was.
