# Roadmap

Updated 2026-09-05. One concept per file in `src/`, each with a `__main__` that
tests it. Run any file directly.

## Done

- `forward_process.py` — linear beta schedule + `q_sample`, buffers on an
  `nn.Module`. `extract` lives here.
- `timestep_embedding.py` — sinusoidal `t -> [B, dim]`, block layout.
- `resblock.py` — GroupNorm/SiLU/conv ×2, temb added in the middle, zero-init
  `conv2` so a fresh block is exactly its skip.
- `unet.py` — 28→14→7, mults (1,2,2), 4.17M params. Zero-init output conv.
- `sampler.py` — ancestral DDPM, `sigma="beta"|"posterior"`.
- `main.py` — train on MNIST, sample a grid per epoch into `artifacts/`.

## Results

15 epochs on MNIST, RTX 4060: 48s/epoch train, 15s to sample 64. Loss 0.0740 →
0.0267 (epoch 2) → 0.0221 (epoch 15). Samples are clean, well-formed digits by
epoch 15; epoch 5 is visibly worse.

**The loss lies.** It looks flat from epoch 2 on, while sample quality keeps
improving a lot. eps-MSE is dominated by high `t`, where nobody beats chance, so
the average mostly measures an irreducible floor. Don't early-stop on it and
don't use it to compare models.

Sample range settles around `[-1.05, 1.09]` — slightly outside the data, since
sampling doesn't clip.

## Next

1. **DDIM** — deterministic, 50 steps instead of 1000. Turns 15s into <1s.
2. **Loss bucketed by `t`** — the metric that tracks what the eye sees.
3. **Self-attention** at 7×7, wired into the U-Net.
4. **`clip_denoised`** — clamp implied x0 to [-1,1] each step.
5. **EMA of weights** — standard in DDPM, usually a visible quality win.
6. **Cosine schedule** — linear betas destroy MNIST's signal early.

## Housekeeping

Done: `data/` untracked and gitignored, `ruff` added (config in `pyproject.toml`,
notebooks excluded), `src/` formatted.

Open: the MNIST blobs are still in git history, so `.git` stays ~25MB. Purging
them means rewriting history and force-pushing to `origin`.
