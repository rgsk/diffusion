"""Did it put the right colour on the right digit?

`compositional.py` asked whether a model can draw a red 3 having never seen one.
Both mechanisms could, so it did not separate them. This asks the question that
does: given "a red 3 in the top left and a blue 7 in the bottom right", is the
top-left digit a *red 3*, or a blue one?

The two failure modes are worth naming, because only one of them is interesting:

    bound    top left is a red 3, bottom right is a blue 7      -- correct
    swapped  top left is a blue 3, bottom right is a red 7      -- every word
                                                                   present, both
                                                                   assignments wrong

A model conditioned on a pooled caption is handed {red, 3, blue, 7} and no
assignment. It should score well on *which colours appear* and badly on *which
colour went where*, and the gap between those two columns is the binding failure
measured directly. Cross-attention has the assignment available to it, in the
attention weights.

Scoring reuses `compositional.py`'s judge without retraining: `isolate` blanks
everything outside one corner, which is exactly the single-object canvas the
judge was trained on. The instruments are checked against real dataset images
first, and that line is the error bar on everything below.

    python binding.py pair_cross_2026-09-09_22-00-00 --prompts 120 --per 8
"""

import argparse
import random

import torch
from torch import Tensor
from torchvision.utils import save_image

from cfg import Guided
from colored_mnist import COLORS, DIGITS, decode, encode_batch, read_color
from compositional import judge_accuracy, train_judge
from ddim import DDIMSampler
from sample import load
from two_objects import (
    CORNERS,
    SEQ_LEN_PAIR,
    TwoObjectMNIST,
    caption_pair,
    isolate,
    swap_colors,
)
from utils import repo_root


def probe_prompts(n: int, per: int, seed: int = 0) -> tuple[list[str], Tensor]:
    """n distinct two-object prompts, `per` samples of each. Colours, digits and
    corners are all distinct within a prompt, so both failure modes are visible."""
    r = random.Random(seed)
    texts, spec = [], []
    seen = set()
    while len(spec) < n:
        c1, c2 = r.sample(range(len(COLORS)), 2)
        d1, d2 = r.sample(range(len(DIGITS)), 2)
        p1, p2 = r.sample(range(len(CORNERS)), 2)
        if (key := (c1, d1, p1, c2, d2, p2)) in seen:
            continue
        seen.add(key)
        spec.append(key)
        texts.append(
            caption_pair(
                COLORS[c1], DIGITS[d1], CORNERS[p1], COLORS[c2], DIGITS[d2], CORNERS[p2]
            )
        )
    out_t = [t for t in texts for _ in range(per)]
    out_s = torch.tensor(spec).repeat_interleave(per, dim=0)
    return out_t, out_s


@torch.no_grad()
def read_objects(x: Tensor, spec: Tensor, judge, dev: str, batch: int = 256):
    """-> (colour at slot 1, digit at slot 1, colour at slot 2, digit at slot 2)."""
    outs = []
    for k in (2, 5):  # the position column of each object
        iso = isolate(x, spec[:, k])
        col = read_color(iso)
        dig = torch.cat(
            [
                judge(iso[i : i + batch].to(dev)).argmax(1).cpu()
                for i in range(0, len(iso), batch)
            ]
        )
        outs += [col, dig]
    return outs


def score(spec: Tensor, read) -> dict[str, Tensor]:
    """Booleans per sample. `set` columns ignore the assignment -- they are what a
    bag of words can get right -- and the bound columns require it."""
    ca, da, cb, db = read
    c1, d1, c2, d2 = spec[:, 0], spec[:, 1], spec[:, 3], spec[:, 4]
    cset = ((ca == c1) & (cb == c2)) | ((ca == c2) & (cb == c1))
    dset = ((da == d1) & (db == d2)) | ((da == d2) & (db == d1))
    return {
        "colours present": cset,
        "colours bound": (ca == c1) & (cb == c2),
        "colours swapped": (ca == c2) & (cb == c1) & (c1 != c2),
        "digits present": dset,
        "digits bound": (da == d1) & (db == d2),
        "both bound": (ca == c1) & (cb == c2) & (da == d1) & (db == d2),
    }


def table(name: str, s: dict[str, Tensor]) -> str:
    return f"{name:22} " + "  ".join(
        f"{k} {v.float().mean():.3f}" for k, v in s.items()
    )


def main(
    run: str,
    prompts: int = 120,
    per: int = 8,
    w: float = 3.0,
    steps: int = 50,
    batch: int = 64,
    weights: str = "ema",
    seed: int = 0,
):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net, fp, shape = load(run, dev, weights)
    assert net.vocab_size is not None, f"{run} is not a captioned run"

    judge = train_judge(dev)
    acc, _ = judge_accuracy(judge, dev)
    print(f"judge on real single-object test digits: {acc:.4f}")

    # the instruments, on real two-object images. Whatever this line falls short
    # of is measurement error, not model error, in every number below.
    ds = TwoObjectMNIST(train=False)
    idx = list(range(0, len(ds), max(1, len(ds) // 512)))[:512]
    xr = torch.stack([ds[i][0] for i in idx])
    words = [decode(ds[i][1]).split() for i in idx]
    spec_r = torch.tensor(
        [
            [
                COLORS.index(w[1]),
                DIGITS.index(w[2]),
                CORNERS.index(" ".join(w[5:7])),
                COLORS.index(w[9]),
                DIGITS.index(w[10]),
                CORNERS.index(" ".join(w[13:15])),
            ]
            for w in words
        ]
    )
    print(
        table(
            "control (real data)", score(spec_r, read_objects(xr, spec_r, judge, dev))
        )
    )

    texts, spec = probe_prompts(prompts, per, seed)
    print(f"\n{len(texts)} samples from {prompts} prompts, w={w}, {steps} DDIM steps")
    smp = DDIMSampler(fp, steps=steps).to(dev)
    xs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            y = encode_batch(texts[i : i + batch], dev, SEQ_LEN_PAIR)
            xs.append(smp.sample(Guided(net, y, w), (y.shape[0], *shape), dev).cpu())
            print(f"  {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    x = torch.cat(xs)
    s = score(spec, read_objects(x, spec, judge, dev))
    print()
    print(table(run.split("_20")[0], s))
    print(
        f"\n  of {int((~s['both bound']).sum())} failures, "
        f"{int(s['colours swapped'].sum())} are a clean colour swap "
        f"({s['colours swapped'].float().mean():.3f} of all samples)"
    )
    print(
        f"  gap between 'colours present' and 'colours bound': "
        f"{(s['colours present'].float().mean() - s['colours bound'].float().mean()):.3f}"
        "  <- the binding failure"
    )

    # a prompt and its colour swap, side by side: if these two rows look the
    # same, the model is not reading the assignment
    base = texts[0]
    pair_texts = [base] * 8 + [swap_colors(base)] * 8
    with torch.no_grad():
        y = encode_batch(pair_texts, dev, SEQ_LEN_PAIR)
        xp = smp.sample(Guided(net, y, w), (16, *shape), dev).cpu()
    path = repo_root() / "artifacts" / run / f"binding_w{w:g}.png"
    save_image(xp, path, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"\n  '{base}'\n  '{swap_colors(base)}'\n  -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", help="folder under artifacts/")
    p.add_argument("--prompts", type=int, default=120, help="distinct captions")
    p.add_argument("--per", type=int, default=8, help="samples per caption")
    p.add_argument("--w", type=float, default=3.0, help="guidance scale")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--batch", type=int, default=64, help="doubled by guidance at w!=1")
    p.add_argument("--weights", default="ema", choices=("ema", "net"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
