"""Can it draw a red 3, having never seen one?

`ColoredMNIST` holds out six colour x digit pairs. Red 3 is absent from training;
red 5s and green 3s are not. So the model has seen every word in "a red 3 in the
top left" and never that combination of them.

    memorised the pairs  ->  asking for a red 3 returns a green 3, or a red 5
    learned the factors  ->  it returns a red 3

That is compositional generalization, and it is the property text-to-image runs
on: no captioned dataset contains every combination anyone will ask for.

Scoring needs three instruments. Colour and position are read straight off the
pixels (`read_color`, `read_position` -- tested against the dataset itself, which
is where they can actually be wrong). The digit needs a classifier, trained here
on the *full* dataset including the held-out pairs: a judge that has never seen a
red 3 cannot score one. The judge is a ruler, not a subject.

The control for judge error is the seen pairs. Both columns are scored by the
same instrument on the same kind of image, so the gap between them is the answer
regardless of what the judge's absolute accuracy is.

    python compositional.py caption_2026-09-09_20-30-00 --n 16 --w 3
"""

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cfg import Guided
from colored_mnist import (
    COLORS,
    DIGITS,
    HELDOUT,
    POSITIONS,
    ColoredMNIST,
    caption,
    encode_batch,
    read_color,
    read_position,
)
from ddim import DDIMSampler
from sample import load
from utils import repo_root


class Judge(nn.Module):
    """Reads the digit off a 32x32 colored canvas. Small on purpose -- it only has
    to be a better digit reader than the diffusion model is a digit writer."""

    def __init__(self, ch: int = 32):
        super().__init__()

        def block(i, o, stride):
            return [nn.Conv2d(i, o, 3, stride, 1), nn.BatchNorm2d(o), nn.ReLU()]

        self.body = nn.Sequential(
            *block(3, ch, 1),  # 32
            *block(ch, ch, 2),  # 16
            *block(ch, 2 * ch, 2),  # 8
            *block(2 * ch, 4 * ch, 2),  # 4
            # flatten rather than global-average-pool: pooling to 1x1 is
            # translation invariant, which sounds right for a digit that moves
            # around the canvas and in fact discards the spatial layout that
            # tells a 3 from an 8. Measured: 0.936 pooled vs 0.99 flattened.
            nn.Flatten(),
            nn.Linear(4 * ch * 16, len(DIGITS)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def train_judge(dev: str, epochs: int = 5, batch: int = 128, force: bool = False):
    """Cached in artifacts/, since it depends on the dataset and not on any run.
    Trained with exclude_heldout=False -- see the module docstring."""
    path = repo_root() / "artifacts" / "judge.pt"
    judge = Judge().to(dev)
    if path.exists() and not force:
        try:
            judge.load_state_dict(torch.load(path, map_location=dev))
            return judge.eval()
        except RuntimeError as e:  # cache predates an architecture change
            print(
                f"  cached judge does not fit this Judge ({e.__class__.__name__});"
                " retraining"
            )
    tr = ColoredMNIST(train=True, exclude_heldout=False)
    loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=4)
    opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for ep in range(epochs):
        judge.train()
        for x, tok in loader:
            # the digit word is token 2 of "a <colour> <digit> in the <pos>", and
            # DIGITS sits at a known offset in the vocabulary
            d = tok[:, 2].to(dev) - min(_digit_ids())
            loss = F.cross_entropy(judge(x.to(dev)), d)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  judge epoch {ep + 1}  loss {loss.item():.4f}")
    torch.save(judge.state_dict(), path)
    return judge.eval()


def _digit_ids() -> list[int]:
    """The digit words occupy a contiguous block of the vocabulary, so `id - lo`
    is the class index. Asserted, because a vocabulary edit that breaks it would
    silently mislabel every training target and every score below."""
    from colored_mnist import WORD2ID

    ids = [WORD2ID[d] for d in DIGITS]
    assert ids == list(range(min(ids), min(ids) + len(DIGITS))), ids
    return ids


@torch.no_grad()
def judge_accuracy(judge: Judge, dev: str, batch: int = 256) -> tuple[float, float]:
    """Accuracy on the real MNIST *test* split, overall and on the held-out pairs
    alone. The second number is the one that matters: it is the error bar on
    every held-out score below."""
    te = ColoredMNIST(train=False, exclude_heldout=False)
    loader = DataLoader(te, batch_size=batch, num_workers=4)
    lo = min(_digit_ids())
    heldout = {(COLORS.index(c), DIGITS.index(d)) for c, d in HELDOUT}
    hits = tot = h_hits = h_tot = 0
    for x, tok in loader:
        d = tok[:, 2] - lo
        pred = judge(x.to(dev)).argmax(1).cpu()
        col = read_color(x)
        mask = torch.tensor([(int(a), int(b)) in heldout for a, b in zip(col, d)])
        hits += int((pred == d).sum())
        tot += len(d)
        h_hits += int((pred == d)[mask].sum())
        h_tot += int(mask.sum())
    return hits / tot, h_hits / max(h_tot, 1)


def probe_prompts(n: int) -> tuple[list[str], Tensor]:
    """Every colour x every digit, n samples each, positions cycled so position is
    scored on something that varies rather than a constant."""
    texts, want = [], []
    for ci, c in enumerate(COLORS):
        for di, d in enumerate(DIGITS):
            for k in range(n):
                pi = k % len(POSITIONS)
                texts.append(caption(c, d, POSITIONS[pi]))
                want.append((ci, di, pi))
    return texts, torch.tensor(want)


@torch.no_grad()
def generate(net, fp, texts: list[str], w: float, steps: int, batch: int, dev: str):
    smp = DDIMSampler(fp, steps=steps).to(dev)
    out = []
    for i in range(0, len(texts), batch):
        y = encode_batch(texts[i : i + batch], dev)
        out.append(smp.sample(Guided(net, y, w), (y.shape[0], 3, 32, 32), dev).cpu())
        print(f"  {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    return torch.cat(out)


def table(rows: list[tuple[str, Tensor]]) -> str:
    head = f"{'':16} {'colour':>8} {'digit':>8} {'position':>9} {'all three':>10}"
    out = [head, "-" * len(head)]
    for name, hit in rows:
        both = hit.all(1).float().mean()  # all(1) needs the bools, mean() needs floats
        c, d, p = hit.float().mean(0)
        out.append(f"{name:16} {c:8.3f} {d:8.3f} {p:9.3f} {both:10.3f}")
    return "\n".join(out)


def main(
    run: str,
    n: int = 16,
    w: float = 3.0,
    steps: int = 50,
    batch: int = 60,
    weights: str = "ema",
    seed: int = 0,
    retrain_judge: bool = False,
):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net, fp, shape = load(run, dev, weights)
    assert net.vocab_size is not None, f"{run} is not a captioned run"
    assert shape == (3, 32, 32), shape

    judge = train_judge(dev, force=retrain_judge)
    acc, h_acc = judge_accuracy(judge, dev)
    print(
        f"judge on real test digits: {acc:.4f} overall, {h_acc:.4f} on held-out pairs"
    )
    assert acc > 0.95, f"judge too weak to measure anything: {acc}"

    texts, want = probe_prompts(n)
    print(f"{len(texts)} samples, w={w}, {steps} DDIM steps")
    x = generate(net, fp, texts, w, steps, batch, dev)

    got = torch.stack(
        [
            read_color(x),
            judge(x.to(dev)).argmax(1).cpu(),
            read_position(x),
        ],
        dim=1,
    )
    hit = got == want
    heldout = torch.tensor(
        [(COLORS[c], DIGITS[d]) in HELDOUT for c, d in want[:, :2].tolist()]
    )
    print()
    print(table([("seen (54 pairs)", hit[~heldout]), ("held out (6)", hit[heldout])]))

    # per-pair, so one bad pair cannot hide behind five good ones
    print("\nheld-out pairs, one at a time:")
    for c, d in HELDOUT:
        m = heldout & (want[:, 0] == COLORS.index(c)) & (want[:, 1] == DIGITS.index(d))
        h = hit[m]
        print(
            f"  {c:8} {d}   colour {h[:, 0].float().mean():.2f}  "
            f"digit {h[:, 1].float().mean():.2f}  both {h[:, :2].all(1).float().mean():.2f}"
        )

    # when it fails, which way does it fall back? A memoriser has to drop one of
    # the two words, and which one it drops says what it did instead of composing.
    miss = heldout & ~hit[:, :2].all(1)
    if int(miss.sum()):
        kc = (hit[miss][:, 0] & ~hit[miss][:, 1]).float().mean()
        kd = (~hit[miss][:, 0] & hit[miss][:, 1]).float().mean()
        both = (~hit[miss][:, 0] & ~hit[miss][:, 1]).float().mean()
        print(
            f"\nof {int(miss.sum())} held-out misses: kept the colour, wrong digit "
            f"{kc:.2f} | kept the digit, wrong colour {kd:.2f} | lost both {both:.2f}"
        )

    # one row per held-out pair, so the numbers above have a picture beside them
    cols = min(n, 12)
    keep = torch.cat(
        [
            torch.nonzero(
                (want[:, 0] == COLORS.index(c)) & (want[:, 1] == DIGITS.index(d))
            )[:cols, 0]
            for c, d in HELDOUT
        ]
    )
    path = repo_root() / "artifacts" / run / f"heldout_w{w:g}.png"
    save_image(x[keep], path, nrow=cols, normalize=True, value_range=(-1, 1))
    print(f"\nheld-out grid ({', '.join(f'{c} {d}' for c, d in HELDOUT)}) -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", help="folder under artifacts/")
    p.add_argument("--n", type=int, default=16, help="samples per colour x digit pair")
    p.add_argument("--w", type=float, default=3.0, help="guidance scale")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--batch", type=int, default=60, help="doubled by guidance at w!=1")
    p.add_argument("--weights", default="ema", choices=("ema", "net"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrain-judge", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
