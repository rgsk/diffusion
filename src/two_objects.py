"""Two digits on one canvas, and the binding problem for real.

`colored_mnist` put one object in the image, and `README.md` records what that
cost: a caption pooled into a single vector scored exactly as well as
cross-attention, because a bag {red, 3, top-left} is unambiguous when there is
only one slot to empty it into. The mechanism was built and never tested.

Two objects is the smallest change that creates the ambiguity:

    "a red 3 in the top left and a blue 7 in the bottom right"

Pooled, that is {red, 3, blue, 7, top-left, bottom-right} with no record of which
colour belongs to which digit. A sum cannot represent the answer -- not because
it lacks capacity, but because the assignment is not a function of the multiset.
Cross-attention can: each position attends to the words about it. The predicted
failure is the swap -- a blue 3 and a red 7, every word present, both bindings
wrong. That is the "red cube and blue sphere" failure, and it is what
`binding.py` counts.

Every training image uses two distinct colours, two distinct digits and two
distinct corners, so a swap is always detectable. Held-out pairs are off here:
compositional generalization was answered in `colored_mnist`, and mixing the two
questions would make neither answerable.
"""

import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from colored_mnist import (
    COLOR_RGB,
    COLORS,
    POSITIONS,
    encode,
    encode_batch,
    offsets,
)
from utils import repo_root

# corners only: the centre box overlaps all four, and two digits sharing pixels
# would make "which object is this" a question about the image rather than the
# caption
CORNERS = POSITIONS[:4]
SEQ_LEN_PAIR = 15  # "a C D in the P1 P2 and a C D in the P1 P2"


def caption_pair(c1: str, d1: str, p1: str, c2: str, d2: str, p2: str) -> str:
    assert p1 in CORNERS and p2 in CORNERS and p1 != p2, (p1, p2)
    assert c1 != c2 and d1 != d2, "a swap has to be detectable"
    return f"a {c1} {d1} in the {p1} and a {c2} {d2} in the {p2}"


def swap_colors(text: str) -> str:
    """The same caption with the two colour words exchanged. Its image should be
    the same two digits in the same two places wearing each other's colour --
    which is exactly what a model that cannot bind will fail to do."""
    w = text.split()
    i, j = 1, w.index("and") + 2
    w[i], w[j] = w[j], w[i]
    return " ".join(w)


def pair_attributes(i: int, seed: int, n: int) -> tuple[int, int, int, int, int]:
    """(partner offset, colour1, corner1, colour2, corner2) for MNIST index i.
    A function of the index alone, as in `colored_mnist`, so the dataset is the
    same set of images every epoch."""
    r = random.Random(seed * 7_000_003 + i)
    c1, c2 = r.sample(range(len(COLORS)), 2)
    p1, p2 = r.sample(range(len(CORNERS)), 2)
    return r.randrange(1, n), c1, p1, c2, p2


class TwoObjectMNIST(Dataset):
    """(image [3,32,32] in [-1,1], tokens [SEQ_LEN_PAIR])."""

    def __init__(
        self,
        root=None,
        train: bool = True,
        size: int = 32,
        digit_px: int = 16,
        seed: int = 0,
    ):
        self.mnist = datasets.MNIST(
            root=root or repo_root() / "data",
            train=train,
            transform=transforms.ToTensor(),
        )
        self.size, self.digit_px = size, digit_px
        self.offsets = offsets(size, digit_px)
        targets = self.mnist.targets.tolist()
        n = len(targets)
        self.index = []
        for i in range(n):
            step, c1, p1, c2, p2 = pair_attributes(i, seed, n)
            j = (i + step) % n
            while targets[j] == targets[i]:  # two digits, never the same digit
                j = (j + 1) % n
            self.index.append((i, j, c1, p1, c2, p2))

    def __len__(self) -> int:
        return len(self.index)

    def _paste(self, canvas: Tensor, mnist_i: int, ci: int, pi: int) -> None:
        d = self.digit_px
        img, _ = self.mnist[mnist_i]
        small = F.interpolate(
            img[None], size=(d, d), mode="bilinear", antialias=True, align_corners=False
        )[0]
        r, c = self.offsets[pi]
        canvas[:, r : r + d, c : c + d] = small * COLOR_RGB[ci][:, None, None]

    def __getitem__(self, k: int) -> tuple[Tensor, Tensor]:
        i, j, c1, p1, c2, p2 = self.index[k]
        canvas = torch.zeros(3, self.size, self.size)
        self._paste(canvas, i, c1, p1)
        self._paste(canvas, j, c2, p2)
        text = caption_pair(
            COLORS[c1],
            str(self.mnist.targets[i].item()),
            CORNERS[p1],
            COLORS[c2],
            str(self.mnist.targets[j].item()),
            CORNERS[p2],
        )
        return canvas * 2 - 1, encode(text, SEQ_LEN_PAIR)


def isolate(x: Tensor, pos: Tensor, size: int = 32, digit_px: int = 16) -> Tensor:
    """[B,3,H,W] and [B] corner indices -> the same canvases with everything
    outside the named box blanked to -1 (the background value).

    This is what lets the single-object judge score a two-object image without
    retraining: a blanked canvas is exactly what it was trained on."""
    assert x.ndim == 4 and pos.shape == (x.shape[0],), (x.shape, pos.shape)
    offs = offsets(size, digit_px)
    out = torch.full_like(x, -1.0)
    for b, p in enumerate(pos.tolist()):
        r, c = offs[p]
        out[b, :, r : r + digit_px, c : c + digit_px] = x[
            b, :, r : r + digit_px, c : c + digit_px
        ]
    return out


def pair_prompt_grid(device=None, per: int = 8) -> tuple[Tensor, int, list[str]]:
    """Four base prompts, each followed by its colour-swapped twin: eight rows,
    `per` samples each. Adjacent rows ask for the same two digits in the same two
    places wearing each other's colour, so binding is legible without measuring
    -- a model that cannot bind draws the two rows the same."""
    base = [
        caption_pair("red", "3", "top left", "blue", "7", "bottom right"),
        caption_pair("green", "1", "top right", "yellow", "8", "bottom left"),
        caption_pair("magenta", "5", "bottom left", "cyan", "2", "top right"),
        caption_pair("blue", "9", "bottom right", "red", "4", "top left"),
    ]
    texts = [t for b in base for t in (b, swap_colors(b)) for _ in range(per)]
    return encode_batch(texts, device, SEQ_LEN_PAIR), per, texts


if __name__ == "__main__":
    from torchvision.utils import save_image

    from colored_mnist import decode, read_color

    torch.manual_seed(0)

    # 1. the caption: fixed length, both objects named, guards on the cases that
    #    would make a swap undetectable
    t = caption_pair("red", "3", "top left", "blue", "7", "bottom right")
    assert t == "a red 3 in the top left and a blue 7 in the bottom right"
    assert len(t.split()) == SEQ_LEN_PAIR
    assert decode(encode(t, SEQ_LEN_PAIR)) == t  # no padding at this length
    for bad in (
        lambda: caption_pair("red", "3", "top left", "red", "7", "bottom right"),
        lambda: caption_pair("red", "3", "top left", "blue", "3", "bottom right"),
        lambda: caption_pair("red", "3", "top left", "blue", "7", "top left"),
        lambda: caption_pair("red", "3", "center", "blue", "7", "bottom right"),
    ):
        try:
            bad()
            raise SystemExit("guard missing")
        except AssertionError:
            pass

    # 2. swap_colors exchanges the two colours and nothing else -- it is the
    #    probe the whole eval rests on, so an off-by-one here would silently
    #    turn the binding test into a comparison of a caption with itself
    sw = swap_colors(t)
    assert sw == "a blue 3 in the top left and a red 7 in the bottom right"
    assert swap_colors(sw) == t  # an involution
    assert sorted(sw.split()) == sorted(t.split()) and sw != t  # same bag, new order

    ds = TwoObjectMNIST()
    x, tok = ds[0]
    print(f"{len(ds)} images  x{tuple(x.shape)}  '{decode(tok)}'")
    assert x.shape == (3, 32, 32) and tok.shape == (SEQ_LEN_PAIR,)
    assert x.min() >= -1 and x.max() <= 1

    # 3. determinism, as in colored_mnist
    for k in (0, 17, 999):
        a, ta = ds[k]
        b, tb = ds[k]
        assert torch.equal(a, b) and torch.equal(ta, tb)

    # 4. THE property that makes the dataset a test: each object is where the
    #    caption says, in the colour the caption gives *it*, not the other one.
    #    A compositing bug that pasted the colours the wrong way round trains
    #    perfectly and inverts the eval's verdict.
    xs, toks = zip(*[ds[k] for k in range(256)])
    xs = torch.stack(xs)
    words = [decode(t).split() for t in toks]
    for slot, (ci, pi) in enumerate(((1, 5), (9, 13))):
        want_c = torch.tensor([COLORS.index(w[ci]) for w in words])
        pos = torch.tensor([CORNERS.index(" ".join(w[pi : pi + 2])) for w in words])
        got = read_color(isolate(xs, pos))
        assert torch.equal(got, want_c), f"object {slot} wears the wrong colour"

    # 5. isolate keeps one box and blanks the rest, exactly
    one = isolate(xs[:4], torch.zeros(4, dtype=torch.long))
    assert (one[:, :, 16:, :] == -1).all() and (one[:, :, :, 16:] == -1).all()
    assert torch.equal(one[:, :, :16, :16], xs[:4, :, :16, :16])
    # and the two objects really are two: isolating each corner recovers both
    lit = [
        int((isolate(xs[:1], torch.tensor([p])) > -0.4).any(1).sum()) for p in range(4)
    ]
    assert sum(v > 0 for v in lit) == 2, lit

    # 6. two distinct digits, two distinct colours, two distinct corners, always
    for w in words:
        assert w[2] != w[10] and w[1] != w[9]
        assert " ".join(w[5:7]) != " ".join(w[13:15])

    y, nrow, texts = pair_prompt_grid()
    assert y.shape == (len(texts), SEQ_LEN_PAIR) and len(texts) == 8 * nrow
    assert texts[nrow] == swap_colors(texts[0])  # row 2 is row 1, colours exchanged
    print(f"grid: {len(texts)} prompts, 8 rows\n  '{texts[0]}'\n  '{texts[nrow]}'")

    out = repo_root() / "scratchpad" / "two_objects.png"
    save_image((xs[:64] + 1) / 2, out, nrow=8)
    print(f"grid -> {out}")
    print("ok")
