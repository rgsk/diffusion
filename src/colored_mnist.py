"""Colored MNIST on a 32x32 canvas, captioned from the attributes that drew it.

Three independent factors per image -- colour, digit, position -- and a caption
that names all three: "a red 3 in the top left". Synthetic, so the captions are
free and cannot be wrong.

The point is what a caption asks of the conditioning path. One label is one
thing, and a single vector added to `temb` says it everywhere at once. A caption
is several facts that have to stay attached to each other, and a sum cannot
record which is which -- the classic "red cube and blue sphere" failure returns a
blue cube with every word present and nothing bound. Cross-attention binds by
letting each position ask its own question, so this dataset is the thing that
makes the difference measurable.

`HELDOUT` is what makes it a test rather than a demo: a handful of colour x digit
pairs never shown. Red 3 is absent; red 5s and green 3s are not. If the model can
draw a red 3 anyway, colour and digit are separate factors it can recombine.
"""

import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from utils import repo_root

COLORS = ("red", "green", "blue", "yellow", "cyan", "magenta")
# corners and edges of the RGB cube: maximally separated, so reading a colour
# back off a generated image is unambiguous
COLOR_RGB = torch.tensor(
    [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0], [0, 1.0, 1.0], [1.0, 0, 1.0]]
)
DIGITS = tuple(str(d) for d in range(10))
POSITIONS = ("top left", "top right", "bottom left", "bottom right", "center")

# index 0 is the null token, the way index `num_classes` is the reserved null row
# for label conditioning: an all-null sequence is the unconditional prompt CFG
# needs, and it must be in the vocabulary before training, not bolted on after.
# <pad> is separate so a short caption is not silently a partly-unconditional one.
VOCAB = (
    "<null>",
    "<pad>",
    "a",
    "in",
    "the",
    *COLORS,
    *DIGITS,
    *sorted({w for p in POSITIONS for w in p.split()}),
)
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
NULL, PAD = 0, 1
SEQ_LEN = 7  # "a <colour> <digit> in the <pos...>", padded

# one pair per colour, six distinct digits, so holding these out starves no
# colour and no digit -- only the combination
HELDOUT = (
    ("red", "3"),
    ("green", "7"),
    ("blue", "1"),
    ("yellow", "5"),
    ("cyan", "9"),
    ("magenta", "0"),
)


def caption(color: str, digit: str, position: str) -> str:
    assert color in COLORS and digit in DIGITS and position in POSITIONS
    return f"a {color} {digit} in the {position}"


def encode(text: str) -> Tensor:
    """Caption -> [SEQ_LEN] token ids, right-padded. Whitespace is the tokenizer:
    the vocabulary is closed, so every word is a known id or a typo worth raising."""
    ids = [WORD2ID[w] for w in text.split()]
    assert len(ids) <= SEQ_LEN, (text, len(ids))
    return torch.tensor(ids + [PAD] * (SEQ_LEN - len(ids)), dtype=torch.long)


def decode(ids: Tensor) -> str:
    return " ".join(VOCAB[i] for i in ids.tolist() if i != PAD)


def null_tokens(n: int, device=None) -> Tensor:
    """The unconditional prompt: no words at all, only the null token."""
    return torch.full((n, SEQ_LEN), NULL, dtype=torch.long, device=device)


def encode_batch(texts: list[str], device=None) -> Tensor:
    return torch.stack([encode(t) for t in texts]).to(device)


def attributes(i: int, seed: int) -> tuple[int, int]:
    """Colour and position for MNIST index i. A function of the index alone, so
    the dataset is the same set of images every epoch and every run -- and so the
    held-out pairs can be filtered up front without loading a single image."""
    r = random.Random(seed * 1_000_003 + i)
    return r.randrange(len(COLORS)), r.randrange(len(POSITIONS))


class ColoredMNIST(Dataset):
    """(image [3,32,32] in [-1,1], tokens [SEQ_LEN]).

    The digit is resized to `digit_px` and dropped at one of five anchors on a
    black canvas. `exclude_heldout=False` keeps every combination -- for a judge,
    which has to recognise a red 3 in order to score one."""

    def __init__(
        self,
        root=None,
        train: bool = True,
        size: int = 32,
        digit_px: int = 16,
        seed: int = 0,
        heldout=HELDOUT,
        exclude_heldout: bool = True,
    ):
        self.mnist = datasets.MNIST(
            root=root or repo_root() / "data",
            train=train,
            transform=transforms.ToTensor(),
        )
        self.size, self.digit_px = size, digit_px
        self.heldout = {tuple(p) for p in heldout}
        assert self.heldout <= {(c, d) for c in COLORS for d in DIGITS}, self.heldout
        self.offsets = offsets(size, digit_px)
        targets = self.mnist.targets.tolist()
        self.index = [  # (mnist index, colour, position); heldout pairs dropped
            (i, c, p)
            for i, (c, p) in ((i, attributes(i, seed)) for i in range(len(targets)))
            if not (exclude_heldout and (COLORS[c], str(targets[i])) in self.heldout)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> tuple[Tensor, Tensor]:
        i, ci, pi = self.index[k]
        img, digit = self.mnist[i]
        d = self.digit_px
        small = F.interpolate(
            img[None], size=(d, d), mode="bilinear", antialias=True, align_corners=False
        )[0]
        canvas = torch.zeros(3, self.size, self.size)
        r, c = self.offsets[pi]
        canvas[:, r : r + d, c : c + d] = small * COLOR_RGB[ci][:, None, None]
        tokens = encode(caption(COLORS[ci], str(digit), POSITIONS[pi]))
        return canvas * 2 - 1, tokens


def offsets(size: int, digit_px: int) -> list[tuple[int, int]]:
    """Top-left corner of the digit box for each entry of POSITIONS."""
    far, mid = size - digit_px, (size - digit_px) // 2
    return [(0, 0), (0, far), (far, 0), (far, far), (mid, mid)]


def anchors(size: int = 32, digit_px: int = 16) -> Tensor:
    """Centre of each position's box, in pixels -- what a centroid is compared to."""
    return torch.tensor(offsets(size, digit_px), dtype=torch.float) + digit_px / 2


def _lit(x: Tensor, thresh: float = 0.3) -> tuple[Tensor, Tensor]:
    """[B,3,H,W] in [-1,1] -> ([0,1] image, mask of pixels the digit actually covers)."""
    assert x.ndim == 4 and x.shape[1] == 3, x.shape
    img = ((x + 1) / 2).clamp(0, 1)
    return img, img.amax(1) > thresh


def read_color(x: Tensor) -> Tensor:
    """[B,3,H,W] -> [B] index into COLORS. Mean RGB over lit pixels, nearest
    direction -- brightness varies with the stroke, hue does not."""
    img, mask = _lit(x)
    n = mask.sum((1, 2)).clamp(min=1).unsqueeze(1)
    rgb = (img * mask.unsqueeze(1)).sum((2, 3)) / n
    rgb = rgb / rgb.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ref = COLOR_RGB / COLOR_RGB.norm(dim=1, keepdim=True)
    return (rgb @ ref.to(rgb).T).argmax(1)


def read_position(x: Tensor) -> Tensor:
    """[B,3,H,W] -> [B] index into POSITIONS, by centroid of the lit pixels."""
    _, mask = _lit(x)
    _, H, W = mask.shape
    m = mask.float()
    n = m.sum((1, 2)).clamp(min=1)
    rows = torch.arange(H, device=x.device, dtype=torch.float)
    colsv = torch.arange(W, device=x.device, dtype=torch.float)
    cy = (m.sum(2) @ rows) / n
    cx = (m.sum(1) @ colsv) / n
    c = torch.stack([cy, cx], 1)
    return torch.cdist(c, anchors(H, H // 2).to(c)).argmin(1)


def prompt_grid(device=None, position: str = "center") -> tuple[Tensor, int, list[str]]:
    """Every colour x every digit at one position: rows are colours, columns are
    digits, and the six held-out cells sit in plain sight. Position is pinned so
    the grid varies on exactly two axes -- `position_grid` is the one that moves
    it."""
    texts = [caption(c, d, position) for c in COLORS for d in DIGITS]
    return encode_batch(texts, device), len(DIGITS), texts


def position_grid(device=None, pairs=HELDOUT) -> tuple[Tensor, int, list[str]]:
    """One row per position, the held-out pairs down the columns. Position is the
    only thing that changes between rows, so a row sitting somewhere other than
    where it was asked to be is visible without measuring anything -- which
    `prompt_grid` alone cannot show, since it pins position to one value."""
    texts = [caption(c, d, p) for p in POSITIONS for c, d in pairs]
    return encode_batch(texts, device), len(pairs), texts


if __name__ == "__main__":
    from torchvision.utils import save_image

    torch.manual_seed(0)

    # 1. the vocabulary is closed and the round trip is exact
    assert VOCAB[NULL] == "<null>" and VOCAB[PAD] == "<pad>"
    assert len(set(VOCAB)) == len(VOCAB), "duplicate word"
    for c in COLORS:
        for d in DIGITS:
            for p in POSITIONS:
                assert decode(encode(caption(c, d, p))) == caption(c, d, p)
    # padding is where variable length shows up: "center" is one word, the
    # corners are two, and both must fit the same fixed tensor
    assert (encode(caption("red", "3", "center"))[-1] == PAD).item()
    assert (encode(caption("red", "3", "top left"))[-1] != PAD).item()
    assert (null_tokens(4) == NULL).all() and null_tokens(4).shape == (4, SEQ_LEN)
    print(f"vocab {len(VOCAB)} words, {SEQ_LEN} tokens/caption: {' '.join(VOCAB)}")

    # 2. an unknown word is a raise, not a silent id -- a typo'd prompt that
    #    quietly encodes to something is the worst possible failure here
    for bad in ("a purple 3 in the top left", "a red three in the center"):
        try:
            encode(bad)
            raise SystemExit("guard missing")
        except KeyError:
            pass

    ds = ColoredMNIST()
    full = ColoredMNIST(exclude_heldout=False)
    x, tok = ds[0]
    print(f"train {len(ds)} of {len(full)}  x{tuple(x.shape)}  '{decode(tok)}'")

    # 3. shapes, range, and determinism: the same index is the same sample, so
    #    epochs repeat and a held-out filter computed from the index is honest
    assert x.shape == (3, 32, 32) and tok.shape == (SEQ_LEN,)
    assert x.min() >= -1 and x.max() <= 1 and x.min() < -0.9
    for k in (0, 17, 999):
        a, ta = ds[k]
        b, tb = ds[k]
        assert torch.equal(a, b) and torch.equal(ta, tb)

    # 4. THE property that makes the caption a label at all: the image agrees
    #    with the words. A colour or offset table off by one trains perfectly and
    #    is only visible here.
    xs, toks = zip(*[ds[k] for k in range(512)])
    xs = torch.stack(xs)
    words = [decode(t).split() for t in toks]
    want_c = torch.tensor([COLORS.index(w[1]) for w in words])
    want_p = torch.tensor([POSITIONS.index(" ".join(w[5:])) for w in words])
    assert torch.equal(read_color(xs), want_c), "caption colour != image colour"
    assert torch.equal(read_position(xs), want_p), "caption position != image position"

    # 5. the held-out pairs are absent from train and present in the judge's copy,
    #    and dropping them starves neither factor
    def pairs(d):
        return [(COLORS[c], str(d.mnist.targets[i].item())) for i, c, _ in d.index]

    seen, seen_full = set(pairs(ds)), set(pairs(full))
    assert not (seen & set(HELDOUT)), sorted(seen & set(HELDOUT))
    assert set(HELDOUT) <= seen_full, "judge split is missing them too"
    assert len(seen) == len(COLORS) * len(DIGITS) - len(HELDOUT) == 54
    for c in COLORS:
        assert sum(p[0] == c for p in seen) == 9  # every colour, minus its one pair
    for d in DIGITS:
        assert sum(p[1] == d for p in seen) >= 5
    print(
        f"{len(seen)} combinations trained on, {len(HELDOUT)} held out: "
        + ", ".join(f"{c} {d}" for c, d in HELDOUT)
    )

    # 6. the loss of data is small enough not to be the explanation for anything
    print(f"held-out pairs cost {1 - len(ds) / len(full):.1%} of the images")
    assert len(ds) / len(full) > 0.85

    # 7. read_color/read_position are the eval's instruments, so they must be
    #    wrong on a wrong image, not just right on a right one
    assert (read_color(xs.roll(1, dims=1)) != want_c).float().mean() > 0.9
    assert (read_position(xs.flip(-1)) != want_p).float().mean() > 0.7

    out = repo_root() / "scratchpad" / "colored_mnist.png"
    save_image((xs[:64] + 1) / 2, out, nrow=8)
    print(f"grid -> {out}")
    print("ok")
