"""2D sinusoidal coordinates: the "where am I" a convolution cannot supply.

A conv is translation-equivariant by construction -- the same filter runs at
every location -- so a feature vector in the top-left is built the same way as
one in the bottom-right and carries no record of which it is. Attention is
worse: it treats its inputs as a set (`attention.py`, test 3). Neither knows
where anything is, and only zero-padding leaks a hint at the border.

That is fine until two objects must be told apart by position. Cross-attention
binds by matching a query against the words that concern it, so a query in the
top-left has to be able to say *I am the top-left one* before it can select the
clause that names it. `README.md` has three runs where it cannot, all assigning
colours at chance.

Fixed, not learned, on purpose: the last positional signal here was a learned
`token_pos` that trained to 2% of the embedding norm and never became usable,
because the loss can be driven down on word identity alone. A closed form cannot
fail to train, and it adds no parameters -- so a run with coordinates and its
control have identical weights, and the comparison is the coordinates alone.

Positions are normalized to a canonical `extent`-wide canvas rather than being
raw indices, so the same place gets the same code at 28x28, 14x14 and 7x7. A
U-Net that changed its mind about where the top-left was on every downsample
would be worse than having no coordinates at all.
"""

from functools import cache

import torch
from torch import Tensor

from timestep_embedding import timestep_embedding


@cache
def coord_embedding(
    height: int,
    width: int,
    dim: int,
    extent: float = 32.0,
    device: torch.device | None = None,
) -> Tensor:
    """-> [1, dim, height, width]. First half of the channels encodes the row,
    second half the column, each by `timestep_embedding` of a coordinate in
    units of an `extent`-pixel canvas.

    Cached: the result is a function of its arguments alone. Callers add it,
    and must not write into it."""
    assert dim % 4 == 0, "dim splits in two axes, each of which splits sin/cos"
    half = dim // 2
    # pixel *centers*, so a cell at 7x7 lands at the middle of the 4x4 block of
    # 28x28 pixels it stands for, rather than at that block's corner
    grid = [
        (torch.arange(n, device=device) + 0.5) / n * extent for n in (height, width)
    ]
    # max_period 2*extent: the slowest frequency is half a cycle across the
    # canvas, so no channel is wasted being flat over the whole image
    ey, ex = (timestep_embedding(g, half, max_period=2 * extent) for g in grid)
    emb = torch.cat(
        [
            ey[:, None, :].expand(height, width, half),  # varies down rows
            ex[None, :, :].expand(height, width, half),  # varies across columns
        ],
        dim=-1,
    )
    return emb.permute(2, 0, 1)[None].contiguous()


if __name__ == "__main__":
    D = 64
    emb = coord_embedding(28, 28, D)

    # 1. shape, range, and the guard
    assert emb.shape == (1, D, 28, 28)
    assert emb.abs().max() <= 1.0
    assert coord_embedding(7, 14, D).shape == (1, D, 7, 14)  # not just squares
    try:
        coord_embedding(8, 8, 6)
        raise SystemExit("guard missing")
    except AssertionError:
        pass

    # 2. the axes are separate and not swapped. The first half is a function of
    #    the row only, the second half of the column only -- a broadcast bug
    #    here gives an embedding that still looks fine by every other check
    y_half, x_half = emb[0, : D // 2], emb[0, D // 2 :]
    assert (y_half - y_half[:, :, :1]).abs().max() == 0.0  # constant across a row
    assert (x_half - x_half[:, :1, :]).abs().max() == 0.0  # constant down a column

    # 3. THE property: every position gets its own code. This is the whole
    #    reason the file exists -- a conv feature map has nothing like it.
    flat = emb.reshape(D, -1).T  # [H*W, D]
    dists = torch.cdist(flat[None], flat[None])[0]
    dists += torch.eye(dists.shape[0]) * 1e9  # ignore the zero diagonal
    print(f"closest two positions: {dists.min():.4f} apart (of {D**0.5:.2f} max)")
    assert dists.min() > 0.1

    # 4. neighbours are nearer than distant positions, so the code is a smooth
    #    map of the image and not an arbitrary lookup table
    d = dists.reshape(28, 28, 28, 28)
    assert d[0, 0, 0, 1] < d[0, 0, 0, 14] < d[0, 0, 0, 27]
    assert d[0, 0, 1, 0] < d[0, 0, 14, 0] < d[0, 0, 27, 0]

    # 5. THE resolution property: the same place is the same code down the
    #    U-Net. For every cell of the 7x7 bottleneck, the nearest 28x28 codes
    #    must be the ones inside the 4x4 block that cell stands for. A net that
    #    changed its mind about where the top-left was on each downsample would
    #    be worse than one with no coordinates at all.
    big = coord_embedding(28, 28, D).reshape(D, -1).T
    worst = 0.0
    for i in range(7):
        for j in range(7):
            cell = coord_embedding(7, 7, D)[0, :, i, j]
            near = (big - cell).norm(dim=1).argmin().item()
            assert (near // 28) // 4 == i and (near % 28) // 4 == j, (i, j, near)
            worst = max(worst, (big - cell).norm(dim=1).min().item())
    print(f"7x7 cells land in their own 28x28 block; worst gap {worst:.4f}")

    # 6. it is cached by argument, so the U-Net rebuilding it per block per step
    #    costs one dict lookup
    assert coord_embedding(28, 28, D) is emb

    print("ok")
