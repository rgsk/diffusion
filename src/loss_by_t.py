"""eps-MSE split by t, accumulated from the training draws themselves.

The pooled epoch loss is dominated by low t (`results.md`), so a gain anywhere
else is invisible in it. Bucketing costs nothing extra: the training step
already draws a t and a per-sample error, this only keeps a running sum per
bucket instead of throwing the split away.
"""

from itertools import pairwise

import torch
from torch import Tensor


class LossByT:
    """Running per-bucket mean of a per-sample loss, keyed by t.

    Not an nn.Module: nothing here is learned or checkpointed, and it is reset
    every epoch."""

    def __init__(self, T: int, n_buckets: int = 10, device=None):
        assert 0 < n_buckets <= T, (n_buckets, T)
        self.T, self.n = T, n_buckets
        self.sums = torch.zeros(n_buckets, device=device)
        self.counts = torch.zeros(n_buckets, device=device)

    def update(self, t: Tensor, per_sample: Tensor) -> None:
        """t and per_sample are both [B], one entry per image in the batch."""
        assert t.shape == per_sample.shape, (t.shape, per_sample.shape)
        # t*n//T, not t//(T//n): the latter overflows into an n+1'th bucket when
        # n does not divide T.
        b = t * self.n // self.T
        self.sums.index_add_(0, b, per_sample.detach().float())
        self.counts.index_add_(0, b, torch.ones_like(self.sums[b]))

    def means(self) -> list[float]:
        """Empty buckets read 0.0 rather than nan -- a bucket with no draws has
        nothing to report, and one nan would poison a log line."""
        return (self.sums / self.counts.clamp(min=1)).tolist()

    def pooled(self) -> float:
        """The scalar the epoch line prints. Count-weighted, so it equals the
        mean over every sample seen regardless of how t landed."""
        return (self.sums.sum() / self.counts.sum().clamp(min=1)).item()

    @property
    def edges(self) -> list[tuple[int, int]]:
        """[lo, hi) per bucket. lo_j = ceil(j*T/n) inverts t*n//T exactly."""
        bound = [-((-j * self.T) // self.n) for j in range(self.n + 1)]
        return list(pairwise(bound))

    @property
    def labels(self) -> list[str]:
        return [f"{lo}-{hi - 1}" for lo, hi in self.edges]


if __name__ == "__main__":
    torch.manual_seed(0)
    T = 1000

    # 1. the labels are not decoration: every t lands in the bucket its own
    #    label claims, for divisible and indivisible n alike
    for n in (10, 7, 3, 1, T):
        acc = LossByT(T, n)
        t = torch.arange(T)
        b = t * n // T
        assert b.min() == 0 and b.max() == n - 1, n
        for j, (lo, hi) in enumerate(acc.edges):
            assert lo < hi, (n, j)
            assert torch.equal(t[b == j], torch.arange(lo, hi)), (n, j)
        assert acc.edges[0][0] == 0 and acc.edges[-1][1] == T, n

    # 2. the means are per-bucket means, checked against a groupby of the same
    #    draws. Two unequal batches, so a bucket spans calls.
    acc = LossByT(T, 10)
    ts, ls = [], []
    for batch in (128, 37):
        t = torch.randint(0, T, (batch,))
        loss = t.float() / T + torch.rand(batch)  # varies with t, as the real one does
        acc.update(t, loss)
        ts.append(t)
        ls.append(loss)
    t, loss = torch.cat(ts), torch.cat(ls)
    for j, m in enumerate(acc.means()):
        sel = loss[t * 10 // T == j]
        assert abs(m - sel.mean().item()) < 1e-5, j

    # 3. THE invariant: pooled is the mean over samples, not over buckets. The
    #    buckets hold unequal counts, so the two differ -- averaging the printed
    #    row would not reproduce the printed scalar.
    assert abs(acc.pooled() - loss.mean().item()) < 1e-5
    row = sum(acc.means()) / 10
    print(f"pooled {acc.pooled():.4f}  mean of buckets {row:.4f}")
    assert abs(row - acc.pooled()) > 1e-4

    # 4. nothing seen yet: zeros, no division by zero
    empty = LossByT(T, 10)
    assert empty.pooled() == 0.0 and empty.means() == [0.0] * 10

    # 5. shape guard -- [B,1] against [B] would broadcast into a silent B×B sum
    try:
        LossByT(T, 10).update(torch.zeros(4, dtype=torch.long), torch.zeros(4, 1))
        raise SystemExit("guard missing")
    except AssertionError:
        pass

    print(" ".join(LossByT(T, 10).labels))
    print("ok")
