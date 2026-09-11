"""eps against flow matching, on the same net, the same data and the same budget.

The two objectives cannot be compared by their training losses -- eps-MSE and
v-MSE score different targets, the same trap `README.md` records for linear
against cosine -- so everything here is measured on generated pixels instead.

Two slices through the same grid, both at a fixed x_T shared by every cell, so
the start point is never a variable:

  - **step budget**: quality against the number of model calls. Flow's straight
    path is the whole argument for the rewrite, and a straight path should be
    cheaper to follow with big steps. `flow.py` shows exactly that against a
    Gaussian oracle; this asks whether it survives a real net.
  - **guidance**: the w sweep the README already has for eps, run again for flow.
    CFG is arithmetic on the net's output and knows nothing about the objective,
    so the prediction is that nothing changes. Worth checking rather than
    assuming, since w>1 extrapolates and the two outputs are different objects.

Obedience is scored by a small CNN trained on MNIST -- a ruler, not a subject,
and its accuracy on real test digits is printed as the error bar on every number
below. Diversity is across-sample pixel std within each class, as in `cfg.ipynb`.

Accuracy alone cannot see a collapsed sampler: a class prototype drawn sixteen
times scores 1.000. So every cell also carries a last column, the correlation
between the mean of the samples drawn for a class and the dataset's own average
image of that class. A working generator's `per` samples average to something
near that image but not exactly -- `per` is small -- while a sampler answering
`E[x_0 | y]` hits it dead on, because every one of its draws already *is* that
image. Real images run through the same statistic are printed as the control,
and a run reading meaningfully above it has collapsed.
"""

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from cfg import Guided
from sample import build_sampler, load
from utils import repo_root


class Judge(nn.Module):
    """Reads the digit off a 28x28 grayscale image. Same shape of model as
    `compositional.Judge`, one channel and one size smaller."""

    def __init__(self, ch: int = 32):
        super().__init__()

        def block(i, o, stride):
            return [nn.Conv2d(i, o, 3, stride, 1), nn.BatchNorm2d(o), nn.ReLU()]

        self.body = nn.Sequential(
            *block(1, ch, 1),  # 28
            *block(ch, ch, 2),  # 14
            *block(ch, 2 * ch, 2),  # 7
            *block(2 * ch, 4 * ch, 2),  # 4
            nn.Flatten(),
            nn.Linear(4 * ch * 16, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def mnist(train: bool):
    """[-1, 1], matching what the diffusion runs were trained on -- a judge fed
    [0, 1] would read every generated sample as the wrong thing."""
    return datasets.MNIST(
        root=repo_root() / "data",
        train=train,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )


def train_judge(dev: str, epochs: int = 3, batch: int = 128, force: bool = False):
    """Cached in artifacts/: it depends on the dataset, not on any run."""
    path = repo_root() / "artifacts" / "mnist_judge.pt"
    judge = Judge().to(dev)
    if path.exists() and not force:
        try:
            judge.load_state_dict(torch.load(path, map_location=dev))
            return judge.eval()
        except RuntimeError as e:  # cache predates an architecture change
            print(f"  cached judge does not fit ({e.__class__.__name__}); retraining")
    loader = DataLoader(mnist(True), batch_size=batch, shuffle=True, num_workers=4)
    opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for ep in range(epochs):
        judge.train()
        for x, d in loader:
            loss = F.cross_entropy(judge(x.to(dev)), d.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  judge epoch {ep + 1}  loss {loss.item():.4f}")
    torch.save(judge.state_dict(), path)
    return judge.eval()


@torch.no_grad()
def real_baseline(means: Tensor, per: int, seed: int = 0) -> tuple[float, float]:
    """(diversity, correlation with the class mean) for `per` *real* images per
    class, run through the same statistics as every generated cell. Whatever
    this reads is what a perfect sampler would read, not 0.273 and not 1.000."""
    ds = mnist(False)
    imgs = (ds.data.float() / 255.0 - 0.5) / 0.5
    g = torch.Generator().manual_seed(seed)
    cls = torch.stack(
        [
            imgs[ds.targets == c][
                torch.randperm(int((ds.targets == c).sum()), generator=g)[:per]
            ]
            for c in range(10)
        ]
    )[:, :, None]  # [10, per, 1, 28, 28]
    got = cls.mean(dim=1).flatten()
    corr = torch.corrcoef(torch.stack([got, means.flatten()]))[0, 1].item()
    return cls.std(dim=1).mean().item(), corr


@torch.no_grad()
def judge_accuracy(judge: Judge, dev: str, batch: int = 512) -> float:
    """On the real MNIST test split. Whatever this falls short of 1.0 is
    measurement error in every score below, not model error."""
    hits = tot = 0
    for x, d in DataLoader(mnist(False), batch_size=batch, num_workers=4):
        hits += int((judge(x.to(dev)).argmax(1).cpu() == d).sum())
        tot += len(d)
    return hits / tot


@torch.no_grad()
def run_chain(smp, model, x: Tensor) -> Tensor:
    """sample() draws its own x_T; this drives the chain from one handed in, so
    every cell of the sweep starts from the same noise. DDIM and the flow
    integrator share the (model, x, i) step interface, so this does not branch."""
    for i in range(smp.steps):
        x = smp.step(model, x, i)
    return x


def class_means() -> Tensor:
    """[10, 1, 28, 28], the average training image of each digit, in [-1, 1]. The
    thing a squared-error loss converges to when it cannot tell two answers
    apart, so it is what a collapsed sampler is compared against."""
    ds = mnist(True)
    imgs = (ds.data.float() / 255.0 - 0.5) / 0.5
    return torch.stack([imgs[ds.targets == c].mean(0) for c in range(10)])[:, None]


@torch.no_grad()
def score(
    judge: Judge, x: Tensor, y: Tensor, per: int, means: Tensor
) -> tuple[float, float, float, float]:
    """(label accuracy, diversity, pixels pinned at +-1, correlation with the
    class mean image).

    Diversity is the across-sample pixel std within one class, meaned over
    pixels and classes. `README.md` records that it stops meaning variety past
    w~8, where speckle reads as variance -- so it is reported beside a grid, not
    alone. The last column is the collapse detector: a sampler that has fallen
    back on E[x_0 | y] correlates ~1.000 with these means, and scores a perfect
    accuracy while doing it."""
    pred = judge(x).argmax(1)
    acc = (pred == y).float().mean().item()
    cls = x.reshape(-1, per, *x.shape[1:])  # [classes, per, 1, H, W]
    div = cls.std(dim=1).mean().item()
    pinned = (x.abs() >= 0.999).float().mean().item()
    got = cls.mean(dim=1).cpu().flatten()
    mean_corr = torch.corrcoef(torch.stack([got, means.flatten()]))[0, 1].item()
    return acc, div, pinned, mean_corr


def main(
    runs: list[str],
    steps: tuple[int, ...] = (2, 5, 10, 20, 50, 100),
    ws: tuple[float, ...] = (1.0, 3.0),
    sweep_ws: tuple[float, ...] = (0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0),
    sweep_steps: int = 50,
    per: int = 16,
    weights: str = "ema",
    seed: int = 0,
    retrain_judge: bool = False,
):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    judge = train_judge(dev, force=retrain_judge)
    means = class_means()
    acc = judge_accuracy(judge, dev)
    print(f"judge on real MNIST test digits: {acc:.4f}")
    assert acc > 0.97, f"judge too weak to measure anything: {acc}"
    r_div, r_corr = real_baseline(means, per)
    r_pin = (((mnist(False).data.float() / 255.0 - 0.5) / 0.5).abs() >= 0.999).float()
    print(
        f"real images, {per} per class, same statistics: diversity {r_div:.3f}  "
        f"pinned {r_pin.mean():.3f}  corr with class mean {r_corr:.3f}"
    )

    loaded = {}
    for run in runs:
        net, proc, shape, _, objective = load(run, dev, weights)
        assert net.num_classes == 10, f"{run} is not a 10-class MNIST run"
        assert shape == (1, 28, 28), shape
        loaded[run] = (net, proc, objective)
        print(f"{run}: objective {objective}")

    # one class per row, and one x_T reused by every cell below -- across step
    # counts, across w, and across both runs. The objective is then the only
    # thing that differs between two numbers in the same column.
    y = torch.arange(10, device=dev).repeat_interleave(per)
    torch.manual_seed(seed)
    x_T = torch.randn(y.shape[0], 1, 28, 28, device=dev)
    out = repo_root() / "artifacts"

    for w in ws:
        print(f"\n=== step budget, w={w} ===")
        head = f"{'steps':>6}" + "".join(f"{r.split('_20')[0]:>29}" for r in runs)
        print(head + "\n" + "-" * len(head))
        for n in steps:
            cells = []
            for run in runs:
                net, proc, objective = loaded[run]
                smp = build_sampler(proc, objective, steps=n).to(dev)
                x = run_chain(smp, Guided(net, y, w), x_T)
                a, d, p, m = score(judge, x, y, per, means)
                cells.append(f"{a:6.3f} {d:6.3f} {p:6.3f} {m:6.3f}")
                if n in (2, 10, 50):
                    save_image(
                        x.cpu(),
                        out / f"{run}/steps{n:03d}_w{w:g}.png",
                        nrow=per,
                        normalize=True,
                        value_range=(-1, 1),
                    )
            print(f"{n:6d}" + "".join(f"{c:>29}" for c in cells))
        print("          (accuracy, diversity, pinned, corr with class mean)")

    print(f"\n=== guidance, {sweep_steps} steps ===")
    head = f"{'w':>6}" + "".join(f"{r.split('_20')[0]:>29}" for r in runs)
    print(head + "\n" + "-" * len(head))
    for w in sweep_ws:
        cells = []
        for run in runs:
            net, proc, objective = loaded[run]
            smp = build_sampler(proc, objective, steps=sweep_steps).to(dev)
            x = run_chain(smp, Guided(net, y, w), x_T)
            a, d, p, m = score(judge, x, y, per, means)
            cells.append(f"{a:6.3f} {d:6.3f} {p:6.3f} {m:6.3f}")
        print(f"{w:6.1f}" + "".join(f"{c:>29}" for c in cells))
    print("          (accuracy, diversity, pinned, corr with class mean)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", help="folders under artifacts/, eps first")
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=(2, 5, 10, 20, 50, 100),
        help="model-call budgets to sweep",
    )
    p.add_argument(
        "--ws", type=float, nargs="+", default=(1.0, 3.0), help="w per step sweep"
    )
    p.add_argument(
        "--sweep-ws",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0),
        help="guidance scales for the w sweep",
    )
    p.add_argument("--sweep-steps", type=int, default=50, help="budget for the w sweep")
    p.add_argument("--per", type=int, default=16, help="samples per class")
    p.add_argument("--weights", default="ema", choices=("ema", "net"))
    p.add_argument("--seed", type=int, default=0, help="fixes the shared x_T")
    p.add_argument("--retrain-judge", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
