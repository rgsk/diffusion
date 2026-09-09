"""Train the eps-predictor on MNIST, sample a grid after every epoch."""

import argparse
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from cfg import Guided, drop_labels
from ddim import DDIMSampler
from ema import EMA
from forward_process import ForwardProcess
from loss_by_t import LossByT
from sampler import DDPMSampler
from unet import UNet
from utils import repo_root


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def run_dir(name: str) -> Path:
    """artifacts/scratch is wiped and reused; every other name gets its own
    timestamped folder so runs never clobber each other."""
    base = repo_root() / "artifacts"
    if name == "scratch":
        out = base / "scratch"
        shutil.rmtree(out, ignore_errors=True)  # else a short run keeps stale epochs
    else:
        out = base / f"{name}_{timestamp()}"
    out.mkdir(parents=True)
    return out


def logger(out: Path):
    """print() that also appends to the run's train.log, so console output survives
    the run without shell redirection."""
    path = out / "train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with path.open("a") as f:
            f.write(msg + "\n")

    return log


def build_sampler(fp: ForwardProcess, kind: str, ddim_steps: int):
    """DDIM at 50 steps costs ~1s a grid against DDPM's ~15s; DDPM stays the
    reference, since it is the process the loss is actually derived from."""
    assert kind in ("ddim", "ddpm"), kind
    return DDPMSampler(fp) if kind == "ddpm" else DDIMSampler(fp, steps=ddim_steps)


def cols(cells: list[str], labels: list[str]) -> str:
    """One column per t bucket, header and rows sharing widths so a loss sits
    under the range it belongs to."""
    return " ".join(c.rjust(max(len(lab), 6)) for c, lab in zip(cells, labels))


def main(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 2e-4,
    n_samples: int = 64,
    name: str = "scratch",
    sampler: str = "ddim",
    ddim_steps: int = 50,
    num_classes: int = 10,
    loss_buckets: int = 10,
    ema_decay: float = 0.999,
    schedule: str = "linear",
    seed: int = 0,
    label_dropout: float = 0.1,
    guidance: float = 1.0,
):
    root = repo_root()
    out = run_dir(name)
    log = logger(out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # init, shuffling, and every noise draw -- so two runs differing in one flag
    # differ in that flag alone
    torch.manual_seed(seed)

    ds = datasets.MNIST(
        root=root / "data",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    fp = ForwardProcess(schedule=schedule).to(dev)
    net = UNet(num_classes=num_classes or None).to(dev)
    smp = build_sampler(fp, sampler, ddim_steps).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ema = EMA(net, ema_decay) if ema_decay else None
    labels = LossByT(fp.T, loss_buckets).labels
    log(
        f"out {out}\n"
        f"epochs {epochs}  batch {batch_size}  lr {lr}  n_samples {n_samples}\n"
        f"sampler {sampler}  {getattr(smp, 'steps', fp.T)} steps  "
        f"schedule {schedule}\n"
        f"classes {num_classes or 'unconditional'}  "
        f"ema {ema_decay or 'off'}  "
        f"label dropout {label_dropout}  guidance {guidance}\n"
        f"seed {seed}\n"
        f"device {dev}  params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M  "
        f"{len(loader)} steps/epoch\n"
        f"loss by t:  {cols(labels, labels)}"
    )

    if num_classes:
        per_class = max(1, n_samples // num_classes)
        y_grid = torch.arange(num_classes, device=dev).repeat_interleave(per_class)
        # save_image's nrow is images *per* row, i.e. a column count. Handing it
        # per_class against a label-major y_grid puts one class on each row.
        nrow = per_class
    else:
        y_grid, nrow = None, 8

    for epoch in range(1, epochs + 1):
        net.train()
        start = time.time()
        by_t = LossByT(fp.T, loss_buckets, dev)
        for x0, y in loader:
            x0 = x0.to(dev)
            # dropped per sample and redrawn every step, so one net learns both
            # eps(x,t,y) and eps(x,t,null) from the same images
            y = (
                drop_labels(y.to(dev), label_dropout, num_classes)
                if num_classes
                else None
            )
            t = fp.sample_t(x0.shape[0], dev)
            noise = torch.randn_like(x0)
            # per image, then mean: same gradient as mse_loss, but the split by t
            # survives instead of being pooled away
            eps = net(fp.q_sample(x0, t, noise), t, y)
            se = (eps - noise).pow(2).flatten(1).mean(1)
            loss = se.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            by_t.update(t, se)
            if ema:
                ema.update(net)
        train_s = time.time() - start

        net.eval()
        start = time.time()
        n = n_samples if y_grid is None else y_grid.shape[0]
        # the grid is what the run is judged on, so draw it from the weights that
        # would actually ship -- the averaged ones
        with ema.as_weights(net) if ema else nullcontext():
            model = net if y_grid is None else Guided(net, y_grid, guidance)
            x = smp.sample(model, (n, 1, 28, 28), dev)
        save_image(
            x.cpu(),
            out / f"samples_epoch{epoch:02d}.png",
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
        )
        log(
            f"epoch {epoch}  loss {by_t.pooled():.4f}  train {train_s:.0f}s  "
            f"sample {time.time() - start:.0f}s  range [{x.min():.2f}, {x.max():.2f}]\n"
            + "  by t:     "
            + cols([f"{m:.4f}" for m in by_t.means()], labels)
        )
        # num_classes rides along; without it the net can't be rebuilt to load this
        torch.save(
            {
                "net": net.state_dict(),
                "ema": ema.shadow if ema else None,  # None so a loader can tell
                "fp": fp.state_dict(),  # the schedule rides along as buffers
                "num_classes": num_classes,
            },
            out / "ckpt.pt",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=5, help="training epochs")
    p.add_argument("--batch-size", type=int, default=128, help="images per step")
    p.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    p.add_argument("--n-samples", type=int, default=64, help="grid size per epoch")
    p.add_argument(
        "--name",
        default="scratch",
        help="run label; 'scratch' reuses artifacts/scratch, anything else gets "
        "its own timestamped folder",
    )
    p.add_argument(
        "--sampler",
        default="ddim",
        choices=("ddim", "ddpm"),
        help="sampler for the per-epoch grid",
    )
    p.add_argument("--ddim-steps", type=int, default=50, help="ignored for ddpm")
    p.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="condition on the MNIST label; 0 trains unconditionally",
    )
    p.add_argument(
        "--loss-buckets",
        type=int,
        default=10,
        help="t buckets the epoch loss is reported in; 1 pools as before",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="weight EMA for the sampled grid and the 'ema' checkpoint key; "
        "0 disables and samples from the trained weights",
    )
    p.add_argument(
        "--schedule",
        default="cosine",
        choices=("linear", "cosine"),
        help="beta schedule for the forward process",
    )
    p.add_argument("--seed", type=int, default=0, help="init, shuffling, and noise")
    p.add_argument(
        "--label-dropout",
        type=float,
        default=0.1,
        help="probability of training a step on the null label; 0 disables CFG",
    )
    p.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="guidance scale w for the epoch grid; 1 is the plain conditional net",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
