"""Train the eps-predictor on MNIST, sample a grid after every epoch."""

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from forward_process import ForwardProcess
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


def main(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 2e-4,
    n_samples: int = 64,
    name: str = "scratch",
):
    root = repo_root()
    out = run_dir(name)
    log = logger(out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

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

    fp = ForwardProcess().to(dev)
    net = UNet().to(dev)
    sampler = DDPMSampler(fp).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    log(
        f"out {out}\n"
        f"epochs {epochs}  batch {batch_size}  lr {lr}  n_samples {n_samples}\n"
        f"device {dev}  params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M  "
        f"{len(loader)} steps/epoch"
    )

    for epoch in range(1, epochs + 1):
        net.train()
        start, total = time.time(), 0.0
        for x0, _ in loader:
            x0 = x0.to(dev)
            t = fp.sample_t(x0.shape[0], dev)
            noise = torch.randn_like(x0)
            loss = F.mse_loss(net(fp.q_sample(x0, t, noise), t), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        train_s = time.time() - start

        net.eval()
        start = time.time()
        x = sampler.sample(net, (n_samples, 1, 28, 28), dev)
        save_image(
            x.cpu(),
            out / f"samples_epoch{epoch:02d}.png",
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )
        log(
            f"epoch {epoch}  loss {total / len(loader):.4f}  train {train_s:.0f}s  "
            f"sample {time.time() - start:.0f}s  range [{x.min():.2f}, {x.max():.2f}]"
        )
        torch.save({"net": net.state_dict(), "fp": fp.state_dict()}, out / "ckpt.pt")


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
    return p.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
