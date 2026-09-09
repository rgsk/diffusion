"""Sample digits from a trained run: pick a label, pick a guidance scale, get a PNG.

Everything else in this project samples inside the training loop. This is the
same machinery with a CLI in front of it.
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from cfg import Guided
from ddim import DDIMSampler
from forward_process import ForwardProcess
from sampler import DDPMSampler
from unet import UNet
from utils import repo_root


def load(run: str, device: str, weights: str = "ema"):
    """Rebuild exactly the net that was trained, from the checkpoint alone."""
    ckpt = repo_root() / "artifacts" / run / "ckpt.pt"
    assert ckpt.exists(), f"no checkpoint at {ckpt}"
    ck = torch.load(ckpt, map_location=device)
    assert weights in ("ema", "net"), weights
    state = ck.get(weights)
    assert state is not None, (
        f"{run} has no '{weights}' weights (trained with --ema-decay 0?)"
    )
    # runs from before --attention have no such key; the weights themselves say
    # so, and reading them can't disagree with what is about to be loaded
    attention = ck.get("attention", any(k.startswith("mid_attn") for k in state))
    # the oldest runs predate conditioning entirely and have no such key either
    C = ck.get("num_classes") or None
    net = UNet(num_classes=C, attention=attention)
    net.load_state_dict(state)
    fp = ForwardProcess()
    fp.load_state_dict(ck["fp"])  # schedule buffers, whichever schedule it was
    return net.to(device).eval(), fp.to(device), C


def main(
    run: str,
    label: int = -1,
    n: int = 16,
    w: float = 3.0,
    steps: int = 50,
    sampler: str = "ddim",
    weights: str = "ema",
    seed: int | None = None,
    out: str = "",
):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, fp, C = load(run, dev, weights)
    smp = DDPMSampler(fp) if sampler == "ddpm" else DDIMSampler(fp, steps=steps)
    smp = smp.to(dev)
    if seed is not None:
        torch.manual_seed(seed)

    if C is None:
        assert label < 0, f"{run} is unconditional -- it has no labels to ask for"
        model, nrow = net, min(n, 8)
    else:
        assert -1 <= label < C, f"label must be 0..{C - 1}, or -1 for all classes"
        if label < 0:
            per = max(1, n // C)  # one class per row, every class
            y = torch.arange(C, device=dev).repeat_interleave(per)
            nrow = per
        else:
            y = torch.full((n,), label, device=dev)
            nrow = min(n, 8)
        model = Guided(net, y, w)

    total = n if C is None or label >= 0 else y.shape[0]
    x = smp.sample(model, (total, 1, 28, 28), dev)
    path = (
        Path(out)
        if out
        else repo_root()
        / "artifacts"
        / run
        / (f"sample_{'all' if label < 0 else label}_w{w:g}.png")
    )
    save_image(x.cpu(), path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"{total} images -> {path}   range [{x.min():.2f}, {x.max():.2f}]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("run", help="folder under artifacts/, e.g. cfg_2026-09-09_18-17-27")
    p.add_argument(
        "--label", type=int, default=-1, help="digit to draw; -1 = all classes"
    )
    p.add_argument(
        "--n", type=int, default=16, help="images (split across classes if -1)"
    )
    p.add_argument(
        "--w", type=float, default=3.0, help="guidance scale; 1 = plain conditional"
    )
    p.add_argument("--steps", type=int, default=50, help="DDIM steps; ignored for ddpm")
    p.add_argument("--sampler", default="ddim", choices=("ddim", "ddpm"))
    p.add_argument(
        "--weights", default="ema", choices=("ema", "net"), help="which copy to sample"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="fix x_T; omit for a fresh draw"
    )
    p.add_argument(
        "--out", default="", help="output path; default is inside the run folder"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
