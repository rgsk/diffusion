"""Sample from a trained run: pick a label or write a prompt, get a PNG.

    python sample.py <run> --label 3 --n 16 --w 3
    python sample.py <run> --prompt "a red 3 in the top left" --n 16

Everything else in this project samples inside the training loop. This is the
same machinery with a CLI in front of it.
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from cfg import Guided
from colored_mnist import SEQ_LEN, encode, prompt_grid
from ddim import DDIMSampler
from forward_process import ForwardProcess
from sampler import DDPMSampler
from two_objects import SEQ_LEN_PAIR, pair_prompt_grid
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
    V = ck.get("vocab_size") or None
    net = UNet(
        in_ch=ck.get("in_ch", 1),
        num_classes=C,
        attention=attention,
        vocab_size=V,
        pooled=ck.get("pooled", False),
        text_layers=ck.get("text_layers", 0),
        coords=ck.get("coords", False),  # adds no weights, so only the key says
    )
    net.load_state_dict(state)
    fp = ForwardProcess()
    fp.load_state_dict(ck["fp"])  # schedule buffers, whichever schedule it was
    shape = (ck.get("in_ch", 1), ck.get("image_size", 28), ck.get("image_size", 28))
    # which dataset wrote this decides how long a caption is; a pair run's
    # prompt names two objects and does not fit the single-object length
    return net.to(device).eval(), fp.to(device), shape, ck.get("dataset", "mnist")


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)


def main(
    run: str,
    label: int = -1,
    prompt: str = "",
    n: int = 16,
    w: float = 3.0,
    steps: int = 50,
    sampler: str = "ddim",
    weights: str = "ema",
    seed: int | None = None,
    out: str = "",
):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, fp, shape, dataset = load(run, dev, weights)
    C, V = net.num_classes, net.vocab_size
    smp = DDPMSampler(fp) if sampler == "ddpm" else DDIMSampler(fp, steps=steps)
    smp = smp.to(dev)
    if seed is not None:
        torch.manual_seed(seed)

    if V is not None:
        assert label < 0, f"{run} is captioned -- ask it with --prompt, not --label"
        seq_len = SEQ_LEN_PAIR if dataset == "pair" else SEQ_LEN
        if prompt:
            # encode() raises on an unknown word rather than encoding a shrug
            y = encode(prompt, seq_len).to(dev).expand(n, seq_len)
            nrow, tag = min(n, 8), slug(prompt)
        elif dataset == "pair":
            y, nrow, _ = pair_prompt_grid(dev)  # each pair beside its colour swap
            tag = "grid"
        else:
            y, nrow, _ = prompt_grid(dev)  # every colour x every digit
            tag = "grid"
        model = Guided(net, y, w)
    elif C is None:
        assert label < 0 and not prompt, f"{run} is unconditional -- nothing to ask"
        model, y, nrow, tag = net, torch.empty(n), min(n, 8), "uncond"
    else:
        assert not prompt, f"{run} is class-conditioned -- use --label, not --prompt"
        assert -1 <= label < C, f"label must be 0..{C - 1}, or -1 for all classes"
        if label < 0:
            per = max(1, n // C)  # one class per row, every class
            y = torch.arange(C, device=dev).repeat_interleave(per)
            nrow, tag = per, "all"
        else:
            y = torch.full((n,), label, device=dev)
            nrow, tag = min(n, 8), str(label)
        model = Guided(net, y, w)

    total = y.shape[0]
    x = smp.sample(model, (total, *shape), dev)
    path = (
        Path(out)
        if out
        else repo_root() / "artifacts" / run / f"sample_{tag}_w{w:g}.png"
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
        "--prompt",
        default="",
        help='caption for a captioned run, e.g. "a red 3 in the top left"; '
        "omit for the full colour x digit grid",
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
