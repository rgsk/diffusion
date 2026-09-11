"""Train the denoiser, sample a grid after every epoch.

`--objective eps` predicts the noise on a variance-preserving beta schedule
(DDPM/DDIM); `--objective flow` predicts the velocity along a straight line from
noise to data (rectified flow, `flow.py`). The loop below is the same either way
-- it asks the process for a t, an x_t and a target, and never learns which one
it got -- so the two are comparable at a fixed --seed.

`--dataset mnist` is 28x28 grayscale conditioned on a class label; `--dataset
colored` is 32x32 RGB conditioned on a caption, and its grid is every colour x
every digit, so the held-out combinations are visible in the picture.
"""

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
from colored_mnist import VOCAB, ColoredMNIST, position_grid, prompt_grid
from ema import EMA
from flow import FlowPath
from forward_process import ForwardProcess
from loss_by_t import LossByT
from negatives import per_sample_mse, swap_hinge
from sample import build_sampler
from two_objects import TwoObjectMNIST, pair_prompt_grid, swap_color_tokens
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
    attention: bool = False,
    dataset: str = "mnist",
    pooled_context: bool = False,
    text_layers: int = 0,
    coords: bool = False,
    swap_weight: float = 0.0,
    swap_margin: float = 0.1,
    objective: str = "eps",
):
    assert dataset in ("mnist", "colored", "pair"), dataset
    assert objective in ("eps", "flow"), objective
    # an ODE has no variance to pick, so there is nothing for --sampler to choose
    assert objective == "eps" or sampler == "ddim", "--sampler is an eps choice"
    # the negative is a colour swap, which only names two objects on the pair set
    assert not swap_weight or dataset == "pair", "--swap-weight needs --dataset pair"
    root = repo_root()
    out = run_dir(name)
    log = logger(out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # init, shuffling, and every noise draw -- so two runs differing in one flag
    # differ in that flag alone
    torch.manual_seed(seed)

    if dataset in ("colored", "pair"):
        ds = (
            ColoredMNIST(root=root / "data", train=True, seed=seed)
            if dataset == "colored"
            else TwoObjectMNIST(root=root / "data", train=True, seed=seed)
        )
        # the caption replaces the label rather than joining it: the whole point
        # is that everything conditional goes through cross-attention
        in_ch, size, num_classes, vocab_size = 3, ds.size, 0, len(VOCAB)
    else:
        ds = datasets.MNIST(
            root=root / "data",
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        in_ch, size, vocab_size = 1, 28, None
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    # the only two lines that know which objective this is. Everything below --
    # the net, EMA, CFG, the hinge, the grids, the checkpoint -- reads it off
    # `proc` through sample_t/q_sample/target and never asks again.
    proc = (
        FlowPath() if objective == "flow" else ForwardProcess(schedule=schedule)
    ).to(dev)
    net = UNet(
        in_ch=in_ch,
        num_classes=num_classes or None,
        attention=attention,
        vocab_size=vocab_size,
        pooled=pooled_context,
        text_layers=text_layers,
        coords=coords,
    ).to(dev)
    smp = build_sampler(proc, objective, sampler, ddim_steps).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ema = EMA(net, ema_decay) if ema_decay else None
    labels = LossByT(proc.T, loss_buckets).labels
    log(
        f"out {out}\n"
        f"dataset {dataset}  {len(ds)} images  {in_ch}x{size}x{size}\n"
        f"epochs {epochs}  batch {batch_size}  lr {lr}  n_samples {n_samples}\n"
        f"objective {objective}  "
        f"sampler {'flow-euler' if objective == 'flow' else sampler}  "
        f"{getattr(smp, 'steps', proc.T)} steps  "
        f"schedule {'-' if objective == 'flow' else schedule}\n"
        f"classes {num_classes or 'unconditional'}  "
        f"vocab {vocab_size or '-'}"
        f"{' (pooled into temb)' if vocab_size and pooled_context else ''}"
        f"{f' text-encoder x{text_layers}' if vocab_size and text_layers else ''}"
        f"{' coords' if vocab_size and coords else ''}"
        f"{f' swap-negatives w{swap_weight} m{swap_margin}' if swap_weight else ''}  "
        f"ema {ema_decay or 'off'}  "
        f"label dropout {label_dropout}  guidance {guidance}  "
        f"attention {attention}\n"
        f"seed {seed}\n"
        f"device {dev}  params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M  "
        f"{len(loader)} steps/epoch\n"
        f"loss by t:  {cols(labels, labels)}"
    )

    # (filename tag, tokens or labels, images per row); the first is the one the
    # log reports a range for
    if dataset == "pair":
        # rows come in pairs: a prompt, then the same prompt with its two colour
        # words exchanged. A model that binds draws them differently; a model
        # that pools draws them the same, and the grid says so without measuring.
        yg, ng, prompts = pair_prompt_grid(dev)
        grids = [("samples", yg, ng)]
        log(f"grid: {len(prompts)} prompts\n  '{prompts[0]}'\n  '{prompts[ng]}'")
    elif vocab_size:
        # every colour x every digit at one position: the six held-out cells are
        # in the picture, so the compositionality question is asked every epoch.
        # positions_ sweeps the axis prompt_grid pins, so a broken position
        # never sits unnoticed behind a grid that cannot show it.
        yg, ng, prompts = prompt_grid(dev)
        yp, np_, pprompts = position_grid(dev)
        grids = [("samples", yg, ng), ("positions", yp, np_)]
        log(
            f"grid: {len(prompts)} prompts, '{prompts[0]}' .. '{prompts[-1]}'\n"
            f"positions: {len(pprompts)} prompts, "
            f"'{pprompts[0]}' .. '{pprompts[-1]}'"
        )
    elif num_classes:
        per_class = max(1, n_samples // num_classes)
        y_grid = torch.arange(num_classes, device=dev).repeat_interleave(per_class)
        # save_image's nrow is images *per* row, i.e. a column count. Handing it
        # per_class against a label-major y_grid puts one class on each row.
        grids = [("samples", y_grid, per_class)]
    else:
        grids = [("samples", None, 8)]

    for epoch in range(1, epochs + 1):
        net.train()
        start = time.time()
        by_t = LossByT(proc.T, loss_buckets, dev)
        hinge_sum = torch.zeros((), device=dev)
        hinge_n = torch.zeros((), device=dev, dtype=torch.long)
        for x0, y in loader:
            x0 = x0.to(dev)
            # dropped per sample and redrawn every step, so one net learns both
            # eps(x,t,y) and eps(x,t,null) from the same images
            # net.null_label is the reserved id either way -- num_classes for a
            # label, the null token for a caption
            y = (
                drop_labels(y.to(dev), label_dropout, net.null_label)
                if net.null_label is not None
                else None
            )
            t = proc.sample_t(x0.shape[0], dev)
            noise = torch.randn_like(x0)
            xt = proc.q_sample(x0, t, noise)
            # eps under the diffusion process, the velocity eps - x0 under flow
            # matching. The net's output is whatever this is, and nothing in the
            # net, the samplers or the wrappers is told which.
            target = proc.target(x0, noise)
            if swap_weight:
                # both branches in one doubled batch, as CFG does it
                yn = swap_color_tokens(y)
                both = net(torch.cat([xt, xt]), torch.cat([t, t]), torch.cat([y, yn]))
                pred, pred_neg = both.chunk(2)
            else:
                pred, pred_neg = net(xt, t, y), None
            # per image, then mean: same gradient as mse_loss, but the split by t
            # survives instead of being pooled away
            se = per_sample_mse(pred, target)
            loss = se.mean()
            if pred_neg is not None:
                # a dropped caption is all-null, so its swap is itself and the
                # hinge would be an unsatisfiable constant on those rows
                real = (y != net.null_label).any(1)
                h = swap_hinge(se, per_sample_mse(pred_neg, target), swap_margin)
                h = torch.where(real, h, torch.zeros_like(h))
                n_real = real.sum()
                # kept on the GPU and read once per epoch: an .item() here is a
                # sync every step, which costs more than the extra branch does
                hinge_sum += h.sum().detach()
                hinge_n += n_real
                loss = loss + swap_weight * h.sum() / n_real.clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            by_t.update(t, se)
            if ema:
                ema.update(net)
        train_s = time.time() - start

        net.eval()
        start = time.time()
        # the grids are what the run is judged on, so draw them from the weights
        # that would actually ship -- the averaged ones
        rng = ""
        with ema.as_weights(net) if ema else nullcontext():
            for tag, gy, gnrow in grids:
                n = n_samples if gy is None else gy.shape[0]
                model = net if gy is None else Guided(net, gy, guidance)
                x = smp.sample(model, (n, in_ch, size, size), dev)
                save_image(
                    x.cpu(),
                    out / f"{tag}_epoch{epoch:02d}.png",
                    nrow=gnrow,
                    normalize=True,
                    value_range=(-1, 1),
                )
                rng = rng or f"[{x.min():.2f}, {x.max():.2f}]"
        log(
            f"epoch {epoch}  loss {by_t.pooled():.4f}  "
            f"{f'hinge {hinge_sum.item() / max(int(hinge_n), 1):.4f}  ' if swap_weight else ''}"
            f"train {train_s:.0f}s  "
            f"sample {time.time() - start:.0f}s  range {rng}\n"
            + "  by t:     "
            + cols([f"{m:.4f}" for m in by_t.means()], labels)
        )
        # num_classes rides along; without it the net can't be rebuilt to load this
        torch.save(
            {
                "net": net.state_dict(),
                "ema": ema.shadow if ema else None,  # None so a loader can tell
                "fp": proc.state_dict(),  # the schedule rides along as buffers
                # eps or flow: the weights are identical in shape and mean
                # different things, so a loader that guesses gets noise
                "objective": objective,
                # everything a loader needs to rebuild this net; without them the
                # state_dict keys don't match and the failure is a stack trace
                "num_classes": num_classes,
                "attention": attention,
                "vocab_size": vocab_size,
                "pooled": pooled_context,
                "text_layers": text_layers,
                "coords": coords,
                "swap_weight": swap_weight,
                "swap_margin": swap_margin,
                "in_ch": in_ch,
                "image_size": size,
                "dataset": dataset,
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
    p.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="ignored for ddpm; also the Euler step count for --objective flow",
    )
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
        help="beta schedule for the forward process; ignored by --objective flow, "
        "which has no schedule to pick",
    )
    p.add_argument("--seed", type=int, default=0, help="init, shuffling, and noise")
    p.add_argument(
        "--objective",
        default="eps",
        choices=("eps", "flow"),
        help="eps: DDPM/DDIM, predict the noise on a variance-preserving "
        "schedule. flow: rectified flow, predict the velocity along a straight "
        "line from noise to data -- no beta schedule, and --sampler and "
        "--schedule do not apply",
    )
    p.add_argument(
        "--dataset",
        default="mnist",
        choices=("mnist", "colored", "pair"),
        help="mnist: 28x28 grayscale, class-conditioned. colored: 32x32 RGB "
        "captioned digits, one per image. pair: two digits per image, where a "
        "pooled caption cannot say which colour goes with which digit",
    )
    p.add_argument(
        "--pooled-context",
        action="store_true",
        help="baseline for --dataset colored: mean the caption into one vector and "
        "add it to temb, the way a class label is added, instead of cross-attention",
    )
    p.add_argument(
        "--text-layers",
        type=int,
        default=0,
        help="self-attention blocks over the token sequence before the U-Net "
        "reads it; 0 reproduces the runs that assign attributes at chance",
    )
    p.add_argument(
        "--coords",
        action="store_true",
        help="2D position code on the cross-attention queries, so a query can "
        "say which position is asking; adds no parameters",
    )
    p.add_argument(
        "--swap-weight",
        type=float,
        default=0.0,
        help="weight on the hinge that requires the colour-swapped caption to "
        "score worse than the true one; 0 is the plain eps objective",
    )
    p.add_argument(
        "--swap-margin",
        type=float,
        default=0.1,
        help="share of the pair's error the swapped caption must own, above an "
        "even 0.5 split; 0.1 asks for 1.5x the true caption's",
    )
    p.add_argument(
        "--attention",
        action="store_true",
        help="self-attention in the 7x7 bottleneck",
    )
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
