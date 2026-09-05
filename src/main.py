"""Train the eps-predictor on MNIST, sample a grid after every epoch."""

import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from forward_process import ForwardProcess
from sampler import DDPMSampler
from unet import UNet
from utils import repo_root


def main(epochs: int = 5, batch_size: int = 128, lr: float = 2e-4, n_samples: int = 64):
    root = repo_root()
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
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
    print(
        f"device {dev}  params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M  "
        f"{len(loader)} steps/epoch",
        flush=True,
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
        print(
            f"epoch {epoch}  loss {total / len(loader):.4f}  train {train_s:.0f}s  "
            f"sample {time.time() - start:.0f}s  range [{x.min():.2f}, {x.max():.2f}]",
            flush=True,
        )
        torch.save({"net": net.state_dict(), "fp": fp.state_dict()}, out / "ckpt.pt")


if __name__ == "__main__":
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
