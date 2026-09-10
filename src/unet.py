"""U-Net eps-predictor: down path, bottleneck, up path with skip concatenation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from attention import Attention
from resblock import ResBlock
from text_encoder import TextBlock, TextEncoder
from timestep_embedding import timestep_embedding


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Nearest x2 then a conv -- transposed conv here gives checkerboard artifacts."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Predicts eps from (x_t, t), and from conditioning: a class label when
    `num_classes` is set, or a caption when `vocab_size` is (`pooled=True` for
    the pooled-into-temb baseline instead; `text_layers>0` contextualises the
    tokens first). Output conv is
    zero-init, so a fresh net predicts exactly zero noise and the MSE loss starts
    at E[eps²] = 1.

    The two conditioning paths are deliberately different. A label is one vector
    added to `temb`, which reaches every pixel identically. A caption is a
    sequence read by cross-attention after every ResBlock, and never touches
    `temb` at all -- so nothing about a caption is pooled before the net sees it,
    and each position picks out the words that concern it. `coords=True` tells
    each of those positions where it is (`coords.py`), which is what "the words
    that concern it" needs in order to mean anything."""

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        mults: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 1,
        time_dim: int = 256,
        groups: int = 8,
        num_classes: int | None = None,
        attention: bool = False,
        vocab_size: int | None = None,
        context_dim: int = 128,
        cross_heads: int = 4,
        max_tokens: int = 32,
        null_token: int = 0,
        pooled: bool = False,
        text_layers: int = 0,
        coords: bool = False,
    ):
        super().__init__()
        self.base, self.mults = base, mults
        self.num_classes, self.vocab_size = num_classes, vocab_size
        assert not (num_classes and vocab_size), "labels or captions, not both"
        self.null_label = None  # the reserved "no conditioning" id, for CFG
        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        if num_classes is not None:
            # one extra row: index num_classes is the null label. Unused here, but
            # CFG needs the same weights to have seen it, so reserve it before
            # training rather than retrofitting it after.
            self.null_label = num_classes
            self.label_emb = nn.Embedding(num_classes + 1, time_dim)
        if vocab_size is not None:
            # index 0 by convention (see colored_mnist.VOCAB): an all-null
            # sequence is the unconditional prompt, so the weights must have seen
            # it during training just like the null label row.
            assert 0 <= null_token < vocab_size, (null_token, vocab_size)
            self.null_label = null_token
            self.token_emb = nn.Embedding(vocab_size, context_dim)
            self.token_pos = nn.Parameter(torch.zeros(1, max_tokens, context_dim))
            # Zero-init starts the sequence as a pure bag of words, so any
            # order-dependence is learned rather than assumed. Measured cost of
            # that purity (`README.md`): token_pos trained to 2% of the
            # embedding norm and never became a usable signal, because the net
            # can drive the loss down on word identity alone. With an encoder to
            # read them, positions get a real init and a running start.
            self.text = (
                TextEncoder(context_dim, text_layers, cross_heads)
                if text_layers
                else None
            )
            if self.text is not None:
                nn.init.normal_(self.token_pos, std=0.02)
            # the baseline this item exists to beat: mean the sequence into one
            # vector and add it to temb, exactly as a class label is added. Every
            # word still reaches the net; nothing records which word goes with
            # which object.
            self.pooled = pooled
            if pooled:
                self.context_pool = nn.Linear(context_dim, time_dim)
        cross_on = vocab_size is not None and not pooled
        # coords go on the cross blocks only: it is the caption lookup that needs
        # to know which position is asking. Free in parameters, so `--coords` and
        # its control have identical state dicts (`attention.py`, test 7).
        cross = {
            "heads": cross_heads,
            "groups": groups,
            "context_dim": context_dim,
            "coords": coords,
        }
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        ch, skip_chs = base, [base]  # channel counts the up path will concat back
        self.down = nn.ModuleList()
        for i, m in enumerate(mults):
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, base * m, time_dim, groups))
                ch = base * m
                if cross_on:
                    self.down.append(Attention(ch, **cross))
                skip_chs.append(ch)
            if i < len(mults) - 1:
                self.down.append(Downsample(ch))
                skip_chs.append(ch)

        self.mid1 = ResBlock(ch, ch, time_dim, groups)
        # bottleneck only: at 28x28 self-attention is 784² pairs per head, and the
        # 7x7 bottleneck already carries what the whole image contributed
        self.mid_attn = Attention(ch, groups=groups) if attention else None
        self.mid_cross = Attention(ch, **cross) if cross_on else None
        self.mid2 = ResBlock(ch, ch, time_dim, groups)

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(num_res_blocks + 1):  # +1 consumes the downsample's skip
                self.up.append(
                    ResBlock(ch + skip_chs.pop(), base * m, time_dim, groups)
                )
                ch = base * m
                if cross_on:
                    self.up.append(Attention(ch, **cross))
            if i > 0:
                self.up.append(Upsample(ch))
        assert not skip_chs, skip_chs

        self.out_norm = nn.GroupNorm(groups, ch)
        self.act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        """y is [B] class labels, or [B, L] token ids -- whichever this net was
        built for. Same argument either way, so `Conditioned` and `Guided` and the
        samplers behind them do not care which."""
        d = 2 ** (len(self.mults) - 1)
        assert x.shape[-1] % d == 0 and x.shape[-2] % d == 0, f"H,W must divide {d}"
        temb = self.time_mlp(timestep_embedding(t, self.base))
        context = None
        if self.vocab_size is not None:
            assert y is not None, "captioned net needs tokens (net.null_label for none)"
            assert y.ndim == 2 and y.shape[0] == t.shape[0], (y.shape, t.shape)
            assert y.shape[1] <= self.token_pos.shape[1], (
                y.shape,
                self.token_pos.shape,
            )
            assert 0 <= int(y.min()) and int(y.max()) < self.vocab_size, y
            context = self.token_emb(y) + self.token_pos[:, : y.shape[1]]
            if self.text is not None:
                # each word rewritten in terms of the others, before any pixel
                # attends to it -- the stage a real model gets from CLIP
                context = self.text(context)
            if self.pooled:
                temb = temb + self.context_pool(context.mean(1))
                context = None  # nothing downstream reads it; there is no reader
        elif self.num_classes is None:
            assert y is None, "unconditional net was handed a label"
        else:
            # silently dropping y would train an unconditional model that looks fine
            assert y is not None, "conditional net needs y (net.null_label for none)"
            assert y.shape == t.shape, (y.shape, t.shape)
            assert 0 <= int(y.min()) and int(y.max()) <= self.num_classes, y
            temb = temb + self.label_emb(y)  # same footing as t, one vector per sample
        h = self.conv_in(x)
        hs = [h]
        for m in self.down:
            if isinstance(m, Attention):
                # refines the block it follows rather than being a stage of its
                # own, so it overwrites that skip instead of pushing a new one
                h = hs[-1] = m(h, context)
            else:
                h = m(h, temb) if isinstance(m, ResBlock) else m(h)
                hs.append(h)
        h = self.mid1(h, temb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        if self.mid_cross is not None:
            h = self.mid_cross(h, context)
        h = self.mid2(h, temb)
        for m in self.up:
            if isinstance(m, Attention):
                h = m(h, context)
            elif isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], dim=1), temb)
            else:
                h = m(h)
        assert not hs, len(hs)
        return self.out_conv(self.act(self.out_norm(h)))


class Conditioned(nn.Module):
    """Freezes y so a conditional net still presents the (x, t) -> eps interface
    the samplers call. CFG wraps the same way, with two calls instead of one."""

    def __init__(self, net: UNet, y: Tensor):
        super().__init__()
        self.net, self.y = net, y

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(x, t, self.y)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = UNet()
    B = 4
    x = torch.randn(B, 1, 28, 28)
    t = torch.randint(0, 1000, (B,))
    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    # 1. eps has the shape of x, and every skip is consumed (asserts in forward)
    assert net(x, t).shape == x.shape
    assert net(torch.randn(1, 1, 28, 28), t[:1]).shape == (1, 1, 28, 28)

    # 2. zero-init output: a fresh net predicts no noise at all, so the MSE
    #    against eps starts at exactly E[eps²] = 1 -- the number to expect on
    #    step 0 of any training run
    assert torch.equal(net(x, t), torch.zeros_like(x))
    eps = torch.randn(B, 1, 28, 28)
    print(f"init loss: {F.mse_loss(net(x, t), eps):.4f}")
    assert abs(F.mse_loss(net(x, t), eps).item() - eps.square().mean().item()) < 1e-6

    # 3. resolutions must line up or the skip concat is wrong. 28 -> 14 -> 7 works;
    #    an odd size divides to a resolution the up path can't rebuild
    try:
        net(torch.randn(B, 1, 30, 30), t)
        raise SystemExit("guard missing")
    except AssertionError:
        pass

    # 4. it learns: overfit 8 MNIST digits, loss must fall well below the 1.0 it
    #    starts at. Every check above passes on a net that cannot train at all
    from torchvision import datasets, transforms

    from forward_process import ForwardProcess
    from utils import repo_root

    ds = datasets.MNIST(
        root=repo_root() / "data",
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x0 = torch.stack([ds[i][0] for i in range(8)]).to(dev)
    fp = ForwardProcess().to(dev)
    net = UNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    for step in range(301):
        ti = fp.sample_t(8, dev)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(net(fp.q_sample(x0, ti, noise), ti), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    assert loss.item() < 0.5

    # 5. t reaches the output, per sample. Show the vacuous case first: on a
    #    fresh net the output conv is zero, so both t give exactly zero and the
    #    check below would pass on a net that never reads t at all. That is why
    #    this test has to run after training, not before
    x, t = x0, fp.sample_t(8, dev)
    t2 = t.clone()
    t2[0] = (t[0] + 500) % 1000
    fresh = UNet().to(dev)
    moved_fresh = (fresh(x, t) - fresh(x, t2)).abs().max().item()
    print(f"row 0 moved by {moved_fresh:.4f} (fresh net)")
    assert moved_fresh == 0.0

    a, b = net(x, t), net(x, t2)
    print(f"row 0 moved by {(a[0] - b[0]).abs().max():.4f} (trained)")
    assert (a[0] - b[0]).abs().max() > 1e-4
    assert torch.equal(a[1:], b[1:])  # rows you didn't touch are untouched, exactly

    # 6. the label guards. Each of these is a bug that trains to a plausible loss
    #    and only shows up as samples that ignore what you asked for
    cnet = UNet(num_classes=10)
    xc, tc = torch.randn(B, 1, 28, 28), torch.randint(0, 1000, (B,))  # cpu; t is cuda
    yc4 = torch.randint(0, 10, (B,))
    assert cnet.null_label == 10 and cnet.label_emb.num_embeddings == 11
    for bad in (
        lambda: UNet()(xc, tc, yc4),  # label handed to an unconditional net
        lambda: cnet(xc, tc),  # no label handed to a conditional one
        lambda: cnet(xc, tc, yc4[:2]),  # y not batched alongside t
        lambda: cnet(xc, tc, yc4 + 11),  # past the null row
    ):
        try:
            bad()
            raise SystemExit("guard missing")
        except AssertionError:
            pass

    # 7. Conditioned is exactly the frozen-y call, so a sampler driving it sees
    #    what a direct call would give
    assert torch.equal(Conditioned(cnet, yc4)(xc, tc), cnet(xc, tc, yc4))

    # 8. THE property: y carries information. Overfit one digit per class, then
    #    score the same x_t with the right label and with a wrong one. A label
    #    that is wired up but ignored -- added to temb and then averaged away,
    #    say -- passes every check above and fails only here.
    per_class = {}
    for img, lab in ds:
        per_class.setdefault(lab, img)
        if len(per_class) == 10:
            break
    x0 = torch.stack([per_class[c] for c in range(10)]).to(dev)
    yc = torch.arange(10, device=dev)
    net = UNet(num_classes=10).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    for step in range(801):
        ti = fp.sample_t(10, dev)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(net(fp.q_sample(x0, ti, noise), ti, yc), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    right = wrong = 0.0
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(200):  # average over t and noise; one draw is far too noisy
            ti = fp.sample_t(10, dev)
            noise = torch.randn_like(x0)
            xt = fp.q_sample(x0, ti, noise)
            right += F.mse_loss(net(xt, ti, yc), noise).item()
            wrong += F.mse_loss(net(xt, ti, yc.roll(1)), noise).item()
    right, wrong = right / 200, wrong / 200
    print(f"eps-MSE  right label {right:.4f}   wrong label {wrong:.4f}")
    assert wrong > right * 2  # a net ignoring y scores these identically (obs. 4x)

    # 9. caption conditioning. Same y argument, a sequence instead of a scalar,
    #    and every way of getting that wrong is a net that trains to a plausible
    #    loss while ignoring most of what it was told
    from colored_mnist import SEQ_LEN, VOCAB, ColoredMNIST, caption, decode, encode

    V = len(VOCAB)
    tnet = UNet(in_ch=3, vocab_size=V)
    xt3 = torch.randn(B, 3, 32, 32)
    toks = torch.randint(0, V, (B, SEQ_LEN))
    assert tnet.num_classes is None and tnet.null_label == 0
    assert tnet(xt3, tc, toks).shape == xt3.shape
    assert torch.equal(tnet(xt3, tc, toks), torch.zeros_like(xt3))  # zero-init holds
    # L is free: that is the whole reason KV comes from tokens and not from pixels
    for lengths in (3, 7, 12):
        assert tnet(xt3, tc, toks[:, :1].repeat(1, lengths)).shape == xt3.shape
    for bad in (
        lambda: tnet(xt3, tc),  # captioned net handed no tokens
        lambda: tnet(xt3, tc, toks[0]),  # [L] instead of [B, L]
        lambda: tnet(xt3, tc, toks[:2]),  # not batched alongside t
        lambda: tnet(xt3, tc, toks + V),  # past the vocabulary
        lambda: tnet(xt3, tc, toks.repeat(1, 8)),  # longer than max_tokens
        lambda: UNet(in_ch=3, num_classes=10)(xt3, tc, toks),  # tokens to a label net
        lambda: UNet(in_ch=3, vocab_size=V, num_classes=10),  # both at once
    ):
        try:
            bad()
            raise SystemExit("guard missing")
        except AssertionError:
            pass

    # 10. THE property: the caption carries information, and it carries it word by
    #     word. Overfit six captioned images, then re-score the same x_t with a
    #     caption differing in exactly one token. A net that pools the sequence
    #     and adds it to temb would still pass "wrong caption is worse"; it is
    #     changing *one word* that separates binding from a bag of concepts.
    #     Six images of the same digit at the same position, one per colour, so
    #     the colour word is the *only* thing telling them apart. Pick six
    #     different digits instead and the digit token alone identifies each
    #     image, the net can ignore colour entirely, and this test passes
    #     vacuously (measured: 1.2x, against 3x here).
    cds = ColoredMNIST()
    picks, want = [], []
    for k in range(len(cds)):
        img, tok = cds[k]
        words = decode(tok).split()  # decode drops <pad>; joining raw ids keeps it
        col, dig, pos = words[1], words[2], " ".join(words[5:])
        # digit 4 is in no held-out pair, so all six colours of it exist in train
        if dig != "4" or pos != "center" or col in want:
            continue
        picks.append((img, tok))
        want.append(col)
        if len(picks) == 6:
            break
    assert len(picks) == 6, want
    x0c = torch.stack([p[0] for p in picks]).to(dev)
    yc2 = torch.stack([p[1] for p in picks]).to(dev)
    tnet = UNet(in_ch=3, vocab_size=V).to(dev)
    print(f"captioned params: {sum(p.numel() for p in tnet.parameters()) / 1e6:.2f}M")
    opt = torch.optim.Adam(tnet.parameters(), lr=3e-4)
    for step in range(2001):  # 800 leaves the margins below at ~2x, too thin to trust
        ti = fp.sample_t(len(picks), dev)
        noise = torch.randn_like(x0c)
        loss = F.mse_loss(tnet(fp.q_sample(x0c, ti, noise), ti, yc2), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    # one token changed: the colour word, and nothing else
    swapped = yc2.clone()
    swapped[:, 1] = yc2.roll(1, 0)[:, 1]
    nulled = torch.zeros_like(yc2)  # the unconditional prompt
    scores = dict.fromkeys(("right", "colour swapped", "null"), 0.0)
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(200):  # one draw of (t, eps) is far too noisy to compare
            ti = fp.sample_t(len(picks), dev)
            noise = torch.randn_like(x0c)
            xt = fp.q_sample(x0c, ti, noise)
            for k, yy in zip(scores, (yc2, swapped, nulled)):
                scores[k] += F.mse_loss(tnet(xt, ti, yy), noise).item() / 200
    print("eps-MSE  " + "  ".join(f"{k} {v:.4f}" for k, v in scores.items()))
    assert scores["colour swapped"] > scores["right"] * 2, scores
    assert scores["null"] > scores["right"] * 2, scores

    # 11. and the only route from caption to pixels is cross-attention. Zero the
    #     attention output projections on the *trained* net and the caption goes
    #     completely inert -- which it could not, if any of it were leaking
    #     through temb the way a class label does
    with torch.no_grad():
        for m in tnet.modules():
            if isinstance(m, Attention):
                nn.init.zeros_(m.proj.weight)
                nn.init.zeros_(m.proj.bias)
        ti = fp.sample_t(len(picks), dev)
        xt = fp.q_sample(x0c, ti)
        assert torch.equal(tnet(xt, ti, yc2), tnet(xt, ti, nulled))
    assert caption("red", "3", "center") == "a red 3 in the center"
    assert encode(caption("red", "3", "center")).shape == (SEQ_LEN,)

    # 12. the pooled baseline: same tokens, meaned into one vector and added to
    #     temb the way a class label is. Structurally it must be the *other*
    #     thing -- no cross-attention anywhere -- or the comparison in README.md
    #     is between a net and itself.
    pnet = UNet(in_ch=3, vocab_size=V, pooled=True)
    assert not any(isinstance(m, Attention) for m in pnet.modules())
    assert pnet.mid_cross is None and pnet.null_label == 0
    n_cross = sum(
        isinstance(m, Attention) for m in UNet(in_ch=3, vocab_size=V).modules()
    )
    print(f"cross-attention blocks: {n_cross} vs pooled {0}")
    assert n_cross == 10
    with torch.no_grad():  # the zero-inits again, or every check below is 0 == 0
        for m in pnet.modules():
            if isinstance(m, ResBlock):
                nn.init.normal_(m.conv2.weight, std=0.05)
        nn.init.normal_(pnet.out_conv.weight, std=0.05)
        a = pnet(xt3, tc, toks)
        assert not torch.allclose(a, pnet(xt3, tc, torch.zeros_like(toks)))
        # and it really is pooled: a permuted caption is the same bag of words,
        # so it must give the same answer once token_pos is zero (which it is at
        # init). This is exactly the property that cannot bind a word to a place.
        assert torch.allclose(
            a, pnet(xt3, tc, toks[:, torch.randperm(SEQ_LEN)]), atol=1e-5
        )

    # 13. the text encoder: the stage that makes the context worth attending to.
    #     Without it, the vector at slot 1 is the word "red" plus a positional
    #     offset that trained to 2% of its norm (`README.md`), so cross-attention
    #     assigns attributes at chance. The property to check is that a token's
    #     vector now depends on the other tokens.
    enc = UNet(in_ch=3, vocab_size=V, text_layers=2)
    assert enc.text is not None and len(enc.text.blocks) == 2
    assert enc.token_pos.abs().max() > 0, "positions must not start at zero here"
    assert UNet(in_ch=3, vocab_size=V).token_pos.abs().max() == 0  # control unchanged
    assert enc(xt3, tc, toks).shape == xt3.shape
    assert torch.equal(enc(xt3, tc, toks), torch.zeros_like(xt3))  # zero-init holds

    def context_of(net, tok):
        c = net.token_emb(tok) + net.token_pos[:, : tok.shape[1]]
        return net.text(c) if net.text is not None else c

    with torch.no_grad():
        for m in enc.text.modules():  # de-zero, or the encoder is the identity
            if isinstance(m, TextBlock):
                nn.init.normal_(m.proj.weight, std=0.05)
                nn.init.normal_(m.mlp[-1].weight, std=0.05)
        swapped = toks.clone()
        swapped[:, [1, 4]] = swapped[:, [4, 1]]  # exchange two words
        plain = UNet(in_ch=3, vocab_size=V)
        # without the encoder, exchanging two words leaves every *other* slot
        # untouched -- the context is a bag with position tags
        a, b = context_of(plain, toks), context_of(plain, swapped)
        assert torch.equal(a[:, 2], b[:, 2])
        # with it, a word that did not move still changes, because it is now
        # described in terms of the words that did
        c, d = context_of(enc, toks), context_of(enc, swapped)
        moved = (c[:, 2] - d[:, 2]).abs().max()
        print(f"unmoved token changes by {moved:.4f} once the encoder reads it")
        assert moved > 1e-3

    # 14. coords: test 13 fixed the text side of the lookup and `README.md`
    #     records that binding still failed at chance, which leaves the image
    #     side -- a query built by translation-equivariant convs does not know
    #     where it is. The check here is only that the coordinates reach the
    #     queries and change the answer; whether that buys binding is a training
    #     run, and `binding.py` is the eval.
    co = UNet(in_ch=3, vocab_size=V, coords=True)
    assert all(m.coords for m in co.modules() if isinstance(m, Attention))
    assert torch.equal(co(xt3, tc, toks), torch.zeros_like(xt3))  # zero-init holds
    # no new weights, so the control's are loadable and the runs differ in the
    # coordinates alone -- not in capacity, which would be a rival explanation
    plain = UNet(in_ch=3, vocab_size=V)
    assert [k for k, _ in co.named_parameters()] == [
        k for k, _ in plain.named_parameters()
    ]
    co.load_state_dict(plain.state_dict())
    with torch.no_grad():  # de-zero, or both nets predict exactly zero
        nn.init.normal_(plain.out_conv.weight, std=0.05)
        for m in plain.modules():
            if isinstance(m, Attention):
                nn.init.normal_(m.proj.weight, std=0.05)
        co.load_state_dict(plain.state_dict())  # copy again, now that it is live
        moved = (co(xt3, tc, toks) - plain(xt3, tc, toks)).abs().max()
    print(f"identical weights, coords move eps by {moved:.4f}")
    assert moved > 1e-3

    print("ok")
