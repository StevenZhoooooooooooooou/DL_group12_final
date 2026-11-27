"""Exp. 1 - MNIST (paper Fig. 1 and Fig. 4, qualitative).

One-class VAE trained on a single digit (official expVAE setup: ConvVAE, z=32, BCE, Adam 1e-3,
batch 128, 100 epochs), then anomaly attention (s = sum_i mu_i) on test digits of all other classes,
and element-wise attention M_i for individual latent dims.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vaex import data
from vaex.attention import FeatureTap, anomaly_attention, elementwise_attention
from vaex.common import seed_all, train_vae, save_json, norm01
from vaex.models import MNISTVAE

p = argparse.ArgumentParser()
p.add_argument("--normal", type=int, nargs="+", default=[1, 3])
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--layer", default="conv2")
p.add_argument("--out", default="outputs/mnist")
p.add_argument("--seed", type=int, default=1)
args = p.parse_args()
dev = "cuda"
os.makedirs(args.out, exist_ok=True)

x_test_all, y_test_all = data.mnist(train=False)

for normal in args.normal:
    seed_all(args.seed)
    x_tr, _ = data.mnist(train=True, digit=normal)
    model = MNISTVAE(32)
    ckpt = f"{args.out}/vae_digit{normal}.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        model.to(dev).eval()
    else:
        iters = args.epochs * len(x_tr) // 128
        print(f"[digit {normal}] training on {len(x_tr)} images for {iters} iters")
        train_vae(model, x_tr, iters, 128, 1e-3, dev, recon="bce", log_every=1000)
        torch.save(model.state_dict(), ckpt)
    tap = FeatureTap(model.attn_layers[args.layer])

    # ---- Fig. 4: anomaly attention for every other digit (6 examples each) ----
    g = torch.Generator().manual_seed(0)
    digits = [d for d in range(10) if d != normal]
    n_ex = 6
    fig, axes = plt.subplots(len(digits), 2 * n_ex, figsize=(1.1 * 2 * n_ex, 1.15 * len(digits)))
    for r, d in enumerate(digits):
        idx = torch.nonzero(y_test_all == d)[:, 0]
        idx = idx[torch.randperm(len(idx), generator=g)[:n_ex]]
        x = x_test_all[idx].to(dev)
        M, _ = anomaly_attention(model, tap, x, mode="relu")
        for c in range(n_ex):
            img = x[c, 0].cpu().numpy()
            axes[r, 2 * c].imshow(img, cmap="gray")
            axes[r, 2 * c + 1].imshow(img, cmap="gray")
            axes[r, 2 * c + 1].imshow(norm01(M[c].detach().cpu().numpy()), cmap="jet", alpha=0.5)
        axes[r, 0].set_ylabel(str(d), rotation=0, labelpad=12, fontsize=12)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Anomaly attention, one-class VAE trained on digit {normal} (layer {args.layer}, ReLU, s=sum mu)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig4_anomaly_attention_digit{normal}.png", dpi=110)
    plt.close(fig)

    # ---- Fig. 1: element-wise attention M_i for the 3 most informative latent dims ----
    x_n, _ = data.mnist(train=False, digit=normal)
    with torch.no_grad():
        mu, lv = model.encode(x_n[:2000].to(dev))
        kl_dim = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)
    top = kl_dim.argsort(descending=True)[:3].tolist()
    xs = x_n[torch.randperm(len(x_n), generator=g)[:5]].to(dev)
    Ms, _ = elementwise_attention(model, tap, xs, dims=top, use_mean=True)
    fig, axes = plt.subplots(len(top) + 1, 5, figsize=(7, 1.5 * (len(top) + 1)))
    for c in range(5):
        axes[0, c].imshow(xs[c, 0].cpu(), cmap="gray")
        for r in range(len(top)):
            axes[r + 1, c].imshow(xs[c, 0].cpu(), cmap="gray")
            axes[r + 1, c].imshow(norm01(Ms[r, c].detach().cpu().numpy()), cmap="jet", alpha=0.5)
    for r, dim in enumerate(top):
        axes[r + 1, 0].set_ylabel(f"z{dim}", rotation=0, labelpad=14)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Element-wise attention (digit-{normal} VAE)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig1_elementwise_digit{normal}.png", dpi=110)
    plt.close(fig)
    save_json({"top_kl_dims": top, "kl_per_dim": kl_dim.tolist()}, f"{args.out}/digit{normal}_latent_kl.json")
    tap.remove()
    print(f"[digit {normal}] figures written to {args.out}")
