"""Exp. 3 - MVTec-AD (paper Table 2 + Fig. 6).

Per category: one-class VAE with a ResNet18 encoder and 32-d latent, 256x256, random mirroring +
rotation augmentation.  Pixel-level ROC AUC and best IoU over the category's whole test set for
  * Vanilla-VAE reconstruction difference (baseline)
  * anomaly attention for every (layer, score, mode) combination; the paper's setting is
    score="mu" (s = sum_i mu_i) with mode="relu" (Eq. 2).
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vaex import data
from vaex.attention import FeatureTap, anomaly_attention
from vaex.common import seed_all, train_vae, pixel_auroc_iou, save_json, norm01
from vaex.models import ResNet18VAE, ResNet18VAESupp

p = argparse.ArgumentParser()
p.add_argument("--classes", nargs="+", default=data.MVTEC_CLASSES)
p.add_argument("--iters", type=int, default=4000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--recon", default="bce")
p.add_argument("--pretrained", type=int, default=0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="outputs/mvtec")
p.add_argument("--arch", choices=["gap", "supp"], default="gap",
               help="gap = global-average-pooled head (original run); supp = flattened head of supplement Table 3")
p.add_argument("--rot", type=float, default=15.0, help="max rotation for augmentation (supplement: 30)")
p.add_argument("--cosine", type=int, default=1, help="cosine lr decay (1) or constant lr (0, supplement)")
p.add_argument("--train_only", type=int, default=0, help="skip the built-in evaluation (use scripts/reeval.py)")
args = p.parse_args()
dev = "cuda"
tag = f"{'supp_' if args.arch == 'supp' else ''}{'pt' if args.pretrained else 'scratch'}_{args.recon}"
out = os.path.join(args.out, tag)
os.makedirs(out, exist_ok=True)

LAYERS = ["layer1", "layer2", "layer3", "layer4"]
SCORES = ["mu", "ndd", "kl", "mahal"]
MODES = ["relu", "abs"]

for cls in args.classes:
    if os.path.exists(f"{out}/{cls}.json") or (args.train_only and os.path.exists(f"{out}/{cls}.pt")):
        print(f"[{cls}] done, skipping"); continue
    seed_all(args.seed)
    t0 = time.time()
    x_tr, _, _ = data.mvtec(cls, train=True)
    x_te, m_te, types = data.mvtec(cls, train=False)
    print(f"[{cls}] train {len(x_tr)}  test {len(x_te)}  anomalous px {m_te.float().mean():.4f}", flush=True)

    model = (ResNet18VAESupp if args.arch == "supp" else ResNet18VAE)(32, pretrained=bool(args.pretrained))
    ckpt = f"{out}/{cls}.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt)); model.to(dev).eval()
    else:
        train_vae(model, x_tr, args.iters, args.bs, args.lr, dev, recon=args.recon,
                  augment=lambda x: data.augment_mirror_rotate(x, args.rot), log_every=2000, x_is_uint8=True,
                  log=lambda s: print(s, flush=True), cosine=bool(args.cosine))
        torch.save(model.state_dict(), ckpt)
    if args.train_only:
        print(f"[{cls}] trained ({time.time() - t0:.0f}s)", flush=True)
        continue

    def batches(x, bs=16):
        for i in range(0, len(x), bs):
            yield x[i:i + bs].to(dev).float().div(255)

    maps = {}
    with torch.no_grad():
        maps["Vanilla-VAE"] = torch.cat([(x - model.decode(model.encode(x)[0])).abs().mean(1).cpu()
                                         for x in batches(x_te)]).numpy()
        enc = [model.encode(x) for x in batches(x_tr)]
        mu_all = torch.cat([e[0] for e in enc])
        normal_stats = (mu_all.mean(0), mu_all.var(0) + torch.cat([e[1] for e in enc]).exp().mean(0))
    torch.manual_seed(0)
    for layer in LAYERS:
        tap = FeatureTap(model.attn_layers[layer])
        for score in SCORES:
            for mode in MODES:
                maps[f"{layer},{score},{mode}"] = torch.cat([
                    anomaly_attention(model, tap, x, mode=mode, score=score, normal_stats=normal_stats)[0].detach().cpu()
                    for x in batches(x_te)]).numpy()
        tap.remove()

    res = {}
    for k, v in maps.items():
        auc, iou = pixel_auroc_iou(v, m_te.numpy())
        res[k] = {"auroc": auc, "best_iou": iou}
    save_json(res, f"{out}/{cls}.json")
    paper_key = "layer2,mu,relu"
    print(f"[{cls}] recon AUROC {res['Vanilla-VAE']['auroc']:.3f} | paper-setting ({paper_key}) "
          f"{res[paper_key]['auroc']:.3f} | best attention "
          f"{max(v['auroc'] for k, v in res.items() if k != 'Vanilla-VAE'):.3f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- Fig. 6 style: one example per defect type ----
    dtypes = [t for t in dict.fromkeys(types) if t != "good"][:4]
    picks = [types.index(t) for t in dtypes]
    show = ["Vanilla-VAE", "layer2,mu,relu", "layer2,kl,abs"]
    fig, axes = plt.subplots(2 + len(show), len(picks), figsize=(2.4 * len(picks), 2.4 * (2 + len(show))), squeeze=False)
    for c, i in enumerate(picks):
        img = x_te[i].permute(1, 2, 0).numpy()
        axes[0, c].imshow(img); axes[0, c].set_title(dtypes[c], fontsize=9)
        axes[1, c].imshow(m_te[i], cmap="gray")
        for r, k in enumerate(show):
            axes[2 + r, c].imshow(img); axes[2 + r, c].imshow(norm01(maps[k][i]), cmap="jet", alpha=0.5)
    for r, name in enumerate(["input", "GT"] + show):
        axes[r, 0].set_ylabel(name, fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(cls); fig.tight_layout()
    fig.savefig(f"{out}/{cls}_qualitative.png", dpi=90)
    plt.close(fig)
    del maps
