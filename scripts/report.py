"""Aggregate all experiment outputs into tables / figures next to the paper's numbers.

Writes outputs/report/{results.md, fig8_disentanglement.png, fig9_attention.png, mvtec_summary.png}.
"""
import glob
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vaex.data import MVTEC_CLASSES

OUT = "outputs/report"
os.makedirs(OUT, exist_ok=True)
md = ["# Reproduction results: Towards Visually Explaining Variational Autoencoders (CVPR 2020)\n"]

# ------------------------------------------------------------------ Table 1: UCSD Ped1
f = "outputs/ucsd/table1_seed1.json"
if os.path.exists(f):
    r = json.load(open(f))
    md += ["## Table 1 - UCSD Ped1, pixel-level AUROC (10 test clips with pixel GT)\n",
           "| method | paper | ours (score = sum mu, ReLU: paper setting) | ours (abs, official code) | best non-paper score (KL / Mahalanobis) |",
           "|---|---|---|---|---|",
           f"| Vanilla-VAE (recon. diff.) | 0.86 | {r['Vanilla-VAE']['auroc']:.3f} | | |"]
    for k, l in [("Conv1", "conv1"), ("Conv2", "conv2"), ("Conv3", "conv3")]:
        alt = max((v["auroc"], n) for n, v in r.items() if n.startswith(f"Ours({l},") and (",kl," in n or ",mahal," in n))
        md.append(f"| Ours({k}) | {r['paper']['Ours(' + k + ')']:.2f} | {r[f'Ours({l},mu,relu)']['auroc']:.3f} | "
                  f"{r[f'Ours({l},mu,abs)']['auroc']:.3f} | {alt[0]:.3f} `{alt[1]}` |")
    md.append("")

# ------------------------------------------------------------------ Table 2: MVTec-AD
PAPER = {  # (AUROC, IoU) of 'ours' in the paper, plus AE(L2) baseline from Bergmann et al.
    "carpet": (0.78, 0.10, 0.59), "grid": (0.73, 0.02, 0.90), "leather": (0.95, 0.24, 0.75),
    "tile": (0.80, 0.23, 0.51), "wood": (0.77, 0.14, 0.73), "bottle": (0.87, 0.27, 0.86),
    "cable": (0.90, 0.18, 0.86), "capsule": (0.74, 0.11, 0.88), "hazelnut": (0.98, 0.44, 0.95),
    "metal_nut": (0.94, 0.49, 0.86), "pill": (0.83, 0.18, 0.85), "screw": (0.97, 0.17, 0.96),
    "toothbrush": (0.94, 0.14, 0.93), "transistor": (0.93, 0.30, 0.86), "zipper": (0.78, 0.06, 0.77)}
for tag in sorted(glob.glob("outputs/mvtec/*/")):
    res = {c: json.load(open(f"{tag}{c}.json")) for c in MVTEC_CLASSES if os.path.exists(f"{tag}{c}.json")}
    if not res:
        continue
    keys = [k for k in next(iter(res.values())) if k != "Vanilla-VAE"]
    # one configuration for all classes (no per-class selection): mean AUROC over classes
    mean_auc = {k: np.mean([res[c][k]["auroc"] for c in res]) for k in keys}
    best_paper_cfg = max((k for k in keys if ",mu,relu" in k), key=mean_auc.get)
    best_cfg = max(keys, key=mean_auc.get)
    name = os.path.basename(tag.rstrip("/"))
    md += [f"## Table 2 - MVTec-AD pixel-level ROC AUC / best IoU  (`{name}`, {len(res)}/15 classes)\n",
           f"Paper-setting column = score `sum mu`, ReLU (Eq. 2), single layer for all classes "
           f"(`{best_paper_cfg}`, chosen by mean AUROC).  Best-variant column = best single configuration "
           f"over all layers/scores/modes (`{best_cfg}`, not in the paper).\n",
           "| class | paper AE(L2) | paper ours | our Vanilla-VAE recon | our attention, paper setting | our attention, best variant |",
           "|---|---|---|---|---|---|"]
    rows = []
    for c in MVTEC_CLASSES:
        if c not in res:
            continue
        pa, pi, ae = PAPER[c]
        rv, rp, rb = res[c]["Vanilla-VAE"], res[c][best_paper_cfg], res[c][best_cfg]
        rows.append((pa, pi, ae, rv["auroc"], rv["best_iou"], rp["auroc"], rp["best_iou"], rb["auroc"], rb["best_iou"]))
        md.append(f"| {c} | {ae:.2f} | {pa:.2f} / {pi:.2f} | {rv['auroc']:.3f} / {rv['best_iou']:.3f} | "
                  f"{rp['auroc']:.3f} / {rp['best_iou']:.3f} | {rb['auroc']:.3f} / {rb['best_iou']:.3f} |")
    m = np.mean(rows, 0)
    md += [f"| **mean** | {m[2]:.2f} | {m[0]:.2f} / {m[1]:.2f} | {m[3]:.3f} / {m[4]:.3f} | "
           f"{m[5]:.3f} / {m[6]:.3f} | {m[7]:.3f} / {m[8]:.3f} |", ""]
    # layer x score x mode mean-AUROC grid
    md += ["Mean AUROC over classes for every attention configuration:\n",
           "| layer | " + " | ".join(f"{s},{mo}" for s in ["mu", "ndd", "kl", "mahal"] for mo in ["relu", "abs"]) + " |",
           "|---|" + "---|" * 8]
    for l in ["layer1", "layer2", "layer3", "layer4"]:
        md.append(f"| {l} | " + " | ".join(f"{mean_auc[f'{l},{s},{mo}']:.3f}" for s in ["mu", "ndd", "kl", "mahal"]
                                           for mo in ["relu", "abs"]) + " |")
    md.append("")
    # bar chart
    cls = [c for c in MVTEC_CLASSES if c in res]
    w = 0.2
    xs = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(13, 4))
    for i, (lab, vals) in enumerate([("paper (ours)", [PAPER[c][0] for c in cls]),
                                     ("our recon. baseline", [res[c]["Vanilla-VAE"]["auroc"] for c in cls]),
                                     (f"our attention, paper setting", [res[c][best_paper_cfg]["auroc"] for c in cls]),
                                     (f"our attention, best variant", [res[c][best_cfg]["auroc"] for c in cls])]):
        ax.bar(xs + (i - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(cls, rotation=30); ax.set_ylim(0.3, 1.0); ax.set_ylabel("pixel ROC AUC")
    ax.legend(ncol=4, fontsize=8, loc="lower center"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/mvtec_summary_{name}.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ Fig. 8: dSprites
runs = {}
for f in sorted(glob.glob("outputs/dsprites/*.json")):
    d = json.load(open(f))
    if not d["evals"] or d["evals"][-1]["it"] < d["args"]["iters"]:
        continue
    cfg = os.path.basename(f).rsplit("_s", 1)[0]
    runs.setdefault(cfg, []).append(d["evals"][-1])
if runs:
    md += ["## Fig. 8 - dSprites: FactorVAE disentanglement metric vs reconstruction error (3e5 iters)\n",
           "Paper: AD-FactorVAE ~0.90, FactorVAE (gamma=40) ~0.82, beta-VAE (beta=4) 0.73, recon. error ~40.\n",
           "| configuration | seeds | disentanglement mean +- std | best seed | recon. error mean |", "|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cfg, ev in sorted(runs.items()):
        s = np.array([e["disentanglement"] for e in ev]); r = np.array([e["recon"] for e in ev])
        md.append(f"| {cfg} | {len(ev)} | {s.mean():.3f} +- {s.std():.3f} | {s.max():.3f} | {r.mean():.1f} |")
        ax.scatter(r, s, label=cfg, s=40)
    paper = {"beta-VAE b=4 (paper)": (None, 0.73), "FactorVAE g=40 (paper)": (40, 0.82), "AD-FactorVAE (paper)": (40, 0.90)}
    for lab, (rx, sy) in paper.items():
        ax.axhline(sy, ls="--", lw=0.8, color="gray"); ax.text(ax.get_xlim()[0], sy + 0.005, lab, fontsize=7, color="gray")
    ax.set_xlabel("reconstruction error (BCE, nats/image)"); ax.set_ylabel("FactorVAE disentanglement metric")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig8_disentanglement.png", dpi=120); plt.close(fig)
    md.append("")

# ------------------------------------------------------------------ Fig. 9 + attention overlap
from vaex.attention import FeatureTap, elementwise_attention, attention_disentanglement_loss
from vaex.data import DSprites
from vaex.models import FactorVAENet, FactorVAENetSupp
models = [m for m in ["factorvae_g40", "adfactorvae_g40_l10_conv2_relu", "adfactorvae_g40_l10_conv2_abs",
                      "adfactorvae_g40_l50_conv2_abs", "factorvae_g40_supp", "adfactorvae_g40_l10_conv2_relu_supp",
                      "adfactorvae_g40_l10_conv2_abs_supp"] if glob.glob(f"outputs/dsprites/{m}_s[0-9].pt")]
if models:
    ds = DSprites()
    g = torch.Generator().manual_seed(3)
    x = ds.batch(torch.randint(0, ds.N, (512,), generator=g), "cuda")
    md += ["Attention overlap between latent dims (Eq. 5 with |.| maps at conv2, averaged over all pairs of active dims "
           "and 512 images; lower = more separated attention):\n", "| model | seed | active dims | mean pairwise overlap |", "|---|---|---|---|"]
    fig, axes = plt.subplots(1 + 2 * len(models), 5, figsize=(8, 1.7 * (1 + 2 * len(models))))
    for c in range(5):
        axes[0, c].imshow(x[c, 0].cpu(), cmap="gray")
    for mi, m in enumerate(models):
        for ck in sorted(glob.glob(f"outputs/dsprites/{m}_s[0-9].pt")):
            net = FactorVAENetSupp(32) if m.endswith("_supp") else FactorVAENet(10)
            net.load_state_dict(torch.load(ck)); net.cuda().eval()
            tap = FeatureTap(net.attn_layers["conv2"])
            M, (mu, lv, z) = elementwise_attention(net, tap, x, use_mean=True, mode="abs")
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)
            act = (kl > 0.1).nonzero()[:, 0].tolist()
            ov = np.mean([attention_disentanglement_loss(M[i], M[j], skip_empty=True).item()
                          for i, j in itertools.combinations(act, 2)]) if len(act) > 1 else float("nan")
            md.append(f"| {m} | {ck[-4]} | {len(act)} | {ov:.3f} |")
            if ck.endswith("_s1.pt"):
                top = kl.argsort(descending=True)[:2].tolist()
                for r, d in enumerate(top):
                    for c in range(5):
                        ax = axes[1 + 2 * mi + r, c]
                        ax.imshow(x[c, 0].cpu(), cmap="gray")
                        ax.imshow(M[d, c].detach().cpu(), cmap="jet", alpha=0.6)
                    axes[1 + 2 * mi + r, 0].set_ylabel(f"{m.replace('_g40', '')}\nz{d}", fontsize=6)
            tap.remove()
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/fig9_attention.png", dpi=120); plt.close(fig)
    md.append("")

open(f"{OUT}/results.md", "w").write("\n".join(md))
print("\n".join(md))
