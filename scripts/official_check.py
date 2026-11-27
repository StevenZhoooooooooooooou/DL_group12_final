"""Check our attention implementation against the official expVAE code on the *same* weights and inputs.

The official checkpoint is loaded into both the official ConvVAE (+ its hook-based GradCAM, target 'encoder.2')
and our MNISTVAE (key-mapped). Our map is computed at the same tensor (conv2's Conv2d output, before ReLU)
in 'abs' mode, which is what the official code does; agreement is measured per image.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "official_run"))
from model import ConvVAE                      # official
from gradcam import GradCAM                    # official
from vaex import data
from vaex.attention import FeatureTap, anomaly_attention
from vaex.common import norm01, save_json
from vaex.models import MNISTVAE

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.join(ROOT, "official_run/ckpt/model_best.pth"))
p.add_argument("--outliers", type=int, nargs="+", default=[7, 2])
p.add_argument("--name", default="official_digit1")
args = p.parse_args()
dev = "cuda"
out = os.path.join(ROOT, "outputs/official_mnist")
os.makedirs(out, exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu")
sd = ck["state_dict"] if "state_dict" in ck else ck
off = ConvVAE(32)
off.load_state_dict(sd)
KEYMAP = {"encoder.0": "conv1.0", "encoder.2": "conv2.0", "encoder.5": "fc.1", "fc1": "fc_mu", "fc2": "fc_lv",
          "decoder.0": "dec.0", "decoder.2": "dec.2", "decoder.6": "dec.5", "decoder.8": "dec.7"}
ours = MNISTVAE(32)
ours.load_state_dict({KEYMAP[k.rsplit(".", 1)[0]] + "." + k.rsplit(".", 1)[1]: v for k, v in sd.items()})
ours.to(dev).eval()

x_all, y_all = data.mnist(train=False)
report, panels = {}, []
for d in args.outliers:
    x = x_all[y_all == d][:128].to(dev)            # = first batch of the official test loader (shuffle=False)
    g = GradCAM(off, target_layer="encoder.2", cuda=True)
    _, mu, logvar = g.forward(x)
    off.zero_grad()
    g.backward(mu, logvar, 0, 1)
    m_off = g.generate()[:, 0].detach().cpu().numpy()

    tap = FeatureTap(ours.conv2[0])                # same tensor as 'encoder.2' (conv output before ReLU)
    m_abs = anomaly_attention(ours, tap, x, mode="absup")[0].detach().cpu().numpy()
    m_relu = anomaly_attention(ours, tap, x, mode="relu")[0].detach().cpu().numpy()
    tap.remove()

    def corr(a, b):
        r = [np.corrcoef(a[i].ravel(), b[i].ravel())[0, 1] for i in range(len(a)) if a[i].std() > 0 and b[i].std() > 0]
        return float(np.min(r)), float(np.mean(r))

    # the official code divides gradients by a batch-wide L2 norm; remove that single scale before comparing
    scale = (m_off * m_abs).sum() / (m_abs * m_abs).sum()
    rel_err = float(np.abs(m_off - scale * m_abs).max() / np.abs(m_off).max())
    report[f"outlier_{d}"] = {"corr_official_vs_ours_abs (min, mean)": corr(m_off, m_abs),
                              "max_rel_error_after_scale": rel_err,
                              "corr_official_vs_paper_relu (min, mean)": corr(m_off, m_relu)}
    print(d, report[f"outlier_{d}"])
    panels.append((d, x[:4].cpu().numpy(), m_off[:4], m_abs[:4], m_relu[:4]))

save_json(report, f"{out}/{args.name}_equivalence.json")
cols = ["input", "official code", "ours (official mode)", "ours (paper: ReLU)"]
fig, axes = plt.subplots(4 * len(panels), 4, figsize=(6.4, 1.65 * 4 * len(panels)))
for pi, (d, xs, a, b, c) in enumerate(panels):
    for i in range(4):
        r = pi * 4 + i
        axes[r, 0].imshow(xs[i, 0], cmap="gray")
        for j, m in enumerate([a[i], b[i], c[i]]):
            axes[r, j + 1].imshow(xs[i, 0], cmap="gray")
            axes[r, j + 1].imshow(norm01(m), cmap="jet", alpha=0.5)
        axes[r, 0].set_ylabel(f"digit {d}", fontsize=8)
for j, t in enumerate(cols):
    axes[0, j].set_title(t, fontsize=8)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(f"{out}/{args.name}_comparison.png", dpi=110)
print("saved", f"{out}/{args.name}_comparison.png")
