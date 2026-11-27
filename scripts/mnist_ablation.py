"""Ablation for Fig. 4: aggregation mode (paper ReLU vs official-code abs) x target layer."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vaex import data
from vaex.attention import FeatureTap, anomaly_attention
from vaex.common import norm01
from vaex.models import MNISTVAE

dev = "cuda"
normal = int(sys.argv[1]) if len(sys.argv) > 1 else 1
model = MNISTVAE(32); model.load_state_dict(torch.load(f"outputs/mnist/vae_digit{normal}.pt")); model.to(dev).eval()
xt, yt = data.mnist(train=False)
g = torch.Generator().manual_seed(0)
digits = [7, 2, 4, 9, 0] if normal == 1 else [8, 5, 2, 7, 0]
cfgs = [(l, m) for l in ["conv1", "conv2"] for m in ["relu", "abs"]]
fig, axes = plt.subplots(len(digits) * 2, 1 + len(cfgs), figsize=(1.6 * (1 + len(cfgs)), 1.6 * 2 * len(digits)))
for r, d in enumerate(digits):
    idx = torch.nonzero(yt == d)[:, 0]; idx = idx[torch.randperm(len(idx), generator=g)[:2]]
    x = xt[idx].to(dev)
    for j in range(2):
        axes[2*r+j, 0].imshow(x[j, 0].cpu(), cmap="gray")
    for c, (l, m) in enumerate(cfgs):
        tap = FeatureTap(model.attn_layers[l])
        M, _ = anomaly_attention(model, tap, x, mode=m)
        tap.remove()
        for j in range(2):
            ax = axes[2*r+j, c+1]
            ax.imshow(x[j, 0].cpu(), cmap="gray"); ax.imshow(norm01(M[j].detach().cpu().numpy()), cmap="jet", alpha=0.5)
            if r == 0 and j == 0: ax.set_title(f"{l}/{m}", fontsize=9)
axes[0, 0].set_title("input", fontsize=9)
for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"normal digit = {normal}"); fig.tight_layout()
fig.savefig(f"outputs/mnist/ablation_mode_layer_digit{normal}.png", dpi=100)
