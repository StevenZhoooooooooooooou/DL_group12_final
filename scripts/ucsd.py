"""Exp. 2 - UCSD Ped1 (paper Table 1 + Fig. 5).

One-class VAE on the 34 normal training clips (100x100 grey).  Pixel-level AUROC on the 10 test
clips that have pixel ground truth, comparing
  * Vanilla-VAE: |x - x_hat| reconstruction difference
  * Ours(ConvK): anomaly attention backpropagated to conv layer K (50x50, 25x25, 12x12)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vaex import data
from vaex.attention import FeatureTap, anomaly_attention
from vaex.common import seed_all, train_vae, pixel_auroc_iou, save_json, norm01
from vaex.models import Ped1VAE, Ped1VAESupp

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=10000)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--recon", default="mse")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="outputs/ucsd")
p.add_argument("--arch", choices=["bn", "supp"], default="bn", help="supp = supplement Table 2 (no BatchNorm)")
p.add_argument("--cosine", type=int, default=1)
p.add_argument("--train_only", type=int, default=0)
args = p.parse_args()
dev = "cuda"
os.makedirs(args.out, exist_ok=True)
seed_all(args.seed)

x_tr, _ = data.ucsd_ped1(train=True)
x_te, m_te = data.ucsd_ped1(train=False)
print(f"train {tuple(x_tr.shape)}  test {tuple(x_te.shape)}  anomalous px {m_te.float().mean():.4f}")

model = Ped1VAESupp(32) if args.arch == "supp" else Ped1VAE(32)
ckpt = f"{args.out}/vae_{'supp_' if args.arch == 'supp' else ''}seed{args.seed}.pt"
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt)); model.to(dev).eval()
else:
    hist = train_vae(model, x_tr, args.iters, args.bs, args.lr, dev, recon=args.recon, log_every=2000,
                     cosine=bool(args.cosine))
    torch.save(model.state_dict(), ckpt)
if args.train_only:
    sys.exit(0)


def run_maps(fn, bs=100):
    outs = []
    for i in range(0, len(x_te), bs):
        outs.append(fn(x_te[i:i + bs].to(dev)).detach().cpu())
    return torch.cat(outs).numpy()


results, maps = {}, {}
with torch.no_grad():
    maps["Vanilla-VAE"] = run_maps(lambda x: (x - model.decode(model.encode(x)[0])).abs()[:, 0])
    mus, lvs = zip(*[model.encode(x_tr[i:i + 500].to(dev)) for i in range(0, len(x_tr), 500)])
    mu_all = torch.cat(mus)
    # statistics of the normal-training embeddings (mu_x, sigma_x^2) used by Eq. 4
    normal_stats = (mu_all.mean(0), mu_all.var(0) + torch.cat(lvs).exp().mean(0))
torch.manual_seed(0)
for layer in ["conv1", "conv2", "conv3"]:
    tap = FeatureTap(model.attn_layers[layer])
    for score in ["mu", "ndd", "kl", "mahal"]:
        for mode in ["relu", "abs"]:
            maps[f"Ours({layer},{score},{mode})"] = run_maps(
                lambda x: anomaly_attention(model, tap, x, mode=mode, score=score, normal_stats=normal_stats)[0])
    tap.remove()

for k, v in maps.items():
    auc, iou = pixel_auroc_iou(v, m_te.numpy())
    results[k] = {"auroc": auc, "best_iou": iou}
    print(f"{k:28s} AUROC {auc:.3f}  bestIoU {iou:.3f}")
results["paper"] = {"Vanilla-VAE": 0.86, "Ours(Conv1)": 0.89, "Ours(Conv2)": 0.92, "Ours(Conv3)": 0.91}
save_json(results, f"{args.out}/table1_seed{args.seed}.json")

# ---- Fig. 5: qualitative examples (frames with the largest anomalous area) ----
area = m_te.flatten(1).float().sum(1)
clip_of = np.repeat(np.arange(10), 200)
picks = [int(np.argmax(np.where(clip_of == c, area.numpy(), -1))) for c in range(0, 10, 2)]
best = max((k for k in maps if k.startswith("Ours") and ",mu," in k), key=lambda k: results[k]["auroc"])
cols = ["input", "GT", f"attention {best}", "Vanilla-VAE |x - x_hat|"]
fig, axes = plt.subplots(len(picks), 4, figsize=(9, 2.3 * len(picks)))
for r, i in enumerate(picks):
    axes[r, 0].imshow(x_te[i, 0], cmap="gray")
    axes[r, 1].imshow(m_te[i], cmap="gray")
    axes[r, 2].imshow(x_te[i, 0], cmap="gray"); axes[r, 2].imshow(norm01(maps[best][i]), cmap="jet", alpha=0.5)
    axes[r, 3].imshow(norm01(maps["Vanilla-VAE"][i]), cmap="jet")
for c, t in enumerate(cols):
    axes[0, c].set_title(t, fontsize=9)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(f"{args.out}/fig5_qualitative_seed{args.seed}.png", dpi=110)
