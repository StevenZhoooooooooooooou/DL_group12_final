"""Re-evaluate the saved one-class VAEs without retraining, to make the anomaly results more reliable.

For every map (reconstruction residual, and attention for every layer x score x mode) it records
pixel AUROC / best IoU under four post-processing variants:
  raw        - attention values as produced (what scripts/mvtec.py and scripts/ucsd.py report)
  norm       - per-image min-max normalisation (the usual Grad-CAM convention)
  blur       - Gaussian smoothing (sigma = 4 px at 256x256, scaled with resolution)
  blur_norm  - smoothing, then per-image normalisation
on the full test set and on two disjoint halves (fold A / B) used for cross-validated selection:
MVTec halves are stratified by defect type, UCSD halves are split by clip.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vaex import data
from vaex.attention import FeatureTap, anomaly_attention
from vaex.common import save_json
from vaex.models import Ped1VAE, Ped1VAESupp, ResNet18VAE, ResNet18VAESupp

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["mvtec", "ucsd"], required=True)
p.add_argument("--tag", default="scratch_bce", help="MVTec checkpoint folder under outputs/mvtec/")
p.add_argument("--classes", nargs="+", default=data.MVTEC_CLASSES)
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="UCSD checkpoints vae_seed{s}.pt")
p.add_argument("--ucsd_arch", choices=["bn", "supp"], default="bn")
args = p.parse_args()
dev = "cuda"
POSTS = ["raw", "norm", "blur", "blur_norm"]
SCORES = ["mu", "ndd", "kl", "mahal"]
MODES = ["relu", "abs", "absup", "chabs"]


def gaussian_kernel(sigma):
    r = int(3 * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return k / k.sum(), r


def postprocess(m, post, sigma):
    """m: B x H x W on GPU."""
    if post in ("blur", "blur_norm"):
        k, r = gaussian_kernel(sigma)
        x = F.pad(m[:, None], (r, r, r, r), mode="reflect")
        x = F.conv2d(x, k.view(1, 1, 1, -1))
        m = F.conv2d(x, k.view(1, 1, -1, 1))[:, 0]
    if post in ("norm", "blur_norm"):
        lo = m.flatten(1).min(1).values[:, None, None]
        hi = m.flatten(1).max(1).values[:, None, None]
        m = (m - lo) / (hi - lo + 1e-12)
    return m


def auroc_iou(s, y):
    """Pixel ROC AUC + best IoU over all thresholds, on GPU (ties handled like sklearn)."""
    order = torch.argsort(s, descending=True)
    s, y = s[order], y[order]
    tps = torch.cumsum(y.long(), 0)
    fps = torch.cumsum((~y).long(), 0)
    last = torch.ones_like(y)
    last[:-1] = s[1:] != s[:-1]
    tps, fps = tps[last].double(), fps[last].double()
    P, N = tps[-1], fps[-1]
    z = torch.zeros(1, dtype=torch.float64, device=dev)
    auc = torch.trapz(torch.cat([z, tps / P]), torch.cat([z, fps / N]))
    iou = (tps / (fps + P)).max()
    return float(auc), float(iou)


def evaluate(maps, masks, folds, sigma, bs=64):
    """maps: N x H x W (CPU), masks: N x H x W bool. Returns {post: {subset: [auc, iou]}}."""
    out = {}
    y = masks.to(dev)
    for post in POSTS:
        pm = torch.cat([postprocess(maps[i:i + bs].to(dev), post, sigma) for i in range(0, len(maps), bs)])
        out[post] = {"all": auroc_iou(pm.flatten(), y.flatten())}
        for name, idx in folds.items():
            idx = torch.as_tensor(idx, device=dev)
            out[post][name] = auroc_iou(pm[idx].flatten(), y[idx].flatten())
        del pm
    return out


def attention_maps(model, layers, x_te, x_tr, uint8, skip=()):
    """All attention configs + reconstruction residual for one model (configs in `skip` are not recomputed)."""
    to_f = (lambda x: x.to(dev).float().div(255)) if uint8 else (lambda x: x.to(dev))
    bs = 16 if uint8 else 100
    batches = lambda x: (to_f(x[i:i + bs]) for i in range(0, len(x), bs))
    maps = {}
    with torch.no_grad():
        if "recon" not in skip:
            maps["recon"] = torch.cat([(x - model.decode(model.encode(x)[0])).abs().mean(1).cpu() for x in batches(x_te)])
        enc = [model.encode(x) for x in batches(x_tr)]
        mu_all = torch.cat([e[0] for e in enc])
        stats = (mu_all.mean(0), mu_all.var(0) + torch.cat([e[1] for e in enc]).exp().mean(0))
    torch.manual_seed(0)
    for layer in layers:
        tap = FeatureTap(model.attn_layers[layer])
        for score in SCORES:
            for mode in MODES:
                if f"{layer},{score},{mode}" in skip:
                    continue
                maps[f"{layer},{score},{mode}"] = torch.cat([
                    anomaly_attention(model, tap, x, mode=mode, score=score, normal_stats=stats)[0].detach().cpu()
                    for x in batches(x_te)])
        tap.remove()
    return maps


if args.dataset == "mvtec":
    out_dir = f"outputs/reeval/mvtec_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    for cls in args.classes:
        prev = json.load(open(f"{out_dir}/{cls}.json")) if os.path.exists(f"{out_dir}/{cls}.json") else {}
        if prev and all(f"{l},{s},{m}" in prev for l in ["layer1", "layer2", "layer3", "layer4"]
                        for s in SCORES for m in MODES):
            print(f"[{cls}] done, skipping"); continue
        x_tr, _, _ = data.mvtec(cls, train=True)
        x_te, m_te, types = data.mvtec(cls, train=False)
        rng = np.random.default_rng(0)
        fold_a = []
        for t in sorted(set(types)):                       # stratified 50/50 split by defect type
            idx = rng.permutation([i for i, tt in enumerate(types) if tt == t])
            fold_a += idx[::2].tolist()
        fold_b = sorted(set(range(len(types))) - set(fold_a))
        model = (ResNet18VAESupp if args.tag.startswith("supp_") else ResNet18VAE)(32)
        model.load_state_dict(torch.load(f"outputs/mvtec/{args.tag}/{cls}.pt"))
        model.to(dev).eval()
        maps = attention_maps(model, ["layer1", "layer2", "layer3", "layer4"], x_te, x_tr, uint8=True, skip=prev)
        res = {**prev, **{k: evaluate(v, m_te, {"A": sorted(fold_a), "B": fold_b}, sigma=4.0) for k, v in maps.items()}}
        save_json(res, f"{out_dir}/{cls}.json")
        print(f"[{cls}] SZU-style layer2 raw {res['layer2,mu,chabs']['raw']['all'][0]:.3f} | "
              f"paper setting layer2 raw {res['layer2,mu,relu']['raw']['all'][0]:.3f} "
              f"blur_norm {res['layer2,mu,relu']['blur_norm']['all'][0]:.3f} | "
              f"recon raw {res['recon']['raw']['all'][0]:.3f} blur {res['recon']['blur']['all'][0]:.3f}", flush=True)
else:
    out_dir = "outputs/reeval/ucsd" + ("_supp" if args.ucsd_arch == "supp" else "")
    os.makedirs(out_dir, exist_ok=True)
    x_tr, _ = data.ucsd_ped1(train=True)
    x_te, m_te = data.ucsd_ped1(train=False)
    clip = np.repeat(np.arange(10), 200)
    folds = {"A": np.nonzero(clip < 5)[0].tolist(), "B": np.nonzero(clip >= 5)[0].tolist()}
    for seed in args.seeds:
        model = Ped1VAESupp(32) if args.ucsd_arch == "supp" else Ped1VAE(32)
        model.load_state_dict(torch.load(f"outputs/ucsd/vae_{'supp_' if args.ucsd_arch == 'supp' else ''}seed{seed}.pt"))
        model.to(dev).eval()
        f = f"{out_dir}/seed{seed}.json"
        prev = json.load(open(f)) if os.path.exists(f) else {}
        maps = attention_maps(model, ["conv1", "conv2", "conv3"], x_te, x_tr, uint8=False, skip=prev)
        res = {**prev, **{k: evaluate(v, m_te, folds, sigma=4.0 * 100 / 256) for k, v in maps.items()}}
        save_json(res, f"{out_dir}/seed{seed}.json")
        print(f"[ucsd seed {seed}] SZU-style conv2 raw {res['conv2,mu,chabs']['raw']['all'][0]:.3f} | paper conv2 raw {res['conv2,mu,relu']['raw']['all'][0]:.3f} "
              f"blur_norm {res['conv2,mu,relu']['blur_norm']['all'][0]:.3f} | recon raw "
              f"{res['recon']['raw']['all'][0]:.3f}", flush=True)
