"""Shared training / evaluation utilities for the one-class (anomaly) VAEs."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vae_loss(x_hat, x, mu, logvar, recon="bce"):
    """Eq. 1: L = L_r(x, x_hat) + KL(q(z|x) || N(0, I)), summed over pixels/dims, averaged over batch."""
    B = x.shape[0]
    if recon == "bce":
        rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / B
    elif recon == "mse":
        rec = F.mse_loss(x_hat, x, reduction="sum") / B
    else:
        raise ValueError(recon)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return rec + kl, rec, kl


def train_vae(model, x_train, iters, batch_size, lr, device, recon="bce", augment=None,
              log_every=500, log=print, x_is_uint8=False, cosine=True):
    """Plain VAE training on an in-memory tensor (kept on the GPU)."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05) if cosine
             else torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0))
    xg = x_train.to(device)
    N, t0, hist = len(xg), time.time(), []
    for it in range(1, iters + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        x = xg[idx].float().div(255) if x_is_uint8 else xg[idx]
        if augment is not None:
            x = augment(x).clamp(0, 1)
        x_hat, mu, logvar = model(x)
        loss, rec, kl = vae_loss(x_hat, x, mu, logvar, recon)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters:
            hist.append(dict(it=it, loss=loss.item(), rec=rec.item(), kl=kl.item()))
            log(f"  it {it:6d}/{iters}  loss {loss.item():10.2f}  rec {rec.item():10.2f}  "
                f"kl {kl.item():7.2f}  ({time.time() - t0:.0f}s)")
    model.eval()
    return hist


def pixel_auroc_iou(scores, masks):
    """Pixel-level ROC AUC over the whole test set + best IoU over all ROC thresholds.

    IoU(t) = TP / (TP + FP + FN) = TPR*P / (FPR*N + P)   (paper: 'best IOU ... based on our ROC curve').
    """
    s = np.asarray(scores, dtype=np.float32).ravel()
    m = np.asarray(masks, dtype=bool).ravel()
    auc = roc_auc_score(m, s)
    fpr, tpr, _ = roc_curve(m, s, drop_intermediate=True)
    P, N = m.sum(), (~m).sum()
    iou = (tpr * P / (fpr * N + P)).max()
    return float(auc), float(iou)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-12)
