"""Gradient-based VAE attention (Liu et al., CVPR 2020, Sec. 3.2-3.4).

Everything is computed with torch.autograd.grad on a captured feature map A, so the
same code serves three purposes:
  * element-wise attention M_i for each latent dim z_i           (Eq. 2-3, Fig. 1/2)
  * anomaly attention from a score s = sum_i mu_i                (Sec. 3.3, Fig. 3)
  * differentiable attention for the attention-disentanglement loss (Eq. 5, create_graph=True)

Batches are handled by summing the score over the batch: with the encoder in eval
mode (or without BatchNorm) samples are independent, so d(sum_b s_b)/dA_b = ds_b/dA_b.
"""
import torch
import torch.nn.functional as F


class FeatureTap:
    """Keeps the forward output of `module` (still attached to the graph)."""

    def __init__(self, module):
        self.out = None
        self._h = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.out = output

    def remove(self):
        self._h.remove()


def cam_from_score(score, A, mode="relu", out_size=None, create_graph=False):
    """Eq. 2-3.  alpha_k = GAP(ds/dA_k),  M = ReLU(sum_k alpha_k A_k).

    mode: "relu" -> paper (Eq. 2); "abs" -> what the official MNIST code does
          (|sum_k alpha_k A_k|); "chabs" -> what the SZU third-party code does
          (mean_k |alpha_k A_k|, abs per channel before aggregating); "absup" -> exactly the official
          order of operations (upsample the signed map first, then abs); "none" -> raw signed map.
    Returns B x H x W.
    """
    g, = torch.autograd.grad(score, A, retain_graph=True, create_graph=create_graph)
    alpha = g.mean(dim=(2, 3), keepdim=True)          # GAP over the h x w map  (Eq. 3)
    if mode == "chabs":
        cam = (alpha * A).abs().mean(dim=1, keepdim=True)
        if out_size is not None:
            cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        return cam[:, 0]
    cam = (alpha * A).sum(dim=1, keepdim=True)        # linear combination      (Eq. 2)
    if mode == "absup":
        if out_size is not None:
            cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        return cam.abs()[:, 0]
    if mode == "relu":
        cam = F.relu(cam)
    elif mode == "abs":
        cam = cam.abs()
    elif mode != "none":
        raise ValueError(mode)
    if out_size is not None:
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
    return cam[:, 0]


def elementwise_attention(model, tap, x, dims=None, use_mean=False, mode="relu", out_size=None,
                          create_graph=False):
    """M_1..M_D (Fig. 2).  Backprop each z_i (reparameterised sample, or mu_i if use_mean).

    Returns D' x B x H x W for the requested dims, plus (mu, logvar, z).
    """
    mu, logvar = model.encode(x)
    A = tap.out
    z = mu if use_mean else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    dims = range(z.shape[1]) if dims is None else dims
    out_size = out_size or x.shape[-2:]
    maps = [cam_from_score(z[:, i].sum(), A, mode, out_size, create_graph) for i in dims]
    return torch.stack(maps), (mu, logvar, z)


def anomaly_attention(model, tap, x, mode="relu", score="mu", normal_stats=None, out_size=None):
    """Anomaly attention map M (Sec. 3.3).

    score="mu":  s = sum_i mu_i  (the variant the paper uses for its results)
    score="ndd": z ~ normal difference distribution N(mu_x - mu_y, sigma_x^2 + sigma_y^2)
                 (Eq. 4) with (mu_x, sigma_x) the embedding statistics of the normal training
                 set, then s = sum_i z_i.
    """
    mu, logvar = model.encode(x)
    A = tap.out
    if score == "mu":
        s = mu.sum()
    elif score == "ndd":
        mu_x, var_x = normal_stats
        std = torch.sqrt(var_x + logvar.exp())
        s = ((mu_x - mu) + torch.randn_like(mu) * std).sum()
    # --- not in the paper; diagnostic variants that measure *deviation* from the normal latent ---
    elif score == "kl":       # KL(q(z|y) || N(0, I)): 'latent does not conform to the standard Gaussian'
        s = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum()
    elif score == "mahal":    # squared z-score of mu w.r.t. the normal-training embedding statistics
        mu_x, var_x = normal_stats
        s = ((mu - mu_x).pow(2) / var_x).sum()
    else:
        raise ValueError(score)
    return cam_from_score(s, A, mode, out_size or x.shape[-2:]), mu


def attention_disentanglement_loss(M1, M2, eps=1e-8, skip_empty=False):
    """Eq. 5, per sample, averaged over the batch:  2 * sum min(A1, A2) / sum(A1 + A2).

    skip_empty: average only over samples where *both* maps are non-zero.  With ReLU maps a
    dimension whose alpha*A is negative everywhere gets an all-zero map, which makes Eq. 5 equal 0
    without any separation (a degenerate optimum the network quickly finds).
    """
    s1, s2 = M1.flatten(1).sum(1), M2.flatten(1).sum(1)
    ratio = 2 * torch.minimum(M1, M2).flatten(1).sum(1) / (s1 + s2 + eps)
    if not skip_empty:
        return ratio.mean()
    valid = (s1 > eps) & (s2 > eps)
    return ratio[valid].mean() if valid.any() else ratio.sum() * 0
