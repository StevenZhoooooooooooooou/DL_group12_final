"""Exp. 4 - dSprites: AD-FactorVAE vs FactorVAE vs beta-VAE (paper Fig. 8 and Fig. 9).

Setup follows Kim & Mnih (2018), which the paper uses as baseline and metric:
z=10, batch 64, Adam(1e-4, 0.9, 0.999) for the VAE, Adam(1e-4, 0.5, 0.9) for the discriminator,
Bernoulli decoder, 3e5 iterations, FactorVAE disentanglement metric (L=100, 800 train / 800 eval votes).

AD-FactorVAE adds lambda * L_AD (Eq. 5-6): per iteration two distinct *active* latent dims
(batch KL > 0.1) are picked at random, their attention maps are computed at `--layer` with
create_graph=True (so L_AD trains the encoder through both A and alpha), and L_AD is averaged
over the batch.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vaex.attention import FeatureTap, cam_from_score, attention_disentanglement_loss
from vaex.common import seed_all, save_json
from vaex.data import DSprites
from vaex.models import FactorVAENet, FactorVAENetSupp, Discriminator, reparameterize

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["betavae", "factorvae", "adfactorvae"], required=True)
p.add_argument("--beta", type=float, default=4.0)
p.add_argument("--gamma", type=float, default=40.0)
p.add_argument("--lamb", type=float, default=10.0)
p.add_argument("--layer", default="conv2")
p.add_argument("--ad_mode", choices=["relu", "abs"], default="relu",
               help="relu = paper (Eq. 2); abs = |map| and skip samples with an empty map")
p.add_argument("--iters", type=int, default=300000)
p.add_argument("--eval_every", type=int, default=50000)
p.add_argument("--z_dim", type=int, default=10)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="outputs/dsprites")
p.add_argument("--tf32", type=int, default=1)
p.add_argument("--deconv_init", choices=["normal", "default"], default="normal")
p.add_argument("--tag", default="")
p.add_argument("--arch", choices=["kim", "supp"], default="kim", help="supp = supplement Table 4 (z=32)")
args = p.parse_args()
if args.arch == "supp":
    args.z_dim, args.tag = 32, args.tag + "_supp"
dev = "cuda"
name = {"betavae": f"betavae_b{args.beta:g}", "factorvae": f"factorvae_g{args.gamma:g}",
        "adfactorvae": f"adfactorvae_g{args.gamma:g}_l{args.lamb:g}_{args.layer}_{args.ad_mode}"}[args.model]
run = f"{name}{args.tag}_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
seed_all(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(args.tf32)

ds = DSprites()
imgs = ds.imgs.pin_memory()                    # 3 GB uint8 kept on the CPU so several runs can share the GPU
batch = lambda idx: imgs[idx.cpu()].to(dev, non_blocking=True).unsqueeze(1).float()


def kl_per_dim(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


@torch.no_grad()
def encode_mu(model, idx, bs=4096):
    return torch.cat([model.encode(batch(idx[i:i + bs]))[0] for i in range(0, len(idx), bs)])


@torch.no_grad()
def factorvae_metric(model, L=100, n_train=800, n_eval=800, seed=0):
    """Kim & Mnih (2018), Sec. 4: majority-vote classifier on argmin normalised variance."""
    rng = np.random.default_rng(seed)
    sub = torch.from_numpy(rng.choice(ds.N, 20000, replace=False)).to(dev)
    mus, lvs = zip(*[model.encode(batch(sub[i:i + 4096])) for i in range(0, len(sub), 4096)])
    mu, lv = torch.cat(mus), torch.cat(lvs)
    scale = mu.std(0)
    active = (kl_per_dim(mu, lv).mean(0) > 1e-2).nonzero()[:, 0]     # prune collapsed dims
    if len(active) == 0:
        return 0.0, 0
    K = len(ds.n_values)

    def votes(n):
        out = []
        for _ in range(n):
            k = rng.integers(K)
            v = rng.integers(ds.n_values[k])
            pool = np.nonzero(ds.classes[:, k] == v)[0]
            idx = torch.from_numpy(rng.choice(pool, L)).to(dev)
            z = encode_mu(model, idx)[:, active] / scale[active]
            out.append((int(z.var(0).argmin()), k))
        return out

    V = np.zeros((len(active), K), dtype=int)
    for d, k in votes(n_train):
        V[d, k] += 1
    clf = V.argmax(1)
    acc = np.mean([clf[d] == k for d, k in votes(n_eval)])
    return float(acc), int(len(active))


@torch.no_grad()
def recon_error(model, n=10000, seed=0):
    idx = torch.from_numpy(np.random.default_rng(seed).choice(ds.N, n, replace=False)).to(dev)
    tot = 0.0
    for i in range(0, n, 2000):
        x = batch(idx[i:i + 2000])
        logits = model.decode(model.encode(x)[0])
        tot += F.binary_cross_entropy_with_logits(logits, x, reduction="sum").item()
    return tot / n


vae = (FactorVAENetSupp(args.z_dim) if args.arch == "supp"
       else FactorVAENet(args.z_dim, init_deconv=args.deconv_init == "normal")).to(dev)
opt = torch.optim.Adam(vae.parameters(), lr=1e-4, betas=(0.9, 0.999))
if args.model != "betavae":
    D = Discriminator(args.z_dim).to(dev)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
tap = FeatureTap(vae.attn_layers[args.layer]) if args.model == "adfactorvae" else None

B = 64
log = {"args": vars(args), "evals": [], "train": []}
t0 = time.time()
acc_ad = acc_tc = acc_rec = acc_kl = max_kl = 0.0
for it in range(1, args.iters + 1):
    x = batch(torch.randint(0, ds.N, (B,), device=dev))
    mu, logvar = vae.encode(x)
    z = reparameterize(mu, logvar)
    logits = vae.decode(z)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / B
    kld_d = kl_per_dim(mu, logvar).mean(0)
    kl = kld_d.sum()

    if args.model == "betavae":
        loss = rec + args.beta * kl
    else:
        d_z = D(z)
        tc = (d_z[:, 0] - d_z[:, 1]).mean()          # density-ratio estimate of TC(z)
        loss = rec + kl + args.gamma * tc
        if args.model == "adfactorvae":
            act = (kld_d.detach() > 0.1).nonzero()[:, 0]
            pool = act if len(act) >= 2 else kld_d.detach().topk(2).indices
            i, j = pool[torch.randperm(len(pool), device=dev)[:2]].tolist()
            # attention maps of the two dims from the *same* forward pass (tap.out is A of this batch)
            A = tap.out
            M1 = cam_from_score(z[:, i].sum(), A, args.ad_mode, None, create_graph=True)
            M2 = cam_from_score(z[:, j].sum(), A, args.ad_mode, None, create_graph=True)
            l_ad = attention_disentanglement_loss(M1, M2, skip_empty=args.ad_mode == "abs")
            loss = loss + args.lamb * l_ad
            acc_ad += l_ad.item()
        acc_tc += tc.item()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    acc_rec += rec.item(); acc_kl += kl.item(); max_kl = max(max_kl, kl.item())

    if args.model != "betavae":
        # discriminator: q(z) (label 0) vs product of marginals via dimension-wise permutation (label 1)
        x2 = batch(torch.randint(0, ds.N, (B,), device=dev))
        with torch.no_grad():
            mu2, lv2 = vae.encode(x2)
            z2 = reparameterize(mu2, lv2)
            z_perm = torch.stack([z2[torch.randperm(B, device=dev), d] for d in range(args.z_dim)], 1)
        zeros = torch.zeros(B, dtype=torch.long, device=dev)
        d_loss = 0.5 * (F.cross_entropy(D(z.detach()), zeros) + F.cross_entropy(D(z_perm), 1 - zeros))
        opt_D.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_D.step()

    if it % 5000 == 0:
        n = 5000
        entry = dict(it=it, rec=acc_rec / n, kl=acc_kl / n, kl_max=max_kl, tc=acc_tc / n, l_ad=acc_ad / n,
                     active=int((kld_d > 0.1).sum()))
        acc_ad = acc_tc = acc_rec = acc_kl = max_kl = 0.0
        log["train"].append(entry)
        print(f"[{run}] it {it} rec {entry['rec']:.1f} kl {entry['kl']:.2f} (max {entry['kl_max']:.0f}) tc {entry['tc']:.3f} "
              f"L_AD {entry['l_ad']:.3f} active {entry['active']} ({time.time() - t0:.0f}s)", flush=True)
    if it % args.eval_every == 0 or it == args.iters:
        vae.eval()
        score, n_act = factorvae_metric(vae)
        r = recon_error(vae)
        vae.train()
        log["evals"].append(dict(it=it, disentanglement=score, recon=r, active_dims=n_act))
        print(f"[{run}] EVAL it {it}: disentanglement {score:.3f}  recon {r:.2f}  active {n_act}", flush=True)
        save_json(log, f"{args.out}/{run}.json")

torch.save(vae.state_dict(), f"{args.out}/{run}.pt")
print(f"[{run}] finished in {time.time() - t0:.0f}s")
