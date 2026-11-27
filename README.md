# Reproduction: *Towards Visually Explaining Variational Autoencoders* (Liu et al., CVPR 2020)

A single modern-PyTorch implementation of all four experiments in the paper. Two existing codebases served as reference:

* `expVAE/`: the official code from the first author (liuem607/expVAE). It covers only MNIST and targets PyTorch 1.0.
* `szu_repro/`: a third-party reproduction (SZU-AdvTech-2023). It covers MNIST, UCSD, MVTec and AD-FactorVAE and targets PyTorch 1.7.

Neither of them is included in this repository. To reproduce the official-code check and the SZU reference run, clone them:
```bash
git clone https://github.com/liuem607/expVAE.git
git clone https://github.com/SZU-AdvTech-2023/326-Towards-Visually-Explaining-Variational-Autoencoders.git szu_repro
cp -r expVAE/code official_run   # then apply the two patches listed in official_run/PATCHES.md
```

**Illustrated write-up:** `docs/paper_explainer.html` (open it in a browser), which covers the method, all results and the critique.

## Environment

```bash
mamba create -n vae-attn python=3.11
# /tmp is on a small partition on this machine -> give pip a temp dir on /home
TMPDIR=$PWD/.piptmp pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-learn matplotlib tqdm pillow h5py opencv-python-headless
```
Exact versions used are pinned in `requirements.txt`. Install torch/torchvision from the cu128 index shown above.
Tested with torch 2.11.0+cu128 on an RTX 5070 Ti Laptop GPU. Blackwell GPUs need CUDA 12.8 or newer builds.

## Data (`data/`)
| dataset | source |
|---|---|
| MNIST | torchvision (downloaded automatically) |
| dSprites | github.com/deepmind/dsprites-dataset |
| UCSD Ped1 | svcl.ucsd.edu/projects/anomaly (UCSD_Anomaly_Dataset.v1p2) |
| MVTec-AD | mydrive.ch mirror used by anomalib (sha256 verified), CC BY-NC-SA 4.0 |

Datasets are not included (about 6 GB). Expected layout: `data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`,
`data/UCSD_Anomaly_Dataset.v1p2/`, `data/mvtec/<class>/`, and `data/mnist/`, which torchvision creates on first run.
Model checkpoints are not included either; every script retrains when a checkpoint is missing.

## Code
```
vaex/attention.py   Eq. 2-3 Grad-CAM from latent units (autograd.grad, differentiable), anomaly attention (Sec. 3.3, Eq. 4), L_AD (Eq. 5)
vaex/models.py      MNIST ConvVAE (official arch), Ped1 ConvVAE, ResNet18-VAE (MVTec), FactorVAE net + discriminator (dSprites)
vaex/data.py        in-memory loaders + GPU mirror/rotation augmentation
vaex/common.py      VAE training loop, pixel AUROC / best-IoU
scripts/mnist.py            Exp. 1  Fig. 1 / Fig. 4
scripts/mnist_ablation.py   Fig. 4 variants: target layer x (ReLU vs abs)
scripts/ucsd.py             Exp. 2  Table 1 / Fig. 5
scripts/mvtec.py            Exp. 3  Table 2 / Fig. 6     (--pretrained 0|1)
scripts/dsprites.py         Exp. 4  Fig. 8 / Fig. 9      (--model betavae|factorvae|adfactorvae)
scripts/run_dsprites_grid.sh  full Fig. 8 grid (5 configs x 3 seeds)
scripts/report.py           aggregates everything into outputs/report/results.md + figures
scripts/reeval.py           re-evaluates saved VAEs: post-processing x layer x score x aggregation, 2-fold CV
scripts/reeval_summary.py   writes outputs/report/reeval.md (post-processing, oracle, cross-validated tables)
scripts/official_check.py   our attention vs the official expVAE code on the same weights (needs official_run/)
scripts/run_dsprites_supp.sh  dSprites re-run with the supplement architecture (z=32)
scripts/start_job.sh / stop_all.sh / watch.sh   run long jobs in their own process groups and watch their logs
explainer/                  template + build script for docs/paper_explainer.html
```

## Results summary
Full tables are in `outputs/report/results.md`, figures in `outputs/report/`, and the illustrated write-up is `../paper_explainer.html` (rebuild with `explainer/build.py`).

| experiment | paper | reproduced |
|---|---|---|
| UCSD Ped1 pixel AUROC | attention 0.89-0.92 > recon. 0.86 | attention (paper setting) 0.35-0.59, recon. 0.90 |
| MVTec-AD mean pixel AUROC / IoU | 0.86 / 0.20 | paper-setting attention 0.550 / 0.046 (scratch), 0.558 / 0.053 (ImageNet init). Recon. 0.794 / 0.806. Attention beats recon. in 0-1 of 15 classes |
| Robustness re-evaluation (`scripts/reeval.py`, `outputs/report/reeval.md`); 2-fold CV over layer and post-processing | n/a | Attention, by aggregation (UCSD / MVTec scratch / MVTec ImageNet): paper Eq. 2 is 0.659 / 0.606 / 0.647; official code (verified identical, upsample then abs) is 0.631 / 0.686 / 0.639; SZU channel-wise abs is 0.843 / 0.769 / 0.719. Recon. residual with blur is 0.955 / 0.862 / 0.883 |
| dSprites beta-VAE (beta=4) | 0.73 | 0.729 +- 0.074 |
| dSprites FactorVAE (gamma=40) | 0.82 | 0.763 +- 0.037 |
| dSprites AD-FactorVAE, as written (ReLU, lambda=10) | 0.90 | 0.665 +- 0.149. Attention overlap is the same as FactorVAE |
| dSprites AD-FactorVAE, fixed (abs + skip empty, lambda=10) | n/a | 0.828 +- 0.026, recon. 34. Beats the same-seed FactorVAE on 3/3 seeds |

## Re-run with the paper's supplementary-material settings
The supplement (CVPR open-access page) specifies settings the first run did not follow. All three were re-run with them
(`--arch supp`; models `Ped1VAESupp`, `ResNet18VAESupp`, `FactorVAENetSupp`):

| experiment | supplement setting | original run -> supplement run | paper |
|---|---|---|---|
| UCSD (3 seeds, CV) | no BatchNorm, lr 1e-4, batch 32 | official-code attention 0.631 -> **0.868**; SZU-style 0.843 -> **0.892**; Eq. 2 0.659 -> 0.695; recon. 0.90 / 0.95 (blur) unchanged | 0.89-0.92 (recon. 0.86) |
| MVTec (15 classes, CV) | flattened ResNet18 features -> 1024 -> 32, rot +-30, lr 1e-4, batch 8 | Eq. 2 0.606 -> 0.609; official 0.686 -> 0.672; best variant 0.812 -> 0.730; recon. (blur) 0.862 -> 0.878 | 0.86 |
| dSprites (3 seeds) | Table 4 architecture, z=32 | FactorVAE 0.763 -> **0.795 +- 0.011**; AD as written 0.665 -> 0.762 +- 0.059; AD fixed 0.828 -> 0.738 +- 0.096 | 0.82 / 0.90 |

Verdict: the low UCSD attention scores were largely caused by our BatchNorm model. With the supplement model, the official-code attention gets close to the paper's numbers, but it still does not beat the reconstruction residual. MVTec and the AD-FactorVAE gain stay unreproduced under the authors' own settings. Our fixed L_AD separates attention, with overlap 0.08-0.09 when converged, but it improves disentanglement only in the z=10 setting.

## Official code check
`official_run/` holds the official expVAE MNIST code, run with two compatibility patches (`official_run/PATCHES.md`). On the same checkpoint and inputs, `scripts/official_check.py` shows our attention equals the official one: per-image correlation 1.0000, provided the signed map is upsampled before `abs` (`mode="absup"`). The official pretrained weights link is dead (Google Drive 404).

## Choices where the paper is underspecified, and deviations

| item | paper | here |
|---|---|---|
| attention aggregation | ReLU(sum_k alpha_k A_k) (Eq. 2) | both ReLU (paper) and abs. The official code uses `abs`, plus a global grad L2-normalisation that only rescales whole batches and is dropped here. |
| anomaly score | sum of the mean-vector elements (results), or the normal difference distribution (Eq. 4) | both are implemented (`mu`, `ndd`). Two extra diagnostic scores that are **not** in the paper are also evaluated: `kl` = KL(q(z\|x)\|\|N(0,I)) and `mahal` = squared z-score of mu against the normal-training embeddings |
| target layer | not given for MNIST/MVTec | every layer is evaluated. Tables report one fixed layer for all classes, with no per-class picking |
| VAE recon. loss | "typically MSE" | BCE (sum), as in both reference codes. MSE-sum led to posterior collapse on UCSD (KL ~ 4.7 nats) |
| MVTec encoder | "ResNet18" | trained from scratch, and separately ImageNet-pretrained |
| MVTec training | not given | 4000 iters, batch 32, Adam 3e-4 with cosine decay, hflip + rotation of ±15° |
| UCSD evaluation | 36 test clips | only the 10 clips that have pixel GT. SZU counted clips without GT as all-normal, although they contain anomalies |
| dSprites setup | "same as FactorVAE" | Kim & Mnih: z=10, batch 64, 3e5 iters, metric with L=100, 800 train + 800 eval votes over 5 factors (the constant colour factor is excluded; SZU included it) |
| FactorVAE D-step | n/a | D trained on `z.detach()`. In the SZU code the D loss also back-propagated into the VAE |
| L_AD pairs | "two attention maps" | 2 distinct random *active* dims (batch KL > 0.1) per iteration, `create_graph=True` so gradients flow through alpha too |
| L_AD degeneracy | n/a | with ReLU maps, an all-zero map makes Eq. 5 = 0 without any separation, and training finds this shortcut. The `--ad_mode abs` variant uses \|map\| and skips empty-map samples. Both are run |

### FactorVAE early-training check (dSprites, gamma=40)
The first 10-20k iterations are very volatile: batch KL spikes above 150 nats and reconstruction error stays above 110. Four single-seed 30k-iteration diagnostics changed one setting each: TF32 off, PyTorch-default ConvTranspose init, z=32. None of them removed the volatility (`outputs/dsprites_diag/`). The SZU reference code, run unchanged in this environment (`outputs/szu_ref/`), shows the same transient: KL about 87 and reconstruction about 243 at 2k iterations, settling by about 8k. Our implementation settles later, at about 20-30k. By 30k both sit at reconstruction about 95-100 and KL about 15, so the full runs use the Kim & Mnih setting (z=10, 3e5 iters) with 3 seeds. Note that SZU's loader holds dSprites as float32, about 12 GB per process. Two parallel copies triggered the OOM killer.
