# Reproduction results: Towards Visually Explaining Variational Autoencoders (CVPR 2020)

## Table 1 - UCSD Ped1, pixel-level AUROC (10 test clips with pixel GT)

| method | paper | ours (score = sum mu, ReLU: paper setting) | ours (abs, official code) | best non-paper score (KL / Mahalanobis) |
|---|---|---|---|---|
| Vanilla-VAE (recon. diff.) | 0.86 | 0.902 | | |
| Ours(Conv1) | 0.89 | 0.352 | 0.363 | 0.595 `Ours(conv1,kl,abs)` |
| Ours(Conv2) | 0.92 | 0.360 | 0.420 | 0.868 `Ours(conv2,kl,abs)` |
| Ours(Conv3) | 0.91 | 0.590 | 0.574 | 0.846 `Ours(conv3,kl,relu)` |

## Table 2 - MVTec-AD pixel-level ROC AUC / best IoU  (`pt_bce`, 15/15 classes)

Paper-setting column = score `sum mu`, ReLU (Eq. 2), single layer for all classes (`layer1,mu,relu`, chosen by mean AUROC).  Best-variant column = best single configuration over all layers/scores/modes (`layer4,mahal,abs`, not in the paper).

| class | paper AE(L2) | paper ours | our Vanilla-VAE recon | our attention, paper setting | our attention, best variant |
|---|---|---|---|---|---|
| carpet | 0.59 | 0.78 / 0.10 | 0.565 / 0.024 | 0.643 / 0.094 | 0.544 / 0.023 |
| grid | 0.90 | 0.73 / 0.02 | 0.728 / 0.095 | 0.620 / 0.036 | 0.558 / 0.013 |
| leather | 0.75 | 0.95 / 0.24 | 0.837 / 0.167 | 0.596 / 0.050 | 0.404 / 0.007 |
| tile | 0.51 | 0.80 / 0.23 | 0.528 / 0.076 | 0.456 / 0.070 | 0.761 / 0.176 |
| wood | 0.73 | 0.77 / 0.14 | 0.680 / 0.116 | 0.524 / 0.051 | 0.680 / 0.066 |
| bottle | 0.86 | 0.87 / 0.27 | 0.883 / 0.285 | 0.628 / 0.082 | 0.603 / 0.073 |
| cable | 0.86 | 0.90 / 0.18 | 0.840 / 0.155 | 0.443 / 0.029 | 0.617 / 0.038 |
| capsule | 0.88 | 0.74 / 0.11 | 0.887 / 0.164 | 0.528 / 0.025 | 0.619 / 0.017 |
| hazelnut | 0.95 | 0.98 / 0.44 | 0.948 / 0.387 | 0.356 / 0.027 | 0.863 / 0.142 |
| metal_nut | 0.86 | 0.94 / 0.49 | 0.870 / 0.321 | 0.576 / 0.152 | 0.583 / 0.140 |
| pill | 0.85 | 0.83 / 0.18 | 0.872 / 0.196 | 0.670 / 0.054 | 0.680 / 0.064 |
| screw | 0.96 | 0.97 / 0.17 | 0.937 / 0.115 | 0.787 / 0.013 | 0.499 / 0.003 |
| toothbrush | 0.93 | 0.94 / 0.14 | 0.946 / 0.237 | 0.534 / 0.036 | 0.604 / 0.023 |
| transistor | 0.86 | 0.93 / 0.30 | 0.810 / 0.233 | 0.548 / 0.054 | 0.743 / 0.130 |
| zipper | 0.77 | 0.78 / 0.06 | 0.756 / 0.073 | 0.455 / 0.021 | 0.476 / 0.021 |
| **mean** | 0.82 | 0.86 / 0.20 | 0.806 / 0.176 | 0.558 / 0.053 | 0.616 / 0.062 |

Mean AUROC over classes for every attention configuration:

| layer | mu,relu | mu,abs | ndd,relu | ndd,abs | kl,relu | kl,abs | mahal,relu | mahal,abs |
|---|---|---|---|---|---|---|---|---|
| layer1 | 0.558 | 0.588 | 0.515 | 0.587 | 0.577 | 0.561 | 0.564 | 0.581 |
| layer2 | 0.543 | 0.563 | 0.521 | 0.563 | 0.573 | 0.561 | 0.577 | 0.571 |
| layer3 | 0.501 | 0.535 | 0.533 | 0.536 | 0.552 | 0.521 | 0.565 | 0.566 |
| layer4 | 0.488 | 0.514 | 0.537 | 0.513 | 0.574 | 0.574 | 0.612 | 0.616 |

## Table 2 - MVTec-AD pixel-level ROC AUC / best IoU  (`scratch_bce`, 15/15 classes)

Paper-setting column = score `sum mu`, ReLU (Eq. 2), single layer for all classes (`layer4,mu,relu`, chosen by mean AUROC).  Best-variant column = best single configuration over all layers/scores/modes (`layer1,kl,abs`, not in the paper).

| class | paper AE(L2) | paper ours | our Vanilla-VAE recon | our attention, paper setting | our attention, best variant |
|---|---|---|---|---|---|
| carpet | 0.59 | 0.78 / 0.10 | 0.546 / 0.020 | 0.472 / 0.016 | 0.551 / 0.052 |
| grid | 0.90 | 0.73 / 0.02 | 0.607 / 0.021 | 0.606 / 0.012 | 0.568 / 0.010 |
| leather | 0.75 | 0.95 / 0.24 | 0.838 / 0.166 | 0.510 / 0.008 | 0.709 / 0.025 |
| tile | 0.51 | 0.80 / 0.23 | 0.522 / 0.072 | 0.404 / 0.070 | 0.695 / 0.166 |
| wood | 0.73 | 0.77 / 0.14 | 0.677 / 0.110 | 0.545 / 0.045 | 0.695 / 0.146 |
| bottle | 0.86 | 0.87 / 0.27 | 0.881 / 0.279 | 0.399 / 0.058 | 0.458 / 0.058 |
| cable | 0.86 | 0.90 / 0.18 | 0.816 / 0.134 | 0.521 / 0.030 | 0.592 / 0.082 |
| capsule | 0.88 | 0.74 / 0.11 | 0.882 / 0.151 | 0.447 / 0.014 | 0.685 / 0.032 |
| hazelnut | 0.95 | 0.98 / 0.44 | 0.950 / 0.386 | 0.799 / 0.076 | 0.622 / 0.117 |
| metal_nut | 0.86 | 0.94 / 0.49 | 0.864 / 0.308 | 0.670 / 0.174 | 0.775 / 0.243 |
| pill | 0.85 | 0.83 / 0.18 | 0.872 / 0.197 | 0.451 / 0.034 | 0.609 / 0.086 |
| screw | 0.96 | 0.97 / 0.17 | 0.936 / 0.107 | 0.544 / 0.003 | 0.787 / 0.013 |
| toothbrush | 0.93 | 0.94 / 0.14 | 0.949 / 0.237 | 0.656 / 0.035 | 0.791 / 0.077 |
| transistor | 0.86 | 0.93 / 0.30 | 0.824 / 0.236 | 0.656 / 0.076 | 0.734 / 0.178 |
| zipper | 0.77 | 0.78 / 0.06 | 0.749 / 0.071 | 0.563 / 0.031 | 0.746 / 0.061 |
| **mean** | 0.82 | 0.86 / 0.20 | 0.794 / 0.166 | 0.550 / 0.046 | 0.668 / 0.090 |

Mean AUROC over classes for every attention configuration:

| layer | mu,relu | mu,abs | ndd,relu | ndd,abs | kl,relu | kl,abs | mahal,relu | mahal,abs |
|---|---|---|---|---|---|---|---|---|
| layer1 | 0.494 | 0.648 | 0.592 | 0.649 | 0.616 | 0.668 | 0.614 | 0.634 |
| layer2 | 0.520 | 0.652 | 0.549 | 0.650 | 0.600 | 0.656 | 0.612 | 0.651 |
| layer3 | 0.542 | 0.573 | 0.508 | 0.574 | 0.585 | 0.626 | 0.573 | 0.645 |
| layer4 | 0.550 | 0.536 | 0.496 | 0.536 | 0.635 | 0.635 | 0.648 | 0.646 |

## Fig. 8 - dSprites: FactorVAE disentanglement metric vs reconstruction error (3e5 iters)

Paper: AD-FactorVAE ~0.90, FactorVAE (gamma=40) ~0.82, beta-VAE (beta=4) 0.73, recon. error ~40.

| configuration | seeds | disentanglement mean +- std | best seed | recon. error mean |
|---|---|---|---|---|
| adfactorvae_g40_l10_conv2_abs | 3 | 0.828 +- 0.026 | 0.864 | 34.4 |
| adfactorvae_g40_l10_conv2_abs_supp | 3 | 0.738 +- 0.096 | 0.829 | 57.2 |
| adfactorvae_g40_l10_conv2_relu | 3 | 0.665 +- 0.149 | 0.815 | 82.7 |
| adfactorvae_g40_l10_conv2_relu_supp | 3 | 0.762 +- 0.059 | 0.809 | 28.2 |
| adfactorvae_g40_l50_conv2_abs | 3 | 0.747 +- 0.082 | 0.812 | 31.6 |
| betavae_b4 | 3 | 0.729 +- 0.074 | 0.806 | 47.8 |
| factorvae_g40 | 3 | 0.763 +- 0.037 | 0.792 | 62.5 |
| factorvae_g40_supp | 3 | 0.795 +- 0.011 | 0.810 | 34.1 |

Attention overlap between latent dims (Eq. 5 with |.| maps at conv2, averaged over all pairs of active dims and 512 images; lower = more separated attention):

| model | seed | active dims | mean pairwise overlap |
|---|---|---|---|
| factorvae_g40 | 1 | 3 | 0.471 |
| factorvae_g40 | 2 | 5 | 0.483 |
| factorvae_g40 | 3 | 4 | 0.569 |
| adfactorvae_g40_l10_conv2_relu | 1 | 10 | 0.548 |
| adfactorvae_g40_l10_conv2_relu | 2 | 10 | 0.442 |
| adfactorvae_g40_l10_conv2_relu | 3 | 5 | 0.564 |
| adfactorvae_g40_l10_conv2_abs | 1 | 5 | 0.085 |
| adfactorvae_g40_l10_conv2_abs | 2 | 6 | 0.139 |
| adfactorvae_g40_l10_conv2_abs | 3 | 6 | 0.149 |
| adfactorvae_g40_l50_conv2_abs | 1 | 5 | 0.060 |
| adfactorvae_g40_l50_conv2_abs | 2 | 5 | 0.071 |
| adfactorvae_g40_l50_conv2_abs | 3 | 5 | 0.068 |
| factorvae_g40_supp | 1 | 5 | 0.551 |
| factorvae_g40_supp | 2 | 5 | 0.492 |
| factorvae_g40_supp | 3 | 5 | 0.420 |
| adfactorvae_g40_l10_conv2_relu_supp | 1 | 5 | 0.499 |
| adfactorvae_g40_l10_conv2_relu_supp | 2 | 5 | 0.658 |
| adfactorvae_g40_l10_conv2_relu_supp | 3 | 5 | 0.463 |
| adfactorvae_g40_l10_conv2_abs_supp | 1 | 5 | 0.082 |
| adfactorvae_g40_l10_conv2_abs_supp | 2 | 9 | 0.370 |
| adfactorvae_g40_l10_conv2_abs_supp | 3 | 5 | 0.094 |
