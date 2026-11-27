# Re-evaluation without retraining: post-processing, oracle upper bound, cross-validated selection

Post-processing: `raw` = maps as produced, `norm` = per-image min-max, `blur` = Gaussian (sigma 4 px at 256), `blur_norm` = both. The same four options are given to the reconstruction residual.

## MVTec-AD, encoder `pt_bce` (15/15 classes), paper mean AUROC 0.86 / IoU 0.20

### A. Effect of post-processing (one configuration for all classes, layer chosen by mean AUROC)

| post-processing | recon. residual | paper setting (best fixed layer) | best fixed attention variant |
|---|---|---|---|
| raw | 0.806 | 0.558 (`layer1,mu,relu`) | 0.635 (`layer2,mahal,chabs`) |
| norm | 0.781 | 0.565 (`layer1,mu,relu`) | 0.687 (`layer1,mu,chabs`) |
| blur | 0.884 | 0.595 (`layer1,mu,relu`) | 0.634 (`layer2,mu,chabs`) |
| blur_norm | 0.835 | 0.617 (`layer1,mu,relu`) | 0.696 (`layer1,mu,chabs`) |

### B. Per-class oracle (upper bound: best option per class chosen on the test set itself)

| class | paper | recon. oracle | paper-setting oracle | official-code oracle |
|---|---|---|---|---|
| carpet | 0.78 | 0.671 | 0.697 <sub>layer1,mu,relu / blur_norm</sub> | 0.716 |
| grid | 0.73 | 0.864 | 0.620 <sub>layer1,mu,relu / raw</sub> | 0.588 |
| leather | 0.95 | 0.986 | 0.740 <sub>layer1,mu,relu / blur_norm</sub> | 0.739 |
| tile | 0.80 | 0.529 | 0.549 <sub>layer3,mu,relu / raw</sub> | 0.676 |
| wood | 0.77 | 0.797 | 0.549 <sub>layer1,mu,relu / blur_norm</sub> | 0.673 |
| bottle | 0.87 | 0.943 | 0.665 <sub>layer1,mu,relu / blur_norm</sub> | 0.566 |
| cable | 0.90 | 0.916 | 0.578 <sub>layer2,mu,relu / blur</sub> | 0.598 |
| capsule | 0.74 | 0.956 | 0.679 <sub>layer1,mu,relu / blur_norm</sub> | 0.626 |
| hazelnut | 0.98 | 0.983 | 0.716 <sub>layer2,mu,relu / blur_norm</sub> | 0.590 |
| metal_nut | 0.94 | 0.934 | 0.633 <sub>layer1,mu,relu / blur_norm</sub> | 0.648 |
| pill | 0.83 | 0.940 | 0.763 <sub>layer1,mu,relu / blur_norm</sub> | 0.700 |
| screw | 0.97 | 0.983 | 0.861 <sub>layer1,mu,relu / blur_norm</sub> | 0.822 |
| toothbrush | 0.94 | 0.984 | 0.658 <sub>layer1,mu,relu / blur</sub> | 0.676 |
| transistor | 0.93 | 0.888 | 0.591 <sub>layer1,mu,relu / blur_norm</sub> | 0.596 |
| zipper | 0.78 | 0.900 | 0.603 <sub>layer2,mu,relu / blur_norm</sub> | 0.573 |
| **mean AUROC** | 0.86 | 0.885 | 0.660 | 0.652 |
| **mean best IoU** | 0.20 | 0.267 | 0.062 | 0.060 |

### C. 2-fold cross-validated selection (per class: choose on one half of the test set, score on the other)

| method family | mean AUROC | mean best IoU | classes where it beats CV recon. |
|---|---|---|---|
| recon. residual | 0.883 | 0.266 |  |
| paper setting (sum mu, ReLU) | 0.647 | 0.062 | 2/15 |
| official code exactly (sum mu, upsample then abs) | 0.639 | 0.059 | 2/15 |
| SZU third-party code (sum mu, channel-wise abs) | 0.719 | 0.074 | 3/15 |
| any sum-mu aggregation (ReLU / abs / channel-wise abs) | 0.738 | 0.081 | 3/15 |
| any attention variant (+ Eq.4 / KL / Mahalanobis, not in paper) | 0.769 | 0.109 | 3/15 |

## MVTec-AD, encoder `scratch_bce` (15/15 classes), paper mean AUROC 0.86 / IoU 0.20

### A. Effect of post-processing (one configuration for all classes, layer chosen by mean AUROC)

| post-processing | recon. residual | paper setting (best fixed layer) | best fixed attention variant |
|---|---|---|---|
| raw | 0.794 | 0.550 (`layer4,mu,relu`) | 0.711 (`layer1,kl,chabs`) |
| norm | 0.770 | 0.546 (`layer4,mu,relu`) | 0.704 (`layer1,mu,chabs`) |
| blur | 0.863 | 0.554 (`layer4,mu,relu`) | 0.714 (`layer1,kl,chabs`) |
| blur_norm | 0.813 | 0.550 (`layer4,mu,relu`) | 0.715 (`layer1,mu,chabs`) |

### B. Per-class oracle (upper bound: best option per class chosen on the test set itself)

| class | paper | recon. oracle | paper-setting oracle | official-code oracle |
|---|---|---|---|---|
| carpet | 0.78 | 0.581 | 0.541 <sub>layer1,mu,relu / blur_norm</sub> | 0.668 |
| grid | 0.73 | 0.668 | 0.609 <sub>layer4,mu,relu / blur_norm</sub> | 0.523 |
| leather | 0.95 | 0.986 | 0.520 <sub>layer4,mu,relu / blur</sub> | 0.872 |
| tile | 0.80 | 0.522 | 0.475 <sub>layer3,mu,relu / raw</sub> | 0.660 |
| wood | 0.77 | 0.798 | 0.622 <sub>layer3,mu,relu / blur</sub> | 0.652 |
| bottle | 0.87 | 0.941 | 0.496 <sub>layer3,mu,relu / raw</sub> | 0.502 |
| cable | 0.90 | 0.898 | 0.822 <sub>layer2,mu,relu / blur_norm</sub> | 0.749 |
| capsule | 0.74 | 0.953 | 0.599 <sub>layer2,mu,relu / raw</sub> | 0.608 |
| hazelnut | 0.98 | 0.984 | 0.823 <sub>layer3,mu,relu / blur</sub> | 0.800 |
| metal_nut | 0.94 | 0.927 | 0.800 <sub>layer1,mu,relu / blur</sub> | 0.800 |
| pill | 0.83 | 0.941 | 0.461 <sub>layer3,mu,relu / blur</sub> | 0.571 |
| screw | 0.97 | 0.981 | 0.550 <sub>layer4,mu,relu / blur</sub> | 0.916 |
| toothbrush | 0.94 | 0.986 | 0.797 <sub>layer1,mu,relu / blur_norm</sub> | 0.795 |
| transistor | 0.93 | 0.908 | 0.672 <sub>layer4,mu,relu / blur</sub> | 0.618 |
| zipper | 0.78 | 0.890 | 0.580 <sub>layer4,mu,relu / blur</sub> | 0.774 |
| **mean AUROC** | 0.86 | 0.864 | 0.624 | 0.701 |
| **mean best IoU** | 0.20 | 0.252 | 0.068 | 0.083 |

### C. 2-fold cross-validated selection (per class: choose on one half of the test set, score on the other)

| method family | mean AUROC | mean best IoU | classes where it beats CV recon. |
|---|---|---|---|
| recon. residual | 0.862 | 0.253 |  |
| paper setting (sum mu, ReLU) | 0.606 | 0.068 | 0/15 |
| official code exactly (sum mu, upsample then abs) | 0.686 | 0.084 | 2/15 |
| SZU third-party code (sum mu, channel-wise abs) | 0.769 | 0.111 | 3/15 |
| any sum-mu aggregation (ReLU / abs / channel-wise abs) | 0.788 | 0.117 | 3/15 |
| any attention variant (+ Eq.4 / KL / Mahalanobis, not in paper) | 0.812 | 0.134 | 3/15 |

## MVTec-AD, encoder `supp_scratch_bce` (15/15 classes), paper mean AUROC 0.86 / IoU 0.20

### A. Effect of post-processing (one configuration for all classes, layer chosen by mean AUROC)

| post-processing | recon. residual | paper setting (best fixed layer) | best fixed attention variant |
|---|---|---|---|
| raw | 0.805 | 0.572 (`layer4,mu,relu`) | 0.671 (`layer2,mahal,chabs`) |
| norm | 0.782 | 0.562 (`layer4,mu,relu`) | 0.654 (`layer1,ndd,chabs`) |
| blur | 0.878 | 0.575 (`layer4,mu,relu`) | 0.671 (`layer2,mahal,chabs`) |
| blur_norm | 0.831 | 0.564 (`layer4,mu,relu`) | 0.657 (`layer1,ndd,chabs`) |

### B. Per-class oracle (upper bound: best option per class chosen on the test set itself)

| class | paper | recon. oracle | paper-setting oracle | official-code oracle |
|---|---|---|---|---|
| carpet | 0.78 | 0.589 | 0.571 <sub>layer1,mu,relu / blur_norm</sub> | 0.541 |
| grid | 0.73 | 0.904 | 0.493 <sub>layer4,mu,relu / blur_norm</sub> | 0.558 |
| leather | 0.95 | 0.962 | 0.715 <sub>layer4,mu,relu / blur</sub> | 0.823 |
| tile | 0.80 | 0.508 | 0.606 <sub>layer2,mu,relu / blur</sub> | 0.720 |
| wood | 0.77 | 0.814 | 0.645 <sub>layer4,mu,relu / blur</sub> | 0.694 |
| bottle | 0.87 | 0.950 | 0.621 <sub>layer4,mu,relu / blur</sub> | 0.622 |
| cable | 0.90 | 0.916 | 0.517 <sub>layer4,mu,relu / blur_norm</sub> | 0.755 |
| capsule | 0.74 | 0.953 | 0.487 <sub>layer4,mu,relu / blur_norm</sub> | 0.630 |
| hazelnut | 0.98 | 0.982 | 0.827 <sub>layer3,mu,relu / blur_norm</sub> | 0.823 |
| metal_nut | 0.94 | 0.930 | 0.581 <sub>layer4,mu,relu / blur</sub> | 0.660 |
| pill | 0.83 | 0.919 | 0.510 <sub>layer1,mu,relu / blur</sub> | 0.697 |
| screw | 0.97 | 0.983 | 0.900 <sub>layer2,mu,relu / blur_norm</sub> | 0.905 |
| toothbrush | 0.94 | 0.986 | 0.764 <sub>layer3,mu,relu / blur_norm</sub> | 0.797 |
| transistor | 0.93 | 0.901 | 0.559 <sub>layer4,mu,relu / blur</sub> | 0.599 |
| zipper | 0.78 | 0.896 | 0.546 <sub>layer2,mu,relu / raw</sub> | 0.490 |
| **mean AUROC** | 0.86 | 0.879 | 0.623 | 0.688 |
| **mean best IoU** | 0.20 | 0.257 | 0.054 | 0.072 |

### C. 2-fold cross-validated selection (per class: choose on one half of the test set, score on the other)

| method family | mean AUROC | mean best IoU | classes where it beats CV recon. |
|---|---|---|---|
| recon. residual | 0.878 | 0.254 |  |
| paper setting (sum mu, ReLU) | 0.609 | 0.052 | 1/15 |
| official code exactly (sum mu, upsample then abs) | 0.672 | 0.065 | 1/15 |
| SZU third-party code (sum mu, channel-wise abs) | 0.714 | 0.078 | 1/15 |
| any sum-mu aggregation (ReLU / abs / channel-wise abs) | 0.714 | 0.082 | 1/15 |
| any attention variant (+ Eq.4 / KL / Mahalanobis, not in paper) | 0.730 | 0.097 | 1/15 |

## UCSD Ped1 `ucsd` (3 seeds: seed1, seed2, seed3), pixel AUROC, mean +- std over seeds

Paper Table 1: Vanilla-VAE 0.86, Conv1 0.89, Conv2 0.92, Conv3 0.91. Folds for C = test clips 1-5 vs 6-10.

### A. Table 1 under each post-processing

| post-processing | recon. residual | paper setting Conv1 | Conv2 | Conv3 |
|---|---|---|---|---|
| raw | 0.904 +- 0.001 | 0.486 +- 0.095 | 0.492 +- 0.095 | 0.628 +- 0.103 |
| norm | 0.900 +- 0.002 | 0.486 +- 0.084 | 0.485 +- 0.093 | 0.629 +- 0.109 |
| blur | 0.955 +- 0.001 | 0.522 +- 0.131 | 0.534 +- 0.117 | 0.654 +- 0.095 |
| blur_norm | 0.946 +- 0.001 | 0.518 +- 0.121 | 0.531 +- 0.112 | 0.656 +- 0.101 |

### B. Oracle and C. cross-validated selection

| method family | oracle (upper bound) | 2-fold CV |
|---|---|---|
| recon. residual | 0.955 +- 0.001 | 0.955 +- 0.001 |
| paper setting (sum mu, ReLU) | 0.664 +- 0.092 | 0.659 +- 0.098 |
| official code exactly (sum mu, upsample then abs) | 0.642 +- 0.081 | 0.631 +- 0.084 |
| SZU third-party code (sum mu, channel-wise abs) | 0.845 +- 0.045 | 0.843 +- 0.044 |
| any sum-mu aggregation (ReLU / abs / channel-wise abs) | 0.845 +- 0.045 | 0.843 +- 0.044 |
| any attention variant (+ Eq.4 / KL / Mahalanobis, not in paper) | 0.897 +- 0.010 | 0.894 +- 0.010 |

## UCSD Ped1 `ucsd_supp` (3 seeds: seed1, seed2, seed3), pixel AUROC, mean +- std over seeds

Paper Table 1: Vanilla-VAE 0.86, Conv1 0.89, Conv2 0.92, Conv3 0.91. Folds for C = test clips 1-5 vs 6-10.

### A. Table 1 under each post-processing

| post-processing | recon. residual | paper setting Conv1 | Conv2 | Conv3 |
|---|---|---|---|---|
| raw | 0.902 +- 0.000 | 0.420 +- 0.098 | 0.630 +- 0.085 | 0.575 +- 0.197 |
| norm | 0.899 +- 0.000 | 0.416 +- 0.097 | 0.619 +- 0.113 | 0.574 +- 0.197 |
| blur | 0.953 +- 0.000 | 0.435 +- 0.094 | 0.638 +- 0.073 | 0.569 +- 0.202 |
| blur_norm | 0.944 +- 0.000 | 0.430 +- 0.095 | 0.630 +- 0.102 | 0.567 +- 0.203 |

### B. Oracle and C. cross-validated selection

| method family | oracle (upper bound) | 2-fold CV |
|---|---|---|
| recon. residual | 0.953 +- 0.000 | 0.953 +- 0.000 |
| paper setting (sum mu, ReLU) | 0.698 +- 0.129 | 0.695 +- 0.124 |
| official code exactly (sum mu, upsample then abs) | 0.873 +- 0.014 | 0.868 +- 0.012 |
| SZU third-party code (sum mu, channel-wise abs) | 0.893 +- 0.002 | 0.892 +- 0.002 |
| any sum-mu aggregation (ReLU / abs / channel-wise abs) | 0.893 +- 0.002 | 0.892 +- 0.002 |
| any attention variant (+ Eq.4 / KL / Mahalanobis, not in paper) | 0.893 +- 0.002 | 0.889 +- 0.007 |
