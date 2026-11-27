"""Summarise outputs/reeval/* into outputs/report/reeval.md.

A. post-processing effect (one fixed configuration for all classes)
B. per-class oracle = most generous upper bound for the paper (best layer x post-processing chosen on the test set)
C. 2-fold cross-validation: configuration chosen per class on one half of the test set, scored on the other half
"""
import glob
import json
import os

import numpy as np

PAPER_MVTEC = {"carpet": .78, "grid": .73, "leather": .95, "tile": .80, "wood": .77, "bottle": .87, "cable": .90,
               "capsule": .74, "hazelnut": .98, "metal_nut": .94, "pill": .83, "screw": .97, "toothbrush": .94,
               "transistor": .93, "zipper": .78}
POSTS = ["raw", "norm", "blur", "blur_norm"]
FAMILIES = {
    "recon. residual": lambda k: k == "recon",
    "paper setting (sum mu, ReLU)": lambda k: k.endswith(",mu,relu"),
    "official code exactly (sum mu, upsample then abs)": lambda k: k.endswith(",mu,absup"),
    "SZU third-party code (sum mu, channel-wise abs)": lambda k: k.endswith(",mu,chabs"),
    "any sum-mu aggregation (ReLU / abs / channel-wise abs)": lambda k: ",mu," in k,
    "any attention variant (+ Eq.4 / KL / Mahalanobis, not in paper)": lambda k: k != "recon",
}


def candidates(res, fam):
    return [(k, p) for k in res if FAMILIES[fam](k) for p in POSTS]


def oracle(res, fam):
    k, p = max(candidates(res, fam), key=lambda c: res[c[0]][c[1]]["all"][0])
    return res[k][p]["all"], f"{k} / {p}"


def cross_val(res, fam):
    """Pick on fold A -> score on fold B and vice versa; return mean [auc, iou] and the picks."""
    scores, picks = [], []
    for sel, ev in (("A", "B"), ("B", "A")):
        k, p = max(candidates(res, fam), key=lambda c: res[c[0]][c[1]][sel][0])
        scores.append(res[k][p][ev])
        picks.append(f"{k}/{p}")
    return np.mean(scores, 0), picks


md = ["# Re-evaluation without retraining: post-processing, oracle upper bound, cross-validated selection\n",
      "Post-processing: `raw` = maps as produced, `norm` = per-image min-max, `blur` = Gaussian (sigma 4 px at 256), "
      "`blur_norm` = both. The same four options are given to the reconstruction residual.\n"]

# ------------------------------------------------------------------ MVTec
for d in sorted(glob.glob("outputs/reeval/mvtec_*")):
    tag = d.split("mvtec_")[1]
    R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json"))}
    if not R:
        continue
    cls = [c for c in PAPER_MVTEC if c in R]
    md.append(f"## MVTec-AD, encoder `{tag}` ({len(cls)}/15 classes), paper mean AUROC 0.86 / IoU 0.20\n")

    md += ["### A. Effect of post-processing (one configuration for all classes, layer chosen by mean AUROC)\n",
           "| post-processing | recon. residual | paper setting (best fixed layer) | best fixed attention variant |",
           "|---|---|---|---|"]
    keys = [k for k in R[cls[0]] if k != "recon"]
    for p in POSTS:
        rec = np.mean([R[c]["recon"][p]["all"][0] for c in cls])
        mean = {k: np.mean([R[c][k][p]["all"][0] for c in cls]) for k in keys}
        kp = max((k for k in keys if k.endswith(",mu,relu")), key=mean.get)
        kb = max(keys, key=mean.get)
        md.append(f"| {p} | {rec:.3f} | {mean[kp]:.3f} (`{kp}`) | {mean[kb]:.3f} (`{kb}`) |")
    md.append("")

    md += ["### B. Per-class oracle (upper bound: best option per class chosen on the test set itself)\n",
           "| class | paper | recon. oracle | paper-setting oracle | official-code oracle |", "|---|---|---|---|---|"]
    rows = []
    for c in cls:
        r = oracle(R[c], "recon. residual")[0]
        a, ka = oracle(R[c], "paper setting (sum mu, ReLU)")
        b = oracle(R[c], "official code exactly (sum mu, upsample then abs)")[0]
        rows.append([PAPER_MVTEC[c], r[0], a[0], b[0], r[1], a[1], b[1]])
        md.append(f"| {c} | {PAPER_MVTEC[c]:.2f} | {r[0]:.3f} | {a[0]:.3f} <sub>{ka}</sub> | {b[0]:.3f} |")
    m = np.mean(rows, 0)
    md += [f"| **mean AUROC** | {m[0]:.2f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} |",
           f"| **mean best IoU** | 0.20 | {m[4]:.3f} | {m[5]:.3f} | {m[6]:.3f} |", ""]

    md += ["### C. 2-fold cross-validated selection (per class: choose on one half of the test set, score on the other)\n",
           "| method family | mean AUROC | mean best IoU | classes where it beats CV recon. |", "|---|---|---|---|"]
    cv = {f: {c: cross_val(R[c], f)[0] for c in cls} for f in FAMILIES}
    for f in FAMILIES:
        auc = np.mean([cv[f][c][0] for c in cls]); iou = np.mean([cv[f][c][1] for c in cls])
        wins = "" if f == "recon. residual" else f"{sum(cv[f][c][0] > cv['recon. residual'][c][0] for c in cls)}/{len(cls)}"
        md.append(f"| {f} | {auc:.3f} | {iou:.3f} | {wins} |")
    md.append("")

# ------------------------------------------------------------------ UCSD
for udir in sorted(glob.glob("outputs/reeval/ucsd*")):
  U = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{udir}/seed*.json"))}
  if U:
      seeds = sorted(U)
      md += [f"## UCSD Ped1 `{os.path.basename(udir)}` ({len(seeds)} seeds: {', '.join(seeds)}), pixel AUROC, mean +- std over seeds\n",
             "Paper Table 1: Vanilla-VAE 0.86, Conv1 0.89, Conv2 0.92, Conv3 0.91. Folds for C = test clips 1-5 vs 6-10.\n",
             "### A. Table 1 under each post-processing\n",
             "| post-processing | recon. residual | paper setting Conv1 | Conv2 | Conv3 |", "|---|---|---|---|---|"]
      ms = lambda vals: f"{np.mean(vals):.3f} +- {np.std(vals):.3f}"
      for p in POSTS:
          cells = [ms([U[s]["recon"][p]["all"][0] for s in seeds])]
          cells += [ms([U[s][f"{l},mu,relu"][p]["all"][0] for s in seeds]) for l in ["conv1", "conv2", "conv3"]]
          md.append(f"| {p} | " + " | ".join(cells) + " |")
      md += ["", "### B. Oracle and C. cross-validated selection\n",
             "| method family | oracle (upper bound) | 2-fold CV |", "|---|---|---|"]
      for f in FAMILIES:
          o = [oracle(U[s], f)[0][0] for s in seeds]
          c = [cross_val(U[s], f)[0][0] for s in seeds]
          md.append(f"| {f} | {ms(o)} | {ms(c)} |")
      md.append("")

os.makedirs("outputs/report", exist_ok=True)
open("outputs/report/reeval.md", "w").write("\n".join(md))
print("\n".join(md))
