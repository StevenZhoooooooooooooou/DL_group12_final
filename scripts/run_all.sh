#!/bin/bash
# Runs every remaining experiment. Start with scripts/start_all.sh, stop with scripts/stop_all.sh.
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/vae-attn/bin/python
# MVTec: from-scratch encoder, then ImageNet-pretrained encoder (finished classes are skipped)
(
  $PY scripts/mvtec.py --pretrained 0 >> outputs/mvtec_scratch_bce.log 2>&1
  $PY scripts/mvtec.py --pretrained 1 >> outputs/mvtec_pt_bce.log 2>&1
  echo "[mvtec] all done $(date)"
) &
# dSprites Fig. 8 grid: 5 configs x 3 seeds, 3 in parallel
( scripts/run_dsprites_grid.sh; echo "[dsprites] all done $(date)" ) &
wait
echo "[run_all] finished $(date)"
