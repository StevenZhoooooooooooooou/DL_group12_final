#!/bin/bash
# Full Fig. 8 grid: 5 configurations x 3 seeds, 3 runs in parallel on one GPU.
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/vae-attn/bin/python
mkdir -p outputs/dsprites/logs
{
for s in 1 2 3; do
  echo "--model factorvae --gamma 40 --seed $s"
  echo "--model adfactorvae --gamma 40 --lamb 10 --ad_mode abs --seed $s"
  echo "--model adfactorvae --gamma 40 --lamb 10 --ad_mode relu --seed $s"
  echo "--model adfactorvae --gamma 40 --lamb 50 --ad_mode abs --seed $s"
  echo "--model betavae --beta 4 --seed $s"
done
} | xargs -P 3 -I{} bash -c 'a="{}"; n=$(echo $a | tr -d "-" | tr " " "_"); '"$PY"' scripts/dsprites.py $a > outputs/dsprites/logs/$n.log 2>&1'
