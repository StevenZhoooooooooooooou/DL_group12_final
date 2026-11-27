#!/bin/bash
# Fig. 8 re-run with the supplement architecture (Table 4, z=32): 3 configs x 3 seeds, 3 in parallel.
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/vae-attn/bin/python
mkdir -p outputs/dsprites/logs
{
for s in 1 2 3; do
  echo "--arch supp --model factorvae --gamma 40 --seed $s"
  echo "--arch supp --model adfactorvae --gamma 40 --lamb 10 --ad_mode relu --seed $s"
  echo "--arch supp --model adfactorvae --gamma 40 --lamb 10 --ad_mode abs --seed $s"
done
} | xargs -P 3 -I{} bash -c 'a="{}"; n=$(echo $a | tr -d "-" | tr " " "_"); '"$PY"' scripts/dsprites.py $a > outputs/dsprites/logs/$n.log 2>&1'
