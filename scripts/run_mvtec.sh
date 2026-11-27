#!/bin/bash
# MVTec: from-scratch encoder, then ImageNet-pretrained encoder (finished classes are skipped)
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/vae-attn/bin/python
$PY scripts/mvtec.py --pretrained 0 >> outputs/mvtec_scratch_bce.log 2>&1
$PY scripts/mvtec.py --pretrained 1 >> outputs/mvtec_pt_bce.log 2>&1
