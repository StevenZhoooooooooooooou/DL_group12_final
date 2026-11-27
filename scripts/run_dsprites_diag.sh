#!/bin/bash
# FactorVAE (gamma=40) stability diagnostics: one change at a time, 30k iterations each, run in parallel.
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/vae-attn/bin/python
mkdir -p outputs/dsprites_diag
C="--model factorvae --gamma 40 --seed 1 --iters 30000 --eval_every 30000 --out outputs/dsprites_diag"
$PY scripts/dsprites.py $C --tag _A_asis                       > outputs/dsprites_diag/A_asis.log 2>&1 &
$PY scripts/dsprites.py $C --tag _B_notf32 --tf32 0            > outputs/dsprites_diag/B_notf32.log 2>&1 &
$PY scripts/dsprites.py $C --tag _C_deconvdefault --deconv_init default > outputs/dsprites_diag/C_deconvdefault.log 2>&1 &
$PY scripts/dsprites.py $C --tag _D_z32 --z_dim 32             > outputs/dsprites_diag/D_z32.log 2>&1 &
wait
