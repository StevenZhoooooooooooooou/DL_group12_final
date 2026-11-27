# Changes to the official expVAE code (liuem607/expVAE, code/) needed to run it on PyTorch 2.11 / NumPy 2

1. `test_expVAE.py`: `np.float` -> `float` (the alias was removed in NumPy 1.24).
2. Data: the original downloader points at yann.lecun.com, which is no longer reliable. `data/processed/{training,test}.pt`
   were written from the torchvision copy of MNIST in exactly the format `OneClassMnist.py` expects
   (`(uint8 images, labels)` tuples), so the dataset code itself is unchanged.

Everything else (model, training loop, loss, optimiser, Grad-CAM with deprecated `register_backward_hook`,
`F.upsample`, target layer `encoder.2`) is run as published.

## Pretrained weights
The README's Google Drive link for pretrained MNIST models (file id 1LLJkAEVA7G5ktzo7EfIsyHyHDGTIVxyG) returned
HTTP 404 "Page Not Found" on 2026-10-11, so the official training script was run as published instead
(`python train_expVAE.py --one_class 1 --epochs 100`, final val loss 42.8, `train_digit1.log`).

## Equivalence with our implementation (`scripts/official_check.py`)
Same checkpoint and the same 128 test images per outlier digit, official hook-based GradCAM at 'encoder.2' vs our
autograd implementation at the same tensor:
- official hook gradient vs exact autograd gradient: per-image cosine 1.000 (the deprecated hook is still correct)
- the official code upsamples the signed 7x7 map *before* taking abs; with that order (`mode="absup"`) our maps match
  exactly: per-image correlation 1.0000, max relative error about 2e-7. Taking abs before upsampling, which is our
  original `abs` mode, gives only 0.45-0.72 minimum / 0.84-0.93 mean correlation.
