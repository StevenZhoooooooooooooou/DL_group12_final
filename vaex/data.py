"""Datasets, loaded fully into memory as tensors (all four are small enough)."""
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MVTEC_CLASSES = ["carpet", "grid", "leather", "tile", "wood",
                 "bottle", "cable", "capsule", "hazelnut", "metal_nut",
                 "pill", "screw", "toothbrush", "transistor", "zipper"]
MVTEC_TEXTURES = MVTEC_CLASSES[:5]


def mnist(train, digit=None, root=ROOT):
    """Returns (x in [0,1], N x 1 x 28 x 28; labels)."""
    import torchvision
    ds = torchvision.datasets.MNIST(os.path.join(root, "mnist"), train=train, download=True)
    x = ds.data.float().div(255).unsqueeze(1)
    y = ds.targets
    if digit is not None:
        x, y = x[y == digit], y[y == digit]
    return x, y


def _load_gray(path, size):
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())


def ucsd_ped1(train, size=100, root=ROOT):
    """UCSD Ped1. Train: 34 normal clips. Test: only the 10 clips with pixel-level GT.

    Returns x (N x 1 x size x size, float [0,1]) and, for test, masks (N x size x size, bool).
    """
    base = os.path.join(root, "UCSD_Anomaly_Dataset.v1p2", "UCSDped1", "Train" if train else "Test")
    clips = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.endswith("_gt"))
    if not train:
        clips = [c for c in clips if os.path.isdir(os.path.join(base, c + "_gt"))]
    xs, ms = [], []
    for c in clips:
        for f in sorted(glob.glob(os.path.join(base, c, "*.tif"))):
            xs.append(_load_gray(f, size))
            if not train:
                g = os.path.join(base, c + "_gt", os.path.basename(f).replace(".tif", ".bmp"))
                m = Image.open(g).convert("L").resize((size, size), Image.NEAREST)
                ms.append(torch.from_numpy(np.asarray(m) > 127))
    x = torch.stack(xs).unsqueeze(1).float().div(255)
    return (x, None) if train else (x, torch.stack(ms))


def mvtec(cls, train, size=256, root=ROOT):
    """MVTec-AD. Returns x (N x 3 x size x size, uint8), masks (N x size x size bool), defect types."""
    base = os.path.join(root, "mvtec", cls, "train" if train else "test")
    xs, ms, types = [], [], []
    for t in sorted(os.listdir(base)):
        for f in sorted(glob.glob(os.path.join(base, t, "*.png"))):
            im = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
            xs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
            if t == "good":
                ms.append(torch.zeros(size, size, dtype=torch.bool))
            else:
                g = os.path.join(root, "mvtec", cls, "ground_truth", t, os.path.basename(f)[:-4] + "_mask.png")
                m = Image.open(g).convert("L").resize((size, size), Image.NEAREST)
                ms.append(torch.from_numpy(np.asarray(m) > 127))
            types.append(t)
    return torch.stack(xs), torch.stack(ms), types


def augment_mirror_rotate(x, max_deg=15.0):
    """Random horizontal mirroring + random rotation (paper: 'random mirroring and random rotation').

    x: B x C x H x W float on any device. Rotation uses reflection padding to avoid black corners.
    """
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    ang = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(max_deg)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


class DSprites:
    """dSprites (737,280 64x64 binary images) with its ground-truth factor classes."""

    # factor columns in latents_classes: color, shape, scale, orientation, posX, posY
    FACTOR_NAMES = ["shape", "scale", "orientation", "posX", "posY"]

    def __init__(self, root=ROOT):
        d = np.load(os.path.join(root, "dsprites", "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"))
        self.imgs = torch.from_numpy(d["imgs"])                   # N x 64 x 64 uint8 {0,1}
        self.classes = d["latents_classes"][:, 1:]                # drop the constant 'color' factor
        self.n_values = self.classes.max(0) + 1                   # [3, 6, 40, 32, 32]
        self.N = len(self.imgs)

    def batch(self, idx, device):
        return self.imgs[idx].to(device, non_blocking=True).unsqueeze(1).float()
