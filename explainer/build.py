"""Fill template.html with reproduction results and figures -> ../../paper_explainer.html (self-contained)."""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
CLASSES = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule",
           "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
PAPER = {"carpet": .78, "grid": .73, "leather": .95, "tile": .80, "wood": .77, "bottle": .87, "cable": .90,
         "capsule": .74, "hazelnut": .98, "metal_nut": .94, "pill": .83, "screw": .97, "toothbrush": .94,
         "transistor": .93, "zipper": .78}
KEY = "layer2,mu,relu"


def b64(path, width):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def mvtec_rows():
    sc = {c: json.load(open(f"{OUT}/mvtec/scratch_bce/{c}.json")) for c in CLASSES}
    pt = {c: json.load(open(f"{OUT}/mvtec/pt_bce/{c}.json")) for c in CLASSES}
    rows, cols = [], []
    for c in CLASSES:
        v = (sc[c]["Vanilla-VAE"]["auroc"], pt[c]["Vanilla-VAE"]["auroc"], sc[c][KEY]["auroc"], pt[c][KEY]["auroc"])
        cols.append(v)
        rows.append(f'<tr><td>{c}</td><td>{PAPER[c]:.2f}</td><td>{v[0]:.3f} / {v[1]:.3f}</td>'
                    f'<td class="lo">{v[2]:.3f} / {v[3]:.3f}</td></tr>')
    m = np.mean(cols, 0)
    rows.append(f'<tr><td><b>平均</b></td><td><b>{np.mean(list(PAPER.values())):.2f}</b></td>'
                f'<td><b>{m[0]:.3f} / {m[1]:.3f}</b></td><td class="lo"><b>{m[2]:.3f} / {m[3]:.3f}</b></td></tr>')
    return "\n".join(rows)


html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
fill = {
    "{{MVTEC_ROWS}}": mvtec_rows(),
    "{{UCSD_IMG}}": b64(f"{OUT}/ucsd/fig5_qualitative_seed1.png", 900),
    "{{HAZEL_IMG}}": b64(f"{OUT}/mvtec/scratch_bce/hazelnut_qualitative.png", 760),
    "{{MNIST_IMG}}": b64(f"{OUT}/mnist/fig4_anomaly_attention_digit1.png", 900),
    "{{MVTEC_BAR_IMG}}": b64(f"{OUT}/report/mvtec_summary_scratch_bce.png", 1100),
    "{{FIG8_IMG}}": b64(f"{OUT}/report/fig8_disentanglement.png", 780),
    "{{FIG9_IMG}}": b64(f"{OUT}/report/fig9_attention.png", 640),
    "{{OFFICIAL_IMG}}": b64(f"{OUT}/official_mnist/official_digit1_comparison.png", 560),
}
for k, v in fill.items():
    assert k in html, k
    html = html.replace(k, v)
assert "{{" not in html
dst = os.path.join(HERE, "..", "..", "paper_explainer.html")
open(dst, "w", encoding="utf-8").write(html)
print(os.path.normpath(dst), os.path.getsize(dst) // 1024, "KB")
