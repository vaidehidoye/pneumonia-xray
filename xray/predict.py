"""Predict on a single chest X-ray image.

    python -m xray.predict path/to/xray.png
    python -m xray.predict path/to/xray.png --heatmap out.png
"""

import argparse
import json

import matplotlib
import numpy as np
from PIL import Image

from xray.data import IMAGE_SIZE, MEAN, STD, normalize
from xray.evaluate import RESULTS_DIR, load_checkpoint
from xray.gradcam import GradCAM
from xray.metrics import sigmoid
from xray.models import pick_device
from xray.train import RUNS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_image(path):
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return normalize(np.asarray(image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--heatmap", help="save a Grad-CAM overlay to this path")
    args = parser.parse_args()

    with open(RESULTS_DIR / args.model / "metrics.json") as f:
        metrics = json.load(f)
    threshold_key = next(k for k in metrics if k.startswith("at_val_sensitivity"))
    threshold = metrics[threshold_key]["threshold"]

    device = pick_device()
    model = load_checkpoint(RUNS_DIR / args.model / "best.pt", device)
    x = load_image(args.image).unsqueeze(0).to(device)

    heatmap, logits = GradCAM(model)(x)
    prob = float(sigmoid(logits.cpu().numpy()[0] / metrics["temperature"]))
    label = "pneumonia" if prob >= threshold else "normal"
    print(f"{label}  (p = {prob:.3f}, threshold {threshold:.2f})")

    if args.heatmap:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(x[0, 0].cpu().numpy() * STD + MEAN, cmap="gray")
        ax.imshow(heatmap[0].cpu().numpy(), cmap="jet", alpha=0.35)
        ax.set_title(f"{label}, p = {prob:.2f}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.heatmap, dpi=150)
        print(f"heatmap saved to {args.heatmap}")


if __name__ == "__main__":
    main()
