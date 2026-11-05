"""Evaluate a trained model on the test set.

    python -m xray.evaluate --model resnet18

Everything that gets tuned (temperature, decision threshold) is fit on the
validation set. The test set is only used to report numbers.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from xray.data import MEAN, STD, make_loader
from xray.gradcam import GradCAM
from xray.metrics import (
    binary_metrics,
    bootstrap_auc,
    expected_calibration_error,
    fit_temperature,
    sigmoid,
    threshold_for_sensitivity,
)
from xray.models import build_model, pick_device
from xray.train import RUNS_DIR, predict_logits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def plot_roc(labels, probs, point, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(labels, probs):.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.scatter([1 - point["specificity"]], [point["sensitivity"]], color="black", zorder=3,
               label=f"threshold {point['threshold']:.2f}")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title("ROC, test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(labels, before, after, path, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    for probs, name in [(before, "before"), (after, "after temperature scaling")]:
        idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
        observed = [labels[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
        ece = expected_calibration_error(labels, probs, bins)
        ax.plot(centers, observed, marker="o", label=f"{name} (ECE {ece:.3f})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed pneumonia rate")
    ax.set_title("Calibration, test set")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gradcam(model, device, images, labels, probs, threshold, path, per_group=3):
    """Examples of each outcome: true/false positives and negatives."""
    preds = probs >= threshold
    groups = {
        "true positive": np.where(preds & (labels == 1))[0],
        "true negative": np.where(~preds & (labels == 0))[0],
        "false positive": np.where(preds & (labels == 0))[0],
        "false negative": np.where(~preds & (labels == 1))[0],
    }
    # Most confident examples within each group.
    chosen = []
    for name, idx in groups.items():
        confidence = np.abs(probs[idx] - threshold)
        for i in idx[np.argsort(-confidence)][:per_group]:
            chosen.append((name, i))

    cam = GradCAM(model)
    x = torch.stack([images[i] for _, i in chosen]).to(device)
    heatmaps, _ = cam(x)

    fig, axes = plt.subplots(len(groups), per_group, figsize=(2.2 * per_group, 2.4 * len(groups)))
    for ax in axes.flat:
        ax.axis("off")
    counts = {name: 0 for name in groups}
    for (name, i), heat in zip(chosen, heatmaps.cpu().numpy()):
        row = list(groups).index(name)
        ax = axes[row, counts[name]]
        counts[name] += 1
        ax.imshow(images[i][0].numpy() * STD + MEAN, cmap="gray")
        ax.imshow(heat, cmap="jet", alpha=0.35)
        ax.set_title(f"{name}\np={probs[i]:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    args = parser.parse_args()

    device = pick_device()
    model = load_checkpoint(RUNS_DIR / args.model / "best.pt", device)
    out_dir = RESULTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    val_logits, val_labels = predict_logits(model, make_loader("val"), device)
    test_loader = make_loader("test")
    test_logits, test_labels = predict_logits(model, test_loader, device)

    temperature = fit_temperature(val_logits, val_labels)
    val_probs = sigmoid(val_logits / temperature)
    test_probs_raw = sigmoid(test_logits)
    test_probs = sigmoid(test_logits / temperature)

    threshold = threshold_for_sensitivity(val_labels, val_probs, args.target_sensitivity)
    at_threshold = binary_metrics(test_labels, test_probs, threshold)
    at_half = binary_metrics(test_labels, test_probs, 0.5)
    auc = roc_auc_score(test_labels, test_probs)
    auc_low, auc_high = bootstrap_auc(test_labels, test_probs)

    results = {
        "model": args.model,
        "temperature": temperature,
        "test_auc": auc,
        "test_auc_95ci": [auc_low, auc_high],
        "val_auc": roc_auc_score(val_labels, val_probs),
        "ece_before": expected_calibration_error(test_labels, test_probs_raw),
        "ece_after": expected_calibration_error(test_labels, test_probs),
        f"at_val_sensitivity_{args.target_sensitivity:.2f}": at_threshold,
        "at_threshold_0.5": at_half,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    plot_roc(test_labels, test_probs, at_threshold, out_dir / "roc.png")
    plot_reliability(test_labels, test_probs_raw, test_probs, out_dir / "calibration.png")
    test_images = [x for batch, _ in test_loader for x in batch]
    plot_gradcam(model, device, test_images, test_labels, test_probs, threshold, out_dir / "gradcam.png")

    print(f"{args.model}")
    print(f"  test AUC        {auc:.3f}  (95% CI {auc_low:.3f}-{auc_high:.3f})")
    print(f"  temperature     {temperature:.2f}   ECE {results['ece_before']:.3f} -> {results['ece_after']:.3f}")
    for name, m in [(f"threshold {threshold:.2f} (val sens {args.target_sensitivity:.0%})", at_threshold),
                    ("threshold 0.50", at_half)]:
        print(f"  {name}: sensitivity {m['sensitivity']:.3f}  specificity {m['specificity']:.3f}  "
              f"accuracy {m['accuracy']:.3f}")
    print(f"  saved to {out_dir}")


if __name__ == "__main__":
    main()
