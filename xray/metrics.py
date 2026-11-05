import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def binary_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_sensitivity(labels, probs, target=0.95):
    """Highest threshold that still reaches the target sensitivity.

    Missing pneumonia is worse than a false alarm, so the operating point is
    chosen for sensitivity first. Pick it on validation data, not test data.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    ok = np.where(tpr >= target)[0]
    return float(min(thresholds[ok[0]], 1.0))


def bootstrap_auc(labels, probs, n=1000, seed=0):
    """95% confidence interval for AUC by resampling the test set."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def expected_calibration_error(labels, probs, bins=10):
    """Average gap between predicted probability and observed frequency."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(probs[in_bin].mean() - labels[in_bin].mean())
    return float(ece)


def fit_temperature(logits, labels):
    """Temperature scaling: find T so that sigmoid(logits / T) is calibrated.

    One parameter, fit on validation logits by minimizing log loss. It changes
    the confidence of predictions but not their ranking, so AUC is unchanged.
    """
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())
