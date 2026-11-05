import numpy as np
import pytest
import torch

from xray.data import XrayDataset, normalize
from xray.gradcam import GradCAM
from xray.metrics import (
    binary_metrics,
    expected_calibration_error,
    fit_temperature,
    sigmoid,
    threshold_for_sensitivity,
)
from xray.models import build_model


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_model_outputs_one_logit_per_image(name):
    model = build_model(name, pretrained=False).eval()
    out = model(torch.randn(3, 1, 128, 128))
    assert out.shape == (3,)


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_gradcam_heatmap_matches_input(name):
    model = build_model(name, pretrained=False)
    heatmaps, logits = GradCAM(model)(torch.randn(2, 1, 128, 128))
    assert heatmaps.shape == (2, 128, 128)
    assert logits.shape == (2,)
    assert heatmaps.min() >= 0 and heatmaps.max() <= 1


def test_dataset_item():
    images = np.random.randint(0, 256, (4, 128, 128), dtype=np.uint8)
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    x, y = XrayDataset(images, labels, augment=True)[1]
    assert x.shape == (1, 128, 128)
    assert y == 1


def test_normalize_adds_channel():
    assert normalize(np.zeros((128, 128), dtype=np.uint8)).shape == (1, 128, 128)


def test_binary_metrics():
    labels = np.array([1, 1, 1, 0, 0])
    probs = np.array([0.9, 0.8, 0.2, 0.7, 0.1])
    m = binary_metrics(labels, probs, threshold=0.5)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    probs = np.clip(labels * 0.3 + rng.random(500) * 0.7, 0, 1)
    threshold = threshold_for_sensitivity(labels, probs, target=0.95)
    assert binary_metrics(labels, probs, threshold)["sensitivity"] >= 0.95


def test_ece_is_zero_when_calibrated():
    probs = np.full(1000, 0.7)
    labels = np.array([1] * 700 + [0] * 300)
    assert expected_calibration_error(labels, probs) == pytest.approx(0.0, abs=1e-9)


def test_temperature_undoes_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 5000)
    labels = (rng.random(5000) < sigmoid(true_logits)).astype(float)
    # A model that is 3x too confident: temperature should come out near 3.
    assert fit_temperature(true_logits * 3, labels) == pytest.approx(3.0, rel=0.1)
