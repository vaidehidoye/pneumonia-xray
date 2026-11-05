"""Grad-CAM: which parts of the image pushed the prediction towards pneumonia.

For the last conv layer, take the gradient of the output logit with respect
to each feature map, average it over space to get one weight per channel,
and sum the feature maps with those weights. ReLU keeps only the regions that
increase the score.
"""

import torch
import torch.nn.functional as F

from xray.models import last_conv_layer


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        last_conv_layer(model).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad

    def __call__(self, x):
        """x: (N, 1, H, W). Returns heatmaps (N, H, W) in [0, 1] and logits (N,)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits.sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        flat = cam.flatten(1)
        lo = flat.min(dim=1).values[:, None, None]
        hi = flat.max(dim=1).values[:, None, None]
        cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
        return cam.detach(), logits.detach()
