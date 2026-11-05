import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Four conv blocks, 128x128 -> 8x8, then global average pooling."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class ResNet18(nn.Module):
    """ImageNet ResNet-18 adapted to one input channel and one output logit."""

    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Collapse the RGB filters of the first layer into one grayscale filter,
        # so the pretrained weights still apply.
        rgb = net.conv1.weight.data
        net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        net.conv1.weight.data = rgb.sum(dim=1, keepdim=True)
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_model(name, pretrained=True):
    if name == "cnn":
        return SmallCNN()
    if name == "resnet18":
        return ResNet18(pretrained=pretrained)
    raise ValueError(f"unknown model: {name}")


def last_conv_layer(model):
    """The layer Grad-CAM reads activations and gradients from."""
    if isinstance(model, SmallCNN):
        return model.features[-1]
    return model.net.layer4


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
