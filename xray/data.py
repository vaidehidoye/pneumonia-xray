from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_SIZE = 128

# Pixel statistics of the training set, after scaling to [0, 1].
MEAN = 0.5717
STD = 0.1762


def download(root=DATA_DIR):
    from medmnist import PneumoniaMNIST

    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        PneumoniaMNIST(split=split, size=IMAGE_SIZE, download=True, root=str(root))


def load_split(split, root=DATA_DIR):
    path = root / f"pneumoniamnist_{IMAGE_SIZE}.npz"
    if not path.exists():
        download(root)
    data = np.load(path)
    return data[f"{split}_images"], data[f"{split}_labels"].ravel().astype(np.float32)


def normalize(x):
    """uint8 image(s) -> float tensor with a channel dim, normalized."""
    x = torch.as_tensor(x, dtype=torch.float32) / 255.0
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return (x - MEAN) / STD


class XrayDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = torch.as_tensor(labels)
        # No horizontal flips: the heart sits on the left, and a flipped chest
        # X-ray isn't a realistic image.
        self.augment = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)) if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = normalize(self.images[i])
        if self.augment is not None:
            x = self.augment(x)
        return x, self.labels[i]


def make_loader(split, batch_size=64, shuffle=False, augment=False):
    images, labels = load_split(split)
    return DataLoader(XrayDataset(images, labels, augment=augment), batch_size=batch_size, shuffle=shuffle)
