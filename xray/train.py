"""Train a pneumonia classifier.

    python -m xray.train --model resnet18
    python -m xray.train --model cnn --epochs 30
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from xray.data import make_loader
from xray.models import build_model, pick_device

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="default: 1e-3 for cnn, 3e-4 for resnet18")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    lr = args.lr or (1e-3 if args.model == "cnn" else 3e-4)
    out_dir = RUNS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader("train", args.batch_size, shuffle=True, augment=True)
    val_loader = make_loader("val", args.batch_size)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters, device {device}")
    history = []
    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_logits, val_labels = predict_logits(model, val_loader, device)
        val_loss = loss_fn(torch.from_numpy(val_logits), torch.from_numpy(val_labels)).item()
        val_auc = roc_auc_score(val_labels, val_logits)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})

        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": args.model, "state_dict": model.state_dict()}, out_dir / "best.pt")
            marker = "  *"
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val AUC {val_auc:.4f}  ({time.time() - start:.0f}s){marker}")

    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"best val AUC {best_auc:.4f}, saved to {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
