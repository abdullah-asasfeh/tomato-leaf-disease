"""
Fine-tune EfficientNet-B0 to classify a tomato photo as fresh or rotten.

Reads the train/val splits made by prepare_data.py, keeps the epoch with the
best validation accuracy, and saves its state_dict to MODEL_PATH (same format
as the leaf model, so the app loads it the same way).

    python produce-freshness/train.py
    python produce-freshness/train.py --epochs 3 --data-dir <folder> --out <file>   # quick test
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------
# Shared settings.
# TEMPORARY copy of the agreed values until Salem's config.py is merged.
# Before opening the PR, replace this block with:
#     from config import CLASSES, IMG_SIZE, MEAN, STD, SPLITS_DIR, MODEL_PATH
# ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CLASSES = ["fresh", "rotten"]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet
STD = [0.229, 0.224, 0.225]
SPLITS_DIR = HERE / "data" / "splits"
MODEL_PATH = HERE / "fresh_model.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=SPLITS_DIR,
                   help="folder with train/ and val/ (default: %(default)s)")
    p.add_argument("--out", type=Path, default=MODEL_PATH,
                   help="where to save the best model (default: %(default)s)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Light augmentation on train only. No hue/saturation changes: colour is one of
# the main signs of rot, so we must not teach the model to ignore it.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Same preprocessing the app uses: squash to IMG_SIZE x IMG_SIZE, no crop.
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_split(root, name, tf):
    ds = datasets.ImageFolder(root / name, transform=tf)
    # ImageFolder numbers classes alphabetically; make sure that matches CLASSES,
    # otherwise "fresh" and "rotten" would be silently swapped.
    if ds.classes != CLASSES:
        raise SystemExit(f"{root / name}: found classes {ds.classes}, expected {CLASSES}")
    return ds


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, otherwise only evaluates.
    Returns (mean loss, accuracy, per-class recall)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = torch.zeros(len(CLASSES))
    seen = torch.zeros(len(CLASSES))

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            pred = out.argmax(1)
            for c in range(len(CLASSES)):
                mask = y == c
                seen[c] += mask.sum().item()
                correct[c] += (pred[mask] == c).sum().item()

    n = seen.sum().item()
    recall = (correct / seen.clamp(min=1)).tolist()
    return total_loss / n, correct.sum().item() / n, recall


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_split(args.data_dir, "train", train_tf)
    val_ds = load_split(args.data_dir, "val", eval_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    # The data has roughly twice as many fresh photos as rotten ones. Weight the
    # loss by inverse class frequency so "always say fresh" isn't an easy win.
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(CLASSES)).float()
    if (counts == 0).any():
        raise SystemExit(f"Empty class in train split: {dict(zip(CLASSES, counts.tolist()))}")
    class_weights = counts.sum() / (len(CLASSES) * counts)

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} images {dict(zip(CLASSES, counts.int().tolist()))}")
    print(f"Val:   {len(val_ds)} images")
    print(f"Class weights: {dict(zip(CLASSES, [round(w, 2) for w in class_weights.tolist()]))}")

    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    recall_hdr = "  ".join(f"{c[:6]:>6}" for c in CLASSES)
    print(f"\n{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  "
          f"{'val_acc':>7}  {recall_hdr}  {'time':>5}")

    best_acc, best_epoch = -1.0, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_dl, loss_fn, device, optimizer)
        va_loss, va_acc, va_recall = run_epoch(model, val_dl, loss_fn, device)
        scheduler.step()

        improved = va_acc > best_acc
        if improved:
            best_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), args.out)

        recalls = "  ".join(f"{r:6.1%}" for r in va_recall)
        print(f"{epoch:5d}  {tr_loss:10.4f}  {tr_acc:9.1%}  {va_loss:8.4f}  {va_acc:7.1%}  "
              f"{recalls}  {time.time() - t0:4.0f}s{'  *saved' if improved else ''}")

    print(f"\nBest val accuracy {best_acc:.1%} at epoch {best_epoch}. Saved to {args.out}")
    print("Per-class numbers above are val recall (share of each class predicted correctly).")


if __name__ == "__main__":
    main()
