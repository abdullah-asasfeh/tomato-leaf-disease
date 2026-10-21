"""
Fine-tune EfficientNet-B0 to classify a tomato photo as fresh or rotten.

Reads the train/val splits made by prepare_data.py, keeps the epoch with the
best validation accuracy, and saves its state_dict to config.MODEL_PATH (same
format as the leaf model, so the app loads it the same way). The per-epoch
history goes to config.REPORTS_DIR/train_history.csv.

    python produce-freshness/train.py
    python produce-freshness/train.py --epochs 3 --data-dir <folder> --out <file>   # quick test
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config

HISTORY_PATH = config.REPORTS_DIR / "train_history.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=config.SPLITS_DIR,
                   help="folder with train/ and val/ (default: %(default)s)")
    p.add_argument("--out", type=Path, default=config.MODEL_PATH,
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
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])


def load_split(root, name, tf):
    ds = datasets.ImageFolder(root / name, transform=tf)
    # ImageFolder numbers classes alphabetically; make sure that matches CLASSES,
    # otherwise "fresh" and "rotten" would be silently swapped.
    if ds.classes != config.CLASSES:
        raise SystemExit(f"{root / name}: found classes {ds.classes}, expected {config.CLASSES}")
    return ds


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, otherwise only evaluates.
    Returns (mean loss, accuracy, per-class recall)."""
    training = optimizer is not None
    model.train(training)
    n_classes = len(config.CLASSES)
    total_loss = 0.0
    correct = torch.zeros(n_classes)
    seen = torch.zeros(n_classes)

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
            for c in range(n_classes):
                mask = y == c
                seen[c] += mask.sum().item()
                correct[c] += (pred[mask] == c).sum().item()

    n = seen.sum().item()
    recall = (correct / seen.clamp(min=1)).tolist()
    return total_loss / n, correct.sum().item() / n, recall


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config.CLASSES

    train_ds = load_split(args.data_dir, "train", train_tf)
    val_ds = load_split(args.data_dir, "val", config.eval_transform())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    # The data has roughly twice as many fresh photos as rotten ones. Weight the
    # loss by inverse class frequency so "always say fresh" isn't an easy win.
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(classes)).float()
    if (counts == 0).any():
        raise SystemExit(f"Empty class in train split: {dict(zip(classes, counts.tolist()))}")
    class_weights = counts.sum() / (len(classes) * counts)

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} images {dict(zip(classes, counts.int().tolist()))}")
    print(f"Val:   {len(val_ds)} images")
    print(f"Class weights: {dict(zip(classes, [round(w, 2) for w in class_weights.tolist()]))}")

    model = config.build_model(pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    recall_hdr = "  ".join(f"{c[:6]:>6}" for c in classes)
    print(f"\n{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  "
          f"{'val_acc':>7}  {recall_hdr}  {'time':>5}")

    best_acc, best_epoch = -1.0, 0
    history = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_dl, loss_fn, device, optimizer)
        va_loss, va_acc, va_recall = run_epoch(model, val_dl, loss_fn, device)
        scheduler.step()
        seconds = time.time() - t0

        improved = va_acc > best_acc
        if improved:
            best_acc, best_epoch = va_acc, epoch
            # Save CPU tensors so a GPU-trained model loads in the CPU-only app.
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out)

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc,
                        **{f"val_recall_{c}": r for c, r in zip(classes, va_recall)},
                        "seconds": round(seconds, 1), "saved": improved})

        recalls = "  ".join(f"{r:6.1%}" for r in va_recall)
        print(f"{epoch:5d}  {tr_loss:10.4f}  {tr_acc:9.1%}  {va_loss:8.4f}  {va_acc:7.1%}  "
              f"{recalls}  {seconds:4.0f}s{'  *saved' if improved else ''}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print(f"\nBest val accuracy {best_acc:.1%} at epoch {best_epoch}. Saved to {args.out}")
    print(f"History saved to {HISTORY_PATH}")
    print("Per-class numbers above are val recall (share of each class predicted correctly).")


if __name__ == "__main__":
    main()
