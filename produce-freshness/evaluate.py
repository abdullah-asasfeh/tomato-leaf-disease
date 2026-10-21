"""
Evaluate the fresh/rotten model on the test split and, separately, on our own
store photos (the real-world score, like PlantDoc was for the leaf model).

For each set: accuracy, per-class precision / recall, confusion matrix, how often
the app's "not sure" rule triggers and how accurate the answers it does give are,
plus a few Grad-CAM examples of right and wrong answers. Plots go to
config.REPORTS_DIR (gitignored); a Markdown summary is printed for the README.

    python produce-freshness/evaluate.py
    python produce-freshness/evaluate.py --model <file> --test-dir <folder> --store-dir <folder>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write image files only, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageOps

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def open_image(path):
    """Open a photo upright, the way the app does: phone photos often store their
    rotation in an EXIF tag instead of rotating the pixels."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=config.MODEL_PATH,
                   help="trained state_dict (default: %(default)s)")
    p.add_argument("--test-dir", type=Path, default=config.SPLITS_DIR / "test",
                   help="test split, {class}/ subfolders (default: %(default)s)")
    p.add_argument("--store-dir", type=Path, default=config.STORE_TEST_DIR,
                   help="our store photos, {class}/ subfolders (default: %(default)s)")
    p.add_argument("--gradcam", type=int, default=4,
                   help="Grad-CAM examples per set for right and for wrong answers (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def list_images(root):
    """[(path, class index)] for root/{class}/*. Missing class folders count as empty."""
    items = []
    for i, c in enumerate(config.CLASSES):
        folder = root / c
        if folder.is_dir():
            items += [(p, i) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if root.is_dir():
        extra = [d.name for d in root.iterdir() if d.is_dir() and d.name not in config.CLASSES]
        if extra:
            print(f"  warning: {root} has folders that are not classes, ignored: {extra}")
    return items


@torch.no_grad()
def predict(model, items, batch_size):
    """Softmax probabilities, shape (N, n_classes)."""
    tf = config.eval_transform()
    probs = []
    for start in range(0, len(items), batch_size):
        batch = [tf(open_image(p)) for p, _ in items[start:start + batch_size]]
        probs.append(F.softmax(model(torch.stack(batch)), dim=1))
    return torch.cat(probs).numpy()


def not_sure(probs):
    """The app's rule: answer "not sure" when the top class is below THRESHOLD
    or less than MARGIN ahead of the second one."""
    top2 = np.sort(probs, axis=1)[:, -2:]
    return (top2[:, 1] < config.THRESHOLD) | (top2[:, 1] - top2[:, 0] < config.MARGIN)


def metrics(y, pred, unsure):
    n = len(config.CLASSES)
    cm = np.zeros((n, n), dtype=int)          # rows = true class, columns = predicted
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(config.CLASSES):
        tp = cm[i, i]
        per_class[c] = {
            "precision": tp / cm[:, i].sum() if cm[:, i].sum() else float("nan"),
            "recall": tp / cm[i, :].sum() if cm[i, :].sum() else float("nan"),
            "n": int(cm[i, :].sum()),
        }
    answered = ~unsure
    return {
        "n": len(y),
        "accuracy": float((pred == y).mean()),
        "per_class": per_class,
        "cm": cm,
        "not_sure_rate": float(unsure.mean()),
        "answered_accuracy": float((pred[answered] == y[answered]).mean()) if answered.any() else float("nan"),
    }


def gradcam(model, img):
    """Heatmap (IMG_SIZE x IMG_SIZE, 0..1) for the predicted class, same method as the leaf app."""
    saved = {}

    def hook(module, inp, out):
        saved["act"] = out
        out.register_hook(lambda g: saved.__setitem__("grad", g))

    handle = model.features[-1].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            out = model(config.eval_transform()(img).unsqueeze(0))
            model.zero_grad()
            out[0, int(out.argmax())].backward()
    finally:
        handle.remove()
    weights = saved["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * saved["act"]).sum(dim=1))[0]
    cam = cam / (cam.max() + 1e-8)
    return F.interpolate(cam[None, None], size=(config.IMG_SIZE, config.IMG_SIZE),
                         mode="bilinear")[0, 0].detach().numpy()


def overlay(img, cam):
    base = img.resize((config.IMG_SIZE, config.IMG_SIZE))
    heat = (colormaps["jet"](cam)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(heat), alpha=0.45)


def save_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ticks = range(len(config.CLASSES))
    ax.set_xticks(ticks, config.CLASSES)
    ax.set_yticks(ticks, config.CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_gradcam_examples(model, items, y, pred, probs, unsure, k, title, path):
    """Top row: correct answers (spread across confidence). Bottom row: the most
    confident wrong answers, which are the most useful to look at.
    Titles mark the ones the app would answer with "not sure"."""
    right = np.flatnonzero(pred == y)
    wrong = np.flatnonzero(pred != y)
    right = right[np.linspace(0, len(right) - 1, min(k, len(right))).astype(int)] if len(right) else right
    wrong = wrong[np.argsort(-probs[wrong].max(axis=1))][:k]
    rows = [("right", right), ("wrong", wrong)]
    cols = max(len(right), len(wrong), 1)

    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 6.4), squeeze=False,
                             layout="constrained")
    for r, (label, idx) in enumerate(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(idx):
                if c == 0:
                    ax.text(0.5, 0.5, f"no {label} answers", ha="center", va="center")
                continue
            i = idx[c]
            img = open_image(items[i][0])
            ax.imshow(overlay(img, gradcam(model, img)))
            ax.set_title(f"{label}: {config.CLASSES[y[i]]}\n-> {config.CLASSES[pred[i]]} "
                         f"({probs[i].max():.0%}){', not sure' if unsure[i] else ''}",
                         fontsize=9, color="green" if label == "right" else "red")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_set(model, name, root, args):
    print(f"\n[{name}] {root}")
    items = list_images(root)
    if not items:
        print("  no images found, skipped")
        return None
    y = np.array([i for _, i in items])
    probs = predict(model, items, args.batch_size)
    pred = probs.argmax(axis=1)
    unsure = not_sure(probs)
    m = metrics(y, pred, unsure)

    slug = name.lower().replace(" ", "_")
    save_confusion(m["cm"], name, config.REPORTS_DIR / f"confusion_{slug}.png")
    save_gradcam_examples(model, items, y, pred, probs, unsure, args.gradcam, name,
                          config.REPORTS_DIR / f"gradcam_{slug}.png")

    print(f"  {m['n']} images, accuracy {pct(m['accuracy'])}, "
          f"not sure {pct(m['not_sure_rate'])}, accuracy when answering {pct(m['answered_accuracy'])}")
    print(f"  confusion matrix (rows = true, columns = predicted, order {config.CLASSES}):")
    for c, row in zip(config.CLASSES, m["cm"]):
        print(f"    {c:>8}  {row.tolist()}")
    return m


def pct(x):
    return "n/a" if x != x else f"{x:.1%}"   # x != x is True only for NaN


def summary(results, model_path):
    """Markdown table, one column per set, ready to paste into the README."""
    sets = [(n, m) for n, m in results.items()]
    head = "| | " + " | ".join(n for n, _ in sets) + " |"
    rows = [head, "|---|" + "---|" * len(sets)]

    def row(label, f):
        rows.append(f"| {label} | " + " | ".join("not available yet" if m is None else f(m)
                                                for _, m in sets) + " |")

    row("Images", lambda m: " / ".join(f"{m['per_class'][c]['n']} {c}" for c in config.CLASSES))
    row("Accuracy", lambda m: pct(m["accuracy"]))
    for c in config.CLASSES:
        row(f"{c.capitalize()}: precision / recall",
            lambda m, c=c: f"{pct(m['per_class'][c]['precision'])} / {pct(m['per_class'][c]['recall'])}")
    row('"Not sure" rate', lambda m: pct(m["not_sure_rate"]))
    row("Accuracy when it answers", lambda m: pct(m["answered_accuracy"]))
    return (f"Model: `{model_path.name}`, \"not sure\" when confidence < {config.THRESHOLD:.0%} "
            f"or the top two are within {config.MARGIN:.0%}\n\n" + "\n".join(rows) +
            "\n\nStore photos were taken by us and never used for training, so they are the "
            "real-world score.")


def main():
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model} (run train.py first, or pass --model)")
    model = config.build_model(pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "Test split": evaluate_set(model, "Test split", args.test_dir, args),
        "Store photos": evaluate_set(model, "Store photos", args.store_dir, args),
    }
    text = summary(results, args.model)
    (config.REPORTS_DIR / "summary.md").write_text(text + "\n", encoding="utf-8")

    print(f"\nPlots saved to {config.REPORTS_DIR}")
    print("\n----- summary (Markdown) -----\n")
    print(text)


if __name__ == "__main__":
    main()
