"""
Download the Khatun tomato quality dataset, check it, and split it into
train / val / test folders for train.py and evaluate.py.

Source: Khatun et al. (2023), "Tomato Maturity Detection and Quality Grading Dataset",
Mendeley Data, doi:10.17632/s42kpg8h37.1, CC BY 4.0 (credit required). We use the
Hugging Face copy (Project-AgML/tomato_quality_classification), "raw" config only:
the 1,986 original photos, not the pre-augmented 6,000.

Steps:
  1. Download the raw parquet file (~150 MB) into config.RAW_DIR, once.
  2. Drop unreadable and tiny images, and exact duplicates.
  3. Group near-duplicates (perceptual hash) so a group never spans train and test.
     Groups that mix fresh and rotten are dropped as unreliable labels.
  4. Stratified 70 / 15 / 15 split by group, fixed seed (config.SEED).
  5. Save upright RGB JPEGs (EXIF rotation applied, metadata removed) to
     config.SPLITS_DIR/{train,val,test}/{fresh,rotten}/, and a manifest to splits.csv.

Running it again gives the identical split. Needs pyarrow (requirements-train.txt).

    python produce-freshness/prepare_data.py
"""
import argparse
import csv
import hashlib
import io
import random
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config

# Pinned to a dataset revision so the download never changes under us.
REVISION = "c4847c689b7289817ddc2fe2e34510031ada03ce"
URL = ("https://huggingface.co/datasets/Project-AgML/tomato_quality_classification/"
       f"resolve/{REVISION}/raw/train-00000-of-00001.parquet")
EXPECTED_BYTES = 149_944_018
RAW_FILE = config.RAW_DIR / "khatun_raw.parquet"
SOURCE_LABELS = {0: "fresh", 1: "rotten"}   # from the dataset card: 0 = Fresh, 1 = Rotten

MANIFEST = config.ROOT / "splits.csv"       # small, committed: lets us check we have the same split.
                                            # Only a default run writes it; test runs write theirs into --out.
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_SIDE = 64          # smaller images are dropped
MAX_SIDE = 1024        # larger images are shrunk on save (training uses 224 anyway)
HASH_SIZE = 16         # dHash grid: 16 x 16 = 256 bits
NEAR_DUP_BITS = 24     # distance (out of 256) at or below which two photos count as near-duplicates.
                       # Picked by eye on this dataset: <= 24 is the same scene with a tomato moved
                       # or removed; >= 27 is different tomatoes. 64-bit hashes were too coarse:
                       # similar-looking photos chained into one group of 1,000+.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-file", type=Path, default=RAW_FILE,
                   help="parquet file to use instead of downloading (default: %(default)s)")
    p.add_argument("--out", type=Path, default=config.SPLITS_DIR,
                   help="output folder, replaced on every run (default: %(default)s)")
    return p.parse_args()


def download(path):
    if path.is_file() and path.stat().st_size == EXPECTED_BYTES:
        print(f"Using downloaded file {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading {URL}\n  -> {path} ({EXPECTED_BYTES / 1e6:.0f} MB)")
    with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:6.1f} MB", end="", flush=True)
    print()
    if tmp.stat().st_size != EXPECTED_BYTES:
        raise SystemExit(f"Download incomplete: got {tmp.stat().st_size} bytes, expected {EXPECTED_BYTES}")
    tmp.replace(path)


def read_rows(path):
    """[(image bytes, class name)] in file order."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["image", "label"])
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    return [(img["bytes"], SOURCE_LABELS[lab]) for img, lab in zip(images, labels)]


def dhash(img):
    """256-bit difference hash as 32 packed bytes: survives resizing and re-compression,
    so it finds the same photo saved twice, or the same scene shot again."""
    g = np.asarray(img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def distances(h, hashes):
    """Bit distance from one hash to each row of hashes."""
    return np.unpackbits(h ^ hashes, axis=1).sum(axis=1)


def load_images(rows):
    """Decode and check every image. Returns kept items and counts of what was dropped."""
    kept, dropped = [], {"unreadable": 0, "too small": 0, "exact duplicate": 0}
    seen = set()
    for idx, (data, cls) in enumerate(rows):
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception:
            dropped["unreadable"] += 1
            continue
        if min(img.size) < MIN_SIDE:
            dropped["too small"] += 1
            continue
        digest = hashlib.sha256(img.tobytes()).hexdigest()
        if digest in seen:
            dropped["exact duplicate"] += 1
            continue
        seen.add(digest)
        kept.append({"source_index": idx, "class": cls, "img": img, "dhash": dhash(img)})
    return kept, dropped


def group_near_duplicates(items):
    """Union-find over pairs whose dHash differs in <= NEAR_DUP_BITS bits.
    Sets item["group"] to the smallest member index of its group."""
    hashes = np.stack([it["dhash"] for it in items])
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in np.flatnonzero(distances(hashes[i], hashes[i + 1:]) <= NEAR_DUP_BITS) + i + 1:
            a, b = find(i), find(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    for i, it in enumerate(items):
        it["group"] = find(i)


def split_groups(items):
    """Assign whole groups to train/val/test, separately per class, so each split
    keeps the fresh/rotten ratio. Mixed-label groups are dropped."""
    groups = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it)
    mixed = [g for g, members in groups.items() if len({m["class"] for m in members}) > 1]
    n_mixed = sum(len(groups[g]) for g in mixed)
    for g in mixed:
        del groups[g]

    rng = random.Random(config.SEED)
    for cls in config.CLASSES:
        ids = sorted(g for g, members in groups.items() if members[0]["class"] == cls)
        rng.shuffle(ids)
        total = sum(len(groups[g]) for g in ids)
        filled = 0
        for g in ids:
            # Put the group in the first split that still has room, by image count.
            share = filled / total
            split = "train" if share < FRACTIONS["train"] else \
                    "val" if share < FRACTIONS["train"] + FRACTIONS["val"] else "test"
            for m in groups[g]:
                m["split"] = split
            filled += len(groups[g])
    kept = [m for members in groups.values() for m in members]
    return kept, n_mixed


def clear_output(out):
    """Delete a previous split before writing a new one, but only if the folder holds
    nothing except our split folders (and our manifest), so a wrong --out can't wipe
    anything else."""
    if not out.exists():
        return
    ours = set(config.SPLITS) | {"splits.csv"}
    other = sorted(c.name for c in out.iterdir() if c.name not in ours)
    if other:
        raise SystemExit(f"Refusing to delete {out}: it contains {other}, which this script "
                         f"didn't create. Pick an empty folder for --out.")
    shutil.rmtree(out)


def save(items, out, manifest):
    clear_output(out)
    for split in config.SPLITS:
        for cls in config.CLASSES:
            (out / split / cls).mkdir(parents=True, exist_ok=True)
    rows = []
    for it in sorted(items, key=lambda m: m["source_index"]):
        img = it["img"]
        if max(img.size) > MAX_SIDE:
            img = img.copy()
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        name = f"khatun_{it['source_index']:04d}.jpg"
        img.save(out / it["split"] / it["class"] / name, "JPEG", quality=95)   # no EXIF written
        rows.append({"file": f"{it['split']}/{it['class']}/{name}", "source_index": it["source_index"],
                     "class": it["class"], "split": it["split"], "group": it["group"],
                     "dhash": it["dhash"].tobytes().hex()})
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_no_leak(items):
    """No near-duplicate pair may sit in two different splits."""
    by_split = {s: [it["dhash"] for it in items if it["split"] == s] for s in config.SPLITS}
    leaks = 0
    for a in config.SPLITS:
        for b in config.SPLITS:
            if a >= b or not len(by_split[a]) or not len(by_split[b]):
                continue
            other = np.stack(by_split[b])
            for h in by_split[a]:
                leaks += int((distances(h, other) <= NEAR_DUP_BITS).sum())
    return leaks


def main():
    args = parse_args()
    if args.raw_file == RAW_FILE:
        download(RAW_FILE)
    rows = read_rows(args.raw_file)
    print(f"Read {len(rows)} images "
          f"({', '.join(f'{sum(1 for _, c in rows if c == k)} {k}' for k in config.CLASSES)})")

    items, dropped = load_images(rows)
    group_near_duplicates(items)
    n_groups = len({it["group"] for it in items})
    items, n_mixed = split_groups(items)
    dropped["near-duplicates with mixed labels"] = n_mixed
    default_run = args.raw_file == RAW_FILE and args.out == config.SPLITS_DIR
    manifest = MANIFEST if default_run else args.out / "splits.csv"
    save(items, args.out, manifest)
    leaks = check_no_leak(items)

    print("Dropped: " + ", ".join(f"{v} {k}" for k, v in dropped.items()))
    print(f"Near-duplicate groups: {n_groups} groups for {len(items) + n_mixed} images "
          f"(each group kept in one split)")
    print(f"\n{'':8}" + "".join(f"{c:>8}" for c in config.CLASSES) + f"{'total':>8}")
    for split in config.SPLITS:
        counts = [sum(1 for it in items if it["split"] == split and it["class"] == c)
                  for c in config.CLASSES]
        print(f"{split:8}" + "".join(f"{n:8d}" for n in counts) + f"{sum(counts):8d}")
    print(f"\nNear-duplicate pairs across splits: {leaks}")
    print(f"Saved to {args.out}\nManifest: {manifest}")
    if leaks:
        raise SystemExit("Leak between splits: near-duplicates ended up in different splits")


if __name__ == "__main__":
    main()
