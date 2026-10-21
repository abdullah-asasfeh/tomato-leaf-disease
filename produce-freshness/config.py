"""
Shared settings for the produce freshness app (v1: tomato, Fresh / Rotten).

Every script imports from here — prepare_data.py, train.py, evaluate.py, app.py —
so classes, image size, paths and the model shape are defined once.
Change this file only through a PR (issue #4).

Run `python produce-freshness/config.py` to print the settings.
"""
from pathlib import Path

# ---------------------------------------------------------------
# Classes (order = model output order; never reorder after training)
# ---------------------------------------------------------------
CLASSES = ["fresh", "rotten"]

# ---------------------------------------------------------------
# Images (same preprocessing as the leaf model)
# ---------------------------------------------------------------
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet
STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------
# Paths (relative to this file, so scripts work from any folder)
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                   # gitignored: never commit images
RAW_DIR = DATA_DIR / "raw"                 # downloaded datasets, as-is
SPLITS_DIR = DATA_DIR / "splits"           # splits/{train,val,test}/{class}/
STORE_TEST_DIR = DATA_DIR / "store_test"   # store_test/{class}/ — testing only, never training
SPLITS = ["train", "val", "test"]
REPORTS_DIR = ROOT / "reports"             # gitignored: plots, Grad-CAM images, history
MODEL_PATH = ROOT / "fresh_model.pt"

SEED = 42

# ---------------------------------------------------------------
# App: when to answer "not sure" (same rule as the leaf app)
# ---------------------------------------------------------------
THRESHOLD = 0.60   # minimum confidence
MARGIN = 0.20      # minimum gap between the top two classes


# ---------------------------------------------------------------
# Model: one definition, used by train / evaluate / app
# ---------------------------------------------------------------
def build_model(pretrained=False):
    """EfficientNet-B0 with a len(CLASSES)-way head.

    pretrained=True  -> ImageNet weights (for training)
    pretrained=False -> empty model (load MODEL_PATH into it for evaluation / the app)
    """
    from torch import nn
    from torchvision import models

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


# ---------------------------------------------------------------
# Preprocessing: one definition for train.py's validation, evaluate.py and app.py
# (train-time augmentation stays in train.py)
# ---------------------------------------------------------------
def eval_transform():
    """PIL RGB image -> normalised tensor of shape (3, IMG_SIZE, IMG_SIZE).

    Squashes to IMG_SIZE x IMG_SIZE (no crop), same as the leaf app.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


if __name__ == "__main__":
    for name in ["CLASSES", "IMG_SIZE", "MEAN", "STD", "SEED", "THRESHOLD", "MARGIN",
                 "DATA_DIR", "SPLITS_DIR", "STORE_TEST_DIR", "REPORTS_DIR", "MODEL_PATH"]:
        print(f"{name:15s} {globals()[name]}")
