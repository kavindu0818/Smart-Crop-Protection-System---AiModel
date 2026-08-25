"""
generate_notebook.py
Run this once to produce Animal_CNN_Research.ipynb in the same directory.
    python generate_notebook.py
"""

import json, os

# ---------------------------------------------------------------------------
# Helper to create cells
# ---------------------------------------------------------------------------

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

def code(source: str, tags: list = None) -> dict:
    meta = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": source
    }

# ---------------------------------------------------------------------------
# Build cells list
# ---------------------------------------------------------------------------

cells = []

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# 🦁 Custom CNN Animal Classification Model
## Research Notebook — Developed from Scratch using TensorFlow / Keras

---

> **Author**: Research Project  
> **Purpose**: Academic Research — Animal Detection & Classification  
> **Architecture**: Custom CNN (No pretrained weights, No transfer learning)  
> **Framework**: TensorFlow · Keras · OpenCV · Scikit-learn  

---
"""))

# ══════════════════════════════════════════════════════════════════════════
# 01 Research Objective
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 01 · Research Objective

### Problem Statement
Automatic animal identification from images is a fundamental computer-vision challenge with wide applications in wildlife monitoring, smart-farming, and crop-protection systems. The goal of this research module is to design, train, and evaluate a **custom Convolutional Neural Network (CNN)** that can classify images of five animal species (bird, cow, elephant, monkey, pig) with high reliability.

### Research Constraints
| Constraint | Value |
|---|---|
| Pretrained models | ❌ Prohibited |
| Transfer learning / Fine-tuning | ❌ Prohibited |
| YOLO / MobileNet / ResNet / VGG / EfficientNet | ❌ Prohibited |
| Custom CNN from scratch | ✅ Required |
| Training data | User-provided dataset only |

### Research Contributions
1. Novel CNN architecture specifically designed for multi-species animal classification.  
2. Reproducible training pipeline with systematic hyperparameter tracking.  
3. Full evaluation suite: accuracy, precision, recall, F1, confusion matrix.  
4. OpenCV inference module for deployment on new, unseen images.  
"""))

# ══════════════════════════════════════════════════════════════════════════
# 02 Import Libraries
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 02 · Import Libraries"))
cells.append(code("""\
# ── Standard library ──────────────────────────────────────────────────────
import os
import sys
import json
import time
import hashlib
import datetime
import warnings
import random
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# ── Numerical & Visualisation ──────────────────────────────────────────────
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── Deep Learning ──────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.models import Model

# ── Image Processing ────────────────────────────────────────────────────────
import cv2
from PIL import Image

# ── Machine Learning Utilities ──────────────────────────────────────────────
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

keras_version = getattr(keras, "__version__", getattr(tf, "__version__", "unknown"))
print(f"Python        : {sys.version.split()[0]}")
print(f"TensorFlow    : {tf.__version__}")
print(f"Keras         : {keras_version}")
print(f"NumPy         : {np.__version__}")
print(f"OpenCV        : {cv2.__version__}")
print(f"Matplotlib    : {matplotlib.__version__}")
"""))

# ══════════════════════════════════════════════════════════════════════════
# 03 Configuration Class
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 03 · Configuration

All project-wide parameters are centralised in the `Config` class.  
To adapt this notebook to your own dataset, **only change the three directory paths** below.
"""))
cells.append(code("""\
class Config:
    \"\"\"
    Central configuration class.
    Change TRAIN_DIR, VALIDATION_DIR, TEST_DIR to point at your dataset.
    \"\"\"

    # ── Dataset Paths ──────────────────────────────────────────────────────
    TRAIN_DIR      = "dataset/train"
    VALIDATION_DIR = "dataset/validation"
    TEST_DIR       = "dataset/test"

    # ── Image Settings ─────────────────────────────────────────────────────
    IMG_SIZE       = (224, 224)      # (height, width)
    IMG_SHAPE      = (224, 224, 3)   # (H, W, C)
    NUM_CHANNELS   = 3

    # ── Training Hyperparameters ──────────────────────────────────────────
    BATCH_SIZE     = 32
    EPOCHS         = 60
    LEARNING_RATE  = 1e-3

    # ── Regularisation ─────────────────────────────────────────────────────
    DROPOUT_RATE   = 0.40
    L2_REG         = 1e-4

    # ── Callbacks ──────────────────────────────────────────────────────────
    EARLY_STOP_PATIENCE   = 12
    LR_REDUCE_FACTOR      = 0.50
    LR_REDUCE_PATIENCE    = 5
    LR_MIN                = 1e-6

    # ── Augmentation ───────────────────────────────────────────────────────
    AUG_ROTATION     = 0.15
    AUG_WIDTH_SHIFT  = 0.10
    AUG_HEIGHT_SHIFT = 0.10
    AUG_ZOOM         = 0.10
    AUG_BRIGHTNESS   = (-0.15, 0.15)   # delta range

    # ── Saving ─────────────────────────────────────────────────────────────
    MODEL_PATH       = "animal_cnn.keras"
    CLASS_NAMES_PATH = "class_names.json"
    RESULTS_DIR      = "results"

    # ── Prediction ─────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD = 0.60

    # ── Reproducibility ────────────────────────────────────────────────────
    RANDOM_SEED = 42

    @classmethod
    def apply_seeds(cls):
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        tf.random.set_seed(cls.RANDOM_SEED)
        os.environ["PYTHONHASHSEED"] = str(cls.RANDOM_SEED)

    @classmethod
    def make_dirs(cls):
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)

Config.apply_seeds()
Config.make_dirs()
print("✅ Configuration loaded and seeds set.")
print(f"   Train      → {Config.TRAIN_DIR}")
print(f"   Validation → {Config.VALIDATION_DIR}")
print(f"   Test       → {Config.TEST_DIR}")
print(f"   IMG_SIZE   → {Config.IMG_SIZE}")
print(f"   BATCH_SIZE → {Config.BATCH_SIZE}")
print(f"   EPOCHS     → {Config.EPOCHS}")
"""))

# ══════════════════════════════════════════════════════════════════════════
# 04 Dataset Manager
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 04 · Dataset Manager

The `DatasetManager` class handles all dataset-related operations:
- Directory validation
- Class detection from folder names
- Image counting per class
- Duplicate/leakage detection via MD5 hashing
- Dataset statistics reporting
"""))
cells.append(code("""\
class DatasetManager:
    \"\"\"
    Manages dataset validation, statistics, and leakage checking.
    Class labels are automatically inferred from sub-folder names.
    \"\"\"

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, cfg: Config):
        self.cfg         = cfg
        self.class_names = []
        self.num_classes = 0
        self._stats      = {}   # {split: {class: count}}
        self._hashes     = {}   # {split: {class: [hash, ...]}}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        \"\"\"Check that all three split directories exist and are non-empty.\"\"\"
        ok = True
        for name, path in [
            ("Train",      self.cfg.TRAIN_DIR),
            ("Validation", self.cfg.VALIDATION_DIR),
            ("Test",       self.cfg.TEST_DIR),
        ]:
            if not os.path.isdir(path):
                print(f"  ❌  {name} directory not found : {path}")
                ok = False
            else:
                folders = [e for e in os.scandir(path) if e.is_dir()]
                if not folders:
                    print(f"  ❌  {name} directory has no class sub-folders : {path}")
                    ok = False
                else:
                    print(f"  ✅  {name} directory OK ({len(folders)} class folders)")
        return ok

    # ------------------------------------------------------------------
    # Class detection
    # ------------------------------------------------------------------

    def detect_classes(self) -> list:
        \"\"\"Detect class names from training directory sub-folder names.\"\"\"
        self.class_names = sorted(
            e.name for e in os.scandir(self.cfg.TRAIN_DIR) if e.is_dir()
        )
        self.num_classes = len(self.class_names)
        return self.class_names

    # ------------------------------------------------------------------
    # Image counting
    # ------------------------------------------------------------------

    def _count_images(self, split_dir: str) -> dict:
        \"\"\"Return {class_name: image_count} for one split directory.\"\"\"
        counts = {}
        for cls in self.class_names:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                counts[cls] = sum(
                    1 for f in os.listdir(cls_dir)
                    if Path(f).suffix.lower() in self.SUPPORTED_EXTS
                )
            else:
                counts[cls] = 0
        return counts

    def compute_stats(self):
        \"\"\"Compute image counts for all splits.\"\"\"
        self._stats = {
            "train"      : self._count_images(self.cfg.TRAIN_DIR),
            "validation" : self._count_images(self.cfg.VALIDATION_DIR),
            "test"       : self._count_images(self.cfg.TEST_DIR),
        }

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    def print_summary(self):
        \"\"\"Print a formatted dataset summary.\"\"\"
        if not self._stats:
            self.compute_stats()

        total_train = sum(self._stats["train"].values())
        total_val   = sum(self._stats["validation"].values())
        total_test  = sum(self._stats["test"].values())
        total_all   = total_train + total_val + total_test

        w = max(len(c) for c in self.class_names) + 2

        print("\\n" + "=" * 55)
        print("  DATASET SUMMARY")
        print("=" * 55)
        print(f"  Classes detected : {self.num_classes}")
        print(f"  Class names      : {self.class_names}")
        print("-" * 55)

        for split_name, split_key in [
            ("Train Images",      "train"),
            ("Validation Images", "validation"),
            ("Test Images",       "test"),
        ]:
            total = sum(self._stats[split_key].values())
            print(f"\\n  {split_name} (Total: {total})")
            for cls, cnt in self._stats[split_key].items():
                bar = "█" * (cnt // max(1, total // 20))
                print(f"    {cls:<{w}}: {cnt:>5}  {bar}")

        print("-" * 55)
        print(f"  Grand Total      : {total_all}")
        print("=" * 55)

    # ------------------------------------------------------------------
    # Data leakage detection
    # ------------------------------------------------------------------

    def _hash_file(self, filepath: str) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _collect_hashes(self, split_dir: str) -> dict:
        \"\"\"Return {class: [md5, ...]} for one split.\"\"\"
        result = {}
        for cls in self.class_names:
            cls_dir = os.path.join(split_dir, cls)
            hashes = []
            if os.path.isdir(cls_dir):
                for fn in os.listdir(cls_dir):
                    fp = os.path.join(cls_dir, fn)
                    if Path(fn).suffix.lower() in self.SUPPORTED_EXTS and os.path.isfile(fp):
                        try:
                            hashes.append(self._hash_file(fp))
                        except Exception:
                            pass
            result[cls] = hashes
        return result

    def check_leakage(self):
        \"\"\"
        Compare MD5 hashes across train / validation / test.
        Report any identical files (potential data leakage).
        DO NOT delete any files.
        \"\"\"
        print("\\n[Leakage Check] Computing MD5 hashes …")
        train_h = self._collect_hashes(self.cfg.TRAIN_DIR)
        val_h   = self._collect_hashes(self.cfg.VALIDATION_DIR)
        test_h  = self._collect_hashes(self.cfg.TEST_DIR)

        train_set = set(h for v in train_h.values() for h in v)
        val_set   = set(h for v in val_h.values()   for h in v)
        test_set  = set(h for v in test_h.values()  for h in v)

        tv_overlap = train_set & val_set
        tt_overlap = train_set & test_set
        vt_overlap = val_set   & test_set

        print(f"  Train ∩ Validation duplicates : {len(tv_overlap)}")
        print(f"  Train ∩ Test duplicates        : {len(tt_overlap)}")
        print(f"  Validation ∩ Test duplicates   : {len(vt_overlap)}")

        if not (tv_overlap or tt_overlap or vt_overlap):
            print("  ✅ No data leakage detected.")
        else:
            print("  ⚠️  Possible data leakage detected — review the duplicates above.")
            print("      No files were modified. Please remove duplicates manually.")

    # ------------------------------------------------------------------
    # Class-imbalance analysis
    # ------------------------------------------------------------------

    def imbalance_report(self):
        \"\"\"Report and visualise class imbalance in the training split.\"\"\"
        if not self._stats:
            self.compute_stats()
        counts = self._stats["train"]
        total  = sum(counts.values())
        mx     = max(counts.values())
        mn     = min(counts.values())
        ratio  = mx / max(mn, 1)

        print("\\n[Imbalance Report — Training Split]")
        for cls, cnt in counts.items():
            pct = cnt / total * 100
            print(f"  {cls:<20} : {cnt:>5} ({pct:.1f}%)")
        print(f"  Max/Min ratio : {ratio:.2f}x", end="  ")
        if ratio > 3:
            print("  ⚠️  Significant imbalance — class weights will be applied.")
        elif ratio > 1.5:
            print("  ⚠️  Mild imbalance — monitor per-class performance.")
        else:
            print("  ✅ Balanced")

    # ------------------------------------------------------------------
    # Compute class weights for imbalanced datasets
    # ------------------------------------------------------------------

    def compute_class_weights(self) -> dict:
        \"\"\"
        Compute balanced class weights using the inverse-frequency formula.
        Returns {class_index: weight} dict suitable for model.fit(class_weight=...).
        \"\"\"
        if not self._stats:
            self.compute_stats()
        counts = [self._stats["train"].get(cls, 1) for cls in self.class_names]
        total  = sum(counts)
        n_cls  = len(counts)
        weights = {
            i: total / (n_cls * max(cnt, 1))
            for i, cnt in enumerate(counts)
        }
        return weights
"""))

cells.append(code("""\
# ── Run dataset validation & stats ───────────────────────────────────────
dm = DatasetManager(Config)

print("\\n[Step 1] Validating dataset directories …")
if not dm.validate():
    print("\\n⚠️  Fix the directory issues above before continuing.")
else:
    print("\\n[Step 2] Detecting classes …")
    class_names = dm.detect_classes()
    print(f"  Classes: {class_names}")

    print("\\n[Step 3] Computing dataset statistics …")
    dm.compute_stats()
    dm.print_summary()
"""))

# ══════════════════════════════════════════════════════════════════════════
# 05 Data Leakage Check
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 05 · Data Leakage Check"))
cells.append(code("""\
dm.check_leakage()
"""))

# ══════════════════════════════════════════════════════════════════════════
# 06 Class Imbalance Report
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 06 · Class Imbalance Analysis"))
cells.append(code("""\
dm.imbalance_report()
class_weights = dm.compute_class_weights()
print(f"\\nClass weights: {class_weights}")
"""))

# ══════════════════════════════════════════════════════════════════════════
# 07 Dataset Visualisation
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 07 · Dataset Visualisation

Display representative images from each class for dataset inspection.
"""))
cells.append(code("""\
def visualise_dataset_samples(cfg, class_names, n_per_class=4):
    \"\"\"Display n_per_class sample images for each animal class.\"\"\"
    n_cls = len(class_names)
    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(n_per_class * 3, n_cls * 3))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Dataset Samples per Class", color="white", fontsize=16, fontweight="bold", y=1.01)

    for row, cls in enumerate(class_names):
        cls_dir = os.path.join(cfg.TRAIN_DIR, cls)
        imgs    = [
            f for f in os.listdir(cls_dir)
            if Path(f).suffix.lower() in DatasetManager.SUPPORTED_EXTS
        ] if os.path.isdir(cls_dir) else []

        for col in range(n_per_class):
            ax = axes[row][col] if n_cls > 1 else axes[col]
            ax.set_facecolor("#161b22")
            if col < len(imgs):
                fp  = os.path.join(cls_dir, imgs[col])
                img = cv2.imread(fp)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]
                    ax.imshow(img)
                    if col == 0:
                        ax.set_ylabel(cls.capitalize(), color="#58a6ff",
                                      fontsize=11, fontweight="bold", rotation=0,
                                      labelpad=50, va="center")
                    ax.set_title(f"{w}×{h}", color="#8b949e", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Load Error", ha="center", va="center", color="red")
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", color="#8b949e")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "dataset_samples.png"),
                dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("Dataset samples figure saved.")

visualise_dataset_samples(Config, class_names)
"""))

cells.append(code("""\
def plot_class_distribution(dm):
    \"\"\"Bar chart of image counts per class across all splits.\"\"\"
    splits = ["train", "validation", "test"]
    colors = ["#238636", "#1f6feb", "#da3633"]
    x      = np.arange(len(dm.class_names))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for i, (split, color) in enumerate(zip(splits, colors)):
        counts = [dm._stats[split].get(cls, 0) for cls in dm.class_names]
        bars   = ax.bar(x + i * width, counts, width, label=split.capitalize(), color=color, alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(cnt), ha="center", va="bottom", color="white", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in dm.class_names], color="white", fontsize=11)
    ax.set_ylabel("Image Count", color="white")
    ax.set_title("Class Distribution across Splits", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#21262d", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="y", alpha=0.2, color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "class_distribution.png"),
                dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

plot_class_distribution(dm)
"""))

# ══════════════════════════════════════════════════════════════════════════
# 08 Image Preprocessor
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 08 · Image Preprocessor & Data Augmentation

### Preprocessing Pipeline
```
Image → Resize (224×224) → RGB conversion → Pixel normalisation [0,1] → Tensor
```

### Augmentation (Training Only)
Random flips, rotations, zoom, translation, and brightness jitter help the model 
generalise to diverse real-world conditions without overfitting to the training set.
"""))
cells.append(code("""\
class ImagePreprocessor:
    \"\"\"
    Builds tf.data preprocessing pipelines for each split.
    Augmentation is ONLY applied to the training split.
    Validation and Test splits receive only normalisation (no randomness).
    \"\"\"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Augmentation layer
    # ------------------------------------------------------------------

    def _augmentation_layer(self) -> keras.Sequential:
        \"\"\"Returns a Keras layer stack that applies random augmentations.\"\"\"
        lo, hi = self.cfg.AUG_BRIGHTNESS
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(self.cfg.AUG_ROTATION),
            layers.RandomTranslation(
                height_factor=self.cfg.AUG_HEIGHT_SHIFT,
                width_factor=self.cfg.AUG_WIDTH_SHIFT,
            ),
            layers.RandomZoom(self.cfg.AUG_ZOOM),
            layers.RandomBrightness(factor=(lo, hi)),
        ], name="augmentation")

    # ------------------------------------------------------------------
    # tf.data pipelines
    # ------------------------------------------------------------------

    def _make_raw_dataset(self, directory: str, class_names: list, shuffle: bool):
        \"\"\"Load image batches from directory using Keras utility.\"\"\"
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            color_mode="rgb",
            batch_size=self.cfg.BATCH_SIZE,
            image_size=self.cfg.IMG_SIZE,
            shuffle=shuffle,
            seed=self.cfg.RANDOM_SEED,
        )

    def build_train_dataset(self, class_names: list) -> tf.data.Dataset:
        normalise = layers.Rescaling(1.0 / 255.0)
        aug       = self._augmentation_layer()
        raw_ds    = self._make_raw_dataset(self.cfg.TRAIN_DIR, class_names, shuffle=True)

        def preprocess(imgs, lbls):
            imgs = normalise(imgs)
            imgs = aug(imgs, training=True)
            return imgs, lbls

        return (
            raw_ds
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    def build_eval_dataset(self, directory: str, class_names: list) -> tf.data.Dataset:
        \"\"\"Validation / Test — normalise only, no augmentation.\"\"\"
        normalise = layers.Rescaling(1.0 / 255.0)
        raw_ds    = self._make_raw_dataset(directory, class_names, shuffle=False)

        def preprocess(imgs, lbls):
            return normalise(imgs), lbls

        return (
            raw_ds
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

    # ------------------------------------------------------------------
    # OpenCV-compatible single-image preprocessor
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_single_image_opencv(image_bgr: np.ndarray, img_size: tuple) -> np.ndarray:
        \"\"\"
        Preprocess a single BGR OpenCV image for model inference.
        Must match the training normalisation exactly.
        Returns: float32 array of shape (1, H, W, 3)
        \"\"\"
        h, w    = img_size
        resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm    = rgb.astype(np.float32) / 255.0
        return np.expand_dims(norm, axis=0)
"""))

cells.append(code("""\
# ── Build datasets ────────────────────────────────────────────────────────
preprocessor = ImagePreprocessor(Config)

print("Building training dataset (with augmentation) …")
train_ds = preprocessor.build_train_dataset(class_names)

print("Building validation dataset …")
val_ds   = preprocessor.build_eval_dataset(Config.VALIDATION_DIR, class_names)

print("Building test dataset …")
test_ds  = preprocessor.build_eval_dataset(Config.TEST_DIR, class_names)

print("\\n✅ All three tf.data pipelines ready.")
"""))

# ══════════════════════════════════════════════════════════════════════════
# 09 Augmentation Preview
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 09 · Augmentation Preview"))
cells.append(code("""\
def preview_augmentation(cfg, class_names, n_aug=6):
    \"\"\"Show the same image before and after augmentation.\"\"\"
    aug_fn = ImagePreprocessor(cfg)._augmentation_layer()
    norm   = layers.Rescaling(1.0 / 255.0)

    # Pick the first image from the first class
    cls_dir = os.path.join(cfg.TRAIN_DIR, class_names[0])
    files   = [f for f in os.listdir(cls_dir)
               if Path(f).suffix.lower() in DatasetManager.SUPPORTED_EXTS]
    if not files:
        print("No images found for augmentation preview.")
        return

    fp    = os.path.join(cls_dir, files[0])
    img   = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, (cfg.IMG_SIZE[1], cfg.IMG_SIZE[0]))
    t     = tf.expand_dims(tf.cast(img_r, tf.float32) / 255.0, 0)

    fig, axes = plt.subplots(1, n_aug + 1, figsize=((n_aug + 1) * 3, 3.5))
    fig.patch.set_facecolor("#0d1117")

    axes[0].imshow(img_r)
    axes[0].set_title("Original", color="#58a6ff", fontsize=10)
    axes[0].axis("off")

    for i in range(n_aug):
        aug_t = aug_fn(t, training=True).numpy()[0]
        axes[i + 1].imshow(np.clip(aug_t, 0, 1))
        axes[i + 1].set_title(f"Aug {i+1}", color="#8b949e", fontsize=9)
        axes[i + 1].axis("off")

    fig.suptitle(f"Data Augmentation Preview — {class_names[0].capitalize()}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "augmentation_preview.png"),
                dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

preview_augmentation(Config, class_names)
"""))

# ══════════════════════════════════════════════════════════════════════════
# 10 Custom CNN Architecture
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 10 · Custom CNN Architecture

### Design Philosophy

The architecture follows a **hierarchical feature-learning** strategy:

| Stage | Layers | What the Model Learns |
|---|---|---|
| Block 1 (32 filters) | Conv → BN → ReLU → Conv → BN → ReLU → MaxPool | Low-level: edges, colour gradients, simple textures |
| Block 2 (64 filters) | Conv → BN → ReLU → Conv → BN → ReLU → MaxPool | Patterns: fur texture, feather patterns, skin details |
| Block 3 (128 filters) | Conv → BN → ReLU → Conv → BN → ReLU → MaxPool | Body parts: legs, trunks, tails, wings |
| Block 4 (256 filters) | Conv → BN → ReLU → Conv → BN → ReLU → MaxPool | Semantic: full animal body structure |
| Head | GAP → Dense(512) → Dropout → Softmax | Classification |

**Why two Conv layers per block?**  
Two consecutive 3×3 convolutions have the same effective receptive field as one 5×5 but with fewer parameters and an extra non-linearity — allowing richer feature extraction without over-parameterisation.

### Architecture Diagram
```
Input (224×224×3)
       ↓
 ┌─ Conv Block 1 ──────────────────────┐
 │  Conv(32,3×3) → BN → ReLU          │ → Spatial: 224×224
 │  Conv(32,3×3) → BN → ReLU          │
 │  MaxPool(2×2)                       │ → Spatial: 112×112
 └─────────────────────────────────────┘
       ↓
 ┌─ Conv Block 2 ──────────────────────┐
 │  Conv(64,3×3) → BN → ReLU          │
 │  Conv(64,3×3) → BN → ReLU          │
 │  MaxPool(2×2)                       │ → Spatial: 56×56
 └─────────────────────────────────────┘
       ↓
 ┌─ Conv Block 3 ──────────────────────┐
 │  Conv(128,3×3) → BN → ReLU         │
 │  Conv(128,3×3) → BN → ReLU         │
 │  MaxPool(2×2)                       │ → Spatial: 28×28
 └─────────────────────────────────────┘
       ↓
 ┌─ Conv Block 4 ──────────────────────┐
 │  Conv(256,3×3) → BN → ReLU         │
 │  Conv(256,3×3) → BN → ReLU         │
 │  MaxPool(2×2)                       │ → Spatial: 14×14
 └─────────────────────────────────────┘
       ↓
  Global Average Pooling → (256,)
       ↓
  Dense(512) + ReLU + L2
       ↓
  Dropout(0.40)
       ↓
  Dense(N) → Softmax   (N = number of classes)
```

**No pretrained weights. No transfer learning. All weights randomly initialised.**
"""))
cells.append(code("""\
class CustomCNN:
    \"\"\"
    Factory class that builds the custom CNN model from scratch.

    All weights are initialised using He Normal (suitable for ReLU-activated
    networks). There are no pretrained weights, no external backbones, and
    no transfer learning of any kind.
    \"\"\"

    def __init__(self, cfg: Config, num_classes: int):
        self.cfg         = cfg
        self.num_classes = num_classes
        self.model       = None

    # ------------------------------------------------------------------
    # Internal block builder
    # ------------------------------------------------------------------

    def _conv_block(self, x, filters: int, block_name: str):
        \"\"\"
        Two consecutive Conv2D layers with Batch Normalisation and ReLU,
        followed by MaxPool2D.
        \"\"\"
        kreg = regularizers.L2(self.cfg.L2_REG)

        # First conv layer
        x = layers.Conv2D(
            filters, (3, 3), padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=kreg,
            use_bias=False,
            name=f"{block_name}_conv1",
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
        x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

        # Second conv layer — doubles representational depth without new params overhead
        x = layers.Conv2D(
            filters, (3, 3), padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=kreg,
            use_bias=False,
            name=f"{block_name}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
        x = layers.Activation("relu", name=f"{block_name}_relu2")(x)

        # Spatial downsampling
        x = layers.MaxPool2D((2, 2), name=f"{block_name}_pool")(x)
        return x

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> keras.Model:
        \"\"\"Construct and return the uncompiled Keras model.\"\"\"
        inputs = keras.Input(shape=self.cfg.IMG_SHAPE, name="input_image")
        x = inputs

        # ── Feature Extraction ────────────────────────────────────────
        x = self._conv_block(x, filters=32,  block_name="block1")  # 112×112
        x = self._conv_block(x, filters=64,  block_name="block2")  # 56×56
        x = self._conv_block(x, filters=128, block_name="block3")  # 28×28
        x = self._conv_block(x, filters=256, block_name="block4")  # 14×14

        # ── Classification Head ───────────────────────────────────────
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(
            512,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.L2(self.cfg.L2_REG),
            name="dense_head",
        )(x)
        x = layers.Dropout(self.cfg.DROPOUT_RATE, name="dropout")(x)
        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            kernel_initializer="glorot_uniform",
            name="classifier",
        )(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="AnimalCNN_Custom")
        return self.model

    # ------------------------------------------------------------------
    # Architecture summary
    # ------------------------------------------------------------------

    def print_architecture_summary(self):
        if self.model is None:
            self.build()
        total     = self.model.count_params()
        trainable = sum(int(tf.size(v)) for v in self.model.trainable_variables)
        non_train = total - trainable
        print("\\n" + "=" * 60)
        print("  Custom Animal CNN — Architecture Summary")
        print("=" * 60)
        print(f"  Input Shape          : {self.cfg.IMG_SHAPE}")
        print(f"  Number of Classes    : {self.num_classes}")
        print(f"  Total Parameters     : {total:,}")
        print(f"  Trainable Parameters : {trainable:,}")
        print(f"  Non-Trainable Params : {non_train:,}")
        print("-" * 60)
        blocks = [(32, 112), (64, 56), (128, 28), (256, 14)]
        for i, (f, s) in enumerate(blocks, 1):
            print(f"  Block {i}  : 2× Conv({f:>3}, 3×3) → BN → ReLU → MaxPool  → {s}×{s}")
        print(f"  Head    : GAP → Dense(512) → Dropout({self.cfg.DROPOUT_RATE}) → Softmax({self.num_classes})")
        print("=" * 60)
        print("  ✅ All weights randomly initialised — NO pretrained weights.")
        print("=" * 60 + "\\n")
        self.model.summary()
"""))

cells.append(code("""\
# ── Instantiate and inspect the model ────────────────────────────────────
num_classes = dm.num_classes
cnn_builder = CustomCNN(Config, num_classes=num_classes)
model       = cnn_builder.build()
cnn_builder.print_architecture_summary()
"""))

# ══════════════════════════════════════════════════════════════════════════
# 11 Model Trainer
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 11 · Training Configuration & Model Trainer

### Training Strategy
- **Optimiser**: Adam with initial LR = 0.001  
- **Loss**: Categorical Cross-Entropy (one-hot labels)  
- **Metrics**: Accuracy, Precision, Recall  
- **Callbacks**: EarlyStopping · ModelCheckpoint · ReduceLROnPlateau · CSVLogger  
- **Class Weights**: Applied to compensate for dataset imbalance
"""))
cells.append(code("""\
class ModelTrainer:
    \"\"\"
    Handles model compilation, callback registration, and training execution.
    \"\"\"

    def __init__(self, model: keras.Model, cfg: Config):
        self.model    = model
        self.cfg      = cfg
        self.history  = None

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )
        print(f"✅ Model compiled.")
        print(f"   Optimiser  : Adam (lr={self.cfg.LEARNING_RATE})")
        print(f"   Loss       : categorical_crossentropy")
        print(f"   Metrics    : accuracy · precision · recall")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def build_callbacks(self) -> list:
        os.makedirs("models", exist_ok=True)
        os.makedirs(self.cfg.RESULTS_DIR, exist_ok=True)

        cb_list = [
            callbacks.ModelCheckpoint(
                filepath=self.cfg.MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                mode="max",
                verbose=1,
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.cfg.LR_REDUCE_FACTOR,
                patience=self.cfg.LR_REDUCE_PATIENCE,
                min_lr=self.cfg.LR_MIN,
                verbose=1,
            ),
            callbacks.CSVLogger(
                os.path.join(self.cfg.RESULTS_DIR, "training_log.csv"),
                append=False,
            ),
        ]
        return cb_list

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds:   tf.data.Dataset,
        class_weights: dict | None = None,
    ):
        \"\"\"
        Execute the training loop.
        Returns the Keras History object.
        \"\"\"
        print("\\n" + "=" * 55)
        print("  Starting Training …")
        print("=" * 55)
        print(f"  Max epochs     : {self.cfg.EPOCHS}")
        print(f"  Batch size     : {self.cfg.BATCH_SIZE}")
        print(f"  Early stopping : patience={self.cfg.EARLY_STOP_PATIENCE}")
        print()

        t0 = time.perf_counter()
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.EPOCHS,
            callbacks=self.build_callbacks(),
            class_weight=class_weights,
            verbose=1,
        )
        elapsed = time.perf_counter() - t0
        self.training_time = str(datetime.timedelta(seconds=int(elapsed)))

        best_val_acc  = max(self.history.history.get("val_accuracy", [0]))
        best_val_loss = min(self.history.history.get("val_loss",     [float("inf")]))
        epochs_run    = len(self.history.history.get("accuracy", []))

        print("\\n" + "=" * 55)
        print("  Training Complete")
        print("=" * 55)
        print(f"  Epochs run          : {epochs_run}")
        print(f"  Training time       : {self.training_time}")
        print(f"  Best val accuracy   : {best_val_acc*100:.2f}%")
        print(f"  Best val loss       : {best_val_loss:.4f}")
        print(f"  Model saved to      : {self.cfg.MODEL_PATH}")

        return self.history
"""))

cells.append(code("""\
# ── Compile and train the model ───────────────────────────────────────────
trainer = ModelTrainer(model, Config)
trainer.compile()
history = trainer.train(train_ds, val_ds, class_weights=class_weights)
"""))

# ══════════════════════════════════════════════════════════════════════════
# 12 Training History Graphs
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 12 · Training & Validation Graphs"))
cells.append(code("""\
def plot_training_history(history, save_dir: str = Config.RESULTS_DIR):
    \"\"\"
    Plot accuracy, loss, and learning rate curves side-by-side.
    Saves figure to results directory.
    \"\"\"
    hist   = history.history
    acc    = hist.get("accuracy",     [])
    vacc   = hist.get("val_accuracy", [])
    loss   = hist.get("loss",         [])
    vloss  = hist.get("val_loss",     [])
    lr     = hist.get("lr",           [])
    ep     = range(1, len(acc) + 1)

    fig = plt.figure(figsize=(17, 5))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    def style_ax(ax, title, ylabel):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="#8b949e")
        ax.set_ylabel(ylabel, color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for s in ax.spines.values():
            s.set_edgecolor("#30363d")
        ax.legend(facecolor="#21262d", labelcolor="white", fontsize=9)
        ax.grid(True, alpha=0.15, color="white")

    # Accuracy
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(ep, [a * 100 for a in acc],  color="#3fb950", lw=2, marker="o", ms=3, label="Train")
    ax1.plot(ep, [a * 100 for a in vacc], color="#f78166", lw=2, marker="s", ms=3, label="Validation", ls="--")
    ax1.set_ylim(0, 105)
    style_ax(ax1, "Accuracy (%)", "Accuracy (%)")

    # Loss
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(ep, loss,  color="#3fb950", lw=2, marker="o", ms=3, label="Train")
    ax2.plot(ep, vloss, color="#f78166", lw=2, marker="s", ms=3, label="Validation", ls="--")
    style_ax(ax2, "Loss", "Loss")

    # Learning Rate
    ax3 = fig.add_subplot(gs[2])
    if lr:
        ax3.plot(ep, lr, color="#d2a8ff", lw=2, marker="^", ms=3, label="LR")
        ax3.set_yscale("log")
    else:
        ax3.text(0.5, 0.5, "LR not logged", ha="center", va="center", color="#8b949e")
    style_ax(ax3, "Learning Rate", "LR (log scale)")

    fig.suptitle("Custom Animal CNN — Training Progress",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    out = os.path.join(save_dir, "training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Training history plot saved → {out}")

plot_training_history(history)
"""))

# ══════════════════════════════════════════════════════════════════════════
# 13 Model Evaluator
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 13 · Test Evaluation

### Methodology
The model is evaluated on the **independent test set** — images that were never seen during training or hyperparameter selection.

Metrics computed:
- **Test Accuracy** — proportion of correct predictions
- **Precision** — of all positive predictions, how many were correct
- **Recall** — of all actual positives, how many were found
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — per-class error analysis
"""))
cells.append(code("""\
class ModelEvaluator:
    \"\"\"
    Evaluates the trained CNN on the test dataset and generates
    full reporting artefacts: classification report, confusion matrix,
    incorrect prediction gallery.
    \"\"\"

    def __init__(self, model: keras.Model, test_ds: tf.data.Dataset,
                 class_names: list, cfg: Config):
        self.model       = model
        self.test_ds     = test_ds
        self.class_names = class_names
        self.cfg         = cfg
        self.y_true      = None
        self.y_pred      = None
        self.y_probs     = None
        self.test_images = None   # raw tensors for visualisation
        self.test_loss   = None
        self.test_acc    = None

    # ------------------------------------------------------------------
    # Collect predictions
    # ------------------------------------------------------------------

    def run_inference(self):
        \"\"\"Iterate over the test dataset and gather predictions.\"\"\"
        y_true_lst, y_pred_lst, y_prob_lst, imgs_lst = [], [], [], []

        for imgs, lbls in self.test_ds:
            probs  = self.model(imgs, training=False).numpy()
            truths = np.argmax(lbls.numpy(), axis=1)
            preds  = np.argmax(probs, axis=1)
            y_true_lst.append(truths)
            y_pred_lst.append(preds)
            y_prob_lst.append(probs)
            imgs_lst.append(imgs.numpy())

        self.y_true      = np.concatenate(y_true_lst)
        self.y_pred      = np.concatenate(y_pred_lst)
        self.y_probs     = np.concatenate(y_prob_lst)
        self.test_images = np.concatenate(imgs_lst)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self):
        \"\"\"Run Keras evaluate() and store test loss / accuracy.\"\"\"
        self.test_loss, self.test_acc = self.model.evaluate(self.test_ds, verbose=1)
        print(f"\\n  Test Loss     : {self.test_loss:.4f}")
        print(f"  Test Accuracy : {self.test_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def metrics_summary(self):
        \"\"\"Print precision, recall, F1 per class and macro averages.\"\"\"
        prec, rec, f1, sup = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None,
            labels=list(range(len(self.class_names)))
        )
        mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average="macro"
        )

        print("\\n" + "=" * 65)
        print("  Per-Class Metrics")
        print("=" * 65)
        print(f"  {'Class':<22} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6}")
        print("-" * 65)
        for i, cls in enumerate(self.class_names):
            print(f"  {cls:<22} {prec[i]*100:>7.2f}% {rec[i]*100:>7.2f}% {f1[i]*100:>7.2f}% {int(sup[i]):>6}")
        print("-" * 65)
        print(f"  {'Macro Average':<22} {mac_p*100:>7.2f}% {mac_r*100:>7.2f}% {mac_f1*100:>7.2f}%")
        print("=" * 65)
        return {"precision": float(mac_p), "recall": float(mac_r), "f1": float(mac_f1)}

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------

    def classification_report(self):
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names, digits=4
        )
        print("\\n--- Sklearn Classification Report ---")
        print(report)
        path = os.path.join(self.cfg.RESULTS_DIR, "classification_report.txt")
        header = (
            "=" * 60 + "\\n"
            "  Custom Animal CNN — Classification Report\\n"
            "=" * 60 + "\\n"
            f"  Test Loss     : {self.test_loss:.4f}\\n"
            f"  Test Accuracy : {self.test_acc*100:.2f}%\\n"
            "=" * 60 + "\\n\\n"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + report)
        print(f"Classification report saved → {path}")
        return report

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        n  = len(self.class_names)

        fig, ax = plt.subplots(figsize=(max(8, n * 1.6), max(6, n * 1.3)))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        im   = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors="white")

        ax.set(
            xticks=np.arange(n),
            yticks=np.arange(n),
            xticklabels=[c.capitalize() for c in self.class_names],
            yticklabels=[c.capitalize() for c in self.class_names],
            ylabel="Actual Class", xlabel="Predicted Class",
            title="Confusion Matrix — Custom Animal CNN",
        )
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(colors="white")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color="white")
        plt.setp(ax.get_yticklabels(), color="white")

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] < thresh else "black",
                        fontsize=12, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()
        out = os.path.join(self.cfg.RESULTS_DIR, "confusion_matrix.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()
        print(f"Confusion matrix saved → {out}")

    # ------------------------------------------------------------------
    # Per-class accuracy
    # ------------------------------------------------------------------

    def per_class_accuracy(self):
        cm   = confusion_matrix(self.y_true, self.y_pred)
        diag = cm.diagonal()
        row_sums = cm.sum(axis=1).clip(min=1)
        accs = diag / row_sums
        print("\\n  Per-Class Accuracy")
        print("  " + "-" * 40)
        for cls, acc in zip(self.class_names, accs):
            bar = "█" * int(acc * 20)
            print(f"  {cls:<20} : {acc*100:5.1f}%  {bar}")
        return {cls: float(a) for cls, a in zip(self.class_names, accs)}

    # ------------------------------------------------------------------
    # Incorrect predictions gallery
    # ------------------------------------------------------------------

    def show_incorrect_predictions(self, max_show: int = 12):
        \"\"\"Display a grid of misclassified test images.\"\"\"
        wrong_idx = np.where(self.y_true != self.y_pred)[0]
        if len(wrong_idx) == 0:
            print("  ✅ No incorrect predictions on the test set!")
            return

        n_show = min(max_show, len(wrong_idx))
        cols   = min(4, n_show)
        rows   = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        fig.patch.set_facecolor("#0d1117")
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax_i, idx in enumerate(wrong_idx[:n_show]):
            ax  = axes_flat[ax_i]
            img = self.test_images[idx]
            ax.imshow(np.clip(img, 0, 1))
            actual    = self.class_names[self.y_true[idx]].capitalize()
            predicted = self.class_names[self.y_pred[idx]].capitalize()
            conf      = self.y_probs[idx][self.y_pred[idx]] * 100
            ax.set_title(f"Actual: {actual}\\nPred: {predicted} ({conf:.1f}%)",
                         color="#f78166", fontsize=8)
            ax.axis("off")

        for ax_i in range(n_show, len(axes_flat)):
            axes_flat[ax_i].axis("off")

        fig.suptitle(f"Incorrect Predictions ({len(wrong_idx)} total misclassified)",
                     color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(self.cfg.RESULTS_DIR, "incorrect_predictions.png")
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()
        print(f"Incorrect predictions figure saved → {out}")
        print(f"\\n  Total misclassified: {len(wrong_idx)} / {len(self.y_true)}")

    # ------------------------------------------------------------------
    # Sample correct predictions
    # ------------------------------------------------------------------

    def show_correct_predictions(self, max_show: int = 8):
        \"\"\"Display a sample of correctly classified test images.\"\"\"
        correct_idx = np.where(self.y_true == self.y_pred)[0]
        n_show      = min(max_show, len(correct_idx))
        cols        = min(4, n_show)
        rows        = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        fig.patch.set_facecolor("#0d1117")
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax_i, idx in enumerate(correct_idx[:n_show]):
            ax  = axes_flat[ax_i]
            img = self.test_images[idx]
            ax.imshow(np.clip(img, 0, 1))
            cls  = self.class_names[self.y_true[idx]].capitalize()
            conf = self.y_probs[idx][self.y_pred[idx]] * 100
            ax.set_title(f"✅ {cls}\\nConfidence: {conf:.1f}%",
                         color="#3fb950", fontsize=8)
            ax.axis("off")

        for ax_i in range(n_show, len(axes_flat)):
            axes_flat[ax_i].axis("off")

        fig.suptitle("Sample Correct Predictions",
                     color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(self.cfg.RESULTS_DIR, "correct_predictions.png")
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()

    # ------------------------------------------------------------------
    # Sample-level prediction table
    # ------------------------------------------------------------------

    def print_sample_predictions(self, n: int = 30):
        w = max(len(c) for c in self.class_names) + 2
        print("\\n" + "=" * 72)
        print(f"  {'#':<5} {'Actual':<{w}} {'Predicted':<{w}} {'Conf':>8}  {'✓/✗'}")
        print("-" * 72)
        for i in range(min(n, len(self.y_true))):
            actual    = self.class_names[self.y_true[i]]
            predicted = self.class_names[self.y_pred[i]]
            conf      = self.y_probs[i][self.y_pred[i]] * 100
            correct   = "✓" if self.y_true[i] == self.y_pred[i] else "✗"
            print(f"  {i+1:<5} {actual:<{w}} {predicted:<{w}} {conf:>7.2f}%  {correct}")
        if len(self.y_true) > n:
            print(f"  … (showing first {n} of {len(self.y_true)} samples)")
        print("=" * 72)
"""))

cells.append(code("""\
# ── Load best saved model for evaluation ─────────────────────────────────
print("Loading best saved model …")
best_model = keras.models.load_model(Config.MODEL_PATH)
print(f"✅ Model loaded from {Config.MODEL_PATH}")

evaluator = ModelEvaluator(best_model, test_ds, class_names, Config)

print("\\n[1/5] Running Keras evaluate() …")
evaluator.evaluate()

print("\\n[2/5] Running full inference …")
evaluator.run_inference()
"""))

cells.append(code("""\
# ── Metrics summary ───────────────────────────────────────────────────────
print("\\n[3/5] Computing per-class metrics …")
macro_metrics = evaluator.metrics_summary()
per_cls_acc   = evaluator.per_class_accuracy()
"""))

cells.append(code("""\
# ── Classification report ─────────────────────────────────────────────────
print("\\n[4/5] Generating classification report …")
evaluator.classification_report()
"""))

cells.append(code("""\
# ── Confusion matrix ──────────────────────────────────────────────────────
print("\\n[5/5] Plotting confusion matrix …")
evaluator.plot_confusion_matrix()
"""))

cells.append(code("""\
# ── Sample predictions table ──────────────────────────────────────────────
evaluator.print_sample_predictions(n=40)
"""))

cells.append(code("""\
# ── Correct vs incorrect galleries ───────────────────────────────────────
evaluator.show_correct_predictions(max_show=8)
evaluator.show_incorrect_predictions(max_show=12)
"""))

# ══════════════════════════════════════════════════════════════════════════
# 14 Incorrect Prediction Analysis (Research)
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 14 · Incorrect Prediction Analysis

The following analysis considers the most common error patterns.  
Each explanation is labeled as a **possible cause** — not a confirmed root cause.

| Error Pattern | Possible Cause |
|---|---|
| Monkey ↔ Bird | Similar colour profiles; overlapping habitat backgrounds |
| Cow ↔ Elephant | Both large-body animals; possible background/grass bias |
| Pig ↔ Cow | Similar body proportions; possible lighting/angle variation |

Strategies to reduce errors in future work:
1. Increase per-class image diversity (different angles, lighting, backgrounds)
2. Remove background-heavy images where the animal is partially occluded
3. Apply attention mechanisms (Squeeze-and-Excitation blocks)
4. Use test-time augmentation (TTA) for borderline cases
"""))

# ══════════════════════════════════════════════════════════════════════════
# 15 Save Model & Class Names
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 15 · Save Model and Class Mapping"))
cells.append(code("""\
# Model is already saved by ModelCheckpoint; confirm it exists
if os.path.exists(Config.MODEL_PATH):
    size_mb = os.path.getsize(Config.MODEL_PATH) / 1_048_576
    print(f"✅ Model already saved: {Config.MODEL_PATH}  ({size_mb:.1f} MB)")
else:
    best_model.save(Config.MODEL_PATH)
    print(f"✅ Model saved: {Config.MODEL_PATH}")

# Save class names to JSON so the OpenCV predictor uses the same mapping
class_data = {"class_names": class_names, "num_classes": len(class_names)}
with open(Config.CLASS_NAMES_PATH, "w", encoding="utf-8") as fh:
    json.dump(class_data, fh, indent=2)
print(f"✅ Class names saved: {Config.CLASS_NAMES_PATH}")
print(f"   Mapping: {class_data}")
"""))

# ══════════════════════════════════════════════════════════════════════════
# 16 OpenCV Predictor
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 16 · OpenCV Prediction Module

### Inference Pipeline
```
New Image
     ↓
cv2.imread()   — load as BGR array
     ↓
Validate image (not None, correct file)
     ↓
BGR → RGB
     ↓
Resize → 224×224
     ↓
Normalise → [0, 1]
     ↓
Expand dims → (1, 224, 224, 3)
     ↓
Custom CNN (trained weights)
     ↓
Softmax probabilities
     ↓
Confidence threshold check
     ↓
Animal Class + Confidence
     ↓
cv2.putText() overlay
     ↓
cv2.imshow()
```
"""))
cells.append(code("""\
class OpenCVPredictor:
    \"\"\"
    Loads the saved custom CNN and class mapping, then performs
    single-image inference using OpenCV for image I/O.

    Usage
    -----
    predictor = OpenCVPredictor(Config)
    result    = predictor.predict("sample_images/test_elephant.jpg")
    \"\"\"

    def __init__(self, cfg: Config):
        self.cfg         = cfg
        self.model       = None
        self.class_names = []
        self._load()

    # ------------------------------------------------------------------
    # Load assets
    # ------------------------------------------------------------------

    def _load(self):
        # Load class names
        if not os.path.exists(self.cfg.CLASS_NAMES_PATH):
            raise FileNotFoundError(
                f"Class names file not found: {self.cfg.CLASS_NAMES_PATH}\\n"
                "Run training first."
            )
        with open(self.cfg.CLASS_NAMES_PATH, "r") as fh:
            data = json.load(fh)
        self.class_names = data["class_names"]

        # Load model
        if not os.path.exists(self.cfg.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {self.cfg.MODEL_PATH}\\n"
                "Run training first."
            )
        self.model = keras.models.load_model(self.cfg.MODEL_PATH)
        print(f"✅ OpenCVPredictor ready.")
        print(f"   Model      : {self.cfg.MODEL_PATH}")
        print(f"   Classes    : {self.class_names}")
        print(f"   Threshold  : {self.cfg.CONFIDENCE_THRESHOLD}")

    # ------------------------------------------------------------------
    # Preprocess (identical to training normalisation)
    # ------------------------------------------------------------------

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w    = self.cfg.IMG_SIZE
        resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm    = rgb.astype(np.float32) / 255.0
        return np.expand_dims(norm, axis=0)

    # ------------------------------------------------------------------
    # Overlay on image
    # ------------------------------------------------------------------

    def _draw_overlay(
        self,
        img_bgr:    np.ndarray,
        label:      str,
        confidence: float,
        is_unknown: bool,
    ) -> np.ndarray:
        img = img_bgr.copy()
        h, w = img.shape[:2]

        # Semi-transparent banner
        banner_h = max(55, int(h * 0.13))
        overlay  = img.copy()
        bg_color = (20, 30, 60) if not is_unknown else (40, 20, 20)
        cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        font       = cv2.FONT_HERSHEY_DUPLEX
        fscale     = max(0.5, w / 900)
        thickness  = max(1, int(fscale * 1.8))
        text_color = (0, 220, 100) if not is_unknown else (0, 180, 255)
        conf_color = (180, 255, 180) if not is_unknown else (180, 220, 255)

        conf_pct  = confidence * 100
        main_text = f"Predicted: {label}"
        conf_text = f"Confidence: {conf_pct:.2f}%"

        (_, th1), _  = cv2.getTextSize(main_text, font, fscale, thickness)
        (_, th2), _  = cv2.getTextSize(conf_text, font, fscale * 0.85, thickness)

        cv2.putText(img, main_text,
                    (12, int(banner_h * 0.42) + th1 // 2),
                    font, fscale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(img, conf_text,
                    (12, int(banner_h * 0.80) + th2 // 2),
                    font, fscale * 0.85, conf_color, thickness - 1, cv2.LINE_AA)

        # Confidence bar
        bar_y = banner_h + 4
        cv2.rectangle(img, (0, bar_y), (w, bar_y + 6), (50, 50, 50), -1)
        cv2.rectangle(img, (0, bar_y), (int(w * confidence), bar_y + 6),
                      (0, 200, 80) if not is_unknown else (0, 180, 255), -1)
        return img

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        image_path: str,
        show_window: bool = True,
        threshold:   float | None = None,
    ) -> dict:
        \"\"\"
        Run inference on a single image.

        Parameters
        ----------
        image_path  : str   — Path to the image file.
        show_window : bool  — Open an OpenCV display window.
        threshold   : float — Override the configured confidence threshold.

        Returns
        -------
        dict: label, confidence, is_unknown, all_probs
        \"\"\"
        thresh = threshold if threshold is not None else self.cfg.CONFIDENCE_THRESHOLD

        # ── 1. Load image ─────────────────────────────────────────────
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(
                f"OpenCV could not read the image: {image_path}\\n"
                "Verify the file is a valid image (JPG/PNG/BMP/WEBP)."
            )

        orig_h, orig_w = image_bgr.shape[:2]
        print(f"[predict] Image loaded: {os.path.basename(image_path)}  ({orig_w}×{orig_h})")

        # ── 2. Preprocess ─────────────────────────────────────────────
        tensor = self._preprocess(image_bgr)

        # ── 3. Inference ──────────────────────────────────────────────
        probs     = self.model(tensor, training=False).numpy()[0]
        pred_idx  = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        is_unknown = confidence < thresh

        label = "Unknown / Low Confidence" if is_unknown else self.class_names[pred_idx]

        # ── 4. Console output ─────────────────────────────────────────
        print("\\n" + "=" * 44)
        print("  Animal Classification Result")
        print("=" * 44)
        if is_unknown:
            print(f"  Prediction  : Unknown / Low Confidence")
        else:
            print(f"  Predicted Animal  : {label.capitalize()}")
        print(f"  Confidence        : {confidence*100:.2f}%")
        print("-" * 44)
        print("  All class probabilities:")
        for cls, p in zip(self.class_names, probs):
            bar = "█" * int(p * 25)
            print(f"    {cls:<22} {p*100:5.2f}%  {bar}")
        print("=" * 44 + "\\n")

        # ── 5. Visual overlay & display ───────────────────────────────
        annotated = self._draw_overlay(image_bgr, label, confidence, is_unknown)

        if show_window:
            win_title = "Animal CNN — Result  (press any key to close)"
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
            disp_w = min(orig_w, 900)
            disp_h = int(orig_h * disp_w / max(orig_w, 1))
            cv2.resizeWindow(win_title, disp_w, disp_h)
            cv2.imshow(win_title, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "label"     : label,
            "confidence": confidence,
            "is_unknown": is_unknown,
            "all_probs" : {c: float(p) for c, p in zip(self.class_names, probs)},
        }

    # ------------------------------------------------------------------
    # Predict and display inside notebook (matplotlib instead of cv2.imshow)
    # ------------------------------------------------------------------

    def predict_notebook(self, image_path: str, threshold: float | None = None) -> dict:
        \"\"\"
        Same as predict() but renders the annotated image inline in Jupyter
        using matplotlib instead of cv2.imshow (avoids GUI issues in notebooks).
        \"\"\"
        result    = self.predict(image_path, show_window=False, threshold=threshold)
        image_bgr = cv2.imread(image_path)
        annotated = self._draw_overlay(
            image_bgr, result["label"], result["confidence"], result["is_unknown"]
        )
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.imshow(annotated_rgb)
        ax.axis("off")
        status = "✅" if not result["is_unknown"] else "⚠️"
        ax.set_title(
            f"{status} {result['label'].capitalize()}  —  {result['confidence']*100:.2f}%",
            color="white", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()
        return result
"""))

cells.append(code("""\
# ── Instantiate the predictor ─────────────────────────────────────────────
predictor = OpenCVPredictor(Config)
"""))

# ══════════════════════════════════════════════════════════════════════════
# 17 New Image Prediction
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 17 · New Image Prediction

Change `IMAGE_PATH` to the path of any image you want to classify.
"""))
cells.append(code("""\
# ─────────────────────────────────────────────────────────────────────────
#  ↓ CHANGE THIS PATH to your image ↓
IMAGE_PATH = "sample_images/test_image.jpg"
# ─────────────────────────────────────────────────────────────────────────

if os.path.isfile(IMAGE_PATH):
    result = predictor.predict_notebook(IMAGE_PATH)
else:
    print(f"⚠️  Image not found: {IMAGE_PATH}")
    print("   Place an image in the sample_images/ folder and update IMAGE_PATH.")
    print("\\n   Demo using a test image from the test set …")

    # Fall back to the first test image found
    fallback = None
    for cls in class_names:
        cls_dir = os.path.join(Config.TEST_DIR, cls)
        if os.path.isdir(cls_dir):
            files = [f for f in os.listdir(cls_dir)
                     if Path(f).suffix.lower() in DatasetManager.SUPPORTED_EXTS]
            if files:
                fallback = os.path.join(cls_dir, files[0])
                break

    if fallback:
        print(f"   Using: {fallback}")
        result = predictor.predict_notebook(fallback)
    else:
        print("   No test images found either. Add images to dataset/test/ first.")
"""))

# ══════════════════════════════════════════════════════════════════════════
# 18 Confidence Threshold Demo
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 18 · Confidence Threshold Behaviour"))
cells.append(code("""\
# Demo: show how different thresholds affect the output label
thresholds = [0.30, 0.60, 0.80, 0.95]

if os.path.isfile(IMAGE_PATH):
    image_bgr = cv2.imread(IMAGE_PATH)
    tensor    = ImagePreprocessor.preprocess_single_image_opencv(image_bgr, Config.IMG_SIZE)
    probs     = predictor.model(tensor, training=False).numpy()[0]
    conf      = float(np.max(probs))
    pred_cls  = class_names[int(np.argmax(probs))]

    print(f"  Raw prediction : {pred_cls.capitalize()} @ {conf*100:.2f}%\\n")
    print(f"  {'Threshold':<12} {'Output'}")
    print("  " + "-" * 40)
    for t in thresholds:
        label = f"{pred_cls.capitalize()}" if conf >= t else "Unknown / Low Confidence"
        flag  = "✅" if conf >= t else "⚠️ "
        print(f"  {t:<12.2f} {flag} {label}")
else:
    print("Set IMAGE_PATH above to run this demo.")
"""))

# ══════════════════════════════════════════════════════════════════════════
# 19 Hyperparameter Experiment Tracker (lightweight)
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 19 · Hyperparameter Experiment Notes

> **Note**: Running full hyperparameter experiments requires additional training runs.  
> Record your experiment results in the table below after each run.

| Experiment | LR | Batch | Dropout | Val Accuracy | Notes |
|---|---|---|---|---|---|
| Baseline | 0.001 | 32 | 0.40 | — | Initial configuration |
| Exp-2 | 0.0005 | 32 | 0.40 | — | Lower LR |
| Exp-3 | 0.001 | 16 | 0.50 | — | Smaller batch, more dropout |
| Exp-4 | 0.001 | 32 | 0.30 | — | Less dropout |

Replace the `—` values with actual validation accuracy from each run.

**Important**: Use **only the validation set** to compare configurations.  
The test set must remain unseen until the final evaluation.
"""))

# ══════════════════════════════════════════════════════════════════════════
# 20 Final Research Summary
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("## 20 · Final Research Summary"))
cells.append(code("""\
def print_final_summary(dm, evaluator, trainer, history, class_names, model):
    \"\"\"Print the complete research summary table.\"\"\"

    train_total = sum(dm._stats["train"].values())
    val_total   = sum(dm._stats["validation"].values())
    test_total  = sum(dm._stats["test"].values())

    final_train_acc = history.history.get("accuracy", [0])[-1] * 100
    best_val_acc    = max(history.history.get("val_accuracy", [0])) * 100
    test_acc        = (evaluator.test_acc or 0) * 100

    total_params     = model.count_params()
    trainable_params = sum(int(tf.size(v)) for v in model.trainable_variables)

    print("\\n" + "=" * 55)
    print("  FINAL MODEL PERFORMANCE")
    print("=" * 55)
    print(f"  Number of Classes       : {len(class_names)}")
    print(f"  Class Names             : {class_names}")
    print(f"  Training Images         : {train_total}")
    print(f"  Validation Images       : {val_total}")
    print(f"  Test Images             : {test_total}")
    print(f"  Input Image Size        : {Config.IMG_SIZE[0]} × {Config.IMG_SIZE[1]}")
    print("-" * 55)
    print(f"  CNN Architecture        : Custom CNN (4 Conv Blocks)")
    print(f"  Pretrained Model        : NO")
    print(f"  Transfer Learning       : NO")
    print(f"  Total Parameters        : {total_params:,}")
    print(f"  Trainable Parameters    : {trainable_params:,}")
    print("-" * 55)
    print(f"  Training Accuracy       : {final_train_acc:.2f}%")
    print(f"  Best Validation Acc     : {best_val_acc:.2f}%")
    print(f"  Test Accuracy           : {test_acc:.2f}%")
    print(f"  Macro Precision         : {(evaluator.metrics_summary()['precision']*100 if evaluator.y_true is not None else 0):.2f}%")
    print(f"  Macro Recall            : {(evaluator.metrics_summary()['recall']*100 if evaluator.y_true is not None else 0):.2f}%")
    print(f"  Macro F1-Score          : {(evaluator.metrics_summary()['f1']*100 if evaluator.y_true is not None else 0):.2f}%")
    print(f"  Training Time           : {trainer.training_time}")
    print("=" * 55)
    print("  All values generated from actual training and evaluation.")
    print("=" * 55 + "\\n")

print_final_summary(dm, evaluator, trainer, history, class_names, best_model)
"""))

# ══════════════════════════════════════════════════════════════════════════
# 21 Research Explanation
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 21 · Research Documentation

### Why CNN?

Convolutional Neural Networks are the most suitable architecture for image classification tasks because:
- **Local connectivity**: Conv filters detect spatially local features (edges, textures) irrespective of position.
- **Parameter sharing**: The same filter is applied across the entire image, dramatically reducing the number of parameters compared to fully connected networks.
- **Hierarchical learning**: Stacking multiple convolutional layers allows the network to learn increasingly abstract features from simple edges to complex animal body structures.
- **Translation equivariance**: CNNs naturally handle animals appearing at different positions in the image.

### Why Custom CNN (No Transfer Learning)?

This research project requires demonstrating that a CNN can be designed, trained, and evaluated entirely from scratch. The custom architecture:
- Starts with **randomly initialised weights** (He Normal for Conv, Glorot Uniform for Dense).
- Learns **only from the provided dataset** — no ImageNet or other pretraining.
- Makes the model directly attributable to this research, not to a third-party backbone.
- Avoids potential domain mismatch between natural image pretraining (ImageNet) and the specific visual characteristics of these five animal classes.

### Architecture Design Rationale

| Choice | Reason |
|---|---|
| 3×3 kernels | Standard and efficient; two 3×3 stacked ≡ one 5×5 but with extra non-linearity |
| He Normal init | Optimal for ReLU networks — avoids vanishing/exploding gradients |
| Batch Normalisation | Stabilises training, allows higher LR, acts as mild regulariser |
| Global Average Pooling | Drastically reduces parameters vs Flatten; improves generalisation |
| Dropout (0.40) | Reduces co-adaptation of neurons; prevents overfitting |
| L2 regularisation | Penalises large weights; improves generalisation |
| Adam optimiser | Adaptive learning rate; robust to noisy gradients |
| ReduceLROnPlateau | Automatically fine-tunes learning rate when validation plateaus |
| EarlyStopping | Prevents overfitting by halting training when val_loss stops improving |

### Preprocessing

All images are:
1. Loaded via OpenCV (prediction) or TensorFlow (training)
2. Resized to **224×224** — standard input suitable for the 4-block architecture
3. Converted to RGB (from BGR if using OpenCV)
4. Normalised to **[0, 1]** by dividing by 255.0

Validation and test images receive **only these steps** — no augmentation — to reflect true real-world evaluation conditions.

### Data Augmentation

Training images additionally receive:
- **Random horizontal flip** — animals may face either direction
- **Random rotation ±15%** — camera tilt variation
- **Random zoom ±10%** — distance from animal
- **Random translation ±10%** — animal not always centred
- **Random brightness ±15%** — lighting variation in field conditions

None of these transforms are applied to validation or test images.

### Limitations

| Limitation | Description |
|---|---|
| Dataset size | Smaller datasets increase overfitting risk; more images generally improve robustness |
| Visual similarity | Some species (e.g., monkey vs. bird at distance) may share visual characteristics |
| Background bias | Model may learn habitat cues rather than animal-specific features |
| Resolution | Low-resolution images may lack discriminative detail |
| Occlusion | Partially visible animals are harder to classify correctly |
| Pose/Angle | Unusual viewing angles may not be well represented in training data |
| Colour bias | Greyscale or unusual colouration may reduce confidence |

### Future Improvements

1. **Attention mechanisms** (e.g., CBAM, SE blocks) to focus on discriminative animal body parts
2. **Test-Time Augmentation (TTA)** — average predictions across multiple augmented views of the same image
3. **Larger dataset** with greater diversity of backgrounds, angles, and lighting
4. **Data-driven augmentation** using learned augmentation policies (AutoAugment / RandAugment)
5. **Class Activation Maps (CAM)** to visualise which image regions the CNN uses for classification
6. **Uncertainty quantification** via Monte Carlo Dropout for production deployment
7. **Ensemble methods** — combine multiple independently trained CNNs for improved robustness
"""))

# ══════════════════════════════════════════════════════════════════════════
# 22 Conclusion
# ══════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 22 · Research Conclusion

This notebook presents a complete, end-to-end implementation of a **custom Convolutional Neural Network** for animal image classification, built entirely from scratch.

### Key Contributions
✅ Novel 4-block CNN architecture designed specifically for 5-class animal classification  
✅ Systematic dataset validation and leakage checking  
✅ Class-imbalance analysis with balanced class-weight compensation  
✅ Training-only data augmentation preserving test-set integrity  
✅ Full evaluation suite: accuracy, precision, recall, F1, confusion matrix  
✅ Incorrect-prediction analysis for research insight  
✅ OpenCV inference module for deployment on new unseen images  
✅ Confidence-based uncertainty reporting  
✅ Reproducible training pipeline with fixed random seeds  

### Verification of "From Scratch" Requirement
- No `hub.load()`, `tf.keras.applications.*`, or external backbone calls
- No pretrained weight files downloaded
- All layers created with `keras.layers.*` using random initialisation
- Model trained exclusively on the researcher's own dataset

---
*Custom CNN Animal Classification — Research Notebook*  
*Developed using TensorFlow · Keras · OpenCV · Scikit-learn*
"""))

# ══════════════════════════════════════════════════════════════════════════
# Build the notebook dict
# ══════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

# ── Write to file ─────────────────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Animal_CNN_Research.ipynb")
with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)

print(f"[OK] Notebook written to: {output_path}")
print(f"     Cells: {len(cells)}")
