"""
=============================================================================
config.py — Central Configuration for Custom CNN Animal Classification
=============================================================================
All hyperparameters, directory paths, and training settings are defined here.
To adapt the project to your own dataset, update the directory paths below.
=============================================================================
"""

import os
import random
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Reproducibility Seeds
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ---------------------------------------------------------------------------
# Dataset Paths  <- Change these to point to your actual dataset folders
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR       = os.path.join(DATASET_DIR, "train")
VALIDATION_DIR  = os.path.join(DATASET_DIR, "validation")
TEST_DIR        = os.path.join(DATASET_DIR, "test")

# ---------------------------------------------------------------------------
# Output Directories
# ---------------------------------------------------------------------------
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
SAMPLE_DIR      = os.path.join(BASE_DIR, "sample_images")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "animal_cnn.keras")
HISTORY_PLOT    = os.path.join(RESULTS_DIR, "training_history.png")
CONFUSION_PLOT  = os.path.join(RESULTS_DIR, "confusion_matrix.png")
REPORT_PATH     = os.path.join(RESULTS_DIR, "classification_report.txt")
METADATA_PATH   = os.path.join(RESULTS_DIR, "training_metadata.json")

# ---------------------------------------------------------------------------
# Image Settings
# ---------------------------------------------------------------------------
IMG_SIZE     = (224, 224)          # (height, width) — model input resolution
IMG_SHAPE    = (224, 224, 3)       # (H, W, C) — 3-channel RGB
NUM_CHANNELS = 3

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE      = 32
EPOCHS          = 60
LEARNING_RATE   = 1e-3
LR_DECAY_FACTOR = 0.5
LR_PATIENCE     = 5
LR_MIN          = 1e-6
EARLY_STOP_PAT  = 12

# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------
DROPOUT_RATE = 0.4
L2_REG       = 1e-4

# ---------------------------------------------------------------------------
# CNN Architecture
# ---------------------------------------------------------------------------
DENSE_UNITS = 512

# ---------------------------------------------------------------------------
# Data Augmentation (training only)
# ---------------------------------------------------------------------------
AUG_ROTATION        = 0.15
AUG_WIDTH_SHIFT     = 0.10
AUG_HEIGHT_SHIFT    = 0.10
AUG_ZOOM            = 0.10
AUG_BRIGHTNESS      = (0.85, 1.15)

# ---------------------------------------------------------------------------
# Prediction Settings
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in [MODELS_DIR, RESULTS_DIR, SAMPLE_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_dirs()
