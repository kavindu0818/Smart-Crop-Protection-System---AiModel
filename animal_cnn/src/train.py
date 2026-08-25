"""
=============================================================================
train.py — Training Pipeline for Custom Animal CNN
=============================================================================
Usage:
    python -m src.train
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import datetime
import numpy as np
import tensorflow as tf

from src import config
from src.dataset_loader import load_train_dataset, load_validation_dataset
from src.cnn_model import build_model
from src.utils import plot_training_history, save_metadata, count_images, Timer


def build_callbacks() -> list:
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PAT,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_DECAY_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.LR_MIN,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.RESULTS_DIR, "training_log.csv"),
            append=False,
        ),
    ]


def train() -> None:
    print("\n" + "=" * 60)
    print("  Custom Animal CNN — Training Pipeline")
    print("=" * 60)

    timer = Timer().start()

    print("\n[Step 1/6] Loading datasets ...")
    train_ds, class_names, num_classes = load_train_dataset()
    val_ds, _, _ = load_validation_dataset(class_names)

    print(f"\n  Classes detected ({num_classes}): {class_names}")

    train_counts = count_images(config.TRAIN_DIR, class_names)
    val_counts = count_images(config.VALIDATION_DIR, class_names)

    print("\n[Step 2/6] Building custom CNN from scratch ...")
    model = build_model(num_classes=num_classes)
    model.summary()

    total_params = model.count_params()
    trainable_params = sum(int(tf.size(v)) for v in model.trainable_variables)

    print("\n[Step 3/6] Registering callbacks ...")
    callbacks = build_callbacks()

    print(f"\n[Step 4/6] Training for up to {config.EPOCHS} epochs ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = timer.elapsed_str()

    print("\n[Step 5/6] Training complete.")
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    best_val_loss = min(history.history.get("val_loss", [float("inf")]))
    final_train_acc = history.history.get("accuracy", [0.0])[-1]
    epochs_run = len(history.history.get("accuracy", []))

    print(f"  Epochs run             : {epochs_run}")
    print(f"  Final train accuracy   : {final_train_acc * 100:.2f}%")
    print(f"  Best validation acc    : {best_val_acc * 100:.2f}%")
    print(f"  Best validation loss   : {best_val_loss:.4f}")
    print(f"  Training time          : {elapsed}")
    print(f"  Best model saved to    : {config.MODEL_SAVE_PATH}")

    print("\n[Step 6/6] Saving training artefacts ...")
    plot_training_history(history)

    # Save class_names.json in base dir for OpenCVPredictor
    class_data = {"class_names": class_names, "num_classes": len(class_names)}
    class_json_path = os.path.join(config.BASE_DIR, "class_names.json")
    with open(class_json_path, "w", encoding="utf-8") as fh:
        json.dump(class_data, fh, indent=2)

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "training_time": elapsed,
        "num_classes": num_classes,
        "class_names": class_names,
        "img_size": list(config.IMG_SIZE),
        "batch_size": config.BATCH_SIZE,
        "epochs_configured": config.EPOCHS,
        "epochs_run": epochs_run,
        "learning_rate_init": config.LEARNING_RATE,
        "optimizer": "Adam",
        "loss_function": "categorical_crossentropy",
        "total_parameters": total_params,
        "trainable_parameters": int(trainable_params),
        "dropout_rate": config.DROPOUT_RATE,
        "l2_regularisation": config.L2_REG,
        "final_train_accuracy": float(final_train_acc),
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "train_image_counts": train_counts,
        "val_image_counts": val_counts,
        "model_save_path": config.MODEL_SAVE_PATH,
        "seed": config.SEED,
    }
    save_metadata(meta)

    print("\n" + "=" * 60)
    print("  Training pipeline completed successfully.")
    print("  Run  python -m src.evaluate  to test the model.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()
