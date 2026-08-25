"""
=============================================================================
evaluate.py — Test Evaluation Module for Custom Animal CNN
=============================================================================
Usage:
    python -m src.evaluate
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from src import config
from src.dataset_loader import load_test_dataset
from src.utils import plot_confusion_matrix, save_classification_report, per_class_accuracy, save_metadata


def load_trained_model() -> tf.keras.Model:
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"[evaluate] Trained model not found at: {config.MODEL_SAVE_PATH}\n"
            "Please run  python -m src.train  first."
        )
    print(f"[evaluate] Loading model from: {config.MODEL_SAVE_PATH}")
    return tf.keras.models.load_model(config.MODEL_SAVE_PATH)


def load_class_names() -> list:
    class_json_path = os.path.join(config.BASE_DIR, "class_names.json")
    if os.path.exists(class_json_path):
        with open(class_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("class_names", [])

    if os.path.exists(config.METADATA_PATH):
        with open(config.METADATA_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("class_names"):
            return meta["class_names"]

    return sorted(entry.name for entry in os.scandir(config.TEST_DIR) if entry.is_dir())


def collect_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset):
    y_true_list, y_probs_list = [], []

    for images, labels in test_ds:
        probs = model(images, training=False).numpy()
        truths = np.argmax(labels.numpy(), axis=1)
        y_probs_list.append(probs)
        y_true_list.append(truths)

    y_probs = np.concatenate(y_probs_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.argmax(y_probs, axis=1)
    return y_true, y_pred, y_probs


def print_sample_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, class_names: list, max_display: int = 40):
    n = min(len(y_true), max_display)
    col = 22
    print("\n" + "=" * 72)
    print(f"  {'#':<5} {'Actual':<{col}} {'Predicted':<{col}} {'Confidence':<12} {'Correct?'}")
    print("-" * 72)
    for i in range(n):
        actual = class_names[y_true[i]]
        predicted = class_names[y_pred[i]]
        conf = y_probs[i][y_pred[i]] * 100
        correct = "[OK]" if y_true[i] == y_pred[i] else "[X]"
        print(f"  {i+1:<5} {actual:<{col}} {predicted:<{col}} {conf:<12.2f}% {correct}")
    print("=" * 72)


def evaluate() -> None:
    print("\n" + "=" * 60)
    print("  Custom Animal CNN — Test Evaluation")
    print("=" * 60)

    print("\n[Step 1/5] Loading trained model ...")
    model = load_trained_model()

    print("\n[Step 2/5] Loading class names ...")
    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")

    print("\n[Step 3/5] Loading test dataset ...")
    test_ds, _, _ = load_test_dataset(class_names)

    print("\n[Step 4/5] Computing test metrics ...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    y_true, y_pred, y_probs = collect_predictions(model, test_ds)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    per_cls_acc = per_class_accuracy(y_true, y_pred, class_names)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test Loss              : {test_loss:.4f}")
    print(f"  Test Accuracy          : {test_acc * 100:.2f}%")
    print(f"  Macro Precision        : {macro_p * 100:.2f}%")
    print(f"  Macro Recall           : {macro_r * 100:.2f}%")
    print(f"  Macro F1-Score         : {macro_f1 * 100:.2f}%")
    print("-" * 60)
    for i, cls in enumerate(class_names):
        print(f"  {cls:<22} Prec: {precision[i]*100:>6.2f}% | Rec: {recall[i]*100:>6.2f}% | F1: {f1[i]*100:>6.2f}% | Acc: {per_cls_acc[cls]*100:>6.2f}% (N={int(support[i])})")
    print("=" * 60)

    print_sample_predictions(y_true, y_pred, y_probs, class_names)

    print("\n[Step 5/5] Saving evaluation artefacts ...")
    plot_confusion_matrix(y_true, y_pred, class_names)
    save_classification_report(y_true, y_pred, class_names, test_loss, test_acc)

    print("\n  Evaluation completed.")


if __name__ == "__main__":
    evaluate()
