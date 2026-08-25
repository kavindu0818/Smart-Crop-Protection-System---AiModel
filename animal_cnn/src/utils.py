"""
=============================================================================
utils.py — Shared Utilities for Custom CNN Animal Classification
=============================================================================
"""

import os
import json
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report

from src import config


def plot_training_history(history, save_path: str = config.HISTORY_PLOT) -> None:
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    lr = history.history.get("lr", [])
    epochs = range(1, len(acc) + 1)

    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#16213e")
    ax1.plot(epochs, acc, color="#00d4aa", linewidth=2, marker="o", markersize=3, label="Train Acc")
    ax1.plot(epochs, val_acc, color="#ff6b6b", linewidth=2, marker="s", markersize=3, label="Val Acc", linestyle="--")
    ax1.set_title("Model Accuracy", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#cccccc")
    ax1.set_ylabel("Accuracy", color="#cccccc")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")
    ax1.tick_params(colors="#cccccc")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#555555")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.2, color="white")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#16213e")
    ax2.plot(epochs, loss, color="#00d4aa", linewidth=2, marker="o", markersize=3, label="Train Loss")
    ax2.plot(epochs, val_loss, color="#ff6b6b", linewidth=2, marker="s", markersize=3, label="Val Loss", linestyle="--")
    ax2.set_title("Model Loss", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#cccccc")
    ax2.set_ylabel("Loss", color="#cccccc")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")
    ax2.tick_params(colors="#cccccc")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#555555")
    ax2.grid(True, alpha=0.2, color="white")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#16213e")
    if lr:
        ax3.plot(epochs, lr, color="#f7b731", linewidth=2, marker="^", markersize=3, label="LR")
        ax3.set_yscale("log")
        ax3.set_title("Learning Rate Schedule", color="white", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Epoch", color="#cccccc")
        ax3.set_ylabel("Learning Rate (log)", color="#cccccc")
        ax3.legend(facecolor="#1a1a2e", labelcolor="white")
    else:
        ax3.text(0.5, 0.5, "LR not logged", ha="center", va="center", color="white")
        ax3.set_title("Learning Rate", color="white", fontsize=13, fontweight="bold")
    ax3.tick_params(colors="#cccccc")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#555555")
    ax3.grid(True, alpha=0.2, color="white")

    fig.suptitle("Custom Animal CNN — Training Progress", color="white", fontsize=15, fontweight="bold", y=1.02)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[utils] Training history plot saved -> {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: str = config.CONFUSION_PLOT) -> None:
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), max(6, n * 1.2)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual Class",
        xlabel="Predicted Class",
        title="Confusion Matrix — Custom Animal CNN",
    )
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", color="white")
    plt.setp(ax.get_yticklabels(), color="white")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] < thresh else "black",
                fontsize=11,
            )

    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[utils] Confusion matrix saved -> {save_path}")


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, test_loss: float, test_acc: float, save_path: str = config.REPORT_PATH) -> str:
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    header = (
        "=" * 60 + "\n"
        "  Custom Animal CNN — Classification Report\n"
        "=" * 60 + "\n"
        f"  Test Loss     : {test_loss:.4f}\n"
        f"  Test Accuracy : {test_acc * 100:.2f}%\n"
        "=" * 60 + "\n\n"
    )
    full_text = header + report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(full_text)
    print(f"[utils] Classification report saved -> {save_path}")
    return report


def save_metadata(meta: dict, save_path: str = config.METADATA_PATH) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[utils] Metadata saved -> {save_path}")


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return {cls: float(acc) for cls, acc in zip(class_names, per_class)}


class Timer:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()
        return self

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    def elapsed_str(self) -> str:
        secs = int(self.elapsed())
        return str(datetime.timedelta(seconds=secs))


def count_images(directory: str, class_names: list) -> dict:
    counts = {}
    for cls in class_names:
        cls_dir = os.path.join(directory, cls)
        if os.path.isdir(cls_dir):
            counts[cls] = len([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ])
        else:
            counts[cls] = 0
    return counts
