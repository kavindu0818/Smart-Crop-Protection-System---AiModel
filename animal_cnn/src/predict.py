"""
=============================================================================
predict.py — OpenCV Prediction Module for Custom Animal CNN
=============================================================================
Usage:
    python -m src.predict --image sample_images/test_image.jpg
"""

import os
import sys
import argparse
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import cv2
import tensorflow as tf

from src import config


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

    return sorted(e.name for e in os.scandir(config.TEST_DIR) if e.is_dir())


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    h, w = config.IMG_SIZE
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm = rgb.astype(np.float32) / 255.0
    return np.expand_dims(norm, axis=0)


def draw_prediction_overlay(image_bgr: np.ndarray, label: str, confidence: float, is_unknown: bool) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]

    banner_h = max(60, int(h * 0.14))
    overlay = img.copy()
    banner_color = (40, 30, 20) if is_unknown else (20, 60, 20)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    conf_pct = confidence * 100
    main_text = f"Predicted: {label}"
    conf_text = f"Confidence: {conf_pct:.2f}%"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.5, w / 900)
    thickness = max(1, int(font_scale * 1.8))
    text_color = (0, 200, 80) if not is_unknown else (0, 180, 220)
    conf_color = (180, 255, 180) if not is_unknown else (180, 230, 255)

    (tw, th), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
    y1 = int(banner_h * 0.40) + th // 2
    cv2.putText(img, main_text, (10, y1), font, font_scale, text_color, thickness, cv2.LINE_AA)

    (tw2, th2), _ = cv2.getTextSize(conf_text, font, font_scale * 0.85, thickness)
    y2 = int(banner_h * 0.78) + th2 // 2
    cv2.putText(img, conf_text, (10, y2), font, font_scale * 0.85, conf_color, thickness - 1, cv2.LINE_AA)

    bar_y = banner_h + 4
    bar_h = 6
    bar_fill_w = int(w * confidence)
    bar_color = (0, 200, 80) if not is_unknown else (0, 180, 220)
    cv2.rectangle(img, (0, bar_y), (w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(img, (0, bar_y), (bar_fill_w, bar_y + bar_h), bar_color, -1)

    return img


def predict_image(image_path: str, threshold: float = config.CONFIDENCE_THRESHOLD, show_window: bool = True) -> dict:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"[predict] Image not found: {image_path}")

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"[predict] Trained model not found at: {config.MODEL_SAVE_PATH}\n"
            "Please run  python -m src.train  first."
        )

    class_names = load_class_names()
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"[predict] OpenCV failed to load image: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]
    tensor = preprocess_image(image_bgr)
    probs = model(tensor, training=False).numpy()[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    is_unknown = confidence < threshold
    label = "Unknown / Low Confidence" if is_unknown else class_names[pred_idx]

    print("\n" + "=" * 44)
    print("  Animal Classification Result")
    print("=" * 44)
    if is_unknown:
        print(f"  Prediction        : Unknown / Low Confidence")
        print(f"  Confidence        : {confidence*100:.2f}%")
    else:
        print(f"  Predicted Animal  : {label}")
        print(f"  Confidence        : {confidence*100:.2f}%")
    print("-" * 44)
    print("  All class probabilities:")
    for cls, prob in zip(class_names, probs):
        bar = "█" * int(prob * 20)
        print(f"    {cls:<20} {prob*100:5.2f}%  {bar}")
    print("=" * 44 + "\n")

    annotated = draw_prediction_overlay(image_bgr, label, confidence, is_unknown)

    if show_window:
        window_title = "Animal CNN — Prediction Result [Press any key to close]"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        disp_w = min(orig_w, 900)
        disp_h = int(orig_h * (disp_w / orig_w))
        cv2.resizeWindow(window_title, disp_w, disp_h)
        cv2.imshow(window_title, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "label": label,
        "confidence": confidence,
        "is_unknown": is_unknown,
        "all_probs": {cls: float(p) for cls, p in zip(class_names, probs)},
    }


def main():
    parser = argparse.ArgumentParser(description="Predict animal class from an image.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--threshold", "-t", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Skip GUI display window")
    args = parser.parse_args()

    predict_image(image_path=args.image, threshold=args.threshold, show_window=not args.no_display)


if __name__ == "__main__":
    main()
