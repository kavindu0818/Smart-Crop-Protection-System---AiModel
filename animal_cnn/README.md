# 🦁 Custom CNN Animal Classification System

> **Academic Research Project** | Custom CNN built from scratch using TensorFlow/Keras | OpenCV Inference

---

## Table of Contents

1. [Research Overview](#research-overview)
2. [Project Structure](#project-structure)
3. [Why CNN?](#why-cnn)
4. [Why Custom CNN (No Transfer Learning)?](#why-custom-cnn)
5. [CNN Architecture](#cnn-architecture)
6. [Dataset Structure](#dataset-structure)
7. [Image Preprocessing](#image-preprocessing)
8. [Data Augmentation](#data-augmentation)
9. [Training Methodology](#training-methodology)
10. [Evaluation Methodology](#evaluation-methodology)
11. [OpenCV Prediction Pipeline](#opencv-prediction-pipeline)
12. [Quick Start](#quick-start)
13. [Usage — Jupyter Notebook](#usage--jupyter-notebook)
14. [Usage — Python Scripts](#usage--python-scripts)
15. [Model Limitations](#model-limitations)
16. [Future Improvements](#future-improvements)

---

## Research Overview

This project implements a **completely custom Convolutional Neural Network (CNN)** for multi-class animal image classification. The model is designed, implemented, and trained entirely from scratch — no pretrained weights, no transfer learning, no existing backbones.

| Property | Value |
|---|---|
| Task | Multi-class image classification |
| Classes | Auto-detected from dataset folders |
| Framework | TensorFlow / Keras |
| Architecture | Custom 4-block CNN |
| Pretrained weights | ❌ None |
| Transfer learning | ❌ None |
| Input size | 224 × 224 × 3 |
| Inference engine | OpenCV + TensorFlow |

---

## Project Structure

```text
animal_cnn/
│
├── Animal_CNN_Research.ipynb   ← Main research notebook
│
├── dataset/
│   ├── train/
│   │   ├── bird/
│   │   ├── cow/
│   │   ├── elephant/
│   │   ├── monkey/
│   │   └── pig/
│   ├── validation/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── dataset_samples.png
│   ├── class_distribution.png
│   ├── augmentation_preview.png
│   ├── correct_predictions.png
│   ├── incorrect_predictions.png
│   └── training_log.csv
│
├── models/
│
├── sample_images/              ← Place new images here for prediction
│
├── animal_cnn.keras            ← Saved best model (generated after training)
├── class_names.json            ← Class mapping (generated after training)
│
├── src/                        ← Modular Python scripts (optional)
│   ├── config.py
│   ├── dataset_loader.py
│   ├── cnn_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── generate_notebook.py        ← Script that regenerates the .ipynb
├── requirements.txt
└── README.md
```

---

## Why CNN?

Convolutional Neural Networks are the most suitable architecture for image classification because:

1. **Local Feature Detection**: Conv filters detect spatially local patterns (edges, textures) regardless of their position in the image.
2. **Parameter Efficiency**: Weight sharing across the entire image drastically reduces the number of parameters compared to fully connected networks.
3. **Hierarchical Learning**: Stacking multiple conv layers allows the network to build increasingly abstract feature representations.
4. **Translation Equivariance**: Animals at different positions in the frame are recognised correctly without explicit location encoding.
5. **Proven Performance**: CNNs have consistently achieved state-of-the-art results on image classification benchmarks since AlexNet (2012).

---

## Why Custom CNN?

This research project requires demonstrating independent CNN design capability:

- **Research Integrity**: The architecture is entirely attributable to this project — not borrowed from ImageNet-era models.
- **Domain Specificity**: The architecture is sized and shaped specifically for this 5-class animal classification problem, not general large-scale recognition.
- **No Domain Mismatch**: Transfer learning introduces biases from source domain (ImageNet) that may not be appropriate for this agricultural/wildlife context.
- **Transparency**: Every layer, every weight initialisation, every regularisation choice is explicitly documented and under researcher control.
- **Academic Requirement**: The research mandate is to demonstrate that effective classification can be achieved with a custom-built model.

---

## CNN Architecture

### Design Philosophy

The architecture uses **4 progressive convolutional blocks** with increasing filter depth (32 → 64 → 128 → 256), each consisting of two consecutive 3×3 Conv layers.

**Why two Conv layers per block?**  
Two consecutive 3×3 convolutions share the same effective receptive field as a single 5×5 convolution but with fewer parameters and an additional non-linearity — a design principle validated in deep learning literature.

### Architecture Diagram

```
Input (224 × 224 × 3)
        ↓
┌─ Conv Block 1 ─────────────────────────────┐
│  Conv2D(32, 3×3, padding=same) → BN → ReLU │
│  Conv2D(32, 3×3, padding=same) → BN → ReLU │  Spatial: 224×224
│  MaxPool2D(2×2)                             │  → 112×112
└────────────────────────────────────────────┘
        ↓
┌─ Conv Block 2 ─────────────────────────────┐
│  Conv2D(64, 3×3, padding=same) → BN → ReLU │
│  Conv2D(64, 3×3, padding=same) → BN → ReLU │
│  MaxPool2D(2×2)                             │  → 56×56
└────────────────────────────────────────────┘
        ↓
┌─ Conv Block 3 ──────────────────────────────┐
│  Conv2D(128, 3×3, padding=same) → BN → ReLU │
│  Conv2D(128, 3×3, padding=same) → BN → ReLU │
│  MaxPool2D(2×2)                              │  → 28×28
└─────────────────────────────────────────────┘
        ↓
┌─ Conv Block 4 ──────────────────────────────┐
│  Conv2D(256, 3×3, padding=same) → BN → ReLU │
│  Conv2D(256, 3×3, padding=same) → BN → ReLU │
│  MaxPool2D(2×2)                              │  → 14×14
└─────────────────────────────────────────────┘
        ↓
  GlobalAveragePooling2D  →  (256,)
        ↓
  Dense(512, activation=relu, L2 regularisation)
        ↓
  Dropout(0.40)
        ↓
  Dense(N, activation=softmax)    ← N = number of animal classes
        ↓
  Output: Class Probabilities
```

### Layer Details

| Component | Specification | Purpose |
|---|---|---|
| Conv2D | 3×3 kernel, same padding | Local feature extraction |
| Kernel initialisation | He Normal | Optimal for ReLU networks |
| use_bias | False (with Batch Norm) | BN makes bias redundant |
| Batch Normalisation | After each Conv | Training stability, faster convergence |
| Activation | ReLU | Non-linearity, avoids vanishing gradient |
| MaxPool2D | 2×2, stride 2 | Spatial downsampling, translation invariance |
| Global Average Pooling | — | Replaces Flatten; reduces parameters, overfitting |
| Dense(512) | ReLU + L2 | Final feature combination |
| Dropout | 0.40 | Regularisation against overfitting |
| Dense(N) | Softmax | Probability distribution over classes |

---

## Dataset Structure

Provide images in this exact folder structure:

```text
dataset/
├── train/
│   ├── bird/          ← JPG/PNG images of birds
│   ├── cow/
│   ├── elephant/
│   ├── monkey/
│   └── pig/
├── validation/
│   ├── bird/
│   ├── cow/
│   ├── elephant/
│   ├── monkey/
│   └── pig/
└── test/
    ├── bird/
    ├── cow/
    ├── elephant/
    ├── monkey/
    └── pig/
```

- **No annotation files required** — labels come from folder names automatically.
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`
- **Recommended split**: ~70% train / 15% validation / 15% test

The notebook auto-detects class names from the sub-folder names — **you never need to hard-code them**.

---

## Image Preprocessing

| Step | Details |
|---|---|
| Loading | TensorFlow (training) or OpenCV cv2.imread() (inference) |
| Resize | (224, 224) using bilinear interpolation |
| Colour space | BGR → RGB conversion for OpenCV images |
| Normalisation | Pixel values divided by 255.0 → [0.0, 1.0] |
| Tensor shape | (1, 224, 224, 3) for single-image inference |

The **same normalisation** is applied during training and inference — preventing preprocessing mismatch.

---

## Data Augmentation

Applied **only to the training split**. Validation and test images are never augmented.

| Transform | Setting | Rationale |
|---|---|---|
| Horizontal Flip | Enabled | Animals may face either direction |
| Rotation | ±15% | Camera tilt variation |
| Zoom | ±10% | Distance from the animal |
| Translation | ±10% H & W | Animal not always centred |
| Brightness | ±15% delta | Lighting variation (day/night/shade) |

---

## Training Methodology

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Max Epochs | 60 |
| Early Stopping | patience=12, monitor=val_loss |
| LR Reduction | factor=0.5, patience=5, min=1e-6 |
| Model Checkpoint | Save best val_accuracy |
| Class Weights | Balanced inverse-frequency |
| Regularisation | L2 (1e-4) + Dropout (0.40) |
| Random Seed | 42 (fully reproducible) |

---

## Evaluation Methodology

The model is evaluated on the **independent test set** (never seen during training or hyperparameter selection).

Metrics computed:
- **Test Accuracy**: Overall fraction of correct predictions
- **Precision** (per-class & macro): Of all predicted positives, how many were correct
- **Recall** (per-class & macro): Of all actual positives, how many were found
- **F1-Score** (per-class & macro): Harmonic mean of precision and recall
- **Confusion Matrix**: Visualises which classes are confused with each other
- **Per-Class Accuracy**: Individual class performance
- **Incorrect Prediction Gallery**: Visual analysis of misclassified samples

---

## OpenCV Prediction Pipeline

```
New Image File
      ↓
cv2.imread(image_path)          ← Load as BGR array
      ↓
Validate (not None, file exists)
      ↓
cv2.cvtColor(BGR → RGB)
      ↓
cv2.resize(224, 224)
      ↓
/ 255.0 normalisation
      ↓
np.expand_dims → (1, 224, 224, 3)
      ↓
model(tensor, training=False)   ← Custom CNN inference
      ↓
Softmax probabilities[0]        ← (num_classes,)
      ↓
np.argmax → predicted class index
      ↓
Confidence threshold check (default: 0.60)
      ↓
If conf >= threshold:  Animal Class + Confidence
If conf <  threshold:  Unknown / Low Confidence
      ↓
cv2.putText() overlay on image
      ↓
cv2.imshow() display
```

### Example Output

```text
==========================================
  Animal Classification Result
==========================================

  Predicted Animal  : Elephant
  Confidence        : 94.72%

------------------------------------------
  All class probabilities:
    bird                    0.01%  
    cow                     0.02%  
    elephant               94.72%  ████████████████████████
    monkey                  0.02%  
    pig                     5.23%  █
==========================================
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Add Your Dataset

Place images in `dataset/train/`, `dataset/validation/`, `dataset/test/` following the folder structure above.

### 3. Open the Research Notebook

```bash
jupyter notebook Animal_CNN_Research.ipynb
```

Run all cells from top to bottom.

### 4. Predict a New Image

Inside the notebook (Section 17):

```python
IMAGE_PATH = "sample_images/your_image.jpg"
result = predictor.predict_notebook(IMAGE_PATH)
```

Or using the Python script:

```bash
python -m src.predict --image sample_images/your_image.jpg
```

---

## Usage — Jupyter Notebook

The main deliverable is `Animal_CNN_Research.ipynb`. Open it in Jupyter Lab or Jupyter Notebook and run cells sequentially.

| Section | Description |
|---|---|
| 01 | Research objective and constraints |
| 02 | Library imports |
| 03 | Config class (edit paths here) |
| 04 | Dataset validation and statistics |
| 05 | Data leakage detection |
| 06 | Class imbalance analysis |
| 07 | Dataset visualisation |
| 08 | ImagePreprocessor + tf.data pipelines |
| 09 | Augmentation preview |
| 10 | CustomCNN architecture |
| 11 | ModelTrainer — compile and train |
| 12 | Training/validation graphs |
| 13 | ModelEvaluator — test metrics |
| 14 | Incorrect prediction analysis |
| 15 | Model and class-name saving |
| 16 | OpenCVPredictor class |
| 17 | New image prediction |
| 18 | Confidence threshold demo |
| 19 | Hyperparameter experiment notes |
| 20 | Final research summary table |
| 21 | Full research documentation |
| 22 | Conclusion |

---

## Usage — Python Scripts

The `src/` folder contains standalone modules if you prefer running scripts:

```bash
# Train
python -m src.train

# Evaluate on test set
python -m src.evaluate

# Predict a single image
python -m src.predict --image sample_images/test.jpg --threshold 0.70
```

---

## Model Limitations

| Limitation | Impact |
|---|---|
| Dataset size | Smaller datasets increase overfitting risk |
| Background bias | Model may learn habitat cues rather than animal features |
| Visual similarity | Superficially similar species may be confused |
| Occlusion | Partially visible animals reduce confidence |
| Resolution | Very low-resolution images may lack discriminative detail |
| Lighting extremes | Very dark or overexposed images may degrade performance |
| Pose/Angle | Rare viewing angles not in training data may be misclassified |

---

## Future Improvements

1. **Attention Mechanisms**: CBAM or SE blocks to focus on discriminative body parts
2. **Test-Time Augmentation (TTA)**: Average predictions over multiple augmented views
3. **Class Activation Maps (CAM)**: Visualise which image regions drive the classification decision
4. **Larger & More Diverse Dataset**: More images per class from varied conditions
5. **Ensemble Learning**: Combine multiple independently trained CNNs
6. **Uncertainty Quantification**: Monte Carlo Dropout for confidence calibration
7. **Learned Augmentation**: AutoAugment / RandAugment policies
8. **Deeper Architecture**: Add a 5th convolutional block for larger datasets
9. **Background Removal Preprocessing**: Isolate the animal from background noise
10. **Grad-CAM Visualisation**: Explain model decisions for research transparency

---

## Reproducibility

All random seeds are fixed:

```python
RANDOM_SEED = 42
random.seed(42)
numpy.random.seed(42)
tensorflow.random.set_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
```

This ensures that model weight initialisation, data shuffling, and augmentation randomness are consistent across runs on the same hardware.

---

*Custom CNN Animal Classification — Academic Research Project*  
*Built with TensorFlow · Keras · OpenCV · Scikit-learn*
