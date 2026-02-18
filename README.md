# DermAssist AI — Clinical Decision Support for Skin Lesion Analysis

DermAssist AI is an **offline** desktop application for **multi-class dermoscopic skin lesion classification** using **EfficientNet-V2-S**. It provides **Top-3 predictions**, **confidence scores**, and **Grad-CAM++ explainability heatmaps** inside a modern **PyQt6 GUI**, with **privacy-preserving local processing**.

> ⚠️ **Disclaimer:** This project is **NOT a medical device**. It is intended for **research and educational** purposes only. Do **NOT** use it to diagnose or treat patients.

---

## Features

- **7-class skin lesion classification** (HAM10000 label space)
- **EfficientNet-V2-S** with **transfer learning** (ImageNet pretrained)
- **Explainability** via **Grad-CAM++** heatmaps
- **Top-3 predictions** table + confidence progress bar
- **Test-Time Augmentation (TTA)** option in inference
- **Dermoscopy-oriented preprocessing**
  - Hair artifact removal (morphology + inpainting)
  - Contrast enhancement (CLAHE)
- **Privacy-first**
  - Face anonymization (Haar Cascade + blur)
  - Fully **offline** (no uploads, no server)

---

## Classes

The model predicts one of the following categories:

- Actinic Keratoses  
- Basal Cell Carcinoma  
- Benign Keratosis-like Lesions  
- Dermatofibroma  
- Melanocytic Nevi  
- Melanoma  
- Vascular Lesions  

---

## Project Structure

```text
.
├── main.py                     # GUI entry point
├── requirements.txt            # Dependencies
├── prepare_ham10000.py         # Dataset preparation (group-aware split)
├── train.py                    # Training script
├── train.txt                   # Example training log
├── models/
│   ├── skin_classifier.pth     # Trained checkpoint (generated)
│   └── skin_classifier_metadata.txt  # Optional (generated)
├── data/                       # Prepared dataset output (generated)
│   ├── train/
│   └── val/
└── src/                        # Application modules (engine/ui/utils)
```

---

## Installation

### 1) Create and activate a virtual environment

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset (HAM10000)

This project is trained on the **HAM10000** dermoscopic dataset (10,015 images, 7 classes).  
You must have:

- `HAM10000_metadata.csv`
- a folder containing all images (e.g., `ham10000/images/`)

---

## Step-by-step: Prepare Data → Train → Run GUI

### Step 1 — Prepare the dataset (ImageFolder format)

```bash
python prepare_ham10000.py \
  --meta ham10000/HAM10000_metadata.csv \
  --images ham10000/images \
  --out data \
  --val-ratio 0.15 \
  --mode hardlink
```

This creates:
```text
data/
├── train/<class_name>/
└── val/<class_name>/
```

**Note:** The split is **group-aware** using `lesion_id` to prevent **data leakage** (the same lesion cannot appear in both train and val).

---

### Step 2 — Train the model

```bash
mkdir models

python train.py \
  --mode imagefolder \
  --data-root data \
  --pretrained \
  --use-weighted-sampler \
  --use-class-weights \
  --save-metadata \
  --out models/skin_classifier.pth
```

Outputs:
- `models/skin_classifier.pth`
- `models/skin_classifier_metadata.txt` (if `--save-metadata` is enabled)

---

### Step 3 — Run the desktop application

```bash
python main.py
```

In the GUI:
1. Click **Upload Skin Image**
2. Choose a dermoscopic image (JPG/PNG)
3. View:
   - Predicted diagnosis + confidence
   - Top-3 predictions
   - Grad-CAM++ heatmap explanation

---

## Inference Options (GUI)

You can toggle:
- **Hair removal**
- **CLAHE contrast enhancement**
- **Test-time augmentation (TTA)**

These directly affect preprocessing/inference and may change probabilities and heatmap focus.

---

## Example Training Result

An example run (CPU training) achieved:
- **Validation Top-1 Accuracy:** **77.10%**
- **Validation Top-3 Accuracy:** ~**97%**

(See `train.txt` for the full log.)

---

## Privacy & Security

- Fully offline processing (no uploads)
- Face anonymization enabled by default
- Images are processed locally on the user’s device

---

## Acknowledgments

- HAM10000 dataset creators  
- PyTorch / TorchVision  
- PyQt6  
- `pytorch-grad-cam` (Grad-CAM++ implementation)


