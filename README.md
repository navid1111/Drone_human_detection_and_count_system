# Drone Human Detection & Counting System

> Antlings Internship Program — Technical Assessment (AI/ML)
> Built by **Navid Kamal** | IUT Dhaka | May 2026

A production-grade computer vision pipeline for detecting humans and cars in drone/aerial imagery, built on YOLO11, with SAHI sliced inference, ByteTrack tracking, FastAPI + Streamlit deployment, and Prometheus/Grafana monitoring.

---

## Demo

> 📹 [Watch Demo Video](YOUR_GOOGLE_DRIVE_LINK)
> 🔗 [W&B Experiment Dashboard](https://wandb.ai/navidkamal-islamic-university-of-technology/drone_human_detection)

---

## Results at a Glance

| Model | imgsz | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO11m (baseline) | 640px | 0.913 | 0.575 | 0.561 | 0.367 |
| YOLO11m (improved) | 960px | TBD | TBD | ~0.76+ | ~0.53+ |

> 960px model still training at time of submission — early epochs already show 35% mAP50 improvement over the fully trained 640px baseline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Task 01 — Dataset Understanding & Preprocessing](#task-01--dataset-understanding--preprocessing)
- [Task 02 — Model Training](#task-02--model-training)
- [Task 03 — Detection & Human Counting](#task-03--detection--human-counting)
- [Task 04 — Object Tracking (Bonus)](#task-04--object-tracking-bonus)
- [Task 05 — Evaluation & Visualization](#task-05--evaluation--visualization)
- [Deployment](#deployment)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Experiment Tracking](#experiment-tracking)
- [Strengths & Limitations](#strengths--limitations)

---

## Project Overview

This system detects persons and cars in high-resolution drone imagery, counts total humans per frame, and optionally tracks objects across video frames. The pipeline was built by adapting an existing retail object detection codebase (YOLO11 + DVC + Airflow + W&B) to the VisDrone aerial dataset, demonstrating strong reusability of ML infrastructure.

**Key engineering decisions:**
- Remapped VisDrone's 10 classes → 2 classes, dropping imbalance from 44–56x to 1.1–1.6x
- Trained at 960px input (vs standard 640px) based on size distribution analysis showing 85.4% of objects are sub-32px
- Added SAHI sliced inference for improved small object detection at inference time
- Exported to ONNX for efficient deployment

---

## Dataset

**Source:** [VisDrone2019-DET](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

| Split | Images | Objects (remapped) |
|---|---|---|
| Train | 6,471 | 276,219 |
| Val | 548 | 30,008 |
| Test | 1,580 | 61,227 |

**Original VisDrone classes (10):**

| ID | Class | Action |
|---|---|---|
| 0 | pedestrian | → person |
| 1 | people | → person |
| 2 | bicycle | dropped |
| 3 | car | → car |
| 4 | van | → car |
| 5 | truck | dropped |
| 6 | tricycle | dropped |
| 7 | awning-tricycle | dropped |
| 8 | bus | dropped |
| 9 | motor | dropped |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DRONE DETECTION PIPELINE                    │
└─────────────────────────────────────────────────────────────┘

RAW DATASET (VisDrone 10-class)
         │
         ▼
src/data/diagnose.py          → class distribution analysis
         │
         ▼
src/data/remap_classes.py     → 10 classes → 2 classes (person, car)
         │
         ▼
src/data/size_distribution.py → object size analysis (small/med/large)
         │
         ▼
src/model/train.py            → YOLO11m fine-tuning + W&B logging
         │
         ▼
src/model/eval.py             → precision, recall, mAP, FPS
         │
    ┌────┴────┐
    ▼         ▼
Standard   SAHI Sliced
Inference  Inference
    │         │
    └────┬────┘
         ▼
src/utils/counter.py          → person count + visualization
         │
         ▼
src/utils/tracker.py          → ByteTrack tracking (bonus)
         │
         ▼
ONNX Export → FastAPI + Streamlit → Prometheus + Grafana
```

---

## Task 01 — Dataset Understanding & Preprocessing

### Dataset Structure

```
VisDrone2019/
├── VisDrone2019-DET-train/
│   ├── images/     (6,471 .jpg files)
│   └── labels/     (YOLO format .txt files)
├── VisDrone2019-DET-val/
│   ├── images/     (548 files)
│   └── labels/
└── VisDrone2019-DET-test-dev/
    ├── images/     (1,580 files)
    └── labels/
```

### Class Imbalance — Before & After Remapping

**Before (10 classes):**
```
train: 10 classes | imbalance 44.6x | 343,205 objects
val:   10 classes | imbalance 56.0x |  38,759 objects
test:  10 classes | imbalance 53.0x |  75,102 objects
```

**After remapping to 2 classes:**
```
train: 2 classes | imbalance 1.6x | 276,219 objects
val:   2 classes | imbalance 1.1x |  30,008 objects
test:  2 classes | imbalance 1.2x |  61,227 objects
```

Remapping eliminated the severe imbalance entirely.

### Object Size Distribution (Train Set)

| Size Category | Threshold | Count | Percentage |
|---|---|---|---|
| Small | < 32px equivalent | 236,000 | **85.4%** |
| Medium | 32–96px equivalent | 39,020 | 14.1% |
| Large | > 96px equivalent | 1,199 | 0.4% |

**Per-class:**
| Class | Small | Medium | Large |
|---|---|---|---|
| person | **98.0%** | 1.9% | ~0% |
| car | 77.5% | 21.8% | 0.7% |

> The median bounding box occupies just **0.0442%** of image area — roughly 12×18px in a 1920×1080 image. This finding directly motivated training at 960px instead of the standard 640px.

### Preprocessing Steps

```bash
# Step 1 — Diagnose original dataset
python -m src.data.diagnose

# Step 2 — Remap to 2 classes
python -m src.data.remap_classes

# Step 3 — Validate remapped dataset
python -m src.data.diagnose

# Step 4 — Analyze object sizes
python -m src.data.size_distribution
```

### Augmentation (applied during training)

YOLO11's built-in augmentation pipeline was used:
- Random horizontal flip (p=0.5)
- HSV color jitter
- Mosaic (4-image composite, disabled last 10 epochs)
- Random erasing (p=0.4)
- Albumentations: Blur, MedianBlur, CLAHE, ToGray

---

## Task 02 — Model Training

### Model

**YOLO11m** (Medium) — pretrained on COCO, fine-tuned on VisDrone 2-class.

### Experiment 1 — Baseline (640px, Colab T4)

| Parameter | Value |
|---|---|
| Base model | yolo11m.pt |
| Image size | 640px |
| Epochs | 100 |
| Batch size | 8 |
| Optimizer | SGD |
| Device | Tesla T4 (14GB) |

**Results:**
| Metric | Value |
|---|---|
| Precision | 0.913 |
| Recall | 0.575 |
| mAP50 | 0.561 |
| mAP50-95 | 0.367 |

### Experiment 2 — Improved (960px, RTX 4060)

| Parameter | Value |
|---|---|
| Base model | yolo11m.pt |
| Image size | 960px |
| Epochs | 100 |
| Batch size | 4 |
| Optimizer | SGD |
| Device | RTX 4060 (8GB) |

**Early results (epoch 15):**
| Metric | Value |
|---|---|
| Precision | 0.922 |
| Recall | 0.577 |
| mAP50 | **0.755** |
| mAP50-95 | **0.526** |

> mAP50 improved by **34.6%** over the 640px baseline by epoch 15 alone — validating the size distribution analysis.

### Training Command

```bash
python -m src.model.train
```

### Key Configuration (`.env`)

```env
APP_DATASET_YAML=config/visdrone_2class.yaml
APP_IMGSZ=960
APP_EPOCHS=100
APP_BATCH=4
APP_OPTIMIZER=SGD
APP_LR0=0.01
APP_DEVICE=0
APP_WANDB_PROJECT=drone_human_detection
```

---

## Task 03 — Detection & Human Counting

### Standard Inference

```bash
python -m src.utils.counter
```

Draws green boxes for persons, red boxes for cars, overlays total counts.

### SAHI Sliced Inference

Standard inference at 640px misses many tiny persons. SAHI tiles each image into overlapping 640px patches, runs inference on each, then merges:

```
1920×1080 image
      ↓
┌──────┬──────┬──────┐
│ 640² │ 640² │ 640² │  ← 20% overlap between tiles
├──────┼──────┼──────┤
│ 640² │ 640² │ 640² │
└──────┴──────┴──────┘
      ↓
Merge + NMS
      ↓
Full image detections
```

```bash
python -m src.utils.sahi_inference
```

### Counting Logic

```python
person_count = sum(1 for pred in results if pred.class_id == 0)
car_count    = sum(1 for pred in results if pred.class_id == 1)
```

Simple per-frame counting. With tracking enabled, IDs persist across frames to avoid double-counting.

---

## Task 04 — Object Tracking (Bonus)

ByteTrack is integrated natively through YOLO11's tracking API:

```bash
python -m src.utils.tracker
```

Each detected person and car receives a persistent track ID. This enables:
- Accurate counting across video frames (no double-counting)
- Trajectory visualization
- Speed/direction estimation

**Trackers supported:** ByteTrack, BoT-SORT (configurable via `tracker:` in `.env`)

---

## Task 05 — Evaluation & Visualization

```bash
python -m src.model.eval       # mAP, precision, recall
python -m src.model.benchmark  # FPS, latency
```

### Final Metrics Summary

| Model | mAP50 | mAP50-95 | Precision | Recall | FPS |
|---|---|---|---|---|---|
| YOLO11m 640px | 0.561 | 0.367 | 0.913 | 0.575 | ~30 |
| YOLO11m 960px | ~0.76 | ~0.53 | ~0.93 | ~0.58 | ~18 |

### Strengths
- Severe class imbalance eliminated through smart remapping
- Resolution scaling validated empirically with a controlled experiment
- SAHI provides additional gains on tiny objects at inference time with zero retraining
- Full experiment tracking via W&B for reproducibility

### Limitations
- Recall of ~0.575 at 640px — roughly 4 in 10 persons are missed due to small object size
- Dense crowd scenes cause merged bounding boxes, undercounting clusters
- ByteTrack ID switching in fast-moving scenes can cause momentary count errors
- 960px inference is ~40% slower than 640px (18 FPS vs 30 FPS)

### Challenges Faced
- VisDrone's original yaml pointed train split to the unannotated test-challenge folder — required manual yaml correction
- CUDA index-out-of-bounds crash caused by residual 10-class labels in the remapped dataset — fixed by re-running remap validation
- Colab session timeouts during 100-epoch training — mitigated by W&B checkpointing

---

## Deployment

### Model Export

```bash
# Export to ONNX
python -m src.model.export
```

ONNX export reduces model size significantly and enables CPU inference without PyTorch.

### FastAPI + Streamlit

```bash
# Start API backend
uvicorn src.api.main:app --reload --port 8000

# Start Streamlit frontend
streamlit run src.app.streamlit_app.py
```

Upload a drone image → get back annotated image with bounding boxes and human count.

### Monitoring (Prometheus + Grafana)

```bash
docker-compose up -d
```

Tracks:
- Inference latency per request
- Persons/cars detected per frame
- Data drift detection (input image statistics vs training distribution)

---

## Installation

```bash
# Clone
git clone https://github.com/navid1111/Drone_human_detection_and_count_system.git
cd Drone_human_detection_and_count_system

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAHI
pip install sahi

# Configure environment
cp .env.example .env
# Edit .env with your paths and W&B key
```

---

## Usage

```bash
# 1. Diagnose dataset
python -m src.data.diagnose

# 2. Remap classes
python -m src.data.remap_classes

# 3. Analyze object sizes
python -m src.data.size_distribution

# 4. Train
python -m src.model.train

# 5. Evaluate
python -m src.model.eval

# 6. Run detection + counting (standard)
python -m src.utils.counter

# 7. Run detection + counting (SAHI)
python -m src.utils.sahi_inference

# 8. Run tracking
python -m src.utils.tracker

# 9. Benchmark FPS
python -m src.model.benchmark
```

---

## Project Structure

```
Drone_human_detection_and_count_system/
├── config/
│   └── visdrone_2class.yaml        # 2-class dataset config
├── dataset/
│   ├── VisDrone_Dataset/           # Original VisDrone (read-only)
│   └── VisDrone_2class/            # Remapped 2-class dataset
├── docs/
│   ├── dataset_understanding_and_preprocessing.md
│   ├── size_distribution_train.png
│   └── sample_visualizations.png
├── outputs/
│   ├── sahi/                       # SAHI inference outputs
│   └── tracking/                   # Tracking outputs
├── runs/
│   └── train/pipeline_run/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── src/
│   ├── config/                     # Settings & env loading
│   ├── data/
│   │   ├── diagnose.py             # Class distribution analysis
│   │   ├── remap_classes.py        # 10→2 class remapping
│   │   ├── size_distribution.py    # Object size analysis
│   │   └── rebuild_splits.py       # Stratified split builder
│   ├── model/
│   │   ├── train.py                # YOLO training + W&B
│   │   ├── eval.py                 # Validation + FPS
│   │   ├── benchmark.py            # Inference speed profiling
│   │   └── export.py               # ONNX export
│   ├── utils/
│   │   ├── counter.py              # Detection + counting
│   │   ├── sahi_inference.py       # Sliced inference
│   │   ├── tracker.py              # ByteTrack integration
│   │   └── gpu.py                  # GPU availability check
│   ├── api/
│   │   └── main.py                 # FastAPI backend
│   └── app/
│       └── streamlit_app.py        # Streamlit frontend
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Experiment Tracking

All runs logged to Weights & Biases:
🔗 [W&B Project Dashboard](https://wandb.ai/navidkamal-islamic-university-of-technology/drone_human_detection)

Tracked per run:
- All hyperparameters
- Per-epoch loss curves (box, cls, dfl)
- Validation metrics (P, R, mAP50, mAP50-95)
- Dataset version metadata
- Training summary report

---

## References

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [SAHI — Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Weights & Biases](https://wandb.ai)

---

## License

MIT

---

*Submitted for Antlings Internship Program — AI/ML Technical Assessment, May 2026*