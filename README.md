<div align="center">

<!-- BANNER -->
<img src="demo/banner.png" alt="Helmet Violation Detector" width="100%"/>

<br/>

<h1>
  <img src="https://img.icons8.com/fluency/48/000000/helmet.png" width="36" style="vertical-align:middle"/> 
  Helmet Violation Detector
</h1>

<p><strong>Automated traffic surveillance powered by multi-stage YOLOv8</strong></p>

<p>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>
  <a href="https://ultralytics.com"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFF?style=for-the-badge&logo=yolo&logoColor=white"/></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/></a>
  <a href="https://opencv.org"><img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/></a>
</p>

<p>
  <a href="#-overview">Overview</a> ·
  <a href="#-pipeline">Pipeline</a> ·
  <a href="#-model-performance">Performance</a> ·
  <a href="#-quickstart">Quickstart</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-project-structure">Structure</a> ·
  <a href="#-roadmap">Roadmap</a>
</p>

<br/>

> **Detects helmet violations in traffic video · Extracts license plates · Exports evidence**  
> Built for real-world traffic enforcement scenarios — fast, modular, and production-ready.

<br/>

---

</div>

## 🌍 Overview

Helmet violations are a **leading cause of fatal road accidents**, especially in high-density urban traffic. Manual monitoring is slow, inconsistent, and impossible to scale across city-wide camera networks.

**Helmet Violation Detector** is an end-to-end automated enforcement pipeline that:

- 🔍 **Detects** motorcycles and riders in each video frame
- 🪖 **Classifies** whether each rider is wearing a helmet
- 🔢 **Extracts** license plates from violating vehicles
- 📋 **Logs** all violations with annotated evidence frames and metadata
- 📥 **Exports** structured reports ready for enforcement action

All of this runs from a single traffic video, with a clean Streamlit dashboard and zero manual intervention.

---

## ⚙️ Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         HELMET VIOLATION DETECTOR PIPELINE                      │
├─────────────┬──────────────────┬──────────────────┬──────────────┬─────────────┤
│  Stage 1    │    Stage 2       │    Stage 3       │   Stage 4    │   Stage 5   │
│             │                  │                  │              │             │
│  Input      │  Bike & Rider    │  Helmet & LP     │  Violation   │  Evidence   │
│  Video  ──► │  Detection   ──► │  Detection   ──► │  Logic   ──► │  Export     │
│             │  (YOLOv8)        │  (YOLOv8)        │              │             │
└─────────────┴──────────────────┴──────────────────┴──────────────┴─────────────┘
```

### Stage 1 — Video Ingestion
Raw traffic video is ingested frame by frame. Each frame is preprocessed and passed downstream for inference.

### Stage 2 — Bike & Rider Detection
A fine-tuned YOLOv8 model identifies **motorcycles** and **persons** with near-perfect accuracy (`mAP@0.5 = 0.994`). Only relevant bounding regions are cropped and forwarded — minimizing compute overhead.

### Stage 3 — Helmet & License Plate Classification
A second YOLOv8 model operates on the cropped regions, classifying each detected rider into one of three categories:

| Class | Label | Description |
|-------|-------|-------------|
| ✅ | `helmet` | Rider is wearing a helmet |
| ❌ | `no_helmet` | Rider is not wearing a helmet |
| 🔢 | `license_plate` | License plate region detected |

### Stage 4 — Violation Logic & Smart Filtering

A violation is triggered **only when a rider is continuously detected without a helmet for ≥ 1 second** inside the active detection zone. This temporal threshold dramatically reduces false positives from brief occlusion events.

Additional filtering mechanisms:

| Filter | Purpose |
|--------|---------|
| **Size threshold** | Ignores vehicles that are too small/distant to classify reliably |
| **IoU-based bike tracking** | Assigns persistent IDs across frames without a dedicated tracker |
| **Cooldown mechanism** | Prevents duplicate violation saves for the same vehicle |

### Stage 5 — Evidence Generation

For every confirmed violation, the system automatically generates:

| Output | Description |
|--------|-------------|
| 📸 Annotated frame | The exact frame of violation, saved to disk |
| 🔢 License plate crop | Best-confidence plate region, cached across all frames per bike |
| 📋 Metadata log | Frame number, timestamp, bike ID, confidence scores |
| 📹 Annotated video | Full output video with overlays, downloadable from dashboard |
| 📥 CSV export | Structured violation log for enforcement reporting |

---

## 📊 Model Performance

### 🚦 Stage 1: Bike + Person Detector

| Metric | Score |
|--------|-------|
| **Precision** | 0.976 |
| **Recall** | 0.979 |
| **mAP@0.5** | **0.994** ✅ |
| **mAP@0.5:0.95** | 0.839 |

> Near-perfect detection performance. Misses are extremely rare — suitable for production deployment.

---

### 🪖 Stage 2: Helmet + License Plate Detector

| Metric | Score |
|--------|-------|
| **Precision** | 0.921 |
| **Recall** | 0.927 |
| **mAP@0.5** | **0.937** ✅ |
| **mAP@0.5:0.95** | 0.696 |

**Class-wise mAP@0.5 breakdown:**

| Class | mAP@0.5 | Notes |
|-------|---------|-------|
| 🔢 License Plate | **0.871** | Strong — robust evidence capture |
| ✅ Helmet | **0.619** | Moderate — impacted by occlusion & motion blur |
| ❌ No Helmet | **0.598** | Moderate — challenging at small scales |

> **Helmet classification is the harder sub-problem.** Key challenges include small object sizes at typical traffic camera distances, partial occlusion by hair or clothing, and motion blur at high frame rates. These are active areas for improvement (see [Roadmap](#-roadmap)).

---

## 🧠 Datasets

Both models were trained on open Roboflow datasets and fine-tuned for real-world traffic conditions.

| Model | Dataset | Source |
|-------|---------|--------|
| Bike + Person Detector | Motobike Detection | [Roboflow Universe](https://universe.roboflow.com/cdio-zmfmj/motobike-detection) |
| Helmet + License Plate Detector | Helmet LP Detection | [Roboflow Universe](https://universe.roboflow.com/cdio-zmfmj/helmet-lincense-plate-detection-gevlq) |

---

## 🚀 Quickstart

### Prerequisites

- Python 3.8 or higher
- Git
- A CUDA-capable GPU is recommended but not required

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/helmet-violation-detector.git
cd helmet-violation-detector
```

### 2. Create a Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Model Weights

Download your trained `.pt` files and place them in the `weights/` directory:

```
weights/
├── bike_person.pt       ← Stage 1 detector
└── helmet_lp.pt         ← Stage 2 detector
```

### 5. Verify Class Names ⚠️ Important

Mismatched class names are the most common source of silent errors. Always verify after placing weights:

```python
from ultralytics import YOLO

model_1 = YOLO("weights/bike_person.pt")
model_2 = YOLO("weights/helmet_lp.pt")

print("Bike/Person classes:", model_1.names)
print("Helmet/LP classes:  ", model_2.names)
```

If the output differs from what the detectors expect, update the class name mappings in:

```
models/bike_person_detector.py
models/helmet_detector.py
```

### 6. Launch the Dashboard

```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser. Upload a traffic video and hit **Run**.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🎯 **Multi-stage YOLOv8 pipeline** | Two fine-tuned models: bikes/persons → helmets/plates |
| 🎥 **Real-time frame display** | Live annotated preview during inference in the dashboard |
| 📸 **Violation evidence saving** | Annotated frames and license plate crops saved per violation |
| 🔢 **Best-confidence LP caching** | Tracks the highest-confidence plate crop for each bike across the full video |
| 🧠 **IoU-based bike tracking** | Persistent IDs assigned via overlap matching — no external tracker needed |
| 🔁 **Cooldown-based deduplication** | Per-bike cooldown timer prevents repeated saves for the same vehicle |
| 📹 **Annotated video export** | Full H.264 output video with bounding boxes and labels, downloadable from UI |
| 📋 **CSV violation log** | One-click export of all violations with metadata for enforcement |
| ⚙️ **Configurable thresholds** | Detection confidence, minimum size, cooldown, and zone all tunable from sidebar |
| 🧩 **Modular architecture** | Swap models or add detection stages with minimal code changes |

---

## 📁 Project Structure

```
helmet_violation_detector/
│
├── app.py                        # Streamlit dashboard — entry point
├── requirements.txt              # Python dependencies
├── README.md
│
├── models/
│   ├── bike_person_detector.py   # Stage 1: bike & rider detection wrapper
│   └── helmet_detector.py        # Stage 2: helmet & license plate detection wrapper
│
├── utils/
│   ├── video_processor.py        # Frame-by-frame inference loop
│   ├── violation_handler.py      # Violation timing logic, cooldown, deduplication
│   └── drawing.py                # Bounding box annotation & overlay rendering
│
├── training/                     # Training scripts, configs, and notebooks
│
├── weights/                      # ← Place .pt model files here
│   ├── bike_person.pt
│   └── helmet_lp.pt
│
├── demo/                         # Demo video and banner image
│
└── violations/                   # Auto-generated output: frames, plates, CSV log
```

---

## 🔑 Key Insights

**What works well:**

- **Bike detection is highly reliable** — mAP@0.5 of 0.994 means virtually no vehicles are missed in normal conditions.
- **License plate detection is robust** — 0.871 mAP enables usable evidence crops in the vast majority of real-world frames.
- **The modular pipeline design** makes it straightforward to upgrade individual stages independently, add new classes, or port to a different inference backend.

**Where the challenges lie:**

- **Helmet classification is the bottleneck** — small object size, occlusion by hair/clothing, motion blur, and variable lighting all impact per-frame accuracy.
- **The temporal threshold (≥1s)** provides a practical workaround: a single missed frame doesn't trigger a false positive, but consistency improves accuracy at the cost of latency.
- **IoU-based tracking** is effective for moderate traffic but may lose IDs during severe occlusion or rapid lane changes. A proper multi-object tracker (e.g. BotSort, DeepSort) would improve identity consistency.

---

## 🔭 Roadmap

| Priority | Feature |
|----------|---------|
| 🔴 High | **OCR for license plate text recognition** — convert plate crops to readable text |
| 🔴 High | **Dedicated MOT integration** — replace IoU tracking with BotSort/ByteTrack |
| 🟡 Medium | **Night-time / low-light improvements** — domain adaptation or synthetic augmentation |
| 🟡 Medium | **Streamlit performance profiling** — optimize UI refresh rate for longer videos |
| 🟢 Low | **Multi-camera tracking support** — associate violations across overlapping views |
| 🟢 Low | **Edge deployment (ONNX / TensorRT)** — real-time inference on CCTV hardware |
| 🟢 Low | **Alert system** — SMS / email notification on confirmed violation |
| 🟢 Low | **Web API** — REST endpoint for integration with third-party enforcement systems |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software with attribution.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — state-of-the-art object detection
- [Streamlit](https://streamlit.io) — rapid ML dashboard development
- [OpenCV](https://opencv.org) — video I/O and frame processing
- [Roboflow Universe](https://universe.roboflow.com) — open annotation datasets

---

<div align="center">
  <sub>Built with ❤️ using YOLOv8 · Streamlit · OpenCV</sub>
  <br/>
  <sub>© 2024 — MIT Licensed</sub>
</div>
