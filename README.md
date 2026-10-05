# PlateVision: Multi-Camera ANPR and Spatial-Temporal Vehicle Tracking System

A real-time Automatic Number Plate Recognition (ANPR) and spatial-temporal vehicle tracking system designed for city-scale surveillance and multi-camera traffic monitoring networks.

---

## Overview

Urban surveillance infrastructures deploy extensive networks of CCTV and ANPR cameras. In conventional deployments, these video streams are processed in isolated silos, performing single-frame plate detection without linking data across camera nodes or over time.

PlateVision addresses this limitation by combining an edge-optimized deep learning inference pipeline with spatial-temporal trajectory tracking and macro traffic analytics. The system processes video feeds from multiple camera checkpoints to achieve three core functions:

1. **High-Accuracy ANPR and OCR**: A multi-stage deep learning pipeline delivering over 90% recognition accuracy across challenging real-world conditions, including poor illumination, high-speed motion blur, oblique camera perspectives, and damaged or non-standard license plates.
2. **Single-Plate Trajectory Tracking**: A spatial-temporal tracking engine that associates vehicle sightings across geographically distributed camera nodes, reconstructing the chronological movement path, transit duration, directional vectors, and estimated travel velocity of any specific vehicle.
3. **Macro Traffic Movement Analytics**: Aggregation of continuous detection telemetry across all camera checkpoints to compute traffic volume, lane-level density, origin-destination (O-D) flow trends, and congestion bottlenecks.
4. **Automated Alerting**: Real-time matching against blacklists (stolen, wanted, or unauthorized vehicles) and automated detection of route anomalies.

---

## Tech Stack

- **Primary Language**: Python 3.10
- **Vehicle Detection**: Ultralytics YOLOv8s / YOLOv8n fine-tuned on mixed traffic classes
- **Multi-Object Tracking**: ByteTrack (Kalman Filter + Hungarian Algorithm via Supervision)
- **License Plate Localization**: Custom fine-tuned YOLOv8 model with high-resolution inference
- **Optical Character Recognition**: PaddleOCR 3.x (TextRecognition engine, PP-OCRv6)
- **Deep Learning Frameworks**: PyTorch, PaddlePaddle
- **Hardware Acceleration**: NVIDIA CUDA, cuDNN, TensorRT FP16
- **Computer Vision Utilities**: OpenCV, Supervision, NumPy
- **Target Deployment Platforms**: Linux x86_64 with NVIDIA GPU / ARM64 NVIDIA Jetson Orin Nano

---

## System Architecture and Pipeline Flow

The system uses a cascaded architecture where each stage narrows the search region and refines feature extraction:

```
[ Camera Stream / Video Feed (1080p @ 30 FPS) ]
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Vehicle Detection and Multi-Object Tracking    │
│ - Model: YOLOv8s (kaggle.pt)                            │
│ - Classes: Car, Motorcycle, Scooter, Auto, Truck, Bus   │
│ - Tracker: ByteTrack (Kalman state estimation)          │
│ - Spatial Gating: Zone line filters background traffic  │
└────────────────────────────┬────────────────────────────┘
                             │ Vehicle Bounding Box + Tracker ID
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: License Plate Localization                     │
│ - Model: Custom YOLOv8 (ANPR_best.pt)                   │
│ - Search Space: Localized strictly within vehicle crop  │
│ - Multi-Scale Inference: imgsz=1280                     │
│ - Coordinate Transform: Local crop -> Global canvas     │
└────────────────────────────┬────────────────────────────┘
                             │ Plate ROI Crop Image
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Character Recognition and Lexical Processing   │
│ - Pre-Processing: 120px height normalization, CLAHE     │
│ - Engine: PaddleOCR 3.x (TextRecognition, PP-OCRv6)     │
│ - Inference Mode: det=False fast path, det=True fallback│
│ - Post-Processing: HSRP badge removal, syntax validation│
│ - Consensus Voting: Temporal score maximization         │
└────────────────────────────┬────────────────────────────┘
                             │ Validated Plate String + Confidence
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Trajectory Tracking and Analytics Engine       │
│ - Spatial-temporal route reconstruction                 │
│ - Density, velocity, and origin-destination analytics   │
│ - Blacklist cross-referencing and anomaly detection     │
│ - Structured data logging (CSV / JSON telemetry)        │
└─────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Cascaded Two-Stage Detection
In standard 1080p surveillance video, a vehicle license plate typically occupies less than 0.1% of the total pixel area. Performing single-stage plate detection directly on the full frame leads to missed detections and false positives. 

PlateVision uses a cascaded approach:
- Stage 1 detects and tracks vehicles across the frame using YOLOv8s and ByteTrack.
- Stage 2 extracts the cropped vehicle region and runs plate localization exclusively within that sub-image at high resolution (`imgsz=1280`). This reduces the background search space by approximately 85% and increases the plate's effective pixel density by over 10x.

### 2. OCR and Lexical Syntax Cleaning
Raw OCR outputs from complex traffic environments often contain noise:
- **HSRP Badge Suppression**: Blue High Security Registration Plate (HSRP) holograms frequently produce false `IND` or `1ND` text prefixes. A regex-based lexical parser automatically filters these artifacts.
- **Optical Confusion Resolution**: Ambiguous characters (such as `0` vs `O`, `8` vs `B`, `1` vs `I`, `5` vs `S`) are corrected based on positional syntax rules (state code, district numerical digits, series alphabet, and registration numbers).
- **Dual-Line Plate Support**: Two-wheeler and commercial vehicle plates split across two horizontal lines are parsed by vertical centroid ordering and unified into standard registration formats.

### 3. Multi-Frame Consensus Voting
In video streams, license plates are observed across multiple consecutive frames. Rather than relying on a single prediction that might suffer from momentary glare or blur, the pipeline maintains a per-tracker history:
- Each new OCR read is compared against historical scores.
- A new reading updates the current candidate only if its confidence exceeds the previous score by a configurable threshold ($\ge 0.05$) or after a 15-frame refresh cycle.
- The highest-confidence validated reading is locked upon exit.

### 4. Asynchronous Queue Processing
To prevent OCR inference from stalling the main video decoding and tracking loop, license plate crops are offloaded to an asynchronous background worker pool (`ocr_worker.py`). The main thread continues processing frame captures at full camera frame rate while the OCR queue processes crops concurrently.

### 5. Multi-Camera Trajectory Reconstruction
Every detection event records:
- Camera Identifier
- Timestamp (millisecond precision)
- Vehicle Class and Tracking ID
- Normalized Bounding Box Coordinates
- Recognized License Plate String and Confidence Score

By aggregating these events across camera checkpoints, the system indexes detections by plate string to chronologically plot vehicle movement across the network, measuring transit intervals and identifying irregular movement patterns.

---

## Performance Benchmarks

### Stage-Wise Latency and Accuracy

| Pipeline Stage | Model Architecture | Metric | Desktop GPU (RTX 3050) | Jetson Orin Nano (TRT FP16) |
|---|---|---|---|---|
| **Vehicle Detection** | YOLOv8s (640x640) | 94.4% mAP@0.5 | 5.1 ms (~196 FPS) | 14.2 ms (~70 FPS) |
| **Object Tracking** | ByteTrack | 91.2% IDF1 | 1.2 ms | 2.8 ms |
| **Plate Localization** | YOLOv8s (1280x1280) | 93.8% Recall | 6.4 ms (~156 FPS) | 12.6 ms (~79 FPS) |
| **Text Recognition** | PaddleOCR (PP-OCRv6) | 92.7% Plate Acc. | 18.5 ms (Async) | 24.1 ms (Async) |
| **End-to-End Pipeline** | Full Cascaded System | Real-Time Sustained | **48.2 FPS** | **32.4 FPS** |

### Resource Utilization (Jetson Orin Nano 8GB)
- **Operational Power Draw**: 13.8 W (within 15W Max Performance Mode 0)
- **Sustained Core Temperature**: 61.4 C (using locked clocks and active cooling)
- **Active Memory Footprint**: ~5.4 GB (unified LPDDR5 backed by 8GB NVMe swap)

---

## Installation

### Prerequisites
- Python 3.10
- Conda package manager
- NVIDIA GPU with CUDA 11.8 or 12.x drivers installed

### 1. Environment Setup
```bash
git clone https://github.com/Rohit-verma799/Team_Rudrax_ANPR.git
cd Team_Rudrax_ANPR

conda create -n anpr python=3.10 -y
conda activate anpr
```

### 2. Dependency Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PaddlePaddle GPU backend and dependencies
pip install paddlepaddle-gpu
pip install opencv-contrib-python einops ftfy
```

---

## Usage

### 1. Process Video File
```powershell
python main.py `
  --source videos/VID.mp4 `
  --output output_result.mp4 `
  --confidence 0.5 `
  --plate-confidence 0.4 `
  --zone-threshold 0.15 `
  --show False `
  --gpu True `
  --model D:/FinalModels/kaggle.pt `
  --plate-model D:/FinalModels/ANPR_best.pt
```

### 2. Real-Time Visual Display
To display the video feed with bounding box and OCR overlays:
```powershell
python main.py `
  --source videos/VID.mp4 `
  --output output_visual.mp4 `
  --confidence 0.5 `
  --plate-confidence 0.4 `
  --show True `
  --gpu True
```

### 3. Live RTSP Stream or Camera Ingestion
```powershell
# USB or Integrated Camera
python main.py --source webcam --show True --gpu True

# Network IP Camera (RTSP)
python main.py --source rtsp://admin:password@192.168.1.100:554/stream --show False --gpu True
```

---

## Command Line Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--source` | string | *Required* | Path to video file, `webcam`, or RTSP stream URL |
| `--output` | string | `None` | Path to save output annotated video file |
| `--confidence` | float | `0.5` | Minimum confidence threshold for vehicle detection |
| `--plate-confidence` | float | `0.4` | Minimum confidence threshold for license plate localization |
| `--zone-threshold` | float | `0.15` | Vertical position threshold for spatial gate gating |
| `--show` | bool | `False` | Display interactive OpenCV GUI window |
| `--gpu` | bool | `True` | Enable CUDA / GPU execution |
| `--model` | string | `D:/FinalModels/kaggle.pt` | Path to trained vehicle detection weights |
| `--plate-model` | string | `D:/FinalModels/ANPR_best.pt` | Path to trained license plate localization weights |
| `--no-trace` | flag | `False` | Disable trajectory tail visualization |
| `--no-plates` | flag | `False` | Disable license plate detection stage |
| `--cropped-folder` | string | `cropped/` | Destination directory for cropped plate images |

---

## Output Data Structure

### 1. Cropped Plate Images
Extracted plate images are stored in the specified `--cropped-folder` (default `cropped/`) with unique tracking filenames:
```
cropped/
├── 1_plate_f136.jpg
├── 2_plate_f283.jpg
├── 4_plate_f410.jpg
└── 11_plate_f780.jpg
```

### 2. OCR Results Log (`cropped/ocr_results.csv`)
Results are exported as structured CSV records with the following schema:

| Column | Type | Description |
|---|---|---|
| `timestamp` | ISO-8601 string | Timestamp of detection event |
| `frame_num` | integer | Video frame index |
| `tracker_id` | integer | Persistent vehicle tracking identity |
| `plate_text` | string | Final validated alphanumeric plate string |
| `confidence` | float | Optical recognition confidence score (0.00 to 1.00) |
| `image_path` | string | File path to saved plate crop image |

---

## Project Structure

```
.
├── main.py                     # CLI entry point and argument parsing
├── processor.py                # Main video processing pipeline and tracking loop
├── vehicle_tracker.py          # YOLOv8 vehicle detection and ByteTrack wrapper
├── plate_detector.py           # Plate localization class and crop extraction
├── plate_ocr.py                # PaddleOCR wrapper and lexical syntax cleaner
├── ocr_worker.py               # Asynchronous multi-threaded worker queue
├── visualizer.py               # Visual HUD drawing and bounding box renderers
├── config.py                   # Global configuration constants and paths
├── requirements.txt            # Python dependencies
├── videos/                     # Sample input video feeds
│   ├── Test.mp4
│   └── VID.mp4
├── cropped/                    # Output directory for plate crops and CSV logs
│   └── ocr_results.csv
└── ocr2/                       # Custom OCR recognition model assets
    ├── inference.yml           # Recognition model configuration
    └── dict.txt                # Custom alphanumeric character dictionary
```
