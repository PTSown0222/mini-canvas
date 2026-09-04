<div align="center">

# 🎨 Mini Canvas: Image Enhancement and Detection

<p align="center">
  <em>An interactive Computer Vision workspace combining classical digital image processing with deep learning inference.</em>
</p>

<!-- Status Badges -->
[![Python](https://img.shields.io/badge/Python-3.12-20232a?style=flat&logo=python&logoColor=3776AB)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-20232a?style=flat&logo=streamlit&logoColor=FF4B4B)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-20232a?style=flat&logo=opencv&logoColor=5C3EE8)](https://opencv.org/)
[![uv](https://img.shields.io/badge/uv-Fast%20Packaging-20232a?style=flat&logo=astral&logoColor=DE5FE9)](https://astral.sh/uv)
<!-- [![Docker](https://img.shields.io/badge/Docker-Ready-20232a?style=flat&logo=docker&logoColor=2496ED)](https://www.docker.com/) -->

</div>

---

## 🤗 Demo

> Demo 1: Denoise
<p align="center">
  <img src="assets/demo1.png" alt="Mini Canvas Demo 1" width="800"/>
</p>

> Demo 2: Edge Detection
<p align="center">
  <img src="assets/demo2.png" alt="Mini Canvas Demo 2" width="800"/>
</p>

> Demo 3: Car Detection (Finetune)
<p align="center">
  <img src="assets/demo3.jpg" alt="Mini Canvas Demo 3" width="800"/>
</p>

> Demo 4: UI/UX segmentation and object detections
<p align="center">
  <img src="assets/demo4.png" alt="Mini Canvas Demo 4" width="800"/>
</p>

---
## 🌟 Key Features

* **Image Enhancement & Denoising:** Gaussian Blur, Bilateral Filter, Median Filter, and 2D Kernel Sharpening.
* **Edge Detection:** Multi-algorithm gradient extraction including Canny, Sobel, and Prewitt filters.
* **YOLO11 Vision Tasks:**
  * 📦 **Object Detection:** Real-time multi-class bounding box localization.
  * 🎭 **Instance Segmentation:** Pixel-level mask segmentation.
  <!-- * 🤸 **Pose Estimation:** Human keypoint tracking and skeleton drawing. -->
* **Multi-Page UI:** Clean navigation architecture powered by `st.navigation`.
<!-- * **Fast Dependency Management:** Managed with **uv** for ultra-fast environment resolution and containerized via **Docker**. -->

---

## 📁 Project Structure

```text
miniCanvas/
├── assets/                       # Sample images for testing & demo
│   ├── demo1.png
│   ├── demo2.png
│   ├── demo3.jpg
│   └── demo4.png
├── data/                         # Datasets directory (car_dataset: train/valid/test)
├── models/                       # Checkpoints & model weights
│   ├── yolo11n.pt                # Base COCO detection model
│   ├── yolo11n-seg.pt            # Base COCO instance segmentation model
│   ├── yolo11n_car_detect_20ep.pt# Fine-tuned car detection model
│   └── fasterrcnn_mobilenet_v3.pt# Faster R-CNN model weights
├── notebook/                     # Jupyter notebooks for experiments & training
├── outputs/                      # Training artifacts, charts & evaluation metrics
│   └── car_detection_finetune/
│       ├── BoxF1_curve.png
│       ├── BoxP_curve.png
│       ├── BoxPR_curve.png
│       ├── BoxR_curve.png
│       ├── confusion_matrix.png
│       └── results.csv
├── script/                       # Standalone shell execution scripts
│   ├── run.sh
│   └── run_faster_rcnn.sh
├── src/                          # Core processing modules
│   ├── faster_rcnn/              # Faster R-CNN pipeline implementation
│   ├── yolo11/                   # YOLO11 training, validation & inference scripts
│   ├── basic_adjust.py           # Basic image operations (contrast, brightness)
│   └── enhance_image.py          # Classical CV filters (denoise, smoothing, edges)
├── views/                        # Streamlit multi-page UI views
│   ├── home.py                   # Image upload hub & landing page
│   ├── manipulation.py           # Image filtering & enhancement view
│   └── yolo_detect_view.py       # YOLO11 detection & instance segmentation view
├── app.py                        # Main Streamlit router & entrypoint
├── Dockerfile                    # Containerization config
├── Makefile                      # Command shortcuts (build, run, infer)
├── pyproject.toml                # Project metadata & dependency definitions
└── uv.lock                       # Locked dependency tree (managed with uv)
```
## 🚀 Quick Start
Ensure you have uv installed on your system:

More sources with uv here: https://docs.astral.sh/uv/getting-started/installation/

#### macOS / Linux
`curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh`

#### Windows (PowerShell)
`powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"`

## Local Installation
Clone the repository
```shell
git clone https://github.com/PTSown0222/mini-canvas.git
cd mini-canvas
```

Sync dependencies:
```shell
uv sync
```

Run the application
```shell
bash script/run.sh
# Or directly via uv:
uv run streamlit run app.py
```

#### TODO:

- [x] Build mini canvas to enhance quality images
- [ ] Reads and Conducts this process: `https://docs.astral.sh/uv/guides/integration/docker/installing-uv`
- [x] Add more features to enhance quality images such as brightness,...
- [x] Training and Inference Yolo Model
- [X] Build UI UX
- [ ] Training and Inference Faster-RCNN (MobileNet Backbone)
- [ ] Write API
- [ ] Deploy Model with web apps

#### Requirements
```text
uv add torch>=2.13.0
uv add opencv-python-headless
uv add numpy>=2.5.2
uv add streamlit>=1.61.1
uv add ultralytics>=8.4.138
uv add torchvision>=0.28.0
```

