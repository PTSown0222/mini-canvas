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

> Demo 3: Object Detection (coming soon)

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
├── assets/
│   ├── demo1.png
|   └── demo2.png
|
├── data/                  # Sample test images
├── models/                # Pre-trained YOLO11 weights (.pt)
│   ├── yolo11n.pt
│   └── yolo11n-seg.pt
│   
├── src/                   # Core logic and CV functions
│   ├── enhance_image.py   # OpenCV filtering algorithms
│   └── yolo_service.py    # Model loader and inference engine (coming soon)
├── views/                 # Streamlit page views
│   ├── home.py            # Upload hub and landing view
│   └── manipulation.py    # Enhancement and edge detection view
├── app.py                 # Main entrypoint and router
├── Dockerfile             # Container configuration
├── pyproject.toml         # Project dependencies & metadata
├── uv.lock                # Locked dependency tree
└── run.sh                 # Startup bash script
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
bash run.sh
# Or directly via uv:
uv run streamlit run app.py
```

#### TODO:

- [x] Build mini canvas to enhance quality images
- [ ] Reads and Conducts this process: `https://docs.astral.sh/uv/guides/integration/docker/installing-uv`
- [ ] Add more features to enhance quality images such as brightness,...
- [ ] Training and Inference Yolo Model
- [ ] Deploy Model with web apps

#### Requirements
```text
uv add opencv-python-headless
uv add numpy>=2.5.2
uv add streamlit>=1.61.1
```

