## 🏎️ YOLO11 Fine-tuning on Car Dataset

Fine-tuned `yolo11n.pt` on the custom Stanford Cars dataset over **20 epochs** using Apple Silicon (`mps` device acceleration).

### 1. Training Setup & Hyperparameters

* **Base Model:** `yolo11n.pt`
* **Epochs:** 20
* **Input Resolution (`imgsz`):** 640
* **Compute Device:** `mps` (Apple Silicon)
* **Checkpoint Artifact:** `models/yolo11n_car_detect_20ep.pt`
* **Outputs & Metrics Directory:** `outputs/car_detection_finetune/`

---

### 2. Training Progression & Key Milestones

| Stage | Epoch | Train Box / Cls Loss | Val Box / Cls Loss | Precision (B) | Recall (B) | mAP50 (B) | mAP50-95 (B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Warmup / Start** | 1 | 1.136 / 3.758 | 0.945 / 4.079 | 0.00% | 0.00% | 0.00% | 0.00% |
| **Rapid Learning** | 2 | 1.136 / 2.780 | 1.147 / 3.575 | 86.36% | 84.44% | 79.80% | 45.59% |
| **Mid-point** | 10 | 1.104 / 0.861 | 1.117 / 0.996 | 82.33% | 95.55% | 93.24% | 61.16% |
| **Peak Performance** | 19 | 0.687 / 0.528 | 1.055 / 0.542 | 98.37% | **100.0%** | **99.50%** | **72.29%** |
| **Final Checkpoint** | 20 | 0.671 / 0.533 | 1.058 / 0.543 | 95.62% | **100.0%** | **99.21%** | **71.90%** |

---

### 3. Evaluation & Key Metrics

* **Precision ($P$):** **95.62%** (peaked at 99.8% in epoch 18)
* **Recall ($R$):** **100.0%** (achieved zero false negatives across validation vehicles from epoch 16 onward)
* **mAP@0.5:** **99.21%**
* **mAP@0.5:0.95:** **71.90%** (peaked at **72.29%** in epoch 19)
* **Convergence:**
  * Training classification loss (`train/cls_loss`) dropped sharply from **3.758** to **0.533**.
  * Validation classification loss (`val/cls_loss`) declined consistently from **4.079** down to **0.543**, showing clean generalization without overfitting.

---
## Car Tracking
>[!IMPORTANT]
> **Processing flow: Car Detecting and Tracking:**
> $$\text{Frame Video} \xrightarrow{\text{YOLO11/26}} \text{BBoxes} \xrightarrow{\text{ByteTrack + Kalman}} \text{Track ID} \xrightarrow{\text{Smoother}} \text{Point In Polygon} \xrightarrow{\text{Set()}} \text{Counting}$$