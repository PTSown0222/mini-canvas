import streamlit as st
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

st.title("🎯 Object Detections - YOLO Model")

if "current_image" not in st.session_state or st.session_state.current_image is None:
    st.warning("⚠️ No image found. Please go to **Home Page** to upload an image first!")
    st.stop()

image = st.session_state.current_image

# treat channels images
if len(image.shape) == 2:
    input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
else:
    input_image = image

@st.cache_resource
def load_yolo_model(weights_path: str):
    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    return YOLO(weights_path), device

st.sidebar.header("⚙️ Model & Inference Settings")
model_choice = st.sidebar.selectbox(
    "Select Model Mode",
    [
        "🚗 Car Detection (Finetune)",
        "🌐 General Object Detection",
        "🎭 Instance Segmentation"
    ]
)

if "Finetune" in model_choice:
    weights_target = "models/yolo11n_car_detect_20ep.pt"
    if not Path(weights_target).exists():
        weights_target = "models/yolo11n.pt"
elif "Segmentation" in model_choice:
    weights_target = "models/yolo11n-seg.pt"
else:
    weights_target = "models/yolo11n.pt"

st.sidebar.caption(f"Loaded weights: `{weights_target}`")
model, device = load_yolo_model(weights_target)
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
iou_thresh = st.sidebar.slider("IoU Threshold (NMS)", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
run_btn = st.sidebar.button("🚀 Run Detection", type="primary", use_container_width = True)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    st.image(image, channels="BGR", use_container_width=True)

if run_btn:
    with st.spinner("Model is running inference..."):
        results = model.predict(
            source=input_image,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False
        )

        res = results[0]
        if res.masks is not None:
            annotated_img = input_image.copy()
            h, w = input_image.shape[:2]

            colors = [
                (255, 100, 0),   # Xanh dương
                (0, 200, 100),   # Xanh lục
                (200, 50, 255),  # Tím hồng
                (0, 165, 255),   # Cam
                (50, 220, 255),  # Vàng
            ]

            masks_tensor = res.masks.data.cpu().numpy()

            for idx, mask in enumerate(masks_tensor):
                color = colors[idx % len(colors)]
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                binary_mask = mask_resized > 0.5

                colored_layer = np.zeros_like(annotated_img, dtype=np.uint8)
                colored_layer[binary_mask] = color

                alpha = 0.40
                annotated_img[binary_mask] = cv2.addWeighted(
                    annotated_img[binary_mask], 1 - alpha,
                    colored_layer[binary_mask], alpha,
                    0
                )

                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated_img, contours, -1, color, 2)

            if res.boxes is not None:
                for idx, box in enumerate(res.boxes):
                    color = colors[idx % len(colors)]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    cls_name = res.names[cls_id]
                    conf = float(box.conf[0])

                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y1 = max(0, y1 - th - baseline - 4)
                    cv2.rectangle(annotated_img, (x1, label_y1), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(
                        annotated_img, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                    )

        else:
            annotated_img = res.plot(boxes=True)

        with col2:
            st.subheader(f"Result: Found {len(res.boxes)} object(s)")
            st.image(annotated_img, channels="BGR", use_container_width=True)
            is_success, buffer = cv2.imencode(".png", annotated_img)
            if is_success:
                st.download_button(
                    label="⬇️ Download Result Image",
                    data=buffer.tobytes(),
                    file_name="yolo11_detected.png",
                    mime="image/png",
                    use_container_width=True
                )

        st.divider()
        st.subheader("📊 Detected Objects Breakdown")
        if len(res.boxes) > 0:
            data = []
            has_masks = res.masks is not None
            for idx, box in enumerate(res.boxes):
                cls_id = int(box.cls[0])
                cls_name = res.names[cls_id]
                conf = float(box.conf[0])
                coords = [round(x, 1) for x in box.xyxy[0].tolist()]
                row = {
                    "ID": idx + 1,
                    "Class": cls_name,
                    "Confidence": f"{conf:.2%}",
                    "Bounding Box [x1, y1, x2, y2]": str(coords),
                    "Has Mask": "✅" if has_masks else "❌"
                }
                data.append(row)
            st.dataframe(data, use_container_width=True)
        else:
            st.info("No objects detected. Try lowering the Confidence Threshold.")
else:
    with col2:
        st.subheader("Detection Result")
        st.info("👈 Choose model mode and click **Run Detection** on the sidebar.")