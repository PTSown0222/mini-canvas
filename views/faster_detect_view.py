import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from src.faster_rcnn.train import MobileNetDetection
import streamlit as st
import datetime
import os
import pandas as pd

st.title("🎯 Face Detection - Faster R-CNN (MobileNetV3)")

input_source = st.radio("Select Input Source", ["Upload Image", "Use Webcam (Snapshot)"], horizontal=True)

LOG_FILE = "attendance_log.csv"
if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("ID,Person,Time,Confidence\n")

@st.cache_resource
def load_rcnn_model(checkpoint_path: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = MobileNetDetection(num_classes=num_classes).to(device)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    return model, device

st.sidebar.header("⚙️ Model & Inference Settings")
checkpoint_file = "models/faster_mobilenet_facedetect.pt"
st.sidebar.caption("Checkpoint Path: `models/faster_mobilenet_facedetect.pt`")
image_size = st.sidebar.slider("Model Input Image Size", min_value=128, max_value=640, value=224, step=32)
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
model, device = load_rcnn_model(checkpoint_file, num_classes=2)

image = None

if input_source == "Upload Image":
    if "current_image" not in st.session_state or st.session_state.current_image is None:
        st.warning("⚠️ No image found. Please go to **Home Page** to upload an image first!")
        st.stop()
    image = st.session_state.current_image
else:
    st.subheader("📸 Webcam Snapshot Attendance")
    st.info("💡 Put button **Take Photo** below here to take a photo.")
    camera_file = st.camera_input("Capture a picture from your webcam")
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image is not None:
    if len(image.shape) == 2:
        input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        input_image = image

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, channels="BGR", width="stretch")

    with st.spinner("Faster R-CNN is running inference..."):
        h, w, _ = input_image.shape
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_image, (image_size, image_size))
        
        normalized_img = resized_img.astype(np.float32) / 255.0
        normalized_img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        normalized_img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        tensor_img = torch.from_numpy(np.transpose(normalized_img, (2, 0, 1))).to(device).float()
        
        with torch.no_grad():
            predictions = model([tensor_img])
            
        annotated_img = input_image.copy()
        pred_boxes = predictions[0]["boxes"].cpu().numpy()
        pred_scores = predictions[0]["scores"].cpu().numpy()
        
        valid_indices = pred_scores >= conf_thresh
        filtered_boxes = pred_boxes[valid_indices]
        filtered_scores = pred_scores[valid_indices]
        
        if len(filtered_boxes) > 0:
            max_score = float(np.max(filtered_scores))
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                df_existing = pd.read_csv(LOG_FILE)
                next_id = len(df_existing) + 1
            except Exception:
                next_id = 1

            last_time = st.session_state.get("last_snapshot_time", "")
            if last_time != current_time:
                st.session_state.last_snapshot_time = current_time
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{next_id},Person_Snapshot,{current_time},{max_score:.2%}\n")

        for box, score in zip(filtered_boxes, filtered_scores):
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin / image_size * w)
            ymin = int(ymin / image_size * h)
            xmax = int(xmax / image_size * w)
            ymax = int(ymax / image_size * h)
            
            cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"Face {score:.2f}"
            cv2.putText(annotated_img, label, (xmin, max(ymin - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        with col2:
            st.subheader(f"Result: Found {len(filtered_boxes)} face(s)")
            st.image(annotated_img, channels="BGR", width="stretch")
            
            if len(filtered_boxes) > 0:
                if st.button("📝 Confirm your submission", type="primary", width="stretch"):
                    max_score = float(np.max(filtered_scores))
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    try:
                        df_existing = pd.read_csv(LOG_FILE)
                        next_id = len(df_existing) + 1
                    except Exception:
                        next_id = 1

                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{next_id},Person_Snapshot,{current_time},{max_score:.2%}\n")
                    st.success(f"✅ Sucessfully {current_time}!")
                    st.rerun()

            is_success, buffer = cv2.imencode(".png", annotated_img)
            if is_success:
                st.download_button(
                    label="⬇️ Download Result Image",
                    data=buffer.tobytes(),
                    file_name="faster_rcnn_face_detected.png",
                    mime="image/png",
                    width="stretch"
                )