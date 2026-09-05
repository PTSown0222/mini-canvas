import streamlit as st
import cv2
import tempfile
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import supervision as sv
from ultralytics import YOLO
import torch

st.set_page_config(layout="wide")
st.title("🚦 Zone Tracking, Counting and Classification Traffic")

uploaded_video = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

if not uploaded_video:
    st.info("👈 Please upload an MP4/MOV traffic video to start analysis.")
    st.stop()

# Lưu file tạm
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_video.read())
video_path = tfile.name

video_info = sv.VideoInfo.from_video_path(video_path)
W, H = video_info.width, video_info.height
fps = video_info.fps if video_info.fps > 0 else 30
BASE_TIME = datetime(2026, 9, 4, 15, 0, 0)

# --- 1. Quản lý trạng thái trong session_state ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0
# Lưu ID duy nhất theo từng class: {"car": set(), "truck": set(), ...}
if "counted_by_class" not in st.session_state:
    st.session_state.counted_by_class = {"car": set(), "bus": set(), "truck": set(), "motorcycle": set()}
if "vehicle_metadata" not in st.session_state:
    st.session_state.vehicle_metadata = {}

# --- 2. Sidebar Controls ---
st.sidebar.header("⚙️ Detection & Zone Settings")
conf_thresh = st.sidebar.slider("Detection Confidence", 0.15, 0.80, 0.25, 0.05)

# Điều chỉnh vùng chữ nhật giám sát
zone_y_ratio = st.sidebar.slider("Zone Top Position (% Height)", 0.20, 0.85, 0.50, 0.02)
zone_thickness_ratio = st.sidebar.slider("Zone Thickness (% Height)", 0.05, 0.45, 0.30, 0.02)

st.sidebar.markdown("---")
st.sidebar.header("🔥 Heatmap Settings")
enable_heatmap = st.sidebar.toggle("Enable Traffic Heatmap", value=True)
heat_opacity = st.sidebar.slider("Heatmap Opacity", 0.2, 0.9, 0.55, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🕹️ Playback Controls")
speed_option = st.sidebar.select_slider(
    "Playback Speed",
    options=["0.5x (Slow)", "1.0x (Normal)", "2.0x (Fast)", "3.0x (Ultra)"],
    value="1.0x (Normal)"
)
SPEED_CONFIG = {
    "0.5x (Slow)": {"skip": 1, "delay": 0.04},
    "1.0x (Normal)": {"skip": 1, "delay": 0.00},
    "2.0x (Fast)": {"skip": 2, "delay": 0.00},
    "3.0x (Ultra)": {"skip": 3, "delay": 0.00},
}
cfg_speed = SPEED_CONFIG[speed_option]

# Tọa độ Zone
y1 = int(H * zone_y_ratio)
y2 = min(H, int(y1 + H * zone_thickness_ratio))
zone_polygon = np.array([[0, y1], [W, y1], [W, y2], [0, y2]], dtype=np.int32)

@st.cache_resource
def load_yolo():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return YOLO("models/yolo26s.pt"), device

model, device = load_yolo()
VEHICLE_NAMES = {'car', 'motorcycle', 'bus', 'truck'}
valid_ids = [k for k, v in model.names.items() if v in VEHICLE_NAMES]

# ByteTrack với buffer 60 frame chống trùng ID
if "tracker" not in st.session_state:
    st.session_state.tracker = sv.ByteTrack(
        track_activation_threshold=conf_thresh, 
        lost_track_buffer=60, 
        frame_rate=fps
    )
tracker = st.session_state.tracker

# HeatMapAnnotator duy trì vết nhiệt toàn luồng video
if "heat_annotator" not in st.session_state:
    st.session_state.heat_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=heat_opacity,
        radius=25,
        kernel_size=25
    )
heat_annotator = st.session_state.heat_annotator
heat_annotator.opacity = heat_opacity

# --- 3. Nút điều khiển Play / Pause / Reset ---
col_c1, col_c2, col_c3 = st.columns(3)
if col_c1.button("🚀 Play / Resume", type="primary", width="stretch"):
    st.session_state.is_running = True
if col_c2.button("⏸️ Pause", width="stretch"):
    st.session_state.is_running = False
if col_c3.button("⏹️ Reset", width="stretch"):
    st.session_state.is_running = False
    st.session_state.last_frame = None
    st.session_state.frame_index = 0
    st.session_state.counted_by_class = {"car": set(), "bus": set(), "truck": set(), "motorcycle": set()}
    st.session_state.vehicle_metadata = {}
    st.session_state.heat_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=heat_opacity,
        radius=25,
        kernel_size=25
    )
    st.session_state.tracker = sv.ByteTrack(
        track_activation_threshold=conf_thresh, 
        lost_track_buffer=60, 
        frame_rate=fps
    )
    st.rerun()

# --- 4. Layout Video & Metrics Thống Kê Loại Xe ---
col_view, col_stats = st.columns([3.2, 1])
st_frame = col_view.empty()

# Tính tổng số xe duy nhất
total_unique = sum(len(ids) for ids in st.session_state.counted_by_class.values())

with col_stats:
    st.subheader("📊 Vehicle Breakdown")
    m_total = st.metric("Total Vehicles", total_unique)
    m_cars = st.metric("🚗 Cars", len(st.session_state.counted_by_class["car"]))
    m_trucks = st.metric("🚚 Trucks", len(st.session_state.counted_by_class["truck"]))
    m_buses = st.metric("🚌 Buses", len(st.session_state.counted_by_class["bus"]))
    m_motor = st.metric("🏍️ Motorcycles", len(st.session_state.counted_by_class["motorcycle"]))
    m_in_zone = st.metric("Currently in Zone", 0)
    m_fps = st.metric("FPS", 0)

if not st.session_state.is_running and st.session_state.last_frame is not None:
    st_frame.image(st.session_state.last_frame, channels="BGR", width="stretch")

st.markdown("---")
st.subheader("📋 Detailed Vehicle Registry (Metadata)")
table_placeholder = st.empty()
col_dl1, col_dl2 = st.columns(2)
btn_csv_placeholder = col_dl1.empty()
btn_json_placeholder = col_dl2.empty()

# --- 5. Vòng lặp phát video ---
if st.session_state.is_running:
    box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.CLASS, roundness=0.2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.45, 
        text_thickness=1, 
        text_position=sv.Position.TOP_CENTER,
        color_lookup=sv.ColorLookup.CLASS
    )
    trace_annotator = sv.TraceAnnotator(trace_length=20, thickness=2, color_lookup=sv.ColorLookup.TRACK)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
    prev_time = cv2.getTickCount()

    while cap.isOpened() and st.session_state.is_running:
        step = cfg_speed["skip"]
        if step > 1:
            curr_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos + step - 1)

        ret, frame = cap.read()
        if not ret:
            st.session_state.is_running = False
            st.session_state.frame_index = 0
            break

        current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        st.session_state.frame_index = current_frame_no

        sec_elapsed = round(current_frame_no / fps, 2)
        current_ts = (BASE_TIME + timedelta(seconds=sec_elapsed)).strftime("%Y-%m-%d %H:%M:%S")

        # Inference
        results = model.predict(frame, conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, valid_ids)]
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()

        # 1. Vẽ Heatmap mật độ xe
        if enable_heatmap:
            annotated_frame = heat_annotator.annotate(scene=annotated_frame, detections=detections)

        # 2. Vẽ Overlay Zone giám sát (Xanh lá trong suốt)
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [zone_polygon], (0, 255, 180))
        cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)
        cv2.polylines(annotated_frame, [zone_polygon], isClosed=True, color=(0, 255, 180), thickness=2)

        currently_inside = 0

        # 3. Phân loại & Ghi nhận xe
        if len(detections) > 0 and detections.tracker_id is not None:
            anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

            for track_id, cid, (cx, cy) in zip(detections.tracker_id, detections.class_id, anchors):
                track_id = int(track_id)
                cls_name = model.names[cid]

                # Kiểm tra xe đi vào Zone
                if y1 <= cy <= y2:
                    currently_inside += 1
                    cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 255, 0), cv2.FILLED)

                    # Ghi nhận ID duy nhất theo từng class
                    if cls_name in st.session_state.counted_by_class:
                        st.session_state.counted_by_class[cls_name].add(track_id)

                    # Ghi nhận Metadata
                    if track_id not in st.session_state.vehicle_metadata:
                        st.session_state.vehicle_metadata[track_id] = {
                            "Track ID": track_id,
                            "Vehicle Type": cls_name.upper(),
                            "First Seen": current_ts,
                            "Last Seen": current_ts,
                            "Detected Location": "Inside Zone"
                        }
                    else:
                        st.session_state.vehicle_metadata[track_id]["Last Seen"] = current_ts

            labels = [
                f"#{tid} {model.names[cid]}" 
                for tid, cid in zip(detections.tracker_id, detections.class_id)
            ]
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Đo FPS
        curr_time = cv2.getTickCount()
        time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
        fps_proc = int(1.0 / time_diff) if time_diff > 0 else 0
        prev_time = curr_time

        st.session_state.last_frame = annotated_frame

        # Cập nhật giao diện
        st_frame.image(annotated_frame, channels="BGR", width="stretch")
        total_unique = sum(len(ids) for ids in st.session_state.counted_by_class.values())
        m_total.metric("Total Vehicles", total_unique)
        m_cars.metric("🚗 Cars", len(st.session_state.counted_by_class["car"]))
        m_trucks.metric("🚚 Trucks", len(st.session_state.counted_by_class["truck"]))
        m_buses.metric("🚌 Buses", len(st.session_state.counted_by_class["bus"]))
        m_motor.metric("🏍️ Motorcycles", len(st.session_state.counted_by_class["motorcycle"]))
        m_in_zone.metric("Currently in Zone", currently_inside)
        m_fps.metric("FPS", fps_proc)

        if cfg_speed["delay"] > 0:
            time.sleep(cfg_speed["delay"])

    cap.release()

# --- 6. Hiển thị bảng dữ liệu Logs & Download khi Pause / Hoàn tất ---
if len(st.session_state.vehicle_metadata) > 0:
    df_final = pd.DataFrame(list(st.session_state.vehicle_metadata.values()))
    cols_to_show = ["Track ID", "Vehicle Type", "First Seen", "Last Seen", "Detected Location"]
    df_clean = df_final[cols_to_show]
    table_placeholder.dataframe(df_clean, width="stretch")

    csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
    btn_csv_placeholder.download_button(
        label="📥 Download Vehicle Registry (CSV)",
        data=csv_bytes,
        file_name="traffic_vehicle_breakdown.csv",
        mime="text/csv",
        width="stretch"
    )

    json_bytes = json.dumps(df_clean.to_dict(orient="records"), indent=4).encode('utf-8')
    btn_json_placeholder.download_button(
        label="📥 Download Vehicle Registry (JSON)",
        data=json_bytes,
        file_name="traffic_vehicle_breakdown.json",
        mime="application/json",
        width="stretch"
    )