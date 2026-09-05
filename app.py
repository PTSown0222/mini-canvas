"""
Mini Canvas: Image Enhancement & Detection
"""
import streamlit as st
import cv2

# Page Layout
st.set_page_config(
    page_title="Mini Canvas",
    page_icon="📷",
    layout="wide"
)
st.logo(
    image = "🤗",
    size = "large",
    icon_image = "🤗",
)

st.sidebar.markdown(
    """
    **Author 👨‍💻:** Phuong The Son
    [![GitHub Repo](https://img.shields.io/badge/GitHub-mini--canvas-8A2BE2?style=flat&logo=github)](https://github.com/PTSown0222/mini-canvas)
    """
)

active = st.toggle('active snow')
if active:
    st.snow()

if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None

home_p = st.Page("./views/home.py", title="Home Page and Upload", icon="🏠", default=True)
enhance_p = st.Page("./views/manipulation.py", title="Enhance Image", icon="📷")
# Navigate to models
yolo11n_detection = st.Page("./views/yolo_detect_view.py", title="Omni YOLO Detection", icon="📦")
faster_rcnn_detection = st.Page("./views/faster_detect_view.py", title="Face Detection (Faster R-CNN)", icon="🎭")
track_p = st.Page("./views/yolo_tracking_view.py", title="Vehicle Tracking", icon="📹")

page = st.navigation({
    "Home Page": [home_p],
    "Processing Image": [enhance_p],
    "Predict": [yolo11n_detection, track_p, faster_rcnn_detection]
})

page.run()

