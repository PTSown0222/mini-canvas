"""
Mini Canvas: Image Enhancement & Detection
"""
import streamlit as st
import cv2
import numpy as np
from src.enhance_image import denoiseAndSmoothImage

# Page Layout
st.set_page_config(
    page_title="Mini Canvas - Enhance Images",
    page_icon="📷",
    layout="wide"
)
st.title("Mini Canvas - Image Enhancement & Detection")

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "detected_image" not in st.session_state:
    st.session_state.detected_image = None

# 2. Sidebar cấu hình
st.sidebar.header("Images System")
model_options = ["yolo pose", "yolo segmentation", "yolo detection"]
selected_model = st.sidebar.selectbox("Select Model", options=model_options, index=0)

st.sidebar.subheader("Methods")
image_enhance = [
    "Gaussian",
    "Bilateral",
    "Median",
    "Sharpen"
]
selected_methods = st.sidebar.selectbox("Select Filter Method", options=image_enhance, index=0)
filter_params = {}
if selected_methods == "Gaussian":
    k_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=31, value=9, step=2)
    filter_params["ksize"] = (k_size, k_size)
elif selected_methods == "Bilateral":
    filter_params["d"] = st.sidebar.slider("Diameter (d)", min_value=5, max_value=25, value=15, step=2)
    filter_params["sigma_color"] = st.sidebar.slider("Sigma Color", min_value=10, max_value=200, value=100, step=10)
elif selected_methods == "Median":
    filter_params["ksize"] = st.sidebar.slider("Kernel Size", min_value=3, max_value=21, value=5, step=2)

# 4. Upload file
uploaded_file = st.file_uploader(
    "Drag & drop an image or click to upload",
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

if uploaded_file is not None:
    # Reset processed image nếu upload ảnh mới
    if st.session_state.uploaded_image != uploaded_file.name:
        st.session_state.uploaded_image = uploaded_file.name
        st.session_state.processed_image = None

    # Dùng getvalue() an toàn thay vì read() để tránh mất buffer khi rerun
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(opencv_image, channels="BGR", width="stretch")

    if st.sidebar.button("Apply Filter"):
        st.session_state.processed_image = denoiseAndSmoothImage(
            opencv_image, 
            method=selected_methods,
            **filter_params
        )

    with col2:
        st.subheader(f"Processed ({selected_methods})")
        if st.session_state.processed_image is not None:
            st.image(st.session_state.processed_image, channels="BGR", width="stretch")
        else:
            st.info("Nhấn **'Apply Filter'** ở sidebar để xem kết quả.")