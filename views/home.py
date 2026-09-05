
import streamlit as st
import cv2
import numpy as np

st.title("📷 Mini Canvas - Enhance Images and Detection")
uploaded_file = st.file_uploader(
    "Uploaded Image (JPG, PNG)",
    type = ["jpg", "jpeg", "png"],
    key = "global_uploader"
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if st.session_state.get("image_name") != uploaded_file.name:
        st.session_state.pop("bg_removed_result", None)
    st.session_state.current_image = opencv_image
    st.session_state.image_name = uploaded_file.name

if st.session_state.current_image is not None:
    st.success(f"working with file: **{st.session_state.image_name}**")
    img_width = st.slider("Resize Images)", min_value = 200, max_value = 800, value= 350, step=50)
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.image(
            st.session_state.current_image,
            channels= "BGR",
            caption= " Original Image",
            width = img_width
        )
else:
    st.info("Upload Images")

