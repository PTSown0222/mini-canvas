import streamlit as st
import cv2
import numpy as np

st.title("🏠 Mini Canvas - Studio Xử lý Ảnh & AI")
st.write("Tải ảnh của bạn lên tại đây, sau đó chọn các công cụ ở menu bên trái để xử lý.")

uploaded_file = st.file_uploader(
    "Tải ảnh lên (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    key="global_uploader"
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.session_state.current_image = opencv_image
    st.session_state.image_name = uploaded_file.name

if st.session_state.current_image is not None:
    st.success(f"Đang làm việc với file: **{st.session_state.image_name}**")
    st.image(st.session_state.current_image, channels="BGR", caption="Ảnh gốc hiện tại", width="stretch")
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")