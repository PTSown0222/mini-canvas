from src.enhance_image import (
    denoiseAndSmoothImage,
    edgeDetection
)
import cv2
import streamlit as st

if st.session_state.current_image is None:
    st.warning(f"No Images. Please Access to home page to upload your images to adjust")
    st.stop()

image = st.session_state.current_image

st.sidebar.header("Processing Mode")
category = st.sidebar.radio(
    "Choose Category",
    ["Denoise / Smoothing", "Edge Detection"],
    key=" process_category"
)

filter_params = {}
res = None

if category == "Denoise / Smoothing":
    st.sidebar.subheader("Smoothing Filters")
    method = st.sidebar.selectbox(
        "Select Filter",
        ["Gaussian", "Bilateral", "Median", "Sharpen"],
        key= " enhance_method"
    )

    if method == "Gaussian":
        k_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=31, value=9, step=2)
        filter_params["ksize"] = (k_size, k_size)
    elif method == "Bilateral":
        filter_params["d"] = st.sidebar.slider("Diameter (d)", min_value=5, max_value=25, value=15, step=2)
        filter_params["sigma_color"] = st.sidebar.slider("Sigma Color", min_value=10, max_value=200, value=100, step=10)
    elif method == "Median":
        filter_params["ksize"] = st.sidebar.slider("Kernel Size", min_value=3, max_value=21, value=5, step=2)
    
    res = denoiseAndSmoothImage(image, method=method, **filter_params)
else:
    st.sidebar.subheader("Edge Detection Algorithms")
    method = st.sidebar.selectbox(
        "Select Algorithm",
        ["canny", "sobel", "prewitt"],
        key="edge_method"
    )
    
    res = edgeDetection(image, method=method)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, channels="BGR", width="stretch")

with col2:
    st.subheader(f"Processed: {method.capitalize()}")
    if res is not None:
        channels_mode = "GRAY" if len(res.shape) == 2 else "BGR"
        st.image(res, channels=channels_mode, width="stretch")

        # download
        is_success, buffer = cv2.imencode(".png", res)
        if is_success:
            btn_clicked = st.download_button(
                label=f"⬇️ Download {method.capitalize()} Image",
                data=buffer.tobytes(),
                file_name=f"{method.lower()}_result.png",
                mime="image/png",
                use_container_width=True
            )
            if btn_clicked:
                st.toast("Downloading Processed Images!", icon="😍")

# --- Explanation Section ---
with st.expander(f"ℹ️ See {method.capitalize()} Explanation"):
    if method == "Gaussian":
        st.markdown("""
        **Gaussian Blur** is a low-pass linear filter that convolves the image with a Gaussian kernel.
        
        * **Kernel Size ($ksize$):** Controls the neighborhood window dimensions. Larger values increase blur intensity.
        * **Use Case:** Removing general high-frequency noise and pre-processing images before edge detection.
        """)

    elif method == "Bilateral":
        st.markdown("""
        **Bilateral Filter** is an edge-preserving, non-linear smoothing filter that considers both spatial distance and pixel intensity differences.
        
        * **Diameter ($d$):** Pixel neighborhood diameter used for filtering.
        * **Sigma Color:** Defines how much difference in color is tolerated before considering it an edge (larger values blend wider color ranges).
        * **Use Case:** Smoothing skin tones or textures while keeping boundaries and contours razor-sharp.
        """)

    elif method == "Median":
        st.markdown("""
        **Median Filter** replaces each pixel value with the median value of neighboring pixels in the kernel window.
        
        * **Kernel Size:** Must be an odd integer (e.g., 3, 5, 7).
        * **Use Case:** Highly effective at eliminating **salt-and-pepper noise** without blurring edges as heavily as Gaussian.
        """)

    elif method == "Sharpen":
        st.markdown("""
        **Sharpening** applies a high-pass kernel matrix to emphasize local contrast and high-frequency details.
        
        * **Kernel Matrix:** $\\begin{bmatrix} 0 & -1 & 0 \\\\ -1 & 5 & -1 \\\\ 0 & -1 & 0 \\end{bmatrix}$
        * **Use Case:** Restoring blurry images, highlighting fine textures, and preparing inputs for feature extraction models.
        """)

    elif method == "canny":
        st.markdown("""
        **Canny Edge Detector** is a multi-stage edge detection pipeline:
        1. Smooths the image with a Gaussian filter to reduce noise.
        2. Computes intensity gradients using Sobel operators.
        3. Applies **Non-Maximum Suppression (NMS)** to thin the edges.
        4. Applies **Hysteresis Thresholding** with upper and lower bounds to link valid edges.
        * **Use Case:** Precise boundary tracing and object localization.
        """)

    elif method == "sobel":
        st.markdown("""
        **Sobel Operator** calculates the first-order image derivative (gradient) separately along the X and Y axes.
        
        * **Sobel X:** Highlights vertical edges.
        * **Sobel Y:** Highlights horizontal edges.
        * **Gradient Magnitude:** Combined via $\\sqrt{G_x^2 + G_y^2}$.
        * **Use Case:** Rapid edge detection and orientation estimation.
        """)

    elif method == "prewitt":
        st.markdown("""
        **Prewitt Operator** works similarly to Sobel by estimating image gradients using discrete differentiation kernels.
        
        * **Difference from Sobel:** Prewitt applies uniform weights without placing higher statistical weight on the center pixel.
        * **Use Case:** Simple horizontal and vertical edge extraction where noise sensitivity is low.
        """)