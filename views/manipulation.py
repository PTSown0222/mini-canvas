import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from typing import Literal

from src.ai_segmentation import remove_background_ai
from src.basic_adjust import (
    adjust_brightness_contrast,
    adjust_saturation,
    rotate_image,
    flip_image,
    adjust_color_temperature,
    adjust_tint,
    adjust_vibrance,
    apply_clahe,
    apply_unsharp_mask,
    crop_aspect_ratio,
)
from src.enhance_image import (
    denoiseAndSmoothImage,
    edgeDetection,
)

# --- 1. Session State Image Verification ---
if "current_image" not in st.session_state or st.session_state.current_image is None:
    st.warning("⚠️ No image found. Please navigate to the Home page to upload an image.")
    st.stop()

image = st.session_state.current_image

# --- 2. Sidebar Configuration ---
st.sidebar.header("🛠️ Processing Pipeline")

category = st.sidebar.radio(
    "Choose Category",
    [
        "Color & Tone Grading",
        "Geometric Transform",
        "Denoise / Smoothing",
        "Edge Detection",
        "🤖 AI Background Removal"
    ],
    key="process_category",
)

filter_params = {}
res = image.copy()
method = ""

# Deep learning remove background
if category == "🤖 AI Background Removal":
    st.sidebar.subheader("One-Click Foreground Cutout")
    method = "AI Cutout"
    
    model_choice = st.sidebar.selectbox(
        "Select Model Weight",
        ["u2netp", "birefnet-general-lite (High Precision)"]
    )
    chosen_model = "u2netp" if "u2netp" in model_choice else "birefnet-general-lite"
    
    if st.sidebar.button("✨Remove Background Now", width = "stretch"):
        with st.spinner("AI is analyzing foreground..."):
            st.session_state["bg_removed_result"] = remove_background_ai(image, model_name = chosen_model)
            
    if "bg_removed_result" in st.session_state:
        res = st.session_state["bg_removed_result"]
    else:
        st.info("👈 Click the button in sidebar to run AI Background Removal.")

# --- Color & Tone Grading ---
elif category == "Color & Tone Grading":
    method = "color_grading"
    
    # 1. White Balance & Temperature
    st.sidebar.subheader("🌡️ White Balance & Temperature")
    temperature = st.sidebar.slider(
        "Color Temperature (Cool ❄️ / Warm ☀️)",
        min_value=-80,
        max_value=80,
        value=0,
        step=2,
        help="Adjusts balance along Blue-Red spectrum."
    )
    tint = st.sidebar.slider(
        "Tint (Green 🌿 / Magenta 🌸)",
        min_value=-60,
        max_value=60,
        value=0,
        step=2,
        help="Adjusts balance along Green-Magenta axis."
    )

    # 2. Exposure, Contrast & Vibrance
    st.sidebar.markdown("---")
    st.sidebar.subheader("☀️ Exposure & Dynamics")
    brightness = st.sidebar.slider("Exposure (Brightness)", min_value=-100, max_value=100, value=0, step=5)
    contrast = st.sidebar.slider("Contrast Multiplier", min_value=0.2, max_value=2.5, value=1.0, step=0.05)
    vibrance = st.sidebar.slider("Vibrance (Smart Saturation)", min_value=0.0, max_value=2.5, value=1.0, step=0.05)
    saturation = st.sidebar.slider("Global Saturation Scale", min_value=0.0, max_value=2.5, value=1.0, step=0.05)

    # 3. Adaptive Contrast & Sharpening
    st.sidebar.markdown("---")
    st.sidebar.subheader("✨ Dynamic Detail Recovery")
    enable_clahe = st.sidebar.checkbox("Enable CLAHE (Shadow Recovery)", value=False)
    clahe_clip = 2.0
    if enable_clahe:
        clahe_clip = st.sidebar.slider("CLAHE Clip Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.2)

    enable_unsharp = st.sidebar.checkbox("Enable Unsharp Masking", value=False)
    unsharp_strength = 0.6
    if enable_unsharp:
        unsharp_strength = st.sidebar.slider("Sharpening Strength", min_value=0.1, max_value=2.0, value=0.6, step=0.1)

    # Pipeline Execution
    if enable_clahe:
        res = apply_clahe(res, clip_limit=clahe_clip)

    res = adjust_brightness_contrast(res, brightness=brightness, contrast=contrast)
    res = adjust_color_temperature(res, temperature=temperature)
    res = adjust_tint(res, tint=tint)
    res = adjust_vibrance(res, vibrance_scale=vibrance)
    res = adjust_saturation(res, saturation_scale=saturation)

    if enable_unsharp:
        res = apply_unsharp_mask(res, strength=unsharp_strength)

# --- Geometric Transform ---
elif category == "Geometric Transform":
    st.sidebar.subheader("Geometric Operations")
    method = st.sidebar.selectbox(
        "Action", 
        ["Rotate", "Flip", "Aspect Ratio Crop"],
        key="geom_action"
    )

    if method == "Rotate":
        angle = st.sidebar.select_slider("Select Angle (Clockwise)", options=[0, 90, 180, 270], value=0)
        res = rotate_image(image, angle=angle) if angle != 0 else image
    elif method == "Flip":
        direction = st.sidebar.radio("Direction", ["horizontal", "vertical"], key="flip_direction")
        res = flip_image(image, direction=direction)
    elif method == "Aspect Ratio Crop":
        ratio_preset = st.sidebar.selectbox("Preset Ratio", ["1:1", "16:9", "4:3", "9:16"])
        res = crop_aspect_ratio(image, ratio_type=ratio_preset)

# --- C. Denoise / Smoothing ---
elif category == "Denoise / Smoothing":
    st.sidebar.subheader("Smoothing Filters")
    method = st.sidebar.selectbox(
        "Select Filter",
        ["Gaussian", "Bilateral", "Median", "Sharpen"],
        key="enhance_method",
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

# --- D. Edge Detection ---
elif category == "Edge Detection":
    st.sidebar.subheader("Edge Detection Algorithms")
    method = st.sidebar.selectbox(
        "Select Algorithm",
        ["canny", "sobel", "prewitt"],
        key="edge_method",
    )
    res = edgeDetection(image, method=method)

# --- 3. Main View Display ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, channels="BGR", width= "stretch")

with col2:
    display_title = method.replace("_", " ").title() if method else category
    st.subheader(f"Processed: {display_title}")

    if res is not None:
        if len(res.shape) == 2:
            channels_mode = "GRAY"
            save_img = res
        elif res.shape[2] == 4:
            channels_mode = "RGBA"  # Dùng RGBA cho ảnh tách nền
            save_img = cv2.cvtColor(
                res, cv2.COLOR_RGBA2BGRA
            )  # Đổi sang BGRA riêng cho OpenCV encode
        else:
            channels_mode = "BGR"
            save_img = res

        # 1. Hiển thị lên web bằng đúng thứ tự RGBA
        st.image(res, channels=channels_mode, width="stretch")

        # 2. Lưu file PNG trong suốt
        is_success, buffer = cv2.imencode(".png", save_img)
        if is_success:
            btn_clicked = st.download_button(
                label=f"⬇️ Download {display_title} Image",
                data=buffer.tobytes(),
                file_name=f"{method.lower()}_result.png",
                mime="image/png",
                width="stretch",
            )
            if btn_clicked:
                st.toast("Image downloaded successfully!", icon="✅")

# --- 4. Real-time Histogram Display ---
if category == "Color & Tone Grading" and len(res.shape) == 3:
    st.markdown("---")
    with st.expander("📊 Real-Time RGB Color Distribution (Histogram)", expanded=False):
        b_hist = cv2.calcHist([res], [0], None, [256], [0, 256]).flatten()
        g_hist = cv2.calcHist([res], [1], None, [256], [0, 256]).flatten()
        r_hist = cv2.calcHist([res], [2], None, [256], [0, 256]).flatten()

        df_hist = pd.DataFrame({"Blue": b_hist, "Green": g_hist, "Red": r_hist})
        st.line_chart(df_hist, color=["#0000FF", "#00FF00", "#FF0000"])

# --- 5. Technical Documentation ---
with st.expander(f"ℹ️ Technical Specification: {display_title}"):
    if category == "Color & Tone Grading":
        st.markdown("""
        * **Color Temperature:** Shifts the white point towards warm (orange/red) or cool (cyan/blue) tones by modifying BGR cross-channel ratios.
        * **Tint Correction:** Offsets chromatic imbalance along the Green-Magenta orthogonal axis.
        * **Vibrance vs. Saturation:** Vibrance selectively boosts lower-saturated background tones while protecting near-saturated areas and human skin hues, whereas Saturation scales all pixels linearly.
        * **CLAHE:** Equalizes dynamic range locally within split CIELAB lightness tiles, avoiding highlight burnout while boosting shadow details.
        * **Unsharp Masking:** Generates high-frequency detail masks by subtracting a Gaussian-blurred duplicate from the original image matrix.
        """)

    elif category == "Geometric Transform":
        st.markdown("""
        * **Orthogonal Rotation:** Matrix transposition and axis flipping via `cv2.rotate` for zero-loss 90-degree step transformations.
        * **Reflection (Flip):** Reverses coordinate arrays along horizontal or vertical symmetry axes via `cv2.flip`.
        * **Mathematical Aspect Ratio ($R$):** Defined as the ratio between width and height:
          $$R = \\frac{W}{H}$$
        * **Center-Cropping Logic:**
          * **Case 1 ($R_{\\text{current}} > R_{\\text{target}}$):** The image is wider than the target frame. Height is preserved ($H_{\\text{new}} = H$), while excess width is symmetrically trimmed from both edges:
            $$W_{\\text{new}} = H \\times R_{\\text{target}}, \\quad \\Delta X = \\frac{W - W_{\\text{new}}}{2}$$
          * **Case 2 ($R_{\\text{current}} < R_{\\text{target}}$):** The image is taller than the target frame. Width is preserved ($W_{\\text{new}} = W$), while excess height is symmetrically trimmed from top and bottom:
            $$H_{\\text{new}} = \\frac{W}{R_{\\text{target}}}, \\quad \\Delta Y = \\frac{H - H_{\\text{new}}}{2}$$
        * **Zero Geometric Distortion:** Unlike interpolation resizing (`cv2.resize`), center-cropping extracts native NumPy sub-arrays (`image[y:y+h, x:x+w]`), preserving subject geometry, sharpness, and pixel aspect ratios without stretching.
        * **Preset Profiles:**
          * **1:1:** Square profile optimized for social avatars and profile pictures.
          * **16:9:** High-definition widescreen standard for desktop monitors and YouTube banners.
          * **4:3:** Standard format matching classic digital camera sensors and medium format photography.
          * **9:16:** Vertical full-screen aspect ratio targeted for mobile feeds (TikTok, Reels, Shorts).
        """)

    elif method == "Gaussian":
        st.markdown("""
        * **Gaussian Filtering:** 2D spatial convolution kernel weighting neighboring pixels according to a bivariate Gaussian distribution. Reduces high-frequency Gaussian noise.
        """)

    elif method == "Bilateral":
        st.markdown("""
        * **Bilateral Filtering:** Non-linear combination of geometric farness Gaussian weighting and radiometric pixel value similarity. Smooths textures while preserving sharp structural edges.
        """)

    elif method == "Median":
        st.markdown("""
        * **Median Filtering:** Non-linear statistical filter replacing center pixels with local rank medians. Highly resistant to impulse and salt-and-pepper noise.
        """)

    elif method == "Sharpen":
        st.markdown("""
        * **Kernel Sharpening:** Convolves a Discrete Laplacian kernel $\\begin{bmatrix} 0 & -1 & 0 \\\\ -1 & 5 & -1 \\\\ 0 & -1 & 0 \\end{bmatrix}$ to amplify edge transitions.
        """)

    elif method == "canny":
        st.markdown("""
        * **Canny Edge Pipeline:** Computes Sobel gradient magnitudes, applies Non-Maximum Suppression (NMS) for directional edge thinning, and resolves continuous lines via dual-threshold hysteresis.
        """)

    elif method == "sobel":
        st.markdown("""
        * **Sobel Operator:** Computes directional image derivatives via discrete difference approximations: $G = \\sqrt{G_x^2 + G_y^2}$.
        """)

    elif method == "prewitt":
        st.markdown("""
        * **Prewitt Operator:** Computes horizontal and vertical gradient vectors using uniform smoothing kernels.
        """)
    elif method == "ai_cutout":
        st.markdown("""
        * **Deep Semantic Segmentation:** Uses an ONNX-optimized neural network (`u2netp` or `birefnet`) to predict an accurate alpha transparency matte.
        * **Alpha Matting:** Generates continuous edge probabilities around complex regions (e.g., hair strands, fabrics) to produce an anti-aliased 4-channel BGRA matrix.
        * **Lossless PNG Export:** Exports directly with full transparency preservation, suitable for compositing.
        """)