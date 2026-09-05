import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["U2NET_HOME"] = str(PROJECT_ROOT / "models")
import cv2
import numpy as np
import streamlit as st
from rembg import remove, new_session


@st.cache_resource
def get_segmentation_session(model_name: str = "u2netp"):
    """
    Initialize and cache an ONNX inference session to prevent redundant model
    reloading across Streamlit rerun cycles.

    Args:
        model_name (str, optional): Pretrained model weight profile.
            - 'u2netp': Lightweight U-Net variant (~4.7MB) optimized for CPU throughput.
            - 'birefnet-general-lite': High-resolution bilateral reference model (~100MB).
            Defaults to "u2netp".

    Returns:
        rembg.Session: Persistent inference session instance.
    """
    return new_session(model_name)


def remove_background_ai(image: np.ndarray, model_name: str = "u2netp") -> np.ndarray:
    """Perform one-click foreground segmentation and alpha matting.

    Args:
        image (np.ndarray): Input image array in BGR format (OpenCV standard).
        model_name (str, optional): Model weight identifier. Defaults to
        "u2netp".

    Returns:
        np.ndarray: 4-channel RGBA image array.
    """
    session = get_segmentation_session(model_name)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_rgba = remove(rgb_image, session=session)

    return output_rgba