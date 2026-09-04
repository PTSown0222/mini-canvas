"""
This module provides core utility functions for basic image transformations,
including brightness and contrast adjustments, color saturation scaling,
geometric rotations, and flipping operations using OpenCV.
"""

import os
import glob
import cv2
import numpy as np
from typing import Literal

def adjust_brightness_contrast(
    image: np.ndarray, brightness: int = 0, contrast: float = 1.0
) -> np.ndarray:
    """
    Adjust brightness and contrast using a linear transformation formula:
    g(x) = alpha * f(x) + beta

    Args:
        image (np.ndarray): Input image in BGR or Grayscale format.
        brightness (int, optional): Brightness offset (beta), range [-100, 100]. Defaults to 0.
        contrast (float, optional): Contrast multiplier (alpha), range [0.0, 3.0]. Defaults to 1.0.

    Returns:
        np.ndarray: Scaled and clipped image array.
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def adjust_saturation(
    image: np.ndarray, saturation_scale: float = 1.0
) -> np.ndarray:
    """
    Adjust the color saturation level via HSV color space.

    Args:
        image (np.ndarray): Input image in BGR format.
        saturation_scale (float, optional): Scaling factor for the Saturation (S) channel.
            - 0.0: Complete grayscale (monochrome).
            - 1.0: Original color saturation.
            - >1.0: Enhanced color saturation.
            Defaults to 1.0.

    Returns:
        np.ndarray: Saturated image in BGR format.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # S (Saturation) channel is located at index 1
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate an image by orthogonal angles (90, 180, or 270 degrees).

    Args:
        image (np.ndarray): Input image array.
        angle (int): Rotation angle in degrees clockwise (90, 180, or 270).

    Returns:
        np.ndarray: Rotated image array.
    """
    rotation_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(image, rotation_map[angle]) if angle in rotation_map else image

def flip_image(image: np.ndarray, direction: Literal["horizontal", "vertical"]) -> np.ndarray:
    """
    Flip an image horizontally or vertically.

    Args:
        image (np.ndarray): Input image array.
        direction (Literal["horizontal", "vertical"]):
            - 'horizontal': Flip around the y-axis (flipCode = 1).
            - 'vertical': Flip around the x-axis (flipCode = 0).

    Returns:
        np.ndarray: Flipped image array.
    """
    flip_code = 1 if direction == "horizontal" else 0
    return cv2.flip(image, flip_code)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(data_dir, ext)))

    if image_paths:
        test_path = image_paths[3]
        print(f"Opening test image: {test_path}")
        
        original = cv2.imread(test_path)
        
        if original is not None:
            adjusted = adjust_brightness_contrast(original, brightness=30, contrast=1.3)
            saturated = adjust_saturation(original, saturation_scale=1.8)
            rotated = rotate_image(original, angle=90)
            flipped = flip_image(original, direction="horizontal")

            cv2.imshow("1. Original Image", original)
            cv2.imshow("2. Brightness & Contrast Adjusted", adjusted)
            cv2.imshow("3. Saturation Enhanced", saturated)
            cv2.imshow("4. Rotated 90 Deg", rotated)
            cv2.imshow("5. Flipped Horizontal", flipped)

            print("Type any keyboards to quit...")
            cv2.waitKey(0)  
            cv2.destroyAllWindows() 
        else:
            print("Cannot read Image")
    else:
        print(f"No Images in this folder: {data_dir}")