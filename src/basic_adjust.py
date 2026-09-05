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

import cv2
import numpy as np


def adjust_color_temperature(image: np.ndarray, temperature: int = 0) -> np.ndarray:
    """
    Adjust the color temperature to make the image appear warmer (golden/orange)
    or cooler (blueish).

    Args:
        image (np.ndarray): Input image in BGR format.
        temperature (int, optional): Temperature shift offset, range [-100, 100].
            - Positive values (> 0): Warmer tone (boosts Red, attenuates Blue).
            - Negative values (< 0): Cooler tone (boosts Blue, attenuates Red).
            Defaults to 0.

    Returns:
        np.ndarray: Temperature-adjusted image in BGR format.
    """
    if temperature == 0:
        return image

    img_float = image.astype(np.float32)
    # BGR indices: 0: Blue, 1: Green, 2: Red
    if temperature > 0:
        img_float[:, :, 2] += temperature
        img_float[:, :, 0] -= temperature * 0.4
    else:
        img_float[:, :, 0] += abs(temperature)
        img_float[:, :, 2] -= abs(temperature) * 0.4

    return np.clip(img_float, 0, 255).astype(np.uint8)


def adjust_tint(image: np.ndarray, tint: int = 0) -> np.ndarray:
    """
    Adjust the color tint balance along the Green-Magenta axis.

    Args:
        image (np.ndarray): Input image in BGR format.
        tint (int, optional): Tint shift offset, range [-100, 100].
            - Positive values (> 0): Magenta shift (decreases Green channel).
            - Negative values (< 0): Green shift (increases Green channel).
            Defaults to 0.

    Returns:
        np.ndarray: Tint-adjusted image in BGR format.
    """
    if tint == 0:
        return image

    img_float = image.astype(np.float32)
    # Green is at index 1
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] - tint, 0, 255)
    return img_float.astype(np.uint8)


def adjust_vibrance(image: np.ndarray, vibrance_scale: float = 1.0) -> np.ndarray:
    """
    Smart saturation adjustment that enhances muted colors while preventing
    already saturated areas and human skin tones from becoming over-saturated.

    Args:
        image (np.ndarray): Input image in BGR format.
        vibrance_scale (float, optional): Scaling factor for vibrance, range [0.0, 3.0].
            - 1.0: No adjustment.
            - > 1.0: Selectively increases saturation of less saturated pixels.
            - < 1.0: Reduces overall vibrance.
            Defaults to 1.0.

    Returns:
        np.ndarray: Vibrance-enhanced image in BGR format.
    """
    if abs(vibrance_scale - 1.0) < 1e-4:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0

    # Non-linear enhancement: lower saturated pixels gain more boost
    if vibrance_scale > 1.0:
        factor = (vibrance_scale - 1.0) * (1.0 - saturation)
        hsv[:, :, 1] = np.clip((saturation + factor * saturation) * 255.0, 0, 255)
    else:
        hsv[:, :, 1] = np.clip(saturation * vibrance_scale * 255.0, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel
    in CIELAB color space to recover shadow details without blowing out highlights.

    Args:
        image (np.ndarray): Input image in BGR format.
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid_size (int, optional): Dimension of grid tiles for local histogram calculation. Defaults to 8.

    Returns:
        np.ndarray: Locally contrast-enhanced image in BGR format.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    l_enhanced = clahe.apply(l_channel)

    lab_merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    image: np.ndarray, strength: float = 0.6, kernel_size: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """
    Sharpen the image by extracting and amplifying high-frequency edge details
    via an unsharp masking convolution technique.

    Args:
        image (np.ndarray): Input image in BGR format.
        strength (float, optional): Edge amplification strength, range [0.0, 3.0]. Defaults to 0.6.
        kernel_size (int, optional): Gaussian blur kernel size (must be odd). Defaults to 5.
        sigma (float, optional): Gaussian blur standard deviation. Defaults to 1.0.

    Returns:
        np.ndarray: Sharpened image in BGR format.
    """
    if strength <= 0:
        return image

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    # Formula: Result = Original * (1 + strength) - Blurred * strength
    sharpened = cv2.addWeighted(
        image, 1.0 + strength, blurred, -strength, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)

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

def crop_aspect_ratio(
    image: np.ndarray, ratio_type: Literal["1:1", "16:9", "4:3", "9:16"] = "1:1"
) -> np.ndarray:
    """
    Center crop an image to standard aspect ratios: 1:1, 16:9, 4:3, or 9:16.

    Args:
        image (np.ndarray): Input image array in BGR or Grayscale format.
        ratio_type (Literal["1:1", "16:9", "4:3", "9:16"]): Target crop ratio.

    Returns:
        np.ndarray: Center-cropped image array.
    """
    h, w = image.shape[:2]
    ratio_dict = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "9:16": (9, 16),
    }

    target_rw, target_rh = ratio_dict.get(ratio_type, (w, h))
    target_ratio = target_rw / target_rh
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is too wide: crop horizontally from the center
        new_w = int(h * target_ratio)
        x_offset = (w - new_w) // 2
        return image[:, x_offset : x_offset + new_w]
    else:
        # Image is too tall: crop vertically from the center
        new_h = int(w / target_ratio)
        y_offset = (h - new_h) // 2
        return image[y_offset : y_offset + new_h, :]

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