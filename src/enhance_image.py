"""
This file contain some functions to adjust low quality images into high quality images
1. Denoising / Smoothing
2. Sharpening
3. Edge Detection filter: Sobel, Prewitt, Canny Edge 
"""

import cv2
import PIL
import glob
import os
import numpy as np

def imageAnalysis(image: np.ndarray) -> None:
    print("=" * 40)
    if image is None:
        print("Image is None! Check directions")
        print("=" * 40)
        return
    # check shape
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
        img_type = "Grayscale"
    elif len(image.shape) == 3:
        h, w, channels = image.shape
        img_type = f"Color ({channels} channels - BGR/RGB)" if channels == 3 else f"{channels}-channel image"
    else:
        img_type = "Multi-dimensional tensor"

    print(f"Type:             {img_type}")
    print(f"Shape:            {image.shape} -> (Height = {h}, Width = {w}, Channels = {channels})")
    print(f"Data type:        {image.dtype}")
    print(f"Pixel Range:      Min = {image.min()} | Max = {image.max()}")
    print(f"Pixel Stats:      Mean = {image.mean():.2f} | Std = {image.std():.2f}")
    print(f"Memory Size:      {image.nbytes / 1024:.2f} KB")
    print("=" * 40)

def denoiseAndSmoothImage(image: np.ndarray, method: str = "Bilateral", **kwargs) -> np.ndarray:
    if method == "Gaussian":
        ksize = kwargs.get("ksize", (9, 9))
        return cv2.GaussianBlur(image, ksize, 0)
        
    elif method == "Bilateral":
        d = kwargs.get("d", 15)
        sigma_color = kwargs.get("sigma_color", 100)
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_color)
        
    elif method == "Median":
        ksize = kwargs.get("ksize", 5)
        return cv2.medianBlur(image, ksize)
        
    elif method == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        
    return image


def edgeDetection(image: np.ndarray, method: str = "canny") -> np.ndarray:
    # Chuẩn hóa ảnh về Grayscale nếu là ảnh màu
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    method = method.lower()

    if method == "canny":
        return cv2.Canny(gray_image, threshold1=100, threshold2=200)

    elif method == "sobel":
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.hypot(sobel_x, sobel_y)
        return cv2.convertScaleAbs(magnitude)

    elif method == "prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
        magnitude = np.hypot(prewitt_x, prewitt_y)
        return cv2.convertScaleAbs(magnitude)

    else:
        raise ValueError(f"Method '{method}' is not supported. Choose from: 'canny', 'sobel', 'prewitt'")

if __name__ == "__main__":
    root = "data"
    image_paths = glob.glob(os.path.join(root, "*.png"))
    
    if image_paths:
        image_path = image_paths[0]
        image = cv2.imread(image_path)
        imageAnalysis(image)

        # Test các filter
        canny_edges = edgeDetection(image, method="canny")
        sobel_edges = edgeDetection(image, method="sobel")
        prewitt_edges = edgeDetection(image, method="prewitt")

        cv2.imshow("image",prewitt_edges)
        cv2.waitKey(0)

        print("Edge detection executed successfully.")
    else:
        print("No .png files found in directory.")


