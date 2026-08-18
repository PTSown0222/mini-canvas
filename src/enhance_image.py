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
    """Apply denoising, smoothing, or sharpening filters to an image.

    Processes the input image using selected spatial filtering algorithms from
    OpenCV. Hyperparameters for specific filters can be configured via keyword
    arguments.

    Args:
        image (np.ndarray): Input image as a NumPy array (Grayscale or BGR).
        method (str, optional): The filtering technique to apply. Supported
            options are 'Gaussian', 'Bilateral', 'Median', and 'Sharpen'.
            Defaults to 'Bilateral'.
        **kwargs: Additional filter-specific parameters:
            - ksize (tuple[int, int] or int, optional): Kernel size. A tuple of
              two odd integers for 'Gaussian' (defaults to (9, 9)), or an odd
              integer for 'Median' (defaults to 5).
            - d (int, optional): Diameter of each pixel neighborhood used during
              filter application in 'Bilateral'. Defaults to 15.
            - sigma_color (float, optional): Filter sigma in the color space for
              'Bilateral'. Defaults to 100.

    Returns:
        np.ndarray: Filtered image array matching the shape and data type of
            the input.

    Example:
        >>> img = cv2.imread("data/input.png")
        >>> smooth_img = denoiseAndSmoothImage(img, method="Gaussian", ksize=(5, 5))
        >>> bilateral_img = denoiseAndSmoothImage(img, method="Bilateral", d=9, sigma_color=75)
    """
    
    if method == "Gaussian":
        ksize = kwargs.get("ksize", (9, 9))
        return cv2.GaussianBlur(image, ksize, 0)
        
    elif method == "Bilateral":
        d = kwargs.get("d", 15)
        sigma_color = kwargs.get("sigma_color", 100)
        return cv2.bilateralFilter(image, d = d, sigmaColor=sigma_color, sigmaSpace = sigma_color)
        
    elif method == "Median":
        ksize = kwargs.get("ksize", 5)
        return cv2.medianBlur(image, ksize)
        
    elif method == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(src = image, ddepth = -1, kernel = kernel)
        
    return image


def edgeDetection(image: np.ndarray, method: str = "canny") -> np.ndarray:
    """Detect edges in an image using specified edge detection algorithms.

    Converts the input image to grayscale if it is a color image (3-channel BGR),
    then applies the selected edge detection filter.

    Args:
        image (np.ndarray): Input image as a NumPy array. Can be a 2D grayscale
            array (H, W) or a 3D BGR color array (H, W, C).
        method (str, optional): The edge detection algorithm to apply.
            Supported methods are 'canny', 'sobel', and 'prewitt'.
            Case-insensitive. Defaults to 'canny'.

    Returns:
        np.ndarray: A 2D uint8 NumPy array representing the detected edge map.

    Raises:
        ValueError: If `method` is not one of 'canny', 'sobel', or 'prewitt'.

    Example:
        >>> img = cv2.imread("data/dora.png")
        >>> sobel_edges = edgeDetection(img, method = "sobel")
        >>> print(sobel_edges.shape)
        (2160, 3840)
    """

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

        canny_edges = edgeDetection(image, method="canny")
        sobel_edges = edgeDetection(image, method="sobel")
        prewitt_edges = edgeDetection(image, method="prewitt")

        # cv2.imshow("image",prewitt_edges)
        # cv2.waitKey(0)

        print(f"Edge detection has {sobel_edges.shape} executed successfully.")
    else:
        print("No .png files found in directory.")


