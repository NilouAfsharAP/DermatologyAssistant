import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _hair_removal_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Remove hair-like dark strands using blackhat + inpainting."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, 1, cv2.INPAINT_TELEA)

def _clahe_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on L channel and return RGB uint8."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def prepare_input(
    cv_img: np.ndarray,
    *,
    hair_removal: bool = True,
    clahe: bool = True,
    img_size: int = 224,
):
    """Preprocess BGR image for EfficientNet-V2-S.

    Args:
        cv_img: BGR uint8 image (as read by OpenCV).
        hair_removal: Enable hair removal (inpainting).
        clahe: Enable local contrast enhancement (CLAHE).
        img_size: Final square size.

    Returns:
        tensor: Float tensor (1, 3, img_size, img_size) normalized with ImageNet stats.
        rgb_norm: Float RGB image in [0, 1] (img_size, img_size, 3) for Grad-CAM overlay.
    """
    if cv_img is None:
        raise ValueError("Input image is None")

    img = cv_img
    if hair_removal:
        img = _hair_removal_bgr(img)

    if clahe:
        rgb = _clahe_rgb(img)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb_norm = resized.astype(np.float32) / 255.0

    tensor_data = (rgb_norm - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(tensor_data, (2, 0, 1))).float().unsqueeze(0)
    return tensor, rgb_norm
