import cv2
import torch
import numpy as np


def load_image(path):
    """
    Load image in grayscale
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image not found at {path}")

    print(f"[Preprocess] Loaded: {path} | shape: {img.shape}")
    return img


def resize_image(img, size=(256, 256)):
    """
    Resize image to fixed size
    """
    resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    print(f"[Preprocess] Resized: {resized.shape}")
    return resized


def normalize_image(img):
    """
    Normalize to [0, 1]
    """
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    normalized = (img - min_val) / (max_val - min_val + 1e-8)

    print(f"[Preprocess] Normalized: min={normalized.min():.4f}, max={normalized.max():.4f}")
    return normalized


def to_tensor(img):
    """
    Convert numpy image to torch tensor
    Shape: (1, H, W)
    """
    tensor = torch.from_numpy(img).unsqueeze(0)  # add channel
    print(f"[Preprocess] Tensor shape: {tensor.shape}")
    return tensor


def preprocess(image_path):

    img = load_image(image_path)
    img = resize_image(img)
    img = normalize_image(img)
    img = to_tensor(img)
    return img


def enhance_image(img):
    img = (img * 255).astype(np.uint8)

    # Mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    img = clahe.apply(img)

    return img.astype(np.float32) / 255.0
#
# def enhance_image(img):
#     img = (img * 255).astype(np.uint8)

#     # CLAHE (reduce strength)
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
#     img = clahe.apply(img)

#     # VERY LIGHT sharpening (optional)
#     kernel = np.array([[0, -1, 0],
#                        [-1, 4,-1],
#                        [0, -1, 0]])
#     img = cv2.filter2D(img, -1, kernel)

#     img = img.astype(np.float32) / 255.0
#     return img