from pathlib import Path

import cv2
import numpy as np
import torch

from src.fusion.decomposition import multiscale_fuse


def initial_fusion(source1, source2):
    return (0.5 * source1 + 0.5 * source2).clamp(0.0, 1.0)


def fusion_reference(source1, source2, use_msfd=True):
    if use_msfd:
        try:
            return multiscale_fuse(source1=source1, source2=source2).clamp(0.0, 1.0)
        except Exception:
            pass
    return initial_fusion(source1, source2)


def tensor_to_uint8(tensor):
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    return (image * 255.0).round().astype("uint8")


def tensor_to_numpy01(tensor):
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy().astype("float32")
    if image.ndim == 3:
        image = image[0]
    return image


def uint8_to_tensor(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy((image.astype("float32") / 255.0).clip(0.0, 1.0)).unsqueeze(0)


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), tensor_to_uint8(tensor), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def save_uint8_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def percentile_normalize(image, low=1.0, high=99.0):
    image = np.asarray(image, dtype="float32")
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo + 1e-8:
        return np.clip(image, 0.0, 1.0)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def postprocess_grayscale(image, normalize_output=False, postprocess=False, enhance_edges=False):
    image = np.asarray(image, dtype="float32")
    if normalize_output:
        image = percentile_normalize(image)
    image = np.clip(image, 0.0, 1.0)
    if postprocess:
        u8 = (image * 255.0).round().astype("uint8")
        u8 = cv2.bilateralFilter(u8, d=3, sigmaColor=8, sigmaSpace=3)
        image = u8.astype("float32") / 255.0
    if enhance_edges:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=0.8)
        image = np.clip(image + 0.10 * (image - blurred), 0.0, 1.0)
    return image


def presentation_enhance_grayscale(image, output_size=512, local_contrast=True, unsharp=True):
    image = np.clip(np.asarray(image, dtype="float32"), 0.0, 1.0)
    u8 = (image * 255.0).round().astype("uint8")
    if local_contrast:
        clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
        u8 = clahe.apply(u8)
    enhanced = u8.astype("float32") / 255.0
    if unsharp:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.65)
        enhanced = np.clip(enhanced + 0.18 * (enhanced - blurred), 0.0, 1.0)
    if output_size and int(output_size) > 0 and enhanced.shape[:2] != (int(output_size), int(output_size)):
        enhanced = cv2.resize(enhanced, (int(output_size), int(output_size)), interpolation=cv2.INTER_LANCZOS4)
    return enhanced


def resize_rgb_for_presentation(image_rgb, output_size=512):
    image = np.asarray(image_rgb)
    if not output_size or int(output_size) <= 0 or image.shape[:2] == (int(output_size), int(output_size)):
        return image
    return cv2.resize(image, (int(output_size), int(output_size)), interpolation=cv2.INTER_LANCZOS4)


COLORMAPS = {
    "hot": cv2.COLORMAP_HOT,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def apply_medical_colormap(functional_gray, colormap="hot"):
    key = str(colormap or "hot").lower()
    if key not in COLORMAPS:
        allowed = ", ".join(sorted(COLORMAPS))
        raise ValueError(f"Unsupported colormap '{colormap}'. Allowed values: {allowed}")
    functional = percentile_normalize(functional_gray)
    color_bgr = cv2.applyColorMap((functional * 255.0).round().astype("uint8"), COLORMAPS[key])
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def colorize_functional_fusion(grayscale, functional_source, alpha=0.35, colormap="hot"):
    gray = np.clip(grayscale, 0.0, 1.0)
    functional = np.clip(functional_source, 0.0, 1.0)
    if functional.shape != gray.shape:
        functional = cv2.resize(functional, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    gray_rgb = np.repeat((gray * 255.0).round().astype("uint8")[:, :, None], 3, axis=2).astype("float32")
    functional_rgb = apply_medical_colormap(functional, colormap=colormap).astype("float32")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.clip((1.0 - alpha) * gray_rgb + alpha * functional_rgb, 0, 255).round().astype("uint8")


def read_rgb_image01(path, image_size=None):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return (image.astype("float32") / 255.0).clip(0.0, 1.0)


def image_has_color_style(rgb_image, min_channel_delta=0.015):
    image = np.asarray(rgb_image, dtype="float32")
    if image.ndim != 3 or image.shape[2] < 3:
        return False
    channel_delta = np.mean(np.max(image[:, :, :3], axis=2) - np.min(image[:, :, :3], axis=2))
    return bool(channel_delta > min_channel_delta)


def blend_grayscale_with_rgb_style(grayscale, style_rgb, alpha=0.35):
    gray = np.clip(grayscale, 0.0, 1.0)
    style = np.clip(style_rgb, 0.0, 1.0)
    if style.shape[:2] != gray.shape:
        style = cv2.resize(style, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    gray_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    colored = np.clip((1.0 - alpha) * gray_rgb + alpha * style[:, :, :3], 0.0, 1.0)
    return (colored * 255.0).round().astype("uint8")


def contrast_adjust01(image, contrast=1.0):
    image = np.clip(np.asarray(image, dtype="float32"), 0.0, 1.0)
    contrast = float(max(0.0, contrast))
    if contrast == 1.0:
        return image
    mask = image > 0.0
    pivot = float(np.mean(image[mask])) if np.any(mask) else 0.5
    return np.clip((image - pivot) * contrast + pivot, 0.0, 1.0)


def hsv_color_transfer_from_reference(
    grayscale,
    reference_rgb,
    alpha=0.65,
    color_strength=1.0,
    saturation_boost=1.35,
    contrast_boost=1.15,
    background_threshold=0.03,
):
    gray = np.clip(np.asarray(grayscale, dtype="float32"), 0.0, 1.0)
    reference = np.clip(np.asarray(reference_rgb, dtype="float32"), 0.0, 1.0)
    if reference.shape[:2] != gray.shape:
        reference = cv2.resize(reference, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)

    reference_u8 = (reference[:, :, :3] * 255.0).round().astype("uint8")
    hsv = cv2.cvtColor(reference_u8, cv2.COLOR_RGB2HSV).astype("float32")
    value = contrast_adjust01(percentile_normalize(gray), contrast=contrast_boost)
    mask = value > float(background_threshold)

    saturation_scale = max(0.0, float(saturation_boost)) * max(0.0, float(color_strength))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0.0, 255.0)
    hsv[:, :, 2] = np.clip(value * 255.0, 0.0, 255.0)

    transferred = cv2.cvtColor(hsv.round().astype("uint8"), cv2.COLOR_HSV2RGB).astype("float32") / 255.0
    gray_rgb = np.repeat(value[:, :, None], 3, axis=2)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    colored = np.clip((1.0 - alpha) * gray_rgb + alpha * transferred, 0.0, 1.0)
    colored[~mask] = gray_rgb[~mask]
    return (colored * 255.0).round().astype("uint8")


def hsv_colorize_colormap_fallback(
    grayscale,
    functional_source,
    alpha=0.65,
    color_strength=1.0,
    saturation_boost=1.35,
    contrast_boost=1.15,
    background_threshold=0.03,
    colormap="hot",
):
    functional_rgb = apply_medical_colormap(functional_source, colormap=colormap).astype("float32") / 255.0
    return hsv_color_transfer_from_reference(
        grayscale,
        functional_rgb,
        alpha=alpha,
        color_strength=color_strength,
        saturation_boost=saturation_boost,
        contrast_boost=contrast_boost,
        background_threshold=background_threshold,
    )


def gan_colored_reference_path(gan_root, pair, split, stem):
    if gan_root is None:
        return None
    root = Path(gan_root)
    candidates = [
        root / "images" / "aanlib" / split / pair / "fused_color" / f"{stem}_fused.png",
        root / "images" / "aanlib" / split / pair / "fused_colored" / f"{stem}_fused.png",
    ]
    return next((path for path in candidates if path.exists()), None)


def smart_colorize_functional_fusion(
    grayscale,
    source_path,
    pair,
    split,
    stem,
    gan_root=None,
    color_mode="smart",
    alpha=0.65,
    color_strength=1.0,
    saturation_boost=1.35,
    contrast_boost=1.15,
    background_threshold=0.03,
    colormap="hot",
):
    if pair == "ct_mri":
        return None, "none", "none", ""

    requested = str(color_mode or "smart").strip().lower().replace("_", "-")
    if requested not in {"smart", "source", "gan-style", "colormap"}:
        raise ValueError("color_mode must be one of: smart, source, gan-style, colormap")

    gan_ref = gan_colored_reference_path(gan_root, pair, split, stem)
    if requested in {"smart", "gan-style"} and gan_ref is not None:
        gan_rgb = read_rgb_image01(gan_ref, image_size=grayscale.shape[::-1])
        colored = hsv_color_transfer_from_reference(
            grayscale,
            gan_rgb,
            alpha=alpha,
            color_strength=color_strength,
            saturation_boost=saturation_boost,
            contrast_boost=contrast_boost,
            background_threshold=background_threshold,
        )
        return colored, "gan-style", "gan_style", str(gan_ref)
    if requested == "gan-style":
        requested = "source"

    if requested in {"smart", "source"}:
        source_rgb = read_rgb_image01(source_path, image_size=grayscale.shape[::-1])
        if image_has_color_style(source_rgb):
            colored = hsv_color_transfer_from_reference(
                grayscale,
                source_rgb,
                alpha=alpha,
                color_strength=color_strength,
                saturation_boost=saturation_boost,
                contrast_boost=contrast_boost,
                background_threshold=background_threshold,
            )
            return colored, "source", "source_rgb", str(source_path)
        if requested == "source":
            requested = "colormap"

    functional = read_gray_image(source_path, image_size=grayscale.shape[::-1])
    colored = hsv_colorize_colormap_fallback(
        grayscale,
        functional,
        alpha=alpha,
        color_strength=color_strength,
        saturation_boost=saturation_boost,
        contrast_boost=contrast_boost,
        background_threshold=background_threshold,
        colormap=colormap,
    )
    return colored, "colormap", "colormap_fallback", str(source_path)


def read_gray_image(path, image_size=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image_size is not None:
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = image.astype("float32")
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def read_gray_uint01(path, image_size=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image_size is not None:
        size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return (image.astype("float32") / 255.0).clip(0.0, 1.0)


def save_comparison_panel(images, labels, path):
    arrays = [tensor_to_uint8(image) for image in images]
    panel = np.concatenate(arrays, axis=1)
    label_h = 30
    canvas = np.full((panel.shape[0] + label_h, panel.shape[1]), 255, dtype=np.uint8)
    canvas[label_h:, :] = panel
    width = arrays[0].shape[1]
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (index * width + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save comparison image: {path}")


def read_gray_tensor(path, image_size=256):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype("float32")
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
