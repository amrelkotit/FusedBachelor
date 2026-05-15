import torch
import torch.nn.functional as F


def _as_bchw(image):
    """Return image as float tensor shaped (B, C, H, W) in [0, 1]."""
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    image = image.float()
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D image tensor, got {image.shape}")

    if image.max() > 1.0:
        image = image / 255.0
    return image.clamp(0.0, 1.0)


def gaussian_kernel(kernel_size=5, sigma=1.0, channels=1, device=None, dtype=None):
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    image = _as_bchw(image)
    channels = image.shape[1]
    kernel = gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=image.device,
        dtype=image.dtype,
    )
    padding = kernel_size // 2
    padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels)


def local_activity(image, kernel_size=5):
    """Local variance proxy used for adaptive base/detail fusion."""
    padding = kernel_size // 2
    mean = F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(image * image, kernel_size=kernel_size, stride=1, padding=padding)
    return (mean_sq - mean.pow(2)).clamp_min(0.0)


def msfd_decompose(image, levels=3, kernel_size=5, sigma=1.0):
    """Laplacian-pyramid style MSFD approximation."""
    current = _as_bchw(image)
    highs = []

    for _ in range(levels):
        low = gaussian_blur(current, kernel_size=kernel_size, sigma=sigma)
        highs.append(current - low)
        current = low

    return {"low": current, "highs": highs}


def reconstruct_from_decomposition(low, highs):
    """Reconstruct an image from a low-frequency base and detail bands."""
    fused = _as_bchw(low)
    for high in highs:
        fused = fused + high
    return fused.clamp(0.0, 1.0)


def histogram_match(source, reference, bins=256):
    """Match source intensity distribution to reference for rule-based fusion."""
    source = _as_bchw(source)
    reference = _as_bchw(reference).to(source.device)

    matched = torch.empty_like(source)
    source_flat = source.view(-1, source.shape[-2], source.shape[-1])
    reference_flat = reference.view(-1, reference.shape[-2], reference.shape[-1])
    matched_flat = matched.view(-1, matched.shape[-2], matched.shape[-1])

    for index in range(source_flat.shape[0]):
        src = source_flat[index]
        ref = reference_flat[index]

        src_values = torch.clamp((src.flatten() * (bins - 1)).long(), 0, bins - 1)
        ref_values = torch.clamp((ref.flatten() * (bins - 1)).long(), 0, bins - 1)

        src_hist = torch.bincount(src_values, minlength=bins).float()
        ref_hist = torch.bincount(ref_values, minlength=bins).float()
        src_cdf = torch.cumsum(src_hist, dim=0) / src_hist.sum().clamp_min(1.0)
        ref_cdf = torch.cumsum(ref_hist, dim=0) / ref_hist.sum().clamp_min(1.0)

        lookup = torch.searchsorted(ref_cdf, src_cdf).clamp(0, bins - 1).float()
        matched_flat[index] = lookup[src_values].view_as(src) / (bins - 1)

    return matched


def fuse_low_frequency(source2_low, source1_low, histogram_weight=0.5):
    """Blend base layers while favoring the locally more informative modality."""
    matched_source1 = histogram_match(source1_low, source2_low)
    source2_activity = local_activity(source2_low)
    source1_activity = local_activity(matched_source1)
    source2_weight = (source2_activity + 1e-6) / (source2_activity + source1_activity + 2e-6)
    adaptive = source2_weight * source2_low + (1.0 - source2_weight) * matched_source1
    average = histogram_weight * source2_low + (1.0 - histogram_weight) * matched_source1
    return 0.7 * adaptive + 0.3 * average


def fuse_high_frequency(source2_highs, source1_highs):
    """Soft-select sharp detail coefficients instead of hard max switching."""
    fused_highs = []
    for source2_high, source1_high in zip(source2_highs, source1_highs):
        source2_strength = torch.abs(source2_high)
        source1_strength = torch.abs(source1_high)
        weights = torch.softmax(torch.cat([source2_strength, source1_strength], dim=1) * 8.0, dim=1)
        source2_weight = weights[:, 0:1]
        source1_weight = weights[:, 1:2]
        fused_highs.append(source2_weight * source2_high + source1_weight * source1_high)
    return fused_highs


def multiscale_fuse(source1=None, source2=None, levels=3, kernel_size=5, sigma=1.0, **legacy_kwargs):
    """Fuse paired grayscale tensors using MSFD-like decomposition."""
    if source1 is None and "ct" in legacy_kwargs:
        source1 = legacy_kwargs.pop("ct")
    if source2 is None and "mri" in legacy_kwargs:
        source2 = legacy_kwargs.pop("mri")
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    if source1 is None or source2 is None:
        raise TypeError("multiscale_fuse() requires source1 and source2 tensors")

    source1 = _as_bchw(source1)
    source2 = _as_bchw(source2).to(source1.device)

    if source1.shape != source2.shape:
        raise ValueError(f"Input tensors must have same shape, got {source1.shape} and {source2.shape}")

    source1_parts = msfd_decompose(source1, levels=levels, kernel_size=kernel_size, sigma=sigma)
    source2_parts = msfd_decompose(source2, levels=levels, kernel_size=kernel_size, sigma=sigma)

    fused_low = fuse_low_frequency(source2_parts["low"], source1_parts["low"])
    fused_highs = fuse_high_frequency(source2_parts["highs"], source1_parts["highs"])
    return reconstruct_from_decomposition(fused_low, fused_highs)
