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
    return F.conv2d(image, kernel, padding=padding, groups=channels)


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


def fuse_low_frequency(mri_low, ct_low, histogram_weight=0.5):
    """Blend low-frequency structure after matching CT contrast to MRI."""
    matched_ct = histogram_match(ct_low, mri_low)
    return histogram_weight * mri_low + (1.0 - histogram_weight) * matched_ct


def fuse_high_frequency(mri_highs, ct_highs):
    """Keep the sharper detail coefficient at each pixel and scale."""
    fused_highs = []
    for mri_high, ct_high in zip(mri_highs, ct_highs):
        use_mri = torch.abs(mri_high) >= torch.abs(ct_high)
        fused_highs.append(torch.where(use_mri, mri_high, ct_high))
    return fused_highs


def multiscale_fuse(mri, ct, levels=3, kernel_size=5, sigma=1.0):
    """Fuse paired grayscale MRI/CT tensors using MSFD-like decomposition."""
    mri = _as_bchw(mri)
    ct = _as_bchw(ct).to(mri.device)

    if mri.shape != ct.shape:
        raise ValueError(f"MRI and CT tensors must have same shape, got {mri.shape} and {ct.shape}")

    mri_parts = msfd_decompose(mri, levels=levels, kernel_size=kernel_size, sigma=sigma)
    ct_parts = msfd_decompose(ct, levels=levels, kernel_size=kernel_size, sigma=sigma)

    fused_low = fuse_low_frequency(mri_parts["low"], ct_parts["low"])
    fused_highs = fuse_high_frequency(mri_parts["highs"], ct_parts["highs"])
    return reconstruct_from_decomposition(fused_low, fused_highs)
