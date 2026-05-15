import math
import torch
import torch.nn.functional as F


def _as_bchw(image):
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


def psnr(image, reference, data_range=1.0):
    image = _as_bchw(image)
    reference = _as_bchw(reference).to(image.device)
    mse = F.mse_loss(image, reference)
    if mse.item() == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item())


def ssim(image, reference, data_range=1.0, window_size=7):
    image = _as_bchw(image)
    reference = _as_bchw(reference).to(image.device)

    channels = image.shape[1]
    window = torch.ones((channels, 1, window_size, window_size), device=image.device) / (window_size ** 2)
    padding = window_size // 2

    mu_x = F.conv2d(image, window, padding=padding, groups=channels)
    mu_y = F.conv2d(reference, window, padding=padding, groups=channels)
    sigma_x = F.conv2d(image * image, window, padding=padding, groups=channels) - mu_x.pow(2)
    sigma_y = F.conv2d(reference * reference, window, padding=padding, groups=channels) - mu_y.pow(2)
    sigma_xy = F.conv2d(image * reference, window, padding=padding, groups=channels) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().item()


def spatial_frequency(image):
    image = _as_bchw(image)
    row_freq = torch.sqrt(torch.mean((image[..., 1:, :] - image[..., :-1, :]).pow(2)))
    col_freq = torch.sqrt(torch.mean((image[..., :, 1:] - image[..., :, :-1]).pow(2)))
    return torch.sqrt(row_freq.pow(2) + col_freq.pow(2)).item()


def average_gradient(image):
    image = _as_bchw(image)
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.mean(torch.sqrt((dx.pow(2) + dy.pow(2)) * 0.5 + 1e-8)).item()


def mutual_information(image, reference, bins=64):
    image = _as_bchw(image).detach().flatten().clamp(0.0, 1.0)
    reference = _as_bchw(reference).to(image.device).detach().flatten().clamp(0.0, 1.0)
    if image.numel() == 0 or reference.numel() == 0:
        return 0.0
    x = torch.clamp((image * (bins - 1)).long(), 0, bins - 1)
    y = torch.clamp((reference * (bins - 1)).long(), 0, bins - 1)
    joint = torch.bincount(x * bins + y, minlength=bins * bins).float().view(bins, bins)
    joint = joint / joint.sum().clamp_min(1.0)
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    expected = px @ py
    mask = joint > 0
    value = (joint[mask] * torch.log2(joint[mask] / expected[mask].clamp_min(1e-12))).sum()
    return _finite_float(value)


def entropy(image, bins=256):
    image = _as_bchw(image).detach().flatten().clamp(0.0, 1.0)
    indices = torch.clamp((image * (bins - 1)).long(), 0, bins - 1)
    hist = torch.bincount(indices, minlength=bins).float()
    probs = hist / hist.sum().clamp_min(1.0)
    probs = probs[probs > 0]
    return -(probs * torch.log2(probs)).sum().item()


def correlation_coefficient(image, reference):
    image = _as_bchw(image).detach().flatten()
    reference = _as_bchw(reference).to(image.device).detach().flatten()
    image = image - image.mean()
    reference = reference - reference.mean()
    denom = torch.sqrt((image.pow(2).sum() * reference.pow(2).sum()).clamp_min(1e-12))
    return (image * reference).sum().div(denom).item()


def edge_preservation_index(fused, mri, ct):
    fused = _as_bchw(fused)
    source = torch.maximum(_as_bchw(mri).to(fused.device), _as_bchw(ct).to(fused.device))
    fused_grad = _gradient_magnitude(fused).flatten()
    source_grad = _gradient_magnitude(source).flatten()
    fused_centered = fused_grad - fused_grad.mean()
    source_centered = source_grad - source_grad.mean()
    denom = torch.sqrt((fused_centered.pow(2).sum() * source_centered.pow(2).sum()).clamp_min(1e-12))
    return (fused_centered * source_centered).sum().div(denom).item()


def _finite_float(value, default=0.0):
    value = value.item() if torch.is_tensor(value) else float(value)
    return value if math.isfinite(value) else default


def _feature_normalize(feature, eps=1e-8):
    feature = feature.detach().float()
    flat = feature.flatten(start_dim=1)
    low = torch.quantile(flat, 0.01, dim=1).view(-1, 1, 1, 1)
    high = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1)
    span = high - low
    normalized = (feature - low) / span.clamp_min(eps)
    normalized = normalized.clamp(0.0, 1.0)
    constant = span <= eps
    if constant.any():
        normalized = torch.where(constant, torch.zeros_like(normalized), normalized)
    return normalized


def _normalized_mutual_information(image, reference, bins=64, eps=1e-12):
    image = _as_bchw(image).detach().flatten().clamp(0.0, 1.0)
    reference = _as_bchw(reference).to(image.device).detach().flatten().clamp(0.0, 1.0)
    if image.numel() == 0 or reference.numel() == 0:
        return 0.0

    x = torch.clamp((image * (bins - 1)).long(), 0, bins - 1)
    y = torch.clamp((reference * (bins - 1)).long(), 0, bins - 1)
    joint = torch.bincount(x * bins + y, minlength=bins * bins).float().view(bins, bins)
    joint = joint / joint.sum().clamp_min(1.0)
    px = joint.sum(dim=1)
    py = joint.sum(dim=0)
    expected = px[:, None] * py[None, :]
    mask = joint > 0
    mi = (joint[mask] * torch.log2(joint[mask] / expected[mask].clamp_min(eps))).sum()
    hx = -(px[px > 0] * torch.log2(px[px > 0])).sum()
    hy = -(py[py > 0] * torch.log2(py[py > 0])).sum()
    denom = torch.sqrt((hx * hy).clamp_min(eps))
    if hx <= eps and hy <= eps:
        return 1.0
    if hx <= eps or hy <= eps:
        return 0.0
    return max(0.0, min(1.0, _finite_float(mi / denom)))


def feature_mutual_information(fused, source1, source2):
    fused_grad = _feature_normalize(_gradient_magnitude(_as_bchw(fused)))
    source1_grad = _feature_normalize(_gradient_magnitude(_as_bchw(source1).to(fused_grad.device)))
    source2_grad = _feature_normalize(_gradient_magnitude(_as_bchw(source2).to(fused_grad.device)))
    value = 0.5 * (
        _normalized_mutual_information(fused_grad, source1_grad, bins=64)
        + _normalized_mutual_information(fused_grad, source2_grad, bins=64)
    )
    return value if math.isfinite(value) else 0.0


def artifact_noise_indicator(image):
    image = _as_bchw(image)
    smooth = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
    residual = image - smooth
    return residual.std().item()


def _gradient_magnitude(image):
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)


def multi_scale_ssim(image, reference, levels=3):
    """MS metric: multi-scale SSIM averaged over progressively pooled images."""
    image = _as_bchw(image)
    reference = _as_bchw(reference).to(image.device)

    scores = []
    for level in range(levels):
        scores.append(ssim(image, reference))
        if level < levels - 1:
            image = F.avg_pool2d(image, kernel_size=2, stride=2, ceil_mode=True)
            reference = F.avg_pool2d(reference, kernel_size=2, stride=2, ceil_mode=True)
    return sum(scores) / len(scores)


def evaluate_fusion(fused, mri, ct):
    """Return no-reference and source-reference metrics for one fused image."""
    fused = _as_bchw(fused)
    mri = _as_bchw(mri).to(fused.device)
    ct = _as_bchw(ct).to(fused.device)

    avg_reference = 0.5 * (mri + ct)
    return {
        "PSNR_MRI": psnr(fused, mri),
        "PSNR_CT": psnr(fused, ct),
        "SSIM_MRI": ssim(fused, mri),
        "SSIM_CT": ssim(fused, ct),
        "MI_MRI": mutual_information(fused, mri),
        "MI_CT": mutual_information(fused, ct),
        "EN": entropy(fused),
        "CC_MRI": correlation_coefficient(fused, mri),
        "CC_CT": correlation_coefficient(fused, ct),
        "CC": correlation_coefficient(fused, avg_reference),
        "FMI": feature_mutual_information(fused, mri, ct),
        "SF": spatial_frequency(fused),
        "AG": average_gradient(fused),
        "EPI": edge_preservation_index(fused, mri, ct),
        "NOISE": artifact_noise_indicator(fused),
        "MS_MRI": multi_scale_ssim(fused, mri),
        "MS_CT": multi_scale_ssim(fused, ct),
    }
