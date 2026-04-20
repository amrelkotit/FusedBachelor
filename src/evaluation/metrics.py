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

    return {
        "PSNR_MRI": psnr(fused, mri),
        "PSNR_CT": psnr(fused, ct),
        "SSIM_MRI": ssim(fused, mri),
        "SSIM_CT": ssim(fused, ct),
        "SF": spatial_frequency(fused),
        "MS_MRI": multi_scale_ssim(fused, mri),
        "MS_CT": multi_scale_ssim(fused, ct),
    }
