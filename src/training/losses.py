import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_edges(image):
    """Sobel edge magnitude for grayscale tensors shaped (B, 1, H, W)."""
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = image.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)


def gaussian_kernel(image, kernel_size=5, sigma=1.0):
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype)
    coords = coords - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(image.shape[1], 1, 1, 1)


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    padding = kernel_size // 2
    kernel = gaussian_kernel(image, kernel_size=kernel_size, sigma=sigma)
    padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[1])


def high_frequency(image):
    """Gaussian high-pass residual used only as a supervision signal."""
    return image - gaussian_blur(image)


def foreground_mask(mri, ct, threshold=0.03, dilation=7):
    mask = ((mri > threshold) | (ct > threshold)).float()
    if dilation > 1:
        padding = dilation // 2
        mask = F.max_pool2d(mask, kernel_size=dilation, stride=1, padding=padding)
    return mask.clamp(0.0, 1.0)


def masked_l1_loss(image, target, mask):
    mask = mask.to(device=image.device, dtype=image.dtype)
    denom = mask.sum().clamp_min(1.0)
    return torch.abs(image - target).mul(mask).sum() / denom


def gradient_loss(fused, mri, ct, levels=1):
    """Preserve strongest CT/MRI Sobel gradients without adding edge maps to output."""
    loss = fused.new_tensor(0.0)
    fused_scale = fused
    mri_scale = mri
    ct_scale = ct
    mask_scale = foreground_mask(mri, ct)

    for level in range(levels):
        fused_edges = sobel_edges(fused_scale)
        target_edges = torch.maximum(sobel_edges(mri_scale), sobel_edges(ct_scale))
        scale_weight = 1.0 / (2 ** level)
        loss = loss + scale_weight * masked_l1_loss(fused_edges, target_edges, mask_scale)

        if level < levels - 1:
            fused_scale = F.avg_pool2d(fused_scale, kernel_size=2, stride=2, ceil_mode=True)
            mri_scale = F.avg_pool2d(mri_scale, kernel_size=2, stride=2, ceil_mode=True)
            ct_scale = F.avg_pool2d(ct_scale, kernel_size=2, stride=2, ceil_mode=True)
            mask_scale = F.avg_pool2d(mask_scale, kernel_size=2, stride=2, ceil_mode=True)

    return loss


def intensity_loss(fused, mri, ct):
    """Preserve bright structures and soft tissue from either source image."""
    target = torch.maximum(mri, ct)
    return masked_l1_loss(fused, target, foreground_mask(mri, ct))


def ssim_loss(image, reference, data_range=1.0, window_size=7):
    channels = image.shape[1]
    window = image.new_ones((channels, 1, window_size, window_size)) / (window_size ** 2)
    padding = window_size // 2

    mu_x = F.conv2d(image, window, padding=padding, groups=channels)
    mu_y = F.conv2d(reference, window, padding=padding, groups=channels)
    sigma_x = F.conv2d(image * image, window, padding=padding, groups=channels) - mu_x.pow(2)
    sigma_y = F.conv2d(reference * reference, window, padding=padding, groups=channels) - mu_y.pow(2)
    sigma_xy = F.conv2d(image * reference, window, padding=padding, groups=channels) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim.mean()


def structural_loss(fused, mri, ct):
    """Keep structure close to both modalities."""
    mask = foreground_mask(mri, ct)
    return ssim_loss(fused * mask, mri * mask) + ssim_loss(fused * mask, ct * mask)


def texture_loss(fused, mri, ct):
    detail_ct = high_frequency(ct)
    detail_mri = high_frequency(mri)
    detail_fused = high_frequency(fused)
    target_detail = torch.where(torch.abs(detail_ct) >= torch.abs(detail_mri), detail_ct, detail_mri)
    return masked_l1_loss(detail_fused, target_detail, foreground_mask(mri, ct))


def fusion_loss(fused, mri, ct):
    return intensity_loss(fused, mri, ct) + gradient_loss(fused, mri, ct) + structural_loss(fused, mri, ct) + texture_loss(fused, mri, ct)


def gan_generator_loss(fake_logits):
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


def gan_discriminator_loss(real_logits, fake_logits, label_smoothing=0.9):
    """LSGAN with smoothed real labels reduces discriminator dominance."""
    real_targets = torch.full_like(real_logits, label_smoothing)
    fake_targets = torch.zeros_like(fake_logits)
    real_loss = F.mse_loss(real_logits, real_targets)
    fake_loss = F.mse_loss(fake_logits, fake_targets)
    return 0.5 * (real_loss + fake_loss)


class MedicalFusionGANLoss(nn.Module):
    """Weighted medical fusion objective with edge/detail terms used only as losses."""

    def __init__(
        self,
        lambda_intensity=1.0,
        lambda_gradient=5.0,
        lambda_ssim=2.0,
        lambda_texture=3.0,
        lambda_gan=0.1,
    ):
        super().__init__()
        self.lambda_intensity = lambda_intensity
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim
        self.lambda_texture = lambda_texture
        self.lambda_gan = lambda_gan

    def forward(self, fused, mri, ct, d1_fake_logits, d2_fake_logits):
        intensity = intensity_loss(fused, mri, ct)
        grad = gradient_loss(fused, mri, ct)
        ssim = structural_loss(fused, mri, ct)
        texture = texture_loss(fused, mri, ct)
        gan = gan_generator_loss(d1_fake_logits) + gan_generator_loss(d2_fake_logits)
        total = (
            self.lambda_intensity * intensity
            + self.lambda_gradient * grad
            + self.lambda_ssim * ssim
            + self.lambda_texture * texture
            + self.lambda_gan * gan
        )
        return {
            "total": total,
            "fusion": (intensity + grad + ssim + texture).detach(),
            "intensity": intensity.detach(),
            "gradient": grad.detach(),
            "ssim": ssim.detach(),
            "texture": texture.detach(),
            "gan": gan.detach(),
        }


def total_generator_loss(fused, mri, ct, fake_logits=None, fusion_weight=1.0, gan_weight=0.1):
    base_loss = fusion_loss(fused, mri, ct)
    if fake_logits is None:
        gan_loss = fused.new_tensor(0.0)
    else:
        gan_loss = gan_generator_loss(fake_logits)
    return fusion_weight * base_loss + gan_weight * gan_loss
