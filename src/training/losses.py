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


def high_frequency(image):
    """Small Laplacian-style detail map used to discourage over-smoothing."""
    blurred = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
    return image - blurred


def gradient_loss(fused, mri, ct, levels=3):
    """Preserve strongest CT/MRI Sobel edges at multiple scales.

    Strong boundaries receive slightly higher weight, which helps skull edges,
    tissue borders, and small bright structures survive the adversarial stage.
    """
    loss = fused.new_tensor(0.0)
    fused_scale = fused
    mri_scale = mri
    ct_scale = ct

    for level in range(levels):
        fused_edges = sobel_edges(fused_scale)
        target_edges = torch.maximum(sobel_edges(mri_scale), sobel_edges(ct_scale))
        edge_weight = 1.0 + 2.0 * target_edges.detach()
        scale_weight = 1.0 / (2 ** level)
        loss = loss + scale_weight * F.l1_loss(fused_edges * edge_weight, target_edges * edge_weight)

        if level < levels - 1:
            fused_scale = F.avg_pool2d(fused_scale, kernel_size=2, stride=2, ceil_mode=True)
            mri_scale = F.avg_pool2d(mri_scale, kernel_size=2, stride=2, ceil_mode=True)
            ct_scale = F.avg_pool2d(ct_scale, kernel_size=2, stride=2, ceil_mode=True)

    target_detail = torch.maximum(torch.abs(high_frequency(mri)), torch.abs(high_frequency(ct)))
    detail_loss = F.l1_loss(torch.abs(high_frequency(fused)), target_detail)
    return loss + 0.25 * detail_loss


def intensity_loss(fused, mri, ct):
    """Preserve bright structures and soft tissue from either source image."""
    strongest = torch.maximum(mri, ct)
    mean_source = 0.5 * (mri + ct)
    return 0.7 * F.l1_loss(fused, strongest) + 0.3 * F.l1_loss(fused, mean_source)


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
    return 0.5 * ssim_loss(fused, mri) + 0.5 * ssim_loss(fused, ct)


def fusion_loss(fused, mri, ct, intensity_weight=1.0, structural_weight=0.75):
    return intensity_weight * intensity_loss(fused, mri, ct) + structural_weight * structural_loss(fused, mri, ct)


def gan_generator_loss(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def gan_discriminator_loss(real_logits, fake_logits, label_smoothing=0.9):
    """Smoothed real labels reduce discriminator dominance."""
    real_targets = torch.full_like(real_logits, label_smoothing)
    fake_targets = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    return 0.5 * (real_loss + fake_loss)


class MedicalFusionGANLoss(nn.Module):
    """Total G loss = fusion + lambda_gan * GAN + lambda_grad * edge loss.

    Defaults are intentionally conservative for medical fusion:
    lambda_gan=0.005 lowers adversarial pressure after discriminator dominance,
    lambda_grad=7.5 gives stronger boundary/detail preservation.
    """

    def __init__(self, lambda_gan=0.005, lambda_grad=7.5, lambda_fusion=1.0):
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_grad = lambda_grad
        self.lambda_fusion = lambda_fusion

    def forward(self, fused, mri, ct, d1_fake_logits, d2_fake_logits):
        fusion = fusion_loss(fused, mri, ct)
        grad = gradient_loss(fused, mri, ct)
        gan = gan_generator_loss(d1_fake_logits) + gan_generator_loss(d2_fake_logits)
        total = self.lambda_fusion * fusion + self.lambda_gan * gan + self.lambda_grad * grad
        return {
            "total": total,
            "fusion": fusion.detach(),
            "gradient": grad.detach(),
            "gan": gan.detach(),
        }


def total_generator_loss(fused, mri, ct, fake_logits=None, fusion_weight=1.0, gan_weight=0.005):
    base_loss = fusion_loss(fused, mri, ct) + gradient_loss(fused, mri, ct)
    if fake_logits is None:
        gan_loss = fused.new_tensor(0.0)
    else:
        gan_loss = gan_generator_loss(fake_logits)
    return fusion_weight * base_loss + gan_weight * gan_loss
