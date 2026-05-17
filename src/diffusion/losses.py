import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses import foreground_mask, gradient_loss, high_frequency, masked_l1_loss, ssim_loss, sobel_edges


def laplacian_detail(image):
    kernel = image.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    return F.conv2d(image, kernel, padding=1)


def multi_scale_ssim_loss(image, reference, levels=3):
    loss = image.new_tensor(0.0)
    image_scale = image
    reference_scale = reference
    total_weight = 0.0
    for level in range(levels):
        weight = 1.0 / (2 ** level)
        loss = loss + weight * ssim_loss(image_scale, reference_scale)
        total_weight += weight
        if level < levels - 1:
            image_scale = F.avg_pool2d(image_scale, kernel_size=2, stride=2, ceil_mode=True)
            reference_scale = F.avg_pool2d(reference_scale, kernel_size=2, stride=2, ceil_mode=True)
    return loss / max(total_weight, 1e-8)


def local_contrast(image, kernel_size=9):
    padding = kernel_size // 2
    mean = F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(image * image, kernel_size=kernel_size, stride=1, padding=padding)
    return torch.sqrt((mean_sq - mean.pow(2)).clamp_min(1e-8))


class DiffusionFusionLoss(nn.Module):
    def __init__(
        self,
        lambda_l1=1.2,
        lambda_ssim=2.5,
        lambda_grad=1.6,
        lambda_hf=1.0,
        lambda_msfd=1.0,
        lambda_ms_ssim=0.8,
        lambda_local_contrast=0.4,
    ):
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_grad = float(lambda_grad)
        self.lambda_hf = float(lambda_hf)
        self.lambda_msfd = float(lambda_msfd)
        self.lambda_ms_ssim = float(lambda_ms_ssim)
        self.lambda_local_contrast = float(lambda_local_contrast)

    def forward(self, x0_pred, target, source1, source2, msfd_target=None, use_msfd=False):
        l1 = F.l1_loss(x0_pred, target)
        ssim_target = ssim_loss(x0_pred, target)
        ssim_mri = ssim_loss(x0_pred, source2)
        ssim = 0.5 * (ssim_target + ssim_mri)
        mask = foreground_mask(source2, source1)
        grad = gradient_loss(x0_pred, source2, source1, levels=2)
        edge = F.l1_loss(sobel_edges(x0_pred), torch.maximum(sobel_edges(source1), sobel_edges(source2)))
        hf_target = torch.where(
            torch.abs(high_frequency(source1)) >= torch.abs(high_frequency(source2)),
            high_frequency(source1),
            high_frequency(source2),
        )
        hf = masked_l1_loss(high_frequency(x0_pred), hf_target, mask)
        lap_target = torch.where(
            torch.abs(laplacian_detail(source1)) >= torch.abs(laplacian_detail(source2)),
            laplacian_detail(source1),
            laplacian_detail(source2),
        )
        lap = masked_l1_loss(laplacian_detail(x0_pred), lap_target, mask)
        ms_ssim = 0.5 * (multi_scale_ssim_loss(x0_pred * mask, target * mask) + multi_scale_ssim_loss(x0_pred * mask, source2 * mask))
        contrast_target = torch.maximum(local_contrast(source1), local_contrast(source2))
        local_contrast_loss = masked_l1_loss(local_contrast(x0_pred), contrast_target, mask)
        msfd = F.l1_loss(x0_pred, msfd_target) if use_msfd and msfd_target is not None else x0_pred.new_tensor(0.0)
        total = (
            self.lambda_l1 * l1
            + self.lambda_ssim * ssim
            + self.lambda_grad * (grad + 0.5 * edge)
            + self.lambda_hf * (hf + 0.5 * lap)
            + self.lambda_ms_ssim * ms_ssim
            + self.lambda_local_contrast * local_contrast_loss
            + self.lambda_msfd * msfd
        )
        return {
            "total": total,
            "l1": l1.detach(),
            "ssim": ssim.detach(),
            "gradient": grad.detach(),
            "edge": edge.detach(),
            "hf": hf.detach(),
            "laplacian": lap.detach(),
            "ms_ssim": ms_ssim.detach(),
            "local_contrast": local_contrast_loss.detach(),
            "msfd": msfd.detach(),
        }
