import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses import foreground_mask, gradient_loss, high_frequency, masked_l1_loss, sobel_edges


def laplacian_detail(image):
    C = image.shape[1]
    kernel = image.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    return F.conv2d(image, kernel, padding=1, groups=C)



def local_contrast(image, kernel_size=9):
    padding = kernel_size // 2
    mean = F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(image * image, kernel_size=kernel_size, stride=1, padding=padding)
    return torch.sqrt((mean_sq - mean.pow(2)).clamp_min(1e-8))


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using relu2_2 and relu3_3 feature layers.

    Inputs must be 3-channel tensors in [0, 1].  Normalises with ImageNet
    mean/std before passing through VGG.  Falls back to zero loss if
    torchvision is not installed.
    """

    def __init__(self):
        super().__init__()
        self._available = False
        try:
            import torchvision.models as models  # noqa: PLC0415
            try:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except AttributeError:
                # older torchvision (<= 0.12)
                vgg = models.vgg16(pretrained=True)
            feats = list(vgg.features.children())
            # relu2_2 is at index 8 (0-based), relu3_3 is at index 15
            self.relu2_2 = nn.Sequential(*feats[:9])
            self.relu3_3 = nn.Sequential(*feats[:16])
            for param in self.parameters():
                param.requires_grad = False
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
            self._available = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"PerceptualLoss disabled – torchvision unavailable or VGG load failed: {exc}. "
                "Install torchvision to enable perceptual loss.",
                stacklevel=2,
            )

    def forward(self, pred, target):
        """pred, target: (B, 3, H, W) in [0, 1].  Returns scalar loss."""
        if not self._available:
            return pred.new_tensor(0.0)
        # torch.no_grad() prevents PyTorch from storing any activation graphs
        # for backward through VGG — we use .item() to detach anyway, so grads
        # through the perceptual path are never needed.  This cuts VGG VRAM by
        # ~50 % (no saved_tensors for backward), which is critical on 4 GB GPUs.
        #
        # torch.autocast(..., enabled=False) forces float32 — VGG16 in float16
        # overflows and produces NaN under AMP.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            pred_f = pred.float().clamp(0.0, 1.0)
            tgt_f  = target.float().clamp(0.0, 1.0)
            mean   = self.mean.float()
            std    = self.std.float()
            pred_n = (pred_f - mean) / std
            tgt_n  = (tgt_f  - mean) / std

            # Compute each feature block separately and delete intermediates
            # immediately so only one block's activations live in VRAM at a time.
            fp = self.relu2_2(pred_n)
            ft = self.relu2_2(tgt_n)
            loss = F.l1_loss(fp, ft)
            del fp, ft

            fp = self.relu3_3(pred_n)
            ft = self.relu3_3(tgt_n)
            loss = loss + F.l1_loss(fp, ft)
            del fp, ft

            scalar = loss.item() * 0.5  # detach to Python float

        # Return as a scalar in the caller's original dtype.
        return pred.new_tensor(scalar)


class DiffusionFusionLoss(nn.Module):
    def __init__(
        self,
        lambda_l1=1.5,
        lambda_ssim=0,              # disabled – SSIM removed from training loss
        lambda_grad=4.0,
        lambda_hf=2.5,
        lambda_msfd=1.0,
        lambda_ms_ssim=0,           # disabled – MS-SSIM removed from training loss
        lambda_local_contrast=0.6,
        lambda_perceptual=0.5,
    ):
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_grad = float(lambda_grad)
        self.lambda_hf = float(lambda_hf)
        self.lambda_msfd = float(lambda_msfd)
        self.lambda_ms_ssim = float(lambda_ms_ssim)
        self.lambda_local_contrast = float(lambda_local_contrast)
        self.lambda_perceptual = float(lambda_perceptual)
        self.perceptual_loss = PerceptualLoss() if self.lambda_perceptual > 0 else None

    def forward(self, x0_pred, target, source1, source2, msfd_target=None, use_msfd=False):
        # Cast everything to float32 — AMP (float16) can silently overflow in
        # convolution-heavy terms like Laplacian, local contrast, and Sobel edges,
        # producing NaN/Inf that poisons the entire training run.
        x0_pred    = x0_pred.float()
        target     = target.float()
        source1    = source1.float()
        source2    = source2.float()
        if msfd_target is not None:
            msfd_target = msfd_target.float()

        # Collapse accidental multi-channel source inputs to single channel.
        # Guards against augmentation returning RGB tensors instead of grayscale.
        if source1.shape[1] != 1:
            source1 = source1.mean(dim=1, keepdim=True)
        if source2.shape[1] != 1:
            source2 = source2.mean(dim=1, keepdim=True)

        l1 = F.l1_loss(x0_pred, target)
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
        contrast_target = torch.maximum(local_contrast(source1), local_contrast(source2))
        local_contrast_loss = masked_l1_loss(local_contrast(x0_pred), contrast_target, mask)
        msfd = F.l1_loss(x0_pred, msfd_target) if use_msfd and msfd_target is not None else x0_pred.new_tensor(0.0)

        # Perceptual loss: convert single-channel → 3-channel for VGG
        if self.lambda_perceptual > 0:
            perceptual = self.perceptual_loss(
                x0_pred.repeat(1, 3, 1, 1),
                target.repeat(1, 3, 1, 1),
            )
        else:
            perceptual = x0_pred.new_tensor(0.0)

        total = (
            self.lambda_l1 * l1
            + self.lambda_grad * (grad + 0.5 * edge)
            + self.lambda_hf * (hf + 0.5 * lap)
            + self.lambda_local_contrast * local_contrast_loss
            + self.lambda_msfd * msfd
            + self.lambda_perceptual * perceptual
        )
        return {
            "total": total,
            "l1": l1.detach(),
            # "ssim" and "ms_ssim" are disabled (lambda=0) but must be present
            # so that trainer.update_totals() does not raise KeyError.
            "ssim": x0_pred.new_tensor(0.0),
            "gradient": grad.detach(),
            "edge": edge.detach(),
            "hf": hf.detach(),
            "laplacian": lap.detach(),
            "ms_ssim": x0_pred.new_tensor(0.0),
            "local_contrast": local_contrast_loss.detach(),
            "msfd": msfd.detach(),
            "perceptual": perceptual.detach(),
        }
