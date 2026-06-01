import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sff import SFF


def edge_guidance(x):
    """Fixed edge map, no trainable params, so old checkpoints still load."""
    gray = x.mean(dim=1, keepdim=True)
    dx = F.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 0, 1))
    edge = dx + dy
    edge = edge / (edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
    return edge


def source_edge_attention(ct, mri):
    return torch.maximum(edge_guidance(ct), edge_guidance(mri))


def apply_source_attention(features, attention, strength=0.15):
    if attention.shape[-2:] != features.shape[-2:]:
        attention = F.interpolate(attention, size=features.shape[-2:], mode="bilinear", align_corners=False)
    return features * (1.0 + strength * attention)


class MultiScaleConvBlock(nn.Module):
    """Parallel receptive fields help preserve fine and wider anatomical detail."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 3
        remaining = out_channels - 2 * branch_channels
        self.conv3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, remaining, kernel_size=7, padding=3)
        self.mix = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        # A small fixed edge gate makes the block more boundary-aware while
        # preserving checkpoint compatibility because it has no parameters.
        return self.mix(x) * (1.0 + 0.1 * edge_guidance(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = MultiScaleConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        skip = self.block(x)
        down = F.leaky_relu(self.down(skip), 0.2, inplace=True)
        return skip, down


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = MultiScaleConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class FusionGenerator(nn.Module):
    """U-Net generator with dual-encoder + SFF.

    CT and MRI are encoded through independent parallel encoder paths.
    At every scale — skip1 (full-res, 32ch), skip2 (½-res, 64ch),
    skip3 (¼-res, 128ch), and the bottleneck input (⅛-res, 128ch) — a
    dedicated SFF block cross-attends the two modality feature maps in both
    the spatial domain and the Fourier domain before handing the fused
    representation to the decoder.

    ⚠️  Breaking change vs. the previous single-encoder architecture:
    existing checkpoints trained with the old cat([ct, mri]) design are
    NOT compatible with this new layout.
    """

    def __init__(self, base_channels=32):
        super().__init__()
        # --- Dual encoder paths (1-channel input each) ---
        self.ct_down1  = DownBlock(1, base_channels)
        self.ct_down2  = DownBlock(base_channels,     base_channels * 2)
        self.ct_down3  = DownBlock(base_channels * 2, base_channels * 4)

        self.mri_down1 = DownBlock(1, base_channels)
        self.mri_down2 = DownBlock(base_channels,     base_channels * 2)
        self.mri_down3 = DownBlock(base_channels * 2, base_channels * 4)

        # --- SFF at every skip level + bottleneck input ---
        self.sff1    = SFF(base_channels)          # 32-ch  full-res skips
        self.sff2    = SFF(base_channels * 2)      # 64-ch  ½-res   skips
        self.sff3    = SFF(base_channels * 4)      # 128-ch ¼-res   skips
        self.sff_bot = SFF(base_channels * 4)      # 128-ch ⅛-res   bottleneck input

        # --- Bottleneck + decoder (channel counts unchanged) ---
        self.bottleneck = MultiScaleConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels,     base_channels)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _encode(self, ct, mri, attention):
        """Run dual encoder and fuse every scale with SFF.

        Returns (skip1, skip2, skip3, x) where all tensors are already
        cross-modality fused and ready for the decoder.
        """
        # Scale 1 — full resolution
        ct_s1,  ct_x  = self.ct_down1(ct)
        mri_s1, mri_x = self.mri_down1(mri)
        ct_s1  = apply_source_attention(ct_s1,  attention)
        mri_s1 = apply_source_attention(mri_s1, attention)
        ct_x   = apply_source_attention(ct_x,   attention)
        mri_x  = apply_source_attention(mri_x,  attention)

        # Scale 2 — ½ resolution
        ct_s2,  ct_x  = self.ct_down2(ct_x)
        mri_s2, mri_x = self.mri_down2(mri_x)
        ct_s2  = apply_source_attention(ct_s2,  attention)
        mri_s2 = apply_source_attention(mri_s2, attention)
        ct_x   = apply_source_attention(ct_x,   attention)
        mri_x  = apply_source_attention(mri_x,  attention)

        # Scale 3 — ¼ resolution
        ct_s3,  ct_x  = self.ct_down3(ct_x)
        mri_s3, mri_x = self.mri_down3(mri_x)
        ct_s3  = apply_source_attention(ct_s3,  attention)
        mri_s3 = apply_source_attention(mri_s3, attention)
        ct_x   = apply_source_attention(ct_x,   attention)
        mri_x  = apply_source_attention(mri_x,  attention)

        # SFF fusion at every scale
        skip1 = self.sff1(ct_s1, mri_s1)
        skip2 = self.sff2(ct_s2, mri_s2)
        skip3 = self.sff3(ct_s3, mri_s3)
        x     = self.sff_bot(ct_x, mri_x)        # feeds into bottleneck
        return skip1, skip2, skip3, x

    def forward(self, ct, mri):
        attention = source_edge_attention(ct, mri)
        skip1, skip2, skip3, x = self._encode(ct, mri, attention)
        x = self.bottleneck(x)
        x = apply_source_attention(x, attention)
        x = self.up3(x, skip3)
        x = apply_source_attention(x, attention)
        x = self.up2(x, skip2)
        x = apply_source_attention(x, attention)
        x = self.up1(x, skip1)
        x = apply_source_attention(x, attention)
        return self.out(x)

    def forward_with_debug(self, ct, mri):
        """Return (final_output, pre_activation, post_activation)."""
        attention = source_edge_attention(ct, mri)
        skip1, skip2, skip3, x = self._encode(ct, mri, attention)
        x = self.bottleneck(x)
        x = apply_source_attention(x, attention)
        x = self.up3(x, skip3)
        x = apply_source_attention(x, attention)
        x = self.up2(x, skip2)
        x = apply_source_attention(x, attention)
        x = self.up1(x, skip1)
        x = apply_source_attention(x, attention)
        x = self.out[0](x)
        x = self.out[1](x)
        pre_activation  = self.out[2](x)
        post_activation = self.out[3](pre_activation)
        return post_activation, pre_activation, post_activation


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for local medical texture and edge realism."""

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image):
        return self.net(image)
