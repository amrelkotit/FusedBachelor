import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_guidance(x):
    """Fixed edge map, no trainable params, so old checkpoints still load."""
    gray = x.mean(dim=1, keepdim=True)
    dx = F.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 0, 1))
    edge = dx + dy
    edge = edge / (edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
    return edge


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
    """U-Net style generator: input [CT, MRI], output one fused grayscale image."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.down1 = DownBlock(2, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = MultiScaleConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, ct, mri):
        x = torch.cat([ct, mri], dim=1)
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.out(x)

    def forward_with_debug(self, ct, mri):
        """Return final output plus values before/after final activation."""
        x = torch.cat([ct, mri], dim=1)
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = self.out[0](x)
        x = self.out[1](x)
        pre_activation = self.out[2](x)
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
