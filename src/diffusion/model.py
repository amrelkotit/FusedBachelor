import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb)).view(time_emb.shape[0], -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, time_dim=128):
        super().__init__()
        c0 = base_channels
        c1 = base_channels
        c2 = base_channels * 2
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.down1 = ResidualBlock(c0, c1, time_dim)
        self.down2 = ResidualBlock(c1, c2, time_dim)
        self.mid = ResidualBlock(c2, c2, time_dim)

        up1_in_channels = c2 + c1
        up2_in_channels = c1 + c0
        self.up1 = ResidualBlock(up1_in_channels, c1, time_dim)
        self.up2 = ResidualBlock(up2_in_channels, c0, time_dim)
        self.out_norm = nn.GroupNorm(_group_count(c0), c0)
        self.out_conv = nn.Conv2d(c0, 1, 3, padding=1)

    def forward(self, noisy, source1, source2, initial, t):
        time_emb = self.time_mlp(t)
        x = torch.cat([noisy, source1, source2, initial], dim=1)  # [B, 4, H, W]
        x0 = self.in_conv(x)  # [B, base, H, W]
        d1 = self.down1(x0, time_emb)  # [B, base, H, W]
        d2 = self.down2(F.avg_pool2d(d1, 2), time_emb)  # [B, base*2, H/2, W/2]
        mid = self.mid(d2, time_emb)  # [B, base*2, H/2, W/2]
        u1 = F.interpolate(mid, size=d1.shape[-2:], mode="bilinear", align_corners=False)  # [B, base*2, H, W]
        u1 = self.up1(torch.cat([u1, d1], dim=1), time_emb)  # [B, base*2 + base, H, W] -> [B, base, H, W]
        u2 = self.up2(torch.cat([u1, x0], dim=1), time_emb)  # [B, base + base, H, W] -> [B, base, H, W]
        return self.out_conv(F.silu(self.out_norm(u2)))
