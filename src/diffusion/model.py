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


class _CAF(nn.Module):
    """Cross-Attention Fusion block used in legacy SFF source conditioning."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.norm_ff = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, dim, 3, padding=1)

    _ATTN_SIZE = 32  # max spatial side for attention (32×32 = 1024 tokens)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self._ATTN_SIZE
        xs = F.adaptive_avg_pool2d(x, s)          # [B, C, s, s]
        seq = xs.flatten(2).permute(0, 2, 1)       # [B, s², C]
        q, k = self.norm1(seq), self.norm2(seq)
        out, _ = self.attn(q, k, seq, need_weights=False)
        seq = seq + out
        seq = seq + self.ff(self.norm_ff(seq))
        feat = seq.permute(0, 2, 1).reshape(B, C, s, s)
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        return self.proj(feat)


class _SFF(nn.Module):
    """Spatial-Frequency Fusion used in legacy SFF source conditioning."""

    def __init__(self, dim):
        super().__init__()
        self.spatial_caf = _CAF(dim)
        self.freq_caf = _CAF(dim)
        self.merge_caf = _CAF(dim)

    def forward(self, x):
        s = self.spatial_caf(x)
        f = self.freq_caf(x)
        return self.merge_caf(s + f)


class LegacyConditionalUNet(nn.Module):
    """SFF-based model matching checkpoints trained before the concat refactor.

    Checkpoint signature: base_channels=64, time_dim=256, in_channels=3.
    Source conditioning: source1 → source_lift → source_sff → source_proj (1-ch),
    then concat [noisy, source_proj_out, initial] → in_conv.
    """

    def __init__(self, in_channels=3, base_channels=64, time_dim=256):
        super().__init__()
        c0, c2 = base_channels, base_channels * 2
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.source_lift = nn.Conv2d(1, c0, 3, padding=1)
        self.source_sff = _SFF(c0)
        self.source_proj = nn.Conv2d(c0, 1, 3, padding=1)
        self.in_conv = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.down1 = ResidualBlock(c0, c0, time_dim)
        self.down2 = ResidualBlock(c0, c2, time_dim)
        self.mid   = ResidualBlock(c2, c2, time_dim)
        self.up1   = ResidualBlock(c2 + c0, c0, time_dim)
        self.up2   = ResidualBlock(c0 + c0, c0, time_dim)
        self.out_norm = nn.GroupNorm(_group_count(c0), c0)
        self.out_conv = nn.Conv2d(c0, 1, 3, padding=1)

    def forward(self, noisy, source1, source2, initial, t):
        time_emb = self.time_mlp(t)
        src_cond = self.source_proj(self.source_sff(self.source_lift(source1)))
        x = torch.cat([noisy, src_cond, initial], dim=1)
        x0  = self.in_conv(x)
        d1  = self.down1(x0, time_emb)
        d2  = self.down2(F.avg_pool2d(d1, 2), time_emb)
        mid = self.mid(d2, time_emb)
        u1  = F.interpolate(mid, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1  = self.up1(torch.cat([u1, d1], dim=1), time_emb)
        u2  = self.up2(torch.cat([u1, x0], dim=1), time_emb)
        return self.out_conv(F.silu(self.out_norm(u2)))


class ConditionalUNet(nn.Module):
    """Conditional diffusion U-Net.

    Source conditioning via direct 4-channel concatenation:
      [noisy, source1, source2, initial] → in_channels=4
    """

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
        self.mid   = ResidualBlock(c2, c2, time_dim)
        self.up1   = ResidualBlock(c2 + c1, c1, time_dim)
        self.up2   = ResidualBlock(c1 + c0, c0, time_dim)
        self.out_norm = nn.GroupNorm(_group_count(c0), c0)
        self.out_conv = nn.Conv2d(c0, 1, 3, padding=1)

    def forward(self, noisy, source1, source2, initial, t):
        time_emb = self.time_mlp(t)
        x = torch.cat([noisy, source1, source2, initial], dim=1)  # [B, 4, H, W]

        x0  = self.in_conv(x)
        d1  = self.down1(x0, time_emb)
        d2  = self.down2(F.avg_pool2d(d1, 2), time_emb)
        mid = self.mid(d2, time_emb)
        u1  = F.interpolate(mid, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1  = self.up1(torch.cat([u1, d1], dim=1), time_emb)
        u2  = self.up2(torch.cat([u1, x0], dim=1), time_emb)
        return self.out_conv(F.silu(self.out_norm(u2)))
