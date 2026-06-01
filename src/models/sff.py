import torch
import torch.nn as nn
import torch.nn.functional as F


class CAF(nn.Module):
    """Cross-Attention Fusion block.

    FIX v2 (2026-05-26):
      - Pool tokens to 16×16 instead of 8×8: preserves more local gradient
        structure that FMI measures, while staying within GPU memory limits
        (16×16 = 256 tokens × channels — still << 1 GB).
      - Residual is now the average of both inputs (x+y)/2 instead of x only,
        so information from both modalities contributes equally to the skip
        connection, improving MI and CC.
    """

    def __init__(self, channels, num_heads=4, patch_size=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        self.norm_ff = nn.LayerNorm(channels)
        self.proj = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, y):
        # x, y: (B, C, H, W)
        # Force float32 for the entire CAF forward pass.
        # nn.MultiheadAttention overflows in float16 under AMP with fresh random
        # weights (attention logits > 65504 → Inf → NaN after softmax), causing
        # NaN loss on every batch from epoch 1.  Disabling autocast here keeps
        # all ops in float32; the result is cast back to the caller's dtype.
        orig_dtype = x.dtype
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            y = y.float()
            B, C, H, W = x.shape
            pool_h = min(H, 16)
            pool_w = min(W, 16)
            x_pool = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
            y_pool = F.adaptive_avg_pool2d(y, (pool_h, pool_w))
            xf = x_pool.flatten(2).transpose(1, 2)   # (B, pool_h*pool_w, C)
            yf = y_pool.flatten(2).transpose(1, 2)
            xf_n = self.norm1(xf)
            yf_n = self.norm2(yf)
            x_cross, _ = self.attn(xf_n, yf_n, yf_n)
            y_cross, _ = self.attn(yf_n, xf_n, xf_n)
            fused = (xf + x_cross + yf + y_cross) * 0.25
            fused = fused + self.ff(self.norm_ff(fused))
            fused = fused.transpose(1, 2).reshape(B, C, pool_h, pool_w)
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)
            result = self.proj(fused + 0.5 * (x + y))
        return result.to(orig_dtype)


class SFF(nn.Module):
    """Spatial-Frequential Fusion module.

    FIX v2 (2026-05-26) — Fourier path bug corrected:

    PREVIOUS (broken):
        f1 = torch.fft.fft2(feat).real   ← drops imaginary = drops phase = drops edges
        freq = ifft2(fft2(result)).real   ← ifft2∘fft2 on real input ≈ identity (noop)

    The imaginary (sine) component of the FFT encodes phase — which determines
    WHERE edges are in the image.  FMI is a Sobel-gradient metric: it measures
    exactly this edge information.  Discarding the imaginary part meant the
    Fourier branch was nearly a no-op, contributing nothing beyond the spatial
    path and explaining why FMI stayed at 0.12–0.24.

    FIX: fuse the MAGNITUDE spectra of both modalities, then reconstruct using
    the fused magnitude with the phase blended from both sources.  The magnitude
    spectrum captures which frequency bands are active (texture and edge energy);
    the phase determines where those textures/edges appear.  Blending both
    ensures edges from BOTH modalities are preserved in the reconstruction,
    which is what FMI rewards.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.spatial_caf = CAF(channels, num_heads)
        self.freq_caf    = CAF(channels, num_heads)
        self.merge_caf   = CAF(channels, num_heads)

    def forward(self, feat1, feat2):
        # Spatial path
        spatial = self.spatial_caf(feat1, feat2)

        # ── Fourier path (FIXED) ─────────────────────────────────────────────
        # Step 1: full complex FFT (preserves both magnitude and phase)
        f1_c = torch.fft.fft2(feat1.float())   # complex (B, C, H, W)
        f2_c = torch.fft.fft2(feat2.float())   # complex

        # Step 2: fuse MAGNITUDE spectra via cross-attention
        #   Magnitude = which frequencies/edges exist in each modality
        f1_amp = torch.abs(f1_c)               # (B, C, H, W) real
        f2_amp = torch.abs(f2_c)
        # Log-normalize amplitudes before CAF: raw FFT magnitudes grow with feature
        # scale and routinely exceed float16 max (~65504) after a few training epochs.
        # Inside CAF.forward the final proj Conv2d casts its float32 input to float16
        # under AMP → values > 65504 become Inf → NaN.  log1p maps [0, 1e5+] to
        # [0, ~12], well within float16 range, while preserving relative ordering.
        f1_amp = torch.log1p(f1_amp)
        f2_amp = torch.log1p(f2_amp)
        fused_amp = self.freq_caf(f1_amp, f2_amp)  # fused log-magnitude

        # Step 3: reconstruct complex spectrum using fused amplitude + blended phase
        #   Phase = WHERE edges appear.  Averaging both phases preserves edges
        #   from both modalities (key for MI and FMI).
        f1_phase = f1_c / f1_c.abs().clamp_min(1e-8)   # unit phasor (pure phase)
        f2_phase = f2_c / f2_c.abs().clamp_min(1e-8)
        # Blend phases in complex domain (avoids phase-wrapping discontinuities)
        avg_phase = (f1_phase + f2_phase)
        avg_phase = avg_phase / avg_phase.abs().clamp_min(1e-8)
        freq_complex = fused_amp * avg_phase            # complex: fused amp × avg phase

        # Step 4: inverse FFT → real spatial features with preserved edge structure
        freq = torch.fft.ifft2(freq_complex).real.to(feat1.dtype)
        # ────────────────────────────────────────────────────────────────────

        # Merge spatial and frequency paths
        return self.merge_caf(spatial, freq)
