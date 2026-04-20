import numpy as np
import torch
import torch.nn as nn

from src.fusion.decomposition import multiscale_fuse
from src.models.feature_extractor import MultiScaleFeatureExtractor


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = MultiScaleFeatureExtractor()

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, mri, ct):
        feat_mri = self.feature_extractor(mri)
        feat_ct = self.feature_extractor(ct)

        # Feature-level max fusion preserves the strongest modality response.
        fused = torch.max(feat_mri, feat_ct)
        return self.decoder(fused)


def fuse_images(ct, mri, levels=3):
    """Fuse one CT/MRI pair with the practical multi-scale decomposition rule.

    This helper keeps the original public API used by main.py, but moves the
    actual fusion logic to src.fusion.decomposition.
    """
    with torch.no_grad():
        fused = multiscale_fuse(mri=mri, ct=ct, levels=levels)

    fused_np = fused.squeeze().detach().cpu().numpy()
    fused_np = np.clip(fused_np, 0.0, 1.0)
    return (fused_np * 255.0).astype(np.uint8)
