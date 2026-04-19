import torch
import torch.nn as nn
import cv2
import numpy as np
from src.models.feature_extractor import MultiScaleFeatureExtractor
from src.preprocessing.preprocessing import enhance_image


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = MultiScaleFeatureExtractor()

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, mri, ct):
        feat_mri = self.feature_extractor(mri)
        feat_ct  = self.feature_extractor(ct)

        # 🔥 Max Fusion (best before training)
        fused = torch.max(feat_mri, feat_ct)

        output = self.decoder(fused)

        return output
    
def fuse_images(ct, mri):

    # 🔥 Convert Tensor → NumPy
    if hasattr(ct, "numpy"):
        ct = ct.squeeze().cpu().numpy()

    if hasattr(mri, "numpy"):
        mri = mri.squeeze().cpu().numpy()

    # Continue normally
    ct = enhance_image(ct)
    mri = enhance_image(mri)

    # Low-frequency
    low_ct = cv2.GaussianBlur(ct, (5,5), 0)
    low_mri = cv2.GaussianBlur(mri, (5,5), 0)

    # High-frequency
    high_ct = ct - low_ct
    high_mri = mri - low_mri

    # Fusion
    low_fused = 0.5 * low_ct + 0.5 * low_mri
    high_fused = 0.8 * high_ct + 0.8 * high_mri

    fused = low_fused + high_fused

    # Smooth
    fused = cv2.GaussianBlur(fused, (3,3), 0)

    # Normalize
    fused = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX)
    fused = fused.astype(np.uint8)

    return fused