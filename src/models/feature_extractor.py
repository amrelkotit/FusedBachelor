import torch
import torch.nn as nn


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(1, 16, kernel_size=7, padding=3)

        self.relu = nn.ReLU()

        # Combine features
        self.fusion = nn.Conv2d(16 * 3, 64, kernel_size=1)

    def forward(self, x):
        f3 = self.relu(self.conv3(x))
        f5 = self.relu(self.conv5(x))
        f7 = self.relu(self.conv7(x))

        combined = torch.cat([f3, f5, f7], dim=1)
        out = self.fusion(combined)

        return out