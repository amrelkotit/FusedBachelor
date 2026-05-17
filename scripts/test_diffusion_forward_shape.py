import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.model import ConditionalUNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalUNet().to(device)
    batch_size = 2
    noisy = torch.randn(batch_size, 1, 256, 256, device=device)
    source1 = torch.randn(batch_size, 1, 256, 256, device=device)
    source2 = torch.randn(batch_size, 1, 256, 256, device=device)
    initial = torch.randn(batch_size, 1, 256, 256, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    out = model(noisy, source1, source2, initial, t)
    print(out.shape)
    assert out.shape == noisy.shape


if __name__ == "__main__":
    main()
