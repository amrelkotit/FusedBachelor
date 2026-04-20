from pathlib import Path
import re

import cv2
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _list_images(folder):
    folder = Path(folder)
    return sorted(
        (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )


def _read_grayscale(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)
    image = image.astype("float32")
    min_value = image.min()
    max_value = image.max()
    image = (image - min_value) / (max_value - min_value + 1e-8)
    return torch.from_numpy(image).unsqueeze(0)


class PairedMedicalImageDataset(Dataset):
    """Loads paired grayscale CT/MRI images from sibling CT and MRI folders."""

    def __init__(self, root_dir, image_size=256, max_items=None):
        self.root_dir = Path(root_dir)
        self.ct_dir = self.root_dir / "CT"
        self.mri_dir = self.root_dir / "MRI"
        self.image_size = (image_size, image_size)

        if not self.ct_dir.exists() or not self.mri_dir.exists():
            raise FileNotFoundError(f"Expected CT and MRI folders under: {self.root_dir}")

        ct_paths = _list_images(self.ct_dir)
        mri_paths = _list_images(self.mri_dir)
        pair_count = min(len(ct_paths), len(mri_paths))
        if max_items is not None:
            pair_count = min(pair_count, max_items)

        self.pairs = list(zip(ct_paths[:pair_count], mri_paths[:pair_count]))
        if not self.pairs:
            raise ValueError(f"No paired images found under: {self.root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ct_path, mri_path = self.pairs[index]
        ct = _read_grayscale(ct_path, self.image_size)
        mri = _read_grayscale(mri_path, self.image_size)
        return {
            "ct": ct,
            "mri": mri,
            "ct_path": str(ct_path),
            "mri_path": str(mri_path),
        }
