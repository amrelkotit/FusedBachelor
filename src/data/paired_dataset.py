from pathlib import Path
import re

import cv2
import torch
from torch.utils.data import ConcatDataset, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


FINAL_DATASET_ROOT = Path(r"E:\El Gam3a\My bachelor\Fused bachelor\data\raw\final_dataset")
AANLIB_TRAIN = FINAL_DATASET_ROOT / "AANLIB" / "train"
AANLIB_TEST = FINAL_DATASET_ROOT / "AANLIB" / "test"
BRATS_TRAIN = FINAL_DATASET_ROOT / "BRATS" / "train"
BRATS_TEST = FINAL_DATASET_ROOT / "BRATS" / "test"

DEFAULT_TRAIN_ROOTS = [AANLIB_TRAIN, BRATS_TRAIN]
DEFAULT_TEST_ROOTS = [AANLIB_TEST, BRATS_TEST]


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


def modality_dirs(root_dir):
    """Final structure uses lowercase ct/mri folders."""
    root_dir = Path(root_dir)
    return root_dir / "ct", root_dir / "mri"


def verify_dataset_root(root_dir, dataset_name=None, strict=True):
    root_dir = Path(root_dir)
    ct_dir, mri_dir = modality_dirs(root_dir)
    label = dataset_name or root_dir.name

    missing = [str(path) for path in (ct_dir, mri_dir) if not path.exists()]
    if missing:
        message = f"[Dataset] {label}: missing folder(s): {missing}"
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return {"name": label, "root": root_dir, "pairs": 0, "missing": missing, "unmatched_ct": 0, "unmatched_mri": 0}

    ct_paths = _list_images(ct_dir)
    mri_paths = _list_images(mri_dir)
    ct_by_name = {path.name.lower(): path for path in ct_paths}
    mri_by_name = {path.name.lower(): path for path in mri_paths}
    matched_names = sorted(set(ct_by_name) & set(mri_by_name), key=lambda name: _natural_key(Path(name)))
    unmatched_ct = sorted(set(ct_by_name) - set(mri_by_name))
    unmatched_mri = sorted(set(mri_by_name) - set(ct_by_name))

    print(
        f"[Dataset] {label}: {len(matched_names)} matched pairs | "
        f"{len(unmatched_ct)} unmatched CT | {len(unmatched_mri)} unmatched MRI"
    )
    return {
        "name": label,
        "root": root_dir,
        "pairs": len(matched_names),
        "missing": [],
        "unmatched_ct": len(unmatched_ct),
        "unmatched_mri": len(unmatched_mri),
    }


class PairedMedicalImageDataset(Dataset):
    """Loads paired grayscale CT/MRI images from final lowercase ct/mri folders."""

    def __init__(self, root_dir, image_size=256, max_items=None, dataset_name=None, strict=True):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name or self.root_dir.parent.name
        self.ct_dir, self.mri_dir = modality_dirs(self.root_dir)
        self.image_size = (image_size, image_size)

        verify_dataset_root(self.root_dir, self.dataset_name, strict=strict)
        if not self.ct_dir.exists() or not self.mri_dir.exists():
            self.pairs = []
            return

        ct_paths = _list_images(self.ct_dir)
        mri_paths = _list_images(self.mri_dir)
        ct_by_name = {path.name.lower(): path for path in ct_paths}
        mri_by_name = {path.name.lower(): path for path in mri_paths}
        matched_names = sorted(set(ct_by_name) & set(mri_by_name), key=lambda name: _natural_key(Path(name)))

        if max_items is not None:
            matched_names = matched_names[:max_items]

        self.pairs = [(ct_by_name[name], mri_by_name[name]) for name in matched_names]
        if not self.pairs and strict:
            raise ValueError(f"No matched CT/MRI filename pairs found under: {self.root_dir}")

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
            "dataset": self.dataset_name,
        }


class CombinedMedicalFusionDataset(ConcatDataset):
    """Concatenates AANLIB and BRATS_SPLIT while keeping source datasets separate."""

    def __init__(self, roots, image_size=256, max_items=None, split_name="train", strict=True):
        self.roots = [Path(root) for root in roots]
        datasets = []
        for root in self.roots:
            name = "BRATS" if "BRATS_SPLIT" in str(root) else root.parent.name
            dataset_name = f"{name} {split_name}"
            dataset = PairedMedicalImageDataset(
                root,
                image_size=image_size,
                max_items=max_items,
                dataset_name=dataset_name,
                strict=strict,
            )
            if len(dataset) > 0:
                datasets.append(dataset)

        if not datasets and strict:
            raise ValueError(f"No valid datasets found for roots: {self.roots}")

        super().__init__(datasets)
