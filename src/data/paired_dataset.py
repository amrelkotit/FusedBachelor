from pathlib import Path
import re

import cv2
import torch
from torch.utils.data import ConcatDataset, Dataset, random_split


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AANLIB_ROOT = PROJECT_ROOT / "data" / "raw" / "AANLIB"
BRATS_ROOT = PROJECT_ROOT / "data" / "raw" / "final_dataset" / "BRATS"
GAN_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "gan"
DIFFUSION_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "diffusion"

PAIR_CONFIGS = {
    "ct_mri": {"folder": "CT-MRI", "source1": "CT", "source2": "MRI"},
    "pet_mri": {"folder": "PET-MRI", "source1": "PET", "source2": "MRI"},
    "spect_mri": {"folder": "SPECT-MRI", "source1": "SPECT", "source2": "MRI"},
}

DEFAULT_TRAIN_ROOTS = [AANLIB_ROOT]
DEFAULT_TEST_ROOTS = [AANLIB_ROOT]


def normalize_pair(pair):
    key = str(pair or "ct_mri").strip().lower().replace("-", "_")
    if key not in PAIR_CONFIGS:
        supported = ", ".join(sorted(PAIR_CONFIGS))
        raise ValueError(f"Invalid modality pair '{pair}'. Supported pairs: {supported}")
    return key


def pair_config(pair):
    return PAIR_CONFIGS[normalize_pair(pair)]


def pair_labels(pair):
    config = pair_config(pair)
    return config["source1"], config["source2"], "Fused"


def aanlib_split_root(dataset_root, pair, split):
    split = str(split).strip().lower()
    if split not in {"train", "val", "valid", "validation", "test"}:
        raise ValueError(f"Invalid split '{split}'. Use train, val, or test.")
    split = "val" if split in {"valid", "validation"} else split
    return Path(dataset_root) / pair_config(pair)["folder"] / split


def gan_experiment_name(pair, dataset_name="aanlib"):
    return f"{dataset_name}_{normalize_pair(pair)}"


def gan_checkpoint_dir(pair, dataset_name="aanlib"):
    return GAN_OUTPUT_ROOT / gan_experiment_name(pair, dataset_name) / "checkpoints"


def gan_image_dir(dataset_name, split, pair):
    return GAN_OUTPUT_ROOT / "images" / dataset_name / split / normalize_pair(pair)


def gan_graph_dir():
    return GAN_OUTPUT_ROOT / "graphs"


def gan_metrics_dir():
    return GAN_OUTPUT_ROOT / "metrics"


def gan_logs_dir():
    return GAN_OUTPUT_ROOT / "logs"


def gan_thesis_figures_dir():
    return GAN_OUTPUT_ROOT / "thesis_figures"


def diffusion_experiment_name(pair, dataset_name="aanlib"):
    return f"{dataset_name}_{normalize_pair(pair)}"


def diffusion_pair_dir(pair, dataset_name="aanlib", output_root=None):
    root = Path(output_root) if output_root else DIFFUSION_OUTPUT_ROOT
    return root / diffusion_experiment_name(pair, dataset_name)


def diffusion_checkpoint_dir(pair, dataset_name="aanlib", output_root=None):
    return diffusion_pair_dir(pair, dataset_name, output_root=output_root) / "checkpoints"


def diffusion_sample_dir(pair, dataset_name="aanlib", output_root=None):
    return diffusion_pair_dir(pair, dataset_name, output_root=output_root) / "samples"


def diffusion_pair_logs_dir(pair, dataset_name="aanlib", output_root=None):
    return diffusion_pair_dir(pair, dataset_name, output_root=output_root) / "logs"


def diffusion_image_dir(dataset_name, pair, split, output_root=None):
    root = Path(output_root) if output_root else DIFFUSION_OUTPUT_ROOT
    return root / "images" / dataset_name / normalize_pair(pair) / split


def diffusion_after_msfd_dir(dataset_name, pair, output_root=None):
    root = Path(output_root) if output_root else DIFFUSION_OUTPUT_ROOT
    return root / "images_after_msfd" / dataset_name / normalize_pair(pair)


def diffusion_graph_dir(output_root=None):
    root = Path(output_root) if output_root else DIFFUSION_OUTPUT_ROOT
    return root / "graphs"


def diffusion_metrics_dir(output_root=None):
    root = Path(output_root) if output_root else DIFFUSION_OUTPUT_ROOT
    return root / "metrics"


def diffusion_logs_dir(output_root=None):
    root = Path(output_root) if output_root else DIFFUSION_OUTPUT_ROOT
    return root / "logs"


def diffusion_final_assets_dir(output_root=None):
    root = Path(output_root) if output_root else DIFFUSION_OUTPUT_ROOT
    return root / "final_thesis_assets"


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).name)]


def _list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )


def _read_grayscale(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    interpolation = cv2.INTER_AREA if image.shape[0] > size[1] or image.shape[1] > size[0] else cv2.INTER_CUBIC
    image = cv2.resize(image, size, interpolation=interpolation)
    image = image.astype("float32")
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return torch.from_numpy(image).unsqueeze(0)


def find_case_insensitive_dir(root_dir, names):
    root_dir = Path(root_dir)
    wanted = {name.lower() for name in names}
    for child in root_dir.iterdir() if root_dir.exists() else []:
        if child.is_dir() and child.name.lower() in wanted:
            return child
    expected = ", ".join(str(root_dir / name) for name in names)
    raise FileNotFoundError(f"Dataset folder missing. Expected one of: {expected}")


def modality_dirs(root_dir, pair="ct_mri"):
    config = pair_config(pair)
    root_dir = Path(root_dir)
    source1 = config["source1"]
    source2 = config["source2"]
    return (
        find_case_insensitive_dir(root_dir, (source1, source1.lower())),
        find_case_insensitive_dir(root_dir, (source2, source2.lower())),
    )


def verify_dataset_root(root_dir, dataset_name=None, strict=True, pair="ct_mri"):
    root_dir = Path(root_dir)
    label = dataset_name or str(root_dir)
    source1_label, source2_label, _ = pair_labels(pair)
    try:
        source1_dir, source2_dir = modality_dirs(root_dir, pair)
    except FileNotFoundError as error:
        if strict:
            raise
        print(f"[Dataset] {label}: {error}")
        return {"name": label, "root": root_dir, "pairs": 0, "missing": [str(error)]}

    source1_paths = _list_images(source1_dir)
    source2_paths = _list_images(source2_dir)
    source1_by_name = {path.name.lower(): path for path in source1_paths}
    source2_by_name = {path.name.lower(): path for path in source2_paths}
    matched_names = sorted(set(source1_by_name) & set(source2_by_name), key=lambda name: _natural_key(Path(name)))
    unmatched_source1 = sorted(set(source1_by_name) - set(source2_by_name))
    unmatched_source2 = sorted(set(source2_by_name) - set(source1_by_name))
    print(
        f"[Dataset] {label}: {len(matched_names)} matched {source1_label}/{source2_label} pairs | "
        f"{len(unmatched_source1)} unmatched {source1_label} | {len(unmatched_source2)} unmatched {source2_label}"
    )
    if strict and not matched_names:
        raise ValueError(f"No matched {source1_label}/{source2_label} image pairs found under: {root_dir}")
    return {
        "name": label,
        "root": root_dir,
        "pairs": len(matched_names),
        "missing": [],
        "unmatched_source1": len(unmatched_source1),
        "unmatched_source2": len(unmatched_source2),
    }


class PairedMedicalImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256, max_items=None, dataset_name=None, strict=True, pair="ct_mri"):
        self.root_dir = Path(root_dir)
        self.pair = normalize_pair(pair)
        self.dataset_name = dataset_name or self.root_dir.name
        self.source1_label, self.source2_label, _ = pair_labels(self.pair)
        self.image_size = image_size

        verify_dataset_root(self.root_dir, self.dataset_name, strict=strict, pair=self.pair)
        try:
            self.source1_dir, self.source2_dir = modality_dirs(self.root_dir, self.pair)
        except FileNotFoundError:
            self.pairs = []
            return

        source1_by_name = {path.name.lower(): path for path in _list_images(self.source1_dir)}
        source2_by_name = {path.name.lower(): path for path in _list_images(self.source2_dir)}
        matched_names = sorted(set(source1_by_name) & set(source2_by_name), key=lambda name: _natural_key(Path(name)))
        if max_items is not None:
            matched_names = matched_names[:max_items]
        self.pairs = [(source1_by_name[name], source2_by_name[name]) for name in matched_names]
        if strict and not self.pairs:
            raise ValueError(f"No matched image pairs found under: {self.root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source1_path, source2_path = self.pairs[index]
        source1 = _read_grayscale(source1_path, self.image_size)
        source2 = _read_grayscale(source2_path, self.image_size)
        return {
            "source1": source1,
            "source2": source2,
            "ct": source1,
            "mri": source2,
            "source1_path": str(source1_path),
            "source2_path": str(source2_path),
            "ct_path": str(source1_path),
            "mri_path": str(source2_path),
            "source1_label": self.source1_label,
            "source2_label": self.source2_label,
            "dataset": self.dataset_name,
            "pair": self.pair,
        }


class CombinedMedicalFusionDataset(ConcatDataset):
    def __init__(self, roots, image_size=256, max_items=None, split_name="train", strict=True, pair="ct_mri"):
        self.roots = [Path(root) for root in roots]
        datasets = []
        for root in self.roots:
            dataset = PairedMedicalImageDataset(
                root,
                image_size=image_size,
                max_items=max_items,
                dataset_name=f"AANLIB {normalize_pair(pair)} {split_name}",
                strict=strict,
                pair=pair,
            )
            if len(dataset) > 0:
                datasets.append(dataset)
        if not datasets and strict:
            raise ValueError(f"No valid AANLIB datasets found for roots: {self.roots}")
        super().__init__(datasets)


def build_aanlib_train_val_datasets(dataset_root, pair, image_size=256, max_items=None, val_split=0.15, seed=42):
    train_root = aanlib_split_root(dataset_root, pair, "train")
    val_root = aanlib_split_root(dataset_root, pair, "val")
    train_dataset = PairedMedicalImageDataset(
        train_root,
        image_size=image_size,
        max_items=max_items,
        dataset_name=f"AANLIB {normalize_pair(pair)} train",
        strict=True,
        pair=pair,
    )
    if val_root.exists():
        val_dataset = PairedMedicalImageDataset(
            val_root,
            image_size=image_size,
            max_items=max_items,
            dataset_name=f"AANLIB {normalize_pair(pair)} val",
            strict=True,
            pair=pair,
        )
        return train_dataset, val_dataset

    if len(train_dataset) <= 1 or val_split <= 0:
        return train_dataset, None
    val_count = max(1, int(len(train_dataset) * val_split))
    train_count = len(train_dataset) - val_count
    return random_split(
        train_dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(seed),
    )
