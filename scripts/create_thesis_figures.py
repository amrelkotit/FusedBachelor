import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.paired_dataset import (
    AANLIB_ROOT,
    PairedMedicalImageDataset,
    aanlib_split_root,
    gan_image_dir,
    gan_thesis_figures_dir,
    normalize_pair,
    pair_labels,
)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_gray(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def read_color_if_available(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 2:
        return None
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.shape[2] < 3:
        return None
    bgr = image[:, :, :3]
    channel_delta = bgr.max(axis=2).astype("int16") - bgr.min(axis=2).astype("int16")
    if channel_delta.max() <= 3 or channel_delta.mean() <= 0.5:
        return None
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_display_image(path, size):
    color = read_color_if_available(path, size)
    if color is not None:
        return color, None
    return read_gray(path, size), "gray"


def to_uint8(tensor):
    image = tensor.detach().clamp(0, 1).squeeze().numpy()
    return (image * 255).round().astype("uint8")


def default_fused_dir(pair, split):
    original_dir = gan_image_dir("aanlib", split, pair) / "fused_original"
    if pair in {"pet_mri", "spect_mri"}:
        color_dir = gan_image_dir("aanlib", split, pair) / "fused_color"
        if color_dir.exists() and any(color_dir.glob("*_fused.png")):
            return color_dir
    return original_dir


def create_figure(pair, dataset_root, split, fused_dir, output_dir, image_size, index):
    dataset = PairedMedicalImageDataset(
        aanlib_split_root(dataset_root, pair, split),
        image_size=image_size,
        dataset_name=f"AANLIB {pair} {split}",
        strict=True,
        pair=pair,
    )
    fused_paths = sorted(fused_dir.glob("*_fused.png"))
    if not fused_paths:
        raise FileNotFoundError(f"No fused images found in: {fused_dir}")
    sample_index = min(index, len(dataset) - 1, len(fused_paths) - 1)
    sample = dataset[sample_index]
    source1_color = read_color_if_available(sample["source1_path"], image_size) if pair in {"pet_mri", "spect_mri"} else None
    fused_image, fused_cmap = read_display_image(fused_paths[sample_index], image_size)
    images = [source1_color if source1_color is not None else to_uint8(sample["source1"]), to_uint8(sample["source2"]), fused_image]
    cmaps = [None if source1_color is not None else "gray", "gray", fused_cmap]
    labels = pair_labels(pair)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=220)
    for ax, image, label, cmap in zip(axes, images, labels, cmaps):
        if cmap == "gray":
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.04)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pair}_qualitative_comparison.png"
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create thesis-quality qualitative comparison figures.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], required=True)
    parser.add_argument("--split", choices=["test", "train", "val"], default="test")
    parser.add_argument("--fused-dir", default=None)
    parser.add_argument("--output-dir", default=str(gan_thesis_figures_dir()))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    pair = normalize_pair(args.pair)
    fused_dir = resolve_path(args.fused_dir) if args.fused_dir else default_fused_dir(pair, args.split)
    output_dir = resolve_path(args.output_dir)
    print(f"Checkpoint path: thesis figures use existing generated images")
    print(f"Fused image folder: {fused_dir}")
    print(f"Graph folder: {PROJECT_ROOT / 'outputs' / 'models' / 'gan' / 'graphs'}")
    print(f"Metrics folder: {PROJECT_ROOT / 'outputs' / 'models' / 'gan' / 'metrics'}")
    path = create_figure(pair, resolve_path(args.dataset_root), args.split, fused_dir, output_dir, args.image_size, args.index)
    print(f"Saved thesis figure: {path}")


if __name__ == "__main__":
    main()
