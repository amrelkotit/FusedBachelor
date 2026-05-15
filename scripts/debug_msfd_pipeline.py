import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import (
    AANLIB_ROOT,
    PairedMedicalImageDataset,
    aanlib_split_root,
    gan_checkpoint_dir,
    normalize_pair,
    pair_labels,
)
from src.fusion.decomposition import multiscale_fuse
from src.models.gan import FusionGenerator


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "gan" / "debug_msfd_pipeline"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def tensor_to_uint8(tensor):
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    return (image * 255.0).round().astype("uint8")


def normalize_state_dict_keys(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_generator_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("generator", "generator_state_dict", "model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_generator(checkpoint_path, device):
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device)
    generator = FusionGenerator().to(device)
    generator.load_state_dict(normalize_state_dict_keys(extract_generator_state_dict(checkpoint)), strict=True)
    generator.eval()
    return generator


def save_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), tensor_to_uint8(tensor), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def save_grid(images, labels, path):
    uint8_images = [tensor_to_uint8(image) for image in images]
    panel = np.concatenate(uint8_images, axis=1)
    label_h = 30
    canvas = np.full((panel.shape[0] + label_h, panel.shape[1]), 255, dtype=np.uint8)
    canvas[label_h:, :] = panel
    tile_w = uint8_images[0].shape[1]
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (index * tile_w + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save grid: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Debug the optional MSFD guidance path on one paired sample.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], default="ct_mri")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--checkpoint", default=None, help="Optional generator or full checkpoint. Defaults to the pair best_generator.pt when present.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    pair = normalize_pair(args.pair)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    split_root = aanlib_split_root(resolve_path(args.dataset_root), pair, args.split)
    dataset = PairedMedicalImageDataset(split_root, image_size=args.image_size, dataset_name=f"AANLIB {pair} {args.split}", strict=True, pair=pair)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"--sample-index must be in [0, {len(dataset) - 1}], got {args.sample_index}")

    sample = dataset[args.sample_index]
    source1 = sample["source1"].unsqueeze(0).to(device)
    source2 = sample["source2"].unsqueeze(0).to(device)
    msfd_guidance = multiscale_fuse(source1=source1.float(), source2=source2.float()).clamp(0.0, 1.0)

    checkpoint = args.checkpoint
    if checkpoint is None:
        default_checkpoint = gan_checkpoint_dir(pair) / "best_generator.pt"
        checkpoint = str(default_checkpoint) if default_checkpoint.exists() else None

    if checkpoint:
        generator = load_generator(checkpoint, device)
        gan_fused = generator(source1.float(), source2.float()).clamp(0.0, 1.0)
    else:
        gan_fused = torch.zeros_like(msfd_guidance)

    labels = pair_labels(pair)
    output_dir = resolve_path(args.output_dir) / pair / args.split / f"sample_{args.sample_index:04d}"
    save_image(source1[0], output_dir / "source1.png")
    save_image(source2[0], output_dir / "source2.png")
    save_image(msfd_guidance[0], output_dir / "msfd_guidance.png")
    save_image(gan_fused[0], output_dir / "gan_fused.png")
    save_grid(
        [source1[0], source2[0], msfd_guidance[0], gan_fused[0]],
        [labels[0], labels[1], "MSFD guidance", "GAN fused" if checkpoint else "GAN fused (no checkpoint)"],
        output_dir / "debug_grid.png",
    )

    print(f"Saved debug outputs: {output_dir}")
    print(f"Source1: {sample['source1_path']}")
    print(f"Source2: {sample['source2_path']}")
    print(f"Checkpoint: {checkpoint if checkpoint else 'none; GAN fused tile is blank'}")


if __name__ == "__main__":
    main()
