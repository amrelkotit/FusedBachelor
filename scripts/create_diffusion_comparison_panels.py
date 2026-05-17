import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import AANLIB_ROOT, PairedMedicalImageDataset, aanlib_split_root, diffusion_final_assets_dir, diffusion_image_dir, normalize_pair, pair_config


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_image(path, color=False, size=None):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None:
        return None
    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        target_w, target_h = size
        h, w = image.shape[:2]
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        canvas_shape = (target_h, target_w, 3) if resized.ndim == 3 else (target_h, target_w)
        canvas = np.zeros(canvas_shape, dtype=resized.dtype)
        y0 = (target_h - resized_h) // 2
        x0 = (target_w - resized_w) // 2
        canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
        image = canvas
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


def save_panel(items, destination):
    images = [(label, image) for label, image in items if image is not None]
    if not images:
        return None
    size = (images[0][1].shape[1], images[0][1].shape[0])
    resized = [(label, cv2.resize(image, size, interpolation=cv2.INTER_AREA)) for label, image in images]
    panel = np.concatenate([image for _, image in resized], axis=1)
    label_h = 48
    canvas = np.full((panel.shape[0] + label_h, panel.shape[1], 3), 255, dtype=np.uint8)
    canvas[label_h:, :] = panel
    for index, (label, _) in enumerate(resized):
        font_scale = 0.8
        max_width = size[0] - 16
        while font_scale > 0.45 and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] > max_width:
            font_scale -= 0.04
        cv2.putText(canvas, label, (index * size[0] + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return destination if ok else None


def create_pair_panels(dataset_root, model_root, gan_root, pair, split, image_size, max_items):
    pair = normalize_pair(pair)
    cfg = pair_config(pair)
    dataset = PairedMedicalImageDataset(
        aanlib_split_root(dataset_root, pair, split),
        image_size=image_size,
        max_items=max_items,
        dataset_name=f"AANLIB {pair} {split}",
        strict=True,
        pair=pair,
    )
    image_root = diffusion_image_dir("aanlib", pair, split, output_root=model_root)
    comparison_dir = model_root / "comparisons" / "aanlib" / pair / split
    thesis_dir = diffusion_final_assets_dir(model_root) / "comparison_panels"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, sample in enumerate(dataset):
        source1 = read_image(sample["source1_path"], size=(image_size, image_size))
        mri = read_image(sample["source2_path"], size=(image_size, image_size))
        gray_path = first_existing(
            [
                image_root / "fused_grayscale_presentation" / f"{index:04d}_fused.png",
                image_root / "fused_grayscale" / f"{index:04d}_fused.png",
            ]
        )
        original_path = image_root / "fused_original" / f"{index:04d}_fused.png"
        diffusion_gray = read_image(gray_path if gray_path.exists() else original_path, size=(image_size, image_size))
        diffusion_color = None
        if pair != "ct_mri":
            color_path = first_existing(
                [
                    image_root / "fused_colored_presentation" / f"{index:04d}_fused.png",
                    image_root / "fused_colored" / f"{index:04d}_fused.png",
                ]
            )
            diffusion_color = read_image(color_path, color=True, size=(image_size, image_size))
        gan_dir = "fused_original" if pair == "ct_mri" else "fused_color"
        gan = read_image(gan_root / "images" / "aanlib" / split / pair / gan_dir / f"{index:04d}_fused.png", color=(pair != "ct_mri"), size=(image_size, image_size))
        baseline = read_image(model_root / "baselines" / "average" / "aanlib" / pair / split / f"{index:04d}_average.png", size=(image_size, image_size))
        if pair == "ct_mri":
            items = [("MRI", mri), ("CT", source1), ("Diffusion Fused Grayscale", diffusion_gray), ("GAN Fused", gan), ("Average Baseline", baseline)]
        else:
            items = [
                ("MRI", mri),
                (cfg["source1"], source1),
                ("Diffusion Fused Grayscale", diffusion_gray),
                ("Diffusion Fused Colored", diffusion_color),
                ("GAN Fused Colored", gan),
                ("Average Baseline", baseline),
            ]
        destination = comparison_dir / f"{index:04d}_comparison_panel.png"
        path = save_panel(items, destination)
        if path:
            written.append(path)
            if len(written) <= 5:
                shutil.copy2(path, thesis_dir / f"{pair}_{index:04d}_comparison_panel.png")
    print(f"Saved {len(written)} comparison panel(s): {comparison_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create diffusion thesis comparison panels.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--model-root", default="outputs/models/diffusion")
    parser.add_argument("--gan-root", default="outputs/models/gan")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    model_root = resolve_path(args.model_root)
    gan_root = resolve_path(args.gan_root)
    pairs = PAIRS if args.pair == "all" else [normalize_pair(args.pair)]
    for pair in pairs:
        create_pair_panels(dataset_root, model_root, gan_root, pair, args.split, args.image_size, args.max_items)


if __name__ == "__main__":
    main()
