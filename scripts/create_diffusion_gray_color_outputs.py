import argparse
import csv
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import AANLIB_ROOT, PairedMedicalImageDataset, aanlib_split_root, diffusion_image_dir, normalize_pair
from src.diffusion.utils import postprocess_grayscale, presentation_enhance_grayscale, read_gray_uint01, resize_rgb_for_presentation, save_uint8_image, smart_colorize_functional_fusion


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def should_save_colored(args, pair):
    return bool(pair in {"pet_mri", "spect_mri"} and (args.save_colored or args.default_colored))


def deprecate_ct_colored_folder(image_root):
    wrong_dir = image_root / "fused_colored"
    if not wrong_dir.exists():
        return None
    destination = image_root / "_deprecated_fused_colored_wrong"
    if destination.exists():
        suffix = 1
        while (image_root / f"_deprecated_fused_colored_wrong_{suffix}").exists():
            suffix += 1
        destination = image_root / f"_deprecated_fused_colored_wrong_{suffix}"
    shutil.move(str(wrong_dir), str(destination))
    return destination


def process_pair(dataset_root, model_root, pair, split, args):
    pair = normalize_pair(pair)
    image_root = diffusion_image_dir("aanlib", pair, split, output_root=model_root)
    original_dir = image_root / "fused_original"
    grayscale_dir = image_root / "fused_grayscale"
    grayscale_presentation_dir = image_root / "fused_grayscale_presentation"
    colored_dir = image_root / "fused_colored"
    colored_presentation_dir = image_root / "fused_colored_presentation"
    if pair == "ct_mri":
        moved = deprecate_ct_colored_folder(image_root)
        if moved:
            print(f"[Cleanup] Moved old CT-MRI colored diffusion folder to: {moved}")
    if not original_dir.exists() or not any(original_dir.glob("*_fused.png")):
        raise FileNotFoundError("No diffusion generated images found. Run generate_all_fused_diffusion.py first.")
    if args.save_grayscale:
        grayscale_dir.mkdir(parents=True, exist_ok=True)
        if args.save_presentation:
            grayscale_presentation_dir.mkdir(parents=True, exist_ok=True)
    if should_save_colored(args, pair):
        colored_dir.mkdir(parents=True, exist_ok=True)
        if args.save_presentation:
            colored_presentation_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairedMedicalImageDataset(
        aanlib_split_root(dataset_root, pair, split),
        image_size=args.image_size,
        dataset_name=f"AANLIB {pair} {split}",
        strict=True,
        pair=pair,
    )
    rows = []
    for index, sample in enumerate(dataset):
        original_path = original_dir / f"{index:04d}_fused.png"
        if not original_path.exists():
            continue
        raw = read_gray_uint01(original_path, image_size=args.image_size)
        processed = postprocess_grayscale(
            raw,
            normalize_output=args.normalize_output,
            postprocess=args.postprocess,
            enhance_edges=args.enhance_edges,
        )
        grayscale_path = ""
        grayscale_presentation_path = ""
        colored_path = ""
        colored_presentation_path = ""
        color_mode = "none"
        color_source_used = "none"
        color_reference_path = ""
        if args.save_grayscale:
            grayscale_path = grayscale_dir / f"{index:04d}_fused.png"
            save_uint8_image((processed * 255.0).round().astype("uint8"), grayscale_path)
            if args.save_presentation:
                presentation = presentation_enhance_grayscale(processed, output_size=args.presentation_size)
                grayscale_presentation_path = grayscale_presentation_dir / f"{index:04d}_fused.png"
                save_uint8_image((presentation * 255.0).round().astype("uint8"), grayscale_presentation_path)
        if should_save_colored(args, pair):
            colored, color_mode, color_source_used, color_reference_path = smart_colorize_functional_fusion(
                processed,
                sample["source1_path"],
                pair,
                split,
                f"{index:04d}",
                gan_root=args.gan_root,
                color_mode=args.color_mode,
                alpha=args.color_alpha,
                color_strength=args.color_strength,
                saturation_boost=args.saturation_boost,
                contrast_boost=args.contrast_boost,
                background_threshold=args.background_threshold,
                colormap=args.colormap,
            )
            colored_path = colored_dir / f"{index:04d}_fused.png"
            save_uint8_image(cv2.cvtColor(colored, cv2.COLOR_RGB2BGR), colored_path)
            if args.save_presentation:
                if args.save_grayscale:
                    presentation_base = presentation
                else:
                    presentation_base = presentation_enhance_grayscale(processed, output_size=args.presentation_size)
                colored_presentation, _, _, _ = smart_colorize_functional_fusion(
                    presentation_base,
                    sample["source1_path"],
                    pair,
                    split,
                    f"{index:04d}",
                    gan_root=args.gan_root,
                    color_mode=args.color_mode,
                    alpha=args.color_alpha,
                    color_strength=args.color_strength,
                    saturation_boost=args.saturation_boost,
                    contrast_boost=args.contrast_boost,
                    background_threshold=args.background_threshold,
                    colormap=args.colormap,
                )
                colored_presentation = resize_rgb_for_presentation(colored_presentation, output_size=args.presentation_size)
                colored_presentation_path = colored_presentation_dir / f"{index:04d}_fused.png"
                save_uint8_image(cv2.cvtColor(colored_presentation, cv2.COLOR_RGB2BGR), colored_presentation_path)
        rows.append(
            {
                "index": index,
                "pair": pair,
                "split": split,
                "source_modality_path": sample["source1_path"],
                "mri_path": sample["source2_path"],
                "fused_original_path": str(original_path),
                "fused_grayscale_path": str(grayscale_path) if grayscale_path else "",
                "fused_grayscale_presentation_path": str(grayscale_presentation_path) if grayscale_presentation_path else "",
                "fused_colored_path": str(colored_path) if colored_path else "",
                "fused_colored_presentation_path": str(colored_presentation_path) if colored_presentation_path else "",
                "normalize_output_used": args.normalize_output,
                "postprocess_used": args.postprocess,
                "enhance_edges_used": args.enhance_edges,
                "color_alpha": args.color_alpha,
                "colormap": args.colormap,
                "color_mode": color_mode,
                "color_source_used": color_source_used,
                "color_strength": args.color_strength,
                "saturation_boost": args.saturation_boost,
                "contrast_boost": args.contrast_boost,
                "background_threshold": args.background_threshold,
                "color_reference_path": color_reference_path,
            }
        )
    manifest_path = image_root / "gray_color_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "index",
            "pair",
            "split",
            "source_modality_path",
            "mri_path",
            "fused_original_path",
            "fused_grayscale_path",
            "fused_grayscale_presentation_path",
            "fused_colored_path",
            "fused_colored_presentation_path",
            "normalize_output_used",
            "postprocess_used",
            "enhance_edges_used",
            "color_alpha",
            "colormap",
            "color_mode",
            "color_source_used",
            "color_strength",
            "saturation_boost",
            "contrast_boost",
            "background_threshold",
            "color_reference_path",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} processed diffusion image set(s): {image_root}")
    if pair != "ct_mri":
        used = {row.get("color_source_used") for row in rows if row.get("fused_colored_path")}
        if "gan_style" in used:
            print("Colored output created using GAN-style HSV color transfer")
        if "source_rgb" in used:
            print("Colored output created using source RGB HSV color transfer")
        if "colormap_fallback" in used:
            print("Colored output created using colormap fallback")
    print(f"Saved manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create diffusion grayscale/color outputs from existing fused_original images.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--model-root", default="outputs/models/diffusion")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--normalize-output", dest="normalize_output", action="store_true", default=True)
    parser.add_argument("--no-normalize-output", dest="normalize_output", action="store_false")
    parser.add_argument("--postprocess", dest="postprocess", action="store_true", default=True)
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false")
    parser.add_argument("--enhance-edges", dest="enhance_edges", action="store_true", default=True)
    parser.add_argument("--no-enhance-edges", dest="enhance_edges", action="store_false")
    parser.add_argument("--save-grayscale", dest="save_grayscale", action="store_true", default=True)
    parser.add_argument("--no-save-grayscale", dest="save_grayscale", action="store_false")
    parser.add_argument("--save-presentation", dest="save_presentation", action="store_true", default=True)
    parser.add_argument("--no-save-presentation", dest="save_presentation", action="store_false")
    parser.add_argument("--presentation-size", type=int, default=512)
    parser.add_argument("--save-colored", action="store_true")
    parser.add_argument("--no-default-colored", dest="default_colored", action="store_false", default=True)
    parser.add_argument("--gan-root", default="outputs/models/gan")
    parser.add_argument("--color-mode", choices=["smart", "source", "gan-style", "colormap"], default="smart")
    parser.add_argument("--color-alpha", type=float, default=0.65)
    parser.add_argument("--color-strength", type=float, default=1.0)
    parser.add_argument("--saturation-boost", type=float, default=1.35)
    parser.add_argument("--contrast-boost", type=float, default=1.15)
    parser.add_argument("--background-threshold", type=float, default=0.03)
    parser.add_argument("--colormap", choices=["hot", "jet", "inferno", "magma", "viridis"], default="hot")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    model_root = resolve_path(args.model_root)
    args.gan_root = resolve_path(args.gan_root)
    pairs = PAIRS if args.pair == "all" else [normalize_pair(args.pair)]
    for pair in pairs:
        process_pair(dataset_root, model_root, pair, args.split, args)


if __name__ == "__main__":
    main()
