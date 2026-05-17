import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.paired_dataset import AANLIB_ROOT, PairedMedicalImageDataset, aanlib_split_root, diffusion_after_msfd_dir, diffusion_checkpoint_dir, diffusion_graph_dir, diffusion_image_dir, diffusion_metrics_dir, normalize_pair, pair_labels
from src.diffusion.model import ConditionalUNet
from src.diffusion.sampler import sample_refined
from src.diffusion.scheduler import DiffusionScheduler
from src.diffusion.utils import (
    fusion_reference,
    postprocess_grayscale,
    presentation_enhance_grayscale,
    resize_rgb_for_presentation,
    save_comparison_panel,
    save_tensor_image,
    save_uint8_image,
    smart_colorize_functional_fusion,
    tensor_to_numpy01,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_state_dict_keys(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_model_state_dict(checkpoint, checkpoint_path=None):
    if not isinstance(checkpoint, dict):
        return checkpoint
    checkpoint_name = Path(checkpoint_path).stem.lower() if checkpoint_path is not None else ""
    prefer_ema = bool(checkpoint.get("is_ema", False) or "ema" in checkpoint_name)
    keys = ("ema_model_state_dict", "model_state_dict", "state_dict") if prefer_ema else ("model_state_dict", "state_dict", "ema_model_state_dict")
    for key in keys:
        if checkpoint.get(key) is not None:
            return checkpoint[key]
    return checkpoint


def load_model(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = ConditionalUNet(in_channels=4, base_channels=32, time_dim=128).to(device)
    model.load_state_dict(normalize_state_dict_keys(extract_model_state_dict(checkpoint, checkpoint_path)), strict=True)
    model.eval()
    timesteps = int(config.get("timesteps", 1000))
    is_ema = bool(isinstance(checkpoint, dict) and checkpoint.get("is_ema", "ema" in checkpoint_path.stem))
    return model, DiffusionScheduler(timesteps=timesteps, device=device), checkpoint_path, is_ema


def select_default_checkpoint(pair, output_root):
    checkpoint_dir = diffusion_checkpoint_dir(pair, output_root=output_root)
    ema_path = checkpoint_dir / "best_diffusion_ema.pt"
    best_path = checkpoint_dir / "best_diffusion.pt"
    return ema_path if ema_path.is_file() else best_path


def prefer_ema_checkpoint(checkpoint_path, enabled=True):
    checkpoint_path = resolve_path(checkpoint_path)
    if not enabled:
        return checkpoint_path
    if checkpoint_path.name != "best_diffusion.pt":
        return checkpoint_path
    ema_path = checkpoint_path.with_name("best_diffusion_ema.pt")
    return ema_path if ema_path.is_file() else checkpoint_path


def should_save_colored(args):
    return bool(args.pair in {"pet_mri", "spect_mri"} and (args.save_colored or args.default_colored))


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


def save_processed_outputs(fused_tensor, source1_path, dirs, stem, args):
    raw = tensor_to_numpy01(fused_tensor)
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
        grayscale_path = dirs["grayscale"] / f"{stem}_fused.png"
        save_uint8_image((processed * 255.0).round().astype("uint8"), grayscale_path)
        if args.save_presentation:
            presentation = presentation_enhance_grayscale(processed, output_size=args.presentation_size)
            grayscale_presentation_path = dirs["grayscale_presentation"] / f"{stem}_fused.png"
            save_uint8_image((presentation * 255.0).round().astype("uint8"), grayscale_presentation_path)
    if should_save_colored(args):
        colored, color_mode, color_source_used, color_reference_path = smart_colorize_functional_fusion(
            processed,
            source1_path,
            args.pair,
            args.split,
            stem,
            gan_root=args.gan_root,
            color_mode=args.color_mode,
            alpha=args.color_alpha,
            color_strength=args.color_strength,
            saturation_boost=args.saturation_boost,
            contrast_boost=args.contrast_boost,
            background_threshold=args.background_threshold,
            colormap=args.colormap,
        )
        colored_path = dirs["colored"] / f"{stem}_fused.png"
        save_uint8_image(cv2.cvtColor(colored, cv2.COLOR_RGB2BGR), colored_path)
        if args.save_presentation:
            if args.save_grayscale:
                presentation_base = presentation
            else:
                presentation_base = presentation_enhance_grayscale(processed, output_size=args.presentation_size)
            colored_presentation, _, _, _ = smart_colorize_functional_fusion(
                presentation_base,
                source1_path,
                args.pair,
                args.split,
                stem,
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
            colored_presentation_path = dirs["colored_presentation"] / f"{stem}_fused.png"
            save_uint8_image(cv2.cvtColor(colored_presentation, cv2.COLOR_RGB2BGR), colored_presentation_path)
    return (
        processed,
        str(grayscale_path) if grayscale_path else "",
        str(grayscale_presentation_path) if grayscale_presentation_path else "",
        str(colored_path) if colored_path else "",
        str(colored_presentation_path) if colored_presentation_path else "",
        color_mode,
        color_source_used,
        color_reference_path,
    )


@torch.no_grad()
def generate_split(model, scheduler, dataset, output_root, msfd_root, labels, checkpoint_path, args, device):
    fused_dir = output_root / "fused_original"
    grayscale_dir = output_root / "fused_grayscale"
    grayscale_presentation_dir = output_root / "fused_grayscale_presentation"
    colored_dir = output_root / "fused_colored"
    colored_presentation_dir = output_root / "fused_colored_presentation"
    panel_dir = output_root / "comparison_panels"
    initial_dir = output_root / "initial_fused"
    msfd_dir = msfd_root / args.split
    dirs = {
        "grayscale": grayscale_dir,
        "grayscale_presentation": grayscale_presentation_dir,
        "colored": colored_dir,
        "colored_presentation": colored_presentation_dir,
    }
    if args.pair == "ct_mri":
        moved = deprecate_ct_colored_folder(output_root)
        if moved:
            print(f"[Cleanup] Moved old CT-MRI colored diffusion folder to: {moved}")
    folders = [panel_dir, initial_dir, msfd_dir]
    if args.save_original:
        folders.append(fused_dir)
    if args.save_grayscale:
        folders.append(grayscale_dir)
        if args.save_presentation:
            folders.append(grayscale_presentation_dir)
    if should_save_colored(args):
        folders.append(colored_dir)
        if args.save_presentation:
            folders.append(colored_presentation_dir)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    rows = []
    written = 0
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for batch in tqdm(loader, desc=f"Generate diffusion {args.pair} {args.split}", dynamic_ncols=True, file=sys.stdout):
        source1 = batch["source1"].to(device)
        source2 = batch["source2"].to(device)
        fused, initial = sample_refined(model, scheduler, source1.float(), source2.float(), sampling_steps=args.sampling_steps)
        msfd = fusion_reference(source1.float(), source2.float(), use_msfd=True).to(device)
        for batch_index in range(fused.shape[0]):
            if args.max_items is not None and written >= args.max_items:
                return rows
            stem = f"{written:04d}"
            fused_path = fused_dir / f"{stem}_fused.png"
            initial_path = initial_dir / f"{stem}_initial.png"
            panel_path = panel_dir / f"{stem}_comparison.png"
            msfd_path = msfd_dir / f"{stem}_msfd_reference.png"
            if args.save_original:
                save_tensor_image(fused[batch_index].clamp(0.0, 1.0), fused_path)
            save_tensor_image(initial[batch_index], initial_path)
            save_tensor_image(msfd[batch_index], msfd_path)
            (
                processed,
                grayscale_path,
                grayscale_presentation_path,
                colored_path,
                colored_presentation_path,
                color_mode,
                color_source_used,
                color_reference_path,
            ) = save_processed_outputs(
                fused[batch_index].clamp(0.0, 1.0),
                batch["source1_path"][batch_index],
                dirs,
                stem,
                args,
            )
            save_comparison_panel(
                [source1[batch_index], source2[batch_index], initial[batch_index], fused[batch_index]],
                [labels[0], labels[1], "Initial", "Diffusion"],
                panel_path,
            )
            rows.append(
                {
                    "index": written,
                    "pair": args.pair,
                    "split": args.split,
                    "source_modality_path": batch["source1_path"][batch_index],
                    "mri_path": batch["source2_path"][batch_index],
                    "initial_fused_path": str(initial_path),
                    "fused_original_path": str(fused_path) if args.save_original else "",
                    "fused_grayscale_path": grayscale_path,
                    "fused_grayscale_presentation_path": grayscale_presentation_path,
                    "fused_colored_path": colored_path,
                    "fused_colored_presentation_path": colored_presentation_path,
                    "comparison_panel": str(panel_path),
                    "msfd_reference_path": str(msfd_path),
                    "checkpoint_path": str(checkpoint_path),
                    "sampling_steps": args.sampling_steps,
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
            written += 1
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fused images using only a diffusion checkpoint.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], default="ct_mri")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prefer-ema", dest="prefer_ema", action="store_true", default=True)
    parser.add_argument("--no-prefer-ema", dest="prefer_ema", action="store_false")
    parser.add_argument("--output-root", default="outputs/models/diffusion")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=150)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize-output", dest="normalize_output", action="store_true", default=True)
    parser.add_argument("--no-normalize-output", dest="normalize_output", action="store_false")
    parser.add_argument("--postprocess", dest="postprocess", action="store_true", default=True)
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false")
    parser.add_argument("--enhance-edges", dest="enhance_edges", action="store_true", default=True)
    parser.add_argument("--no-enhance-edges", dest="enhance_edges", action="store_false")
    parser.add_argument("--save-original", dest="save_original", action="store_true", default=True)
    parser.add_argument("--no-save-original", dest="save_original", action="store_false")
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


def print_color_validation(rows, pair):
    if pair == "ct_mri":
        return
    used = {row.get("color_source_used") for row in rows if row.get("fused_colored_path")}
    if "gan_style" in used:
        print("Colored output created using GAN-style HSV color transfer")
    if "source_rgb" in used:
        print("Colored output created using source RGB HSV color transfer")
    if "colormap_fallback" in used:
        print("Colored output created using colormap fallback")


def update_global_manifest(output_root, rows):
    metrics_dir = diffusion_metrics_dir(output_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "diffusion_generation_manifest_all.csv"
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
        "checkpoint_path",
        "sampling_steps",
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
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as csv_file:
            existing = list(csv.DictReader(csv_file))
    keys = {(str(row["pair"]), str(row["split"]), str(row["index"])) for row in rows}
    existing = [row for row in existing if (str(row.get("pair")), str(row.get("split")), str(row.get("index"))) not in keys]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in existing + rows])
    return path


def main():
    args = parse_args()
    args.pair = normalize_pair(args.pair)
    output_root = resolve_path(args.output_root)
    args.gan_root = resolve_path(args.gan_root)
    checkpoint = prefer_ema_checkpoint(args.checkpoint or select_default_checkpoint(args.pair, output_root), enabled=args.prefer_ema)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    try:
        model, scheduler, checkpoint_path, is_ema = load_model(checkpoint, device)
    except Exception:
        fallback = resolve_path(args.checkpoint) if args.checkpoint else diffusion_checkpoint_dir(args.pair, output_root=output_root) / "best_diffusion.pt"
        if resolve_path(checkpoint) == fallback:
            raise
        print(f"[Checkpoint] Could not load preferred EMA checkpoint, falling back to: {fallback}")
        model, scheduler, checkpoint_path, is_ema = load_model(fallback, device)
    split_root = aanlib_split_root(resolve_path(args.dataset_root), args.pair, args.split)
    image_root = diffusion_image_dir("aanlib", args.pair, args.split, output_root=output_root)
    msfd_root = diffusion_after_msfd_dir("aanlib", args.pair, output_root=output_root)
    print("[Model] Selected model: Diffusion")
    print(f"[Output] Root: {output_root}")
    print(f"[Pair] {args.pair}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint type: {'EMA' if is_ema else 'non-EMA'}")
    print(f"Fused image folder: {image_root / 'fused_original'}")
    print(f"Fused grayscale folder: {image_root / 'fused_grayscale'}")
    if should_save_colored(args):
        print(f"Fused colored folder: {image_root / 'fused_colored'}")
    print(f"MSFD/intermediate folder: {msfd_root / args.split}")
    print(f"Graph folder: {diffusion_graph_dir(output_root)}")
    print(f"Metrics folder: {diffusion_metrics_dir(output_root)}")
    dataset = PairedMedicalImageDataset(split_root, image_size=args.image_size, dataset_name=f"aanlib {args.split}", strict=True, pair=args.pair)
    rows = generate_split(model, scheduler, dataset, image_root, msfd_root, pair_labels(args.pair), checkpoint_path, args, device)
    manifest_path = image_root / "generation_manifest.csv"
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
            "checkpoint_path",
            "sampling_steps",
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
            "initial_fused_path",
            "comparison_panel",
            "msfd_reference_path",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    global_manifest = update_global_manifest(output_root, rows)
    print_color_validation(rows, args.pair)
    print(f"Generated {len(rows)} fused image(s)")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved global manifest: {global_manifest}")


if __name__ == "__main__":
    main()
