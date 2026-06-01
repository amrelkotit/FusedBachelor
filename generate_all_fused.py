import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.paired_dataset import (
    AANLIB_ROOT,
    BRATS_ROOT,
    PairedMedicalImageDataset,
    aanlib_split_root,
    gan_checkpoint_dir,
    gan_graph_dir,
    gan_image_dir,
    gan_metrics_dir,
    normalize_pair,
    pair_labels,
)
from src.fusion.decomposition import multiscale_fuse
from src.models.gan import FusionGenerator


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def tensor_to_uint8(tensor):
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    return (image * 255.0).round().astype("uint8")


def save_uint8_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def read_source_color(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read source image: {path}")
    if image.ndim == 2:
        return None
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.shape[2] < 3:
        return None
    bgr = image[:, :, :3]
    channel_delta = np.max(bgr, axis=2).astype(np.int16) - np.min(bgr, axis=2).astype(np.int16)
    if channel_delta.max() <= 3 or channel_delta.mean() <= 0.5:
        return None
    return cv2.resize(bgr, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def _enhance_fused_gray_for_spect(fused_gray):
    """CLAHE + unsharp-mask + Laplacian edge boost on a denoised SPECT luma channel."""
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
    enhanced = clahe.apply(fused_gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.7)
    sharpened = cv2.addWeighted(enhanced, 1.75, blur, -0.75, 0)
    lap = cv2.Laplacian(sharpened, cv2.CV_32F)
    return np.clip(sharpened.astype(np.float32) - 0.35 * lap, 0, 255).astype(np.uint8)


def colorize_fused_pet_spect(source_bgr, fused_gray):
    """Colorize a PET/SPECT fused image.

    Applies CLAHE + unsharp-mask to the (already denoised) luma, then either:
    - transfers it into the source YCrCb space with a mild chrominance boost, or
    - maps it through COLORMAP_HOT when the source has no colour information.
    """
    enhanced = _enhance_fused_gray_for_spect(fused_gray)
    if source_bgr is None:
        return cv2.applyColorMap(enhanced, cv2.COLORMAP_HOT)
    ycrcb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = enhanced
    for ch in (1, 2):
        deviation = ycrcb[:, :, ch].astype(np.int16) - 128
        ycrcb[:, :, ch] = np.clip(128 + deviation * 1.50, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def ycrcb_colorize(source_bgr, mri_u8, alpha=0.70):
    """Color from source SPECT/PET Cr/Cb channels + MRI as luminance + brain mask.

    Using MRI as Y gives crisp anatomical structure while the source color channels
    preserve the functional (SPECT/PET) pseudocolor encoding.  The result is then
    blended 70/30 with the enhanced MRI gray background for the same layered look
    as the AdaFuse reference images.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mri_enhanced = clahe.apply(mri_u8)
    ycrcb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.normalize(mri_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    colored_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    mri_bgr = cv2.cvtColor(mri_enhanced, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(colored_bgr, alpha, mri_bgr, 1.0 - alpha, 0)
    _, mask = cv2.threshold(mri_enhanced, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_and(blended, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))


def adafuse_colorize(functional_u8, mri_u8, alpha=0.65):
    """Fallback colorizer when source has no color: hot colormap over MRI background."""
    functional_norm = cv2.normalize(functional_u8, None, 0, 255, cv2.NORM_MINMAX)
    hot_bgr = cv2.applyColorMap(functional_norm, cv2.COLORMAP_HOT)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mri_enhanced = clahe.apply(mri_u8)
    mri_bgr = cv2.cvtColor(mri_enhanced, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(hot_bgr, alpha, mri_bgr, 1.0 - alpha, 0)
    _, mask = cv2.threshold(mri_enhanced, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_and(blended, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))


def colorize_fused_from_source(source_bgr, fused_gray):
    """Colorize a CT-MRI fused image using CLAHE + unsharp mask + YCbCr transfer.

    Used only for CT-MRI where adaptive contrast and sharpening improve
    structural detail in soft tissue and bone edges.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(fused_gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    ycbcr = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2YCrCb)
    ycbcr[:, :, 0] = enhanced
    for ch in (1, 2):
        deviation = ycbcr[:, :, ch].astype(np.int16) - 128
        ycbcr[:, :, ch] = np.clip(128 + deviation * 1.30, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)


def save_comparison_panel(source1, source2, fused, labels, path, msfd_guidance=None):
    images = [tensor_to_uint8(source1), tensor_to_uint8(source2), tensor_to_uint8(fused)]
    panel_labels = list(labels)
    if msfd_guidance is not None:
        images.insert(2, tensor_to_uint8(msfd_guidance))
        panel_labels = [labels[0], labels[1], "MSFD guidance", labels[2]]
    panel = np.concatenate(images, axis=1)
    label_h = 30
    canvas = np.full((panel.shape[0] + label_h, panel.shape[1]), 255, dtype=np.uint8)
    canvas[label_h:, :] = panel
    for index, label in enumerate(panel_labels):
        cv2.putText(canvas, label, (index * images[0].shape[1] + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    save_uint8_image(canvas, path)


def save_color_comparison_panel(source1_bgr, source2_gray, fused_bgr, labels, path):
    source2_bgr = cv2.cvtColor(tensor_to_uint8(source2_gray), cv2.COLOR_GRAY2BGR)
    images = [source1_bgr, source2_bgr, fused_bgr]
    panel = np.concatenate(images, axis=1)
    label_h = 30
    canvas = np.full((panel.shape[0] + label_h, panel.shape[1], 3), 255, dtype=np.uint8)
    canvas[label_h:, :] = panel
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (index * images[0].shape[1] + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    save_uint8_image(canvas, path)


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
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = FusionGenerator().to(device)
    generator.load_state_dict(normalize_state_dict_keys(extract_generator_state_dict(checkpoint)), strict=True)
    generator.eval()
    return generator, checkpoint_path


def dataset_root_for_args(args, pair):
    if args.external_test == "brats":
        if pair != "ct_mri":
            raise ValueError("BRATS external generalization testing is supported only for ct_mri.")
        return resolve_path(args.brats_root), "brats", "test"
    return aanlib_split_root(resolve_path(args.dataset_root), pair, args.split), "aanlib", args.split


@torch.no_grad()
def generate_split(generator, dataset, output_root, labels, checkpoint_path, args, device):
    fused_dir = output_root / "fused_original"
    panel_dir = output_root / "comparison_panels"
    color_dir = output_root / "fused_color"
    color_panel_dir = output_root / "comparison_panels_color"
    msfd_preview_dir = output_root / "msfd_guidance_preview"
    supports_color_visualization = args.pair in {"pet_mri", "spect_mri"}
    fused_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    if args.save_msfd_guidance_preview:
        msfd_preview_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    written = 0
    color_written = 0
    bar = tqdm(loader, desc=f"Generate {args.pair} {args.split}", leave=False, dynamic_ncols=True, file=sys.stdout, ascii=True)
    for batch in bar:
        source1 = batch["source1"].to(device)
        source2 = batch["source2"].to(device)
        fused = generator(source1.float(), source2.float()).clamp(0.0, 1.0)
        msfd_guidance = None
        if args.save_msfd_guidance_preview:
            msfd_guidance = multiscale_fuse(source1=source1.float(), source2=source2.float()).to(device=device, dtype=fused.dtype).clamp(0.0, 1.0)
        for batch_index in range(fused.shape[0]):
            if args.max_items is not None and written >= args.max_items:
                return rows
            stem = f"{written:04d}"
            fused_path = fused_dir / f"{stem}_fused.png"
            panel_path = panel_dir / f"{stem}_comparison.png"
            msfd_preview_path = ""
            fused_gray = tensor_to_uint8(fused[batch_index])
            # Remove isolated noise dots (common in PET/SPECT) without blurring edges.
            # d=7 / sigmaColor=20 / sigmaSpace=12 gives cleaner input for the
            # CLAHE + unsharp-mask that follows in colorize_fused_pet_spect.
            if args.pair in {"pet_mri", "spect_mri"}:
                fused_gray = cv2.bilateralFilter(fused_gray, d=5, sigmaColor=30, sigmaSpace=8)
            save_uint8_image(fused_gray, fused_path)
            if msfd_guidance is not None:
                msfd_preview_path = msfd_preview_dir / f"{stem}_msfd_guidance.png"
                save_uint8_image(tensor_to_uint8(msfd_guidance[batch_index]), msfd_preview_path)
            save_comparison_panel(
                source1[batch_index],
                source2[batch_index],
                fused[batch_index],
                labels,
                panel_path,
                msfd_guidance=msfd_guidance[batch_index] if msfd_guidance is not None else None,
            )
            color_path = ""
            color_panel_path = ""
            if supports_color_visualization:
                mri_u8 = tensor_to_uint8(source2[batch_index])
                source1_color = read_source_color(batch["source1_path"][batch_index], fused_gray.shape)
                if source1_color is not None:
                    fused_color = ycrcb_colorize(source1_color, mri_u8)
                else:
                    fused_color = adafuse_colorize(
                        tensor_to_uint8(source1[batch_index]),
                        mri_u8,
                    )
                color_path = color_dir / f"{stem}_fused.png"
                color_panel_path = color_panel_dir / f"{stem}_comparison.png"
                save_uint8_image(fused_color, color_path)
                panel_source1 = source1_color if source1_color is not None else cv2.cvtColor(tensor_to_uint8(source1[batch_index]), cv2.COLOR_GRAY2BGR)
                save_color_comparison_panel(panel_source1, source2[batch_index], fused_color, labels, color_panel_path)
                color_written += 1
            rows.append(
                {
                    "index": written,
                    "source1_path": batch["source1_path"][batch_index],
                    "source2_path": batch["source2_path"][batch_index],
                    "fused_path": str(fused_path),
                    "comparison_panel": str(panel_path),
                    "fused_color_path": str(color_path) if color_path else "",
                    "comparison_panel_color": str(color_panel_path) if color_panel_path else "",
                    "msfd_guidance_preview_path": str(msfd_preview_path) if msfd_preview_path else "",
                    "checkpoint": str(checkpoint_path),
                }
            )
            written += 1
        bar.set_postfix(written=written, color=color_written)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate raw GAN fused images for one explicit AANLIB pair/split.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT), help="AANLIB root.")
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], default="ct_mri")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--external-test", choices=["brats"], default=None, help="Opt-in external generalization testing only.")
    parser.add_argument("--brats-root", default=str(BRATS_ROOT / "test"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-msfd-guidance-preview", action="store_true", help="Save rule-based MSFD guidance beside generator output for visual comparison only.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.pair = normalize_pair(args.pair)
    checkpoint = args.checkpoint or str(gan_checkpoint_dir(args.pair) / "best_generator.pt")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    generator, checkpoint_path = load_generator(checkpoint, device)
    split_root, dataset_name, split = dataset_root_for_args(args, args.pair)
    output_root = resolve_path(args.output_dir) if args.output_dir else gan_image_dir(dataset_name, split, args.pair)

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Fused image folder: {output_root / 'fused_original'}")
    if args.save_msfd_guidance_preview:
        print(f"MSFD guidance preview folder: {output_root / 'msfd_guidance_preview'}")
    if args.pair in {"pet_mri", "spect_mri"}:
        print(f"Visualization-only color folder: {output_root / 'fused_color'}")
    print(f"Graph folder: {gan_graph_dir()}")
    print(f"Metrics folder: {gan_metrics_dir()}")
    if dataset_name == "brats":
        print("Dataset label: BRATS external generalization testing")

    dataset = PairedMedicalImageDataset(split_root, image_size=args.image_size, dataset_name=f"{dataset_name} {split}", strict=True, pair=args.pair)
    rows = generate_split(generator, dataset, output_root, pair_labels(args.pair), checkpoint_path, args, device)
    manifest_path = output_root / "generation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "index",
            "source1_path",
            "source2_path",
            "fused_path",
            "comparison_panel",
            "fused_color_path",
            "comparison_panel_color",
            "msfd_guidance_preview_path",
            "checkpoint",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} fused image(s)")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
