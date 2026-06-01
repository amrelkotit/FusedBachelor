#!/usr/bin/env python3
"""
Re-colorize existing GAN fused images without re-running the model.

Reads the generation_manifest.csv produced by generate_all_fused.py,
applies the improved colorization pipeline (CLAHE → unsharp mask →
YCbCr luma swap → chrominance boost) to every fused_original grayscale
image, and overwrites the fused_color / comparison_panels_color outputs
in-place.

Usage:
    python scripts/recolorize_fused.py                        # all pairs
    python scripts/recolorize_fused.py --pair spect_mri       # one pair
    python scripts/recolorize_fused.py --pair pet_mri spect_mri

The script is idempotent — safe to run multiple times.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import gan_image_dir  # noqa: E402

ALL_PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
COLOR_PAIRS = {"pet_mri", "spect_mri"}          # ct_mri is grayscale — skip


# ─────────────────────────────────────────────────────────────────────────────
# Core colorization (same as generate_all_fused.py – keep them in sync)
# ─────────────────────────────────────────────────────────────────────────────

def read_source_color(path, target_hw):
    """Load a color SPECT/PET image; return None if it is actually grayscale."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return None
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.shape[2] < 3:
        return None
    bgr = image[:, :, :3]
    delta = np.max(bgr, axis=2).astype(np.int16) - np.min(bgr, axis=2).astype(np.int16)
    if delta.max() <= 3 or delta.mean() <= 0.5:
        return None                                  # truly grayscale
    h, w = target_hw
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


def colorize_fused(source_bgr: np.ndarray, fused_gray: np.ndarray) -> np.ndarray:
    """
    Improved YCbCr colorization pipeline:
      1. CLAHE         — restore local contrast flattened by the GAN
      2. Unsharp mask  — sharpen sulci / lesion borders
      3. YCbCr swap    — inject enhanced luma into source color channels
      4. Chroma boost  — keep PET/SPECT colors vivid after luma change
    """
    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(fused_gray)

    # 2. Unsharp mask
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # 3. YCbCr luma swap
    ycbcr = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2YCrCb)
    ycbcr[:, :, 0] = enhanced

    # 4. Chrominance boost (×1.30)
    for ch in (1, 2):
        dev = ycbcr[:, :, ch].astype(np.int16) - 128
        ycbcr[:, :, ch] = np.clip(128 + dev * 1.30, 0, 255).astype(np.uint8)

    return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)


def rebuild_color_panel(source_bgr, mri_gray_tensor_uint8, fused_bgr, labels, path):
    """Three-panel: source color | MRI gray | fused color."""
    mri_bgr = cv2.cvtColor(mri_gray_tensor_uint8, cv2.COLOR_GRAY2BGR)
    panel = np.concatenate([source_bgr, mri_bgr, fused_bgr], axis=1)
    label_h = 30
    canvas = np.full((panel.shape[0] + label_h, panel.shape[1], 3), 255, dtype=np.uint8)
    canvas[label_h:] = panel
    for idx, label in enumerate(labels):
        cv2.putText(canvas, label, (idx * source_bgr.shape[1] + 10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair processing
# ─────────────────────────────────────────────────────────────────────────────

def process_pair(pair: str, split: str = "test"):
    output_root = gan_image_dir("aanlib", split, pair)
    manifest_path = output_root / "generation_manifest.csv"

    if not manifest_path.exists():
        print(f"  [SKIP] Manifest not found: {manifest_path}")
        return 0

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    color_dir = output_root / "fused_color"
    panel_dir = output_root / "comparison_panels_color"
    color_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Determine display labels (SPECT/PET label, MRI label, Fused)
    label_map = {
        "pet_mri":   ("PET",   "MRI", "Fused"),
        "spect_mri": ("SPECT", "MRI", "Fused"),
    }
    labels = label_map.get(pair, ("Source", "MRI", "Fused"))

    done = 0
    for row in rows:
        stem = f"{int(row['index']):04d}"
        fused_path  = Path(row["fused_path"])
        source_path = Path(row["source1_path"])
        mri_path    = Path(row["source2_path"])
        color_out   = color_dir  / f"{stem}_fused.png"
        panel_out   = panel_dir  / f"{stem}_comparison.png"

        # Load fused grayscale (GAN output)
        fused_gray = cv2.imread(str(fused_path), cv2.IMREAD_GRAYSCALE)
        if fused_gray is None:
            print(f"  [WARN] Cannot read fused: {fused_path}")
            continue

        # Load source COLOR image
        source_bgr = read_source_color(source_path, fused_gray.shape)
        if source_bgr is None:
            print(f"  [SKIP] Source is not a color image: {source_path}")
            continue

        # Colorize with improved pipeline
        fused_bgr = colorize_fused(source_bgr, fused_gray)

        # Save color output
        cv2.imwrite(str(color_out), fused_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Rebuild comparison panel
        mri_gray = cv2.imread(str(mri_path), cv2.IMREAD_GRAYSCALE)
        if mri_gray is not None:
            h, w = fused_gray.shape
            mri_gray = cv2.resize(mri_gray, (w, h), interpolation=cv2.INTER_CUBIC)
            rebuild_color_panel(source_bgr, mri_gray, fused_bgr, labels, panel_out)

        done += 1

    return done


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs="+",
                        choices=ALL_PAIRS, default=None,
                        help="Which pairs to process (default: pet_mri and spect_mri).")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = args.pair if args.pair else list(COLOR_PAIRS)
    # Filter to only color-capable pairs
    pairs = [p for p in pairs if p in COLOR_PAIRS]
    if not pairs:
        print("No color-capable pairs selected (only pet_mri / spect_mri support colorization).")
        return

    print("Re-colorizing fused outputs with improved pipeline")
    print("  CLAHE clipLimit=2.5  |  unsharp sigma=1.0 x1.5  |  chroma boost x1.30")
    print()

    total = 0
    for pair in pairs:
        print(f"[{pair}] split={args.split}")
        n = process_pair(pair, split=args.split)
        print(f"  -> {n} images written")
        total += n

    print(f"\nDone. {total} fused_color images updated.")


if __name__ == "__main__":
    main()
