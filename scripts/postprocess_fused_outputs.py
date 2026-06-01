#!/usr/bin/env python3
"""
PART 1 — Post-process existing diffusion fusion outputs (no GPU required).

Enhancement pipeline per image:
  1. Histogram matching   : align intensity distribution to MRI reference
  2. Guided filter        : edge-preserving denoising (requires opencv-contrib)
  3. Weighted source blend: 0.70*filtered + 0.20*mri_ref + 0.10*source_modality
  4. Unsharp mask         : recover sharpness lost in step 1

Saves post-processed images to:
  outputs/models/diffusion/images_postprocessed/aanlib/{pair}/test/{variant}/

Re-evaluates all images and saves:
  outputs/models/diffusion/metrics/postprocessed_metrics_summary.csv
  outputs/models/diffusion/metrics/postprocessed_vs_original_delta.csv

Prints a before/after comparison table to terminal.
"""

import sys
import subprocess

# ------------------------------------------------------------------
# Auto-install missing packages
# ------------------------------------------------------------------
for _pip, _imp in [
    ("scikit-image", "skimage"),
    ("scipy", "scipy"),
    ("opencv-contrib-python", "cv2"),
]:
    try:
        __import__(_imp)
    except ImportError:
        print(f"[INSTALL] pip install {_pip} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pip, "-q"])

# ------------------------------------------------------------------
# Standard imports (after install)
# ------------------------------------------------------------------
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from skimage import exposure
    from skimage.filters import unsharp_mask as _unsharp_mask
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    warnings.warn("scikit-image not found; histogram matching / unsharp mask will use fallbacks.")

from src.evaluation.metrics import evaluate_fusion  # noqa: E402 — after sys.path insert

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
VARIANTS = ["fused_original", "fused_grayscale", "fused_colored"]
DIFFUSION_ROOT = PROJECT_ROOT / "outputs" / "models" / "diffusion"
INPUT_ROOT = DIFFUSION_ROOT / "images" / "aanlib"
OUTPUT_ROOT = DIFFUSION_ROOT / "images_postprocessed" / "aanlib"
METRICS_DIR = DIFFUSION_ROOT / "metrics"
DETAILED_CSV = METRICS_DIR / "diffusion_metrics_detailed.csv"

METRIC_NAMES = ["SSIM", "PSNR", "MI", "EN", "SF", "AG", "CC", "FMI"]

# ------------------------------------------------------------------
# Image I/O helpers
# ------------------------------------------------------------------

def load_float(path):
    """Load image from *path* as float32 ndarray in [0, 1].  Returns None on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


def save_float(arr, path):
    """Save float32 [0, 1] image as 8-bit PNG to *path* (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(arr * 255.0, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------
# Enhancement steps
# ------------------------------------------------------------------

def _histogram_match(fused: np.ndarray, mri_ref: np.ndarray) -> np.ndarray:
    if not _HAS_SKIMAGE:
        return fused
    try:
        result = exposure.match_histograms(fused, mri_ref)
    except TypeError:
        # older scikit-image used multichannel keyword
        result = exposure.match_histograms(fused, mri_ref, multichannel=False)
    return result.astype(np.float32)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    try:
        guide_u8 = np.clip(guide * 255, 0, 255).astype(np.uint8)
        result = cv2.ximgproc.guidedFilter(
            guide=guide_u8, src=src.astype(np.float32), radius=radius, eps=eps
        )
        return np.clip(result, 0.0, 1.0).astype(np.float32)
    except AttributeError:
        print("  [WARN] cv2.ximgproc unavailable (need opencv-contrib-python). Skipping guided filter.")
        return src


def _unsharp(arr: np.ndarray, radius: float = 1.5, amount: float = 0.35) -> np.ndarray:
    if _HAS_SKIMAGE:
        return np.clip(_unsharp_mask(arr, radius=radius, amount=amount), 0.0, 1.0).astype(np.float32)
    # Manual fallback
    blurred = cv2.GaussianBlur(arr, (0, 0), radius)
    return np.clip(arr + amount * (arr - blurred), 0.0, 1.0).astype(np.float32)


def enhance(fused: np.ndarray, mri_ref: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Apply the full 4-step enhancement pipeline."""
    step1 = _histogram_match(fused, mri_ref)
    step2 = _guided_filter(guide=mri_ref, src=step1, radius=8, eps=0.01)
    step3 = np.clip(0.70 * step2 + 0.20 * mri_ref + 0.10 * source, 0.0, 1.0).astype(np.float32)
    return _unsharp(step3, radius=1.5, amount=0.35)


# ------------------------------------------------------------------
# Metrics helper
# ------------------------------------------------------------------

def compute_metrics(fused: np.ndarray, mri: np.ndarray, source: np.ndarray) -> dict:
    fused_t = torch.from_numpy(fused).unsqueeze(0)
    mri_t   = torch.from_numpy(mri).unsqueeze(0)
    src_t   = torch.from_numpy(source).unsqueeze(0)
    with torch.no_grad():
        m = evaluate_fusion(fused_t, mri_t, src_t)
    return {k: m.get(k, 0.0) for k in METRIC_NAMES}


# ------------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------------

def load_detailed_csv():
    if not DETAILED_CSV.exists():
        print(f"[WARN] Detailed CSV not found: {DETAILED_CSV}")
        return {}
    lookup = {}
    with open(DETAILED_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pair    = row.get("pair", "").strip()
            img_idx = row.get("image", "").strip()
            variant = row.get("variant", "").strip()
            source_path = row.get("source_modality_path", "").strip()
            mri_path    = row.get("mri_path", "").strip()
            lookup[(pair, img_idx, variant)] = (source_path, mri_path)
    return lookup


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("POST-PROCESSING DIFFUSION FUSION OUTPUTS")
    print("=" * 70)

    lookup = load_detailed_csv()

    orig_by_key  = defaultdict(list)  # (pair, variant) -> list of metric dicts
    post_by_key  = defaultdict(list)

    files_created = []

    for pair in PAIRS:
        print(f"\n--- Pair: {pair} ---")
        for variant_dir in VARIANTS:
            variant_key = f"diffusion_{variant_dir}"   # e.g. "diffusion_fused_grayscale"
            input_dir  = INPUT_ROOT / pair / "test" / variant_dir
            output_dir = OUTPUT_ROOT / pair / "test" / variant_dir

            if not input_dir.exists():
                print(f"  [SKIP] Input dir not found: {input_dir}")
                continue

            img_files = sorted(
                [p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            )
            if not img_files:
                print(f"  [SKIP] No images in {input_dir}")
                continue

            print(f"  {variant_dir}: {len(img_files)} images")
            output_dir.mkdir(parents=True, exist_ok=True)
            processed = 0

            for img_file in img_files:
                img_idx = img_file.stem.replace("_fused", "")
                key = (pair, img_idx, variant_key)

                paths = lookup.get(key)
                if paths is None:
                    print(f"    [WARN] No CSV entry for {key} — skipping")
                    continue

                source_path, mri_path = paths
                fused  = load_float(img_file)
                mri    = load_float(mri_path)
                source = load_float(source_path)

                if fused is None:
                    print(f"    [WARN] Cannot read {img_file}")
                    continue
                if mri is None:
                    print(f"    [WARN] Cannot read MRI {mri_path}")
                    continue
                if source is None:
                    print(f"    [WARN] Cannot read source {source_path}")
                    continue

                # Ensure same spatial size
                h, w = fused.shape
                if mri.shape != (h, w):
                    mri = cv2.resize(mri, (w, h), interpolation=cv2.INTER_CUBIC)
                if source.shape != (h, w):
                    source = cv2.resize(source, (w, h), interpolation=cv2.INTER_CUBIC)

                # Original metrics
                orig_m = compute_metrics(fused, mri, source)
                orig_m.update({"pair": pair, "variant": variant_key, "image": img_idx})
                orig_by_key[(pair, variant_key)].append(orig_m)

                # Enhance
                enhanced = enhance(fused, mri, source)
                out_path = output_dir / img_file.name
                save_float(enhanced, out_path)
                if processed == 0:
                    files_created.append(str(output_dir))

                # Post-processed metrics
                post_m = compute_metrics(enhanced, mri, source)
                post_m.update({"pair": pair, "variant": variant_key, "image": img_idx})
                post_by_key[(pair, variant_key)].append(post_m)
                processed += 1

            print(f"    → {processed} images saved to {output_dir}")

    if not orig_by_key:
        print("\n[WARN] No images were processed. Verify that input directories exist.")
        return

    # ------------------------------------------------------------------
    # Save postprocessed_metrics_summary.csv
    # ------------------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = METRICS_DIR / "postprocessed_metrics_summary.csv"
    summary_rows = []
    for (pair, variant) in sorted(orig_by_key.keys()):
        for mname in METRIC_NAMES:
            for typ, store in [("original", orig_by_key), ("postprocessed", post_by_key)]:
                vals = [float(r[mname]) for r in store[(pair, variant)]]
                summary_rows.append({
                    "pair": pair, "variant": variant, "type": typ, "metric": mname,
                    "mean": f"{np.mean(vals):.6f}", "std": f"{np.std(vals):.6f}",
                    "n": len(vals),
                })

    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pair", "variant", "type", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(summary_rows)
    files_created.append(str(summary_path))
    print(f"\n[SAVED] {summary_path}")

    # ------------------------------------------------------------------
    # Save postprocessed_vs_original_delta.csv
    # ------------------------------------------------------------------
    delta_path = METRICS_DIR / "postprocessed_vs_original_delta.csv"
    delta_rows = []
    for (pair, variant) in sorted(orig_by_key.keys()):
        for mname in METRIC_NAMES:
            orig_vals = [float(r[mname]) for r in orig_by_key[(pair, variant)]]
            post_vals = [float(r[mname]) for r in post_by_key.get((pair, variant), [])]
            if not orig_vals or not post_vals:
                continue
            om, pm = np.mean(orig_vals), np.mean(post_vals)
            delta_rows.append({
                "pair": pair, "variant": variant, "metric": mname,
                "original_mean": f"{om:.6f}", "postprocessed_mean": f"{pm:.6f}",
                "delta": f"{pm - om:.6f}",
                "delta_pct": f"{100 * (pm - om) / (abs(om) + 1e-8):.2f}",
            })

    with open(delta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "pair", "variant", "metric", "original_mean", "postprocessed_mean", "delta", "delta_pct"
        ])
        writer.writeheader()
        writer.writerows(delta_rows)
    files_created.append(str(delta_path))
    print(f"[SAVED] {delta_path}")

    # ------------------------------------------------------------------
    # Print before/after comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("BEFORE / AFTER COMPARISON (per-variant mean values)")
    print("=" * 84)
    print(f"{'Pair':<12} {'Variant':<28} {'Metric':<8} {'Before':>9} {'After':>9} {'Delta':>9} {'%':>7}")
    print("-" * 84)
    for row in delta_rows:
        print(
            f"{row['pair']:<12} {row['variant']:<28} {row['metric']:<8} "
            f"{float(row['original_mean']):>9.4f} {float(row['postprocessed_mean']):>9.4f} "
            f"{float(row['delta']):>+9.4f} {float(row['delta_pct']):>6.2f}%"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FILES CREATED / WRITTEN:")
    for f in files_created:
        print(f"  {f}")
    print(f"\nPost-processed images → {OUTPUT_ROOT}")
    print(f"Metrics              → {METRICS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
