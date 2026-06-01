"""
enhance_outputs.py
──────────────────
Applies multi-stage post-processing to existing fused images to recover
detail that the current (capacity-limited) models were unable to produce.

Pipeline per image:
  1. CLAHE  – local contrast enhancement (preserves anatomical structure)
  2. Bilateral filter – edge-preserving smooth  (reduce flat region noise)
  3. Unsharp mask  – high-frequency sharpening  (brings back edge detail)
  4. Percentile normalisation – ensures full [0,255] dynamic range

Outputs are written to <original_dir>_enhanced/ alongside originals,
so nothing is overwritten.

Usage:
    python enhance_outputs.py                        # all models & pairs
    python enhance_outputs.py --model gan            # GAN only
    python enhance_outputs.py --model diffusion      # Diffusion only
    python enhance_outputs.py --pair ct_mri          # one pair
    python enhance_outputs.py --clip-limit 2.0       # stronger CLAHE
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# ── parameters ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       choices=["gan", "diffusion", "all"], default="all")
    p.add_argument("--pair",        choices=["ct_mri", "pet_mri", "spect_mri", "all"], default="all")
    p.add_argument("--clip-limit",  type=float, default=1.5,
                   help="CLAHE clip limit (1.0–3.0). Higher = stronger contrast.")
    p.add_argument("--unsharp-strength", type=float, default=0.22,
                   help="Unsharp mask weight (0.1–0.4). Higher = sharper edges.")
    p.add_argument("--unsharp-sigma",    type=float, default=0.9,
                   help="Gaussian sigma for unsharp mask (0.5–1.5).")
    return p.parse_args()


# ── core enhancement ────────────────────────────────────────────────────────

def enhance(img_u8: np.ndarray, clip_limit=1.5, unsharp_strength=0.22, unsharp_sigma=0.9) -> np.ndarray:
    """
    img_u8 : uint8 grayscale image [0,255]
    returns: uint8 grayscale image [0,255]
    """
    assert img_u8.dtype == np.uint8 and img_u8.ndim == 2

    # 1. CLAHE – adaptive local contrast
    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out     = clahe.apply(img_u8)

    # 2. Bilateral filter – smooth flat areas while preserving edges
    out     = cv2.bilateralFilter(out, d=5, sigmaColor=12, sigmaSpace=5)

    # 3. Unsharp mask – sharpen edges
    blurred = cv2.GaussianBlur(out.astype("float32"), (0, 0), sigmaX=unsharp_sigma)
    out_f   = np.clip(out.astype("float32") + unsharp_strength * (out.astype("float32") - blurred),
                      0, 255)

    # 4. Percentile normalisation – stretch dynamic range
    lo, hi  = np.percentile(out_f, [0.5, 99.5])
    if hi > lo + 1e-4:
        out_f = np.clip((out_f - lo) / (hi - lo) * 255.0, 0, 255)

    return out_f.round().astype("uint8")


def process_directory(src_dir: Path, clip_limit, unsharp_strength, unsharp_sigma, suffix="_enhanced"):
    """Process all PNG images in src_dir, save to src_dir + suffix."""
    if not src_dir.exists():
        return 0

    dst_dir = src_dir.parent / (src_dir.name + suffix)
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src_dir.glob("*.png")):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [skip] Cannot read {img_path.name}")
            continue

        enhanced = enhance(img, clip_limit, unsharp_strength, unsharp_sigma)
        out_path = dst_dir / img_path.name
        ok = cv2.imwrite(str(out_path), enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if ok:
            count += 1
        else:
            print(f"  [error] Failed to write {out_path}")

    return count


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    ROOT   = Path(__file__).resolve().parent / "outputs" / "models"

    models = (["gan", "diffusion"] if args.model == "all" else [args.model])
    pairs  = (["ct_mri", "pet_mri", "spect_mri"] if args.pair == "all" else [args.pair])

    kw = dict(
        clip_limit       = args.clip_limit,
        unsharp_strength = args.unsharp_strength,
        unsharp_sigma    = args.unsharp_sigma,
    )

    total = 0

    for model in models:
        for pair in pairs:
            # ── locate the fused_original directory for this model/pair ──
            if model == "gan":
                search_root = ROOT / "gan" / "images" / "aanlib" / "test" / pair
            else:
                search_root = ROOT / "diffusion" / "images" / "aanlib" / pair / "test"

            if not search_root.exists():
                print(f"[skip] {search_root} not found")
                continue

            # Find subdirs named fused_original or fused_grayscale
            target_dirs = [
                d for d in search_root.iterdir()
                if d.is_dir() and any(kw2 in d.name for kw2 in ("fused_original", "fused_grayscale"))
                and "enhanced" not in d.name
                and "presentation" not in d.name
            ]

            for src_dir in target_dirs:
                n = process_directory(src_dir, **kw)
                total += n
                if n:
                    print(f"  ✓  {model}/{pair}/{src_dir.name} → {n} images enhanced")

    print(f"\nDone. {total} images enhanced total.")
    print("Enhanced images saved next to originals in *_enhanced/ folders.")


if __name__ == "__main__":
    main()
