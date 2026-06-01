"""
scripts/enhance_ct_mri_hq.py
─────────────────────────────
High-quality post-processing for CT-MRI fused images produced by both the
diffusion and GAN models.  Re-processes every PNG in the two fused_original
directories and saves results to a sibling fused_hq/ folder.

Enhancement pipeline (applied in this exact order):
  1. Load as grayscale uint8
  2. Percentile-normalise to full [0,255] using 1st / 99th percentile
  3. NLMeans denoising  – removes diffusion-model noise before sharpening
  4. CLAHE              – strong local contrast recovery  (clipLimit=2.2)
  5. Unsharp mask       – high-frequency sharpening       (sigma=1.0, α=0.40)
  6. Percentile-normalise again to restore full dynamic range
  7. Upscale to 512×512 with Lanczos4

Side-by-side comparison panel saved per image:
  original 256px | enhanced-but-not-upscaled 256px | upscaled 512px

Usage:
    python scripts/enhance_ct_mri_hq.py
    python scripts/enhance_ct_mri_hq.py --dry-run   # list files only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── repo root on sys.path so this script can be run from anywhere ────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── target directories ───────────────────────────────────────────────────────
DIFFUSION_SRC = (
    REPO_ROOT
    / "outputs" / "models" / "diffusion"
    / "images" / "aanlib" / "ct_mri" / "test" / "fused_original"
)
GAN_SRC = (
    REPO_ROOT
    / "outputs" / "models" / "gan"
    / "images" / "aanlib" / "test" / "ct_mri" / "fused_original"
)


# ── enhancement parameters ───────────────────────────────────────────────────
NLMEANS_H             = 6       # denoising strength
NLMEANS_TEMPLATE_WIN  = 7       # template patch size (pixels)
NLMEANS_SEARCH_WIN    = 21      # search window size  (pixels)
CLAHE_CLIP_LIMIT      = 2.2
CLAHE_TILE            = (8, 8)
UNSHARP_SIGMA         = 1.0
UNSHARP_STRENGTH      = 0.40
UPSCALE_SIZE          = 512
PERCENTILE_LOW        = 1.0
PERCENTILE_HIGH       = 99.0


# ── core helpers ─────────────────────────────────────────────────────────────

def percentile_stretch_u8(img: np.ndarray) -> np.ndarray:
    """Stretch uint8 image to full [0, 255] using 1st / 99th percentile."""
    assert img.dtype == np.uint8 and img.ndim == 2
    f = img.astype("float32")
    lo, hi = np.percentile(f, [PERCENTILE_LOW, PERCENTILE_HIGH])
    if hi <= lo + 1e-4:
        return img  # flat image – nothing to stretch
    stretched = np.clip((f - lo) / (hi - lo) * 255.0, 0, 255)
    return stretched.round().astype("uint8")


def enhance_hq(img_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Full HQ enhancement pipeline.

    Parameters
    ----------
    img_u8 : uint8 grayscale array [0, 255]

    Returns
    -------
    enhanced_256 : uint8 [H, W]   – enhanced at original resolution (for panel)
    upscaled_512 : uint8 [512,512] – final deliverable
    """
    assert img_u8.dtype == np.uint8 and img_u8.ndim == 2, \
        f"Expected uint8 2-D array, got dtype={img_u8.dtype} ndim={img_u8.ndim}"

    # Step 1 – already uint8 from caller

    # Step 2: percentile normalise → full [0,255]
    out = percentile_stretch_u8(img_u8)

    # Step 3: NLMeans denoising (removes diffusion model noise before sharpening)
    out = cv2.fastNlMeansDenoising(
        out,
        h=NLMEANS_H,
        templateWindowSize=NLMEANS_TEMPLATE_WIN,
        searchWindowSize=NLMEANS_SEARCH_WIN,
    )

    # Step 4: CLAHE – stronger local contrast recovery
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE)
    out = clahe.apply(out)

    # Step 5: Unsharp mask
    f = out.astype("float32")
    blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=UNSHARP_SIGMA)
    f = np.clip(f + UNSHARP_STRENGTH * (f - blurred), 0.0, 255.0)
    out = f.round().astype("uint8")

    # Step 6: Percentile normalise again → restore full dynamic range
    out = percentile_stretch_u8(out)

    enhanced_256 = out  # keep 256-px version for the comparison panel

    # Step 7: Upscale to 512×512 with Lanczos4
    upscaled_512 = cv2.resize(
        out, (UPSCALE_SIZE, UPSCALE_SIZE), interpolation=cv2.INTER_LANCZOS4
    )

    return enhanced_256, upscaled_512


def save_comparison_panel(
    original_256: np.ndarray,
    enhanced_256: np.ndarray,
    upscaled_512: np.ndarray,
    path: Path,
) -> None:
    """
    Save a side-by-side panel:
      [ original 256 | enhanced 256 | upscaled 512 ]
    All panels are padded to the same height (512 px) with a white bar on top
    for labels.
    """
    # Upscale original and enhanced to 512 px for visual consistency in the panel
    orig_panel  = cv2.resize(original_256,  (256, 256), interpolation=cv2.INTER_NEAREST)
    enh_panel   = cv2.resize(enhanced_256,  (256, 256), interpolation=cv2.INTER_NEAREST)
    up_panel    = cv2.resize(upscaled_512,  (512, 512), interpolation=cv2.INTER_NEAREST)

    # Pad left two panels to 512 px height with white
    def pad_height(arr: np.ndarray, target_h: int) -> np.ndarray:
        h, w = arr.shape[:2]
        if h == target_h:
            return arr
        pad = np.full((target_h - h, w), 255, dtype=np.uint8)
        return np.vstack([pad, arr])

    orig_panel = pad_height(orig_panel, 512)
    enh_panel  = pad_height(enh_panel, 512)

    panel = np.concatenate([orig_panel, enh_panel, up_panel], axis=1)

    # Label bar
    label_h = 30
    canvas = np.full((512 + label_h, panel.shape[1]), 255, dtype=np.uint8)
    canvas[label_h:, :] = panel

    labels = ["Original 256px", "Enhanced 256px", "Upscaled 512px (HQ)"]
    offsets = [0, 256, 512]
    for label, x_off in zip(labels, offsets):
        cv2.putText(
            canvas, label,
            (x_off + 8, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save comparison panel: {path}")


def process_directory(
    src_dir: Path,
    dry_run: bool = False,
) -> int:
    """
    Process all PNGs in *src_dir* and write outputs to sibling directories:
      • fused_hq/            – 512×512 upscaled enhanced images
      • fused_hq_comparison/ – side-by-side panels

    Returns the number of images successfully processed.
    """
    if not src_dir.exists():
        print(f"  [skip] Directory not found: {src_dir}")
        return 0

    png_paths = sorted(src_dir.glob("*.png"))
    if not png_paths:
        print(f"  [skip] No PNG files in: {src_dir}")
        return 0

    hq_dir  = src_dir.parent / "fused_hq"
    cmp_dir = src_dir.parent / "fused_hq_comparison"

    if not dry_run:
        hq_dir.mkdir(parents=True, exist_ok=True)
        cmp_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors    = 0

    for img_path in png_paths:
        if dry_run:
            print(f"  [dry-run] {img_path.name}")
            processed += 1
            continue

        # Step 1: load as grayscale uint8
        img_u8 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
            print(f"  [error] Cannot read: {img_path.name}")
            errors += 1
            continue

        try:
            enhanced_256, upscaled_512 = enhance_hq(img_u8)
        except Exception as exc:
            print(f"  [error] Enhancement failed for {img_path.name}: {exc}")
            errors += 1
            continue

        # Save upscaled HQ image
        hq_path = hq_dir / img_path.name
        ok = cv2.imwrite(str(hq_path), upscaled_512, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            print(f"  [error] Failed to write: {hq_path}")
            errors += 1
            continue

        # Save comparison panel
        cmp_path = cmp_dir / img_path.name
        try:
            save_comparison_panel(img_u8, enhanced_256, upscaled_512, cmp_path)
        except IOError as exc:
            print(f"  [warn] Comparison panel failed for {img_path.name}: {exc}")
            # not fatal – the HQ image was already saved

        processed += 1

    label = src_dir.relative_to(REPO_ROOT)
    if dry_run:
        print(f"  [dry-run] {label}: {processed} images would be processed")
    else:
        status = f"{processed} images"
        if errors:
            status += f" ({errors} errors)"
        print(f"  [done] {label} -> {processed} images -> fused_hq/")

    return processed


# ── entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HQ post-processing for CT-MRI fused images (diffusion + GAN)."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List what would be processed without writing any files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 65)
    print("  CT-MRI HQ Enhancement Pipeline")
    print(f"  NLMeans h={NLMEANS_H}  |  CLAHE clip={CLAHE_CLIP_LIMIT}  "
          f"|  unsharp sigma={UNSHARP_SIGMA} strength={UNSHARP_STRENGTH}")
    print(f"  Output size: {UPSCALE_SIZE}x{UPSCALE_SIZE}  |  interpolation: LANCZOS4")
    print("=" * 65)

    sources = [
        ("Diffusion", DIFFUSION_SRC),
        ("GAN",       GAN_SRC),
    ]

    total = 0
    for model_name, src_dir in sources:
        print(f"\n[{model_name}]  {src_dir}")
        n = process_directory(src_dir, dry_run=args.dry_run)
        total += n

    print(f"\n{'[dry-run] ' if args.dry_run else ''}Done.  "
          f"{total} image{'s' if total != 1 else ''} processed in total.")

    if not args.dry_run and total > 0:
        diff_hq = DIFFUSION_SRC.parent / "fused_hq"
        gan_hq  = GAN_SRC.parent       / "fused_hq"
        print(f"\nHQ outputs written to:")
        print(f"  {diff_hq}")
        print(f"  {gan_hq}")
        print(f"\nComparison panels written to:")
        print(f"  {DIFFUSION_SRC.parent / 'fused_hq_comparison'}")
        print(f"  {GAN_SRC.parent       / 'fused_hq_comparison'}")


if __name__ == "__main__":
    main()
