"""
thesis_enhance_images.py
========================
Thesis-quality enhancement for diffusion-model fused medical images.

What it does (in order, per image):
  1. Backup original (first run only) → <folder>_original_backup/
  2. NLMeans denoising  – removes the diffusion "dot" noise
  3. Bilateral filter   – smooths flat regions, keeps edges
  4. Unsharp mask       – restores / boosts edge definition
  5. Percentile clip    – ensures full dynamic range is used

Colour images (spect_mri, pet_mri  →  fused_colored):
  Processing happens ONLY on the L channel of LAB colour space so
  hue and saturation are never touched.

Grayscale images (ct_mri → fused_hq):
  Standard single-channel pipeline.

Usage
-----
  python thesis_enhance_images.py            # enhance all three modalities
  python thesis_enhance_images.py --dry-run  # preview paths, no writes
  python thesis_enhance_images.py --restore  # copy backups back (undo)
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ── folder targets ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
AANLIB = BASE / "outputs" / "models" / "diffusion" / "images" / "aanlib"

TARGETS = [
    # (folder, is_color)
    (AANLIB / "spect_mri" / "test" / "fused_colored",  True),
    (AANLIB / "pet_mri"   / "test" / "fused_colored",  True),
    (AANLIB / "ct_mri"    / "test" / "fused_hq",       False),
]

# ── enhancement parameters ────────────────────────────────────────────────────
# NLMeans – removes diffusion noise dots without blurring edges
NLM_H              = 5     # denoising strength (lower = gentler)
NLM_TEMPLATE_WIN   = 7     # patch size
NLM_SEARCH_WIN     = 21    # search window

# Bilateral – smooths flat regions, hard-preserves edges
BIL_D              = 5     # pixel-neighbourhood diameter
BIL_SIGMA_COLOR    = 20    # colour tolerance  (higher = more smoothing)
BIL_SIGMA_SPACE    = 5     # spatial extent

# Unsharp mask – edge / detail boost
USM_SIGMA          = 1.2   # blur radius for the mask
USM_STRENGTH       = 0.50  # how much of the high-freq detail to add back
                            # (0.5 = visually crisp; keep ≤ 0.7 to avoid haloes)

# ── helpers ───────────────────────────────────────────────────────────────────

def backup_folder(src: Path) -> Path:
    dst = src.parent / (src.name + "_original_backup")
    if dst.exists():
        return dst  # already backed up
    shutil.copytree(src, dst)
    print(f"  [backup] {dst.relative_to(BASE)}")
    return dst


def restore_folder(folder: Path) -> None:
    backup = folder.parent / (folder.name + "_original_backup")
    if not backup.exists():
        print(f"  [skip]  no backup found for {folder.name}")
        return
    for src in backup.glob("*.png"):
        shutil.copy2(src, folder / src.name)
    print(f"  [restored] {folder.relative_to(BASE)}")


def percentile_stretch_u8(u8: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Stretch contrast so the image uses the full 0-255 range."""
    p_lo, p_hi = np.percentile(u8, [lo, hi])
    if p_hi <= p_lo + 1:
        return u8
    stretched = (u8.astype("float32") - p_lo) / (p_hi - p_lo) * 255.0
    return np.clip(stretched, 0, 255).round().astype("uint8")


def enhance_channel(ch: np.ndarray) -> np.ndarray:
    """
    Apply the full noise-removal + sharpening pipeline to a single
    uint8 grayscale channel.
    """
    # Step 1: NLMeans – kills the diffusion dot noise
    ch = cv2.fastNlMeansDenoising(
        ch,
        h=NLM_H,
        templateWindowSize=NLM_TEMPLATE_WIN,
        searchWindowSize=NLM_SEARCH_WIN,
    )

    # Step 2: Bilateral – smooths flat areas, preserves edges
    ch = cv2.bilateralFilter(ch, d=BIL_D, sigmaColor=BIL_SIGMA_COLOR, sigmaSpace=BIL_SIGMA_SPACE)

    # Step 3: Unsharp mask – add back high-frequency edge detail
    f = ch.astype("float32") / 255.0
    blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=USM_SIGMA)
    sharpened = np.clip(f + USM_STRENGTH * (f - blurred), 0.0, 1.0)
    ch = (sharpened * 255.0).round().astype("uint8")

    # Step 4: Percentile stretch – ensure full dynamic range
    ch = percentile_stretch_u8(ch)

    return ch


def enhance_color_image(path: Path) -> np.ndarray:
    """
    Load a BGR colour image and enhance it while keeping hue/saturation
    untouched (only the L channel of LAB is processed).
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read: {path}")

    # Convert to LAB (L: 0-255, A/B: 0-255 in uint8 OpenCV convention)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Enhance only the L (luminance) channel
    L_enh = enhance_channel(L)

    lab_enh = cv2.merge([L_enh, A, B])
    bgr_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    return bgr_enh


def enhance_gray_image(path: Path) -> np.ndarray:
    """
    Load a grayscale image and apply the full enhancement pipeline.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read: {path}")
    return enhance_channel(gray)


def process_folder(folder: Path, is_color: bool, dry_run: bool) -> int:
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        print(f"  [warn]  no PNGs found in {folder}")
        return 0

    if not dry_run:
        backup_folder(folder)

    count = 0
    for img_path in pngs:
        if dry_run:
            print(f"  [dry]   would enhance {img_path.name}")
            count += 1
            continue

        try:
            if is_color:
                result = enhance_color_image(img_path)
            else:
                result = enhance_gray_image(img_path)

            cv2.imwrite(
                str(img_path),
                result,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],  # lossless
            )
            count += 1
        except Exception as exc:
            print(f"  [error] {img_path.name}: {exc}", file=sys.stderr)

    return count


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis-quality image enhancement")
    parser.add_argument("--dry-run",  action="store_true", help="Print paths without modifying files")
    parser.add_argument("--restore",  action="store_true", help="Restore originals from backup")
    args = parser.parse_args()

    if args.restore:
        print("Restoring originals from backup …")
        for folder, _ in TARGETS:
            restore_folder(folder)
        print("Done.")
        return

    label = "[DRY RUN] " if args.dry_run else ""
    print(f"{label}Enhancing fused images for thesis …\n")

    total = 0
    for folder, is_color in TARGETS:
        mode = "colour (LAB)" if is_color else "grayscale"
        print(f"  {folder.relative_to(BASE)}  [{mode}]")
        if not folder.exists():
            print(f"  [skip]  folder not found")
            continue
        n = process_folder(folder, is_color, dry_run=args.dry_run)
        print(f"  → {n} image(s) {'would be ' if args.dry_run else ''}enhanced\n")
        total += n

    print(f"{'Would enhance' if args.dry_run else 'Enhanced'} {total} image(s) total.")
    if not args.dry_run:
        print("Originals are backed up in <folder>_original_backup/ — run with --restore to undo.")


if __name__ == "__main__":
    main()
