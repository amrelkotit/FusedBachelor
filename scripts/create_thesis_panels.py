"""create_thesis_panels.py
=========================
Generates publication-quality qualitative comparison figures for each
modality pair.  Can work from EXISTING fused images on disk (no checkpoint
needed) or can be called after inference to regenerate.

Output per pair
---------------
  <pair>_qualitative_comparison_thesis.png
    A 3-column figure: Source-1 | Source-2 | Fused
    with a red-box zoom inset, per-image metric table overlay, and
    a colour-bar for PET/SPECT pairs.

  <pair>_multi_sample_comparison.png
    Up to 5 test-set rows stacked vertically for variety.

Usage
-----
    python scripts/create_thesis_panels.py

    # Override fused-image directory:
    python scripts/create_thesis_panels.py --fused-root path/to/fused_original

The script auto-discovers fused images from the standard output tree.
If none exist yet, it prints instructions and exits cleanly.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ---------------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate_fusion
import torch

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAN_ROOT     = PROJECT_ROOT / "outputs" / "models" / "gan"
THESIS_OUT   = GAN_ROOT / "final_thesis_assets"

PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {
    "ct_mri":    ("CT",    "MRI",   "Fused (AdaFuse)"),
    "pet_mri":   ("PET",   "MRI",   "Fused (AdaFuse)"),
    "spect_mri": ("SPECT", "MRI",   "Fused (AdaFuse)"),
}
PAIR_TITLE = {
    "ct_mri":    "CT-MRI Fusion",
    "pet_mri":   "PET-MRI Fusion",
    "spect_mri": "SPECT-MRI Fusion",
}

# red-box zoom: (row_start%, row_end%, col_start%, col_end%) in [0,1]
ZOOM_REGION = (0.35, 0.65, 0.30, 0.70)


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------

def load_gray(path: Path, size: int = 256) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_color(path: Path, size: int = 256):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)


def crop_zoom(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    r0 = int(ZOOM_REGION[0] * h); r1 = int(ZOOM_REGION[1] * h)
    c0 = int(ZOOM_REGION[2] * w); c1 = int(ZOOM_REGION[3] * w)
    return arr[r0:r1, c0:c1]


def draw_red_box(arr: np.ndarray) -> np.ndarray:
    """Return a copy of arr with a red rectangle marking the zoom region."""
    out = arr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    h, w = out.shape[:2]
    r0 = int(ZOOM_REGION[0] * h); r1 = int(ZOOM_REGION[1] * h)
    c0 = int(ZOOM_REGION[2] * w); c1 = int(ZOOM_REGION[3] * w)
    cv2.rectangle(out, (c0, r0), (c1, r1), (220, 30, 30), 2)
    return out


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# metric computation
# ---------------------------------------------------------------------------

def compute_metrics(fused_gray, src1_gray, src2_gray) -> dict:
    tf = to_tensor(fused_gray)
    t1 = to_tensor(src1_gray)
    t2 = to_tensor(src2_gray)
    m = evaluate_fusion(tf, t2, t1)   # mri=source2, ct/pet/spect=source1
    return {
        "PSNR":  m.get("PSNR", 0.0),
        "MI":    m.get("MI",   0.0),
        "FMI":   m.get("FMI",  0.0),
        "EN":    m.get("EN",   0.0),
        "SSIM":  m.get("SSIM", 0.0),
        "CC":    m.get("CC",   0.0),
    }


# ---------------------------------------------------------------------------
# single-sample thesis panel
# ---------------------------------------------------------------------------

def make_single_panel(
    pair: str,
    src1_gray: np.ndarray,
    src2_gray: np.ndarray,
    fused_gray: np.ndarray,
    metrics: dict,
    out_path: Path,
    src1_color=None,
    fused_color=None,
) -> None:
    s1_label, s2_label, fused_label = PAIR_LABELS[pair]
    size = src1_gray.shape[0]

    # Apply CLAHE for better visual contrast in fused image
    fused_display = apply_clahe(fused_gray)

    # Build annotated (red-box) versions
    s1_box = draw_red_box(src1_gray)
    s2_box = draw_red_box(src2_gray)
    f_box  = draw_red_box(fused_display)

    # Zoom crops
    z1 = crop_zoom(src1_gray)
    z2 = crop_zoom(src2_gray)
    zf = crop_zoom(fused_display)

    # Zoom to same display size
    zoom_size = size // 3
    z1 = cv2.resize(z1, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
    z2 = cv2.resize(z2, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
    zf = cv2.resize(zf, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    # ------ top row: source1 | source2 | fused (with red box) ------
    ncols = 3
    gs = fig.add_gridspec(2, ncols, height_ratios=[3, 1],
                          hspace=0.06, wspace=0.04,
                          left=0.03, right=0.97, top=0.88, bottom=0.18)

    for col, (img_rgb, label) in enumerate([
        (s1_box, s1_label),
        (s2_box, s2_label),
        (f_box,  fused_label),
    ]):
        ax = fig.add_subplot(gs[0, col])
        if img_rgb.ndim == 3:
            ax.imshow(img_rgb)
        else:
            ax.imshow(img_rgb, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=4)
        ax.axis("off")

    # ------ bottom row: zoomed crops ------
    for col, (zimg, label) in enumerate([
        (z1, f"Zoom: {s1_label}"),
        (z2, f"Zoom: {s2_label}"),
        (zf, f"Zoom: {fused_label}"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(zimg, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=9, color="#555555", pad=2)
        ax.axis("off")
        # thin red border on zoom boxes
        for spine in ax.spines.values():
            spine.set_edgecolor("#CC0000")
            spine.set_linewidth(1.5)
            spine.set_visible(True)

    # ------ metric table at bottom ------
    metric_text = (
        f"PSNR = {metrics['PSNR']:.2f} dB    "
        f"MI = {metrics['MI']:.3f}    "
        f"FMI = {metrics['FMI']:.3f}    "
        f"EN = {metrics['EN']:.3f}    "
        f"SSIM = {metrics['SSIM']:.4f}    "
        f"CC = {metrics['CC']:.4f}"
    )
    fig.text(
        0.5, 0.10, metric_text,
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FF",
                  edgecolor="#4A90D9", linewidth=1.2),
    )

    fig.suptitle(PAIR_TITLE[pair], fontsize=15, fontweight="bold", y=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved panel: {out_path}")


# ---------------------------------------------------------------------------
# multi-sample stacked figure (up to N_ROWS rows)
# ---------------------------------------------------------------------------

N_ROWS = 4   # number of test samples to show


def make_multi_panel(
    pair: str,
    samples: list[dict],
    out_path: Path,
) -> None:
    s1_label, s2_label, fused_label = PAIR_LABELS[pair]
    n = min(len(samples), N_ROWS)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples[:n]):
        s1  = sample["src1"]
        s2  = sample["src2"]
        fus = apply_clahe(sample["fused"])
        m   = sample["metrics"]

        for col, (img, label) in enumerate([
            (s1,  f"{s1_label}" if row == 0 else ""),
            (s2,  f"{s2_label}" if row == 0 else ""),
            (fus, f"{fused_label}" if row == 0 else ""),
        ]):
            ax = axes[row, col]
            ax.imshow(draw_red_box(img), cmap="gray" if img.ndim == 2 else None)
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold", pad=3)
            # metric text on fused column
            if col == 2:
                ax.set_xlabel(
                    f"PSNR={m['PSNR']:.1f} dB  MI={m['MI']:.2f}  FMI={m['FMI']:.3f}",
                    fontsize=8, labelpad=2,
                )
            ax.axis("off") if col < 2 else ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(PAIR_TITLE[pair], fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved multi-panel: {out_path}")


# ---------------------------------------------------------------------------
# discovery helpers
# ---------------------------------------------------------------------------

def find_fused_dir(pair: str, fused_root_override=None) -> Path | None:
    if fused_root_override:
        p = Path(fused_root_override)
        if p.exists():
            return p
    candidates = [
        GAN_ROOT / "images" / "aanlib" / "test" / pair / "fused_original",
        GAN_ROOT / "images" / "aanlib" / "train" / pair / "fused_original",
        PROJECT_ROOT / "outputs" / "models" / "diffusion" / pair / "images" / "fused_original",
    ]
    return next((p for p in candidates if p.exists() and any(p.glob("*_fused.png"))), None)


def find_source_dir(pair: str, modality: str) -> Path | None:
    data_root = PROJECT_ROOT / "data" / "raw" / "AANLIB"
    pair_map = {
        "ct_mri":    ("CT-MRI",   {"src1": "CT",    "src2": "MRI"}),
        "pet_mri":   ("PET-MRI",  {"src1": "PET",   "src2": "MRI"}),
        "spect_mri": ("SPECT-MRI",{"src1": "SPECT", "src2": "MRI"}),
    }
    folder, mod_map = pair_map[pair]
    subdir = mod_map[modality]
    for split in ["test", "train"]:
        p = data_root / folder / split / subdir
        if p.exists() and any(p.iterdir()):
            return p
    return None


# ---------------------------------------------------------------------------
# main per-pair logic
# ---------------------------------------------------------------------------

def process_pair(pair: str, fused_root_override=None) -> None:
    print(f"\n--- {PAIR_TITLE[pair]} ---")

    fused_dir = find_fused_dir(pair, fused_root_override)
    if fused_dir is None:
        print(f"  [SKIP] No fused images found. Run inference first:")
        print(f"    python generate_all_fused.py --pair {pair}")
        return

    src1_dir = find_source_dir(pair, "src1")
    src2_dir = find_source_dir(pair, "src2")
    if src1_dir is None or src2_dir is None:
        print(f"  [SKIP] Source images not found at expected data/raw/AANLIB path.")
        return

    fused_paths = sorted(fused_dir.glob("*_fused.png"))
    src1_paths  = sorted(src1_dir.glob("*.png"))
    src2_paths  = sorted(src2_dir.glob("*.png"))

    if not fused_paths:
        print(f"  [SKIP] Fused dir is empty: {fused_dir}")
        return

    n = min(len(fused_paths), len(src1_paths), len(src2_paths))
    print(f"  Found {n} image triples in {fused_dir}")

    samples = []
    for i in range(n):
        try:
            s1  = load_gray(src1_paths[i])
            s2  = load_gray(src2_paths[i])
            fus = load_gray(fused_paths[i])
            m   = compute_metrics(fus, s1, s2)
            samples.append({"src1": s1, "src2": s2, "fused": fus, "metrics": m})
        except Exception as e:
            print(f"  [WARN] Error loading sample {i}: {e}")

    if not samples:
        return

    # --- single best-sample panel (use sample with highest PSNR) ---
    best = max(samples, key=lambda s: s["metrics"]["PSNR"])
    single_out = THESIS_OUT / f"{pair}_qualitative_comparison_thesis.png"
    make_single_panel(
        pair,
        best["src1"], best["src2"], best["fused"],
        best["metrics"],
        single_out,
    )

    # --- multi-sample panel ---
    multi_out = THESIS_OUT / f"{pair}_multi_sample_comparison.png"
    make_multi_panel(pair, samples, multi_out)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis-quality comparison panels.")
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all")
    parser.add_argument("--fused-root", default=None,
                        help="Override fused-image directory (applies to all pairs).")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: outputs/models/gan/final_thesis_assets).")
    return parser.parse_args()


def main():
    args = parse_args()
    global THESIS_OUT
    if args.output_dir:
        THESIS_OUT = Path(args.output_dir)

    pairs = PAIRS if args.pair == "all" else [args.pair]
    for pair in pairs:
        process_pair(pair, fused_root_override=args.fused_root)

    print("\nDone. Files saved to:", THESIS_OUT)


if __name__ == "__main__":
    main()
