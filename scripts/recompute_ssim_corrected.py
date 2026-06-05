#!/usr/bin/env python3
"""
Diagnostic + corrected SSIM report for all three modality pairs.

Runs on saved outputs only — no retraining.

For each pair this script:
  1. Prints the exact shape / dtype / value-range of every array passed to ssim()
  2. Asserts that no metric is computed on a 3-channel (RGB) tensor
  3. Prints corrected mean SSIM using single-channel pre-colormap images
     consistently across GAN and Diffusion outputs
"""

import csv
import sys
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import (
    AANLIB_ROOT,
    PairedMedicalImageDataset,
    aanlib_split_root,
)
from src.diffusion.utils import read_gray_tensor
from src.evaluation.metrics import ssim

# ── paths ─────────────────────────────────────────────────────────────────────
OUT      = PROJECT_ROOT / "outputs"
GAN_IMG  = OUT / "models/gan/images/aanlib/test"
DIFF_IMG = OUT / "models/diffusion/images/aanlib"

PAIRS = ["ct_mri", "pet_mri", "spect_mri"]

# Grayscale-only diffusion variants (pre-colormap intensity maps)
DIFF_GRAY_VARIANTS = ["fused_grayscale", "fused_original"]

# GAN grayscale fused directories
GAN_GRAY_DIR = {
    "ct_mri":    "fused_original",
    "pet_mri":   "fused_original",
    "spect_mri": "fused_original",
}


def read_tensor(path, image_size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)


def sanity_line(tag, t):
    c = t.shape[1] if t.ndim == 4 else "?"
    dtype = str(t.dtype).replace("torch.", "")
    assert c == 1, f"ASSERTION FAILED — {tag} has {c} channels (must be 1, got shape {tuple(t.shape)})"
    return (f"  [{tag}]  shape={tuple(t.shape)}  dtype={dtype}"
            f"  range=[{t.min():.4f}, {t.max():.4f}]  channels=1 OK")


def mean_ssim(fused_list, src1_list, src2_list):
    scores = []
    for f, s1, s2 in zip(fused_list, src1_list, src2_list):
        ssim_s1 = ssim(f, s1)
        ssim_s2 = ssim(f, s2)
        scores.append(0.5 * (ssim_s1 + ssim_s2))
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def main():
    image_size = 256

    print("=" * 68)
    print(" SSIM diagnostic — single-channel pre-colormap inputs only")
    print("=" * 68)

    results = {}   # pair -> {variant: mean_ssim}

    for pair in PAIRS:
        print(f"\n{'-'*68}")
        print(f" Pair: {pair.upper()}")
        print(f"{'-'*68}")

        dataset_dir = aanlib_split_root(AANLIB_ROOT, pair, "test")
        dataset = PairedMedicalImageDataset(
            dataset_dir, image_size=image_size, pair=pair, strict=True,
        )

        src1_list, src2_list = [], []
        for sample in dataset:
            src1_list.append(sample["source1"].unsqueeze(0))
            src2_list.append(sample["source2"].unsqueeze(0))

        # Print sanity for first sample
        sample0 = dataset[0]
        s1_0 = sample0["source1"].unsqueeze(0)
        s2_0 = sample0["source2"].unsqueeze(0)
        print("\n Inputs to ssim() — sample 0:")
        print(sanity_line("source1 (modality)", s1_0))
        print(sanity_line("source2 (MRI)     ", s2_0))

        results[pair] = {}

        # ── GAN ───────────────────────────────────────────────────────────────
        gan_dir = GAN_IMG / pair / GAN_GRAY_DIR[pair]
        gan_imgs = sorted(gan_dir.glob("*_fused.png")) if gan_dir.exists() else []
        if gan_imgs:
            fused0 = read_tensor(gan_imgs[0], image_size)
            print(sanity_line("GAN fused_original", fused0))

            fused_list = [t for p in gan_imgs[:len(dataset)]
                          if (t := read_tensor(p, image_size)) is not None]
            n = min(len(fused_list), len(src1_list))
            results[pair]["GAN"] = mean_ssim(fused_list[:n], src1_list[:n], src2_list[:n])
        else:
            print("  [GAN fused_original]  NOT FOUND")

        # ── Diffusion (grayscale variants only) ───────────────────────────────
        for variant in DIFF_GRAY_VARIANTS:
            d = DIFF_IMG / pair / "test" / variant
            imgs = sorted(d.glob("*_fused.png")) if d.exists() else []
            if not imgs:
                continue
            fused0 = read_gray_tensor(imgs[0], image_size)
            # Confirm no 3-channel image slips through
            print(sanity_line(f"Diffusion {variant}", fused0))

            fused_list = [read_gray_tensor(p, image_size) for p in imgs[:len(dataset)]]
            n = min(len(fused_list), len(src1_list))
            results[pair][f"Diffusion_{variant}"] = mean_ssim(
                fused_list[:n], src1_list[:n], src2_list[:n]
            )

        # ── Explicitly confirm colored variant would be WRONG ─────────────────
        colored_d = DIFF_IMG / pair / "test" / "fused_colored"
        if colored_d.exists() and any(colored_d.glob("*_fused.png")):
            raw = cv2.imread(str(sorted(colored_d.glob("*_fused.png"))[0]),
                             cv2.IMREAD_UNCHANGED)
            channels = raw.shape[2] if raw.ndim == 3 else 1
            print(f"\n  [fused_colored raw PNG]  shape={raw.shape}  channels={channels}"
                  f"  — EXCLUDED from metrics (jet-colormap RGB → luminance is"
                  f" non-monotonic; would inflate SSIM)")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(" Corrected mean SSIM (single-channel grayscale inputs only)")
    print(f"{'='*68}")
    header = f"{'Pair':<15} {'Variant':<30} {'SSIM':>8}"
    print(header)
    print("-" * len(header))
    for pair in PAIRS:
        for variant, val in results[pair].items():
            print(f"{pair:<15} {variant:<30} {val:>8.4f}")
    print(f"{'='*68}")
    print("\nAll sanity assertions passed — no metric was computed on an RGB array.")


if __name__ == "__main__":
    main()
