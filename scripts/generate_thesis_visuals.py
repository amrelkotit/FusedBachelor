#!/usr/bin/env python3
"""
PART 3 — Generate publication-quality thesis visuals (no GPU required).

Produces:
  3a. Training curves (loss + SSIM/PSNR dual-axis) per pair + combined overview
  3b. Metrics comparison bar chart (Diffusion vs GAN) + radar chart
  3c. Visual comparison panels (top-5 SSIM per pair) + mosaic
  3d. LaTeX booktabs results table

Run from the project root:
    python scripts/generate_thesis_visuals.py
"""

import sys
import subprocess

# Auto-install missing packages
for _pip, _imp in [
    ("scipy", "scipy"),
    ("scikit-image", "skimage"),
    ("opencv-python", "cv2"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
]:
    try:
        __import__(_imp)
    except ImportError:
        print(f"[INSTALL] pip install {_pip} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pip, "-q"])

import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_ROOT = PROJECT_ROOT / "outputs" / "models" / "diffusion"
GAN_ROOT       = PROJECT_ROOT / "outputs" / "models" / "gan"

GRAPH_DIR      = DIFFUSION_ROOT / "graphs"
METRICS_DIR    = DIFFUSION_ROOT / "metrics"
PANELS_DIR     = DIFFUSION_ROOT / "images" / "thesis_panels"
PP_ROOT        = DIFFUSION_ROOT / "images_postprocessed" / "aanlib"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PANELS_DIR.mkdir(parents=True, exist_ok=True)

PAIRS      = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}

FILES_CREATED = []


# ─────────────────────────────────────────────
# Style helper
# ─────────────────────────────────────────────
def _apply_style():
    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"]:
        try:
            plt.style.use(style)
            return
        except OSError:
            pass


# ─────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────
def _read_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"  [WARN] CSV not found: {path}")
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _parse_mean_std(s):
    """'0.337±0.04' or '0.337 +/- 0.038'  →  (0.337, 0.038)"""
    text = str(s)
    for sep in ("±", "+/-"):
        parts = text.split(sep)
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
    try:
        return float(s), 0.0
    except ValueError:
        return 0.0, 0.0


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────
# Aggregate metrics from detailed CSV
# ─────────────────────────────────────────────
def load_detailed_metrics():
    """Return dict: (pair, variant) → {metric: [values]}"""
    rows = _read_csv(METRICS_DIR / "diffusion_metrics_detailed.csv")
    data = defaultdict(lambda: defaultdict(list))
    metric_cols = ["SSIM", "PSNR", "MI", "EN", "SF", "AG", "CC", "FMI"]
    for row in rows:
        pair    = row.get("pair", "").strip()
        variant = row.get("variant", "").strip()
        for m in metric_cols:
            v = _safe_float(row.get(m))
            if v != 0.0 or row.get(m, "") == "0.0":
                data[(pair, variant)][m].append(v)
    return data


def stats(vals):
    """Return (mean, std) from a list of floats."""
    if not vals:
        return 0.0, 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


# ─────────────────────────────────────────────
# 3a — TRAINING CURVES
# ─────────────────────────────────────────────
def _savgol_smooth(y, window=11, polyorder=3):
    try:
        from scipy.signal import savgol_filter
        y = np.array(y, dtype=float)
        w = min(window, len(y))
        if w % 2 == 0:
            w -= 1
        if w < polyorder + 2:
            return y
        return savgol_filter(y, w, polyorder)
    except Exception:
        return np.array(y, dtype=float)


def _find_best_epoch(epochs, values):
    """Return (epoch, value) for the maximum value; None if empty."""
    valid = [(e, v) for e, v in zip(epochs, values) if not math.isnan(v)]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])


def part3a_training_curves():
    print("\n[3a] Training curves ...")
    _apply_style()

    fig_overview, axes_ov = plt.subplots(2, 3, figsize=(18, 8), dpi=300, sharex="col")
    fig_overview.suptitle("Training Overview — All Pairs", fontsize=14, fontweight="bold")

    csv_dir = DIFFUSION_ROOT / "graphs"

    for col, pair in enumerate(PAIRS):
        csv_path = csv_dir / f"{pair}_diffusion_training_history.csv"
        rows = _read_csv(csv_path)
        if not rows:
            print(f"  [WARN] No history CSV for {pair}")
            continue

        epochs = [int(r["epoch"]) for r in rows]
        train_loss = [_safe_float(r.get("train_total_loss")) for r in rows]
        val_loss   = [_safe_float(r.get("val_total_loss", float("nan"))) for r in rows]
        val_ssim   = [_safe_float(r.get("val_ssim", float("nan"))) for r in rows]
        val_psnr   = [_safe_float(r.get("val_psnr", float("nan"))) for r in rows]
        best_ssim_ep = _find_best_epoch(
            [e for e, v in zip(epochs, val_ssim) if not math.isnan(v)],
            [v for v in val_ssim if not math.isnan(v)]
        )

        label = PAIR_LABELS[pair]

        # ── Loss curve ───────────────────────────────────────────────
        fig_loss, ax_loss = plt.subplots(figsize=(9, 5), dpi=300)
        smoothed = _savgol_smooth(train_loss)
        ax_loss.plot(epochs, train_loss, color="#aac4e0", alpha=0.4, linewidth=1, label="Train loss (raw)")
        ax_loss.plot(epochs, smoothed,   color="#1f77b4", linewidth=2,    label="Train loss (smoothed)")
        val_ep  = [e for e, v in zip(epochs, val_loss) if not math.isnan(v)]
        val_vl  = [v for v in val_loss if not math.isnan(v)]
        if val_ep:
            ax_loss.plot(val_ep, val_vl, color="#ff7f0e", linewidth=1.5, linestyle="--", label="Val loss")
        # Mark best epoch
        if best_ssim_ep:
            be, bv = best_ssim_ep
            idx = epochs.index(be) if be in epochs else -1
            if 0 <= idx < len(smoothed):
                ax_loss.scatter([be], [smoothed[idx]], s=120, zorder=5, marker="*",
                                color="gold", edgecolors="black", linewidths=0.5, label=f"Best (ep {be})")
        ax_loss.set_title(f"{label} — Training Loss", fontsize=14, fontweight="bold")
        ax_loss.set_xlabel("Epoch", fontsize=12)
        ax_loss.set_ylabel("Loss", fontsize=12)
        ax_loss.tick_params(labelsize=10)
        ax_loss.legend(fontsize=9)
        out_loss = GRAPH_DIR / f"training_loss_curve_{pair}.png"
        fig_loss.tight_layout()
        fig_loss.savefig(out_loss, dpi=300)
        plt.close(fig_loss)
        FILES_CREATED.append(str(out_loss))

        # ── SSIM / PSNR dual-axis curve ───────────────────────────────
        ssim_ep = [e for e, v in zip(epochs, val_ssim) if not math.isnan(v)]
        ssim_vl = [v for v in val_ssim if not math.isnan(v)]
        psnr_ep = [e for e, v in zip(epochs, val_psnr) if not math.isnan(v)]
        psnr_vl = [v for v in val_psnr if not math.isnan(v)]

        fig_sv, ax_ssim = plt.subplots(figsize=(9, 5), dpi=300)
        ax_psnr = ax_ssim.twinx()
        if ssim_ep:
            ax_ssim.plot(ssim_ep, ssim_vl, color="#2ca02c", linewidth=2, label="Val SSIM")
        if psnr_ep:
            ax_psnr.plot(psnr_ep, psnr_vl, color="#9467bd", linewidth=2, linestyle="--", label="Val PSNR")
        if best_ssim_ep and ssim_ep:
            be = best_ssim_ep[0]
            if be in ssim_ep:
                bi = ssim_ep.index(be)
                ax_ssim.scatter([be], [ssim_vl[bi]], s=120, zorder=5, marker="*",
                                color="gold", edgecolors="black", linewidths=0.5, label=f"Best SSIM (ep {be})")
        ax_ssim.set_title(f"{label} — Validation SSIM & PSNR", fontsize=14, fontweight="bold")
        ax_ssim.set_xlabel("Epoch", fontsize=12)
        ax_ssim.set_ylabel("SSIM", fontsize=12, color="#2ca02c")
        ax_psnr.set_ylabel("PSNR (dB)", fontsize=12, color="#9467bd")
        ax_ssim.tick_params(labelsize=10)
        ax_psnr.tick_params(labelsize=10)
        lines1, labs1 = ax_ssim.get_legend_handles_labels()
        lines2, labs2 = ax_psnr.get_legend_handles_labels()
        ax_ssim.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
        out_sv = GRAPH_DIR / f"validation_ssim_curve_{pair}.png"
        fig_sv.tight_layout()
        fig_sv.savefig(out_sv, dpi=300)
        plt.close(fig_sv)
        FILES_CREATED.append(str(out_sv))

        # ── Fill overview grid ────────────────────────────────────────
        ax_ov_loss = axes_ov[0, col]
        ax_ov_ssim = axes_ov[1, col]
        ax_ov_loss.plot(epochs, smoothed, color="#1f77b4", linewidth=1.5)
        ax_ov_loss.set_title(f"{label}\nLoss", fontsize=10)
        if ssim_ep:
            ax_ov_ssim.plot(ssim_ep, ssim_vl, color="#2ca02c", linewidth=1.5)
        ax_ov_ssim.set_title(f"{label}\nSSIM", fontsize=10)
        ax_ov_ssim.set_xlabel("Epoch", fontsize=9)

    out_ov = GRAPH_DIR / "all_pairs_training_overview.png"
    fig_overview.tight_layout()
    fig_overview.savefig(out_ov, dpi=300)
    plt.close(fig_overview)
    FILES_CREATED.append(str(out_ov))
    print(f"  → Training curves saved to {GRAPH_DIR}")


# ─────────────────────────────────────────────
# 3b — METRICS BAR CHART + RADAR
# ─────────────────────────────────────────────
def part3b_metrics_charts():
    print("\n[3b] Metrics comparison charts ...")
    _apply_style()

    data = load_detailed_metrics()
    METRICS_PLOT = ["SSIM", "PSNR", "MI", "EN", "SF", "AG"]

    # Determine best diffusion variant per pair (highest SSIM mean)
    best_diff_variant = {}
    for pair in PAIRS:
        best_ssim, best_var = -1.0, "diffusion_fused_grayscale"
        for var in ["diffusion_fused_colored", "diffusion_fused_grayscale", "diffusion_fused_original"]:
            vals = data.get((pair, var), {}).get("SSIM", [])
            if vals and np.mean(vals) > best_ssim:
                best_ssim = np.mean(vals)
                best_var = var
        best_diff_variant[pair] = best_var

    # ── Bar chart ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    fig.suptitle("Diffusion vs. GAN — Quantitative Metrics Comparison",
                 fontsize=14, fontweight="bold")

    for idx, metric in enumerate(METRICS_PLOT):
        ax = axes[idx // 3][idx % 3]
        x = np.arange(len(PAIRS))
        w = 0.35
        gan_means, gan_stds   = [], []
        diff_means, diff_stds = [], []

        for pair in PAIRS:
            gm, gs = stats(data.get((pair, "gan"), {}).get(metric, []))
            dm, ds = stats(data.get((pair, best_diff_variant[pair]), {}).get(metric, []))
            gan_means.append(gm);  gan_stds.append(gs)
            diff_means.append(dm); diff_stds.append(ds)

        bars_gan  = ax.bar(x - w/2, gan_means,  w, yerr=gan_stds,  capsize=3,
                           color="steelblue",  label="GAN",       alpha=0.85)
        bars_diff = ax.bar(x + w/2, diff_means, w, yerr=diff_stds, capsize=3,
                           color="darkorange", label="Diffusion",  alpha=0.85)

        for bar, val in zip(bars_gan,  gan_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gan_stds)*0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        for bar, val in zip(bars_diff, diff_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(diff_stds)*0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PAIR_LABELS[p] for p in PAIRS], fontsize=9)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8)

    out_bar = GRAPH_DIR / "metrics_comparison_diffusion_vs_gan.png"
    fig.tight_layout()
    fig.savefig(out_bar, dpi=300)
    plt.close(fig)
    FILES_CREATED.append(str(out_bar))

    # ── Radar chart ──────────────────────────────────────────────────
    RADAR_METRICS = ["SSIM", "PSNR", "MI", "EN", "SF", "AG"]
    N = len(RADAR_METRICS)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # close polygon

    # Collect all values to normalise 0–1 per metric
    all_vals = {m: [] for m in RADAR_METRICS}
    for pair in PAIRS:
        for var in [f"gan", *[f"diffusion_{v}" for v in ["fused_colored","fused_grayscale","fused_original"]]]:
            for m in RADAR_METRICS:
                vals = data.get((pair, var), {}).get(m, [])
                if vals:
                    all_vals[m].extend(vals)

    metric_min = {m: min(all_vals[m]) if all_vals[m] else 0 for m in RADAR_METRICS}
    metric_max = {m: max(all_vals[m]) if all_vals[m] else 1 for m in RADAR_METRICS}

    def normalise(val, m):
        span = metric_max[m] - metric_min[m]
        return (val - metric_min[m]) / max(span, 1e-8)

    fig_r, ax_r = plt.subplots(figsize=(9, 9), dpi=300,
                                subplot_kw=dict(projection="polar"))

    colors_gan  = ["#1f77b4", "#aec7e8", "#c5dbef"]
    colors_diff = ["#ff7f0e", "#ffbb78", "#ffd8a8"]
    linestyles  = ["-", "--", ":"]

    for pi, pair in enumerate(PAIRS):
        label = PAIR_LABELS[pair]
        # GAN
        gm_vals = [stats(data.get((pair, "gan"), {}).get(m, []))[0] for m in RADAR_METRICS]
        gm_norm = [normalise(v, m) for v, m in zip(gm_vals, RADAR_METRICS)] + [normalise(gm_vals[0], RADAR_METRICS[0])]
        ax_r.plot(angles, gm_norm, color=colors_gan[pi], linestyle=linestyles[pi],
                  linewidth=2, label=f"GAN {label}")
        ax_r.fill(angles, gm_norm, color=colors_gan[pi], alpha=0.08)
        # Best diffusion
        dv = best_diff_variant[pair]
        dm_vals = [stats(data.get((pair, dv), {}).get(m, []))[0] for m in RADAR_METRICS]
        dm_norm = [normalise(v, m) for v, m in zip(dm_vals, RADAR_METRICS)] + [normalise(dm_vals[0], RADAR_METRICS[0])]
        ax_r.plot(angles, dm_norm, color=colors_diff[pi], linestyle=linestyles[pi],
                  linewidth=2, label=f"Diffusion {label}")
        ax_r.fill(angles, dm_norm, color=colors_diff[pi], alpha=0.08)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(RADAR_METRICS, fontsize=11)
    ax_r.set_yticklabels([])
    ax_r.set_title("Normalised Metric Radar — Diffusion vs GAN", fontsize=13,
                   fontweight="bold", pad=20)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8)

    out_radar = GRAPH_DIR / "metrics_radar_chart.png"
    fig_r.tight_layout()
    fig_r.savefig(out_radar, dpi=300)
    plt.close(fig_r)
    FILES_CREATED.append(str(out_radar))
    print(f"  → Bar chart → {out_bar}")
    print(f"  → Radar     → {out_radar}")


# ─────────────────────────────────────────────
# 3c — VISUAL COMPARISON PANELS
# ─────────────────────────────────────────────
def _apply_clahe(img_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def _load_gray_uint8(path):
    p = Path(path)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img


def _uint8_to_display(img):
    """Apply CLAHE and return float [0, 1]."""
    if img is None:
        return np.zeros((256, 256), dtype=np.float32)
    return _apply_clahe(img).astype(np.float32) / 255.0


def _best_diffusion_variant_dir(pair):
    """Return the best available diffusion image subdirectory name for *pair*."""
    root = DIFFUSION_ROOT / "images" / "aanlib" / pair / "test"
    for v in ["fused_colored", "fused_grayscale", "fused_original"]:
        d = root / v
        if d.exists() and any(d.iterdir()):
            return v
    return "fused_grayscale"


def _best_gan_dir(pair):
    root = GAN_ROOT / "images" / "aanlib" / "test" / pair
    for v in ["fused_color", "fused_original_enhanced", "fused_original"]:
        d = root / v
        if d.exists() and any(d.iterdir()):
            return d
    return None


def part3c_visual_panels():
    print("\n[3c] Visual comparison panels ...")
    _apply_style()

    detailed = _read_csv(METRICS_DIR / "diffusion_metrics_detailed.csv")

    # Check if postprocessed images exist
    pp_exists = PP_ROOT.exists()

    # Mosaic data: (pair, source, mri, gan, diff) images
    mosaic_images = {}

    for pair in PAIRS:
        label = PAIR_LABELS[pair]
        pair_dir = PANELS_DIR / pair
        pair_dir.mkdir(parents=True, exist_ok=True)

        diff_var_dir = _best_diffusion_variant_dir(pair)
        diff_var_key = f"diffusion_{diff_var_dir}"
        gan_img_dir  = _best_gan_dir(pair)

        # Select rows for this pair / best diffusion variant, filter non-zero SSIM
        cands = [
            r for r in detailed
            if r.get("pair") == pair and r.get("variant") == diff_var_key
            and _safe_float(r.get("SSIM")) > 0
        ]
        if not cands:
            print(f"  [WARN] No rows for {pair}/{diff_var_key}")
            continue

        cands.sort(key=lambda r: _safe_float(r.get("SSIM")), reverse=True)
        top5 = cands[:5]

        for rank, row in enumerate(top5):
            img_idx   = row.get("image", "").strip()
            ssim_val  = _safe_float(row.get("SSIM"))
            psnr_val  = _safe_float(row.get("PSNR"))
            src_path  = row.get("source_modality_path", "")
            mri_path  = row.get("mri_path", "")
            diff_path = row.get("fused_path", "")

            src_img  = _load_gray_uint8(src_path)
            mri_img  = _load_gray_uint8(mri_path)
            diff_img = _load_gray_uint8(diff_path)

            # GAN fused
            gan_img = None
            if gan_img_dir:
                for ext in ["png", "jpg"]:
                    gp = gan_img_dir / f"{img_idx}_fused.{ext}"
                    if gp.exists():
                        gan_img = _load_gray_uint8(gp)
                        break

            # Post-processed diffusion
            pp_path = PP_ROOT / pair / "test" / diff_var_dir / f"{img_idx}_fused.png"
            pp_img  = _load_gray_uint8(pp_path) if pp_exists else None

            panels = [
                (src_img,  f"Source\n(SSIM: {ssim_val:.2f} / PSNR: {psnr_val:.1f} dB)"),
                (mri_img,  "MRI Reference"),
                (gan_img,  "GAN Fused"),
                (diff_img, f"Diffusion\n(SSIM: {ssim_val:.2f} / PSNR: {psnr_val:.1f} dB)"),
            ]
            if pp_img is not None:
                panels.append((pp_img, "Post-Processed\nDiffusion"))

            n_panels = len(panels)
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5), dpi=200)
            if n_panels == 1:
                axes = [axes]

            for ax, (img, lbl) in zip(axes, panels):
                display = _uint8_to_display(img)
                ax.imshow(display, cmap="gray", vmin=0, vmax=1)
                ax.set_title(lbl, fontsize=8)
                ax.axis("off")

            fig.suptitle(f"{label} — Sample {img_idx}", fontsize=10, fontweight="bold")
            out_panel = pair_dir / f"best_sample_{rank:02d}.png"
            fig.tight_layout()
            fig.savefig(out_panel, dpi=200)
            plt.close(fig)
            FILES_CREATED.append(str(out_panel))

            # Store for mosaic (use rank-0 sample)
            if rank == 0:
                mosaic_images[pair] = (src_img, mri_img, gan_img, diff_img)

        print(f"  {pair}: {len(top5)} panels saved to {pair_dir}")

    # ── Mosaic 3×4 ───────────────────────────────────────────────────
    col_labels = ["Source Modality", "MRI Reference", "GAN Fused", "Diffusion Fused"]
    fig_m, axes_m = plt.subplots(len(PAIRS), 4, figsize=(16, 4 * len(PAIRS)), dpi=200)
    fig_m.suptitle("Visual Comparison Mosaic — All Pairs", fontsize=14, fontweight="bold")

    for row_idx, pair in enumerate(PAIRS):
        imgs = mosaic_images.get(pair, (None, None, None, None))
        for col_idx, (img, col_lbl) in enumerate(zip(imgs, col_labels)):
            ax = axes_m[row_idx][col_idx]
            display = _uint8_to_display(img)
            ax.imshow(display, cmap="gray", vmin=0, vmax=1)
            if row_idx == 0:
                ax.set_title(col_lbl, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(PAIR_LABELS[pair], fontsize=10, fontweight="bold")
            ax.axis("off")

    out_mosaic = PANELS_DIR / "mosaic_all_pairs.png"
    fig_m.tight_layout()
    fig_m.savefig(out_mosaic, dpi=200)
    plt.close(fig_m)
    FILES_CREATED.append(str(out_mosaic))
    print(f"  → Mosaic → {out_mosaic}")


# ─────────────────────────────────────────────
# 3d — LaTeX RESULTS TABLE
# ─────────────────────────────────────────────
def part3d_latex_table():
    print("\n[3d] LaTeX results table ...")

    data = load_detailed_metrics()
    METRICS_TEX = ["SSIM", "PSNR", "MI", "EN", "SF", "AG", "CC", "FMI"]

    def fmt(val, std, metric):
        if metric in ("SSIM", "CC", "FMI"):
            return f"{val:.2f} \\pm {std:.2f}"
        if metric == "PSNR":
            return f"{val:.1f} \\pm {std:.1f}"
        return f"{val:.3f} \\pm {std:.3f}"

    rows = []

    def add_method_rows(pair, gan_stats, diff_stats, pp_stats=None):
        """Add rows for one pair, bold better values."""
        label = PAIR_LABELS[pair]

        def bold_better(gs, ds, metric):
            # higher is better for all except nothing special; use max
            g_mean, g_std = gs
            d_mean, d_std = ds
            gf = fmt(g_mean, g_std, metric)
            df = fmt(d_mean, d_std, metric)
            if d_mean > g_mean:
                return gf, f"\\textbf{{{df}}}"
            elif g_mean > d_mean:
                return f"\\textbf{{{gf}}}", df
            return gf, df

        gan_row  = [f"\\multirow{{{'2' if pp_stats is None else '3'}}}{{*}}{{{label}}}", "GAN"]
        diff_row = ["", "Diffusion"]
        pp_row   = ["", "Post-Proc."] if pp_stats else None

        for m in METRICS_TEX:
            gs = gan_stats.get(m, (0.0, 0.0))
            ds = diff_stats.get(m, (0.0, 0.0))
            gf, df = bold_better(gs, ds, m)
            gan_row.append(gf)
            diff_row.append(df)
            if pp_stats:
                ps = pp_stats.get(m, (0.0, 0.0))
                pp_row.append(fmt(*ps, m))

        rows.append(" & ".join(gan_row)  + r" \\")
        rows.append(" & ".join(diff_row) + r" \\" + r"\midrule" if pp_stats is None else r" \\")
        if pp_stats:
            rows.append(" & ".join(pp_row) + r" \\" + r"\midrule")

    # Load postprocessed summary if it exists
    pp_summary_path = METRICS_DIR / "postprocessed_metrics_summary.csv"
    pp_summary = _read_csv(pp_summary_path)
    pp_lookup = {}
    for r in pp_summary:
        if r.get("type") == "postprocessed":
            key = (r.get("pair"), r.get("variant"), r.get("metric"))
            pp_lookup[key] = (_safe_float(r.get("mean")), _safe_float(r.get("std")))

    for pair in PAIRS:
        # GAN stats
        gan_stats  = {}
        diff_stats = {}
        best_var   = None
        best_ssim  = -1.0
        for v in ["diffusion_fused_colored", "diffusion_fused_grayscale", "diffusion_fused_original"]:
            vals = data.get((pair, v), {}).get("SSIM", [])
            if vals and np.mean(vals) > best_ssim:
                best_ssim = np.mean(vals)
                best_var  = v

        for m in METRICS_TEX:
            gan_stats[m]  = stats(data.get((pair, "gan"), {}).get(m, []))
            diff_stats[m] = stats(data.get((pair, best_var), {}).get(m, [])) if best_var else (0.0, 0.0)

        # Postprocessed stats from CSV
        pp_stats = None
        if pp_summary and best_var:
            pp_s = {}
            for m in METRICS_TEX:
                key = (pair, best_var, m)
                if key in pp_lookup:
                    pp_s[m] = pp_lookup[key]
            if pp_s:
                pp_stats = pp_s

        add_method_rows(pair, gan_stats, diff_stats, pp_stats)

    # Build .tex
    col_spec = "l l " + " ".join(["r"] * len(METRICS_TEX))
    header   = "Pair & Method & " + " & ".join(METRICS_TEX)

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Quantitative comparison of GAN and Diffusion fusion methods on the AANLIB dataset.",
        r"  Values are mean $\pm$ std. \textbf{Bold} indicates the better result per metric per pair.}",
        r"\label{tab:fusion_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    tex.extend(rows)
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]

    out_tex = METRICS_DIR / "results_table.tex"
    out_tex.write_text("\n".join(tex), encoding="utf-8")
    FILES_CREATED.append(str(out_tex))
    print(f"  → {out_tex}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("GENERATE THESIS VISUALS")
    print("=" * 70)

    part3a_training_curves()
    part3b_metrics_charts()
    part3c_visual_panels()
    part3d_latex_table()

    print("\n" + "=" * 70)
    print("ALL FILES CREATED / WRITTEN:")
    for f in FILES_CREATED:
        print(f"  {f}")
    print(f"\nTotal: {len(FILES_CREATED)} files")
    print("Done.")


if __name__ == "__main__":
    main()
