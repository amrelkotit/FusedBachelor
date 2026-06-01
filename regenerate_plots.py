"""
Regenerate all training plots from real CSV data.
Produces clean, publication-quality figures saved to outputs/models/plots/.
Run: python regenerate_plots.py
"""

import csv
import math
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent
DIFF_LOG = BASE / "outputs/models/diffusion/logs"
GAN_LOG  = BASE / "outputs/models/gan/graphs"
GAN_MET  = BASE / "outputs/models/gan/metrics"
OUT      = BASE / "outputs/models/plots"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS       = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT–MRI", "pet_mri": "PET–MRI", "spect_mri": "SPECT–MRI"}
COLORS      = {"ct_mri": "#1f77b4", "pet_mri": "#d62728", "spect_mri": "#2ca02c"}
TRAIN_C     = "#1976D2"
VAL_C       = "#D32F2F"

# ── helpers ────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def col_float(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError, TypeError):
            out.append(None)
    return out

def get_epochs(rows):
    return [int(r["epoch"]) for r in rows]

def best_epoch_from_rows(rows):
    for r in reversed(rows):
        try:
            return int(float(r["best_epoch"]))
        except (KeyError, ValueError, TypeError):
            pass
    return None

def best_val_ssim_from_rows(rows):
    for r in reversed(rows):
        try:
            v = float(r["best_val_ssim"])
            if v > 0:
                return v
        except (KeyError, ValueError, TypeError):
            pass
    return None

def style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

style()

# ══════════════════════════════════════════════════════════════════════════
# 1.  DIFFUSION — training loss + val SSIM per pair
# ══════════════════════════════════════════════════════════════════════════
print("1/6  Diffusion training curves …")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Diffusion Model — Training History (120 Epochs)", fontsize=14, fontweight="bold", y=1.01)

LOSS_STYLES = [
    ("train_loss",          "Total Loss",    TRAIN_C,  "-",  1.8, 1.0),
    ("l1_loss",             "L1",            "#FF9800", "--", 1.0, 0.7),
    ("ssim_loss",           "SSIM",          "#9C27B0", "--", 1.0, 0.7),
    ("grad_loss",           "Gradient",      "#4CAF50", "--", 1.0, 0.7),
    ("edge_loss",           "Edge",          "#00BCD4", ":",  0.9, 0.6),
    ("hf_loss",             "High-Freq",     "#FF5722", ":",  0.9, 0.6),
]

for ci, pair in enumerate(PAIRS):
    rows = read_csv(DIFF_LOG / f"{pair}_training_history.csv")
    ep   = get_epochs(rows)

    # Loss plot
    ax = axes[0][ci]
    for key, lbl, c, ls, lw, alpha in LOSS_STYLES:
        vals = col_float(rows, key)
        valid = [(e, v) for e, v in zip(ep, vals) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color=c, linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)
    ax.set_title(PAIR_LABELS[pair])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" if ci == 0 else "")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_xlim(1, max(ep))

    # SSIM plot
    tr_ssim  = col_float(rows, "train_ssim")
    vl_raw   = col_float(rows, "val_ssim")
    vl_ep    = [ep[i] for i, v in enumerate(vl_raw) if v is not None]
    vl_ssim  = [v for v in vl_raw if v is not None]

    ax2 = axes[1][ci]
    ax2.plot(ep,    tr_ssim, color=TRAIN_C, linewidth=1.5, label="Train SSIM")
    ax2.plot(vl_ep, vl_ssim, color=VAL_C,   linewidth=1.8,
             marker="o", markersize=2.5, label="Val SSIM")

    best_ep = best_epoch_from_rows(rows)
    if best_ep and best_ep in vl_ep:
        bi = vl_ep.index(best_ep)
        ax2.axvline(best_ep, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
        ax2.annotate(f"Best ep {best_ep}\n{vl_ssim[bi]:.3f}",
                     xy=(best_ep, vl_ssim[bi]),
                     xytext=(best_ep + 4, vl_ssim[bi] - 0.10),
                     fontsize=7, color="gray",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))

    final = vl_ssim[-1] if vl_ssim else 0
    ax2.set_title(f"{PAIR_LABELS[pair]}  (final val SSIM = {final:.3f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM" if ci == 0 else "")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right")
    ax2.set_xlim(1, max(ep))

plt.tight_layout()
fp = OUT / "diffusion_training_curves.png"
plt.savefig(fp, bbox_inches="tight", dpi=150)
plt.close()
print(f"   saved → {fp}")

# ══════════════════════════════════════════════════════════════════════════
# 2.  GAN — training loss + val SSIM per pair
# ══════════════════════════════════════════════════════════════════════════
print("2/6  GAN training curves …")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("GAN Model — Training History", fontsize=14, fontweight="bold", y=1.01)

for ci, pair in enumerate(PAIRS):
    rows = read_csv(GAN_LOG / f"{pair}_training_history.csv")
    ep   = get_epochs(rows)

    ax = axes[0][ci]
    for key, lbl, c, ls, lw, alpha in [
        ("train_total_loss",  "Train Total",  TRAIN_C,  "-",  1.8, 1.0),
        ("val_total_loss",    "Val Total",    VAL_C,    "-",  1.8, 1.0),
        ("train_fusion_loss", "Train Fusion", "#FF9800", "--", 1.0, 0.7),
        ("val_fusion_loss",   "Val Fusion",   "#FF5722", "--", 1.0, 0.7),
        ("train_gan_loss",    "Gen Loss",     "#9C27B0", ":",  1.0, 0.7),
    ]:
        vals = col_float(rows, key)
        valid = [(e, v) for e, v in zip(ep, vals) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color=c, linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)

    best_ep   = best_epoch_from_rows(rows)
    best_ssim = best_val_ssim_from_rows(rows)
    n_ep = len(rows)
    ax.set_title(f"{PAIR_LABELS[pair]}  ({n_ep} epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" if ci == 0 else "")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_xlim(1, max(ep))

    ax2 = axes[1][ci]
    tr_ssim = col_float(rows, "train_ssim")
    vl_ssim = col_float(rows, "val_ssim")
    ax2.plot(ep, tr_ssim, color=TRAIN_C, linewidth=1.5, label="Train SSIM")
    ax2.plot(ep, vl_ssim, color=VAL_C,   linewidth=1.5, label="Val SSIM")

    if best_ep:
        ax2.axvline(best_ep, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
        y_ann = best_ssim or 0.5
        ax2.annotate(f"Best ep {best_ep}\n{y_ann:.3f}",
                     xy=(best_ep, y_ann),
                     xytext=(best_ep + 1, y_ann - 0.13),
                     fontsize=7, color="gray",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))

    title = (f"{PAIR_LABELS[pair]}  (best val SSIM = {best_ssim:.3f})"
             if best_ssim else PAIR_LABELS[pair])
    ax2.set_title(title)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM" if ci == 0 else "")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right")
    ax2.set_xlim(1, max(ep))

plt.tight_layout()
fp = OUT / "gan_training_curves.png"
plt.savefig(fp, bbox_inches="tight", dpi=150)
plt.close()
print(f"   saved → {fp}")

# ══════════════════════════════════════════════════════════════════════════
# 3.  Side-by-side val SSIM comparison
# ══════════════════════════════════════════════════════════════════════════
print("3/6  Model comparison SSIM …")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Val SSIM Comparison: GAN vs Diffusion", fontsize=13, fontweight="bold")

for ci, pair in enumerate(PAIRS):
    ax = axes[ci]

    gan_rows  = read_csv(GAN_LOG  / f"{pair}_training_history.csv")
    diff_rows = read_csv(DIFF_LOG / f"{pair}_training_history.csv")

    gan_ep    = get_epochs(gan_rows)
    gan_ssim  = col_float(gan_rows, "val_ssim")

    diff_ep_all  = get_epochs(diff_rows)
    diff_vl_raw  = col_float(diff_rows, "val_ssim")
    diff_ep      = [diff_ep_all[i] for i, v in enumerate(diff_vl_raw) if v is not None]
    diff_ssim    = [v for v in diff_vl_raw if v is not None]

    ax.plot(gan_ep,   gan_ssim,  color="#E53935", linewidth=1.6, label="GAN")
    ax.plot(diff_ep,  diff_ssim, color="#1E88E5", linewidth=1.6, label="Diffusion")

    gan_best  = max((v for v in gan_ssim  if v is not None), default=0)
    diff_best = max((v for v in diff_ssim if v is not None), default=0)

    ax.axhline(gan_best,  color="#E53935", lw=0.9, linestyle="--", alpha=0.45)
    ax.axhline(diff_best, color="#1E88E5", lw=0.9, linestyle="--", alpha=0.45)

    ax.text(0.98, gan_best  + 0.01, f"GAN best {gan_best:.3f}",
            ha="right", va="bottom", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#E53935")
    ax.text(0.98, diff_best + 0.01, f"Diff best {diff_best:.3f}",
            ha="right", va="bottom", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#1E88E5")

    ax.set_title(PAIR_LABELS[pair])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val SSIM" if ci == 0 else "")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")

plt.tight_layout()
fp = OUT / "model_comparison_ssim.png"
plt.savefig(fp, bbox_inches="tight", dpi=150)
plt.close()
print(f"   saved → {fp}")

# ══════════════════════════════════════════════════════════════════════════
# 4.  Test metrics bar chart (with PSNR note)
# ══════════════════════════════════════════════════════════════════════════
print("4/6  Test metrics bar chart …")

OFFSET = 20 * math.log10(255)   # ~48.13 dB

gan_test = {}
for pair in PAIRS:
    rows = read_csv(GAN_MET / f"{pair}_metrics.csv")
    rows = [r for r in rows if r.get("SSIM") and "/" not in str(r["SSIM"])]
    if not rows:
        continue
    ssims  = [float(r["SSIM"]) for r in rows]
    psnrs  = [float(r["PSNR"]) - OFFSET for r in rows]  # convert to standard
    mis    = [float(r["MI"])   for r in rows]
    ccs    = [float(r["CC"])   for r in rows]
    fmis   = [float(r["FMI"])  for r in rows]
    ags    = [float(r["AG"])   for r in rows]
    gan_test[pair] = {
        "SSIM":     (statistics.mean(ssims), statistics.stdev(ssims) if len(ssims) > 1 else 0),
        "PSNR":     (statistics.mean(psnrs), statistics.stdev(psnrs) if len(psnrs) > 1 else 0),
        "PSNR_inf": statistics.mean(psnrs) + OFFSET,
        "MI":       statistics.mean(mis),
        "CC":       statistics.mean(ccs),
        "FMI":      statistics.mean(fmis),
        "AG":       statistics.mean(ags),
    }

metrics_to_plot = [
    ("SSIM",  "SSIM",              0.0, 1.0,  True),
    ("PSNR",  "PSNR (dB, std)",    None, None, True),
    ("MI",    "Mutual Information", 0.0, None, False),
    ("CC",    "Cross-Correlation",  0.0, 1.0,  False),
    ("FMI",   "Feature MI",         0.0, 1.0,  False),
    ("AG",    "Avg Gradient",       0.0, None, False),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("GAN Model — Test Set Metrics  (standard data_range=1 PSNR)", fontsize=13, fontweight="bold")

x = np.arange(len(PAIRS))

for idx, (key, title, ymin, ymax, has_err) in enumerate(metrics_to_plot):
    ax = axes[idx // 3][idx % 3]
    vals  = [gan_test[p][key][0] if isinstance(gan_test[p][key], tuple) else gan_test[p][key]
             for p in PAIRS]
    errs  = [gan_test[p][key][1] if isinstance(gan_test[p][key], tuple) else 0
             for p in PAIRS]

    if has_err:
        bars = ax.bar(x, vals, 0.55, yerr=errs, capsize=6, error_kw={"linewidth": 1.5},
                      color=[COLORS[p] for p in PAIRS], alpha=0.85, edgecolor="white")
    else:
        bars = ax.bar(x, vals, 0.55,
                      color=[COLORS[p] for p in PAIRS], alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(vals) * 0.01),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    if key == "PSNR":
        for i, (bar, p) in enumerate(zip(bars, PAIRS)):
            inf_val = gan_test[p]["PSNR_inf"]
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5,
                    f"({inf_val:.1f}†)", ha="center", va="bottom", fontsize=7, color="gray")
        ax.text(0.5, -0.14, "†: reported PSNR using data_range=255 (+48.1 dB over standard)",
                transform=ax.transAxes, ha="center", fontsize=7, color="gray")

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in PAIRS], fontsize=9)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

plt.tight_layout()
fp = OUT / "gan_test_metrics.png"
plt.savefig(fp, bbox_inches="tight", dpi=150)
plt.close()
print(f"   saved → {fp}")

# ══════════════════════════════════════════════════════════════════════════
# 5.  Diffusion stacked loss breakdown
# ══════════════════════════════════════════════════════════════════════════
print("5/6  Diffusion loss breakdown …")

COMPONENTS = [
    ("noise_loss",          "Noise",          "#E53935"),
    ("l1_loss",             "L1",             "#FB8C00"),
    ("ssim_loss",           "SSIM",           "#8E24AA"),
    ("grad_loss",           "Gradient",       "#43A047"),
    ("edge_loss",           "Edge",           "#00ACC1"),
    ("hf_loss",             "High-Freq",      "#F4511E"),
    ("laplacian_loss",      "Laplacian",      "#6D4C41"),
    ("ms_ssim_loss",        "MS-SSIM",        "#1E88E5"),
    ("local_contrast_loss", "Local Contrast", "#546E7A"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Diffusion Model — Stacked Loss Components", fontsize=13, fontweight="bold")

for ci, pair in enumerate(PAIRS):
    rows   = read_csv(DIFF_LOG / f"{pair}_training_history.csv")
    ep     = np.array(get_epochs(rows), dtype=float)
    ax     = axes[ci]
    bottom = np.zeros(len(ep))

    for key, lbl, c in COMPONENTS:
        vals = np.array([float(r.get(key) or 0) for r in rows])
        ax.fill_between(ep, bottom, bottom + vals, alpha=0.8, color=c, label=lbl)
        bottom += vals

    total = np.array(col_float(rows, "train_loss"), dtype=float)
    ax.plot(ep, total, color="black", linewidth=1.2, linestyle="-", alpha=0.6, label="Total (line)")

    ax.set_title(PAIR_LABELS[pair])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (stacked)" if ci == 0 else "")
    ax.set_xlim(1, max(ep))
    if ci == 2:
        ax.legend(loc="upper right", fontsize=7, ncol=2)

plt.tight_layout()
fp = OUT / "diffusion_loss_breakdown.png"
plt.savefig(fp, bbox_inches="tight", dpi=150)
plt.close()
print(f"   saved → {fp}")

# ══════════════════════════════════════════════════════════════════════════
# 6.  GAN discriminator stability
# ══════════════════════════════════════════════════════════════════════════
print("6/6  GAN discriminator stability …")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("GAN Discriminator Loss Stability", fontsize=13, fontweight="bold")

for ci, pair in enumerate(PAIRS):
    rows = read_csv(GAN_LOG / f"{pair}_training_history.csv")
    ep   = get_epochs(rows)
    ax   = axes[ci]

    for key, lbl, c, ls, lw in [
        ("train_d1_loss", "D1 Train", "#1565C0", "-",  1.3),
        ("val_d1_loss",   "D1 Val",   "#1565C0", "--", 1.3),
        ("train_d2_loss", "D2 Train", "#B71C1C", "-",  1.3),
        ("val_d2_loss",   "D2 Val",   "#B71C1C", "--", 1.3),
        ("train_gan_loss","Gen Loss", "#2E7D32", ":",  1.2),
    ]:
        vals  = col_float(rows, key)
        valid = [(e, v) for e, v in zip(ep, vals) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color=c, linestyle=ls, linewidth=lw, label=lbl, alpha=0.88)

    best_ep = best_epoch_from_rows(rows)
    if best_ep:
        ax.axvline(best_ep, color="gray", linestyle=":", lw=1.0, alpha=0.6,
                   label=f"Best ep {best_ep}")

    d1_vals = [v for v in col_float(rows, "val_d1_loss") if v is not None]
    if d1_vals and max(d1_vals) > 0.8:
        ax.axhline(0.8, color="orange", lw=0.8, linestyle=":", alpha=0.55)
        ax.text(1, 0.82, "Instability zone", color="orange", fontsize=7)

    ax.set_title(f"{PAIR_LABELS[pair]}  ({len(rows)} epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" if ci == 0 else "")
    ax.legend(fontsize=7)
    ax.set_xlim(1, max(ep))

plt.tight_layout()
fp = OUT / "gan_discriminator_stability.png"
plt.savefig(fp, bbox_inches="tight", dpi=150)
plt.close()
print(f"   saved → {fp}")

print("\n✓  All 6 plots written to", OUT)
