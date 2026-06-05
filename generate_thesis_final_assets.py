#!/usr/bin/env python3
"""
Generate final thesis assets into outputs/thesis_final_assets/:
  01_comparisons/   – per-pair comparison panels (source | MRI | GAN | Diffusion)
  02_best_gan/      – best-3 GAN fused images per modality pair
  03_best_diffusion/– best-3 Diffusion fused images per modality pair
  04_training_curves/ – 7 training-curve figures
"""

import os, json, csv, shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT   = Path(r"E:\El Gam3a\My bachelor\Finale-Fused\FusedBachelor")
OUT    = ROOT / "outputs"

THESIS = OUT / "thesis_final_assets"
CMP_D  = THESIS / "01_comparisons"
BGAN_D = THESIS / "02_best_gan"
BDIF_D = THESIS / "03_best_diffusion"
CRV_D  = THESIS / "04_training_curves"
for d in [THESIS, CMP_D, BGAN_D, BDIF_D, CRV_D]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Plot style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

COLORS = {"ct_mri": "#1f77b4", "pet_mri": "#ff7f0e", "spect_mri": "#2ca02c"}
LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def epochs(hist):
    return [e["epoch"] for e in hist]

def field(hist, key):
    return [e.get(key) for e in hist]

def valid_pairs(ep, vals):
    """Return (ep_list, val_list) keeping only non-None entries."""
    pairs = [(e, v) for e, v in zip(ep, vals) if v is not None]
    if not pairs:
        return [], []
    e_out, v_out = zip(*pairs)
    return list(e_out), list(v_out)

def load_img_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def robust_normalize(img_arr):
    """Clip [1 %, 99 %] then rescale to uint8 [0, 255].
    Removes outlier hot-pixels (skull ring, reconstruction artefacts) before
    display so that GAN and Diffusion panels share an identical intensity scale.
    """
    img_f = img_arr.astype(np.float32)
    lo, hi = np.percentile(img_f, [1, 99])
    span = hi - lo
    if span < 1e-6:
        return np.zeros_like(img_arr, dtype=np.uint8)
    img_f = np.clip(img_f, lo, hi)
    return ((img_f - lo) / span * 255).astype(np.uint8)

def is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def top3_by_ssim(rows):
    """Indices of top-3 plain-float rows by SSIM descending (skips summary rows)."""
    valid = [(i, float(r["SSIM"])) for i, r in enumerate(rows) if is_float(r.get("SSIM", ""))]
    ranked = sorted(valid, key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:3]]

# ──────────────────────────────────────────────────────────────────────────────
#  Load training histories
# ──────────────────────────────────────────────────────────────────────────────
print("Loading training histories ...")

GAN_LOG  = OUT / "models/gan/logs"
DIFF_LOG = OUT / "models/diffusion"

gan_hist = {
    "ct_mri":    load_json(GAN_LOG / "ct_mri_training_history.json"),
    "pet_mri":   load_json(GAN_LOG / "pet_mri_training_history.json"),
    "spect_mri": load_json(GAN_LOG / "spect_mri_training_history.json"),
}

diff_hist = {
    "ct_mri":    load_json(DIFF_LOG / "aanlib_ct_mri/logs/ct_mri_diffusion_training_history.json"),
    "pet_mri":   load_json(DIFF_LOG / "aanlib_pet_mri/logs/pet_mri_diffusion_training_history.json"),
    "spect_mri": load_json(DIFF_LOG / "aanlib_spect_mri/logs/spect_mri_diffusion_training_history.json"),
}

# ──────────────────────────────────────────────────────────────────────────────
#  Load metrics
# ──────────────────────────────────────────────────────────────────────────────
print("Loading metrics CSVs ...")

GAN_MET  = OUT / "models/gan/metrics"
DIFF_MET = OUT / "models/diffusion/metrics"

gan_metrics = {
    "ct_mri":    load_csv(GAN_MET / "ct_mri_metrics.csv"),
    "pet_mri":   load_csv(GAN_MET / "pet_mri_metrics.csv"),
    "spect_mri": load_csv(GAN_MET / "spect_mri_metrics.csv"),
}

diff_detailed = load_csv(DIFF_MET / "diffusion_metrics_detailed.csv")

# ──────────────────────────────────────────────────────────────────────────────
#  Image directories
# ──────────────────────────────────────────────────────────────────────────────
GAN_IMG  = OUT / "models/gan/images/aanlib/test"
DIFF_IMG = OUT / "models/diffusion/images/aanlib"

PAIR_CFG = {
    "ct_mri": {
        "label":      "CT–MRI",
        "src1_label": "CT",
        # HQ fused for CT; fall back to fused_original_enhanced if absent
        "gan_fused":  GAN_IMG / "ct_mri/fused_hq",
        "diff_fused": DIFF_IMG / "ct_mri/test/fused_grayscale",
    },
    "pet_mri": {
        "label":      "PET–MRI",
        "src1_label": "PET",
        "gan_fused":  GAN_IMG / "pet_mri/fused_color",
        "diff_fused": DIFF_IMG / "pet_mri/test/fused_colored",
    },
    "spect_mri": {
        "label":      "SPECT–MRI",
        "src1_label": "SPECT",
        "gan_fused":  GAN_IMG / "spect_mri/fused_color",
        "diff_fused": DIFF_IMG / "spect_mri/test/fused_colored",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION A – TRAINING CURVES (7 figures)
# ══════════════════════════════════════════════════════════════════════════════

# ── A1. GAN training loss curves – all three pairs ────────────────────────────
def plot_gan_loss_curves():
    fig, ax = plt.subplots(figsize=(10, 5))
    for pair in ["ct_mri", "pet_mri", "spect_mri"]:
        h = gan_hist[pair]
        ax.plot(epochs(h), field(h, "train_total_loss"),
                color=COLORS[pair], lw=2, label=LABELS[pair])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("GAN Training Loss Curves — All Modality Pairs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = CRV_D / "A1_gan_training_loss_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── A2. GAN training curves – loss/SSIM/PSNR/MI on one figure ────────────────
def plot_gan_training_curves():
    metrics = [
        ("train_total_loss", "Total Loss"),
        ("train_ssim",       "SSIM"),
        ("train_psnr",       "PSNR (dB)"),
        ("train_mi",         "Mutual Information"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (key, title) in zip(axes.flatten(), metrics):
        for pair in ["ct_mri", "pet_mri", "spect_mri"]:
            h = gan_hist[pair]
            ep, vals = valid_pairs(epochs(h), field(h, key))
            ax.plot(ep, vals, color=COLORS[pair], lw=2, label=LABELS[pair])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"GAN Training — {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("GAN Training Curves — All Modality Pairs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = CRV_D / "A2_gan_training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── A3. GAN train-vs-val – loss / SSIM / PSNR / MI ───────────────────────────
def plot_gan_train_val_curves():
    pairs_list = ["ct_mri", "pet_mri", "spect_mri"]
    metric_defs = [
        ("train_total_loss", "val_total_loss", "Total Loss"),
        ("train_ssim",       "val_ssim",       "SSIM"),
        ("train_psnr",       "val_psnr",       "PSNR (dB)"),
        ("train_mi",         "val_mi",         "Mutual Information"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    for col, pair in enumerate(pairs_list):
        h = gan_hist[pair]
        ep = epochs(h)
        for row, (t_key, v_key, label) in enumerate(metric_defs):
            ax = axes[row][col]
            t_ep, t_vals = valid_pairs(ep, field(h, t_key))
            v_ep, v_vals = valid_pairs(ep, field(h, v_key))
            ax.plot(t_ep, t_vals, color="#1a6faf", lw=1.8, label="Train")
            ax.plot(v_ep, v_vals, color="#d94f3d", lw=1.8, ls="--", label="Val")
            if row == 0:
                ax.set_title(LABELS[pair], fontweight="bold", fontsize=13)
            if col == 0:
                ax.set_ylabel(label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")
    fig.suptitle("GAN Training vs. Validation Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = CRV_D / "A3_gan_train_val_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── A4. Diffusion training loss curves – all three pairs ─────────────────────
def plot_diff_loss_curves():
    fig, ax = plt.subplots(figsize=(10, 5))
    for pair in ["ct_mri", "pet_mri", "spect_mri"]:
        h = diff_hist[pair]
        ax.plot(epochs(h), field(h, "train_total_loss"),
                color=COLORS[pair], lw=2, label=LABELS[pair])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Diffusion Training Loss Curves — All Modality Pairs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = CRV_D / "A4_diffusion_training_loss_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── A5. Diffusion validation SSIM curves – all three pairs ───────────────────
def plot_diff_val_ssim():
    fig, ax = plt.subplots(figsize=(10, 5))
    for pair in ["ct_mri", "pet_mri", "spect_mri"]:
        h = diff_hist[pair]
        ep, vals = valid_pairs(epochs(h), field(h, "val_ssim"))
        ax.plot(ep, vals, color=COLORS[pair], lw=2,
                marker="o", markersize=4, label=LABELS[pair])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation SSIM")
    ax.set_title("Diffusion Validation Structural Similarity Curves — All Modality Pairs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = CRV_D / "A5_diffusion_validation_ssim_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── A6. Diffusion training curves – loss/SSIM/PSNR/MI ────────────────────────
def plot_diff_training_curves():
    metrics = [
        ("train_total_loss", "Total Loss"),
        ("train_ssim",       "SSIM"),
        ("train_psnr",       "PSNR (dB)"),
        ("train_mi",         "Mutual Information"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (key, title) in zip(axes.flatten(), metrics):
        for pair in ["ct_mri", "pet_mri", "spect_mri"]:
            h = diff_hist[pair]
            ep, vals = valid_pairs(epochs(h), field(h, key))
            ax.plot(ep, vals, color=COLORS[pair], lw=2, label=LABELS[pair])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Diffusion Training — {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Diffusion Training Curves — All Modality Pairs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = CRV_D / "A6_diffusion_training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── A7. Diffusion train-vs-val – loss / SSIM / PSNR / MI ─────────────────────
def plot_diff_train_val_curves():
    pairs_list = ["ct_mri", "pet_mri", "spect_mri"]
    metric_defs = [
        ("train_total_loss", "val_total_loss", "Total Loss"),
        ("train_ssim",       "val_ssim",       "SSIM"),
        ("train_psnr",       "val_psnr",       "PSNR (dB)"),
        ("train_mi",         "val_mi",         "Mutual Information"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    for col, pair in enumerate(pairs_list):
        h = diff_hist[pair]
        ep = epochs(h)
        for row, (t_key, v_key, label) in enumerate(metric_defs):
            ax = axes[row][col]
            t_ep, t_vals = valid_pairs(ep, field(h, t_key))
            v_ep, v_vals = valid_pairs(ep, field(h, v_key))
            ax.plot(t_ep, t_vals, color="#1a6faf", lw=1.8, label="Train")
            if v_ep:
                ax.plot(v_ep, v_vals, color="#d94f3d", lw=1.8, ls="--",
                        marker="o", markersize=3, label="Val")
            if row == 0:
                ax.set_title(LABELS[pair], fontweight="bold", fontsize=13)
            if col == 0:
                ax.set_ylabel(label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")
    fig.suptitle("Diffusion Training vs. Validation Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = CRV_D / "A7_diffusion_train_val_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION B – COMPARISON PANELS
#  Layout: 3 rows (best-3 samples by SSIM) × 4 cols (src1 | MRI | GAN | Diff)
# ══════════════════════════════════════════════════════════════════════════════
def make_comparison_panels():
    for pair, cfg in PAIR_CFG.items():
        rows = gan_metrics[pair]
        top3 = top3_by_ssim(rows)

        fig, axes = plt.subplots(3, 4, figsize=(18, 14))
        col_titles = [cfg["src1_label"], "MRI", "GAN Fused", "Diffusion Fused"]
        for c, t in enumerate(col_titles):
            axes[0][c].set_title(t, fontsize=14, fontweight="bold", pad=10)

        for r, idx in enumerate(top3):
            row = rows[idx]
            img_name = f"{idx:04d}_fused.png"

            def show(ax, path_or_none):
                if path_or_none and Path(path_or_none).exists():
                    ax.imshow(robust_normalize(load_img_rgb(path_or_none)))
                else:
                    ax.set_facecolor("#dddddd")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12)
                ax.axis("off")

            show(axes[r][0], row.get("source1_path"))
            show(axes[r][1], row.get("source2_path"))

            gan_path = cfg["gan_fused"] / img_name
            if not gan_path.exists():
                for fb in ["fused_color", "fused_original_enhanced", "fused_original"]:
                    alt = GAN_IMG / pair / fb / img_name
                    if alt.exists():
                        gan_path = alt
                        break
            show(axes[r][2], gan_path)

            diff_path = cfg["diff_fused"] / img_name
            if not diff_path.exists():
                for fb in ["fused_colored", "fused_grayscale", "fused_original"]:
                    alt = DIFF_IMG / pair / "test" / fb / img_name
                    if alt.exists():
                        diff_path = alt
                        break
            show(axes[r][3], diff_path)

            ssim_v = float(row["SSIM"])
            axes[r][0].set_ylabel(
                f"Sample {r+1}  (idx {idx:04d})\nSSIM = {ssim_v:.3f}",
                fontsize=9, labelpad=4
            )
            axes[r][0].axis("on")
            axes[r][0].tick_params(left=False, bottom=False,
                                   labelleft=True, labelbottom=False)
            for spine in axes[r][0].spines.values():
                spine.set_visible(False)

        fig.suptitle(
            f"Fusion Comparison — {cfg['label']}   "
            f"( {cfg['src1_label']} | MRI | GAN | Diffusion )",
            fontsize=15, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = CMP_D / f"B_{pair}_comparison.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"  Saved {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION C – BEST-3 GAN IMAGES
#  One figure per pair: 3 columns (source1 | MRI | GAN fused)
# ══════════════════════════════════════════════════════════════════════════════
def make_best_gan():
    for pair, cfg in PAIR_CFG.items():
        d = BGAN_D / pair
        d.mkdir(exist_ok=True)
        rows  = gan_metrics[pair]
        top3  = top3_by_ssim(rows)

        fig, axes = plt.subplots(3, 3, figsize=(13, 13))
        col_titles = [cfg["src1_label"], "MRI", "GAN Fused"]
        for c, t in enumerate(col_titles):
            axes[0][c].set_title(t, fontsize=13, fontweight="bold")

        for r, idx in enumerate(top3):
            row = rows[idx]
            img_name = f"{idx:04d}_fused.png"

            def show(ax, path):
                if path and Path(path).exists():
                    ax.imshow(load_img_rgb(path))
                else:
                    ax.set_facecolor("#dddddd")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes)
                ax.axis("off")

            show(axes[r][0], row.get("source1_path"))
            show(axes[r][1], row.get("source2_path"))

            gan_path = cfg["gan_fused"] / img_name
            if not gan_path.exists():
                for fb in ["fused_color", "fused_original_enhanced", "fused_original"]:
                    alt = GAN_IMG / pair / fb / img_name
                    if alt.exists():
                        gan_path = alt
                        break
            show(axes[r][2], gan_path)

            # copy individual fused file
            if gan_path.exists():
                shutil.copy(gan_path, d / f"rank{r+1:02d}_idx{idx:04d}.png")

            ssim_v = float(row["SSIM"])
            psnr_v = float(row["PSNR"])
            mi_v   = float(row["MI"])
            axes[r][0].set_ylabel(
                f"Rank {r+1} (idx {idx:04d})\n"
                f"SSIM={ssim_v:.3f}  PSNR={psnr_v:.1f} dB  MI={mi_v:.2f}",
                fontsize=8, labelpad=4
            )
            axes[r][0].axis("on")
            axes[r][0].tick_params(left=False, bottom=False,
                                   labelleft=True, labelbottom=False)
            for spine in axes[r][0].spines.values():
                spine.set_visible(False)

        fig.suptitle(f"Best 3 GAN Fused Results — {cfg['label']}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = BGAN_D / f"C_{pair}_best3_gan.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  Saved {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION D – BEST-3 DIFFUSION IMAGES
# ══════════════════════════════════════════════════════════════════════════════
def get_diff_top3(pair):
    """Return top-3 image index strings for diffusion, ranked by SSIM."""
    # prefer 'diffusion_fused_grayscale'; fall back to anything for the pair
    subset = [r for r in diff_detailed
              if r["pair"] == pair and "grayscale" in r.get("variant", "")]
    if not subset:
        subset = [r for r in diff_detailed if r["pair"] == pair]
    ranked = sorted(subset, key=lambda r: float(r["SSIM"]), reverse=True)
    seen, top3 = set(), []
    for r in ranked:
        img_id = r["image"]
        if img_id not in seen:
            seen.add(img_id)
            top3.append(r)
        if len(top3) == 3:
            break
    return top3


def make_best_diffusion():
    for pair, cfg in PAIR_CFG.items():
        d = BDIF_D / pair
        d.mkdir(exist_ok=True)
        top3_rows = get_diff_top3(pair)

        fig, axes = plt.subplots(3, 3, figsize=(13, 13))
        col_titles = [cfg["src1_label"], "MRI", "Diffusion Fused"]
        for c, t in enumerate(col_titles):
            axes[0][c].set_title(t, fontsize=13, fontweight="bold")

        for r, dr in enumerate(top3_rows):
            img_id   = dr["image"]
            img_name = f"{int(img_id):04d}_fused.png"

            def show(ax, path):
                if path and Path(path).exists():
                    ax.imshow(load_img_rgb(path))
                else:
                    ax.set_facecolor("#dddddd")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes)
                ax.axis("off")

            show(axes[r][0], dr.get("source_modality_path"))
            show(axes[r][1], dr.get("mri_path"))

            diff_path = cfg["diff_fused"] / img_name
            if not diff_path.exists():
                for fb in ["fused_colored", "fused_grayscale", "fused_original"]:
                    alt = DIFF_IMG / pair / "test" / fb / img_name
                    if alt.exists():
                        diff_path = alt
                        break
            show(axes[r][2], diff_path)

            if diff_path.exists():
                shutil.copy(diff_path, d / f"rank{r+1:02d}_idx{int(img_id):04d}.png")

            ssim_v = float(dr["SSIM"])
            psnr_v = float(dr["PSNR"])
            mi_v   = float(dr["MI"])
            axes[r][0].set_ylabel(
                f"Rank {r+1} (idx {int(img_id):04d})\n"
                f"SSIM={ssim_v:.3f}  PSNR={psnr_v:.1f} dB  MI={mi_v:.2f}",
                fontsize=8, labelpad=4
            )
            axes[r][0].axis("on")
            axes[r][0].tick_params(left=False, bottom=False,
                                   labelleft=True, labelbottom=False)
            for spine in axes[r][0].spines.values():
                spine.set_visible(False)

        fig.suptitle(f"Best 3 Diffusion Fused Results — {cfg['label']}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = BDIF_D / f"D_{pair}_best3_diffusion.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  Saved {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*64)
    print(" Generating Final Thesis Assets")
    print("="*64)

    print("\n--- Training Curves ---")
    print("[A1] GAN training loss curves ...")
    plot_gan_loss_curves()

    print("[A2] GAN training curves (loss/SSIM/PSNR/MI) ...")
    plot_gan_training_curves()

    print("[A3] GAN train-vs-val curves ...")
    plot_gan_train_val_curves()

    print("[A4] Diffusion training loss curves ...")
    plot_diff_loss_curves()

    print("[A5] Diffusion validation SSIM curves ...")
    plot_diff_val_ssim()

    print("[A6] Diffusion training curves (loss/SSIM/PSNR/MI) ...")
    plot_diff_training_curves()

    print("[A7] Diffusion train-vs-val curves ...")
    plot_diff_train_val_curves()

    print("\n--- Comparison Panels ---")
    make_comparison_panels()

    print("\n--- Best-3 GAN Images ---")
    make_best_gan()

    print("\n--- Best-3 Diffusion Images ---")
    make_best_diffusion()

    print("\n" + "="*64)
    print(" Done!  All assets in: " + str(THESIS))
    print("="*64 + "\n")
