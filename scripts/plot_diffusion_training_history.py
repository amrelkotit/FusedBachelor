"""
Plot diffusion model training curves from persisted CSV history logs.

Produces per-pair and combined figures:
  <pair>_diffusion_loss_curve.png     — training loss vs epoch (with smoothing)
  <pair>_diffusion_ssim_curve.png     — validation SSIM vs epoch (with smoothing)
  all_pairs_diffusion_training.png    — combined 3-pair overview

Reads:
    outputs/models/diffusion/logs/<pair>_training_history.csv

Writes:
    outputs/models/diffusion/graphs/

Run:
    python scripts/plot_diffusion_training_history.py --pair ct_mri
    python scripts/plot_diffusion_training_history.py --pair all
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}

DIFFUSION_LOGS_DIR  = PROJECT_ROOT / "outputs" / "models" / "diffusion" / "logs"
DIFFUSION_GRAPH_DIR = PROJECT_ROOT / "outputs" / "models" / "diffusion" / "graphs"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_history(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def numeric(row, key):
    item = row.get(key)
    if item in (None, ""):
        return None
    try:
        return float(item)
    except (ValueError, TypeError):
        return None


def series(rows, key):
    pts = [(int(row["epoch"]), numeric(row, key)) for row in rows if numeric(row, key) is not None]
    if not pts:
        return np.array([]), np.array([])
    epochs, values = zip(*pts)
    return np.array(epochs), np.array(values)


def smooth(y, window=15):
    if len(y) < window:
        return np.array(y, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def best_epoch(rows):
    raw = None
    for row in reversed(rows):
        item = row.get("best_epoch")
        if item not in (None, "", "None"):
            try:
                raw = int(float(item))
                break
            except (ValueError, TypeError):
                pass
    if raw is not None:
        return raw
    val_pts = [(int(row["epoch"]), numeric(row, "val_ssim")) for row in rows if numeric(row, "val_ssim") is not None]
    return max(val_pts, key=lambda p: p[1])[0] if val_pts else None


def _apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=9)


def annotate_best(ax, best_ep, rows, key):
    if best_ep is None:
        return
    pts = {int(row["epoch"]): numeric(row, key) for row in rows if numeric(row, key) is not None}
    if best_ep not in pts:
        return
    y_val = pts[best_ep]
    ax.axvline(best_ep, color="#d62728", linestyle="--", linewidth=1.3, alpha=0.75, zorder=2)
    ax.scatter([best_ep], [y_val], color="#d62728", s=42, zorder=5, edgecolors="white", linewidths=0.8)
    ax.text(best_ep + 1, y_val, f" best={best_ep}", color="#9c1f1f", fontsize=9, va="bottom")


def _loss_key(rows):
    """Pick the right column name for training loss."""
    if not rows:
        return "train_loss"
    for key in ("train_loss", "train_total_loss"):
        if key in rows[0]:
            return key
    return "train_loss"


def plot_loss_curve(pair, rows, output_dir, best_ep):
    loss_key = _loss_key(rows)
    x, y = series(rows, loss_key)
    if x.size == 0:
        print(f"[Warning] No loss data for {pair}, skipping loss curve.")
        return None

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(x, y, alpha=0.22, linewidth=1.0, color="#4878cf")
    ax.plot(x, smooth(y), label="Training loss", linewidth=2.2, color="#4878cf")

    x_val, y_val = series(rows, "val_ssim")
    if x_val.size > 0:
        ax2 = ax.twinx()
        ax2.plot(x_val, y_val, alpha=0.22, linewidth=1.0, color="#2ca02c")
        ax2.plot(x_val, smooth(y_val), label="Val SSIM", linewidth=2.2, color="#2ca02c", linestyle="--")
        ax2.set_ylabel("Validation SSIM", color="#2ca02c", labelpad=4)
        ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=9)
        ax2.spines["top"].set_visible(False)
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    annotate_best(ax, best_ep, rows, loss_key)
    ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Diffusion Training Loss", fontweight="bold", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", labelpad=4)
    _apply_style(ax)

    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=10, loc="upper right")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{pair}_diffusion_loss_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_ssim_curve(pair, rows, output_dir, best_ep):
    x, y = series(rows, "val_ssim")
    if x.size == 0:
        print(f"[Warning] No val_ssim data for {pair}, skipping SSIM curve.")
        return None

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(x, y, alpha=0.22, linewidth=1.0, color="#2ca02c")
    ax.plot(x, smooth(y), label="Validation SSIM", linewidth=2.2, color="#2ca02c")
    annotate_best(ax, best_ep, rows, "val_ssim")
    ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Diffusion Validation SSIM", fontweight="bold", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.legend(frameon=False, fontsize=10)
    _apply_style(ax)
    fig.tight_layout()
    path = output_dir / f"{pair}_diffusion_ssim_curve.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_combined_overview(all_rows_dict, output_dir):
    """2-row × N-col combined figure: loss (row 0) + SSIM (row 1) for each pair."""
    pairs_present = [pair for pair in PAIRS if pair in all_rows_dict]
    if not pairs_present:
        return None

    n_cols = len(pairs_present)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9), dpi=200)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, pair in enumerate(pairs_present):
        rows = all_rows_dict[pair]
        best_ep = best_epoch(rows)
        loss_key = _loss_key(rows)

        # Row 0 — loss
        ax = axes[0, col]
        x, y = series(rows, loss_key)
        if x.size > 0:
            ax.plot(x, y, alpha=0.2, linewidth=0.8, color="#4878cf")
            ax.plot(x, smooth(y), linewidth=2.0, color="#4878cf", label="Train loss")
        if best_ep:
            ax.axvline(best_ep, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.65)
        ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False, fontsize=9)
        _apply_style(ax)

        # Row 1 — validation SSIM
        ax = axes[1, col]
        x, y = series(rows, "val_ssim")
        if x.size > 0:
            ax.plot(x, y, alpha=0.2, linewidth=0.8, color="#2ca02c")
            ax.plot(x, smooth(y), linewidth=2.0, color="#2ca02c", label="Val SSIM")
        if best_ep:
            ax.axvline(best_ep, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.65)
        ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Val SSIM", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SSIM")
        ax.legend(frameon=False, fontsize=9)
        _apply_style(ax)

    fig.suptitle("Diffusion Training Overview — All Pairs", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = output_dir / "all_pairs_diffusion_training.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot diffusion training curves from logs.")
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all")
    parser.add_argument("--logs-dir",   default=str(DIFFUSION_LOGS_DIR))
    parser.add_argument("--output-dir", default=str(DIFFUSION_GRAPH_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    logs_dir   = resolve_path(args.logs_dir)
    output_dir = resolve_path(args.output_dir)
    pairs = PAIRS if args.pair == "all" else [args.pair]

    all_rows_dict = {}
    for pair in pairs:
        history_path = logs_dir / f"{pair}_training_history.csv"
        rows = read_history(history_path)
        if not rows:
            print(f"[Warning] No history found for {pair}: {history_path}")
            continue
        best_ep = best_epoch(rows)
        print(f"Plotting diffusion history for {pair} (best epoch: {best_ep})")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_loss_curve(pair, rows, output_dir, best_ep)
        plot_ssim_curve(pair, rows, output_dir, best_ep)
        all_rows_dict[pair] = rows

    if len(all_rows_dict) > 1:
        plot_combined_overview(all_rows_dict, output_dir)


if __name__ == "__main__":
    main()
