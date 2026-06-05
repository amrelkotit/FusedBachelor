"""
Plot GAN training curves from the persisted CSV history.

Produces per-pair and combined figures:
  <pair>_training_curves.png   — full 10-panel overview
  <pair>_loss_curve.png        — train vs validation loss (with smoothing)
  <pair>_quality_metrics.png   — SSIM / PSNR / MI vs epoch (with smoothing)
  all_pairs_training_overview.png — 3-column combined figure

Run:
    python scripts/plot_training_history.py --pair ct_mri
    python scripts/plot_training_history.py --pair all
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

from src.data.paired_dataset import gan_graph_dir, normalize_pair

PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_history(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def value(row, key):
    item = row.get(key)
    if item in (None, ""):
        return None
    try:
        return float(item)
    except (ValueError, TypeError):
        return None


def best_epoch_from_history(rows):
    for row in reversed(rows):
        item = row.get("best_epoch")
        if item not in (None, "", "None"):
            try:
                return int(float(item))
            except (ValueError, TypeError):
                pass
    candidates = [
        (int(row["epoch"]), value(row, "val_ssim"))
        for row in rows
        if value(row, "val_ssim") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def smooth(y, window=15):
    """Centered rolling mean; falls back to original where window doesn't fit."""
    if len(y) < window:
        return np.array(y, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def series(rows, key):
    """Return (epochs, values) arrays, filtering None."""
    pts = [(int(row["epoch"]), value(row, key)) for row in rows if value(row, key) is not None]
    if not pts:
        return np.array([]), np.array([])
    epochs, values = zip(*pts)
    return np.array(epochs), np.array(values)


def annotate_best(ax, best_epoch, rows, key):
    if best_epoch is None:
        return
    pts = {int(row["epoch"]): value(row, key) for row in rows if value(row, key) is not None}
    if best_epoch not in pts:
        return
    y_val = pts[best_epoch]
    ax.axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.3, alpha=0.75, zorder=2)
    ax.scatter([best_epoch], [y_val], color="#d62728", s=42, zorder=5, edgecolors="white", linewidths=0.8)
    ax.text(
        best_epoch + 1, y_val, f" best={best_epoch}",
        color="#9c1f1f", fontsize=9, va="bottom", zorder=6,
    )


def _apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=9)


def plot_loss_curve(pair, rows, output_dir, best_epoch):
    """Train + validation total loss with smoothed overlay."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    for key, label, color in [
        ("train_total_loss", "Train loss", "#4878cf"),
        ("val_total_loss",   "Validation loss", "#d62728"),
    ]:
        x, y = series(rows, key)
        if x.size == 0:
            continue
        ax.plot(x, y, alpha=0.25, linewidth=1.0, color=color)
        ax.plot(x, smooth(y), label=label, linewidth=2.2, color=color)

    annotate_best(ax, best_epoch, rows, "val_total_loss")
    ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Loss Curve", fontweight="bold", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=10)
    _apply_style(ax)
    fig.tight_layout()
    path = output_dir / f"{pair}_loss_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_quality_metrics(pair, rows, output_dir, best_epoch):
    """Validation SSIM, PSNR, and MI with smoothed overlay."""
    specs = [
        ("val_ssim", "SSIM",  "#2ca02c"),
        ("val_psnr", "PSNR",  "#9467bd"),
        ("val_mi",   "MI",    "#ff7f0e"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    for key, label, color in specs:
        x, y = series(rows, key)
        if x.size == 0:
            continue
        ax.plot(x, y, alpha=0.22, linewidth=1.0, color=color)
        ax.plot(x, smooth(y), label=label, linewidth=2.2, color=color)

    annotate_best(ax, best_epoch, rows, "val_ssim")
    ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Quality Metrics", fontweight="bold", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False, fontsize=10)
    _apply_style(ax)
    fig.tight_layout()
    path = output_dir / f"{pair}_quality_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_full_curves(pair, rows, output_dir, best_epoch):
    """10-panel overview of all tracked metrics."""
    specs = [
        ("train_gan_loss",   "Generator loss"),
        ("train_d1_loss",    "Discriminator loss"),
        ("train_total_loss", "Train loss"),
        ("val_total_loss",   "Validation loss"),
        ("val_ssim",         "SSIM"),
        ("val_psnr",         "PSNR"),
        ("val_mi",           "MI"),
        ("val_ag",           "AG"),
        ("val_sf",           "SF"),
        ("learning_rate",    "Learning rate"),
    ]

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), dpi=160)

    for ax, (key, title) in zip(axes.flatten(), specs):
        x, y = series(rows, key)
        if x.size > 0:
            ax.plot(x, y, alpha=0.22, linewidth=1.0, color="#555555")
            ax.plot(x, smooth(y), linewidth=1.8, color="#1f77b4")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        _apply_style(ax)
        if best_epoch:
            ax.axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.6)

    for ax in axes.flatten()[len(specs):]:
        ax.axis("off")

    fig.suptitle(f"{PAIR_LABELS.get(pair, pair)} GAN training curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{pair}_training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_combined_overview(all_rows_dict, output_dir):
    """3-column × 2-row: loss + quality for each pair side-by-side."""
    pairs_present = [pair for pair in PAIRS if pair in all_rows_dict]
    if not pairs_present:
        return None

    n_cols = len(pairs_present)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9), dpi=200)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    plt.rcParams.update({"font.size": 10})

    for col, pair in enumerate(pairs_present):
        rows = all_rows_dict[pair]
        best_epoch = best_epoch_from_history(rows)

        # Row 0: loss
        ax = axes[0, col]
        for key, label, color in [
            ("train_total_loss", "Train", "#4878cf"),
            ("val_total_loss",   "Val",   "#d62728"),
        ]:
            x, y = series(rows, key)
            if x.size == 0:
                continue
            ax.plot(x, y, alpha=0.2, linewidth=0.8, color=color)
            ax.plot(x, smooth(y), label=label, linewidth=2.0, color=color)
        if best_epoch:
            ax.axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.65)
        ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False, fontsize=9)
        _apply_style(ax)

        # Row 1: quality
        ax = axes[1, col]
        for key, label, color in [
            ("val_ssim", "SSIM", "#2ca02c"),
            ("val_psnr", "PSNR", "#9467bd"),
            ("val_mi",   "MI",   "#ff7f0e"),
        ]:
            x, y = series(rows, key)
            if x.size == 0:
                continue
            ax.plot(x, y, alpha=0.2, linewidth=0.8, color=color)
            ax.plot(x, smooth(y), label=label, linewidth=2.0, color=color)
        if best_epoch:
            ax.axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.65)
        ax.set_title(f"{PAIR_LABELS.get(pair, pair)} — Metrics", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.legend(frameon=False, fontsize=9)
        _apply_style(ax)

    fig.suptitle("GAN Training Overview — All Pairs", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = output_dir / "all_pairs_training_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_history(pair, history_path, output_dir):
    rows = read_history(history_path)
    if not rows:
        raise ValueError(f"No history rows found in: {history_path}")
    best_epoch = best_epoch_from_history(rows)

    paths = []
    paths.append(plot_full_curves(pair, rows, output_dir, best_epoch))
    paths.append(plot_loss_curve(pair, rows, output_dir, best_epoch))
    paths.append(plot_quality_metrics(pair, rows, output_dir, best_epoch))
    return paths, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GAN training curves from persisted CSV history.")
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="ct_mri")
    parser.add_argument("--history", default=None, help="Override CSV history path.")
    parser.add_argument("--output-dir", default=str(gan_graph_dir()))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    pairs = PAIRS if args.pair == "all" else [normalize_pair(args.pair)]

    all_rows_dict = {}
    for pair in pairs:
        history_path = (
            resolve_path(args.history)
            if args.history
            else gan_graph_dir() / f"{pair}_training_history.csv"
        )
        if not history_path.exists():
            print(f"[Warning] History not found, skipping: {history_path}")
            continue
        print(f"Plotting {pair} from {history_path}")
        saved_paths, rows = plot_history(pair, history_path, output_dir)
        all_rows_dict[pair] = rows
        for path in saved_paths:
            print(f"  Saved: {path}")

    if len(all_rows_dict) > 1:
        plot_combined_overview(all_rows_dict, output_dir)


if __name__ == "__main__":
    main()
