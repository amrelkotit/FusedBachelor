import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.paired_dataset import gan_graph_dir, normalize_pair


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
    return float(item)


def best_epoch_from_history(rows):
    for row in reversed(rows):
        item = row.get("best_epoch")
        if item not in (None, "", "None"):
            return int(float(item))
    candidates = [(int(row["epoch"]), value(row, "val_ssim")) for row in rows if value(row, "val_ssim") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def annotate_best_epoch(ax, rows, best_epoch, key):
    if best_epoch is None:
        return
    points = {int(row["epoch"]): value(row, key) for row in rows if value(row, key) is not None}
    if best_epoch not in points:
        return
    ax.axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.scatter([best_epoch], [points[best_epoch]], color="#d62728", s=34, zorder=5)
    ax.text(best_epoch, points[best_epoch], f" best {best_epoch}", color="#9c1f1f", fontsize=10, va="bottom")


def plot_clean_history(pair, rows, output_dir):
    best_epoch = best_epoch_from_history(rows)
    epochs = [int(row["epoch"]) for row in rows]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    for key, label, color in [
        ("train_total_loss", "Train loss", "#1f77b4"),
        ("val_total_loss", "Validation loss", "#d62728"),
    ]:
        points = [(int(row["epoch"]), value(row, key)) for row in rows if value(row, key) is not None]
        if points:
            ax.plot([point[0] for point in points], [point[1] for point in points], label=label, linewidth=2.2, color=color)
    annotate_best_epoch(ax, rows, best_epoch, "val_total_loss")
    ax.set_title(f"{pair.upper()} Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    loss_path = output_dir / f"{pair}_loss_curve.png"
    fig.savefig(loss_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    for key, label, color in [
        ("val_ssim", "SSIM", "#2ca02c"),
        ("val_psnr", "PSNR", "#9467bd"),
        ("val_mi", "MI", "#ff7f0e"),
    ]:
        points = [(int(row["epoch"]), value(row, key)) for row in rows if value(row, key) is not None]
        if points:
            ax.plot([point[0] for point in points], [point[1] for point in points], label=label, linewidth=2.2, color=color)
    annotate_best_epoch(ax, rows, best_epoch, "val_ssim")
    ax.set_title(f"{pair.upper()} Quality Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    quality_path = output_dir / f"{pair}_quality_metrics.png"
    fig.savefig(quality_path)
    plt.close(fig)
    return loss_path, quality_path


def plot_history(pair, history_path, output_dir):
    rows = read_history(history_path)
    if not rows:
        raise ValueError(f"No history rows found in: {history_path}")
    specs = [
        ("train_gan_loss", "Generator loss"),
        ("train_d1_loss", "Discriminator loss"),
        ("train_total_loss", "Train loss"),
        ("val_total_loss", "Validation loss"),
        ("val_ssim", "SSIM"),
        ("val_psnr", "PSNR"),
        ("val_mi", "MI"),
        ("val_ag", "AG"),
        ("val_sf", "SF"),
        ("learning_rate", "Learning rate"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), dpi=160)
    for ax, (key, title) in zip(axes.flatten(), specs):
        points = [(int(row["epoch"]), row.get(key)) for row in rows if row.get(key)]
        if points:
            ax.plot([point[0] for point in points], [float(point[1]) for point in points], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    for ax in axes.flatten()[len(specs):]:
        ax.axis("off")
    fig.suptitle(f"{pair} training curves")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pair}_training_curves.png"
    fig.savefig(output_path)
    plt.close(fig)
    clean_paths = plot_clean_history(pair, rows, output_dir)
    return output_path, *clean_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Plot GAN training curves from persisted CSV history.")
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], required=True)
    parser.add_argument("--history", default=None)
    parser.add_argument("--output-dir", default=str(gan_graph_dir()))
    return parser.parse_args()


def main():
    args = parse_args()
    pair = normalize_pair(args.pair)
    history_path = resolve_path(args.history) if args.history else gan_graph_dir() / f"{pair}_training_history.csv"
    output_dir = resolve_path(args.output_dir)
    print(f"Checkpoint path: plotting from history")
    print(f"Fused image folder: outputs/models/gan/images/aanlib")
    print(f"Graph folder: {output_dir}")
    print(f"Metrics folder: {PROJECT_ROOT / 'outputs' / 'models' / 'gan' / 'metrics'}")
    for path in plot_history(pair, history_path, output_dir):
        print(f"Saved graph: {path}")


if __name__ == "__main__":
    main()
