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
    return output_path


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
    print(f"Saved graph: {plot_history(pair, history_path, output_dir)}")


if __name__ == "__main__":
    main()
