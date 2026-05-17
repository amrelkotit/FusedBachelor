import argparse
import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.paired_dataset import diffusion_final_assets_dir, diffusion_graph_dir, diffusion_logs_dir, diffusion_metrics_dir


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}
METRIC_COLUMNS = ["SSIM", "PSNR", "MI", "EN", "SF", "AG", "Edge_Intensity"]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path):
    if not path.exists():
        print(f"[Warning] Missing file, skipping: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def numeric(value):
    if value in (None, "", "None"):
        return None
    text = str(value)
    if "+/-" in text:
        text = text.split("+/-", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved graph: {path}")


def plot_training_curves(output_root, graph_dir):
    logs_dir = diffusion_logs_dir(output_root)
    best_rows = []
    for pair in PAIRS:
        rows = read_csv(logs_dir / f"{pair}_training_history.csv")
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows if row.get("epoch")]
        if not epochs:
            print(f"[Warning] Training history has no epochs for {pair}.")
            continue
        train_key = "train_loss" if "train_loss" in rows[0] else "train_total_loss"
        train_values = [numeric(row.get(train_key)) for row in rows]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        ax.plot(epochs, train_values, label="Train loss", linewidth=2)
        ax.set_title(f"{PAIR_LABELS[pair]} Diffusion Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        save_fig(fig, graph_dir / f"training_loss_curve_{pair}.png")

        val_points = [(int(row["epoch"]), numeric(row.get("val_ssim"))) for row in rows if numeric(row.get("val_ssim")) is not None]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        if val_points:
            ax.plot([point[0] for point in val_points], [point[1] for point in val_points], label="Validation SSIM", linewidth=2, color="#2ca02c")
        else:
            print(f"[Warning] No validation SSIM values found for {pair}.")
        ax.set_title(f"{PAIR_LABELS[pair]} Diffusion Validation SSIM")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SSIM")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        save_fig(fig, graph_dir / f"validation_ssim_curve_{pair}.png")

        best_epoch = numeric(rows[-1].get("best_epoch"))
        if best_epoch is None and val_points:
            best_epoch = max(val_points, key=lambda item: item[1])[0]
        if best_epoch is not None:
            best_rows.append((PAIR_LABELS[pair], best_epoch))
    if best_rows:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        ax.bar([row[0] for row in best_rows], [row[1] for row in best_rows], color="#4c78a8")
        ax.set_title("Best Epoch Per Pair")
        ax.set_ylabel("Best epoch")
        ax.grid(axis="y", alpha=0.25)
        save_fig(fig, graph_dir / "best_epoch_per_pair.png")
    else:
        print("[Warning] No best epoch data found.")


def summary_by_variant(metrics_dir):
    rows = read_csv(metrics_dir / "diffusion_metrics_summary.csv")
    return {(row.get("pair"), row.get("variant")): row for row in rows}


def best_diffusion_row(summary, pair):
    variants = ["diffusion_fused_grayscale", "diffusion_fused_colored", "diffusion_fused_original"]
    rows = [summary[(pair, variant)] for variant in variants if (pair, variant) in summary]
    if not rows:
        return None
    return max(rows, key=lambda row: numeric(row.get("SSIM")) or 0.0)


def plot_diffusion_pair_metrics(metrics_dir, graph_dir):
    summary = summary_by_variant(metrics_dir)
    if not summary:
        print("[Warning] No diffusion metrics summary found, skipping metric comparison graph.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    for ax, pair in zip(axes, PAIRS):
        row = best_diffusion_row(summary, pair) or {}
        values = [numeric(row.get(metric)) or 0.0 for metric in METRIC_COLUMNS]
        ax.bar(METRIC_COLUMNS, values, color="#4c78a8")
        ax.set_title(f"{PAIR_LABELS[pair]}\n{row.get('variant', 'missing')}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Diffusion Metric Comparison Across Pairs")
    save_fig(fig, graph_dir / "diffusion_metric_comparison_across_pairs.png")


def read_comparison_rows(metrics_dir, filename):
    rows = read_csv(metrics_dir / filename)
    if not rows:
        return {}
    grouped = {}
    for row in rows:
        grouped[(row.get("pair"), row.get("metric"))] = row
    return grouped


def plot_comparison_delta(metrics_dir, graph_dir, filename, title, destination):
    rows = read_comparison_rows(metrics_dir, filename)
    if not rows:
        print(f"[Warning] No rows in {filename}, skipping {title}.")
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180)
    for ax, metric in zip(axes.flatten(), METRIC_COLUMNS):
        values = []
        labels = []
        for pair in PAIRS:
            row = rows.get((pair, metric))
            values.append(numeric(row.get("delta_diffusion_minus_comparison")) if row else 0.0)
            labels.append(PAIR_LABELS[pair])
        colors = ["#4c78a8" if value >= 0 else "#e45756" for value in values]
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    axes.flatten()[-1].axis("off")
    fig.suptitle(title)
    save_fig(fig, graph_dir / destination)


def copy_graphs_to_assets(output_root, graph_dir):
    target = diffusion_final_assets_dir(output_root) / "graphs"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in graph_dir.glob("*.png"):
        shutil.copy2(path, target / path.name)
        copied += 1
    print(f"Copied {copied} graph(s) to: {target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot diffusion graphs under outputs/models/diffusion/graphs.")
    parser.add_argument("--output-root", default="outputs/models/diffusion")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = resolve_path(args.output_root)
    graph_dir = diffusion_graph_dir(output_root)
    metrics_dir = diffusion_metrics_dir(output_root)
    graph_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(output_root, graph_dir)
    plot_diffusion_pair_metrics(metrics_dir, graph_dir)
    plot_comparison_delta(metrics_dir, graph_dir, "diffusion_vs_baseline_summary.csv", "Diffusion vs Average Baseline", "diffusion_vs_average_baseline.png")
    plot_comparison_delta(metrics_dir, graph_dir, "diffusion_vs_gan_summary.csv", "Diffusion vs GAN", "diffusion_vs_gan.png")
    copy_graphs_to_assets(output_root, graph_dir)


if __name__ == "__main__":
    main()
