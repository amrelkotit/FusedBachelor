"""
Generate per-image-pair quantitative comparison figures for the thesis.

Produces one figure per fusion pair (CT-MRI, PET-MRI, SPECT-MRI) with five
metric subplots (EN, PSNR, MI, CC, SF) arranged as 3-top + 2-bottom, each
showing per-image values for all evaluated methods.

Reads:
    outputs/models/diffusion/metrics/diffusion_metrics_detailed.csv

Writes:
    outputs/thesis_figures/quantitative_comparison/<pair>_quantitative_comparison.png

Run:
    python scripts/plot_per_pair_comparison.py
    python scripts/plot_per_pair_comparison.py --pair ct_mri
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}

# Metrics to plot (5 panels like the reference thesis figures)
METRICS = [
    ("EN",   "EN"),
    ("PSNR", "PSNR"),
    ("MI",   "MI"),
    ("CC",   "CC"),
    ("SF",   "SF"),
]

# Variant display names and styling ─────────────────────────────────────────
VARIANT_STYLE = {
    "gan": dict(
        label="GAN",
        color="#1f77b4",
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.4,
    ),
    "diffusion_fused_original": dict(
        label="Diffusion",
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.4,
    ),
    "diffusion_fused_grayscale": dict(
        label="Diffusion (Gray)",
        color="#ff7f0e",
        marker="^",
        linestyle="-.",
        linewidth=1.6,
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.4,
    ),
    "average_baseline": dict(
        label="Avg. Fusion",
        color="#7f7f7f",
        marker="D",
        linestyle=":",
        linewidth=1.5,
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=1.3,
    ),
}

# Preferred display order
VARIANT_ORDER = [
    "gan",
    "diffusion_fused_original",
    "diffusion_fused_grayscale",
    "average_baseline",
]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_detailed(path):
    if not path.exists():
        print(f"[Warning] Detailed metrics file not found: {path}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def numeric(value):
    text = str(value or "").strip()
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def build_per_pair_data(rows, pair):
    """Return {variant: {image_idx: {metric: value}}}."""
    data = defaultdict(dict)
    for row in rows:
        if row.get("pair") != pair:
            continue
        variant = row.get("variant", "")
        idx_str = row.get("image", "")
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        entry = {}
        for metric, _ in METRICS:
            value = numeric(row.get(metric))
            if value is not None:
                entry[metric] = value
        if entry:
            data[variant][idx] = entry
    return data


def select_variants(data):
    """Return variants present in data, in preferred order."""
    available = set(data.keys())
    ordered = [variant for variant in VARIANT_ORDER if variant in available]
    ordered += sorted(available - set(VARIANT_ORDER))
    return ordered


def plot_pair(pair, data, output_dir, num_pairs=24):
    variants = select_variants(data)
    if not variants:
        print(f"[Warning] No data for pair {pair}, skipping.")
        return None

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })

    fig = plt.figure(figsize=(14, 7.5), dpi=200)

    # 3-top + 2-bottom layout matching reference thesis figures
    top_metrics = METRICS[:3]
    bot_metrics = METRICS[3:]

    gs_top = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.91, bottom=0.54, wspace=0.32)
    gs_bot = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.44, bottom=0.08, wspace=0.32)

    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    axes_bot = [fig.add_subplot(gs_bot[0, i]) for i in range(2)]
    # hide unused bottom-right slot
    ax_empty = fig.add_subplot(gs_bot[0, 2])
    ax_empty.axis("off")

    axes = axes_top + axes_bot
    x = np.arange(1, num_pairs + 1)

    handles, labels = [], []

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for variant in variants:
            style = VARIANT_STYLE.get(variant, {})
            values = [data[variant].get(i, {}).get(metric_key) for i in range(num_pairs)]
            if all(value is None for value in values):
                continue
            y = [value if value is not None else np.nan for value in values]
            x_plot = [xi for xi, yi in zip(x, y) if not np.isnan(yi)]
            y_plot = [yi for yi in y if not np.isnan(yi)]
            line, = ax.plot(
                x_plot, y_plot,
                label=style.get("label", variant),
                color=style.get("color", None),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 1.5),
                markersize=style.get("markersize", 4),
                markerfacecolor=style.get("markerfacecolor", "white"),
                markeredgewidth=style.get("markeredgewidth", 1.2),
                markeredgecolor=style.get("color", None),
            )
            if metric_key == "EN":
                handles.append(line)
                labels.append(style.get("label", variant))

        ax.set_title(metric_label, fontweight="bold", pad=4)
        ax.set_xlabel("Image pairs", labelpad=2)
        ax.set_ylabel(metric_label, labelpad=3)
        ax.set_xlim(0.5, num_pairs + 0.5)
        ax.set_xticks(range(1, num_pairs + 1, 3))
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Quantitative comparison on {num_pairs} image pairs from {PAIR_LABELS[pair]} fusion dataset",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(handles),
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            fontsize=10,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pair}_quantitative_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-image-pair quantitative comparison figures.")
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all")
    parser.add_argument(
        "--detailed-csv",
        default="outputs/models/diffusion/metrics/diffusion_metrics_detailed.csv",
        help="Path to diffusion_metrics_detailed.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/thesis_figures/quantitative_comparison",
        help="Directory where figures are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detailed_path = resolve_path(args.detailed_csv)
    output_dir = resolve_path(args.output_dir)
    pairs = PAIRS if args.pair == "all" else [args.pair]

    rows = read_detailed(detailed_path)
    if not rows:
        print("[Error] No data loaded. Run calculate_diffusion_metrics.py first.")
        return

    for pair in pairs:
        data = build_per_pair_data(rows, pair)
        # Infer num_pairs from data
        all_indices = set()
        for variant_data in data.values():
            all_indices.update(variant_data.keys())
        num_pairs = max(all_indices) + 1 if all_indices else 24
        plot_pair(pair, data, output_dir, num_pairs=num_pairs)


if __name__ == "__main__":
    main()
