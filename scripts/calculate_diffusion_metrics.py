import argparse
import csv
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import AANLIB_ROOT, PairedMedicalImageDataset, aanlib_split_root, diffusion_image_dir, diffusion_metrics_dir, normalize_pair
from src.diffusion.utils import read_gray_tensor
from src.evaluation.metrics import evaluate_fusion


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}
METRIC_COLUMNS = ["SSIM", "PSNR", "MI", "EN", "SF", "AG", "Edge_Intensity"]
OPTIONAL_COLUMNS = ["CC", "FMI"]
ALL_METRIC_COLUMNS = [*METRIC_COLUMNS, *OPTIONAL_COLUMNS]
DIFFUSION_VARIANTS = ["diffusion_fused_grayscale", "diffusion_fused_colored", "diffusion_fused_original"]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def summarize(values):
    if not values:
        return ""
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0
    return f"{tensor.mean().item():.6f} +/- {std:.6f}"


def numeric_mean(value):
    text = str(value or "").strip()
    if "+/-" in text:
        text = text.split("+/-", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def metric_row(fused, source1, source2):
    metrics = evaluate_fusion(fused, source2, source1)
    # Use aggregated keys: PSNR with data_range=1.0, MI as sum of MI_MRI+MI_CT
    return {
        "SSIM": metrics.get("SSIM", 0.5 * (metrics["SSIM_MRI"] + metrics["SSIM_CT"])),
        "PSNR": metrics.get("PSNR", 0.5 * (metrics["PSNR_MRI"] + metrics["PSNR_CT"])),
        "MI":   metrics.get("MI", 0.0),   # sum MI_MRI + MI_CT (256 bins)
        "EN":   metrics["EN"],
        "SF":   metrics["SF"],
        "AG":   metrics["AG"],
        "Edge_Intensity": metrics.get("EPI", 0.0),
        "CC":   metrics.get("CC", 0.0),
        "FMI":  metrics.get("FMI", 0.0),
    }


def gan_candidate(gan_root, pair, split, index):
    stem = f"{index:04d}_fused.png"
    candidates = []
    if pair == "ct_mri":
        candidates.append(gan_root / "images" / "aanlib" / split / pair / "fused_original" / stem)
    else:
        candidates.extend(
            [
                gan_root / "images" / "aanlib" / split / pair / "fused_color" / stem,
                gan_root / "images" / "aanlib" / split / pair / "fused_original" / stem,
            ]
        )
    return next((path for path in candidates if path.exists()), None)


def candidate_outputs(output_root, gan_root, pair, split, index):
    stem = f"{index:04d}_fused.png"
    image_dir = diffusion_image_dir("aanlib", pair, split, output_root=output_root)
    candidates = {
        "diffusion_fused_original": image_dir / "fused_original" / stem,
        "diffusion_fused_grayscale": image_dir / "fused_grayscale" / stem,
        "average_baseline": output_root / "baselines" / "average" / "aanlib" / pair / split / f"{index:04d}_average.png",
    }
    if pair in {"pet_mri", "spect_mri"}:
        candidates["diffusion_fused_colored"] = image_dir / "fused_colored" / stem
    gan_path = gan_candidate(gan_root, pair, split, index)
    if gan_path is not None:
        candidates["gan"] = gan_path
    return candidates


def generated_root_exists(output_root, pair, split):
    image_dir = diffusion_image_dir("aanlib", pair, split, output_root=output_root)
    fused_dir = image_dir / "fused_original"
    if fused_dir.exists() and any(fused_dir.glob("*_fused.png")):
        return True
    print(f"[Warning] No diffusion generated images found for {pair} {split}: {fused_dir}")
    return False


def calculate_metrics(dataset_root, output_root, gan_root, pairs, split, image_size, max_items=None):
    detailed = []
    gan_found = False
    baseline_found = False
    for pair in pairs:
        if not generated_root_exists(output_root, pair, split):
            continue
        dataset = PairedMedicalImageDataset(
            aanlib_split_root(dataset_root, pair, split),
            image_size=image_size,
            max_items=max_items,
            dataset_name=f"AANLIB {pair} {split}",
            strict=True,
            pair=pair,
        )
        for index, sample in enumerate(dataset):
            source1 = sample["source1"].unsqueeze(0)
            source2 = sample["source2"].unsqueeze(0)
            for variant, path in candidate_outputs(output_root, gan_root, pair, split, index).items():
                if not path.exists():
                    continue
                gan_found = gan_found or variant == "gan"
                baseline_found = baseline_found or variant == "average_baseline"
                fused = read_gray_tensor(path, image_size=image_size)
                metrics = metric_row(fused, source1, source2)
                detailed.append(
                    {
                        "pair": pair,
                        "split": split,
                        "image": f"{index:04d}",
                        "variant": variant,
                        "source_modality_path": sample["source1_path"],
                        "mri_path": sample["source2_path"],
                        "fused_path": str(path),
                        **{column: f"{metrics[column]:.6f}" for column in ALL_METRIC_COLUMNS},
                    }
                )
    if not gan_found:
        print("GAN outputs not found, skipping GAN comparison.")
    if not baseline_found:
        print("Average baseline outputs not found, skipping average baseline comparison.")
    return detailed, gan_found, baseline_found


def group_summary(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, {column: [] for column in ALL_METRIC_COLUMNS})
        for column in ALL_METRIC_COLUMNS:
            groups[key][column].append(float(row[column]))
    summary = []
    for key, values in sorted(groups.items()):
        row = {name: value for name, value in zip(group_keys, key)}
        row.update({column: summarize(values[column]) for column in ALL_METRIC_COLUMNS})
        summary.append(row)
    return summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])
    print(f"Saved metrics: {path}")


def best_diffusion_rows(summary_rows):
    best = {}
    for pair in PAIRS:
        pair_rows = [row for row in summary_rows if row.get("pair") == pair and row.get("variant") in DIFFUSION_VARIANTS]
        if pair_rows:
            best[pair] = max(pair_rows, key=lambda row: numeric_mean(row.get("SSIM")))
    return best


def comparison_summary(summary_rows, compare_variant):
    by_key = {(row.get("pair"), row.get("variant")): row for row in summary_rows}
    best_rows = best_diffusion_rows(summary_rows)
    rows = []
    for pair, diffusion_row in best_rows.items():
        compare_row = by_key.get((pair, compare_variant))
        if compare_row is None:
            continue
        for metric in ALL_METRIC_COLUMNS:
            diffusion_value = numeric_mean(diffusion_row.get(metric))
            compare_value = numeric_mean(compare_row.get(metric))
            rows.append(
                {
                    "pair": pair,
                    "metric": metric,
                    "diffusion_variant": diffusion_row["variant"],
                    "comparison_variant": compare_variant,
                    "diffusion_mean": f"{diffusion_value:.6f}",
                    "comparison_mean": f"{compare_value:.6f}",
                    "delta_diffusion_minus_comparison": f"{diffusion_value - compare_value:.6f}",
                }
            )
    return rows


def write_legacy_pair_files(metrics_dir, summary_rows):
    for pair in PAIRS:
        pair_rows = [row for row in summary_rows if row.get("pair") == pair and row.get("variant") == "diffusion_fused_grayscale"]
        if not pair_rows:
            pair_rows = [row for row in summary_rows if row.get("pair") == pair and row.get("variant") == "diffusion_fused_original"]
        if pair_rows:
            write_csv(metrics_dir / f"per_pair_metrics_{pair}.csv", [{"image": "mean +/- std", **{column: pair_rows[0][column] for column in METRIC_COLUMNS}}], ["image", *METRIC_COLUMNS])
    final_rows = []
    best_rows = best_diffusion_rows(summary_rows)
    for pair in PAIRS:
        row = best_rows.get(pair)
        if row:
            final_rows.append({"Pair": PAIR_LABELS[pair], "Variant": row["variant"], **{column: row[column] for column in METRIC_COLUMNS}})
    if final_rows:
        write_csv(metrics_dir / "diffusion_final_results_summary.csv", final_rows, ["Pair", "Variant", *METRIC_COLUMNS])
        md_lines = ["| Pair | Variant | " + " | ".join(METRIC_COLUMNS) + " |", "|" + "---|" * len(["Pair", "Variant", *METRIC_COLUMNS])]
        for row in final_rows:
            md_lines.append("| " + " | ".join(row[column] for column in ["Pair", "Variant", *METRIC_COLUMNS]) + " |")
        (metrics_dir / "diffusion_final_results_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_training_summary(metrics_dir, output_root, pairs, split, summary_rows):
    lines = ["# Diffusion Training And Metrics Summary", ""]
    best_rows = best_diffusion_rows(summary_rows)
    for pair in pairs:
        history = output_root / "logs" / f"{pair}_training_history.csv"
        best_epoch = ""
        best_ssim = ""
        if history.exists():
            with history.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            if rows:
                best_epoch = rows[-1].get("best_epoch", "")
                best_ssim = max((numeric_mean(row.get("val_ssim")) for row in rows), default=0.0)
        metric = best_rows.get(pair)
        lines.extend(
            [
                f"## {PAIR_LABELS[pair]}",
                f"- Best epoch: {best_epoch}",
                f"- Best validation SSIM: {best_ssim:.6f}" if best_ssim != "" else "- Best validation SSIM: ",
                f"- Checkpoint path: `{output_root / ('aanlib_' + pair) / 'checkpoints' / 'best_diffusion.pt'}`",
                f"- Generated image folder: `{diffusion_image_dir('aanlib', pair, split, output_root=output_root)}`",
                f"- Best metrics variant: `{metric.get('variant', '') if metric else ''}`",
                f"- Metrics summary: SSIM `{metric.get('SSIM', '') if metric else ''}`, PSNR `{metric.get('PSNR', '') if metric else ''}`",
                "",
            ]
        )
    path = metrics_dir / "diffusion_training_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved metrics summary: {path}")


def write_thesis_summary(metrics_dir, summary_rows, gan_found, baseline_found):
    best_rows = best_diffusion_rows(summary_rows)
    overall_best = None
    if best_rows:
        overall_best = max(best_rows.values(), key=lambda row: numeric_mean(row.get("SSIM")))
    lines = [
        "# Diffusion Thesis Metrics Summary",
        "",
        "The diffusion pipeline is independent from the GAN pipeline. All reported diffusion outputs are stored under `outputs/models/diffusion`.",
        "",
        "## Output Types",
        "",
        "- `ct_mri`: grayscale only. No diffusion colored CT-MRI output is generated or evaluated.",
        "- `pet_mri`: grayscale and colored diffusion outputs are evaluated when present.",
        "- `spect_mri`: grayscale and colored diffusion outputs are evaluated when present.",
        "",
        "## Best Result Type",
        "",
    ]
    if overall_best:
        lines.append(f"- Best overall diffusion result by mean SSIM: `{overall_best['variant']}` for `{overall_best['pair']}` with SSIM `{overall_best['SSIM']}`.")
    for pair in PAIRS:
        row = best_rows.get(pair)
        if row:
            lines.append(f"- Best {PAIR_LABELS[pair]} result by mean SSIM: `{row['variant']}` with SSIM `{row['SSIM']}`.")
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            f"- GAN comparison: {'found and written to diffusion_vs_gan_summary.csv' if gan_found else 'skipped because GAN outputs were not found'}.",
            f"- Average baseline comparison: {'found and written to diffusion_vs_baseline_summary.csv' if baseline_found else 'skipped because average baseline outputs were not found'}.",
            "",
            "## Metrics",
            "",
            "Calculated metrics include SSIM, PSNR, mutual information, entropy, spatial frequency, average gradient, and edge intensity. CC and FMI are also written when available from the project evaluator.",
            "",
        ]
    )
    path = metrics_dir / "diffusion_thesis_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved thesis summary: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate diffusion, baseline, and optional GAN metrics.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--output-root", default="outputs/models/diffusion")
    parser.add_argument("--gan-root", default="outputs/models/gan")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_root = resolve_path(args.output_root)
    gan_root = resolve_path(args.gan_root)
    pairs = PAIRS if args.pair == "all" else [normalize_pair(args.pair)]
    metrics_dir = diffusion_metrics_dir(output_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    detailed, gan_found, baseline_found = calculate_metrics(dataset_root, output_root, gan_root, pairs, args.split, args.image_size, max_items=args.max_items)

    # When called with a specific pair, merge with existing detailed CSV so the
    # combined summaries always contain all previously-computed pairs.  Without
    # this, each per-pair call would overwrite the combined CSVs and only the
    # last pair would survive in the final summary files.
    if args.pair != "all":
        detailed_path = metrics_dir / "diffusion_metrics_detailed.csv"
        if detailed_path.exists():
            with detailed_path.open(newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
            # Drop stale rows for the pairs being updated, then re-add fresh rows.
            existing = [row for row in existing if row.get("pair") not in pairs]
            detailed = existing + detailed
        # Recalculate gan_found / baseline_found across ALL pairs in the merged data.
        gan_found = any(row.get("variant") == "gan" for row in detailed)
        baseline_found = any(row.get("variant") == "average_baseline" for row in detailed)

    detailed_fields = ["pair", "split", "image", "variant", "source_modality_path", "mri_path", "fused_path", *ALL_METRIC_COLUMNS]
    write_csv(metrics_dir / "diffusion_metrics_detailed.csv", detailed, detailed_fields)

    summary = group_summary(detailed, ["pair", "variant"])
    write_csv(metrics_dir / "diffusion_metrics_summary.csv", summary, ["pair", "variant", *ALL_METRIC_COLUMNS])
    write_csv(metrics_dir / "diffusion_vs_baseline_summary.csv", comparison_summary(summary, "average_baseline"), ["pair", "metric", "diffusion_variant", "comparison_variant", "diffusion_mean", "comparison_mean", "delta_diffusion_minus_comparison"])
    write_csv(metrics_dir / "diffusion_vs_gan_summary.csv", comparison_summary(summary, "gan"), ["pair", "metric", "diffusion_variant", "comparison_variant", "diffusion_mean", "comparison_mean", "delta_diffusion_minus_comparison"])
    write_legacy_pair_files(metrics_dir, summary)
    # For per-pair calls, pass all pairs represented in the merged data so the
    # training summary covers every pair that has been processed so far.
    summary_pairs = list(dict.fromkeys(row["pair"] for row in summary if row.get("pair"))) if args.pair != "all" else pairs
    write_training_summary(metrics_dir, output_root, summary_pairs, args.split, summary)
    write_thesis_summary(metrics_dir, summary, gan_found, baseline_found)


if __name__ == "__main__":
    main()
