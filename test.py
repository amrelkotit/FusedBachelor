import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch
from tqdm.auto import tqdm

from src.data.paired_dataset import (
    AANLIB_ROOT,
    BRATS_ROOT,
    PairedMedicalImageDataset,
    aanlib_split_root,
    gan_graph_dir,
    gan_image_dir,
    gan_metrics_dir,
    normalize_pair,
)
from src.evaluation.metrics import evaluate_fusion


PROJECT_ROOT = Path(__file__).resolve().parent
METRIC_FIELDS = ["SSIM", "PSNR", "MI", "EN", "CC", "FMI", "SF", "AG"]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_tensor(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Metric calculation failure. Could not read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def summarize(rows):
    summary = {"image": "mean +/- std"}
    for field in METRIC_FIELDS:
        values = torch.tensor([float(row[field]) for row in rows], dtype=torch.float32)
        summary[field] = f"{values.mean().item():.3f}±{values.std(unbiased=False).item():.2f}"
    return summary


def metrics_for_pair(dataset, fused_dir, image_size):
    rows = []
    fused_paths = sorted(fused_dir.glob("*_fused.png"))
    if not fused_paths:
        raise FileNotFoundError(f"Metric calculation failure. No fused_original images found in: {fused_dir}")
    count = min(len(dataset), len(fused_paths))
    bar = tqdm(range(count), desc=f"Evaluate {dataset.pair}", leave=False, dynamic_ncols=True, file=sys.stdout, ascii=False)
    for index in bar:
        sample = dataset[index]
        fused = read_tensor(fused_paths[index], image_size)
        source1 = sample["source1"].unsqueeze(0)
        source2 = sample["source2"].unsqueeze(0)
        metrics = evaluate_fusion(fused, source2, source1)
        row = {
            "image": fused_paths[index].name,
            "source1_path": sample["source1_path"],
            "source2_path": sample["source2_path"],
            "fused_path": str(fused_paths[index]),
            # Use the aggregated keys from evaluate_fusion:
            #   PSNR  = avg(PSNR_MRI, PSNR_CT)  with data_range=1.0 (images in [0,1])
            #   SSIM  = avg(SSIM_MRI, SSIM_CT)
            #   MI    = MI_MRI + MI_CT  (sum, 256-bin histograms)
            "SSIM": metrics["SSIM"],
            "PSNR": metrics["PSNR"],
            "MI":   metrics["MI"],
            "EN":   metrics["EN"],
            "CC":   metrics["CC"],
            "FMI":  metrics["FMI"],
            "SF":   metrics["SF"],
            "AG":   metrics["AG"],
        }
        rows.append(row)
    return rows


def write_metrics_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "source1_path", "source2_path", "fused_path", *METRIC_FIELDS]
    # utf-8-sig writes a BOM so Excel decodes the "±" character correctly
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summarize(rows))


def update_summary_files(metrics_dir):
    metric_files = sorted(path for path in metrics_dir.glob("*_metrics.csv") if not path.name.startswith("brats_"))
    rows = []
    for path in metric_files:
        with path.open(newline="", encoding="utf-8") as csv_file:
            data = list(csv.DictReader(csv_file))
        if not data:
            continue
        summary = data[-1]
        pair = path.name.removesuffix("_metrics.csv")
        rows.append({"pair": pair, **{field: summary[field] for field in METRIC_FIELDS}})
    if not rows:
        return
    for filename in ("all_pairs_gan_summary.csv", "thesis_comparison_table.csv"):
        with (metrics_dir / filename).open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["pair", *METRIC_FIELDS])
            writer.writeheader()
            writer.writerows(rows)
    md_lines = ["| Pair | " + " | ".join(METRIC_FIELDS) + " |", "|" + "---|" * (len(METRIC_FIELDS) + 1)]
    for row in rows:
        md_lines.append("| " + row["pair"] + " | " + " | ".join(row[field] for field in METRIC_FIELDS) + " |")
    (metrics_dir / "thesis_comparison_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate metrics from fused_original images only.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], default="ct_mri")
    parser.add_argument("--split", choices=["test", "train", "val"], default="test")
    parser.add_argument("--fused-dir", default=None, help="Defaults to outputs/models/gan/images/aanlib/<split>/<pair>/fused_original.")
    parser.add_argument("--output", default=None, help="Defaults to outputs/models/gan/metrics/<pair>_metrics.csv.")
    parser.add_argument("--external-test", choices=["brats"], default=None)
    parser.add_argument("--brats-root", default=str(BRATS_ROOT / "test"))
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    pair = normalize_pair(args.pair)
    dataset_name = "brats" if args.external_test == "brats" else "aanlib"
    split_root = resolve_path(args.brats_root) if dataset_name == "brats" else aanlib_split_root(resolve_path(args.dataset_root), pair, args.split)
    fused_dir = resolve_path(args.fused_dir) if args.fused_dir else gan_image_dir(dataset_name, args.split, pair) / "fused_original"
    output_path = resolve_path(args.output) if args.output else gan_metrics_dir() / (f"brats_{pair}_metrics.csv" if dataset_name == "brats" else f"{pair}_metrics.csv")

    print(f"Checkpoint path: metrics from existing fused_original images")
    print(f"Fused image folder: {fused_dir}")
    print(f"Graph folder: {gan_graph_dir()}")
    print(f"Metrics folder: {gan_metrics_dir()}")
    if dataset_name == "brats":
        print("Dataset label: BRATS external generalization testing")

    dataset = PairedMedicalImageDataset(split_root, image_size=args.image_size, dataset_name=f"{dataset_name} {args.split}", strict=True, pair=pair)
    rows = metrics_for_pair(dataset, fused_dir, args.image_size)
    write_metrics_csv(rows, output_path)
    update_summary_files(gan_metrics_dir())
    print(f"Saved metrics: {output_path}")
    print("Metrics source: fused_original only")
    print("Metrics are computed from fused_original only. fused_color is for visualization only.")


if __name__ == "__main__":
    main()
