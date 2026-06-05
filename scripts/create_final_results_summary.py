import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAN_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "gan"


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
METRIC_COLUMNS = ["SSIM", "PSNR", "MI", "EN", "CC", "FMI", "SF", "AG"]
PAIR_LABELS = {
    "ct_mri": "CT-MRI",
    "pet_mri": "PET-MRI",
    "spect_mri": "SPECT-MRI",
}


def parse_mean_std(value):
    if value is None:
        return ""
    text = str(value).strip()
    if "+/-" not in text:
        return text
    mean, std = [part.strip() for part in text.split("+/-", 1)]
    return f"{mean} \u00b1 {std}"


def read_summary_row(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in reversed(rows):
        if str(row.get("image", "")).strip().lower() == "mean +/- std":
            return row
    if rows:
        return rows[-1]
    raise ValueError(f"No rows found in metrics CSV: {path}")


def build_summary(metrics_dir):
    rows = []
    for pair in PAIRS:
        path = metrics_dir / f"{pair}_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics CSV: {path}")
        summary = read_summary_row(path)
        rows.append(
            {
                "Pair": PAIR_LABELS[pair],
                **{column: parse_mean_std(summary.get(column, "")) for column in METRIC_COLUMNS},
            }
        )
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig writes a BOM so Excel decodes the "±" character correctly
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Pair", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    header = ["Pair", *METRIC_COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in header) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%").replace("\u00b1", "$\\pm$")


def write_latex(rows, path):
    header = ["Pair", *METRIC_COLUMNS]
    lines = [
        "\\begin{tabular}{l" + "c" * len(METRIC_COLUMNS) + "}",
        "\\hline",
        " & ".join(header) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row[column]) for column in header) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Create final mean/std metrics summary tables.")
    parser.add_argument("--metrics-dir", default=str(GAN_OUTPUT_ROOT / "metrics"))
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_absolute():
        metrics_dir = PROJECT_ROOT / metrics_dir

    rows = build_summary(metrics_dir)
    csv_path = metrics_dir / "final_results_summary.csv"
    md_path = metrics_dir / "final_results_summary.md"
    tex_path = metrics_dir / "final_results_summary.tex"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_latex(rows, tex_path)

    print(f"Saved final CSV summary: {csv_path}")
    print(f"Saved final Markdown summary: {md_path}")
    print(f"Saved final LaTeX summary: {tex_path}")


if __name__ == "__main__":
    main()
