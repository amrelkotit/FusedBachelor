"""
Build outputs/models/diffusion/metrics/all_pairs_summary.csv from the existing
per_pair_metrics_*.csv files — the diffusion equivalent of the GAN all_pairs_summary.csv.

Columns match the GAN summary (pair, SSIM, PSNR, MI, EN, CC, FMI, SF, AG).
CC and FMI are included when available (from diffusion_metrics_summary.csv for pairs
that have been fully evaluated); otherwise left empty.

Run:
    python scripts/make_diffusion_all_pairs_summary.py
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "models" / "diffusion" / "metrics"
PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
OUTPUT_COLUMNS = ["pair", "SSIM", "PSNR", "MI", "EN", "CC", "FMI", "SF", "AG"]


def read_summary_optional(metrics_dir):
    """Return {pair: {metric: value}} from diffusion_metrics_summary.csv for best variant per pair."""
    path = metrics_dir / "diffusion_metrics_summary.csv"
    if not path.exists():
        return {}
    # utf-8-sig strips the BOM that write_csv writes, so the "pair" key is intact
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    def ssim_mean(row):
        raw = str(row.get("SSIM", "0"))
        text = raw.split("±")[0].split("+/-")[0].strip()
        try:
            return float(text)
        except ValueError:
            return 0.0

    diffusion_variants = {"diffusion_fused_grayscale", "diffusion_fused_colored", "diffusion_fused_original"}
    best = {}
    for row in rows:
        pair = row.get("pair", "")
        variant = row.get("variant", "")
        if pair not in PAIRS or variant not in diffusion_variants:
            continue
        if pair not in best or ssim_mean(row) > ssim_mean(best[pair]):
            best[pair] = row
    return best


def read_per_pair(metrics_dir, pair):
    path = metrics_dir / f"per_pair_metrics_{pair}.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("image", "").startswith("mean"):
            return row
    return rows[0] if rows else None


def main():
    optional = read_summary_optional(METRICS_DIR)
    output_rows = []
    for pair in PAIRS:
        per_pair = read_per_pair(METRICS_DIR, pair)
        if per_pair is None:
            print(f"[skip] no per_pair_metrics_{pair}.csv found")
            continue
        opt = optional.get(pair, {})
        row = {
            "pair": pair,
            "SSIM": per_pair.get("SSIM", ""),
            "PSNR": per_pair.get("PSNR", ""),
            "MI":   per_pair.get("MI", ""),
            "EN":   per_pair.get("EN", ""),
            "CC":   opt.get("CC", ""),
            "FMI":  opt.get("FMI", ""),
            "SF":   per_pair.get("SF", ""),
            "AG":   per_pair.get("AG", ""),
        }
        output_rows.append(row)
        print(f"[ok] {pair}")

    if not output_rows:
        print("No data found — nothing written.")
        return

    out_path = METRICS_DIR / "all_pairs_diffusion_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nSaved: {out_path}")

    print("\n--- all_pairs_diffusion_summary ---")
    print("\t".join(OUTPUT_COLUMNS))
    for row in output_rows:
        print("\t".join(str(row[c]) for c in OUTPUT_COLUMNS))


if __name__ == "__main__":
    main()
