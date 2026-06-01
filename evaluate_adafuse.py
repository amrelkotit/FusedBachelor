"""
evaluate_adafuse.py
-------------------
Compute FusedBachelor metrics on AdaFuse results for SPECT-MRI and PET-MRI.

AdaFuse results have already been copied and renamed into:
  outputs/models/gan/images/aanlib/test/spect_mri/adafuse_fused/
  outputs/models/gan/images/aanlib/test/pet_mri/adafuse_fused/

Run from the FusedBachelor root:
  python evaluate_adafuse.py
"""

import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PAIRS = ["spect_mri", "pet_mri"]


def main():
    for pair in PAIRS:
        fused_dir = PROJECT_ROOT / "outputs" / "models" / "gan" / "images" / "aanlib" / "test" / pair / "adafuse_fused"
        output_csv = PROJECT_ROOT / "outputs" / "models" / "gan" / "metrics" / f"adafuse_{pair}_metrics.csv"

        if not fused_dir.exists():
            print(f"[SKIP] {pair}: fused dir not found: {fused_dir}")
            continue

        n = len(list(fused_dir.glob("*_fused.png")))
        print(f"\n{'='*60}")
        print(f"Evaluating AdaFuse  |  pair={pair}  |  images={n}")
        print(f"Fused dir : {fused_dir}")
        print(f"Output CSV: {output_csv}")
        print('='*60)

        cmd = [
            sys.executable, "test.py",
            "--pair", pair,
            "--fused-dir", str(fused_dir),
            "--output", str(output_csv),
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"[ERROR] test.py failed for {pair} (exit {result.returncode})")
        else:
            # Print summary row from the CSV
            if output_csv.exists():
                with output_csv.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    summary = rows[-1]
                    fields = ["SSIM", "PSNR", "MI", "EN", "CC", "FMI", "SF", "AG"]
                    print(f"\nAdaFuse {pair} summary:")
                    for field in fields:
                        print(f"  {field:6s}: {summary.get(field, 'N/A')}")

    print("\nDone. Results saved under outputs/models/gan/metrics/adafuse_*_metrics.csv")


if __name__ == "__main__":
    main()
