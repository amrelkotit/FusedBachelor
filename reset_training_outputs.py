"""
reset_training_outputs.py
=========================
Deletes all generated training outputs (checkpoints, logs, metrics, graphs,
images, reports) so you can start training completely from scratch.

Your CODE (src/, scripts/) and DATA (data/) are never touched.

Usage:
    python reset_training_outputs.py          # dry-run — shows what will be deleted
    python reset_training_outputs.py --confirm  # actually deletes everything
"""

import argparse
import shutil
from pathlib import Path

# ── Root of your project ────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Everything that gets wiped ───────────────────────────────────────────────
# These are the folders and files that contain generated outputs only.
# Code (src/, scripts/) and data (data/) are NOT listed here and will NEVER
# be touched.

DIRS_TO_WIPE = [
    # GAN outputs
    "outputs/models/gan/logs",
    "outputs/models/gan/metrics",
    "outputs/models/gan/metrics_corrected",
    "outputs/models/gan/graphs",
    "outputs/models/gan/images",
    "outputs/models/gan/thesis_figures",
    "outputs/models/gan/final_thesis_assets",
    "outputs/models/gan/final_thesis_assets_corrected",
    # GAN per-pair checkpoints + reports
    "outputs/models/gan/aanlib_ct_mri/checkpoints",
    "outputs/models/gan/aanlib_ct_mri/training_reports",
    "outputs/models/gan/aanlib_pet_mri/checkpoints",
    "outputs/models/gan/aanlib_pet_mri/training_reports",
    "outputs/models/gan/aanlib_spect_mri/checkpoints",
    "outputs/models/gan/aanlib_spect_mri/training_reports",

    # Diffusion outputs
    "outputs/models/diffusion/logs",
    "outputs/models/diffusion/metrics",
    "outputs/models/diffusion/graphs",
    "outputs/models/diffusion/images",
    "outputs/models/diffusion/images_after_msfd",
    # Diffusion per-pair checkpoints, logs, samples
    "outputs/models/diffusion/aanlib_ct_mri/checkpoints",
    "outputs/models/diffusion/aanlib_ct_mri/logs",
    "outputs/models/diffusion/aanlib_ct_mri/samples",
    "outputs/models/diffusion/aanlib_pet_mri/checkpoints",
    "outputs/models/diffusion/aanlib_pet_mri/logs",
    "outputs/models/diffusion/aanlib_pet_mri/samples",
    "outputs/models/diffusion/aanlib_spect_mri/checkpoints",
    "outputs/models/diffusion/aanlib_spect_mri/logs",
    "outputs/models/diffusion/aanlib_spect_mri/samples",

    # Training reports folder at root level
    "training_report",
]


def human_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Reset all training outputs.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete files. Without this flag, runs as a dry-run.",
    )
    args = parser.parse_args()

    dry_run = not args.confirm

    if dry_run:
        print("=" * 60)
        print("  DRY-RUN MODE — nothing will be deleted")
        print("  Add --confirm to actually delete.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  DELETING all training outputs …")
        print("=" * 60)

    total_freed = 0
    deleted_dirs = 0
    skipped = 0

    for rel_path in DIRS_TO_WIPE:
        full_path = ROOT / rel_path
        if not full_path.exists():
            print(f"  [SKIP – not found]  {rel_path}")
            skipped += 1
            continue

        size_str = human_size(full_path)
        file_count = sum(1 for _ in full_path.rglob("*") if _.is_file())
        total_freed += sum(f.stat().st_size for f in full_path.rglob("*") if f.is_file())

        if dry_run:
            print(f"  [WOULD DELETE]  {rel_path}  ({file_count} files, {size_str})")
        else:
            shutil.rmtree(full_path)
            full_path.mkdir(parents=True, exist_ok=True)   # recreate empty folder
            print(f"  [DELETED]       {rel_path}  ({file_count} files, {size_str})")
            deleted_dirs += 1

    print()
    freed_mb = total_freed / (1024 ** 2)
    if dry_run:
        print(f"  Would free ≈ {freed_mb:.1f} MB across {len(DIRS_TO_WIPE) - skipped} directories.")
        print()
        print("  Run again with --confirm to actually reset.")
    else:
        print(f"  Done. Freed ≈ {freed_mb:.1f} MB.  {deleted_dirs} directories cleared.")
        print()
        print("  Your code and data are untouched.")
        print("  You can now run training from scratch — fresh metrics, fresh graphs, fresh everything.")

    print("=" * 60)


if __name__ == "__main__":
    main()
