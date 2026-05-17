import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import diffusion_final_assets_dir, diffusion_image_dir


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def copy_some(source_dir, destination_dir, limit=8):
    copied = []
    if not source_dir.exists():
        return copied
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.png"))[:limit]:
        target = destination_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def copy_some_prefixed(source_dir, destination_dir, prefix, limit=8):
    copied = []
    if not source_dir.exists():
        return copied
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.png"))[:limit]:
        target = destination_dir / f"{prefix}_{path.name}"
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def copy_file_if_exists(source, destination):
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def checkpoint_line(output_root, pair):
    checkpoint_dir = output_root / f"aanlib_{pair}" / "checkpoints"
    ema = checkpoint_dir / "best_diffusion_ema.pt"
    best = checkpoint_dir / "best_diffusion.pt"
    if ema.exists():
        return f"- `{pair}`: `{ema}` preferred; `{best}` fallback."
    if best.exists():
        return f"- `{pair}`: `{best}`."
    return f"- `{pair}`: checkpoint not found in `{checkpoint_dir}`."


def export_assets(output_root, split, max_samples):
    assets_dir = diffusion_final_assets_dir(output_root)
    images_dir = assets_dir / "images"
    panels_dir = assets_dir / "comparison_panels"
    graphs_dir = assets_dir / "graphs"
    metrics_dir = assets_dir / "metrics"
    for folder in [images_dir, panels_dir, graphs_dir, metrics_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    copied = []
    for pair in PAIRS:
        image_root = diffusion_image_dir("aanlib", pair, split, output_root=output_root)
        if pair == "ct_mri":
            copied.extend(copy_some(image_root / "fused_grayscale", images_dir / "ct_mri" / "grayscale", limit=max_samples))
            copied.extend(copy_some(image_root / "fused_grayscale_presentation", images_dir / "ct_mri" / "grayscale_presentation", limit=max_samples))
        else:
            copied.extend(copy_some(image_root / "fused_grayscale", images_dir / pair / "grayscale", limit=max_samples))
            copied.extend(copy_some(image_root / "fused_grayscale_presentation", images_dir / pair / "grayscale_presentation", limit=max_samples))
            copied.extend(copy_some(image_root / "fused_colored", images_dir / pair / "colored", limit=max_samples))
            copied.extend(copy_some(image_root / "fused_colored_presentation", images_dir / pair / "colored_presentation", limit=max_samples))

        comparison_source = output_root / "comparisons" / "aanlib" / pair / split
        copied.extend(copy_some_prefixed(comparison_source, panels_dir, pair, limit=max_samples))

    for graph in sorted((output_root / "graphs").glob("*.png")):
        copied_path = copy_file_if_exists(graph, graphs_dir / graph.name)
        if copied_path:
            copied.append(copied_path)

    for filename in [
        "diffusion_metrics_detailed.csv",
        "diffusion_metrics_summary.csv",
        "diffusion_vs_baseline_summary.csv",
        "diffusion_vs_gan_summary.csv",
        "diffusion_thesis_summary.md",
        "diffusion_final_results_summary.csv",
        "diffusion_final_results_summary.md",
        "diffusion_training_summary.md",
    ]:
        copied_path = copy_file_if_exists(output_root / "metrics" / filename, metrics_dir / filename)
        if copied_path:
            copied.append(copied_path)

    readme = assets_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Diffusion Final Thesis Assets",
                "",
                "This folder collects selected outputs from the independent diffusion pipeline only.",
                "",
                "## Folders",
                "",
                "- `images/ct_mri/grayscale`: selected CT-MRI diffusion grayscale fused images. CT-MRI is grayscale only.",
                "- `images/ct_mri/grayscale_presentation`: lightly enhanced/upscaled CT-MRI display copies for thesis figures.",
                "- `images/pet_mri/grayscale`: selected PET-MRI diffusion grayscale fused images.",
                "- `images/pet_mri/grayscale_presentation`: lightly enhanced/upscaled PET-MRI display copies.",
                "- `images/pet_mri/colored`: selected PET-MRI colored fused images using GAN/source style when available.",
                "- `images/pet_mri/colored_presentation`: upscaled PET-MRI colored display copies.",
                "- `images/spect_mri/grayscale`: selected SPECT-MRI diffusion grayscale fused images.",
                "- `images/spect_mri/grayscale_presentation`: lightly enhanced/upscaled SPECT-MRI display copies.",
                "- `images/spect_mri/colored`: selected SPECT-MRI colored fused images using GAN/source style when available.",
                "- `images/spect_mri/colored_presentation`: upscaled SPECT-MRI colored display copies.",
                "- `comparison_panels`: source, diffusion, GAN when available, and average-baseline panels.",
                "- `graphs`: training, validation, and metric comparison graphs.",
                "- `metrics`: detailed and summary CSV/Markdown metric files.",
                "",
                "## Color Rules",
                "",
                "- CT-MRI has no colored diffusion export.",
                "- PET-MRI and SPECT-MRI include grayscale and colored diffusion outputs.",
                "- Colored PET/SPECT outputs use HSV color transfer: hue/saturation from GAN-style color first, source RGB color second, and colormap only as fallback.",
                "- Diffusion grayscale controls brightness/detail; saturation and contrast are mildly boosted while a background mask keeps non-brain regions quiet.",
                "",
                "## Checkpoints",
                "",
                "Generation uses `best_diffusion.pt` by command and prefers `best_diffusion_ema.pt` when a valid sibling EMA checkpoint is available.",
                "",
                checkpoint_line(output_root, "ct_mri"),
                checkpoint_line(output_root, "pet_mri"),
                checkpoint_line(output_root, "spect_mri"),
                "",
                "## Metrics",
                "",
                "Metrics include SSIM, PSNR, mutual information, entropy, spatial frequency, average gradient, and edge intensity. CC and FMI are also exported when available from the evaluator.",
                "",
                "## Thesis Use",
                "",
                "Use `comparison_panels` for the main side-by-side thesis figures, `images/*/grayscale` for native evaluation examples, `images/*/*_presentation` for display in the thesis results chapter, `images/pet_mri/colored_presentation` and `images/spect_mri/colored_presentation` for functional color visualization, `graphs` for training/metric plots, and `metrics/diffusion_thesis_summary.md` for concise results narration.",
                "",
                f"Exported files: {len(copied)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    copied.append(readme)
    return assets_dir, copied


def parse_args():
    parser = argparse.ArgumentParser(description="Export diffusion final thesis assets.")
    parser.add_argument("--output-root", default="outputs/models/diffusion")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--max-samples", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    assets_dir, copied = export_assets(resolve_path(args.output_root), args.split, args.max_samples)
    print(f"Final thesis assets folder: {assets_dir}")
    for path in copied:
        print(f"Saved asset: {path}")
    print(f"Saved {len(copied)} diffusion thesis asset(s).")


if __name__ == "__main__":
    main()
