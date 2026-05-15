import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAN_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "gan"


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]


def copy_if_exists(source, destination_dir, name=None):
    source = Path(source)
    if not source.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (name or source.name)
    shutil.copy2(source, destination)
    return destination


def export_assets(output_dir):
    metrics_dir = GAN_OUTPUT_ROOT / "metrics"
    graph_dir = GAN_OUTPUT_ROOT / "graphs"
    thesis_dir = GAN_OUTPUT_ROOT / "thesis_figures"

    copied = []
    for filename in [
        "final_results_summary.csv",
        "final_results_summary.md",
        "final_results_summary.tex",
    ]:
        path = copy_if_exists(metrics_dir / filename, output_dir)
        if path:
            copied.append(path)

    for pair in PAIRS:
        for filename in [
            f"{pair}_metrics.csv",
            f"{pair}_loss_curve.png",
            f"{pair}_quality_metrics.png",
        ]:
            source_dir = metrics_dir if filename.endswith("_metrics.csv") else graph_dir
            path = copy_if_exists(source_dir / filename, output_dir)
            if path:
                copied.append(path)

        figure = thesis_dir / f"{pair}_qualitative_comparison.png"
        path = copy_if_exists(figure, output_dir)
        if path:
            copied.append(path)

    manifest_sources = sorted((GAN_OUTPUT_ROOT / "images").glob("**/generation_manifest.csv"))
    for manifest in manifest_sources:
        try:
            relative_parent = manifest.parent.relative_to(GAN_OUTPUT_ROOT / "images")
            safe_name = "_".join(relative_parent.parts) + "_generation_manifest.csv"
        except ValueError:
            safe_name = manifest.name
        path = copy_if_exists(manifest, output_dir, safe_name)
        if path:
            copied.append(path)

    return copied


def parse_args():
    parser = argparse.ArgumentParser(description="Copy final thesis assets into one delivery folder.")
    parser.add_argument("--output-dir", default=str(GAN_OUTPUT_ROOT / "final_thesis_assets"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    copied = export_assets(output_dir)
    print(f"Final thesis assets folder: {output_dir}")
    for path in copied:
        print(f"Copied: {path}")
    print(f"Copied {len(copied)} file(s).")


if __name__ == "__main__":
    main()
