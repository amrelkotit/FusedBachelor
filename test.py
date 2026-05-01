import argparse
import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from src.data.paired_dataset import DEFAULT_TEST_ROOTS, PairedMedicalImageDataset, verify_dataset_root
from src.evaluation.metrics import evaluate_fusion
from src.models.gan import FusionGenerator


def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    cv2.imwrite(str(path), (image * 255.0).astype("uint8"))


def load_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = FusionGenerator().to(device)
    state_dict = checkpoint["generator"] if isinstance(checkpoint, dict) and "generator" in checkpoint else checkpoint
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator


@torch.no_grad()
def evaluate_dataset(generator, dataset, device, sample_dir, max_samples):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    totals = {}
    for index, batch in enumerate(loader):
        ct = batch["ct"].to(device)
        mri = batch["mri"].to(device)
        fused = generator(ct, mri)
        metrics = evaluate_fusion(fused, mri, ct)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        if index < max_samples:
            save_tensor_image(fused[0], sample_dir / f"sample_{index + 1:03d}.png")

    count = max(1, len(dataset))
    return {key: value / count for key, value in totals.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GAN fusion on AANLIB/test and BRATS_SPLIT/test.")
    parser.add_argument("--checkpoint", default="outputs/models/gan/checkpoints/generator_latest.pt")
    parser.add_argument("--test-roots", nargs="+", default=[str(path) for path in DEFAULT_TEST_ROOTS])
    parser.add_argument("--output-dir", default="outputs/models/gan/test_results")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_generator(args.checkpoint, device)

    rows = []
    overall_totals = {}
    overall_count = 0

    for root in args.test_roots:
        root_path = Path(root)
        dataset_name = "BRATS test" if "BRATS_SPLIT" in str(root_path) else "AANLIB test"
        verify_dataset_root(root_path, dataset_name=dataset_name, strict=True)
        dataset = PairedMedicalImageDataset(root_path, image_size=args.image_size, dataset_name=dataset_name, strict=True)
        metrics = evaluate_dataset(generator, dataset, device, output_dir / "samples" / dataset_name.replace(" ", "_"), args.max_samples)

        row = {"dataset": dataset_name, "pairs": len(dataset), **metrics}
        rows.append(row)
        for key, value in metrics.items():
            overall_totals[key] = overall_totals.get(key, 0.0) + value * len(dataset)
        overall_count += len(dataset)
        print(row)

    if overall_count > 0:
        overall = {key: value / overall_count for key, value in overall_totals.items()}
        rows.append({"dataset": "overall", "pairs": overall_count, **overall})
        print(rows[-1])

    csv_path = output_dir / "test_metrics.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved test metrics: {csv_path}")


if __name__ == "__main__":
    main()
