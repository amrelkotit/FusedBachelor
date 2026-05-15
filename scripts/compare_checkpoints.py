import argparse
import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Subset

from src.data.paired_dataset import AANLIB_ROOT, PairedMedicalImageDataset, aanlib_split_root, normalize_pair, verify_dataset_root
from src.evaluation.metrics import evaluate_fusion
from src.models.gan import FusionGenerator


def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    cv2.imwrite(str(path), (image * 255.0).astype("uint8"))


def combined_score(metrics):
    avg_ssim = 0.5 * (metrics["SSIM_MRI"] + metrics["SSIM_CT"])
    avg_ms = 0.5 * (metrics["MS_MRI"] + metrics["MS_CT"])
    avg_psnr = 0.5 * (metrics["PSNR_MRI"] + metrics["PSNR_CT"])
    psnr_score = max(0.0, min(avg_psnr / 40.0, 1.0))
    sf_score = metrics["SF"] / (metrics["SF"] + 0.1) if metrics["SF"] > 0 else 0.0
    return 0.35 * avg_ssim + 0.25 * avg_ms + 0.20 * psnr_score + 0.20 * sf_score


def load_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = FusionGenerator().to(device)
    state_dict = checkpoint["generator"] if isinstance(checkpoint, dict) and "generator" in checkpoint else checkpoint
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, loader, device, sample_dir, max_samples):
    generator = load_generator(checkpoint_path, device)
    totals = {}
    sample_count = 0
    batches = 0

    for batch in loader:
        ct = batch["ct"].to(device)
        mri = batch["mri"].to(device)
        fused = generator(ct, mri)
        metrics = evaluate_fusion(fused, mri, ct)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value

        if sample_count < max_samples:
            save_tensor_image(fused[0], sample_dir / f"{checkpoint_path.stem}_sample_{sample_count + 1:02d}.png")
            sample_count += 1
        batches += 1

    averaged = {key: value / max(1, batches) for key, value in totals.items()}
    averaged["combined_score"] = combined_score(averaged)
    return averaged


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GAN fusion checkpoints numerically and save visual samples.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], default="ct_mri")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/models/gan/checkpoint_comparison")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pair = normalize_pair(args.pair)
    root = aanlib_split_root(args.dataset_root, pair, "test")
    verify_dataset_root(root, dataset_name=f"AANLIB {pair} test", strict=True, pair=pair)
    dataset = PairedMedicalImageDataset(root, image_size=args.image_size, dataset_name=f"AANLIB {pair} test", strict=True, pair=pair)
    if args.max_items:
        dataset = Subset(dataset, range(min(args.max_items, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    for checkpoint in args.checkpoints:
        checkpoint_path = Path(checkpoint)
        metrics = evaluate_checkpoint(checkpoint_path, loader, device, sample_dir, args.max_samples)
        row = {"checkpoint": str(checkpoint_path), **metrics}
        rows.append(row)
        print(row)

    rows.sort(key=lambda item: item["combined_score"], reverse=True)
    csv_path = output_dir / "checkpoint_comparison.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    recommendation_path = output_dir / "recommendation.txt"
    best = rows[0]
    recommendation_path.write_text(
        f"Recommended checkpoint: {best['checkpoint']}\n"
        f"Combined score: {best['combined_score']:.6f}\n"
        f"Reason: highest balanced score across SSIM, MS, PSNR, and SF.\n"
    )
    print(f"Saved comparison: {csv_path}")
    print(f"Saved recommendation: {recommendation_path}")


if __name__ == "__main__":
    main()
