import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paired_dataset import AANLIB_ROOT, PairedMedicalImageDataset, aanlib_split_root, normalize_pair
from src.diffusion.utils import initial_fusion, save_tensor_image


PAIRS = ["ct_mri", "pet_mri", "spect_mri"]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_pair_baseline(dataset_root, model_root, pair, split, image_size, max_items=None):
    pair = normalize_pair(pair)
    dataset = PairedMedicalImageDataset(
        aanlib_split_root(dataset_root, pair, split),
        image_size=image_size,
        max_items=max_items,
        dataset_name=f"AANLIB {pair} {split}",
        strict=True,
        pair=pair,
    )
    output_dir = model_root / "baselines" / "average" / "aanlib" / pair / split
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(dataset):
        fused = initial_fusion(sample["source1"].unsqueeze(0), sample["source2"].unsqueeze(0))
        save_tensor_image(fused[0], output_dir / f"{index:04d}_average.png")
    print(f"Saved {len(dataset)} average baseline image(s): {output_dir}")
    if pair == "ct_mri":
        print("CT-MRI baseline is grayscale only; no colored CT baseline is created.")


def parse_args():
    parser = argparse.ArgumentParser(description="Create average fusion baselines for diffusion comparisons.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--model-root", default="outputs/models/diffusion")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    model_root = resolve_path(args.model_root)
    pairs = PAIRS if args.pair == "all" else [normalize_pair(args.pair)]
    for pair in pairs:
        create_pair_baseline(dataset_root, model_root, pair, args.split, args.image_size, max_items=args.max_items)


if __name__ == "__main__":
    main()
