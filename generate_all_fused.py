import argparse
import csv
from pathlib import Path
import re

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.gan import FusionGenerator


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "final_dataset"
DEFAULT_CHECKPOINT = Path("outputs") / "models" / "gan_continued_from_50" / "checkpoints" / "best_generator.pt"
DEFAULT_OUTPUT_DIR = Path("outputs") / "models" / "gan_continued_from_50" / "images"
DATASET_SPLITS = (
    ("AANLIB", "train"),
    ("AANLIB", "test"),
    ("BRATS", "train"),
    ("BRATS", "test"),
)


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_modality_dir(root_dir, names):
    for name in names:
        path = root_dir / name
        if path.is_dir():
            return path
    expected = ", ".join(str(root_dir / name) for name in names)
    raise FileNotFoundError(f"Missing modality folder. Expected one of: {expected}")


def list_images(folder):
    return sorted(
        (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )


def read_grayscale(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    interpolation = cv2.INTER_AREA if image.shape[0] > image_size or image.shape[1] > image_size else cv2.INTER_CUBIC
    image = cv2.resize(image, (image_size, image_size), interpolation=interpolation)
    image = image.astype("float32")
    min_value = image.min()
    max_value = image.max()
    image = (image - min_value) / (max_value - min_value + 1e-8)
    return torch.from_numpy(image).unsqueeze(0)


class PairedSplitDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.ct_dir = find_modality_dir(self.root_dir, ("ct", "CT"))
        self.mri_dir = find_modality_dir(self.root_dir, ("mri", "MRI"))

        ct_by_name = {path.name.lower(): path for path in list_images(self.ct_dir)}
        mri_by_name = {path.name.lower(): path for path in list_images(self.mri_dir)}
        matched_names = sorted(set(ct_by_name) & set(mri_by_name), key=lambda name: natural_key(Path(name)))
        if not matched_names:
            raise ValueError(f"No matched CT/MRI image pairs found in: {self.root_dir}")

        self.pairs = [(ct_by_name[name], mri_by_name[name]) for name in matched_names]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ct_path, mri_path = self.pairs[index]
        return {
            "ct": read_grayscale(ct_path, self.image_size),
            "mri": read_grayscale(mri_path, self.image_size),
            "ct_path": str(ct_path),
            "mri_path": str(mri_path),
        }


def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    ok = cv2.imwrite(str(path), (image * 255.0).round().astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def tensor_to_uint8(tensor):
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    return (image * 255.0).round().astype("uint8")


def enhance_visualization_uint8(image, output_size=512):
    denoised = cv2.fastNlMeansDenoising(image, None, h=3, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.8)
    enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)
    if output_size and enhanced.shape[:2] != (output_size, output_size):
        enhanced = cv2.resize(enhanced, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return enhanced


def save_uint8_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def normalize_state_dict_keys(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_generator_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("generator", "generator_state_dict", "model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_generator(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_state_dict_keys(extract_generator_state_dict(checkpoint))

    generator = FusionGenerator().to(device)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator, checkpoint_path


def ensure_output_tree(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, split_name in DATASET_SPLITS:
        (output_dir / "fused_original" / dataset_name / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "fused_enhanced_visualization" / dataset_name / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "comparisons" / dataset_name / split_name).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def generate_split(generator, dataset_root, output_dir, dataset_name, split_name, args, device, checkpoint_path):
    split_root = dataset_root / dataset_name / split_name
    dataset = PairedSplitDataset(split_root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    original_dir = output_dir / "fused_original" / dataset_name / split_name
    enhanced_dir = output_dir / "fused_enhanced_visualization" / dataset_name / split_name
    comparison_dir = output_dir / "comparisons" / dataset_name / split_name

    rows = []
    written = 0
    for batch in loader:
        ct = batch["ct"].to(device)
        mri = batch["mri"].to(device)
        fused = generator(ct, mri)

        batch_size = fused.shape[0]
        for batch_index in range(batch_size):
            if args.max_items is not None and written >= args.max_items:
                return rows

            original_path = original_dir / f"{written}_fused.png"
            enhanced_path = enhanced_dir / f"{written}_fused_enhanced_512.png"
            comparison_path = comparison_dir / f"{written}_comparison.png"
            save_tensor_image(fused[batch_index], original_path)
            fused_u8 = tensor_to_uint8(fused[batch_index])
            enhanced_u8 = enhance_visualization_uint8(fused_u8, output_size=args.visualization_size)
            save_uint8_image(enhanced_u8, enhanced_path)

            if args.save_comparisons:
                ct_u8 = tensor_to_uint8(batch["ct"][batch_index])
                mri_u8 = tensor_to_uint8(batch["mri"][batch_index])
                enhanced_for_panel = cv2.resize(enhanced_u8, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                panel = cv2.hconcat([ct_u8, mri_u8, fused_u8, enhanced_for_panel])
                save_uint8_image(panel, comparison_path)
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "index": written,
                    "original_output_path": str(original_path.relative_to(PROJECT_ROOT)),
                    "enhanced_visualization_path": str(enhanced_path.relative_to(PROJECT_ROOT)),
                    "comparison_path": str(comparison_path.relative_to(PROJECT_ROOT)) if args.save_comparisons else "",
                    "ct_path": batch["ct_path"][batch_index],
                    "mri_path": batch["mri_path"][batch_index],
                    "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                }
            )
            written += 1

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Generate all fused GAN PNG images for AANLIB and BRATS train/test splits.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Generator checkpoint to load.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Root containing AANLIB and BRATS folders.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Root folder for generated fused images and manifest.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None, help="Optional per-split limit for quick checks.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--visualization-size", type=int, default=512, help="Enhanced visualization output size; raw model output remains image-size.")
    parser.add_argument("--save-comparisons", action="store_true", default=True, help="Save CT | MRI | fused | enhanced comparison panels.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    ensure_output_tree(output_dir)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    generator, checkpoint_path = load_generator(args.checkpoint, device)

    manifest_rows = []
    for dataset_name, split_name in DATASET_SPLITS:
        rows = generate_split(generator, dataset_root, output_dir, dataset_name, split_name, args, device, checkpoint_path)
        manifest_rows.extend(rows)
        print(f"{dataset_name}/{split_name}: generated {len(rows)} fused image(s)")

    manifest_path = output_dir / "generation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "dataset",
            "split",
            "index",
            "original_output_path",
            "enhanced_visualization_path",
            "comparison_path",
            "ct_path",
            "mri_path",
            "checkpoint",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved fused images under: {output_dir}")


if __name__ == "__main__":
    main()
