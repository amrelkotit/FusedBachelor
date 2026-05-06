import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def enhance_image(image, output_size=512):
    denoised = cv2.fastNlMeansDenoising(image, None, h=3, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.8)
    enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)
    if output_size:
        enhanced = cv2.resize(enhanced, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return enhanced


def iter_images(input_dir):
    input_dir = Path(input_dir)
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create presentation-only enhanced visualizations from saved fused PNGs. "
            "This does not replace or retrain the GAN output."
        )
    )
    parser.add_argument("--input-dir", default="outputs/fused_original", help="Folder containing original fused PNGs.")
    parser.add_argument(
        "--output-dir",
        default="outputs/fused_enhanced_visualization",
        help="Separate folder for enhanced visualization PNGs.",
    )
    parser.add_argument("--output-size", type=int, default=512, help="Bicubic visualization size. Use 0 to keep source size.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_size = args.output_size if args.output_size > 0 else None
    count = 0
    for input_path in iter_images(input_dir):
        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable image: {input_path}")
            continue
        enhanced = enhance_image(image, output_size=output_size)
        relative = input_path.relative_to(input_dir)
        output_path = output_dir / relative.with_name(f"{relative.stem}_enhanced_visualization.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            raise IOError(f"Failed to save image: {output_path}")
        count += 1

    print(f"Enhanced {count} image(s).")
    print(f"Original model outputs remain in: {input_dir}")
    print(f"Presentation-only visualizations saved under: {output_dir}")


if __name__ == "__main__":
    main()
