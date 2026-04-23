from pathlib import Path

import cv2

from src.data.paired_dataset import DEFAULT_TEST_ROOTS, PairedMedicalImageDataset, verify_dataset_root
from src.evaluation.metrics import evaluate_fusion
from src.models.fusion_model import fuse_images


output_root = Path(r"E:\El Gam3a\My bachelor\Fused bachelor\outputs\images")

all_metrics = []
for root in DEFAULT_TEST_ROOTS:
    dataset_name = "BRATS" if "BRATS_SPLIT" in str(root) else "AANLIB"
    verify_dataset_root(root, dataset_name=f"{dataset_name} test", strict=True)
    dataset = PairedMedicalImageDataset(root, dataset_name=f"{dataset_name} test", strict=True)
    dataset_output = output_root / f"{dataset_name.lower()}_test"
    dataset_output.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(dataset, start=1):
        ct = sample["ct"]
        mri = sample["mri"]
        fused = fuse_images(ct, mri)
        metrics = evaluate_fusion(fused, mri, ct)
        all_metrics.append(metrics)

        cv2.imwrite(str(dataset_output / f"fused_{index:03d}.png"), fused)

if all_metrics:
    print("Enhanced MSFD-like Fusion Done!")
    print("Average metrics:")
    for key in all_metrics[0]:
        avg = sum(item[key] for item in all_metrics) / len(all_metrics)
        print(f"  {key}: {avg:.4f}")
