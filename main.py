import os

import cv2

from src.evaluation.metrics import evaluate_fusion
from src.models.fusion_model import fuse_images
from src.preprocessing.preprocessing import preprocess


dataset_path = r"E:\El Gam3a\My bachelor\Fused bachelor\data\raw\final_dataset\AANLIB\train"
ct_path = os.path.join(dataset_path, "CT")
mri_path = os.path.join(dataset_path, "MRI")

output_path = r"E:\El Gam3a\My bachelor\Fused bachelor\outputs\images"
os.makedirs(output_path, exist_ok=True)

ct_images = sorted(os.listdir(ct_path))
mri_images = sorted(os.listdir(mri_path))

num_pairs = min(len(ct_images), len(mri_images))
if len(ct_images) != len(mri_images):
    print(f"[Warning] CT/MRI count mismatch. Processing {num_pairs} paired files.")

all_metrics = []

for i in range(num_pairs):
    ct_file = os.path.join(ct_path, ct_images[i])
    mri_file = os.path.join(mri_path, mri_images[i])

    ct = preprocess(ct_file)
    mri = preprocess(mri_file)

    fused = fuse_images(ct, mri)
    metrics = evaluate_fusion(fused, mri, ct)
    all_metrics.append(metrics)

    cv2.imwrite(os.path.join(output_path, f"fused_{i + 1:03d}.png"), fused)

if all_metrics:
    print("Enhanced MSFD-like Fusion Done!")
    print("Average metrics:")
    for key in all_metrics[0]:
        avg = sum(item[key] for item in all_metrics) / len(all_metrics)
        print(f"  {key}: {avg:.4f}")
