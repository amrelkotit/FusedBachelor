from src.models.fusion_model import FusionNet
from src.models.fusion_model import FusionNet, fuse_images
from src.preprocessing.preprocessing import preprocess
import torch
import cv2
import numpy as np
import os
from src.preprocessing.preprocessing import preprocess


dataset_path = r"E:\El Gam3a\My bachelor\Fused bachelor\data\raw\final_dataset\AANLIB\train"

ct_path = os.path.join(dataset_path, "CT")
mri_path = os.path.join(dataset_path, "MRI")

output_path = r"E:\El Gam3a\My bachelor\Fused bachelor\outputs\images"
os.makedirs(output_path, exist_ok=True)

ct_images = sorted(os.listdir(ct_path))
mri_images = sorted(os.listdir(mri_path))

for i in range(len(ct_images)):
    ct_file = os.path.join(ct_path, ct_images[i])
    mri_file = os.path.join(mri_path, mri_images[i])

    ct = preprocess(ct_file)
    mri = preprocess(mri_file)

    fused = fuse_images(ct, mri)

    cv2.imwrite(os.path.join(output_path, f"fused_{i+1:03d}.png"), fused)
print("Enhanced Fusion Done!")
# =========================
# Preprocessing
# =========================
# mri = preprocess(os.path.join(mri_path, mri_images[i]))
# ct  = preprocess(os.path.join(ct_path, ct_images[i]))

# # Add batch dimension
# mri = mri.unsqueeze(0)
# ct  = ct.unsqueeze(0)

# # =========================
# # Deep Model Fusion (still untrained)
# # =========================
# model = FusionNet()

# with torch.no_grad():
#     fused = model(mri, ct)

# fused_img = fused.squeeze().detach().numpy()
# fused_img = (fused_img * 255).astype("uint8")

# # Ensure folder exists
# os.makedirs("outputs/images", exist_ok=True)



# # =========================
# # 🔥 Adaptive Fusion (BEST NOW)
# # =========================

# mri_np = mri.squeeze().numpy()
# ct_np  = ct.squeeze().numpy()

# # Gradients
# mri_np = mri.squeeze().numpy()
# ct_np  = ct.squeeze().numpy()

# # 🔥 FIX TYPE
# mri_np = mri_np.astype(np.float64)
# ct_np  = ct_np.astype(np.float64)

# # Gradients
# mri_grad = cv2.Laplacian(mri_np, cv2.CV_64F)
# ct_grad  = cv2.Laplacian(ct_np, cv2.CV_64F)

# mri_w = np.abs(mri_grad)
# ct_w  = np.abs(ct_grad)

# sum_w = mri_w + ct_w + 1e-8
# mri_w /= sum_w
# ct_w  /= sum_w

# fused_np = mri_w * mri_np + ct_w * ct_np

# fused_np = cv2.normalize(fused_np, None, 0, 1, cv2.NORM_MINMAX)

# fused_img = (fused_np * 255).astype("uint8")

# mri_w = np.abs(mri_grad)
# ct_w  = np.abs(ct_grad)

# # Normalize weights
# sum_w = mri_w + ct_w + 1e-8
# mri_w /= sum_w
# ct_w  /= sum_w

# # Fusion
# fused_np = mri_w * mri_np + ct_w * ct_np

# # Normalize
# fused_np = cv2.normalize(fused_np, None, 0, 1, cv2.NORM_MINMAX)

# fused_img = (fused_np * 255).astype("uint8")

# cv2.imwrite("outputs/images/fused_adaptive.png", fused_img)
# print("Adaptive fusion saved!")