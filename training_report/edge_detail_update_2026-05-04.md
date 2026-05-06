# Edge/Detail Fusion Update - 2026-05-04

## Goal

Reduce smooth averaged CT-MRI fusion by training the old checkpoint-compatible GAN to preserve stronger real anatomical edges and fine high-frequency source detail.

## Architecture

- Kept the old checkpoint-compatible generator and discriminator layer names/shapes.
- Added no-parameter source-edge attention inside `FusionGenerator.forward`.
- Edge attention is computed from CT/MRI gradients and multiplies existing feature maps only.
- Edge maps are not concatenated into the saved output and are not overlaid on the final fused image.

## Loss

Total generator loss:

```text
L_G =
  1.0 * L_intensity
+ 5.0 * L_edge
+ 3.0 * L_detail
+ 2.0 * L_ssim
+ 0.1 * L_gan
```

- `L_intensity`: masked L1 to `max(CT, MRI)`.
- `L_edge`: masked L1 between `Sobel(fused)` and `max(Sobel(CT), Sobel(MRI))`.
- `L_detail`: masked L1 between Gaussian high-pass residual of fused and stronger signed source residual per pixel.
- `L_ssim`: `(1 - SSIM(fused, CT)) + (1 - SSIM(fused, MRI))` inside the foreground mask.
- `L_gan`: LSGAN generator loss.

Foreground masking uses pixels where CT or MRI intensity is above a small threshold, with dilation, so background artifacts are not rewarded.

## Validation Outputs

Saved every validation epoch:

- Raw fused PNG in `samples/fused_original/`
- Visualization-only enhanced PNG in `samples/fused_enhanced_visualization/`
- CT | MRI | fused | enhanced comparison in `samples/comparisons/`
- CT | MRI | fused | Sobel CT | Sobel MRI | Sobel fused | edge error in `samples/edge_diagnostics/`

## 5-Epoch Smoke Test

Command used:

```powershell
& 'C:\Users\LENOVO\AppData\Local\Programs\Python\Python311\python.exe' train.py --resume outputs/models/gan/checkpoints/latest_checkpoint.pt --output-dir outputs/models/gan_edge_detail_test_5epochs --epochs 55 --batch-size 4 --micro-batch 2 --num-workers 0 --val-every 1 --max-items 8 --lr 5e-5 --discriminator-lr-factor 0.25 --discriminator-update-interval 2 --lambda-intensity 1 --lambda-gradient 5 --lambda-ssim 2 --lambda-texture 3 --lambda-gan 0.1
```

Best validation epoch: 55.

| Epoch | Val Loss | Val SSIM | Val PSNR | Val SF | Val AG | Val EPI | Val Noise | Val Edge Loss | Val Detail Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 51 | 8.8947 | 0.4291 | 9.1460 | 0.1217 | 0.0307 | 0.8915 | 0.0487 | 0.2150 | 0.0634 |
| 52 | 3.6092 | 0.7351 | 16.5350 | 0.0869 | 0.0232 | 0.8142 | 0.0381 | 0.2994 | 0.0264 |
| 53 | 3.0361 | 0.6135 | 10.6490 | 0.0848 | 0.0224 | 0.7977 | 0.0366 | 0.3235 | 0.0318 |
| 54 | 2.7827 | 0.6586 | 12.1807 | 0.0810 | 0.0217 | 0.7992 | 0.0345 | 0.3101 | 0.0302 |
| 55 | 2.2306 | 0.7589 | 16.3788 | 0.0822 | 0.0220 | 0.7144 | 0.0346 | 0.3065 | 0.0289 |

## Recommendation

Continue from the old epoch-50 checkpoint with the old-compatible architecture and these losses. Use a low learning rate and watch edge diagnostic panels for halos. If halos or external background curves strengthen, reduce `--lambda-gradient` from `5` to `3` and/or `--lambda-gan` from `0.1` to `0.03`.
