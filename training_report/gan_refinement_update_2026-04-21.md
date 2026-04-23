# GAN Refinement Update - 2026-04-21

## Summary

The GAN training code was refined in-place without rebuilding the architecture. The current Generator, Discriminator 1, and Discriminator 2 structure was preserved.

## Code Improvements

- Strengthened edge preservation in `src/training/losses.py`.
- Added multi-scale Sobel edge loss with stronger weighting on source edges.
- Added a high-frequency detail penalty to reduce over-smoothing.
- Reduced default GAN pressure from `lambda_gan=0.01` to `0.005`.
- Increased default gradient/detail pressure from `lambda_grad=5.0` to `7.5`.
- Added real-label smoothing in discriminator loss to reduce discriminator dominance.
- Added discriminator learning-rate balancing in `train.py` and `src/training/trainer.py`.
- Generator LR remains `lr`; discriminator LR is now `lr * discriminator_lr_factor`, default `0.5`.
- Added persistent CSV and JSONL logging after every epoch:
  - `outputs/gan/logs/training_log.csv`
  - `outputs/gan/logs/training_log.jsonl`
- Added combined validation score for best-model selection.
- Best checkpoint is now based on SSIM, MS, PSNR, and SF instead of one metric.
- Added early stopping with patience and minimum improvement threshold.
- Kept auto-resume and explicit checkpoint resume support.
- Added fixed edge-aware feature modulation in `src/models/gan.py` without new trainable parameters, so existing checkpoints remain loadable.
- Added the same fixed edge awareness to `src/models/feature_extractor.py`.
- Improved `src/fusion/decomposition.py` with reflection-padded Gaussian blur, adaptive low-frequency fusion, and soft high-frequency coefficient fusion.
- Added `scripts/compare_checkpoints.py` to compare checkpoints numerically and save visual samples.

## Checkpoint Comparison Status

Candidate checkpoint/sample artifacts exist for epochs 18, 21, and 24:

- `outputs/gan/checkpoints/gan_epoch_018.pt`
- `outputs/gan/checkpoints/gan_epoch_021.pt`
- `outputs/gan/checkpoints/gan_epoch_024.pt`
- matching sample images under `outputs/gan/samples`

Numerical comparison could not be run in this shell because `python` is not discoverable on PATH, and no previous persistent metric/loss logs exist for the already completed runs. The new logging code fixes this for future runs.

Once a working Python environment is available, run:

```powershell
python scripts/compare_checkpoints.py --checkpoints outputs/gan/checkpoints/gan_epoch_018.pt outputs/gan/checkpoints/gan_epoch_021.pt outputs/gan/checkpoints/gan_epoch_024.pt --output-dir outputs/checkpoint_comparison_18_21_24
```

Based on the stated visual/numerical observation that epochs 18, 21, and 24 are strongest, epoch 24 is the recommended refinement starting checkpoint because it is the latest checkpoint before the reported discriminator-dominance region.

## Recommended Final Model For Current Artifacts

Use `gan_epoch_024.pt` as the current candidate final model unless visual inspection clearly favors epoch 18 or 21. Epoch 24 is the best practical compromise between learned fusion strength and avoiding later instability.

## Recommended Next Run

Start a refined run from epoch 24 into a separate output folder so old epoch 25+ checkpoints are not overwritten:

```powershell
python train.py --epochs 40 --batch-size 4 --resume outputs/gan/checkpoints/gan_epoch_024.pt --output-dir outputs/gan_refined
```

Optional stronger-edge run if outputs are still smooth:

```powershell
python train.py --epochs 40 --batch-size 4 --resume outputs/gan/checkpoints/gan_epoch_024.pt --output-dir outputs/gan_refined_edges --lambda-grad 9.0 --lambda-gan 0.004
```

## Expected Effect

- Sharper skull/tissue boundaries from multi-scale Sobel loss.
- Less over-smoothed texture from high-frequency detail preservation.
- More stable adversarial training from lower discriminator LR and label smoothing.
- Better model selection from combined validation scoring.
- Easier future comparison due persistent logs.
- Cleaner rule-based fusion targets from improved decomposition/reconstruction.
- More edge-aware generator features without breaking checkpoint compatibility.

## Warnings

- Current shell still cannot run `python` directly from PATH.
- Existing epochs 18/21/24 cannot be numerically compared retroactively without a working Python environment and/or saved metric logs.
- Future refined runs will produce the needed CSV/JSONL metrics automatically.
