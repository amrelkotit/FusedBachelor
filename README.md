# Medical Image Fusion GAN

This project uses the Harvard AANLIB dataset as the primary thesis dataset for training, validation, testing, generation, metrics, graphs, and thesis figures. BRATS is not part of the primary benchmark and is available only as explicit external generalization testing.

## Dataset Structure

```text
data/raw/AANLIB/CT-MRI/train/CT
data/raw/AANLIB/CT-MRI/train/MRI
data/raw/AANLIB/CT-MRI/test/CT
data/raw/AANLIB/CT-MRI/test/MRI

data/raw/AANLIB/PET-MRI/train/PET
data/raw/AANLIB/PET-MRI/train/MRI
data/raw/AANLIB/PET-MRI/test/PET
data/raw/AANLIB/PET-MRI/test/MRI

data/raw/AANLIB/SPECT-MRI/train/SPECT
data/raw/AANLIB/SPECT-MRI/train/MRI
data/raw/AANLIB/SPECT-MRI/test/SPECT
data/raw/AANLIB/SPECT-MRI/test/MRI
```

Uppercase and lowercase modality folders are supported: `CT/ct`, `MRI/mri`, `PET/pet`, and `SPECT/spect`.

Supported pairs:

```text
ct_mri    -> CT | MRI | Fused
pet_mri   -> PET | MRI | Fused
spect_mri -> SPECT | MRI | Fused
```

If a `val` folder exists under a pair, it is used directly. Otherwise, training creates a deterministic internal validation split with seed `42` and `--val-split 0.15`.

## Output Layout

All default GAN outputs are under:

```text
outputs/models/gan/
```

Main folders:

```text
outputs/models/gan/aanlib_<pair>/checkpoints/
outputs/models/gan/images/aanlib/<split>/<pair>/fused_original/
outputs/models/gan/images/aanlib/<split>/<pair>/fused_color/
outputs/models/gan/images/aanlib/<split>/<pair>/comparison_panels/
outputs/models/gan/images/aanlib/<split>/<pair>/comparison_panels_color/
outputs/models/gan/graphs/
outputs/models/gan/metrics/
outputs/models/gan/logs/
outputs/models/gan/thesis_figures/
```

Checkpoints include:

```text
best_generator.pt
latest_generator.pt
best_checkpoint.pt
latest_checkpoint.pt
```

`fused_original` is the raw grayscale generator output and is the only fused output used for evaluation and metrics.

`fused_color` is a visualization-only output for `pet_mri` and `spect_mri` when the PET or SPECT source image is actually colored. It preserves PET/SPECT hue and saturation and uses the raw fused grayscale output as the brightness channel. `fused_color` is not created for CT-MRI, is not created when the functional source image is grayscale, and is never used in evaluation.

`comparison_panels` keeps the existing grayscale panels. `comparison_panels_color` is created only for colored PET/SPECT sources and shows PET/SPECT color, MRI grayscale, and fused color.

## Training Commands

```powershell
python -u train.py --dataset-root data/raw/AANLIB --pair ct_mri --epochs 120 --batch-size 204 --micro-batch 4 --auto-resume *>&1 | Tee-Object -FilePath "outputs/models/gan/logs/ct_mri_training_console.txt"
python -u train.py --dataset-root data/raw/AANLIB --pair pet_mri --epochs 120 --batch-size 204 --micro-batch 4 --auto-resume *>&1 | Tee-Object -FilePath "outputs/models/gan/logs/pet_mri_training_console.txt"
python -u train.py --dataset-root data/raw/AANLIB --pair spect_mri --epochs 120 --batch-size 204 --micro-batch 4 --auto-resume *>&1 | Tee-Object -FilePath "outputs/models/gan/logs/spect_mri_training_console.txt"
```

Full epoch logs are persisted automatically as:

```text
outputs/models/gan/logs/ct_mri_training.log
outputs/models/gan/logs/pet_mri_training.log
outputs/models/gan/logs/spect_mri_training.log
```

Best-model selection uses highest SSIM first, then highest FMI, highest PSNR, and lowest validation loss.

## Generation Commands

```powershell
python generate_all_fused.py --dataset-root data/raw/AANLIB --pair ct_mri --split test --checkpoint outputs/models/gan/aanlib_ct_mri/checkpoints/best_generator.pt
python generate_all_fused.py --dataset-root data/raw/AANLIB --pair pet_mri --split test --checkpoint outputs/models/gan/aanlib_pet_mri/checkpoints/best_generator.pt
python generate_all_fused.py --dataset-root data/raw/AANLIB --pair spect_mri --split test --checkpoint outputs/models/gan/aanlib_spect_mri/checkpoints/best_generator.pt
```

Train-split generation uses `--split train`.

## Evaluation Commands

```powershell
python test.py --dataset-root data/raw/AANLIB --pair ct_mri --split test
python test.py --dataset-root data/raw/AANLIB --pair pet_mri --split test
python test.py --dataset-root data/raw/AANLIB --pair spect_mri --split test
```

Outputs:

```text
outputs/models/gan/metrics/ct_mri_metrics.csv
outputs/models/gan/metrics/pet_mri_metrics.csv
outputs/models/gan/metrics/spect_mri_metrics.csv
outputs/models/gan/metrics/all_pairs_summary.csv
outputs/models/gan/metrics/thesis_comparison_table.csv
outputs/models/gan/metrics/thesis_comparison_table.md
```

Metrics include per-image values and mean +/- std for SSIM, PSNR, MI, EN, CC, FMI, SF, and AG.

## Graph Plotting Commands

Training writes the history CSV and graph automatically. To regenerate graphs:

```powershell
python scripts/plot_training_history.py --pair ct_mri
python scripts/plot_training_history.py --pair pet_mri
python scripts/plot_training_history.py --pair spect_mri
```

Outputs:

```text
outputs/models/gan/graphs/ct_mri_training_curves.png
outputs/models/gan/graphs/pet_mri_training_curves.png
outputs/models/gan/graphs/spect_mri_training_curves.png
outputs/models/gan/graphs/ct_mri_training_history.csv
outputs/models/gan/graphs/pet_mri_training_history.csv
outputs/models/gan/graphs/spect_mri_training_history.csv
```

## Thesis Figure Commands

Generate fused test images first, then run:

```powershell
python scripts/create_thesis_figures.py --dataset-root data/raw/AANLIB --pair ct_mri --split test
python scripts/create_thesis_figures.py --dataset-root data/raw/AANLIB --pair pet_mri --split test
python scripts/create_thesis_figures.py --dataset-root data/raw/AANLIB --pair spect_mri --split test
```

Outputs:

```text
outputs/models/gan/thesis_figures/ct_mri_qualitative_comparison.png
outputs/models/gan/thesis_figures/pet_mri_qualitative_comparison.png
outputs/models/gan/thesis_figures/spect_mri_qualitative_comparison.png
```

For PET-MRI and SPECT-MRI, thesis figures use `fused_color` when it exists and fall back to `fused_original` otherwise. CT-MRI thesis figures always use `fused_original`.

## BRATS External Generalization

BRATS is never used automatically for training, validation, generation, or metrics. It is only opt-in external generalization testing:

```powershell
python generate_all_fused.py --pair ct_mri --external-test brats --checkpoint outputs/models/gan/aanlib_ct_mri/checkpoints/best_generator.pt
python test.py --pair ct_mri --external-test brats
```

BRATS outputs are labeled as external generalization testing and saved under:

```text
outputs/models/gan/images/brats/test/ct_mri/fused_original/
outputs/models/gan/images/brats/test/ct_mri/comparison_panels/
outputs/models/gan/metrics/brats_ct_mri_metrics.csv
```
