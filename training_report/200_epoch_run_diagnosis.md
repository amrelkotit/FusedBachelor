# 200-Epoch Training Run Diagnosis
**Generated:** 2026-05-23  
**Source:** `pipeline_instructions.md` → GAN logs in `outputs/models/gan/logs/`

---

## Executive Summary

The 200-epoch training run was configured in `pipeline_instructions.md` and launched for all three GAN pairs. **None of the three GAN runs reached epoch 200.** All three stopped well short — at epochs 60, 71, and 67 respectively — due to a combination of external interruption and data loader crashes. The diffusion Phase 4 (also specified as 200 epochs) was **never started**; the diffusion logs still reflect the earlier 120-epoch pipeline run.

---

## Configured Command (from `pipeline_instructions.md`)

```
# Phase 1 — GAN Training
python -u train.py --dataset-root data/raw/AANLIB --pair ct_mri    --epochs 200 --batch-size 16 --micro-batch 4 --auto-resume
python -u train.py --dataset-root data/raw/AANLIB --pair pet_mri   --epochs 200 --batch-size 16 --micro-batch 4 --auto-resume
python -u train.py --dataset-root data/raw/AANLIB --pair spect_mri --epochs 200 --batch-size 16 --micro-batch 4 --auto-resume

# Phase 4 — Diffusion Training
python -u train_diffusion.py --dataset-root data/raw/AANLIB --pair ct_mri    --epochs 200 ...
python -u train_diffusion.py --dataset-root data/raw/AANLIB --pair pet_mri   --epochs 200 ...
python -u train_diffusion.py --dataset-root data/raw/AANLIB --pair spect_mri --epochs 200 ...
```

---

## GAN Training Results

### CT-MRI GAN

| Field | Value |
|-------|-------|
| Configured epochs | 200 |
| **Epoch stopped at** | **60** |
| Best epoch | 34 |
| Best val SSIM | 0.6396 |
| Best val loss | 1.1667 |
| Final train loss (ep 60) | 1.0933 |
| Stopping reason | External interruption / process kill (no data-loader anomaly; log ends cleanly at epoch 60 with no error markers) |
| Resume evidence | Epochs 52 and 53 each appear **twice** in the log (duplicate entries with slightly different `skipped_updates` values), confirming the run was interrupted and auto-resumed at least once before the final stop |

**Last 3 log entries:**
```
epoch=058 train_loss=1.082831 ... ssim=0.331885 psnr=16.71  is_best=False
epoch=059 train_loss=1.085326 ... ssim=0.361957 psnr=16.98  is_best=False
epoch=060 train_loss=1.093341 ... ssim=0.341380 psnr=16.88  is_best=False
```

No further entries. The process ended between epoch 60 and 61 without writing a crash line.

---

### PET-MRI GAN

| Field | Value |
|-------|-------|
| Configured epochs | 200 |
| **Epoch stopped at** | **71** |
| Best epoch | 48 |
| Best val SSIM | 0.6769 |
| Best val loss | 2.2871 |
| Final train loss (ep 71) | 2.4327 (elevated — mid-crash) |
| Stopping reason | **Data loader crash during validation** — `skipped_batches=52`, all validation metrics zeroed out |

**Critical last log entry (epoch 71):**
```
epoch=071 train_loss=2.432730 ... val_msfd_loss=0.0
         skipped_updates=1 skipped_batches=52 skipped_d_steps=2 skipped_g_steps=0
         val_loss=0.0  ssim=0.0  psnr=0.0  mi=0.0  ag=0.0  sf=0.0
         is_best=False
```

52 out of ~71 expected batches were skipped. The trainer logged the epoch but returned all-zero metrics, then the process died before epoch 72 could be written. This is a **DataLoader worker crash** — typically caused by a file I/O error (missing file, corrupted tensor, or OOM in a worker process) that prevents the iterator from yielding batches.

---

### SPECT-MRI GAN

| Field | Value |
|-------|-------|
| Configured epochs | 200 |
| **Epoch stopped at** | **67** |
| Best epoch | 44 |
| Best val SSIM | 0.7438 |
| Best val loss | 1.7279 |
| Final train loss (ep 67) | 0.000000 |
| Stopping reason | **Complete data loader failure** — `skipped_batches=71` (ALL 71 batches skipped), `train_loss=0.0`, all metrics zeroed |

**Critical last log entry (epoch 67):**
```
epoch=067 train_loss=0.000000 ... train_msfd_loss=0.0  val_msfd_loss=0.0
          skipped_updates=0  skipped_batches=71  skipped_d_steps=0  skipped_g_steps=0
          val_loss=0.0  ssim=0.0  psnr=0.0  mi=0.0  ag=0.0  sf=0.0
          is_best=False
```

Every single training batch was skipped — the data loader produced **zero samples** this epoch. This is more severe than PET-MRI: the iterator failed to yield even one batch, meaning the data pipeline completely collapsed (worker threads crashed before any data could flow). The trainer survived and logged the zeroed epoch, then died before starting epoch 68.

---

## Diffusion Phase — Not Executed

The diffusion logs (`outputs/models/diffusion/logs/`) all show **120-epoch** runs timestamped from the earlier pipeline run. There is no evidence that the 200-epoch diffusion phase from `pipeline_instructions.md` was ever started. The pipeline likely halted after the GAN failures (or the diffusion runs shown were kept from a prior 120-epoch execution and the new run never progressed to Phase 4).

---

## Root Cause Analysis

### Why the data loader crashed (PET-MRI ep 71, SPECT-MRI ep 67)

The `skipped_batches` counter in `train.py`/`trainer.py` increments when a batch raises an exception that the training loop catches and skips. When this reaches 100% of batches, the epoch produces zero gradient signal and all metrics are 0.0. The most common causes in this codebase are:

1. **Worker OOM** — PyTorch DataLoader workers share system RAM. After 60–70 epochs of GAN training the GPU memory may be fragmented enough that worker processes can no longer load and preprocess the full batch simultaneously.
2. **File handle exhaustion** — with 3 GAN runs open in sequence, each with `num_workers > 0`, the cumulative file descriptor count can hit OS limits on Windows.
3. **Corrupted/missing cache file** — if the AANLIB dataset uses any form of preprocessed cache, a corrupt cache entry will cause repeated exceptions until the loader gives up entirely.

### Why CT-MRI stopped at epoch 60 (no zero-batch signature)

The CT-MRI log ends cleanly after epoch 60 with no anomalous entries — the trainer simply stopped. Possible causes:
- The machine was shut down or the terminal was closed (most likely, given the pipeline was running autonomously)
- A Python exception that was **not** caught by the batch-skip handler (e.g. CUDA assertion, OOM at the optimizer step rather than data loading)
- The GAN SSIM was still only 0.34 at epoch 60 (well below the 0.75 threshold in `pipeline_instructions.md`), so the pipeline operator may have manually stopped it

The duplicate epoch=052 and epoch=053 entries (with different `skipped_updates` counts) confirm the run was **auto-resumed at least once** using `--auto-resume` before the final stop at epoch 60.

---

## SSIM Progression — GAN Runs

### CT-MRI GAN (stopped ep 60, target: 200)

| Epoch | val SSIM | is_best |
|-------|----------|---------|
| 1 | 0.1358 | True |
| 7 | 0.3564 | True |
| 10 | 0.6226 | True |
| 34 | 0.6396 | **True** ← best |
| 45 | 0.4689 | False |
| 57 | 0.5062 | False |
| 60 | 0.3414 | False |

SSIM peaked at epoch 34, then became erratic (oscillating 0.28–0.51). Training was still far from the 0.75 thesis threshold.

### PET-MRI GAN (stopped ep 71, target: 200)

| Epoch | val SSIM | is_best |
|-------|----------|---------|
| 1 | 0.1944 | True |
| 37 | 0.6287 | True |
| 42 | 0.6662 | True |
| 48 | 0.6769 | **True** ← best |
| 65 | plateau ~0.63–0.67 | False |
| 71 | 0.0 (crash) | False |

Approaching convergence near epoch 48, still ~10 points below the 0.75 threshold.

### SPECT-MRI GAN (stopped ep 67, target: 200)

| Epoch | val SSIM | is_best |
|-------|----------|---------|
| 1 | 0.1854 | True |
| 17 | 0.7287 | True |
| 26 | 0.7372 | True |
| 44 | 0.7438 | **True** ← best |
| 60–66 | 0.33–0.47 (degraded) | False |
| 67 | 0.0 (crash) | False |

SPECT-MRI was the only pair to exceed 0.74 SSIM — it was the closest to the 0.75 thesis threshold. Notably, SSIM **degraded significantly** after epoch 44, suggesting the GAN was starting to overfit or oscillate.

---

## Error Messages

No Python tracebacks or exception lines appear in any of the three GAN log files. The crash in each case was recorded only as a zeroed-out epoch entry (PET-MRI, SPECT-MRI) or a clean stop (CT-MRI) — the actual exception was printed to stderr/stdout which was not captured in the structured log file.

---

## Current State of Checkpoints

| Model | Best checkpoint | Best SSIM | Thesis threshold | Status |
|-------|----------------|-----------|-----------------|--------|
| GAN CT-MRI | `gan_epoch_034.pt` | 0.6396 | > 0.75 | ❌ Below threshold |
| GAN PET-MRI | `gan_epoch_048.pt` | 0.6769 | > 0.75 | ❌ Below threshold |
| GAN SPECT-MRI | `gan_epoch_044.pt` | 0.7438 | > 0.75 | ⚠️ Very close |
| Diffusion CT-MRI | `best_diffusion.pt` (ep 90) | 0.7262 | > 0.75 | ❌ Below threshold |
| Diffusion PET-MRI | `best_diffusion.pt` (ep 90) | 0.6625 | > 0.75 | ❌ Below threshold |
| Diffusion SPECT-MRI | `best_diffusion.pt` (ep 85) | 0.7181 | > 0.75 | ❌ Below threshold |

---

## Recommendations

1. **Fix the data loader crash before any retraining.** The batch-skip failures in PET-MRI and SPECT-MRI will recur. Add `persistent_workers=True` and `pin_memory=False` to the DataLoader calls in `trainer.py`, reduce `num_workers` to 2 or 0, and add explicit `try/except` logging in the worker collate function to surface the actual error.

2. **Resume GAN training from best checkpoints** using `--auto-resume`. With the data loader fixed, the SPECT-MRI GAN (best SSIM 0.7438) needs only modest further training to clear 0.75. CT-MRI and PET-MRI need more epochs.

3. **Lower the learning rate on resume.** At epoch 60–67 the GAN LR had already decayed to ~6.25e-6 (CT/SPECT) and 2.5e-5 (PET). SSIM oscillation post-best-epoch indicates the LR is still slightly too large for fine convergence — try resuming at 1e-6.

4. **Run the diffusion 200-epoch phase once GAN training succeeds.** The diffusion models were never trained with `--epochs 200`; the existing logs are from the prior 120-epoch run.

5. **Check system resources before re-running.** The data loader crash pattern (starting at epoch 67–71 after 60+ epochs of prior GAN training) is consistent with cumulative memory or file-descriptor pressure. Reboot before the next pipeline run.
