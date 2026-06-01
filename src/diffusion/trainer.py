from pathlib import Path
import copy
import csv
import gc
import json
import math
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.paired_dataset import (
    AANLIB_ROOT,
    build_aanlib_train_val_datasets,
    diffusion_checkpoint_dir,
    diffusion_graph_dir,
    diffusion_logs_dir,
    diffusion_metrics_dir,
    diffusion_pair_logs_dir,
    diffusion_sample_dir,
    normalize_pair,
    pair_labels,
)
from src.diffusion.losses import DiffusionFusionLoss
from src.diffusion.model import ConditionalUNet, LegacyConditionalUNet
from src.diffusion.sampler import sample_refined
from src.diffusion.scheduler import DiffusionScheduler
from src.diffusion.utils import fusion_reference, initial_fusion, save_comparison_panel
from src.evaluation.metrics import evaluate_fusion


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_total_loss",
    "val_total_loss",
    "train_l1_loss",
    "val_l1_loss",
    "train_ssim_loss",
    "val_ssim_loss",
    "train_gradient_loss",
    "val_gradient_loss",
    "train_noise_loss",
    "val_noise_loss",
    "train_msfd_loss",
    "val_msfd_loss",
    "train_hf_loss",
    "val_hf_loss",
    "train_edge_loss",
    "val_edge_loss",
    "train_laplacian_loss",
    "val_laplacian_loss",
    "train_ms_ssim_loss",
    "val_ms_ssim_loss",
    "train_local_contrast_loss",
    "val_local_contrast_loss",
    "train_perceptual_loss",
    "val_perceptual_loss",
    "train_ssim",
    "val_ssim",
    "train_psnr",
    "val_psnr",
    "train_mi",
    "val_mi",
    "train_en",
    "val_en",
    "train_cc",
    "val_cc",
    "train_fmi",
    "val_fmi",
    "train_sf",
    "val_sf",
    "train_ag",
    "val_ag",
    "best_epoch",
    "best_val_ssim",
    "is_best",
]


def monitor_write(message):
    tqdm.write(str(message), file=sys.stdout)


def resolve_training_device(device, allow_cpu=False):
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if allow_cpu:
            return "cpu"
        raise RuntimeError("CUDA is not available. Pass --allow-cpu to train on CPU.")
    if requested in {"cuda", "gpu"} or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA.")
        return "cuda:0" if requested in {"cuda", "gpu"} else requested
    if requested == "cpu":
        if not allow_cpu:
            raise RuntimeError("Refusing CPU training without --allow-cpu.")
        return "cpu"
    raise ValueError(f"Unsupported device: {device}")


def _avg(metrics, left, right):
    return 0.5 * (metrics.get(left, 0.0) + metrics.get(right, 0.0))


class DiffusionFusionTrainer:
    def __init__(
        self,
        dataset_root=None,
        pair="ct_mri",
        output_root=None,
        image_size=256,
        batch_size=8,
        micro_batch=2,
        epochs=200,
        lr=5e-5,
        timesteps=1000,
        sampling_steps=150,
        lambda_l1=2.0,
        lambda_ssim=0.0,
        lambda_grad=2.0,
        lambda_hf=1.0,
        lambda_noise=0.5,
        lambda_ms_ssim=0.0,
        lambda_local_contrast=0.3,
        lambda_msfd=1.0,
        lambda_perceptual=0.0,
        base_channels=64,
        time_dim=256,
        use_msfd_guidance=False,
        use_ema=True,
        ema_decay=0.999,
        early_stopping=True,
        patience=40,
        min_delta=0.0005,
        val_split=0.15,
        val_every=2,
        num_workers=0,
        max_items=None,
        device="auto",
        allow_cpu=False,
        resume=None,
        auto_resume=False,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root else AANLIB_ROOT
        self.pair = normalize_pair(pair)
        self.output_root = Path(output_root) if output_root else Path("outputs") / "models" / "diffusion"
        self.output_root = self.output_root if self.output_root.is_absolute() else Path.cwd() / self.output_root
        self.source1_label, self.source2_label, _ = pair_labels(self.pair)
        self.image_size = image_size
        self.epochs = int(epochs)
        self.timesteps = int(timesteps)
        self.sampling_steps = int(sampling_steps)
        self.use_msfd_guidance = bool(use_msfd_guidance)
        self.lambda_msfd = float(lambda_msfd)
        self.lambda_perceptual = float(lambda_perceptual)
        self.lambda_noise = float(lambda_noise)
        self.base_channels = int(base_channels)
        self.time_dim = int(time_dim)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.no_improve_epochs = 0
        self.device = resolve_training_device(device, allow_cpu=allow_cpu)
        self.amp_enabled = False  # float16 overflows to NaN/Inf on every batch; float32 required
        if self.device.startswith("cuda"):
            # Reduce allocator fragmentation caused by many small alloc/free cycles
            # (e.g. after NaN-skipped batches).  expandable_segments lets PyTorch
            # grow/shrink segments on demand instead of pre-committing large blocks.
            # Must be set before the first CUDA tensor is allocated.
            import os as _os
            _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            # Cap PyTorch VRAM usage to 90% of total GPU memory.  This reserves
            # a buffer so PyTorch raises an OOM exception before the GPU driver
            # exhausts all memory and triggers a Windows BSOD (SYSTEM_SERVICE_EXCEPTION
            # 0x3B / nvlddmkm.sys).  The diffusion model is ~4x larger than before
            # (base_channels 32→64) so this guard is especially important here.
            try:
                torch.cuda.set_per_process_memory_fraction(0.90)
                monitor_write("[VRAM Guard] PyTorch VRAM capped at 90% of total GPU memory to prevent driver BSOD.")
            except Exception as _mem_exc:
                monitor_write(f"[VRAM Guard] Could not set memory fraction: {_mem_exc}")
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled)
        self.micro_batch_size = int(micro_batch)
        self.accumulation_steps = max(1, math.ceil(int(batch_size) / self.micro_batch_size))
        self.num_workers = int(num_workers)
        self.val_every = int(val_every)
        self.start_epoch = 1
        self.best_epoch = None
        self.best_val_ssim = float("-inf")
        self.history = []

        self.pair_dir = diffusion_checkpoint_dir(self.pair, output_root=self.output_root).parent
        self.checkpoint_dir = diffusion_checkpoint_dir(self.pair, output_root=self.output_root)
        self.sample_dir = diffusion_sample_dir(self.pair, output_root=self.output_root)
        self.graph_dir = diffusion_graph_dir(self.output_root)
        self.metrics_dir = diffusion_metrics_dir(self.output_root)
        self.global_log_dir = diffusion_logs_dir(self.output_root)
        self.pair_log_dir = diffusion_pair_logs_dir(self.pair, output_root=self.output_root)
        self.history_csv_path = self.graph_dir / f"{self.pair}_diffusion_training_history.csv"
        self.global_history_csv_path = self.global_log_dir / f"{self.pair}_training_history.csv"
        self.history_json_path = self.pair_log_dir / f"{self.pair}_diffusion_training_history.json"
        self.training_log_path = self.pair_log_dir / f"{self.pair}_diffusion_training.log"
        self.global_training_log_path = self.global_log_dir / f"{self.pair}_diffusion_training.log"
        self._ensure_dirs()

        train_dataset, val_dataset = build_aanlib_train_val_datasets(
            self.dataset_root,
            self.pair,
            image_size=image_size,
            max_items=max_items,
            val_split=val_split,
            seed=42,
            augment_train=True,  # flip + rotation + brightness jitter for diffusion training
        )
        # pin_memory=False avoids Windows file-descriptor exhaustion over long runs.
        loader_kwargs = {"num_workers": self.num_workers, "pin_memory": False}
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        self.train_loader = DataLoader(train_dataset, batch_size=self.micro_batch_size, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs) if val_dataset is not None else None

        self.model = ConditionalUNet(in_channels=4, base_channels=self.base_channels, time_dim=self.time_dim).to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.scheduler = DiffusionScheduler(timesteps=self.timesteps, device=self.device)
        self.loss_fn = DiffusionFusionLoss(
            lambda_l1=lambda_l1,
            lambda_ssim=lambda_ssim,
            lambda_grad=lambda_grad,
            lambda_hf=lambda_hf,
            lambda_msfd=lambda_msfd,
            lambda_ms_ssim=lambda_ms_ssim,
            lambda_local_contrast=lambda_local_contrast,
            lambda_perceptual=lambda_perceptual,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # Cosine annealing with warm restarts — better than ReduceLROnPlateau for diffusion
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        monitor_write("[Model] Selected model: Diffusion")
        monitor_write(f"[Output] Root: {self.output_root}")
        monitor_write(f"[Pair] {self.pair}")
        monitor_write(f"[Device] {self.device}")
        monitor_write(f"Requested batch: {batch_size}")
        monitor_write(f"Micro batch: {self.micro_batch_size}")
        monitor_write(f"Accumulation steps: {self.accumulation_steps}")
        monitor_write(f"AMP mixed precision: {'enabled' if self.amp_enabled else 'disabled'}")
        monitor_write(f"EMA: {'enabled' if self.use_ema else 'disabled'}, decay={self.ema_decay}")
        monitor_write(
            "Loss weights: "
            f"lambda_noise={lambda_noise}, lambda_l1={lambda_l1}, lambda_ssim={lambda_ssim}, "
            f"lambda_grad={lambda_grad}, lambda_hf={lambda_hf}, lambda_ms_ssim={lambda_ms_ssim}, "
            f"lambda_local_contrast={lambda_local_contrast}, lambda_msfd={lambda_msfd}"
        )
        if self.early_stopping:
            monitor_write(f"Early stopping: enabled, patience={self.patience}, min_delta={self.min_delta}")

        self.load_history()
        self.restore_best_from_history()
        if resume and auto_resume:
            raise ValueError("Use either --resume or --auto-resume, not both.")
        if resume:
            self.load_checkpoint(resume)
        elif auto_resume:
            latest = self.find_latest_checkpoint()
            if latest:
                self.load_checkpoint(latest)
            else:
                monitor_write(
                    f"[Auto-resume] No checkpoint found in {self.checkpoint_dir} — starting from epoch 1."
                )

    def _ensure_dirs(self):
        for folder in [
            self.checkpoint_dir,
            self.sample_dir,
            self.graph_dir,
            self.metrics_dir,
            self.global_log_dir,
            self.pair_log_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self):
        path = self.checkpoint_dir / "last_diffusion.pt"
        if path.exists():
            return path
        # Fall back to highest-numbered epoch file if last_diffusion.pt is missing
        epoch_files = sorted(self.checkpoint_dir.glob("diffusion_epoch_*.pt"))
        if epoch_files:
            monitor_write(
                f"[Auto-resume] last_diffusion.pt not found; falling back to {epoch_files[-1].name}"
            )
            return epoch_files[-1]
        return None

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # ── NaN-weight guard ───────────────────────────────────────────────────
        # If a previous run crashed with NaN losses and saved a corrupted
        # checkpoint, resuming from it causes every subsequent batch to produce
        # NaN → OOM cascade.  Detect this BEFORE loading into self.model so we
        # keep the freshly-initialised clean weights.
        nan_in_ckpt = any(
            not v.isfinite().all()
            for k, v in checkpoint.get("model_state_dict", {}).items()
            if isinstance(v, torch.Tensor) and v.is_floating_point()
        )
        if nan_in_ckpt:
            monitor_write(
                f"[WARNING] Checkpoint '{checkpoint_path}' contains NaN/Inf weights "
                "(saved during a crashed run). Discarding corrupted weights — model "
                "starts from random init.\n"
                "To resume from a known-good snapshot, use:\n"
                "  --resume <path>/best_diffusion.pt"
            )
            return
        # ──────────────────────────────────────────────────────────────────────

        _skip_optimizer_load = False
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as arch_err:
            state_dict = checkpoint.get("model_state_dict", {})
            # Legacy SFF architecture: checkpoint has source_lift/source_sff keys and in_channels=3.
            # Rebuild model as LegacyConditionalUNet and retry so training can resume seamlessly.
            if "source_lift.weight" in state_dict:
                in_ch = state_dict["in_conv.weight"].shape[1]
                monitor_write(
                    f"[Checkpoint] Detected legacy SFF architecture (in_channels={in_ch}) — "
                    "rebuilding model as LegacyConditionalUNet to resume training."
                )
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.model = LegacyConditionalUNet(
                    in_channels=in_ch, base_channels=self.base_channels, time_dim=self.time_dim
                ).to(self.device)
                self.ema_model = copy.deepcopy(self.model).to(self.device).eval()
                for p in self.ema_model.parameters():
                    p.requires_grad_(False)
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=current_lr, weight_decay=1e-4
                )
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=10, T_mult=2
                )
                # Optimizer was rebuilt for the new architecture; checkpoint optimizer state
                # belongs to the old architecture and its moment buffer shapes won't match.
                _skip_optimizer_load = True
                try:
                    self.model.load_state_dict(state_dict)
                except RuntimeError as retry_err:
                    monitor_write(
                        f"[WARNING] Legacy architecture rebuild failed: {retry_err}\n"
                        "Starting from scratch."
                    )
                    self.best_val_ssim = float("-inf")
                    self.best_epoch = None
                    self.history = []
                    return
            else:
                # Architecture mismatch (e.g. old in_channels=3 vs new in_channels=4).
                # Start fresh AND reset best_val_ssim — keeping the old best would cause
                # early stopping to fire immediately since the new model can't match a
                # score set by a different architecture.
                monitor_write(
                    f"[WARNING] Checkpoint architecture mismatch — cannot resume weights: {arch_err}\n"
                    "Starting from scratch with the new architecture. "
                    "Resetting best_val_ssim and history so early stopping is not blocked."
                )
                self.best_val_ssim = float("-inf")
                self.best_epoch = None
                self.history = []
                return
        try:
            if checkpoint.get("ema_model_state_dict") is not None:
                self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            else:
                self.ema_model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            # EMA mismatch — fall back to copying the main model weights
            self.ema_model.load_state_dict(self.model.state_dict())
        if "optimizer_state_dict" in checkpoint and not _skip_optimizer_load:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # If Adam moment buffers contain NaN (from a previous crashed update),
                # every subsequent step will produce NaN weights even from clean inputs.
                opt_has_nan = any(
                    not v.isfinite().all()
                    for state in self.optimizer.state.values()
                    for v in state.values()
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                )
                if opt_has_nan:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    monitor_write(
                        "[Optimizer] Loaded optimizer state has NaN/Inf moment buffers — "
                        "resetting to fresh optimizer."
                    )
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(), lr=current_lr, weight_decay=1e-4
                    )
            except Exception:
                pass  # Optimizer mismatch is non-fatal; just continue with fresh optimizer state
        if "lr_scheduler_state_dict" in checkpoint:
            try:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            except Exception:
                pass  # Scheduler state mismatch is non-fatal
        if "scaler" in checkpoint:
            scaler_state = checkpoint["scaler"]
            # Validate the persisted scale value.  A scale ≤ 1.0 (or NaN/Inf)
            # means the previous run kept halving it on every NaN batch until it
            # hit the floor.  Loading that degraded state causes every AMP forward
            # pass in the new run to also produce NaN (float16 underflow), which
            # triggers the NaN→OOM cascade we saw at epoch 98.
            # Solution: if the scale looks corrupted, throw it away and start fresh
            # so the scaler can re-calibrate on the first few good batches.
            raw_scale = scaler_state.get("scale", None)
            # GradScaler.state_dict() stores scale as a torch.Tensor (cpu float32).
            # Convert to Python float for the range check regardless of type.
            try:
                scale_val = float(raw_scale) if raw_scale is not None else 0.0
            except Exception:
                scale_val = 0.0
            scale_ok = scale_val > 1.0 and not (scale_val != scale_val)  # finite & > 1
            if not scale_ok:
                monitor_write(
                    f"[Scaler] Loaded GradScaler scale={raw_scale!r} is degraded (≤ 1.0 or NaN). "
                    "Resetting to fresh GradScaler (init_scale=4096) to prevent NaN cascade."
                )
                self.scaler = GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0)
            else:
                try:
                    self.scaler.load_state_dict(scaler_state)
                except Exception as scaler_err:
                    monitor_write(f"[Scaler] Failed to load scaler state ({scaler_err}); using fresh scaler.")
                    self.scaler = GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0)
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_val_ssim = float(checkpoint.get("best_val_ssim", self.best_val_ssim))
        # Also restore best_epoch from the checkpoint so it matches the checkpoint's
        # ground-truth value, rather than being inherited from a stale history JSON
        # written by a different (older) training run.
        if checkpoint.get("best_epoch") is not None:
            self.best_epoch = int(checkpoint["best_epoch"])
        monitor_write(f"Loaded diffusion checkpoint: {checkpoint_path} (epoch {checkpoint['epoch']})")

    def update_ema(self):
        if not self.use_ema:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                if ema_param.shape != param.shape:
                    monitor_write(
                        f"[EMA] Shape mismatch — ema {tuple(ema_param.shape)} vs model {tuple(param.shape)} — skipping parameter"
                    )
                    continue
                ema_param.mul_(self.ema_decay).add_(param.detach(), alpha=1.0 - self.ema_decay)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                if ema_buffer.shape != buffer.shape:
                    continue
                ema_buffer.copy_(buffer)

    def checkpoint_payload(self, epoch, model_state_dict=None, is_ema=False):
        return {
            "model_state_dict": model_state_dict if model_state_dict is not None else self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict() if self.use_ema else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_ssim": self.best_val_ssim,
            "best_epoch": self.best_epoch,
            "pair": self.pair,
            "is_ema": bool(is_ema),
            "config": {
                "dataset_root": str(self.dataset_root),
                "output_root": str(self.output_root),
                "image_size": self.image_size,
                "in_channels": 4,
                "base_channels": self.base_channels,
                "time_dim": self.time_dim,
                "timesteps": self.timesteps,
                "sampling_steps": self.sampling_steps,
                "use_msfd_guidance": self.use_msfd_guidance,
                "lambda_msfd": self.lambda_msfd,
                "lambda_noise": self.lambda_noise,
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
                "prediction_type": "x0",
            },
            "scaler": self.scaler.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }

    def save_checkpoint(self, epoch, is_best=False):
        # Never persist NaN weights — a corrupted checkpoint causes every subsequent
        # auto-resume run to immediately enter permanent NaN mode.
        with torch.no_grad():
            nan_in_model = any(not p.data.isfinite().all() for p in self.model.parameters())
        if nan_in_model:
            monitor_write(
                f"[Checkpoint] Epoch {epoch}: model weights are NaN/Inf — "
                "skipping checkpoint save to avoid persisting a corrupted state."
            )
            return self.checkpoint_dir / "last_diffusion.pt"  # return path but don't write
        payload = self.checkpoint_payload(epoch)
        last_path = self.checkpoint_dir / "last_diffusion.pt"
        torch.save(payload, last_path)
        if self.use_ema:
            torch.save(self.checkpoint_payload(epoch, model_state_dict=self.ema_model.state_dict(), is_ema=True), self.checkpoint_dir / "last_diffusion_ema.pt")
        epoch_path = self.checkpoint_dir / f"diffusion_epoch_{epoch:03d}.pt"
        torch.save(payload, epoch_path)
        if is_best:
            torch.save(payload, self.checkpoint_dir / "best_diffusion.pt")
            if self.use_ema:
                torch.save(self.checkpoint_payload(epoch, model_state_dict=self.ema_model.state_dict(), is_ema=True), self.checkpoint_dir / "best_diffusion_ema.pt")
        return last_path

    def load_history(self):
        if self.history_json_path.exists():
            try:
                self.history = json.loads(self.history_json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.history = []

    def restore_best_from_history(self):
        for row in self.history:
            value = row.get("val_ssim")
            if value not in (None, "") and float(value) > self.best_val_ssim:
                self.best_val_ssim = float(value)
                self.best_epoch = int(row["epoch"])

    def target_for_batch(self, source1, source2):
        msfd = fusion_reference(source1, source2, use_msfd=True)
        target = msfd if self.use_msfd_guidance else fusion_reference(source1, source2, use_msfd=False)
        return target.detach(), msfd.detach()

    def forward_loss(self, source1, source2):
        target, msfd = self.target_for_batch(source1, source2)
        initial = initial_fusion(source1, source2)
        noise = torch.randn_like(target)
        t = torch.randint(0, self.timesteps, (target.shape[0],), device=self.device)
        noisy = self.scheduler.add_noise(target, noise, t)

        # x0-prediction: the model directly predicts the clean image.
        # This eliminates the 1/sqrt(alpha_bar) amplification that occurs with
        # epsilon-prediction at high timesteps (factor up to ~150× at t=999),
        # which was the root cause of NaN cascades and training instability.
        x0_pred = self.model(noisy, source1, source2, initial, t)

        # Noise-consistency loss: penalises how much the implied noise
        # (derived from x0_pred) deviates from the actual noise.
        # Equivalent to weighted MSE on x0_pred with weight alpha_bar/(1-alpha_bar)
        # — naturally down-weights high-t steps and focuses on low-t refinement.
        alpha_bar_t = self.scheduler.alpha_bars[t].view(-1, 1, 1, 1).to(x0_pred.device, x0_pred.dtype)
        noise_implied = (noisy - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)
        noise_loss = F.mse_loss(noise_implied.float(), noise.float())

        fusion_losses = self.loss_fn(x0_pred, target, source1, source2, msfd_target=msfd, use_msfd=self.use_msfd_guidance)
        fusion_losses["total"] = fusion_losses["total"] + self.lambda_noise * noise_loss
        fusion_losses["noise"] = noise_loss.detach()
        return fusion_losses, x0_pred

    def metrics_from_fused(self, fused, source1, source2):
        metrics = evaluate_fusion(fused, source2, source1)
        # Use aggregated keys from evaluate_fusion (PSNR data_range=1.0, MI=sum 256-bin)
        return {
            "ssim": metrics.get("SSIM", _avg(metrics, "SSIM_MRI", "SSIM_CT")),
            "psnr": metrics.get("PSNR", _avg(metrics, "PSNR_MRI", "PSNR_CT")),
            "mi":   metrics.get("MI", 0.0),   # sum MI_MRI + MI_CT
            "en":   metrics.get("EN", 0.0),
            "cc":   metrics.get("CC", 0.0),
            "fmi":  metrics.get("FMI", 0.0),
            "sf":   metrics.get("SF", 0.0),
            "ag":   metrics.get("AG", 0.0),
        }

    def empty_totals(self):
        return {
            key: 0.0
            for key in [
                "total_loss",
                "l1_loss",
                "ssim_loss",
                "gradient_loss",
                "noise_loss",
                "hf_loss",
                "edge_loss",
                "laplacian_loss",
                "ms_ssim_loss",
                "local_contrast_loss",
                "perceptual_loss",
                "msfd_loss",
                "ssim",
                "psnr",
                "mi",
                "en",
                "cc",
                "fmi",
                "sf",
                "ag",
            ]
        }

    def update_totals(self, totals, losses, metrics):
        totals["total_loss"] += float(losses["total"].detach().cpu())
        totals["l1_loss"] += float(losses["l1"].cpu())
        totals["ssim_loss"] += float(losses["ssim"].cpu())
        totals["gradient_loss"] += float(losses["gradient"].cpu())
        totals["noise_loss"] += float(losses["noise"].cpu())
        totals["hf_loss"] += float(losses["hf"].cpu())
        totals["edge_loss"] += float(losses["edge"].cpu())
        totals["laplacian_loss"] += float(losses["laplacian"].cpu())
        totals["ms_ssim_loss"] += float(losses["ms_ssim"].cpu())
        totals["local_contrast_loss"] += float(losses["local_contrast"].cpu())
        totals["perceptual_loss"] += float(losses.get("perceptual", losses["total"].new_tensor(0.0)).cpu())
        totals["msfd_loss"] += float(losses["msfd"].cpu())
        for key, value in metrics.items():
            totals[key] += float(value)

    def average_totals(self, totals, count):
        count = max(1, count)
        return {key: value / count for key, value in totals.items()}

    def _reload_best_checkpoint(self, epoch):
        """Restore model + EMA + optimizer from the best saved checkpoint.

        Returns True if the reload produced finite weights, False otherwise.
        """
        best_ckpt = self.checkpoint_dir / "best_diffusion.pt"
        if not best_ckpt.exists():
            monitor_write(
                f"[NaN Recovery] Epoch {epoch}: best_diffusion.pt not found — "
                "cannot restore. Re-initialising model from scratch."
            )
            from src.diffusion.model import ConditionalUNet  # local import to avoid circular dep
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.model = ConditionalUNet(in_channels=4, base_channels=self.base_channels, time_dim=self.time_dim).to(self.device)
            self.ema_model = copy.deepcopy(self.model).to(self.device).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=current_lr, weight_decay=1e-4)
            self.scaler = GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0)
            return True  # fresh random weights are finite
        saved_start = self.start_epoch
        self.load_checkpoint(best_ckpt)
        self.start_epoch = saved_start  # keep the current epoch pointer
        # Reset optimizer with a modest LR to avoid another explosion
        current_lr = min(self.optimizer.param_groups[0]["lr"], 1e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=current_lr, weight_decay=1e-4)
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0)
        with torch.no_grad():
            still_nan = any(not p.data.isfinite().all() for p in self.model.parameters())
        if still_nan:
            monitor_write(
                f"[NaN Recovery] best_diffusion.pt also has NaN weights — "
                "re-initialising model from scratch."
            )
            from src.diffusion.model import ConditionalUNet
            self.model = ConditionalUNet(in_channels=4, base_channels=self.base_channels, time_dim=self.time_dim).to(self.device)
            self.ema_model = copy.deepcopy(self.model).to(self.device).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=current_lr, weight_decay=1e-4)
            self.scaler = GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0)
        monitor_write(
            f"[NaN Recovery] Epoch {epoch}: restored from best checkpoint "
            f"(epoch {self.best_epoch}, val_ssim={self.best_val_ssim:.6f}), lr reset to {current_lr:.2e}"
        )
        return True

    def train_epoch(self, epoch):
        # Fast-path: if weights are already NaN at epoch start, skip the entire
        # data loop (68 batches × 40 s = 45 min of useless work) and return a
        # sentinel so fit() can trigger checkpoint recovery before the next epoch.
        with torch.no_grad():
            if any(not p.data.isfinite().all() for p in self.model.parameters()):
                monitor_write(
                    f"[NaN weights] Epoch {epoch}: model weights are NaN/Inf before training starts "
                    "— skipping epoch. fit() will reload best checkpoint."
                )
                result = self.empty_totals()
                result["skipped_batches"] = -1  # sentinel: entire epoch skipped
                return result

        self.model.train()
        totals = self.empty_totals()
        self.optimizer.zero_grad(set_to_none=True)
        processed = 0
        skipped = 0
        consecutive_nan = 0  # tracks back-to-back NaN batches for scaler auto-reset
        bar = tqdm(self.train_loader, desc=f"Diffusion epoch {epoch}/{self.epochs}", dynamic_ncols=True, file=sys.stdout)
        loader_iter = iter(bar)
        batch_index = 0
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            except Exception as dl_exc:
                skipped += 1
                monitor_write(
                    f"[DataLoader ERROR] Diffusion epoch {epoch}, batch ~{batch_index + 1}: "
                    f"{type(dl_exc).__name__}: {dl_exc} — skipping batch"
                )
                continue
            batch_index += 1
            try:
                source1 = batch["source1"].to(self.device)
                source2 = batch["source2"].to(self.device)
                with autocast("cuda", enabled=self.amp_enabled):
                    losses, x0_pred = self.forward_loss(source1, source2)
                    loss = losses["total"] / self.accumulation_steps

                # ── NaN/Inf guard ──────────────────────────────────────────────
                # If the loss is already non-finite after the forward pass, skip
                # this batch entirely.  Do NOT call backward() — calling
                # backward() on NaN/Inf corrupts accumulated gradients for ALL
                # parameters, which then makes the scaler permanently skip every
                # subsequent update and freezes training.
                if not torch.isfinite(loss):
                    skipped += 1
                    consecutive_nan += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    # Explicitly free the forward-pass tensors so they don't
                    # accumulate on the GPU between NaN batches and cause OOM.
                    del losses, x0_pred, loss
                    gc.collect()
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    monitor_write(
                        f"[NaN/Inf loss] Epoch {epoch}, batch {batch_index}: "
                        "loss=nan — skipping batch (AMP overflow?)"
                    )
                    # After 4 consecutive NaN batches the GradScaler scale has
                    # likely collapsed.  Reset it so the next batches can recover
                    # rather than spiralling into permanent NaN → OOM.
                    if consecutive_nan >= 4:
                        n = consecutive_nan
                        self.scaler = GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0)
                        consecutive_nan = 0
                        monitor_write(
                            f"[Scaler] {n} consecutive NaN batches — "
                            "GradScaler reset to fresh state (init_scale=4096)."
                        )
                    continue
                consecutive_nan = 0  # reset streak on any finite loss
                # ──────────────────────────────────────────────────────────────

                self.scaler.scale(loss).backward()
                if batch_index % self.accumulation_steps == 0 or batch_index == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # ── NaN weight detector ────────────────────────────────────
                    # After a successful optimizer step, verify that no model
                    # weights became NaN (can happen when the scaler mis-judges
                    # an overflow and lets a corrupted update through).  If NaN
                    # is detected, restore parameters from the EMA snapshot —
                    # which lags behind by one step and should still be clean —
                    # and reset optimizer momentum so Adam's m/v buffers do not
                    # carry the NaN forward.
                    with torch.no_grad():
                        nan_in_weights = any(
                            not p.data.isfinite().all()
                            for p in self.model.parameters()
                        )
                    if nan_in_weights:
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        # Try EMA first (fast path, stays in-memory)
                        with torch.no_grad():
                            ema_nan = any(not p.data.isfinite().all() for p in self.ema_model.parameters())
                        if not ema_nan:
                            self.model.load_state_dict(self.ema_model.state_dict())
                            self.optimizer = torch.optim.AdamW(
                                self.model.parameters(), lr=current_lr, weight_decay=1e-4
                            )
                            self.scaler = GradScaler("cuda", enabled=self.amp_enabled, init_scale=4096.0)
                            monitor_write(
                                f"[NaN weights] Epoch {epoch}, batch {batch_index}: "
                                "model weights corrupted — restored from EMA and reset optimizer"
                            )
                        else:
                            # EMA is also corrupt; reload from the best checkpoint on disk
                            monitor_write(
                                f"[NaN weights] Epoch {epoch}, batch {batch_index}: "
                                "model AND EMA weights corrupted — reloading best checkpoint"
                            )
                            self._reload_best_checkpoint(epoch)
                            # The batch loop cannot continue after a full reload; break
                            # so fit() starts the next epoch cleanly with restored weights.
                            break
                    # ──────────────────────────────────────────────────────────

                    try:
                        self.update_ema()
                    except Exception as ema_exc:
                        monitor_write(
                            f"[EMA ERROR] Epoch {epoch}, batch {batch_index}: "
                            f"{type(ema_exc).__name__}: {ema_exc}\n"
                            f"{traceback.format_exc()}"
                        )
                    self.optimizer.zero_grad(set_to_none=True)

                # Only accumulate metrics when all loss components are finite to
                # prevent a single bad batch from poisoning the epoch running avg.
                if torch.isfinite(losses["total"]):
                    metrics = self.metrics_from_fused(x0_pred.detach(), source1, source2)
                    self.update_totals(totals, losses, metrics)
                    processed += 1
                    bar.set_postfix(
                        loss=f"{totals['total_loss'] / processed:.4f}",
                        ssim=f"{totals['ssim'] / processed:.4f}",
                        psnr=f"{totals['psnr'] / processed:.2f}",
                    )
            except Exception as batch_exc:
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.startswith("cuda"):
                    # Two-stage VRAM recovery: Python GC first (releases tensor
                    # references), then empty_cache returns freed blocks to the
                    # allocator pool.  Critical after OOM — without GC the
                    # tensors stay alive in locals/cycle and empty_cache does
                    # nothing, so every subsequent batch also OOMs.
                    gc.collect()
                    torch.cuda.empty_cache()
                monitor_write(
                    f"[Batch ERROR] Diffusion epoch {epoch}, batch {batch_index}: "
                    f"{type(batch_exc).__name__}: {batch_exc} — skipping\n"
                    f"{traceback.format_exc()}"
                )
        result = self.average_totals(totals, processed)
        result["skipped_batches"] = skipped
        return result

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals = self.empty_totals()
        processed = 0
        # Use 50-step DDIM sampling — same process as test-time generation.
        # This prevents the train/test SSIM gap caused by single-step x0_pred
        # from a random noisy timestep (which inflated val_ssim to ~0.72).
        val_sampling_steps = 50
        eval_model = self.ema_model if self.use_ema else self.model
        for batch_index, batch in enumerate(tqdm(self.val_loader, desc=f"Validate {self.pair}", leave=False, dynamic_ncols=True, file=sys.stdout), start=1):
            source1 = batch["source1"].to(self.device)
            source2 = batch["source2"].to(self.device)
            # Loss is still computed via forward_loss (fast, needed for LR scheduler).
            losses, _ = self.forward_loss(source1, source2)
            # Metrics are computed on the actual DDIM-denoised output so val_ssim
            # matches what the test evaluator measures.
            fused, initial = sample_refined(
                eval_model, self.scheduler, source1, source2,
                sampling_steps=val_sampling_steps,
                prediction_type="x0",
            )
            metrics = self.metrics_from_fused(fused, source1, source2)
            self.update_totals(totals, losses, metrics)
            if batch_index <= 3:
                sample_path = self.sample_dir / f"epoch_{epoch:03d}_{batch_index:02d}_comparison.png"
                save_comparison_panel(
                    [source1[0], source2[0], initial[0], fused[0]],
                    [self.source1_label, self.source2_label, "Initial", "Diffusion"],
                    sample_path,
                )
            processed += 1
        return self.average_totals(totals, processed)

    def build_history_row(self, epoch, train_values, val_values, is_best):
        row = {"epoch": epoch, "learning_rate": self.optimizer.param_groups[0]["lr"]}
        for prefix, values in [("train", train_values), ("val", val_values)]:
            for key in [
                "total_loss",
                "l1_loss",
                "ssim_loss",
                "gradient_loss",
                "noise_loss",
                "hf_loss",
                "edge_loss",
                "laplacian_loss",
                "ms_ssim_loss",
                "local_contrast_loss",
                "perceptual_loss",
                "msfd_loss",
                "ssim",
                "psnr",
                "mi",
                "en",
                "cc",
                "fmi",
                "sf",
                "ag",
            ]:
                row[f"{prefix}_{key}"] = values.get(key)
        row["best_epoch"] = self.best_epoch
        row["best_val_ssim"] = self.best_val_ssim
        row["is_best"] = is_best
        return row

    def append_history(self, row):
        self.history = [item for item in self.history if int(item["epoch"]) != int(row["epoch"])]
        self.history.append(row)
        self.history.sort(key=lambda item: int(item["epoch"]))
        with self.history_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self.history)
        compact_fields = [
            "epoch",
            "train_loss",
            "noise_loss",
            "l1_loss",
            "ssim_loss",
            "grad_loss",
            "hf_loss",
            "edge_loss",
            "laplacian_loss",
            "ms_ssim_loss",
            "local_contrast_loss",
            "train_ssim",
            "val_ssim",
            "best_epoch",
            "learning_rate",
        ]
        with self.global_history_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=compact_fields)
            writer.writeheader()
            for item in self.history:
                writer.writerow(
                    {
                        "epoch": item.get("epoch"),
                        "train_loss": item.get("train_total_loss"),
                        "noise_loss": item.get("train_noise_loss"),
                        "l1_loss": item.get("train_l1_loss"),
                        "ssim_loss": item.get("train_ssim_loss"),
                        "grad_loss": item.get("train_gradient_loss"),
                        "hf_loss": item.get("train_hf_loss"),
                        "edge_loss": item.get("train_edge_loss"),
                        "laplacian_loss": item.get("train_laplacian_loss"),
                        "ms_ssim_loss": item.get("train_ms_ssim_loss"),
                        "local_contrast_loss": item.get("train_local_contrast_loss"),
                        "train_ssim": item.get("train_ssim"),
                        "val_ssim": item.get("val_ssim"),
                        "best_epoch": item.get("best_epoch"),
                        "learning_rate": item.get("learning_rate"),
                    }
                )
        self.history_json_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def plot_history(self):
        if not self.history:
            return
        epochs = [int(row["epoch"]) for row in self.history]
        for key, filename, title, ylabel in [
            ("train_total_loss", f"training_loss_curve_{self.pair}.png", f"{self.pair} diffusion training loss", "Loss"),
            ("val_ssim", f"validation_ssim_curve_{self.pair}.png", f"{self.pair} validation SSIM", "SSIM"),
        ]:
            points = [(int(row["epoch"]), row.get(key)) for row in self.history if row.get(key) not in (None, "")]
            plt.figure(figsize=(8, 5), dpi=160)
            if key == "train_total_loss":
                plt.plot(epochs, [float(row["train_total_loss"]) for row in self.history], label="train_total_loss", linewidth=2)
                val_points = [(int(row["epoch"]), row.get("val_total_loss")) for row in self.history if row.get("val_total_loss") not in (None, "")]
                if val_points:
                    plt.plot([p[0] for p in val_points], [float(p[1]) for p in val_points], label="val_total_loss", linewidth=2)
            elif points:
                plt.plot([p[0] for p in points], [float(p[1]) for p in points], label=key, linewidth=2)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(self.graph_dir / filename)
            plt.close()

    def append_training_log(self, epoch, train_values, val_values, checkpoint_path, is_best):
        line = (
            f"epoch={epoch:03d} train_loss={train_values.get('total_loss'):.6f} "
            f"noise_loss={train_values.get('noise_loss'):.6f} "
            f"l1_loss={train_values.get('l1_loss'):.6f} "
            f"ssim_loss={train_values.get('ssim_loss'):.6f} "
            f"grad_loss={train_values.get('gradient_loss'):.6f} "
            f"hf_loss={train_values.get('hf_loss'):.6f} "
            f"edge_loss={train_values.get('edge_loss'):.6f} "
            f"laplacian_loss={train_values.get('laplacian_loss'):.6f} "
            f"ms_ssim_loss={train_values.get('ms_ssim_loss'):.6f} "
            f"local_contrast_loss={train_values.get('local_contrast_loss'):.6f} "
            f"perceptual_loss={train_values.get('perceptual_loss'):.6f} "
            f"val_loss={val_values.get('total_loss') if val_values.get('total_loss') is not None else 'skipped'} "
            f"val_ssim={val_values.get('ssim') if val_values.get('ssim') is not None else 'skipped'} "
            f"checkpoint={checkpoint_path} is_best={is_best}"
        )
        for path in [self.training_log_path, self.global_training_log_path]:
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")

    def write_readme(self):
        path = self.output_root / "pipeline_diffusion.md"
        if path.exists():
            return
        path.write_text(
            "# Diffusion Medical Image Fusion Pipeline\n\n"
            "This folder contains an independent conditional diffusion pipeline for AANLIB CT-MRI, PET-MRI, and SPECT-MRI fusion. "
            "It reuses the shared paired-image loader and preprocessing, writes only under `outputs/models/diffusion`, and does not train with the GAN pipeline.\n\n"
            "## Training\n\n"
            "`python train_diffusion.py --dataset-root \"data\\\\raw\\\\AANLIB\" --pair ct_mri --epochs 120 --batch-size 8 --micro-batch 2 --num-workers 0 --val-every 5 --auto-resume`\n\n"
            "Use `pet_mri` or `spect_mri` for the other pairs.\n\n"
            "## Generation\n\n"
            "`python generate_all_fused_diffusion.py --dataset-root \"data\\\\raw\\\\AANLIB\" --pair ct_mri --split test --checkpoint outputs\\\\models\\\\diffusion\\\\aanlib_ct_mri\\\\checkpoints\\\\best_diffusion.pt`\n\n"
            "Generated images are saved in `outputs/models/diffusion/images/aanlib/<pair>/<split>`.\n\n"
            "## Metrics And Graphs\n\n"
            "Run `python scripts/calculate_diffusion_metrics.py` and `python scripts/plot_diffusion_results.py`. Metrics and graphs are saved under the diffusion `metrics` and `graphs` folders.\n\n"
            "## GAN Comparison\n\n"
            "GAN files remain under `outputs/models/gan`. When GAN metrics/images exist, the diffusion scripts can create comparison graphs and thesis panels without overwriting GAN outputs.\n",
            encoding="utf-8",
        )

    def fit(self):
        self.write_readme()
        monitor_write(f"Checkpoint path: {self.checkpoint_dir}")
        monitor_write(f"Sample folder: {self.sample_dir}")
        monitor_write(f"Graph folder: {self.graph_dir}")
        monitor_write(f"Metrics folder: {self.metrics_dir}")
        if self.start_epoch > self.epochs:
            monitor_write("Checkpoint is already at or beyond requested epochs. Nothing to train.")
            return False
        _consecutive_nan_epochs = 0
        _max_nan_recovery = 5  # stop if we can't recover after this many attempts
        for epoch in range(self.start_epoch, self.epochs + 1):
            # ── Pre-epoch NaN weight guard ────────────────────────────────────────
            # Weights become NaN when a corrupted gradient update slips through AMP.
            # Once NaN, every forward pass is NaN, every batch is skipped, and the
            # model never updates — training becomes an infinite no-op.  Detect this
            # before the data loop rather than after 68 useless batches.
            with torch.no_grad():
                nan_in_model = any(not p.data.isfinite().all() for p in self.model.parameters())
            if nan_in_model:
                _consecutive_nan_epochs += 1
                monitor_write(
                    f"[NaN Recovery] Epoch {epoch}: NaN weights detected before training "
                    f"(attempt {_consecutive_nan_epochs}/{_max_nan_recovery}) — reloading best checkpoint."
                )
                if _consecutive_nan_epochs >= _max_nan_recovery:
                    monitor_write(
                        f"[NaN Recovery] {_consecutive_nan_epochs} consecutive NaN epochs — "
                        "stopping training to avoid infinite loop."
                    )
                    break
                self._reload_best_checkpoint(epoch)
                # Don't skip the epoch — proceed with (hopefully) clean weights
            # ─────────────────────────────────────────────────────────────────────
            train_values = self.train_epoch(epoch)
            # ── Post-epoch: detect all-NaN epoch (fast-path or all-batches-skipped) ─
            all_skipped = (
                train_values.get("skipped_batches") == -1  # fast-path sentinel
                or train_values.get("skipped_batches", 0) >= len(self.train_loader)
            )
            if all_skipped:
                _consecutive_nan_epochs += 1
                monitor_write(
                    f"[NaN Recovery] Epoch {epoch}: all batches skipped "
                    f"(consecutive NaN epochs: {_consecutive_nan_epochs}/{_max_nan_recovery})."
                )
                if _consecutive_nan_epochs >= _max_nan_recovery:
                    monitor_write(
                        f"[NaN Recovery] {_consecutive_nan_epochs} consecutive NaN epochs — "
                        "stopping training."
                    )
                    break
                self._reload_best_checkpoint(epoch)
            else:
                _consecutive_nan_epochs = 0  # reset on any productive epoch
            # ─────────────────────────────────────────────────────────────────────
            should_validate = epoch % self.val_every == 0 or epoch == self.epochs
            val_values = self.validate(epoch) if should_validate else {}
            val_loss = val_values.get("total_loss")
            # CosineAnnealingWarmRestarts steps every epoch (not loss-dependent)
            self.lr_scheduler.step()
            val_ssim = val_values.get("ssim")
            previous_best = self.best_val_ssim
            is_best = val_ssim is not None and float(val_ssim) > previous_best
            meaningful_improvement = val_ssim is not None and float(val_ssim) > previous_best + self.min_delta
            if is_best:
                self.best_val_ssim = float(val_ssim)
                self.best_epoch = epoch
            if meaningful_improvement:
                self.no_improve_epochs = 0
            elif should_validate:
                self.no_improve_epochs += 1
            checkpoint_path = self.save_checkpoint(epoch, is_best=is_best)
            row = self.build_history_row(epoch, train_values, val_values, is_best)
            self.append_history(row)
            self.plot_history()
            self.append_training_log(epoch, train_values, val_values, checkpoint_path, is_best)
            val_psnr = val_values.get("psnr")
            monitor_write(
                f"Epoch {epoch:03d}/{self.epochs:03d} | train_loss={train_values['total_loss']:.4f} | "
                f"noise={train_values['noise_loss']:.4f} | l1={train_values['l1_loss']:.4f} | "
                f"ssim_loss={train_values['ssim_loss']:.4f} | grad={train_values['gradient_loss']:.4f} | hf={train_values['hf_loss']:.4f} | "
                f"edge={train_values['edge_loss']:.4f} | lap={train_values['laplacian_loss']:.4f} | "
                f"val_ssim={val_ssim if val_ssim is not None else 'skipped'} | "
                f"val_psnr={f'{val_psnr:.2f}' if val_psnr is not None else 'skipped'} | "
                f"best_epoch={self.best_epoch}"
            )
            if is_best:
                monitor_write(f"Best checkpoint path: {self.checkpoint_dir / 'best_diffusion.pt'}")
                if self.use_ema:
                    monitor_write(f"Best EMA checkpoint path: {self.checkpoint_dir / 'best_diffusion_ema.pt'}")

            # Memory cleanup every epoch to prevent CUDA fragmentation over long runs.
            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            if self.early_stopping and should_validate and self.no_improve_epochs >= self.patience:
                monitor_write(f"Early stopping: no validation SSIM improvement for {self.no_improve_epochs} validation checks.")
                break
        monitor_write(f"Best epoch: {self.best_epoch}")
        monitor_write(f"Best val SSIM: {self.best_val_ssim}")
        monitor_write(f"Best checkpoint path: {self.checkpoint_dir / 'best_diffusion.pt'}")
        return True
