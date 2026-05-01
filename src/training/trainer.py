from pathlib import Path
import csv
import json
import math
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from src.data.paired_dataset import CombinedMedicalFusionDataset, DEFAULT_TRAIN_ROOTS
from src.evaluation.metrics import evaluate_fusion
from src.fusion.decomposition import multiscale_fuse
from src.models.gan import FusionGenerator, PatchDiscriminator
from src.training.losses import MedicalFusionGANLoss, gan_discriminator_loss


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_total_loss",
    "val_total_loss",
    "train_fusion_loss",
    "val_fusion_loss",
    "train_gradient_loss",
    "val_gradient_loss",
    "train_gan_loss",
    "val_gan_loss",
    "train_d1_loss",
    "val_d1_loss",
    "train_d2_loss",
    "val_d2_loss",
    "train_psnr",
    "val_psnr",
    "train_ssim",
    "val_ssim",
    "train_sf",
    "val_sf",
    "train_ms",
    "val_ms",
    "best_epoch",
    "best_val_ssim",
    "best_val_loss",
    "is_best",
]


def monitor_write(message):
    tqdm.write(str(message))


def _save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    cv2.imwrite(str(path), (image * 255.0).astype("uint8"))


def _avg_source_metric(metrics, left_key, right_key):
    return 0.5 * (metrics.get(left_key, 0.0) + metrics.get(right_key, 0.0))


def resolve_training_device(device, allow_cpu=False):
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        print("[Device] WARNING: CUDA is not available. Training is stopped because CPU fallback is disabled.")
        print("[Device] Install/use a CUDA-enabled PyTorch build and NVIDIA driver, or pass --allow-cpu to train on CPU.")
        if allow_cpu:
            return "cpu"
        raise RuntimeError("CUDA is not available. Refusing to train on CPU without --allow-cpu.")
    if requested in {"cuda", "gpu"} or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access an NVIDIA CUDA GPU.")
        return "cuda:0" if requested in {"cuda", "gpu"} else requested
    if requested == "cpu":
        if not allow_cpu:
            print("[Device] WARNING: CPU training was requested, but CPU fallback is disabled.")
            print("[Device] Pass --allow-cpu only if you intentionally want to train without CUDA.")
            raise RuntimeError("Refusing to train on CPU without --allow-cpu.")
        return "cpu"
    raise ValueError(f"Unsupported device: {device}. Use auto, cuda, cuda:0, or cpu.")


def describe_training_device(device):
    if str(device).startswith("cuda"):
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return "cpu"


class GANFusionTrainer:
    def __init__(
        self,
        dataset_root=None,
        train_roots=None,
        output_dir="outputs/models/gan",
        image_size=256,
        batch_size=204,
        micro_batch=4,
        epochs=50,
        lr=1e-4,
        discriminator_lr_factor=0.5,
        lambda_gan=0.005,
        lambda_grad=7.5,
        val_split=0.1,
        val_every=1,
        patience=8,
        min_delta=1e-4,
        num_workers=0,
        device=None,
        allow_cpu=False,
        max_items=None,
        resume=None,
        auto_resume=False,
    ):
        self.dataset_root = dataset_root
        self.train_roots = train_roots or ([dataset_root] if dataset_root else DEFAULT_TRAIN_ROOTS)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.history_dir = self.output_dir / "history"
        self.graph_dir = self.output_dir / "graphs"
        self.report_dir = self.output_dir / "training_reports"
        self.history_csv_path = self.history_dir / "training_history.csv"
        self.history_json_path = self.history_dir / "training_history.json"
        self.history_jsonl_path = self.history_dir / "training_history.jsonl"
        self.summary_path = self.report_dir / "summary.txt"
        self.best_metrics_path = self.report_dir / "best_metrics.json"
        self.fitting_report_path = self.report_dir / "fitting_analysis.txt"
        self.epochs = epochs
        self.start_epoch = 1
        self.best_epoch = None
        self.best_val_ssim = float("-inf")
        self.best_val_loss = float("inf")
        self.best_metric = float("-inf")
        self.no_improve_epochs = 0
        self.patience = patience
        self.min_delta = min_delta
        self.lr_g = lr
        self.lr_d = lr * discriminator_lr_factor
        self.device = resolve_training_device(device, allow_cpu=allow_cpu)
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        monitor_write(f"[Device] Selected runtime device: {describe_training_device(self.device)}")
        self.history = []
        self.requested_batch_size = batch_size
        if micro_batch < 1:
            raise ValueError("--micro-batch must be at least 1.")
        if val_every < 1:
            raise ValueError("--val-every must be at least 1.")
        self.micro_batch_size = micro_batch
        self.accumulation_steps = math.ceil(batch_size / self.micro_batch_size)
        self.effective_batch_size = self.micro_batch_size * self.accumulation_steps
        self.num_workers = num_workers
        self.val_every = val_every
        self.amp_enabled = self.device.startswith("cuda")
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.max_grad_norm = 1.0
        self.printed_first_batch_stats = False
        self.nan_reported = False

        self._ensure_output_dirs()

        dataset = CombinedMedicalFusionDataset(
            self.train_roots,
            image_size=image_size,
            max_items=max_items,
            split_name="train",
            strict=True,
        )
        monitor_write(f"[Dataset] Combined train pairs loaded: {len(dataset)}")
        val_count = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
        train_count = len(dataset) - val_count
        if val_count > 0:
            train_dataset, val_dataset = random_split(
                dataset,
                [train_count, val_count],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            train_dataset, val_dataset = dataset, None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.startswith("cuda"),
            persistent_workers=num_workers > 0,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self.device.startswith("cuda"),
                persistent_workers=num_workers > 0,
            )

        self.generator = FusionGenerator().to(self.device)
        self.discriminator1 = PatchDiscriminator(in_channels=1).to(self.device)
        self.discriminator2 = PatchDiscriminator(in_channels=1).to(self.device)

        self.generator_loss = MedicalFusionGANLoss(lambda_gan=lambda_gan, lambda_grad=lambda_grad)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        self.optimizer_d1 = torch.optim.Adam(self.discriminator1.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        self.optimizer_d2 = torch.optim.Adam(self.discriminator2.parameters(), lr=self.lr_d, betas=(0.5, 0.999))

        monitor_write(f"Requested batch: {self.requested_batch_size}")
        monitor_write(f"Micro batch: {self.micro_batch_size}")
        monitor_write(f"Accumulation steps: {self.accumulation_steps}")
        monitor_write(f"Effective batch: {self.effective_batch_size}")
        monitor_write(f"Num workers: {self.num_workers}")
        monitor_write(f"Validation interval: every {self.val_every} epoch(s), plus final epoch")
        monitor_write(f"AMP mixed precision: {'enabled' if self.amp_enabled else 'disabled'}")
        monitor_write(f"Generator LR: {self.lr_g:.2e}")
        monitor_write(f"Discriminator LR: {self.lr_d:.2e}")

        self.load_history()
        self.restore_best_from_history()

        if resume and auto_resume:
            raise ValueError("Use either resume or auto_resume, not both.")
        if resume:
            self.load_checkpoint(resume)
        elif auto_resume:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint is None:
                monitor_write(f"No checkpoint found in {self.checkpoint_dir}. Starting from scratch.")
            else:
                self.load_checkpoint(latest_checkpoint)

    def _ensure_output_dirs(self):
        for folder in [self.checkpoint_dir, self.sample_dir, self.history_dir, self.graph_dir, self.report_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self):
        checkpoints = []
        for folder in [self.checkpoint_dir, Path("outputs") / "models" / "gan" / "checkpoints"]:
            if not folder.exists():
                continue
            for path in folder.glob("gan_epoch_*.pt"):
                try:
                    epoch = int(path.stem.split("_")[-1])
                except ValueError:
                    continue
                checkpoints.append((epoch, path))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda item: item[0])[1]

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        required_keys = [
            "epoch",
            "generator",
            "discriminator1",
            "discriminator2",
            "optimizer_g",
            "optimizer_d1",
            "optimizer_d2",
        ]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"Checkpoint is missing keys: {missing_keys}")

        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator1.load_state_dict(checkpoint["discriminator1"])
        self.discriminator2.load_state_dict(checkpoint["discriminator2"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d1.load_state_dict(checkpoint["optimizer_d1"])
        self.optimizer_d2.load_state_dict(checkpoint["optimizer_d2"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
        self.no_improve_epochs = int(checkpoint.get("no_improve_epochs", self.no_improve_epochs))
        self._set_optimizer_lr(self.optimizer_g, self.lr_g)
        self._set_optimizer_lr(self.optimizer_d1, self.lr_d)
        self._set_optimizer_lr(self.optimizer_d2, self.lr_d)

        saved_epoch = int(checkpoint["epoch"])
        self.start_epoch = saved_epoch + 1
        monitor_write(f"Loaded checkpoint: {checkpoint_path}")
        monitor_write(f"Resuming from epoch {self.start_epoch}")

    @staticmethod
    def _set_optimizer_lr(optimizer, lr):
        for group in optimizer.param_groups:
            group["lr"] = lr

    @staticmethod
    def _set_requires_grad(model, enabled):
        for parameter in model.parameters():
            parameter.requires_grad_(enabled)

    def is_finite(self, value, name, epoch=None, batch_index=None):
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        ok = torch.isfinite(tensor).all().item()
        if not ok and not self.nan_reported:
            location = ""
            if epoch is not None and batch_index is not None:
                location = f" at epoch {epoch}, batch {batch_index}"
            monitor_write(f"[NaN/Inf detected first] {name}{location}")
            self.nan_reported = True
        return ok

    def tensor_stats(self, name, tensor):
        finite = torch.isfinite(tensor)
        if finite.any():
            safe = tensor[finite]
            return (
                f"{name}: min={safe.min().item():.6f}, "
                f"max={safe.max().item():.6f}, mean={safe.mean().item():.6f}, "
                f"finite={finite.all().item()}"
            )
        return f"{name}: no finite values"

    def safe_fused(self, fused, name, epoch, batch_index):
        if self.is_finite(fused, name, epoch, batch_index):
            return fused.clamp(0.0, 1.0), True
        repaired = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        monitor_write(f"[Clamp/Sanitize] {name} repaired before discriminator/loss at epoch {epoch}, batch {batch_index}")
        return repaired, False

    def print_first_batch_stats(self, ct, mri, pre_activation, post_activation, fused, g_losses, d1_loss, d2_loss):
        if self.printed_first_batch_stats:
            return
        monitor_write("[First batch tensor stats]")
        monitor_write(self.tensor_stats("ct", ct.detach()))
        monitor_write(self.tensor_stats("mri", mri.detach()))
        monitor_write(self.tensor_stats("generator_pre_final_activation", pre_activation.detach()))
        monitor_write(self.tensor_stats("generator_post_final_activation", post_activation.detach()))
        monitor_write(self.tensor_stats("fused", fused.detach()))
        monitor_write(self.tensor_stats("generator_total_loss", g_losses["total"].detach()))
        monitor_write(self.tensor_stats("fusion_loss", g_losses["fusion"].detach()))
        monitor_write(self.tensor_stats("gradient_loss", g_losses["gradient"].detach()))
        monitor_write(self.tensor_stats("gan_loss", g_losses["gan"].detach()))
        monitor_write(self.tensor_stats("d1_loss", d1_loss.detach()))
        monitor_write(self.tensor_stats("d2_loss", d2_loss.detach()))
        self.printed_first_batch_stats = True

    def has_bad_gradients(self, model, name):
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                self.is_finite(parameter.grad, f"{name}.grad")
                return True
        return False

    def zero_all_gradients(self):
        self.optimizer_d1.zero_grad(set_to_none=True)
        self.optimizer_d2.zero_grad(set_to_none=True)
        self.optimizer_g.zero_grad(set_to_none=True)

    def _real_style_target(self, ct, mri):
        selector = torch.rand(ct.shape[0], 1, 1, 1, device=ct.device)
        return torch.where(selector > 0.5, ct, mri)

    def _discriminator_losses(self, fused, ct, mri):
        real_style = self._real_style_target(ct, mri)
        real_fused_target = multiscale_fuse(mri=mri, ct=ct).detach()
        d1_real_logits = self.discriminator1(real_fused_target)
        d1_fake_logits = self.discriminator1(fused.detach())
        d2_real_logits = self.discriminator2(real_style)
        d2_fake_logits = self.discriminator2(fused.detach())
        d1_loss = gan_discriminator_loss(d1_real_logits, d1_fake_logits)
        d2_loss = gan_discriminator_loss(d2_real_logits, d2_fake_logits)
        return d1_loss, d2_loss

    def _train_discriminators(self, fused, ct, mri):
        d1_loss, d2_loss = self._discriminator_losses(fused, ct, mri)

        self.optimizer_d1.zero_grad(set_to_none=True)
        d1_loss.backward(retain_graph=True)
        self.optimizer_d1.step()

        self.optimizer_d2.zero_grad(set_to_none=True)
        d2_loss.backward()
        self.optimizer_d2.step()

        return d1_loss.detach(), d2_loss.detach()

    def _generator_loss_parts(self, fused, ct, mri):
        d1_fake_logits = self.discriminator1(fused)
        d2_fake_logits = self.discriminator2(fused)
        return self.generator_loss(
            fused=fused,
            mri=mri,
            ct=ct,
            d1_fake_logits=d1_fake_logits,
            d2_fake_logits=d2_fake_logits,
        )

    def _train_generator(self, fused, ct, mri):
        self.optimizer_g.zero_grad(set_to_none=True)
        losses = self._generator_loss_parts(fused, ct, mri)
        losses["total"].backward()
        self.optimizer_g.step()
        return losses

    def _metrics_from_fused(self, fused, mri, ct):
        metrics = evaluate_fusion(fused, mri, ct)
        return {
            "psnr": _avg_source_metric(metrics, "PSNR_MRI", "PSNR_CT"),
            "ssim": _avg_source_metric(metrics, "SSIM_MRI", "SSIM_CT"),
            "sf": metrics.get("SF", 0.0),
            "ms": _avg_source_metric(metrics, "MS_MRI", "MS_CT"),
        }

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()

        totals = {
            "total_loss": 0.0,
            "fusion_loss": 0.0,
            "gradient_loss": 0.0,
            "gan_loss": 0.0,
            "d1_loss": 0.0,
            "d2_loss": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "sf": 0.0,
            "ms": 0.0,
        }
        running_loss = 0.0
        self.zero_all_gradients()
        skip_window = False

        bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=True,
            
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(bar, start=1):
            ct = batch["ct"].to(self.device)
            mri = batch["mri"].to(self.device)

            if not self.is_finite(ct, "input ct", epoch, batch_index) or not self.is_finite(mri, "input mri", epoch, batch_index):
                skip_window = True
                continue

            with autocast(enabled=self.amp_enabled):
                fused_for_d, pre_d, post_d = self.generator.forward_with_debug(ct.float(), mri.float())
            self.is_finite(pre_d, "generator output before final activation for discriminator", epoch, batch_index)
            self.is_finite(post_d, "generator output after final activation for discriminator", epoch, batch_index)
            fused_for_d, fused_d_ok = self.safe_fused(fused_for_d, "fused output for discriminator", epoch, batch_index)
            if not fused_d_ok:
                skip_window = True
                continue
            with autocast(enabled=self.amp_enabled):
                d1_loss, d2_loss = self._discriminator_losses(fused_for_d.float(), ct.float(), mri.float())
                d_loss = (d1_loss + d2_loss) / self.accumulation_steps
            if not self.is_finite(d1_loss, "discriminator1 loss", epoch, batch_index) or not self.is_finite(d2_loss, "discriminator2 loss", epoch, batch_index):
                skip_window = True
                continue
            self.scaler.scale(d_loss).backward()

            self._set_requires_grad(self.discriminator1, False)
            self._set_requires_grad(self.discriminator2, False)
            try:
                with autocast(enabled=self.amp_enabled):
                    fused_for_g, pre_g, post_g = self.generator.forward_with_debug(ct.float(), mri.float())
                self.is_finite(pre_g, "generator output before final activation for generator", epoch, batch_index)
                self.is_finite(post_g, "generator output after final activation for generator", epoch, batch_index)
                fused_for_g, fused_g_ok = self.safe_fused(fused_for_g, "fused output for generator", epoch, batch_index)
                if not fused_g_ok:
                    skip_window = True
                    continue
                with autocast(enabled=self.amp_enabled):
                    g_losses = self._generator_loss_parts(fused_for_g.float(), ct.float(), mri.float())
                    g_loss = g_losses["total"] / self.accumulation_steps
                if (
                    not self.is_finite(g_losses["total"], "generator total loss", epoch, batch_index)
                    or not self.is_finite(g_losses["fusion"], "fusion loss", epoch, batch_index)
                    or not self.is_finite(g_losses["gradient"], "gradient loss", epoch, batch_index)
                    or not self.is_finite(g_losses["gan"], "gan loss", epoch, batch_index)
                ):
                    skip_window = True
                    continue
                self.scaler.scale(g_loss).backward()
            finally:
                self._set_requires_grad(self.discriminator1, True)
                self._set_requires_grad(self.discriminator2, True)

            self.print_first_batch_stats(ct, mri, pre_g, post_g, fused_for_g, g_losses, d1_loss, d2_loss)

            should_step = batch_index % self.accumulation_steps == 0 or batch_index == len(self.train_loader)
            if should_step:
                self.scaler.unscale_(self.optimizer_d1)
                self.scaler.unscale_(self.optimizer_d2)
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.discriminator1.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.discriminator2.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)

                bad_grads = (
                    self.has_bad_gradients(self.discriminator1, "discriminator1")
                    or self.has_bad_gradients(self.discriminator2, "discriminator2")
                    or self.has_bad_gradients(self.generator, "generator")
                )
                if skip_window or bad_grads:
                    monitor_write(f"[Skip update] epoch {epoch}, batch {batch_index}: non-finite value in accumulation window.")
                    self.zero_all_gradients()
                    self.scaler.update()
                    skip_window = False
                else:
                    self.scaler.step(self.optimizer_d1)
                    self.scaler.step(self.optimizer_d2)
                    self.scaler.step(self.optimizer_g)
                    self.scaler.update()
                    self.zero_all_gradients()

            metric_values = self._metrics_from_fused(fused_for_g.detach(), mri, ct)

            loss_value = g_losses["total"].item()
            running_loss += loss_value
            totals["total_loss"] += loss_value
            totals["fusion_loss"] += g_losses["fusion"].item()
            totals["gradient_loss"] += g_losses["gradient"].item()
            totals["gan_loss"] += g_losses["gan"].item()
            totals["d1_loss"] += d1_loss.item()
            totals["d2_loss"] += d2_loss.item()
            for key, value in metric_values.items():
                totals[key] += value

            bar.set_postfix(
                lr=f"{self.optimizer_g.param_groups[0]['lr']:.2e}",
                loss=f"{running_loss / batch_index:.4f}",
                accum=f"{batch_index % self.accumulation_steps}/{self.accumulation_steps}",
                best=self.best_epoch or "-",
            )

        count = max(1, len(self.train_loader))
        return {key: value / count for key, value in totals.items()}

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return {}

        self.generator.eval()
        self.discriminator1.eval()
        self.discriminator2.eval()
        totals = {
            "total_loss": 0.0,
            "fusion_loss": 0.0,
            "gradient_loss": 0.0,
            "gan_loss": 0.0,
            "d1_loss": 0.0,
            "d2_loss": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "sf": 0.0,
            "ms": 0.0,
        }

        bar = tqdm(
            self.val_loader,
            desc=f"Validate {epoch}/{self.epochs}",
            leave=True,
            
            dynamic_ncols=True,
        )
        for index, batch in enumerate(bar):
            ct = batch["ct"].to(self.device)
            mri = batch["mri"].to(self.device)
            if not self.is_finite(ct, "validation input ct", epoch, index + 1) or not self.is_finite(mri, "validation input mri", epoch, index + 1):
                continue
            with autocast(enabled=self.amp_enabled):
                fused, pre_val, post_val = self.generator.forward_with_debug(ct.float(), mri.float())
            self.is_finite(pre_val, "validation generator output before final activation", epoch, index + 1)
            self.is_finite(post_val, "validation generator output after final activation", epoch, index + 1)
            fused, fused_ok = self.safe_fused(fused, "validation fused output", epoch, index + 1)
            if not fused_ok:
                continue
            with autocast(enabled=self.amp_enabled):
                g_losses = self._generator_loss_parts(fused.float(), ct.float(), mri.float())
                d1_loss, d2_loss = self._discriminator_losses(fused.float(), ct.float(), mri.float())
            if (
                not self.is_finite(g_losses["total"], "validation generator total loss", epoch, index + 1)
                or not self.is_finite(d1_loss, "validation discriminator1 loss", epoch, index + 1)
                or not self.is_finite(d2_loss, "validation discriminator2 loss", epoch, index + 1)
            ):
                continue
            metric_values = self._metrics_from_fused(fused, mri, ct)

            totals["total_loss"] += g_losses["total"].item()
            totals["fusion_loss"] += g_losses["fusion"].item()
            totals["gradient_loss"] += g_losses["gradient"].item()
            totals["gan_loss"] += g_losses["gan"].item()
            totals["d1_loss"] += d1_loss.item()
            totals["d2_loss"] += d2_loss.item()
            for key, value in metric_values.items():
                totals[key] += value

            if index < 4:
                _save_tensor_image(fused[0], self.sample_dir / f"epoch_{epoch:03d}_sample_{index + 1:02d}.png")

            bar.set_postfix(loss=f"{totals['total_loss'] / (index + 1):.4f}")

        count = max(1, len(self.val_loader))
        return {key: value / count for key, value in totals.items()}

    def save_checkpoint(self, epoch):
        path = self.checkpoint_dir / f"gan_epoch_{epoch:03d}.pt"
        payload = {
            "epoch": epoch,
            "generator": self.generator.state_dict(),
            "discriminator1": self.discriminator1.state_dict(),
            "discriminator2": self.discriminator2.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d1": self.optimizer_d1.state_dict(),
            "optimizer_d2": self.optimizer_d2.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_val_ssim": self.best_val_ssim,
            "best_val_loss": self.best_val_loss,
            "no_improve_epochs": self.no_improve_epochs,
        }
        torch.save(payload, path)
        torch.save(payload, self.checkpoint_dir / "latest_checkpoint.pt")
        torch.save(self.generator.state_dict(), self.checkpoint_dir / "generator_latest.pt")
        return path

    def save_best_checkpoint(self, epoch):
        full_path = self.checkpoint_dir / "best_checkpoint.pt"
        generator_path = self.checkpoint_dir / "best_generator.pt"
        torch.save(
            {
                "epoch": epoch,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "best_val_ssim": self.best_val_ssim,
                "best_val_loss": self.best_val_loss,
                "generator": self.generator.state_dict(),
                "discriminator1": self.discriminator1.state_dict(),
                "discriminator2": self.discriminator2.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d1": self.optimizer_d1.state_dict(),
                "optimizer_d2": self.optimizer_d2.state_dict(),
                "scaler": self.scaler.state_dict(),
                "no_improve_epochs": self.no_improve_epochs,
            },
            full_path,
        )
        torch.save(self.generator.state_dict(), generator_path)
        return full_path, generator_path

    def load_history(self):
        if not self.history_json_path.exists():
            return
        try:
            self.history = json.loads(self.history_json_path.read_text())
        except json.JSONDecodeError:
            monitor_write(f"[Warning] Could not parse history JSON: {self.history_json_path}")
            self.history = []

    def restore_best_from_history(self):
        if not self.history:
            return
        for row in self.history:
            val_ssim = float(row.get("val_ssim") or float("-inf"))
            val_loss = float(row.get("val_total_loss") or float("inf"))
            if self.is_better_epoch(val_ssim, val_loss):
                self.best_epoch = int(row["epoch"])
                self.best_val_ssim = val_ssim
                self.best_val_loss = val_loss
                self.best_metric = val_ssim

    def is_better_epoch(self, val_ssim, val_loss):
        nearly_tied = abs(val_ssim - self.best_val_ssim) <= 1e-4
        return val_ssim > self.best_val_ssim + 1e-4 or (nearly_tied and val_loss < self.best_val_loss)

    def build_history_row(self, epoch, train_values, val_values, is_best):
        return {
            "epoch": epoch,
            "learning_rate": self.optimizer_g.param_groups[0]["lr"],
            "train_total_loss": train_values.get("total_loss"),
            "val_total_loss": val_values.get("total_loss"),
            "train_fusion_loss": train_values.get("fusion_loss"),
            "val_fusion_loss": val_values.get("fusion_loss"),
            "train_gradient_loss": train_values.get("gradient_loss"),
            "val_gradient_loss": val_values.get("gradient_loss"),
            "train_gan_loss": train_values.get("gan_loss"),
            "val_gan_loss": val_values.get("gan_loss"),
            "train_d1_loss": train_values.get("d1_loss"),
            "val_d1_loss": val_values.get("d1_loss"),
            "train_d2_loss": train_values.get("d2_loss"),
            "val_d2_loss": val_values.get("d2_loss"),
            "train_psnr": train_values.get("psnr"),
            "val_psnr": val_values.get("psnr"),
            "train_ssim": train_values.get("ssim"),
            "val_ssim": val_values.get("ssim"),
            "train_sf": train_values.get("sf"),
            "val_sf": val_values.get("sf"),
            "train_ms": train_values.get("ms"),
            "val_ms": val_values.get("ms"),
            "best_epoch": self.best_epoch,
            "best_val_ssim": self.best_val_ssim,
            "best_val_loss": self.best_val_loss,
            "is_best": is_best,
        }

    def append_history(self, row):
        self.history = [item for item in self.history if int(item["epoch"]) != int(row["epoch"])]
        self.history.append(row)
        self.history.sort(key=lambda item: int(item["epoch"]))

        with self.history_csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self.history)

        self.history_json_path.write_text(json.dumps(self.history, indent=2))
        with self.history_jsonl_path.open("w") as jsonl_file:
            for item in self.history:
                jsonl_file.write(json.dumps(item) + "\n")

    def plot_history(self):
        if not self.history:
            return
        graph_specs = [
            ("train_total_loss", "val_total_loss", "train_loss_vs_val_loss.png", "Train Loss vs Validation Loss", "Loss"),
            ("train_ssim", "val_ssim", "train_ssim_vs_val_ssim.png", "Train SSIM vs Validation SSIM", "SSIM"),
            ("train_psnr", "val_psnr", "train_psnr_vs_val_psnr.png", "Train PSNR vs Validation PSNR", "PSNR"),
            ("train_sf", "val_sf", "train_sf_vs_val_sf.png", "Train SF vs Validation SF", "Spatial Frequency"),
            ("train_ms", "val_ms", "train_ms_vs_val_ms.png", "Train MS vs Validation MS", "MS"),
        ]
        epochs = [int(row["epoch"]) for row in self.history]
        for train_key, val_key, filename, title, ylabel in graph_specs:
            train_values = [float(row[train_key]) for row in self.history]
            val_points = [
                (int(row["epoch"]), float(row[val_key]))
                for row in self.history
                if row.get(val_key) is not None
            ]
            plt.figure(figsize=(9, 5), dpi=160)
            plt.plot(epochs, train_values, label=train_key, linewidth=2)
            if val_points:
                val_epochs = [epoch for epoch, _ in val_points]
                val_values = [value for _, value in val_points]
                plt.plot(val_epochs, val_values, label=val_key, linewidth=2)
                if self.best_epoch is not None and self.best_epoch in val_epochs:
                    best_index = val_epochs.index(self.best_epoch)
                    plt.scatter([self.best_epoch], [val_values[best_index]], color="red", zorder=5, label=f"best epoch {self.best_epoch}")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.graph_dir / filename)
            plt.close()

    def fitting_analysis(self):
        validated_history = [row for row in self.history if row.get("val_total_loss") is not None and row.get("val_ssim") is not None]
        if len(validated_history) < 3:
            return "Not enough validated epochs yet for reliable fitting analysis."

        recent = validated_history[-3:]
        train_losses = [float(row["train_total_loss"]) for row in recent]
        val_losses = [float(row["val_total_loss"]) for row in recent]
        train_ssim = [float(row["train_ssim"]) for row in recent]
        val_ssim = [float(row["val_ssim"]) for row in recent]

        train_improves = train_losses[-1] < train_losses[0]
        val_worsens = val_losses[-1] > val_losses[0]
        both_weak = train_ssim[-1] < 0.45 and val_ssim[-1] < 0.45
        val_improves = val_ssim[-1] >= val_ssim[0]
        close_gap = abs(train_ssim[-1] - val_ssim[-1]) < 0.08

        if train_improves and val_worsens:
            return "Overfitting warning: training loss improved while validation loss worsened over recent epochs."
        if both_weak and not val_improves:
            return "Underfitting warning: train and validation SSIM are weak and not improving enough."
        if val_improves and close_gap:
            return "Good fit: validation quality is improving and remains close to training behavior."
        return "Stable/unclear: continue monitoring loss curves and validation SSIM."

    def update_reports(self, row):
        analysis = self.fitting_analysis()
        best_payload = {
            "best_epoch": self.best_epoch,
            "best_val_ssim": self.best_val_ssim,
            "best_val_loss": self.best_val_loss,
            "latest_epoch": row["epoch"],
            "latest_val_ssim": row["val_ssim"],
            "latest_val_loss": row["val_total_loss"],
            "analysis": analysis,
        }
        self.best_metrics_path.write_text(json.dumps(best_payload, indent=2))
        self.summary_path.write_text(
            "\n".join(
                [
                    "GAN Fusion Training Summary",
                    f"Latest epoch: {row['epoch']}",
                    f"Best epoch: {self.best_epoch}",
                    f"Best validation SSIM: {self.best_val_ssim:.6f}",
                    f"Best validation loss: {self.best_val_loss:.6f}",
                    f"Learning rate: {row['learning_rate']}",
                    f"History CSV: {self.history_csv_path}",
                    f"History JSON: {self.history_json_path}",
                    f"Graphs: {self.graph_dir}",
                    f"Analysis: {analysis}",
                ]
            )
        )
        self.fitting_report_path.write_text(analysis + "\n")

    def fit(self):
        monitor_write(f"Training on device: {self.device}")
        monitor_write(f"History: {self.history_dir}")
        monitor_write(f"Graphs: {self.graph_dir}")
        monitor_write(f"Reports: {self.report_dir}")

        if self.start_epoch > self.epochs:
            monitor_write(
                f"Checkpoint is already at epoch {self.start_epoch - 1}. "
                f"Requested total epochs: {self.epochs}. Nothing to train."
            )
            return False

        completed_epoch = None
        epoch_bar = tqdm(
            range(self.start_epoch, self.epochs + 1),
            desc="Training epochs",
            position=0,
            dynamic_ncols=True,
        )
        for epoch in epoch_bar:
            train_values = self.train_epoch(epoch)
            should_validate = epoch % self.val_every == 0 or epoch == self.epochs
            val_values = self.validate(epoch) if should_validate else {}
            val_ssim = float(val_values.get("ssim", float("-inf"))) if should_validate else None
            val_loss = float(val_values.get("total_loss", float("inf"))) if should_validate else None

            is_best = should_validate and self.is_better_epoch(val_ssim, val_loss)
            if is_best:
                self.best_epoch = epoch
                self.best_val_ssim = val_ssim
                self.best_val_loss = val_loss
                self.best_metric = val_ssim
                self.no_improve_epochs = 0
                best_full_path, best_generator_path = self.save_best_checkpoint(epoch)
            else:
                if should_validate:
                    self.no_improve_epochs += 1
                best_full_path = None
                best_generator_path = None

            checkpoint_path = self.save_checkpoint(epoch)
            row = self.build_history_row(epoch, train_values, val_values, is_best)
            self.append_history(row)
            self.plot_history()
            self.update_reports(row)
            completed_epoch = epoch

            epoch_bar.set_postfix(
                lr=f"{self.optimizer_g.param_groups[0]['lr']:.2e}",
                train=f"{train_values['total_loss']:.4f}",
                val=f"{val_loss:.4f}" if val_loss is not None else "skip",
                ssim=f"{val_ssim:.4f}" if val_ssim is not None else "skip",
                best=self.best_epoch,
            )
            val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "skipped"
            val_ssim_text = f"{val_ssim:.4f}" if val_ssim is not None else "skipped"
            monitor_write(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"train_loss={train_values['total_loss']:.4f} | "
                f"val_loss={val_loss_text} | val_ssim={val_ssim_text} | "
                f"best_epoch={self.best_epoch} | checkpoint={checkpoint_path}"
            )
            if is_best:
                monitor_write(f"Best model updated: {best_full_path}")
                monitor_write(f"Best generator updated: {best_generator_path}")
            if should_validate and self.patience > 0 and self.no_improve_epochs >= self.patience:
                monitor_write(f"Early stopping: no validation SSIM/loss improvement for {self.no_improve_epochs} epochs.")
                return False

        return completed_epoch == self.epochs
