from pathlib import Path
import copy
import csv
import json
import math
import os
import sys

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
from src.diffusion.model import ConditionalUNet
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
        epochs=120,
        lr=1e-4,
        timesteps=1000,
        sampling_steps=150,
        lambda_l1=1.2,
        lambda_ssim=2.5,
        lambda_grad=1.6,
        lambda_hf=1.0,
        lambda_noise=1.2,
        lambda_ms_ssim=0.8,
        lambda_local_contrast=0.4,
        lambda_msfd=1.0,
        use_msfd_guidance=False,
        use_ema=True,
        ema_decay=0.999,
        early_stopping=False,
        patience=30,
        min_delta=0.0005,
        val_split=0.15,
        val_every=5,
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
        self.lambda_noise = float(lambda_noise)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.no_improve_epochs = 0
        self.device = resolve_training_device(device, allow_cpu=allow_cpu)
        self.amp_enabled = self.device.startswith("cuda")
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
        )
        loader_kwargs = {"num_workers": self.num_workers, "pin_memory": self.device.startswith("cuda")}
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        self.train_loader = DataLoader(train_dataset, batch_size=self.micro_batch_size, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs) if val_dataset is not None else None

        self.model = ConditionalUNet(in_channels=4, base_channels=32, time_dim=128).to(self.device)
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
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

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
        return path if path.exists() else None

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("ema_model_state_dict") is not None:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        else:
            self.ema_model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_val_ssim = float(checkpoint.get("best_val_ssim", self.best_val_ssim))
        monitor_write(f"Loaded diffusion checkpoint: {checkpoint_path}")

    def update_ema(self):
        if not self.use_ema:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.mul_(self.ema_decay).add_(param.detach(), alpha=1.0 - self.ema_decay)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
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
                "timesteps": self.timesteps,
                "sampling_steps": self.sampling_steps,
                "use_msfd_guidance": self.use_msfd_guidance,
                "lambda_msfd": self.lambda_msfd,
                "lambda_noise": self.lambda_noise,
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
            },
            "scaler": self.scaler.state_dict(),
        }

    def save_checkpoint(self, epoch, is_best=False):
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
        noise_pred = self.model(noisy, source1, source2, initial, t)
        x0_pred = self.scheduler.predict_x0(noisy, noise_pred, t)
        noise_loss = F.mse_loss(noise_pred, noise)
        fusion_losses = self.loss_fn(x0_pred, target, source1, source2, msfd_target=msfd, use_msfd=self.use_msfd_guidance)
        fusion_losses["total"] = fusion_losses["total"] + self.lambda_noise * noise_loss
        fusion_losses["noise"] = noise_loss.detach()
        return fusion_losses, x0_pred

    def metrics_from_fused(self, fused, source1, source2):
        metrics = evaluate_fusion(fused, source2, source1)
        return {
            "ssim": _avg(metrics, "SSIM_MRI", "SSIM_CT"),
            "psnr": _avg(metrics, "PSNR_MRI", "PSNR_CT"),
            "mi": _avg(metrics, "MI_MRI", "MI_CT"),
            "en": metrics.get("EN", 0.0),
            "cc": metrics.get("CC", 0.0),
            "fmi": metrics.get("FMI", 0.0),
            "sf": metrics.get("SF", 0.0),
            "ag": metrics.get("AG", 0.0),
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
        totals["msfd_loss"] += float(losses["msfd"].cpu())
        for key, value in metrics.items():
            totals[key] += float(value)

    def average_totals(self, totals, count):
        count = max(1, count)
        return {key: value / count for key, value in totals.items()}

    def train_epoch(self, epoch):
        self.model.train()
        totals = self.empty_totals()
        self.optimizer.zero_grad(set_to_none=True)
        processed = 0
        bar = tqdm(self.train_loader, desc=f"Diffusion epoch {epoch}/{self.epochs}", dynamic_ncols=True, file=sys.stdout)
        for batch_index, batch in enumerate(bar, start=1):
            source1 = batch["source1"].to(self.device)
            source2 = batch["source2"].to(self.device)
            with autocast("cuda", enabled=self.amp_enabled):
                losses, x0_pred = self.forward_loss(source1, source2)
                loss = losses["total"] / self.accumulation_steps
            self.scaler.scale(loss).backward()
            if batch_index % self.accumulation_steps == 0 or batch_index == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.update_ema()
                self.optimizer.zero_grad(set_to_none=True)
            metrics = self.metrics_from_fused(x0_pred.detach(), source1, source2)
            self.update_totals(totals, losses, metrics)
            processed += 1
            bar.set_postfix(loss=f"{totals['total_loss'] / processed:.4f}", ssim=f"{totals['ssim'] / processed:.4f}")
        return self.average_totals(totals, processed)

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals = self.empty_totals()
        processed = 0
        for batch_index, batch in enumerate(tqdm(self.val_loader, desc=f"Validate {self.pair}", leave=False, dynamic_ncols=True, file=sys.stdout), start=1):
            source1 = batch["source1"].to(self.device)
            source2 = batch["source2"].to(self.device)
            losses, x0_pred = self.forward_loss(source1, source2)
            metrics = self.metrics_from_fused(x0_pred, source1, source2)
            self.update_totals(totals, losses, metrics)
            if batch_index <= 3:
                initial = initial_fusion(source1, source2)
                sample_path = self.sample_dir / f"epoch_{epoch:03d}_{batch_index:02d}_comparison.png"
                save_comparison_panel(
                    [source1[0], source2[0], initial[0], x0_pred[0]],
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
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_values = self.train_epoch(epoch)
            should_validate = epoch % self.val_every == 0 or epoch == self.epochs
            val_values = self.validate(epoch) if should_validate else {}
            val_loss = val_values.get("total_loss")
            if val_loss is not None:
                self.lr_scheduler.step(val_loss)
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
            monitor_write(
                f"Epoch {epoch:03d}/{self.epochs:03d} | train_loss={train_values['total_loss']:.4f} | "
                f"noise={train_values['noise_loss']:.4f} | l1={train_values['l1_loss']:.4f} | "
                f"ssim_loss={train_values['ssim_loss']:.4f} | grad={train_values['gradient_loss']:.4f} | hf={train_values['hf_loss']:.4f} | "
                f"edge={train_values['edge_loss']:.4f} | lap={train_values['laplacian_loss']:.4f} | "
                f"val_ssim={val_ssim if val_ssim is not None else 'skipped'} | best_epoch={self.best_epoch}"
            )
            if is_best:
                monitor_write(f"Best checkpoint path: {self.checkpoint_dir / 'best_diffusion.pt'}")
                if self.use_ema:
                    monitor_write(f"Best EMA checkpoint path: {self.checkpoint_dir / 'best_diffusion_ema.pt'}")
            if self.early_stopping and should_validate and self.no_improve_epochs >= self.patience:
                monitor_write(f"Early stopping: no validation SSIM improvement for {self.no_improve_epochs} validation checks.")
                break
        monitor_write(f"Best epoch: {self.best_epoch}")
        monitor_write(f"Best val SSIM: {self.best_val_ssim}")
        monitor_write(f"Best checkpoint path: {self.checkpoint_dir / 'best_diffusion.pt'}")
        return True
