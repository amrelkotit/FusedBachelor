from pathlib import Path
import csv
import json
import math
import os
import sys
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")
def force_utf8_console():
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass


force_utf8_console()

TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.paired_dataset import (
    AANLIB_ROOT,
    build_aanlib_train_val_datasets,
    gan_checkpoint_dir,
    gan_graph_dir,
    gan_metrics_dir,
    gan_image_dir,
    gan_logs_dir,
    normalize_pair,
    pair_labels,
)
from src.evaluation.metrics import evaluate_fusion
from src.fusion.decomposition import multiscale_fuse
from src.models.gan import FusionGenerator, PatchDiscriminator
from src.training.losses import MedicalFusionGANLoss, gan_discriminator_loss


OLD_GAN_CHECKPOINT_DIR = Path("outputs") / "models" / "gan" / "checkpoints"


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_total_loss",
    "val_total_loss",
    "train_fusion_loss",
    "val_fusion_loss",
    "train_intensity_loss",
    "val_intensity_loss",
    "train_gradient_loss",
    "val_gradient_loss",
    "train_ssim_loss",
    "val_ssim_loss",
    "train_texture_loss",
    "val_texture_loss",
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
    "train_mi",
    "val_mi",
    "train_ag",
    "val_ag",
    "train_fmi",
    "val_fmi",
    "train_en",
    "val_en",
    "train_cc",
    "val_cc",
    "train_epi",
    "val_epi",
    "train_noise",
    "val_noise",
    "train_ms",
    "val_ms",
    "best_epoch",
    "best_val_ssim",
    "best_val_loss",
    "is_best",
]


def monitor_write(message):
    tqdm.write(str(message), file=sys.stdout)


def _save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    ok = cv2.imwrite(str(path), (image * 255.0).round().astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def _tensor_to_uint8(tensor):
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    return (image * 255.0).round().astype("uint8")


def _save_validation_comparison(ct, mri, fused, output_path, labels):
    ct_u8 = _tensor_to_uint8(ct)
    mri_u8 = _tensor_to_uint8(mri)
    fused_u8 = _tensor_to_uint8(fused)
    comparison = np.concatenate([ct_u8, mri_u8, fused_u8], axis=1)
    label_h = 28
    canvas = np.full((comparison.shape[0] + label_h, comparison.shape[1]), 255, dtype=np.uint8)
    canvas[label_h:, :] = comparison
    for idx, label in enumerate(labels):
        x = idx * ct_u8.shape[1] + 10
        cv2.putText(canvas, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError(f"Failed to save comparison image: {output_path}")


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
        pair="ct_mri",
        train_roots=None,
        output_dir=None,
        image_size=256,
        batch_size=204,
        micro_batch=4,
        epochs=50,
        lr=1e-4,
        discriminator_lr_factor=0.25,
        discriminator_update_interval=2,
        lambda_intensity=1.0,
        lambda_gradient=5.0,
        lambda_ssim=2.0,
        lambda_texture=3.0,
        lambda_gan=0.1,
        lambda_source1_intensity=None,
        lambda_source1_gradient=None,
        lambda_source1_ssim=None,
        lambda_source1_texture=None,
        val_split=0.15,
        val_every=1,
        patience=20,
        min_delta=1e-4,
        num_workers=0,
        device=None,
        allow_cpu=False,
        max_items=None,
        resume=None,
        auto_resume=False,
        debug=False,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root else AANLIB_ROOT
        self.pair = normalize_pair(pair)
        self.source1_label, self.source2_label, self.fused_label = pair_labels(self.pair)
        self.train_roots = train_roots
        self.output_dir = Path(output_dir) if output_dir else gan_checkpoint_dir(self.pair).parent
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = gan_image_dir("aanlib", "train", self.pair)
        self.original_sample_dir = self.sample_dir / "fused_original"
        self.comparison_sample_dir = self.sample_dir / "comparison_panels"
        self.history_dir = gan_logs_dir()
        self.graph_dir = gan_graph_dir()
        self.report_dir = self.output_dir / "training_reports"
        self.history_csv_path = self.graph_dir / f"{self.pair}_training_history.csv"
        self.history_json_path = self.history_dir / f"{self.pair}_training_history.json"
        self.history_jsonl_path = self.history_dir / f"{self.pair}_training_history.jsonl"
        self.training_log_path = self.history_dir / f"{self.pair}_training.log"
        self.summary_path = self.report_dir / "summary.txt"
        self.best_metrics_path = self.report_dir / "best_metrics.json"
        self.fitting_report_path = self.report_dir / "fitting_analysis.txt"
        self.epochs = epochs
        self.start_epoch = 1
        self.best_epoch = None
        self.best_val_ssim = float("-inf")
        self.best_val_loss = float("inf")
        self.best_fmi = float("-inf")
        self.best_psnr = float("-inf")
        self.best_metric = float("-inf")
        self.no_improve_epochs = 0
        self.patience = patience
        self.min_delta = min_delta
        self.lr_g = lr
        self.lr_d = lr * discriminator_lr_factor
        self.discriminator_update_interval = discriminator_update_interval
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
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled)
        self.max_grad_norm = 1.0
        self.printed_first_batch_stats = False
        self.nan_reported = False
        self.debug = debug

        self._ensure_output_dirs()

        train_dataset, val_dataset = build_aanlib_train_val_datasets(
            self.dataset_root,
            self.pair,
            image_size=image_size,
            max_items=max_items,
            val_split=val_split,
            seed=42,
        )
        monitor_write(f"[Dataset] AANLIB {self.pair} train pairs loaded: {len(train_dataset)}")
        monitor_write(f"[Dataset] AANLIB {self.pair} validation pairs loaded: {len(val_dataset) if val_dataset is not None else 0}")

        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": self.device.startswith("cuda"),
        }

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            **loader_kwargs,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                **loader_kwargs,
        )

        self.generator = FusionGenerator().to(self.device)
        self.discriminator1 = PatchDiscriminator(in_channels=1).to(self.device)
        self.discriminator2 = PatchDiscriminator(in_channels=1).to(self.device)

        if self.pair in {"pet_mri", "spect_mri"}:
            lambda_source1_intensity = 0.6 if lambda_source1_intensity is None else lambda_source1_intensity
            lambda_source1_gradient = 2.0 if lambda_source1_gradient is None else lambda_source1_gradient
            lambda_source1_ssim = 0.5 if lambda_source1_ssim is None else lambda_source1_ssim
            lambda_source1_texture = 1.5 if lambda_source1_texture is None else lambda_source1_texture
        else:
            lambda_source1_intensity = 0.0 if lambda_source1_intensity is None else lambda_source1_intensity
            lambda_source1_gradient = 0.0 if lambda_source1_gradient is None else lambda_source1_gradient
            lambda_source1_ssim = 0.0 if lambda_source1_ssim is None else lambda_source1_ssim
            lambda_source1_texture = 0.0 if lambda_source1_texture is None else lambda_source1_texture

        self.generator_loss = MedicalFusionGANLoss(
            lambda_intensity=lambda_intensity,
            lambda_gradient=lambda_gradient,
            lambda_ssim=lambda_ssim,
            lambda_texture=lambda_texture,
            lambda_gan=lambda_gan,
            lambda_source1_intensity=lambda_source1_intensity,
            lambda_source1_gradient=lambda_source1_gradient,
            lambda_source1_ssim=lambda_source1_ssim,
            lambda_source1_texture=lambda_source1_texture,
        )
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
        monitor_write(f"Discriminator update interval: every {self.discriminator_update_interval} batch(es)")
        monitor_write(
            f"Source1 preservation weights: intensity={lambda_source1_intensity:.2f}, "
            f"gradient={lambda_source1_gradient:.2f}, ssim={lambda_source1_ssim:.2f}, "
            f"texture={lambda_source1_texture:.2f}"
        )

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
        for folder in [
            self.checkpoint_dir,
            self.sample_dir,
            self.original_sample_dir,
            self.comparison_sample_dir,
            self.history_dir,
            self.graph_dir,
            self.report_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self):
        checkpoints = []
        for folder in [self.checkpoint_dir, OLD_GAN_CHECKPOINT_DIR]:
            if not folder.exists():
                continue
            for path in folder.glob("gan_epoch_*.pt"):
                try:
                    epoch = int(path.stem.split("_")[-1])
                except ValueError:
                    continue
                checkpoints.append((epoch, path))
        if not checkpoints:
            old_latest = OLD_GAN_CHECKPOINT_DIR / "latest_checkpoint.pt"
            if old_latest.exists():
                return old_latest
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
            self.load_generator_only_checkpoint(checkpoint, checkpoint_path, missing_keys)
            return

        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator1.load_state_dict(checkpoint["discriminator1"])
        self.discriminator2.load_state_dict(checkpoint["discriminator2"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d1.load_state_dict(checkpoint["optimizer_d1"])
        self.optimizer_d2.load_state_dict(checkpoint["optimizer_d2"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
        self.best_fmi = float(checkpoint.get("best_fmi", self.best_fmi))
        self.best_psnr = float(checkpoint.get("best_psnr", self.best_psnr))
        self.no_improve_epochs = int(checkpoint.get("no_improve_epochs", self.no_improve_epochs))
        self._set_optimizer_lr(self.optimizer_g, self.lr_g)
        self._set_optimizer_lr(self.optimizer_d1, self.lr_d)
        self._set_optimizer_lr(self.optimizer_d2, self.lr_d)

        saved_epoch = int(checkpoint["epoch"])
        self.start_epoch = saved_epoch + 1
        monitor_write(f"Loaded old checkpoint from {checkpoint_path} and resumed from epoch {self.start_epoch}")

    def load_generator_only_checkpoint(self, checkpoint, checkpoint_path, missing_keys):
        if not isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            for key in ("generator", "generator_state_dict", "model_state_dict", "state_dict"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            else:
                state_dict = checkpoint

        try:
            self.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise KeyError(
                f"Checkpoint is missing full training state keys {missing_keys} and cannot be loaded as generator weights: {error}"
            ) from error

        self.start_epoch = 51
        monitor_write(f"Loaded old checkpoint from {checkpoint_path} and resumed from epoch {self.start_epoch}")
        monitor_write("Loaded generator weights only; optimizer restarted.")

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
        if not ok and self.debug and not self.nan_reported:
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
        if self.debug:
            monitor_write(f"[Clamp/Sanitize] {name} repaired before discriminator/loss at epoch {epoch}, batch {batch_index}")
        return repaired, False

    def print_first_batch_stats(self, ct, mri, pre_activation, post_activation, fused, g_losses, d1_loss, d2_loss):
        if self.printed_first_batch_stats or not self.debug:
            return
        monitor_write("[First batch tensor stats]")
        monitor_write(self.tensor_stats("ct", ct.detach()))
        monitor_write(self.tensor_stats("mri", mri.detach()))
        monitor_write(self.tensor_stats("generator_pre_final_activation", pre_activation.detach()))
        monitor_write(self.tensor_stats("generator_post_final_activation", post_activation.detach()))
        monitor_write(self.tensor_stats("fused", fused.detach()))
        monitor_write(self.tensor_stats("generator_total_loss", g_losses["total"].detach()))
        monitor_write(self.tensor_stats("fusion_loss", g_losses["fusion"].detach()))
        monitor_write(self.tensor_stats("intensity_loss", g_losses["intensity"].detach()))
        monitor_write(self.tensor_stats("gradient_loss", g_losses["gradient"].detach()))
        monitor_write(self.tensor_stats("ssim_loss", g_losses["ssim"].detach()))
        monitor_write(self.tensor_stats("texture_loss", g_losses["texture"].detach()))
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

    @staticmethod
    def has_any_gradient(optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
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
            "mi": _avg_source_metric(metrics, "MI_MRI", "MI_CT"),
            "ag": metrics.get("AG", 0.0),
            "fmi": metrics.get("FMI", 0.0),
            "en": metrics.get("EN", 0.0),
            "cc": metrics.get("CC", 0.0),
            "epi": metrics.get("EPI", 0.0),
            "noise": metrics.get("NOISE", 0.0),
            "ms": _avg_source_metric(metrics, "MS_MRI", "MS_CT"),
        }

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()

        totals = {
            "total_loss": 0.0,
            "fusion_loss": 0.0,
            "intensity_loss": 0.0,
            "gradient_loss": 0.0,
            "ssim_loss": 0.0,
            "texture_loss": 0.0,
            "gan_loss": 0.0,
            "d1_loss": 0.0,
            "d2_loss": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "sf": 0.0,
            "mi": 0.0,
            "ag": 0.0,
            "fmi": 0.0,
            "en": 0.0,
            "cc": 0.0,
            "epi": 0.0,
            "noise": 0.0,
            "ms": 0.0,
        }
        running_loss = 0.0
        processed_batches = 0
        skipped_updates = 0
        skipped_batches = 0
        skipped_d_steps = 0
        skipped_g_steps = 0
        self.zero_all_gradients()
        skip_window = False
        d1_had_backward = False
        d2_had_backward = False
        g_had_backward = False

        bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            total=len(self.train_loader),
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=False,
            bar_format=TQDM_BAR_FORMAT,
            mininterval=0.5,
        )
        for batch_index, batch in enumerate(bar, start=1):
            ct = batch["ct"].to(self.device)
            mri = batch["mri"].to(self.device)

            if not self.is_finite(ct, "input ct", epoch, batch_index) or not self.is_finite(mri, "input mri", epoch, batch_index):
                skip_window = True
                skipped_batches += 1
                continue

            with autocast("cuda", enabled=self.amp_enabled):
                fused_for_d, pre_d, post_d = self.generator.forward_with_debug(ct.float(), mri.float())
            self.is_finite(pre_d, "generator output before final activation for discriminator", epoch, batch_index)
            self.is_finite(post_d, "generator output after final activation for discriminator", epoch, batch_index)
            fused_for_d, fused_d_ok = self.safe_fused(fused_for_d, "fused output for discriminator", epoch, batch_index)
            if not fused_d_ok:
                skip_window = True
                skipped_batches += 1
                continue
            should_update_d = batch_index == 1 or batch_index % self.discriminator_update_interval == 0
            if should_update_d:
                with autocast("cuda", enabled=self.amp_enabled):
                    d1_loss, d2_loss = self._discriminator_losses(fused_for_d.float(), ct.float(), mri.float())
                    d_loss = (d1_loss + d2_loss) / self.accumulation_steps
                if not self.is_finite(d1_loss, "discriminator1 loss", epoch, batch_index) or not self.is_finite(d2_loss, "discriminator2 loss", epoch, batch_index):
                    skip_window = True
                    skipped_batches += 1
                    continue
                if not self.is_finite(d_loss, "discriminator scaled loss", epoch, batch_index):
                    skip_window = True
                    skipped_batches += 1
                    continue
                self.scaler.scale(d_loss).backward()
                d1_had_backward = True
                d2_had_backward = True
            else:
                d1_loss = fused_for_d.new_tensor(0.0)
                d2_loss = fused_for_d.new_tensor(0.0)

            self._set_requires_grad(self.discriminator1, False)
            self._set_requires_grad(self.discriminator2, False)
            try:
                with autocast("cuda", enabled=self.amp_enabled):
                    fused_for_g, pre_g, post_g = self.generator.forward_with_debug(ct.float(), mri.float())
                self.is_finite(pre_g, "generator output before final activation for generator", epoch, batch_index)
                self.is_finite(post_g, "generator output after final activation for generator", epoch, batch_index)
                fused_for_g, fused_g_ok = self.safe_fused(fused_for_g, "fused output for generator", epoch, batch_index)
                if not fused_g_ok:
                    skip_window = True
                    skipped_batches += 1
                    continue
                with autocast("cuda", enabled=self.amp_enabled):
                    g_losses = self._generator_loss_parts(fused_for_g.float(), ct.float(), mri.float())
                    g_loss = g_losses["total"] / self.accumulation_steps
                if (
                    not self.is_finite(g_losses["total"], "generator total loss", epoch, batch_index)
                    or not self.is_finite(g_losses["fusion"], "fusion loss", epoch, batch_index)
                    or not self.is_finite(g_losses["intensity"], "intensity loss", epoch, batch_index)
                    or not self.is_finite(g_losses["gradient"], "gradient loss", epoch, batch_index)
                    or not self.is_finite(g_losses["ssim"], "ssim loss", epoch, batch_index)
                    or not self.is_finite(g_losses["texture"], "texture loss", epoch, batch_index)
                    or not self.is_finite(g_losses["gan"], "gan loss", epoch, batch_index)
                ):
                    skip_window = True
                    skipped_batches += 1
                    self.zero_all_gradients()
                    d1_had_backward = False
                    d2_had_backward = False
                    g_had_backward = False
                    continue
                if not self.is_finite(g_loss, "generator scaled loss", epoch, batch_index):
                    skip_window = True
                    skipped_batches += 1
                    self.zero_all_gradients()
                    d1_had_backward = False
                    d2_had_backward = False
                    g_had_backward = False
                    continue
                self.scaler.scale(g_loss).backward()
                g_had_backward = True
            finally:
                self._set_requires_grad(self.discriminator1, True)
                self._set_requires_grad(self.discriminator2, True)

            self.print_first_batch_stats(ct, mri, pre_g, post_g, fused_for_g, g_losses, d1_loss, d2_loss)

            should_step = batch_index % self.accumulation_steps == 0 or batch_index == len(self.train_loader)
            if should_step:
                d1_has_grads = d1_had_backward and self.has_any_gradient(self.optimizer_d1)
                d2_has_grads = d2_had_backward and self.has_any_gradient(self.optimizer_d2)
                g_has_grads = g_had_backward and self.has_any_gradient(self.optimizer_g)

                if d1_has_grads:
                    self.scaler.unscale_(self.optimizer_d1)
                    torch.nn.utils.clip_grad_norm_(self.discriminator1.parameters(), self.max_grad_norm)
                if d2_has_grads:
                    self.scaler.unscale_(self.optimizer_d2)
                    torch.nn.utils.clip_grad_norm_(self.discriminator2.parameters(), self.max_grad_norm)
                if g_has_grads:
                    self.scaler.unscale_(self.optimizer_g)
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)

                bad_grads = (
                    (d1_has_grads and self.has_bad_gradients(self.discriminator1, "discriminator1"))
                    or (d2_has_grads and self.has_bad_gradients(self.discriminator2, "discriminator2"))
                    or (g_has_grads and self.has_bad_gradients(self.generator, "generator"))
                )
                if skip_window or bad_grads:
                    skipped_updates += 1
                    if not d1_has_grads:
                        skipped_d_steps += 1
                    if not d2_has_grads:
                        skipped_d_steps += 1
                    if not g_has_grads:
                        skipped_g_steps += 1
                    if self.debug:
                        monitor_write(f"[Skip update] epoch {epoch}, batch {batch_index}: non-finite value in accumulation window.")
                    self.zero_all_gradients()
                    if d1_has_grads or d2_has_grads or g_has_grads:
                        self.scaler.update()
                    skip_window = False
                else:
                    if d1_has_grads:
                        self.scaler.step(self.optimizer_d1)
                    else:
                        skipped_d_steps += 1
                    if d2_has_grads:
                        self.scaler.step(self.optimizer_d2)
                    else:
                        skipped_d_steps += 1
                    if g_has_grads:
                        self.scaler.step(self.optimizer_g)
                    else:
                        skipped_g_steps += 1
                    if d1_has_grads or d2_has_grads or g_has_grads:
                        self.scaler.update()
                    self.zero_all_gradients()
                d1_had_backward = False
                d2_had_backward = False
                g_had_backward = False

            metric_values = self._metrics_from_fused(fused_for_g.detach(), mri, ct)

            loss_value = g_losses["total"].item()
            processed_batches += 1
            running_loss += loss_value
            totals["total_loss"] += loss_value
            totals["fusion_loss"] += g_losses["fusion"].item()
            totals["intensity_loss"] += g_losses["intensity"].item()
            totals["gradient_loss"] += g_losses["gradient"].item()
            totals["ssim_loss"] += g_losses["ssim"].item()
            totals["texture_loss"] += g_losses["texture"].item()
            totals["gan_loss"] += g_losses["gan"].item()
            totals["d1_loss"] += d1_loss.item()
            totals["d2_loss"] += d2_loss.item()
            for key, value in metric_values.items():
                totals[key] += value

            bar.set_postfix(
                {
                    "loss": f"{running_loss / max(1, processed_batches):.4f}",
                    "lr": f"{self.optimizer_g.param_groups[0]['lr']:.2e}",
                    "accum": f"{batch_index % self.accumulation_steps}/{self.accumulation_steps}",
                }
            )

        count = max(1, processed_batches)
        values = {key: value / count for key, value in totals.items()}
        values["skipped_updates"] = skipped_updates
        values["skipped_batches"] = skipped_batches
        values["skipped_d_steps"] = skipped_d_steps
        values["skipped_g_steps"] = skipped_g_steps
        return values

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
            "intensity_loss": 0.0,
            "gradient_loss": 0.0,
            "ssim_loss": 0.0,
            "texture_loss": 0.0,
            "gan_loss": 0.0,
            "d1_loss": 0.0,
            "d2_loss": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "sf": 0.0,
            "mi": 0.0,
            "ag": 0.0,
            "fmi": 0.0,
            "en": 0.0,
            "cc": 0.0,
            "epi": 0.0,
            "noise": 0.0,
            "ms": 0.0,
        }

        bar = tqdm(
            self.val_loader,
            desc=f"Validate {epoch}/{self.epochs}",
            leave=False,
            total=len(self.val_loader),
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=False,
            bar_format=TQDM_BAR_FORMAT,
            mininterval=0.5,
        )
        for index, batch in enumerate(bar):
            ct = batch["ct"].to(self.device)
            mri = batch["mri"].to(self.device)
            if not self.is_finite(ct, "validation input ct", epoch, index + 1) or not self.is_finite(mri, "validation input mri", epoch, index + 1):
                continue
            with autocast("cuda", enabled=self.amp_enabled):
                fused, pre_val, post_val = self.generator.forward_with_debug(ct.float(), mri.float())
            self.is_finite(pre_val, "validation generator output before final activation", epoch, index + 1)
            self.is_finite(post_val, "validation generator output after final activation", epoch, index + 1)
            fused, fused_ok = self.safe_fused(fused, "validation fused output", epoch, index + 1)
            if not fused_ok:
                continue
            with autocast("cuda", enabled=self.amp_enabled):
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
            totals["intensity_loss"] += g_losses["intensity"].item()
            totals["gradient_loss"] += g_losses["gradient"].item()
            totals["ssim_loss"] += g_losses["ssim"].item()
            totals["texture_loss"] += g_losses["texture"].item()
            totals["gan_loss"] += g_losses["gan"].item()
            totals["d1_loss"] += d1_loss.item()
            totals["d2_loss"] += d2_loss.item()
            for key, value in metric_values.items():
                totals[key] += value

            if index < 4:
                original_path = self.original_sample_dir / f"epoch_{epoch:03d}_sample_{index + 1:02d}.png"
                comparison_path = self.comparison_sample_dir / f"epoch_{epoch:03d}_sample_{index + 1:02d}_{self.pair}_comparison.png"
                _save_tensor_image(fused[0], original_path)
                _save_validation_comparison(ct[0], mri[0], fused[0], comparison_path, pair_labels(self.pair))

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
            "best_fmi": self.best_fmi,
            "best_psnr": self.best_psnr,
            "best_val_loss": self.best_val_loss,
            "no_improve_epochs": self.no_improve_epochs,
        }
        torch.save(payload, path)
        torch.save(payload, self.checkpoint_dir / "latest_checkpoint.pt")
        torch.save(self.generator.state_dict(), self.checkpoint_dir / "latest_generator.pt")
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
                "best_fmi": self.best_fmi,
                "best_psnr": self.best_psnr,
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
            val_fmi = float(row.get("val_fmi") or float("-inf"))
            val_psnr = float(row.get("val_psnr") or float("-inf"))
            val_loss = float(row.get("val_total_loss") or float("inf"))
            if self.is_better_epoch(val_ssim, val_fmi, val_psnr, val_loss):
                self.best_epoch = int(row["epoch"])
                self.best_val_ssim = val_ssim
                self.best_fmi = val_fmi
                self.best_psnr = val_psnr
                self.best_val_loss = val_loss
                self.best_metric = val_ssim

    def is_better_epoch(self, val_ssim, val_fmi, val_psnr, val_loss):
        candidate = (val_ssim, val_fmi, val_psnr, -val_loss)
        current = (self.best_val_ssim, self.best_fmi, self.best_psnr, -self.best_val_loss)
        return candidate > current

    def build_history_row(self, epoch, train_values, val_values, is_best):
        return {
            "epoch": epoch,
            "learning_rate": self.optimizer_g.param_groups[0]["lr"],
            "train_total_loss": train_values.get("total_loss"),
            "val_total_loss": val_values.get("total_loss"),
            "train_fusion_loss": train_values.get("fusion_loss"),
            "val_fusion_loss": val_values.get("fusion_loss"),
            "train_intensity_loss": train_values.get("intensity_loss"),
            "val_intensity_loss": val_values.get("intensity_loss"),
            "train_gradient_loss": train_values.get("gradient_loss"),
            "val_gradient_loss": val_values.get("gradient_loss"),
            "train_ssim_loss": train_values.get("ssim_loss"),
            "val_ssim_loss": val_values.get("ssim_loss"),
            "train_texture_loss": train_values.get("texture_loss"),
            "val_texture_loss": val_values.get("texture_loss"),
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
            "train_mi": train_values.get("mi"),
            "val_mi": val_values.get("mi"),
            "train_ag": train_values.get("ag"),
            "val_ag": val_values.get("ag"),
            "train_fmi": train_values.get("fmi"),
            "val_fmi": val_values.get("fmi"),
            "train_en": train_values.get("en"),
            "val_en": val_values.get("en"),
            "train_cc": train_values.get("cc"),
            "val_cc": val_values.get("cc"),
            "train_epi": train_values.get("epi"),
            "val_epi": val_values.get("epi"),
            "train_noise": train_values.get("noise"),
            "val_noise": val_values.get("noise"),
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
        overview_specs = [
            ("train_gan_loss", "Generator loss"),
            ("train_d1_loss", "Discriminator loss D1"),
            ("train_d2_loss", "Discriminator loss D2"),
            ("train_total_loss", "Train loss"),
            ("val_total_loss", "Validation loss"),
            ("val_ssim", "SSIM"),
            ("val_psnr", "PSNR"),
            ("val_mi", "MI"),
            ("val_ag", "AG"),
            ("val_sf", "SF"),
            ("learning_rate", "Learning rate"),
        ]
        epochs = [int(row["epoch"]) for row in self.history]
        fig, axes = plt.subplots(4, 3, figsize=(15, 12), dpi=160)
        for ax, (key, title) in zip(axes.flatten(), overview_specs):
            points = [(int(row["epoch"]), row.get(key)) for row in self.history if row.get(key) is not None]
            if points:
                ax.plot([point[0] for point in points], [float(point[1]) for point in points], linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
        axes.flatten()[-1].axis("off")
        fig.suptitle(f"{self.pair} training curves")
        fig.tight_layout()
        fig.savefig(self.graph_dir / f"{self.pair}_training_curves.png")
        plt.close(fig)
        graph_specs = [
            ("train_total_loss", "val_total_loss", "train_loss_vs_val_loss.png", "Train Loss vs Validation Loss", "Loss"),
            ("train_ssim", "val_ssim", "train_ssim_vs_val_ssim.png", "Train SSIM vs Validation SSIM", "SSIM"),
            ("train_psnr", "val_psnr", "train_psnr_vs_val_psnr.png", "Train PSNR vs Validation PSNR", "PSNR"),
            ("train_sf", "val_sf", "train_sf_vs_val_sf.png", "Train SF vs Validation SF", "Spatial Frequency"),
            ("train_mi", "val_mi", "train_mi_vs_val_mi.png", "Train MI vs Validation MI", "Mutual Information"),
            ("train_ag", "val_ag", "train_ag_vs_val_ag.png", "Train AG vs Validation AG", "Average Gradient"),
            ("train_epi", "val_epi", "train_epi_vs_val_epi.png", "Train EPI vs Validation EPI", "Edge Preservation"),
            ("train_noise", "val_noise", "train_noise_vs_val_noise.png", "Train Noise vs Validation Noise", "Noise Indicator"),
            ("train_ms", "val_ms", "train_ms_vs_val_ms.png", "Train MS vs Validation MS", "MS"),
        ]
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

    def append_training_log(self, epoch, train_values, val_values, checkpoint_path, is_best):
        self.training_log_path.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"epoch={epoch:03d} "
            f"train_loss={train_values.get('total_loss'):.6f} "
            f"skipped_updates={int(train_values.get('skipped_updates', 0))} "
            f"skipped_batches={int(train_values.get('skipped_batches', 0))} "
            f"skipped_d_steps={int(train_values.get('skipped_d_steps', 0))} "
            f"skipped_g_steps={int(train_values.get('skipped_g_steps', 0))} "
            f"val_loss={val_values.get('total_loss') if val_values.get('total_loss') is not None else 'skipped'} "
            f"ssim={val_values.get('ssim') if val_values.get('ssim') is not None else 'skipped'} "
            f"psnr={val_values.get('psnr') if val_values.get('psnr') is not None else 'skipped'} "
            f"mi={val_values.get('mi') if val_values.get('mi') is not None else 'skipped'} "
            f"ag={val_values.get('ag') if val_values.get('ag') is not None else 'skipped'} "
            f"sf={val_values.get('sf') if val_values.get('sf') is not None else 'skipped'} "
            f"checkpoint={checkpoint_path} "
            f"is_best={is_best}"
        )
        with self.training_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    def fit(self):
        monitor_write(f"Training on device: {self.device}")
        monitor_write(f"Checkpoint path: {self.checkpoint_dir}")
        monitor_write(f"Fused image folder: {self.original_sample_dir}")
        monitor_write(f"Graph folder: {self.graph_dir}")
        monitor_write(f"Metrics folder: {gan_metrics_dir()}")
        monitor_write(f"Training log: {self.training_log_path}")

        if self.start_epoch > self.epochs:
            monitor_write(
                f"Checkpoint is already at epoch {self.start_epoch - 1}. "
                f"Requested total epochs: {self.epochs}. Nothing to train."
            )
            return False

        completed_epoch = None
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_values = self.train_epoch(epoch)
            should_validate = epoch % self.val_every == 0 or epoch == self.epochs
            val_values = self.validate(epoch) if should_validate else {}
            val_ssim = float(val_values.get("ssim", float("-inf"))) if should_validate else None
            val_fmi = float(val_values.get("fmi", float("-inf"))) if should_validate else None
            val_psnr = float(val_values.get("psnr", float("-inf"))) if should_validate else None
            val_loss = float(val_values.get("total_loss", float("inf"))) if should_validate else None

            is_best = should_validate and self.is_better_epoch(val_ssim, val_fmi, val_psnr, val_loss)
            if is_best:
                self.best_epoch = epoch
                self.best_val_ssim = val_ssim
                self.best_fmi = val_fmi
                self.best_psnr = val_psnr
                self.best_val_loss = val_loss
                self.best_metric = val_ssim
                self.no_improve_epochs = 0
                self.save_best_checkpoint(epoch)
            else:
                if should_validate:
                    self.no_improve_epochs += 1

            checkpoint_path = self.save_checkpoint(epoch)
            row = self.build_history_row(epoch, train_values, val_values, is_best)
            self.append_history(row)
            self.plot_history()
            self.update_reports(row)
            completed_epoch = epoch

            val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "skipped"
            val_ssim_text = f"{val_ssim:.4f}" if val_ssim is not None else "skipped"
            monitor_write(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"train_loss={train_values['total_loss']:.4f} | "
                f"skipped_updates={int(train_values.get('skipped_updates', 0))} | "
                f"skipped_batches={int(train_values.get('skipped_batches', 0))} | "
                f"skipped_d_steps={int(train_values.get('skipped_d_steps', 0))} | "
                f"skipped_g_steps={int(train_values.get('skipped_g_steps', 0))} | "
                f"val_loss={val_loss_text} | val_ssim={val_ssim_text} | "
                f"best_epoch={self.best_epoch} | checkpoint=saved"
            )
            self.append_training_log(epoch, train_values, val_values, checkpoint_path, is_best)
            if should_validate and self.patience > 0 and self.no_improve_epochs >= self.patience:
                monitor_write(f"Early stopping: no validation SSIM/loss improvement for {self.no_improve_epochs} epochs.")
                return False

        return completed_epoch == self.epochs
