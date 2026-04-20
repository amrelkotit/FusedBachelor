from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, random_split

from src.data.paired_dataset import PairedMedicalImageDataset
from src.evaluation.metrics import evaluate_fusion
from src.fusion.decomposition import multiscale_fuse
from src.models.gan import FusionGenerator, PatchDiscriminator
from src.training.losses import MedicalFusionGANLoss, gan_discriminator_loss


def _save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    cv2.imwrite(str(path), (image * 255.0).astype("uint8"))


class GANFusionTrainer:
    def __init__(
        self,
        dataset_root,
        output_dir="outputs/gan",
        image_size=256,
        batch_size=4,
        epochs=50,
        lr=2e-4,
        lambda_gan=0.01,
        lambda_grad=5.0,
        val_split=0.1,
        num_workers=0,
        device=None,
        max_items=None,
    ):
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        dataset = PairedMedicalImageDataset(dataset_root, image_size=image_size, max_items=max_items)
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
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device == "cuda",
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.generator = FusionGenerator().to(self.device)
        self.discriminator1 = PatchDiscriminator(in_channels=1).to(self.device)
        self.discriminator2 = PatchDiscriminator(in_channels=1).to(self.device)

        self.generator_loss = MedicalFusionGANLoss(lambda_gan=lambda_gan, lambda_grad=lambda_grad)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d1 = torch.optim.Adam(self.discriminator1.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d2 = torch.optim.Adam(self.discriminator2.parameters(), lr=lr, betas=(0.5, 0.999))

    def _real_style_target(self, ct, mri):
        """D2 sees source-domain realism from real CT/MRI, alternating by batch."""
        selector = torch.rand(ct.shape[0], 1, 1, 1, device=ct.device)
        return torch.where(selector > 0.5, ct, mri)

    def _train_discriminators(self, fused, ct, mri):
        real_style = self._real_style_target(ct, mri)
        # D1 learns fused-image realism from the current MSFD-like pipeline.
        real_fused_target = multiscale_fuse(mri=mri, ct=ct).detach()

        self.optimizer_d1.zero_grad(set_to_none=True)
        d1_real_logits = self.discriminator1(real_fused_target)
        d1_fake_logits = self.discriminator1(fused.detach())
        d1_loss = gan_discriminator_loss(d1_real_logits, d1_fake_logits)
        d1_loss.backward()
        self.optimizer_d1.step()

        self.optimizer_d2.zero_grad(set_to_none=True)
        # D2 learns source-domain realism directly from real CT/MRI images.
        d2_real_logits = self.discriminator2(real_style)
        d2_fake_logits = self.discriminator2(fused.detach())
        d2_loss = gan_discriminator_loss(d2_real_logits, d2_fake_logits)
        d2_loss.backward()
        self.optimizer_d2.step()

        return d1_loss.detach(), d2_loss.detach()

    def _train_generator(self, fused, ct, mri):
        self.optimizer_g.zero_grad(set_to_none=True)
        d1_fake_logits = self.discriminator1(fused)
        d2_fake_logits = self.discriminator2(fused)
        losses = self.generator_loss(fused=fused, mri=mri, ct=ct, d1_fake_logits=d1_fake_logits, d2_fake_logits=d2_fake_logits)
        losses["total"].backward()
        self.optimizer_g.step()
        return losses

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator1.train()
        self.discriminator2.train()

        totals = {"g": 0.0, "d1": 0.0, "d2": 0.0, "fusion": 0.0, "gradient": 0.0, "gan": 0.0}
        for batch_index, batch in enumerate(self.train_loader, start=1):
            ct = batch["ct"].to(self.device)
            mri = batch["mri"].to(self.device)

            fused_for_d = self.generator(ct, mri)
            d1_loss, d2_loss = self._train_discriminators(fused_for_d, ct, mri)

            fused_for_g = self.generator(ct, mri)
            g_losses = self._train_generator(fused_for_g, ct, mri)

            totals["g"] += g_losses["total"].item()
            totals["d1"] += d1_loss.item()
            totals["d2"] += d2_loss.item()
            totals["fusion"] += g_losses["fusion"].item()
            totals["gradient"] += g_losses["gradient"].item()
            totals["gan"] += g_losses["gan"].item()

            if batch_index % 20 == 0:
                print(
                    f"Epoch {epoch:03d} Batch {batch_index:04d} | "
                    f"G {g_losses['total'].item():.4f} | D1 {d1_loss.item():.4f} | D2 {d2_loss.item():.4f}"
                )

        count = max(1, len(self.train_loader))
        return {key: value / count for key, value in totals.items()}

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return {}

        self.generator.eval()
        metric_totals = {}
        for index, batch in enumerate(self.val_loader):
            ct = batch["ct"].to(self.device)
            mri = batch["mri"].to(self.device)
            fused = self.generator(ct, mri)
            metrics = evaluate_fusion(fused, mri, ct)

            for key, value in metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value

            if index < 4:
                _save_tensor_image(fused[0], self.sample_dir / f"epoch_{epoch:03d}_sample_{index + 1:02d}.png")

        return {key: value / len(self.val_loader) for key, value in metric_totals.items()}

    def save_checkpoint(self, epoch):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"gan_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator": self.generator.state_dict(),
                "discriminator1": self.discriminator1.state_dict(),
                "discriminator2": self.discriminator2.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d1": self.optimizer_d1.state_dict(),
                "optimizer_d2": self.optimizer_d2.state_dict(),
            },
            path,
        )
        torch.save(self.generator.state_dict(), self.checkpoint_dir / "generator_latest.pt")
        return path

    def fit(self):
        print(f"Training on device: {self.device}")
        for epoch in range(1, self.epochs + 1):
            train_losses = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            checkpoint_path = self.save_checkpoint(epoch)

            print(
                f"Epoch {epoch:03d} done | "
                f"G {train_losses['g']:.4f} | Fusion {train_losses['fusion']:.4f} | "
                f"Grad {train_losses['gradient']:.4f} | GAN {train_losses['gan']:.4f} | "
                f"D1 {train_losses['d1']:.4f} | D2 {train_losses['d2']:.4f}"
            )
            if val_metrics:
                metrics_text = " | ".join(f"{key} {value:.4f}" for key, value in val_metrics.items())
                print(f"Validation | {metrics_text}")
            print(f"Saved checkpoint: {checkpoint_path}")
