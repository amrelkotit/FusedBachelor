import argparse
import os
import subprocess
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from src.data.paired_dataset import DEFAULT_TEST_ROOTS, DEFAULT_TRAIN_ROOTS, verify_dataset_root
from src.training.trainer import GANFusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN-based CT/MRI medical image fusion.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Backward-compatible single dataset root. Prefer --train-roots for the final dataset layout.",
    )
    parser.add_argument(
        "--train-roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_TRAIN_ROOTS],
        help="Training split roots. Defaults to AANLIB/train and BRATS_SPLIT/train.",
    )
    parser.add_argument("--output-dir", default="outputs/models/gan_continued_from_50", help="Folder for GAN checkpoints, logs, graphs, reports, and samples.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=204)
    parser.add_argument("--micro-batch", type=int, default=4, help="Mini-batch loaded on the GPU before gradient accumulation.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="auto", help="Training device: auto, cuda, cuda:0, or cpu. Auto uses cuda:0 when available.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow training on CPU if CUDA is not available.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr-factor", type=float, default=0.25, help="D learning rate = lr * factor; lower values reduce discriminator dominance.")
    parser.add_argument("--discriminator-update-interval", type=int, default=2, help="Update D every N batches to reduce noisy adversarial pressure.")
    parser.add_argument("--lambda-intensity", type=float, default=1.0)
    parser.add_argument("--lambda-gradient", type=float, default=5.0)
    parser.add_argument("--lambda-ssim", type=float, default=2.0)
    parser.add_argument("--lambda-texture", type=float, default=3.0)
    parser.add_argument("--lambda-gan", type=float, default=0.1)
    parser.add_argument("--lambda-grad", type=float, default=None, help="Deprecated alias for --lambda-gradient.")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs, plus always on the final epoch.")
    parser.add_argument("--patience", type=int, default=8, help="Stop after this many epochs without combined validation improvement. Use 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum combined-score improvement required to reset patience.")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless multiprocessing is configured.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional small subset for quick debugging.")
    parser.add_argument("--resume", default=None, help="Path to a full GAN checkpoint, for example outputs/models/gan/checkpoints/gan_epoch_012.pt.")
    parser.add_argument("--auto-resume", action="store_true", help="Automatically resume from the latest gan_epoch_XXX.pt checkpoint.")
    parser.add_argument("--shutdown-on-complete", action="store_true", help="Shutdown Windows 60 seconds after all requested epochs finish successfully.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_roots = [args.dataset_root] if args.dataset_root else args.train_roots
    print("[Dataset] Final expected train roots:")
    for root in train_roots:
        verify_dataset_root(root, dataset_name=str(root), strict=True)
    print("[Dataset] Final expected test roots:")
    for root in DEFAULT_TEST_ROOTS:
        verify_dataset_root(root, dataset_name=str(root), strict=False)

    trainer = GANFusionTrainer(
        dataset_root=args.dataset_root,
        train_roots=train_roots,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        epochs=args.epochs,
        device=args.device,
        allow_cpu=args.allow_cpu,
        lr=args.lr,
        discriminator_lr_factor=args.discriminator_lr_factor,
        discriminator_update_interval=args.discriminator_update_interval,
        lambda_intensity=args.lambda_intensity,
        lambda_gradient=args.lambda_gradient if args.lambda_grad is None else args.lambda_grad,
        lambda_ssim=args.lambda_ssim,
        lambda_texture=args.lambda_texture,
        lambda_gan=args.lambda_gan,
        val_split=args.val_split,
        val_every=args.val_every,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        max_items=args.max_items,
        resume=args.resume,
        auto_resume=args.auto_resume,
    )
    training_completed = False
    try:
        training_completed = trainer.fit()
    except BaseException:
        training_completed = False
        raise
    finally:
        if args.shutdown_on_complete and training_completed:
            sys.stdout.flush()
            sys.stderr.flush()
            print("Training completed successfully. Laptop will shutdown in 60 seconds.")
            print("To cancel shutdown, run: shutdown /a")
            sys.stdout.flush()
            sys.stderr.flush()
            subprocess.run(["shutdown", "/s", "/t", "60"], check=True)


if __name__ == "__main__":
    main()
