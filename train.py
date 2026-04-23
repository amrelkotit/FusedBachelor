import argparse
import os

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
    parser.add_argument("--output-dir", default="outputs", help="Folder for checkpoints, history, graphs, reports, and samples.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=204)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr-factor", type=float, default=0.5, help="D learning rate = lr * factor; lower values reduce discriminator dominance.")
    parser.add_argument("--lambda-gan", type=float, default=0.005)
    parser.add_argument("--lambda-grad", type=float, default=7.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8, help="Stop after this many epochs without combined validation improvement. Use 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum combined-score improvement required to reset patience.")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless multiprocessing is configured.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional small subset for quick debugging.")
    parser.add_argument("--resume", default=None, help="Path to a full GAN checkpoint, for example outputs/gan/checkpoints/gan_epoch_012.pt.")
    parser.add_argument("--auto-resume", action="store_true", help="Automatically resume from the latest gan_epoch_XXX.pt checkpoint.")
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
        epochs=args.epochs,
        lr=args.lr,
        discriminator_lr_factor=args.discriminator_lr_factor,
        lambda_gan=args.lambda_gan,
        lambda_grad=args.lambda_grad,
        val_split=args.val_split,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        max_items=args.max_items,
        resume=args.resume,
        auto_resume=args.auto_resume,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
