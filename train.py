import argparse

from src.training.trainer import GANFusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN-based CT/MRI medical image fusion.")
    parser.add_argument(
        "--dataset-root",
        default=r"E:\El Gam3a\My bachelor\Fused bachelor\data\raw\final_dataset\AANLIB\train",
        help="Folder containing CT and MRI subfolders.",
    )
    parser.add_argument("--output-dir", default="outputs/gan", help="Folder for checkpoints and validation samples.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-gan", type=float, default=0.01)
    parser.add_argument("--lambda-grad", type=float, default=5.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless multiprocessing is configured.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional small subset for quick debugging.")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = GANFusionTrainer(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_gan=args.lambda_gan,
        lambda_grad=args.lambda_grad,
        val_split=args.val_split,
        num_workers=args.num_workers,
        max_items=args.max_items,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
