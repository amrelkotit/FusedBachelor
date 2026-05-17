import argparse
import os
import subprocess
import sys
import warnings


def force_utf8_console():
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

from src.data.paired_dataset import AANLIB_ROOT, aanlib_split_root, diffusion_checkpoint_dir, diffusion_graph_dir, diffusion_image_dir, diffusion_logs_dir, diffusion_metrics_dir, normalize_pair, verify_dataset_root
from src.diffusion.trainer import DiffusionFusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train an independent diffusion model for AANLIB medical image fusion.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT))
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], default="ct_mri")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-root", default="outputs/models/diffusion")
    parser.add_argument("--use-msfd-guidance", action="store_true")
    parser.add_argument("--lambda-msfd", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-steps", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-noise", type=float, default=1.2)
    parser.add_argument("--lambda-l1", type=float, default=1.2)
    parser.add_argument("--lambda-ssim", type=float, default=2.5)
    parser.add_argument("--lambda-grad", type=float, default=1.6)
    parser.add_argument("--lambda-hf", type=float, default=1.0)
    parser.add_argument("--lambda-ms-ssim", type=float, default=0.8)
    parser.add_argument("--lambda-local-contrast", type=float, default=0.4)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--shutdown-on-complete", action="store_true", help="Shutdown Windows after diffusion training completes successfully.")
    return parser.parse_args()


def main():
    force_utf8_console()
    args = parse_args()
    pair = normalize_pair(args.pair)
    train_root = aanlib_split_root(args.dataset_root, pair, "train")
    test_root = aanlib_split_root(args.dataset_root, pair, "test")
    print("[Model] Selected model: Diffusion")
    print(f"[Output] Root: {args.output_root}")
    print(f"[Pair] {pair}")
    verify_dataset_root(train_root, dataset_name=f"AANLIB {pair} train", strict=True, pair=pair)
    verify_dataset_root(test_root, dataset_name=f"AANLIB {pair} test", strict=False, pair=pair)
    print(f"Checkpoint path: {diffusion_checkpoint_dir(pair, output_root=args.output_root)}")
    print(f"Fused image folder: {diffusion_image_dir('aanlib', pair, 'train', output_root=args.output_root)}")
    print(f"Graph folder: {diffusion_graph_dir(args.output_root)}")
    print(f"Metrics folder: {diffusion_metrics_dir(args.output_root)}")
    print(f"Training log: {diffusion_logs_dir(args.output_root) / (pair + '_diffusion_training.log')}")

    trainer = DiffusionFusionTrainer(
        dataset_root=args.dataset_root,
        pair=pair,
        output_root=args.output_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        epochs=args.epochs,
        lr=args.lr,
        timesteps=args.timesteps,
        sampling_steps=args.sampling_steps,
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_grad=args.lambda_grad,
        lambda_hf=args.lambda_hf,
        lambda_noise=args.lambda_noise,
        lambda_ms_ssim=args.lambda_ms_ssim,
        lambda_local_contrast=args.lambda_local_contrast,
        lambda_msfd=args.lambda_msfd,
        use_msfd_guidance=args.use_msfd_guidance,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        val_split=args.val_split,
        val_every=args.val_every,
        num_workers=args.num_workers,
        max_items=args.max_items,
        device=args.device,
        allow_cpu=args.allow_cpu,
        resume=args.resume,
        auto_resume=args.auto_resume,
    )
    training_completed = False
    try:
        training_completed = trainer.fit()
    finally:
        if args.shutdown_on_complete and training_completed:
            sys.stdout.flush()
            sys.stderr.flush()
            print("Diffusion training completed successfully. Laptop will shutdown in 60 seconds.")
            print("To cancel shutdown, run: shutdown /a")
            sys.stdout.flush()
            sys.stderr.flush()
            subprocess.run(["shutdown", "/s", "/t", "60"], check=True)


if __name__ == "__main__":
    main()
