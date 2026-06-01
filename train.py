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

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

from src.data.paired_dataset import (
    AANLIB_ROOT,
    aanlib_split_root,
    gan_checkpoint_dir,
    gan_graph_dir,
    gan_image_dir,
    gan_logs_dir,
    gan_metrics_dir,
    normalize_pair,
    verify_dataset_root,
)
from src.training.trainer import GANFusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN fusion on Harvard AANLIB modality pairs.")
    parser.add_argument("--dataset-root", default=str(AANLIB_ROOT), help="AANLIB root containing CT-MRI, PET-MRI, and SPECT-MRI.")
    parser.add_argument("--pair", choices=["ct_mri", "pet_mri", "spect_mri"], default="ct_mri")
    parser.add_argument("--output-dir", default=None, help="Optional experiment folder. Default: outputs/models/gan/aanlib_<pair>.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Effective batch size. Keep ≤ dataset size so accumulation steps "
                             "< number of batches per epoch (avoids 1-step-per-epoch bug). "
                             "Default: 8 (2 accumulation steps of micro-batch 4).")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs. Increased from 50 to 150 to allow "
                             "the model to converge properly with the fixed batch size.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr-factor", type=float, default=0.25)
    parser.add_argument("--discriminator-update-interval", type=int, default=2)
    parser.add_argument("--lambda-intensity", type=float, default=1.0)
    parser.add_argument("--lambda-gradient", type=float, default=5.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.0)  # disabled – not used in MedicalFusionGANLoss.forward()
    parser.add_argument("--lambda-texture", type=float, default=3.0)
    parser.add_argument("--lambda-gan", type=float, default=0.1)
    parser.add_argument("--use-msfd-guidance", dest="use_msfd_guidance", action="store_true", default=True, help="Use MSFD guidance L1 loss during generator training. Enabled by default.")
    parser.add_argument("--no-msfd-guidance", dest="use_msfd_guidance", action="store_false", help="Disable MSFD guidance and use the old normal GAN training objective.")
    parser.add_argument("--lambda-msfd", type=float, default=0.25, help="Weight for MSFD guidance L1 loss.")
    parser.add_argument("--lambda-grad", type=float, default=None, help="Deprecated alias for --lambda-gradient.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Deterministic internal validation split when AANLIB/<pair>/val is absent.")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stopping patience in epochs. 0 (default) disables early stopping so training "
                             "always runs for the full --epochs count. Set to a positive value (e.g. 35) to stop "
                             "early when validation SSIM/loss stops improving.")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lambda-source1-intensity", type=float, default=None)
    parser.add_argument("--lambda-source1-gradient", type=float, default=None)
    parser.add_argument("--lambda-source1-ssim", type=float, default=None)
    parser.add_argument("--lambda-source1-texture", type=float, default=None)
    parser.add_argument("--lambda-source1-hot", type=float, default=None)
    parser.add_argument("--lambda-mri-intensity", type=float, default=None)
    parser.add_argument("--lambda-mri-ssim", type=float, default=None)
    parser.add_argument("--lambda-mri-texture", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Show detailed non-finite diagnostics during training.")
    parser.add_argument("--verbose", action="store_true", help="Alias for --debug.")
    parser.add_argument("--shutdown-on-complete", action="store_true")
    return parser.parse_args()


def main():
    force_utf8_console()
    args = parse_args()
    pair = normalize_pair(args.pair)
    train_root = aanlib_split_root(args.dataset_root, pair, "train")
    test_root = aanlib_split_root(args.dataset_root, pair, "test")

    print("[Startup] AANLIB-only primary training")
    verify_dataset_root(train_root, dataset_name=f"AANLIB {pair} train", strict=True, pair=pair)
    verify_dataset_root(test_root, dataset_name=f"AANLIB {pair} test", strict=False, pair=pair)
    print(f"Checkpoint path: {gan_checkpoint_dir(pair)}")
    print(f"Fused image folder: {gan_image_dir('aanlib', 'train', pair) / 'fused_original'}")
    print(f"Graph folder: {gan_graph_dir()}")
    print(f"Metrics folder: {gan_metrics_dir()}")
    print(f"Training log: {gan_logs_dir() / (pair + '_training.log')}")

    trainer = GANFusionTrainer(
        dataset_root=args.dataset_root,
        pair=pair,
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
        use_msfd_guidance=args.use_msfd_guidance,
        lambda_msfd=args.lambda_msfd,
        lambda_source1_intensity=args.lambda_source1_intensity,
        lambda_source1_gradient=args.lambda_source1_gradient,
        lambda_source1_ssim=args.lambda_source1_ssim,
        lambda_source1_texture=args.lambda_source1_texture,
        lambda_source1_hot=args.lambda_source1_hot,
        lambda_mri_intensity=args.lambda_mri_intensity,
        lambda_mri_ssim=args.lambda_mri_ssim,
        lambda_mri_texture=args.lambda_mri_texture,
        val_split=args.val_split,
        val_every=args.val_every,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        max_items=args.max_items,
        resume=args.resume,
        auto_resume=args.auto_resume,
        debug=args.debug or args.verbose,
    )
    training_completed = False
    try:
        training_completed = trainer.fit()
    except BaseException as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
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
