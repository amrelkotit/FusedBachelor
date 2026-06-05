"""Generate thesis-style loss curves for GAN and Diffusion models."""
import json
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

OUTPUT_DIR = r"e:\El Gam3a\My bachelor\Finale-Fused\FusedBachelor\output\final_thesis_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = r"e:\El Gam3a\My bachelor\Finale-Fused\FusedBachelor\outputs\models"

PAIRS = ["ct_mri", "pet_mri", "spect_mri"]
PAIR_LABELS = {"ct_mri": "CT-MRI", "pet_mri": "PET-MRI", "spect_mri": "SPECT-MRI"}


def plot_loss(epochs, losses, title_suffix, filename, caption):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(epochs, losses, color="red", linewidth=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9,
             color="#007bff", style="italic")
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ── GAN ──────────────────────────────────────────────────────────────────────
for i, pair in enumerate(PAIRS, start=1):
    json_path = os.path.join(BASE, "gan", "logs", f"{pair}_training_history.json")
    with open(json_path) as f:
        history = json.load(f)
    epochs = [e["epoch"] for e in history]
    losses = [e["train_total_loss"] for e in history]
    n_epochs = epochs[-1]
    label = PAIR_LABELS[pair]
    caption = f"GAN Loss Curve – {label} ({n_epochs} epochs)"
    plot_loss(
        epochs, losses,
        title_suffix=label,
        filename=f"gan_{pair}_loss_curve.png",
        caption=caption,
    )

# ── Diffusion ─────────────────────────────────────────────────────────────────
for i, pair in enumerate(PAIRS, start=1):
    csv_path = os.path.join(BASE, "diffusion", "logs", f"{pair}_training_history.csv")
    df = pd.read_csv(csv_path)
    epochs = df["epoch"].tolist()
    losses = df["train_loss"].tolist()
    n_epochs = epochs[-1]
    label = PAIR_LABELS[pair]
    caption = f"Diffusion Loss Curve – {label} ({n_epochs} epochs)"
    plot_loss(
        epochs, losses,
        title_suffix=label,
        filename=f"diffusion_{pair}_loss_curve.png",
        caption=caption,
    )

print("\nAll 6 loss curves saved to:", OUTPUT_DIR)
