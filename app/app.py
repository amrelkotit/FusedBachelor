"""
Medical Image Fusion - Gradio Demo
==================================
GAN-based fusion of CT/PET/SPECT + MRI using a dual-encoder FusionGenerator
with Spatial-Frequential Fusion (SFF) attention.

Usage (from the project root):
    python app/app.py

Requirements:
    pip install gradio torch torchvision Pillow
"""

import sys
import os

# -- Make sure project root is on the Python path ------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math
import numpy as np
import cv2
import torch
import gradio as gr
from PIL import Image

from src.models.gan import FusionGenerator
from src.evaluation.metrics import (
    psnr,
    ssim,
    mutual_information,
    feature_mutual_information,
    correlation_coefficient,
    entropy,
    spatial_frequency,
    average_gradient,
)

# -- Device --------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Checkpoint paths (relative to project root) --------------------------------
CHECKPOINT_PATHS = {
    "CT-MRI":    os.path.join(ROOT, "outputs", "models", "gan", "aanlib_ct_mri",   "checkpoints", "best_generator.pt"),
    "PET-MRI":   os.path.join(ROOT, "outputs", "models", "gan", "aanlib_pet_mri",  "checkpoints", "best_generator.pt"),
    "SPECT-MRI": os.path.join(ROOT, "outputs", "models", "gan", "aanlib_spect_mri","checkpoints", "best_generator.pt"),
}

# Human-readable labels for each source image per modality pair
PAIR_LABELS = {
    "CT-MRI":    ("CT Image",    "MRI Image"),
    "PET-MRI":   ("PET Image",   "MRI Image"),
    "SPECT-MRI": ("SPECT Image", "MRI Image"),
}

# -- Load all three generators at startup ---------------------------------------
print("Loading fusion models...")
MODELS: dict[str, FusionGenerator] = {}

for pair_name, ckpt_path in CHECKPOINT_PATHS.items():
    model = FusionGenerator(base_channels=32).to(DEVICE)
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        # Support both bare state-dicts and wrapped checkpoints
        if isinstance(state, dict) and "generator_state_dict" in state:
            state = state["generator_state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"  OK  {pair_name} loaded from {ckpt_path}")
    else:
        print(f"  WARN  {pair_name} checkpoint NOT found at {ckpt_path} - using random weights")
    model.eval()
    MODELS[pair_name] = model

print("All models ready.\n")


# -- Helper utilities -----------------------------------------------------------

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image -> (1, 1, 256, 256) float tensor in [0, 1]."""
    img = img.convert("L").resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 1, H, W) float tensor in [0, 1] -> grayscale PIL image."""
    arr = tensor.squeeze().detach().cpu().float().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _pil_to_uint8(img: Image.Image) -> np.ndarray:
    """Resize PIL image to 256×256 and return grayscale uint8 array."""
    return np.array(img.convert("L").resize((256, 256), Image.LANCZOS), dtype=np.uint8)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Resize PIL image to 256×256 and return BGR uint8 array (or None if gray)."""
    rgb = np.array(img.convert("RGB").resize((256, 256), Image.LANCZOS))
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    delta = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    if delta.max() <= 10 or delta.mean() <= 1.0:
        return None                               # grayscale — no meaningful colour
    return rgb[:, :, ::-1].copy()                 # RGB → BGR


def _ycrcb_colorize(source_bgr: np.ndarray, mri_u8: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    """MRI anatomy + PET/SPECT colour — exact port of generate_all_fused.ycrcb_colorize."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mri_enhanced = clahe.apply(mri_u8)
    ycrcb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.normalize(mri_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    colored_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    mri_bgr = cv2.cvtColor(mri_enhanced, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(colored_bgr, alpha, mri_bgr, 1.0 - alpha, 0)
    _, mask = cv2.threshold(mri_enhanced, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_and(blended, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))


def _adafuse_colorize(functional_u8: np.ndarray, mri_u8: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """HOT colormap over MRI background — exact port of generate_all_fused.adafuse_colorize."""
    functional_norm = cv2.normalize(functional_u8, None, 0, 255, cv2.NORM_MINMAX)
    hot_bgr = cv2.applyColorMap(functional_norm, cv2.COLORMAP_HOT)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mri_enhanced = clahe.apply(mri_u8)
    mri_bgr = cv2.cvtColor(mri_enhanced, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(hot_bgr, alpha, mri_bgr, 1.0 - alpha, 0)
    _, mask = cv2.threshold(mri_enhanced, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_and(blended, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))


def _colorize_pet_spect(source1_pil: Image.Image, source2_pil: Image.Image) -> Image.Image:
    """Produce the colour PET/SPECT-MRI fusion display.

    Matches generate_all_fused.py exactly:
      • If source1 has colour → ycrcb_colorize(spect_bgr, mri_u8)
        MRI used as luminance, SPECT/PET Cr/Cb as colour, 70/30 blend + brain mask.
      • If source1 is grayscale → adafuse_colorize(functional_u8, mri_u8)
        HOT colormap blended 65/35 over CLAHE-enhanced MRI + brain mask.
    """
    src1_bgr = _pil_to_bgr(source1_pil)           # None when grayscale
    src1_u8  = _pil_to_uint8(source1_pil)          # grayscale fallback
    mri_u8   = _pil_to_uint8(source2_pil)

    if src1_bgr is not None:
        result_bgr = _ycrcb_colorize(src1_bgr, mri_u8)
    else:
        result_bgr = _adafuse_colorize(src1_u8, mri_u8)

    result_rgb = result_bgr[:, :, ::-1].copy()
    return Image.fromarray(result_rgb, "RGB")


def _fmt(value: float, decimals: int = 4) -> str:
    """Format a float; return 'inf' for infinity."""
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{decimals}f}"


# -- Core inference + metrics ---------------------------------------------------

def fuse_images(source1_img, source2_img, pair_choice: str):
    """
    Run the selected generator on the uploaded image pair and compute metrics.

    Returns:
        source1_display  - PIL image (RGB)
        source2_display  - PIL image (RGB)
        fused_display    - PIL image (RGB)
        metrics_df       - list[list] table for gr.Dataframe
        status_msg       - str
    """
    if source1_img is None or source2_img is None:
        err = "WARN: Please upload both source images before fusing."
        return None, None, None, [], err

    try:
        # PIL images come in directly from gr.Image(type="pil")
        src1_pil = source1_img if isinstance(source1_img, Image.Image) else Image.fromarray(source1_img)
        src2_pil = source2_img if isinstance(source2_img, Image.Image) else Image.fromarray(source2_img)

        # Preserve original color for display and PET/SPECT colorization
        src1_display = src1_pil.convert("RGB").resize((256, 256), Image.LANCZOS)
        src2_display = src2_pil.convert("RGB").resize((256, 256), Image.LANCZOS)

        # Always use grayscale for the model
        src1_t = _pil_to_tensor(src1_pil)
        src2_t = _pil_to_tensor(src2_pil)

        model = MODELS[pair_choice]

        with torch.no_grad():
            # FusionGenerator.forward(ct, mri) – source1 maps to the first
            # modality (CT/PET/SPECT), source2 to MRI.
            fused_t = model(src1_t, src2_t)

        # ── Post-process ─────────────────────────────────────────────────────
        is_pet_spect = pair_choice in ("PET-MRI", "SPECT-MRI")

        if not is_pet_spect:
            # CT-MRI only: the generator is biased toward MRI (source2).
            # The training objective was target = max(CT, MRI), so enforce it:
            # wherever CT is brighter than the model output (bone/skull regions),
            # use CT — this restores the white bone ring that the network drops.
            fused_t = torch.maximum(fused_t, src1_t)
            fused_t = fused_t.clamp(0.0, 1.0)

        # ── Colorize fused output for PET/SPECT ───────────────────────────────
        if is_pet_spect:
            # Matches generate_all_fused.py: MRI as luminance, SPECT/PET as colour
            fused_display = _colorize_pet_spect(src1_pil, src2_pil)
        else:
            fused_display = _tensor_to_pil(fused_t)

        # ── Compute metrics (data_range=255.0 matches thesis convention) ──────
        label1, label2 = PAIR_LABELS[pair_choice]

        psnr_s1  = psnr(fused_t, src1_t, data_range=255.0)
        psnr_s2  = psnr(fused_t, src2_t, data_range=255.0)
        ssim_s1  = ssim(fused_t, src1_t)
        ssim_s2  = ssim(fused_t, src2_t)
        mi_s1    = mutual_information(fused_t, src1_t)
        mi_s2    = mutual_information(fused_t, src2_t)
        cc_s1    = correlation_coefficient(fused_t, src1_t)
        cc_s2    = correlation_coefficient(fused_t, src2_t)
        fmi_val  = feature_mutual_information(fused_t, src1_t, src2_t)
        en_val   = entropy(fused_t)
        sf_val   = spatial_frequency(fused_t)
        ag_val   = average_gradient(fused_t)

        metrics_table = [
            ["Metric",                 f"Fused vs {label1}", f"Fused vs {label2}", "Description"],
            ["PSNR (dB)",              _fmt(psnr_s1),         _fmt(psnr_s2),        "Peak Signal-to-Noise Ratio"],
            ["SSIM",                   _fmt(ssim_s1),         _fmt(ssim_s2),        "Structural Similarity Index"],
            ["MI (bits)",              _fmt(mi_s1),           _fmt(mi_s2),          "Mutual Information"],
            ["CC",                     _fmt(cc_s1),           _fmt(cc_s2),          "Correlation Coefficient"],
            ["FMI",                    _fmt(fmi_val),         "-",                  "Feature Mutual Information (both sources)"],
            ["EN (bits)",              _fmt(en_val),          "-",                  "Information Entropy of fused image"],
            ["SF",                     _fmt(sf_val),          "-",                  "Spatial Frequency"],
            ["AG",                     _fmt(ag_val),          "-",                  "Average Gradient"],
        ]

        status = f"OK: Fusion complete [{pair_choice}] | Device: {str(DEVICE).upper()}"
        return (
            src1_display,
            src2_display,
            fused_display,
            metrics_table,
            status,
        )

    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        return None, None, None, [], f"ERROR: {exc}\n\n{tb}"


# -- Dynamic label updater ------------------------------------------------------

def update_labels(pair_choice: str):
    label1, label2 = PAIR_LABELS[pair_choice]
    return gr.update(label=label1), gr.update(label=label2)


# -- Gradio UI ------------------------------------------------------------------

CSS = """
/* ── Global ──────────────────────────────────────────────────────────── */
body, .gradio-container {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header banner ───────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #2d4a6e;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
.app-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #63b3ed !important;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.app-header p {
    font-size: 0.93rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.6;
}
.app-header .badge {
    display: inline-block;
    background: #1e3a5f;
    border: 1px solid #2d6a9f;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #90cdf4;
    margin-top: 8px;
    margin-right: 6px;
}

/* ── Panels ──────────────────────────────────────────────────────────── */
.panel-box {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35);
}
.panel-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #63b3ed;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 8px;
}

/* ── Fuse button ─────────────────────────────────────────────────────── */
.fuse-btn {
    background: linear-gradient(135deg, #2b6cb0, #1e4e8c) !important;
    color: #fff !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 14px 28px !important;
    box-shadow: 0 4px 14px rgba(43,108,176,0.4) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin-top: 8px;
}
.fuse-btn:hover {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    box-shadow: 0 6px 20px rgba(43,108,176,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Output image labels ─────────────────────────────────────────────── */
.output-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-align: center;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

/* ── Metrics table ───────────────────────────────────────────────────── */
.metrics-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.metrics-table th, .metrics-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #2d3748;
    text-align: left;
}
.metrics-table thead tr {
    background: #1e3a5f;
    color: #90cdf4;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.metrics-table tbody tr:nth-child(even) { background: #1a2233; }
.metrics-table tbody tr:hover { background: #1e2d45; }

/* ── Status bar ──────────────────────────────────────────────────────── */
.status-bar {
    font-size: 0.88rem;
    color: #68d391;
    background: #1a2e1a;
    border: 1px solid #2d5a2d;
    border-radius: 6px;
    padding: 8px 14px;
}

/* ── Dropdown & upload overrides ─────────────────────────────────────── */
.gradio-dropdown label span,
.gradio-image label span {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
"""

with gr.Blocks(
    css=CSS,
    title="Medical Image Fusion Demo",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
) as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
        <h1>🧠 Medical Image Fusion</h1>
        <p>
            GAN-based multimodal fusion using a Dual-Encoder + Spatial-Frequential Fusion (SFF) architecture.<br>
            Upload two complementary images, select the modality pair, and generate a fused result with full quality metrics.
        </p>
        <span class="badge">CT · PET · SPECT + MRI</span>
        <span class="badge">256 × 256 | Color PET/SPECT</span>
        <span class="badge">Bachelor Thesis Demo</span>
    </div>
    """)

    # ── Main row ───────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left panel: controls ────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<div class="panel-title">Configuration</div>')

            pair_dropdown = gr.Dropdown(
                choices=["CT-MRI", "PET-MRI", "SPECT-MRI"],
                value="CT-MRI",
                label="Modality Pair",
                info="Selects the pre-trained generator for this combination.",
                interactive=True,
            )

            gr.HTML('<div style="height:12px"></div>')
            gr.HTML('<div class="panel-title" style="margin-top:6px">Source Images</div>')

            src1_input = gr.Image(
                label="CT Image",
                type="pil",
                height=220,
            )
            src2_input = gr.Image(
                label="MRI Image",
                type="pil",
                height=220,
            )

            fuse_btn = gr.Button("⚡  Fuse Images", elem_classes="fuse-btn", variant="primary")

            status_out = gr.Textbox(
                label="",
                interactive=False,
                elem_classes="status-bar",
                show_label=False,
                placeholder="Status will appear here after fusion …",
                max_lines=3,
            )

        # ── Right panel: outputs ────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML('<div class="panel-title">Fusion Result</div>')

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML('<div class="output-label">Source 1</div>')
                    out_src1 = gr.Image(
                        label="Source 1",
                        show_label=False,
                        type="pil",
                        height=256,
                        interactive=False,
                    )

                with gr.Column():
                    gr.HTML('<div class="output-label">Source 2</div>')
                    out_src2 = gr.Image(
                        label="Source 2",
                        show_label=False,
                        type="pil",
                        height=256,
                        interactive=False,
                    )

                with gr.Column():
                    gr.HTML('<div class="output-label" style="color:#68d391">Fused Result</div>')
                    out_fused = gr.Image(
                        label="Fused",
                        show_label=False,
                        type="pil",
                        height=256,
                        interactive=False,
                    )

            gr.HTML('<div style="height:20px"></div>')
            gr.HTML('<div class="panel-title">Quality Metrics</div>')
            gr.HTML("""
            <p style="font-size:0.8rem; color:#718096; margin:-8px 0 12px 0;">
                All metrics computed between the fused image and each source.
                FMI, EN, SF, and AG are global (no per-source breakdown).
            </p>
            """)

            metrics_out = gr.Dataframe(
                headers=["Metric", "Fused vs Source 1", "Fused vs Source 2", "Description"],
                datatype=["str", "str", "str", "str"],
                col_count=(4, "fixed"),
                interactive=False,
                wrap=True,
                elem_classes="metrics-table",
            )

    # ── Footer ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; margin-top:28px; padding:18px; border-top:1px solid #2d3748;">
        <p style="font-size:0.8rem; color:#4a5568; margin:0;">
            Bachelor Thesis · Medical Image Fusion with GANs ·
            Dual-Encoder + Spatial-Frequential Fusion (SFF) Architecture
        </p>
    </div>
    """)

    # ── Event wiring ────────────────────────────────────────────────────────────

    # Update input labels when the modality pair changes
    pair_dropdown.change(
        fn=update_labels,
        inputs=[pair_dropdown],
        outputs=[src1_input, src2_input],
    )

    # Run fusion on button click
    fuse_btn.click(
        fn=fuse_images,
        inputs=[src1_input, src2_input, pair_dropdown],
        outputs=[out_src1, out_src2, out_fused, metrics_out, status_out],
        api_name="fuse",
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        inbrowser=True,
        show_error=True,
    )
