"""
DDIM-style sampler for conditional medical image fusion diffusion.

Supports two prediction modes (controlled by `prediction_type`):

  "x0"      — model predicts the clean image directly.  No amplification
               instability; x0_pred comes straight from the model.  Recommended
               for newly trained checkpoints.

  "epsilon"  — model predicts the added noise (legacy).  x0_pred is recovered
               via x0 = (x - sqrt(1-ᾱ)·ε) / sqrt(ᾱ), which amplifies errors
               by 1/sqrt(ᾱ) ≈ 150× at t≈999.  Kept for backward compatibility
               with pre-existing pet_mri / spect_mri checkpoints.

Other parameters:
  eta=0  → deterministic DDIM (sharper, recommended for inference)
  eta=1  → stochastic DDPM   (matches training distribution exactly)
  max_start_t : cap on starting timestep (default 400 for "epsilon" mode;
                ignored for "x0" mode which is stable at any t).
"""

import torch

from src.diffusion.utils import initial_fusion


@torch.no_grad()
def sample_refined(
    model,
    scheduler,
    source1,
    source2,
    sampling_steps: int     = 200,
    use_initial_noise: bool = True,
    eta: float              = 0.0,
    max_start_t: int        = 400,
    prediction_type: str    = "x0",
):
    """
    Generate a fused image from source1 and source2 via reverse diffusion.

    Args:
        model             : trained ConditionalUNet
        scheduler         : DiffusionScheduler
        source1           : [B,1,H,W] first modality in [0,1]
        source2           : [B,1,H,W] second modality in [0,1]
        sampling_steps    : denoising steps
        use_initial_noise : start from noised average fusion rather than pure noise
        eta               : 0 = deterministic DDIM, 1 = stochastic DDPM
        max_start_t       : cap on starting timestep for "epsilon" mode (default 400)
        prediction_type   : "x0" (direct image prediction, stable) or
                            "epsilon" (noise prediction, legacy)

    Returns:
        (fused, initial): [B,1,H,W] tensors clamped to [0,1]
    """
    model.eval()
    device = source1.device
    B      = source1.shape[0]

    initial     = initial_fusion(source1, source2)   # 0.5*s1 + 0.5*s2
    total_steps = scheduler.timesteps

    # For epsilon-prediction, cap the start timestep to avoid near-pure-noise
    # initialisation where 1/sqrt(alpha_bar) explodes.  x0-prediction is
    # numerically stable at any t, so no capping is needed there.
    if prediction_type == "epsilon":
        effective_start = min(total_steps - 1, int(max_start_t)) if use_initial_noise else total_steps - 1
    else:
        effective_start = total_steps - 1 if not use_initial_noise else min(total_steps - 1, int(max_start_t))

    indices = torch.linspace(
        effective_start, 0,
        steps=max(1, int(sampling_steps)),
        device=device,
    ).long()

    # ── initialise noisy sample ──────────────────────────────────────────
    if use_initial_noise:
        t0 = indices[0].repeat(B)
        x  = scheduler.add_noise(initial, torch.randn_like(initial), t0)
    else:
        x = torch.randn_like(initial)

    # ── reverse diffusion loop ───────────────────────────────────────────
    x0_pred = initial   # fallback if sampling_steps == 1

    for step_idx, t_val in enumerate(indices):
        t           = t_val.repeat(B)
        alpha_bar_t = scheduler.alpha_bars[t].view(B, 1, 1, 1).to(x.device, x.dtype)

        if prediction_type == "x0":
            # Model directly predicts the clean image — no amplification.
            x0_pred    = model(x, source1, source2, initial, t)
            # Derive the noise direction for the DDIM update.
            noise_pred = (x - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)
        else:
            # Legacy epsilon-prediction path.
            noise_pred = model(x, source1, source2, initial, t)
            x0_pred    = (x - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt().clamp_min(1e-8)

        if step_idx == len(indices) - 1:
            break   # final step: use x0_pred directly

        # DDIM update
        next_t         = indices[step_idx + 1].repeat(B)
        alpha_bar_next = scheduler.alpha_bars[next_t].view(B, 1, 1, 1).to(x.device, x.dtype)

        direction = (1.0 - alpha_bar_next - eta ** 2 * (1.0 - alpha_bar_next)).clamp_min(0.0).sqrt() * noise_pred

        stoch = eta * (1.0 - alpha_bar_next).sqrt() * torch.randn_like(x) if eta > 0 else 0.0

        x = alpha_bar_next.sqrt() * x0_pred + direction + stoch

    return x0_pred.clamp(0.0, 1.0), initial
