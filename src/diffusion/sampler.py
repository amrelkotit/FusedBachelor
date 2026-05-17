import torch

from src.diffusion.utils import initial_fusion


@torch.no_grad()
def sample_refined(model, scheduler, source1, source2, sampling_steps=50, use_initial_noise=True):
    model.eval()
    initial = initial_fusion(source1, source2)
    total_steps = scheduler.timesteps
    indices = torch.linspace(total_steps - 1, 0, steps=max(1, int(sampling_steps)), device=source1.device).long()
    if use_initial_noise:
        t0 = indices[0].repeat(source1.shape[0])
        x = scheduler.add_noise(initial, torch.randn_like(initial), t0)
    else:
        x = torch.randn_like(initial)
    for step_index, t_value in enumerate(indices):
        t = t_value.repeat(source1.shape[0])
        noise_pred = model(x, source1, source2, initial, t)
        x0_pred = scheduler.predict_x0(x, noise_pred, t)
        if step_index == len(indices) - 1:
            x = x0_pred
            break
        next_t = indices[step_index + 1].repeat(source1.shape[0])
        next_alpha_bar = scheduler.alpha_bars[next_t].view(-1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        x = next_alpha_bar.sqrt() * x0_pred + (1.0 - next_alpha_bar).sqrt() * noise_pred
        x = x.clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0), initial
