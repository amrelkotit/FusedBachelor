import torch


class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = int(timesteps)
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, self.timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0, noise, t):
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1).to(device=x0.device, dtype=x0.dtype)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def predict_x0(self, xt, noise_pred, t):
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1).to(device=xt.device, dtype=xt.dtype)
        return ((xt - (1.0 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt().clamp_min(1e-8)).clamp(0.0, 1.0)

    def step(self, xt, noise_pred, t):
        beta = self.betas[t].view(-1, 1, 1, 1).to(device=xt.device, dtype=xt.dtype)
        alpha = self.alphas[t].view(-1, 1, 1, 1).to(device=xt.device, dtype=xt.dtype)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1).to(device=xt.device, dtype=xt.dtype)
        mean = (xt - beta * noise_pred / (1.0 - alpha_bar).sqrt().clamp_min(1e-8)) / alpha.sqrt().clamp_min(1e-8)
        noise = torch.randn_like(xt)
        nonzero = (t > 0).float().view(-1, 1, 1, 1)
        return mean + nonzero * beta.sqrt() * noise

