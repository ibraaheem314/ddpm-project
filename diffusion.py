import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=1000, beta_schedule='cosine', device='cuda'):
        self.T = T
        self.device = torch.device(device)

        # Cosine schedule comme dans le papier
        s = 0.008
        ts = torch.arange(T + 1, device=device) / T
        alphas_cumprod = torch.cos((ts + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = torch.clamp(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0, 0.999)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, shape, device):
        model.eval()
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t_tensor = torch.tensor([t] * shape[0], device=device)
            predicted_noise = model(x, t_tensor)
            x = self._reverse_step(x, predicted_noise, t)
        return x.clamp(-1, 1)

    def _reverse_step(self, x, predicted_noise, t):
        if t <= 0:
            return x

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mu_theta = coef1 * (x - coef2 * predicted_noise)

        if t > 1:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            return mu_theta + sigma_t * noise
        else:
            return mu_theta

    def forward_process(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return xt, noise
