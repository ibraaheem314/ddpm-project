import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=1000, beta_schedule='cosine', device='cuda'):
        self.T = T
        self.device = torch.device(device)
        
        # Cosine schedule
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
        # Si t == 0, on retourne x directement
        if t <= 0:
            return x
        beta = self.betas[t-1]  # indice t-1 car betas a T éléments
        alpha = self.alphas[t-1]
        sqrt_recip_alpha = 1. / torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha)
        # Ajout d’un bruit aléatoire sauf pour t==1
        noise = torch.randn_like(x) if t > 1 else 0.
        x_prev = sqrt_recip_alpha * (x - (beta / sqrt_one_minus_alpha) * predicted_noise) + noise
        return x_prev

    def forward_process(self, x0, t, noise=None):
        # Retourne xt et le bruit utilisé (pour visualiser l’effet de la diffusion)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return xt, noise
