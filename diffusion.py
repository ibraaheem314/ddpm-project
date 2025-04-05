import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=1000, beta_schedule='cosine', device='cuda'):
        self.T = T
        self.device = torch.device(device)
        
        # Schedule cosine officiel
        s = 0.008
        ts = torch.arange(T + 1, device=device) / T
        alphas_cumprod = torch.cos((ts + s)/(1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0, 0.999)
        
        # Calculs optimisés
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