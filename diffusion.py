import torch
import math

class Diffusion:
    def __init__(self, T=1000, beta_schedule="linear"):
        self.T = T
        self.beta_schedule = beta_schedule
        self.register_buffer = lambda x: x.to(torch.float32)

        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, T)
        elif beta_schedule == "cosine":
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos((x / T + 0.008) / (1 + 0.008) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clip(betas, 0.0001, 0.9999)
            self.betas = betas

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return (
            self.sqrt_alphas_cumprod[t, None, None, None] * x0 
            + self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

    def p_sample(self, model, x, t):
        t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
        with torch.no_grad():
            out = model(x, t_batch)
            mu, log_var = out.chunk(2, dim=1)
            eps = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            var = self.posterior_variance[t] * torch.ones_like(x)
            return mu + eps * torch.exp(0.5 * log_var) * torch.sqrt(var)

    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)