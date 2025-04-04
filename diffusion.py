import torch
import math

class Diffusion:
    def __init__(self, T=1000, beta_schedule="cosine", device="cuda"):
        self.T = T
        self.device = device
        self.betas = self._get_beta_schedule(beta_schedule).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def _get_beta_schedule(self, schedule):
        if schedule == "cosine":
            betas = self._cosine_beta_schedule()
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T)
        else:
            raise ValueError(f"Planification {schedule} non supportée.")
        return betas

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.T + 1
        x = torch.linspace(0, self.T, steps)
        alphas_cumprod = torch.cos(((x / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward_process(self, x0, t):
        noise = torch.randn_like(x0, device=x0.device)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(x0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(x0.device)
        xt = sqrt_alpha[:, None, None, None] * x0 + sqrt_one_minus_alpha[:, None, None, None] * noise
        return xt, noise

    def sample(self, model, shape):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        for t in range(self.T, 0, -1):
            t_tensor = torch.full((shape[0],), t, device=device)
            with torch.no_grad():
                pred_noise = model(x, t_tensor)
            x = self._reverse_step(x, pred_noise, t-1)
        return x.clamp(-1, 1)

    def _reverse_step(self, x, pred_noise, t_idx):
        alpha_t = self.alphas[t_idx].to(x.device)
        alpha_cumprod_t = self.alphas_cumprod[t_idx].to(x.device)
        beta_t = self.betas[t_idx].to(x.device)
        
        x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_t)
        if t_idx > 0:
            x += torch.sqrt(beta_t) * torch.randn_like(x)
        return x