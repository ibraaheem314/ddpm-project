import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, out_channels)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.group_norm1(x)))
        h += self.time_mlp(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.group_norm2(h)))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.group_norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.group_norm(x)).reshape(B, 3, self.num_heads, C // self.num_heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('b h d i, b h d j -> b h i j', q, k) * scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('b h i j, b h d j -> b h d i', attn, v).reshape(B, C, H, W)
        return x + self.proj(x)

def timestep_embedding(t, channels):
    half_dim = channels // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device)) * -emb
    emb = t[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 out_channels=3,
                 model_channels=128,
                 channel_mult=(1, 2, 2, 2),
                 num_res_blocks=2):
        super().__init__()
        self.model_channels = model_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels))
        
        # Encoder
        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])
        channels = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(channels, mult * model_channels, model_channels),
                    AttentionBlock(mult * model_channels)
                ]
                self.input_blocks.extend(layers)
                channels = mult * model_channels
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1))
        
        # Bottleneck
        self.middle_block = nn.ModuleList([
            ResidualBlock(channels, channels, model_channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels, model_channels)
        ])
        
        # Decoder
        self.output_blocks = nn.ModuleList([])
        self.skip_connection_flags = []  # Indique si un bloc doit recevoir un skip (True) ou non (False)
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                in_ch = channels + (mult * model_channels if i == 0 else 0)
                rb = ResidualBlock(in_ch, mult * model_channels, model_channels)
                self.output_blocks.append(rb)
                # On attend un skip connection uniquement pour le premier bloc de chaque niveau
                self.skip_connection_flags.append(i == 0)
                if level == 0 and i == num_res_blocks:
                    attn = AttentionBlock(mult * model_channels)
                    self.output_blocks.append(attn)
                    self.skip_connection_flags.append(False)
                channels = mult * model_channels
            if level != 0:
                self.output_blocks.append(
                    nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1, output_padding=1)
                )
                self.skip_connection_flags.append(False)
        
        self.out = nn.Conv2d(model_channels, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_embed = self.time_embed(timestep_embedding(t, self.model_channels))
    
    # Encodeur : on stocke toutes les sorties
        skips = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_embed)
            else:
                h = module(h)
            skips.append(h)
    
    # Bottleneck
        for module in self.middle_block:
            if isinstance(module, ResidualBlock):
                h = module(h, t_embed)
            else:
                h = module(h)
    
    # Décodeur : pour chaque module qui attend un skip, on sélectionne dans 'skips'
        for module, use_skip in zip(self.output_blocks, self.skip_connection_flags):
            if use_skip:
                found = False
            # On parcourt la liste des skip connections depuis la fin
                for idx in range(len(skips) - 1, -1, -1):
                    if skips[idx].shape[2:] == h.shape[2:]:
                        h = torch.cat([h, skips.pop(idx)], dim=1)
                        found = True
                        break
                if not found:
                    raise ValueError("Aucune skip connection avec la bonne résolution n'a été trouvée pour h de taille {}.".format(h.shape[2:]))
            if isinstance(module, ResidualBlock):
                h = module(h, t_embed)
            else:
                h = module(h)
    
        return self.out(h)
