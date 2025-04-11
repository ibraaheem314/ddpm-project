import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, out_channels)
        self.group_norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.group_norm1(x)))
        h += self.time_mlp(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.group_norm2(h)))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.group_norm = nn.GroupNorm(8, channels)
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
                 model_channels=64,
                 channel_mult=(1, 2, 2),
                 num_res_blocks=1):
        super().__init__()
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels)
        )

        # Encoder
        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])
        channels = model_channels
        for i, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.input_blocks.append(ResidualBlock(channels, mult * model_channels, model_channels))
                channels = mult * model_channels
            if i != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1))

        # Bottleneck (attention ici uniquement)
        self.middle_block = nn.Sequential(
            ResidualBlock(channels, channels, model_channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels, model_channels)
        )

        # Decoder
        self.output_blocks = nn.ModuleList()
        self.skip_connection_flags = []
        for i, mult in reversed(list(enumerate(channel_mult))):
            for j in range(num_res_blocks + 1):
                in_ch = channels + (mult * model_channels if j == 0 else 0)
                self.output_blocks.append(ResidualBlock(in_ch, mult * model_channels, model_channels))
                self.skip_connection_flags.append(j == 0)
                channels = mult * model_channels
            if i != 0:
                self.output_blocks.append(nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1, output_padding=1))
                self.skip_connection_flags.append(False)

        self.out = nn.Conv2d(model_channels, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_embed = self.time_embed(timestep_embedding(t, self.model_channels))
        skips = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_embed)
            else:
                h = module(h)
            skips.append(h)

        for module in self.middle_block:
            h = module(h, t_embed) if isinstance(module, ResidualBlock) else module(h)

        for module, use_skip in zip(self.output_blocks, self.skip_connection_flags):
            if use_skip:
                for idx in range(len(skips)-1, -1, -1):
                    if skips[idx].shape[2:] == h.shape[2:]:
                        h = torch.cat([h, skips.pop(idx)], dim=1)
                        break
            h = module(h, t_embed) if isinstance(module, ResidualBlock) else module(h)

        return self.out(h)
