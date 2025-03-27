# model.py (Fully corrected U-Net for DDPM CIFAR-10 in PyTorch)

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32) * torch.log(torch.tensor(10000.0)) / (half - 1)
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.block[0:3](x)
        h += self.emb_proj(t_emb)[:, :, None, None]
        h = self.block[3:](h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, ch=64, out_ch=3, ch_mult=(1, 2, 2), num_res_blocks=2, dropout=0.1):
        super().__init__()
        emb_dim = ch * 4

        self.time_embed = nn.Sequential(
            nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

        self.in_conv = nn.Conv2d(3, ch, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = []
        now_ch = ch
        for mult in ch_mult:
            out_channels = ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(now_ch, out_channels, emb_dim, dropout))
                now_ch = out_channels
                channels.append(now_ch)
            self.down_blocks.append(nn.Conv2d(now_ch, now_ch, 4, stride=2, padding=1))

        self.mid_block = ResBlock(now_ch, now_ch, emb_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for mult in reversed(ch_mult):
            out_channels = ch * mult
            self.up_blocks.append(nn.ConvTranspose2d(now_ch, out_channels, 4, 2, 1))
            now_ch = out_channels
            for _ in range(num_res_blocks):
                skip_ch = channels.pop()
                self.up_blocks.append(ResBlock(now_ch + skip_ch, out_channels, emb_dim, dropout))
                now_ch = out_channels

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))
        hs = []

        h = self.in_conv(x)
        for layer in self.down_blocks:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                hs.append(h)
            else:
                h = layer(h)

        h = self.mid_block(h, t_emb)

        for layer in self.up_blocks:
            if isinstance(layer, ResBlock):
                skip = hs.pop()
                h = layer(torch.cat([h, skip], dim=1), t_emb)
            else:
                h = layer(h)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
