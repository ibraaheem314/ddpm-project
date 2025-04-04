import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = self.dropout(self.conv2(F.silu(self.norm2(h))))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).view(B, 3, self.num_heads, C//self.num_heads, H*W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('b h d i, b h d j -> b h i j', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        out = out.view(B, C, H, W)
        return x + self.proj(out)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, image_size=32, channels=128, channels_mult=[1,2,2,4], num_res_blocks=2, num_heads=4):
        super().__init__()
        self.time_dim = channels * 4
        self.channels = channels
        self.image_size = image_size
        self.init_conv = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Encodage temporel
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        channel_mults = [channels * m for m in channels_mult]
        for i in range(len(channel_mults)):
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    channel_mults[i-1] if i > 0 else channels,
                    channel_mults[i],
                    self.time_dim
                ))
                self.downs.append(AttentionBlock(channel_mults[i], num_heads=num_heads))
            if i != len(channel_mults) -1:
                self.downs.append(nn.AvgPool2d(2))

        # Bottleneck
        mid_channels = channel_mults[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, self.time_dim)
        self.mid_attn = AttentionBlock(mid_channels, num_heads=num_heads)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, self.time_dim)

        # Décodage
        for i in reversed(range(len(channel_mults))):
            for _ in range(num_res_blocks + 1):
                in_ch = channel_mults[i] * 2 if _ < num_res_blocks else channel_mults[i]
                out_ch = channel_mults[i]
                self.ups.append(ResidualBlock(in_ch, out_ch, self.time_dim))
                self.ups.append(AttentionBlock(out_ch, num_heads=num_heads))
            if i != 0:
                self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels * 2, 3, padding=1)  # Sortie μ et σ
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        x = self.init_conv(x)
        skips = [x]

        # Encoder
        for layer in self.downs:
            if isinstance(layer, nn.AvgPool2d):
                skips.append(x)
                x = layer(x)
            else:
                x = layer(x, t_emb)
                skips.append(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for layer in self.ups:
            if isinstance(layer, nn.Upsample):
                x = layer(x)
            else:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)

        return self.final_conv(x)