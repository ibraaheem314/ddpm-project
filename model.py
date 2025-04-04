import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.gelu(self.norm1(x)))
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.gelu(self.norm2(h)))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('b h d i, b h d j -> b h i j', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, image_size=32):
        super().__init__()
        self.time_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        self.down1 = ResidualBlock(in_channels, 64, self.time_dim)
        self.down2 = ResidualBlock(64, 128, self.time_dim)
        self.down3 = ResidualBlock(128, 256, self.time_dim)
        self.down_attn = AttentionBlock(256)
        self.pool = nn.MaxPool2d(2)
        
        self.mid_block1 = ResidualBlock(256, 256, self.time_dim)
        self.mid_attn = AttentionBlock(256)
        self.mid_block2 = ResidualBlock(256, 256, self.time_dim)
        
        self.up1 = ResidualBlock(384, 128, self.time_dim)   # 256 + 128 = 384
        self.up2 = ResidualBlock(192, 64, self.time_dim)    # 128 + 64 = 192
        self.up3 = ResidualBlock(64, 64, self.time_dim)     # 64 → 64
        self.out_conv = nn.Conv2d(64, out_channels, 3, padding=1)

    def time_embedding(self, t, dim):
        device = t.device
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def forward(self, x, t):
        t_emb = self.time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        x1 = self.down1(x, t_emb)
        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb)
        x3 = self.pool(x2)
        x3 = self.down3(x3, t_emb)
        x3 = self.down_attn(x3)
        
        # Bottleneck
        x = self.mid_block1(x3, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x, t_emb)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x, t_emb)
        
        x = self.up3(x, t_emb)
        return self.out_conv(x)