import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, emb_dim=192, patch_size=16, stride=10):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x = self.proj(x)  # (B, emb_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, emb_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(mlp_ratio * emb_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * emb_dim), emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ASTMiniViT(nn.Module):
    def __init__(self, sample_rate=44100, patch_size=16, patch_stride=10, overlap=6,
                 num_classes=50, emb_dim=192, depth=6, num_heads=3, f_dim=128):
        super().__init__()

        self.f_dim = f_dim
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        # Calculate time dimension for 10s clips
        self.t_dim = int((sample_rate * 10) / 160) + 1

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=1, emb_dim=emb_dim, patch_size=patch_size, stride=patch_stride)

        # Grid size
        self.grid_size = (
            (self.f_dim - patch_size) // (patch_size - overlap) + 1,
            (self.t_dim - patch_size) // (patch_size - overlap) + 1
        )
        num_patches = self.grid_size[0] * self.grid_size[1]

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(emb_dim=emb_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: (B, F, T) → (B, 1, F, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.patch_embed(x)  # (B, N, D)
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]

        x = self.blocks(x)
        x = self.norm(x)
        return torch.sigmoid(self.head(x[:, 0]))  # Use CLS token output
