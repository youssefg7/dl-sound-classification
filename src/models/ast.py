import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ASTModel(nn.Module):
    def __init__(self, sample_rate=44100, patch_size=16, patch_stride=10, overlap=6,
                 num_classes=50, pretrained_model='deit_base_patch16_384'):
        super().__init__()

        self.f_dim = 128
        self.emb_dim = 768
        self.num_classes = num_classes

        # Estimate time dimension for typical 10-second clips (our preprocessing uses hop_length=160)
        self.t_dim = int((sample_rate * 10) / 160) + 1

        # Positional embedding interpolation grid
        self.old_grid = (24, 24)
        self.new_grid = (
            (self.f_dim - patch_size) // (patch_size - overlap) + 1,
            (self.t_dim - patch_size) // (patch_size - overlap) + 1
        )

        # Load pretrained ViT
        vit = timm.create_model(pretrained_model, pretrained=True)
        self.patch_embed = nn.Conv2d(1, self.emb_dim, kernel_size=patch_size, stride=patch_stride, bias=True)
        with torch.no_grad():
            self.patch_embed.weight.copy_(vit.patch_embed.proj.weight.mean(dim=1, keepdim=True))  # type: ignore
            self.patch_embed.bias.copy_(vit.patch_embed.proj.bias)  # type: ignore

        self.cls_token = nn.Parameter(vit.cls_token.clone())  # type: ignore
        self.pos_embed = nn.Parameter(self.interpolate_pos_embed(vit.pos_embed))  # type: ignore

        self.transformer = vit.blocks  # type: ignore
        self.norm = vit.norm  # type: ignore
        self.head = nn.Linear(self.emb_dim, num_classes)

    def interpolate_pos_embed(self, pos_embed):
        cls = pos_embed[:, :1]
        patch = pos_embed[:, 1:]
        patch = patch.reshape(1, *self.old_grid, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=self.new_grid, mode='bilinear', align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, self.emb_dim)
        return torch.cat((cls, patch), dim=1)

    def forward(self, x):
        # x: (B, F, T) - preprocessed log-mel spectrogram from ASTPreprocessor
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension: (B, 1, F, T)

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        for blk in self.transformer:  # type: ignore
            x = blk(x)
        x = self.norm(x)  # type: ignore
        return torch.sigmoid(self.head(x[:, 0]))
