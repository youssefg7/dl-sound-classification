import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio.transforms as T


class ASTModel(nn.Module):
    def __init__(self, sample_rate=44100, patch_size=16, patch_stride=10, overlap=6,
                 num_classes=50, pretrained_model='deit_base_patch16_384'):
        super().__init__()

        self.f_dim = 128
        self.emb_dim = 768
        self.num_classes = num_classes

        # Audio frontend
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=160,
            win_length=400,
            n_mels=self.f_dim
        )
        self.db = T.AmplitudeToDB(top_db=80)

        # Estimate time dimension
        with torch.no_grad():
            dummy = torch.randn(1, sample_rate * 5)
            self.t_dim = self.db(self.melspec(dummy)).shape[-1]

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
            self.patch_embed.weight.copy_(vit.patch_embed.proj.weight.mean(dim=1, keepdim=True))
            self.patch_embed.bias.copy_(vit.patch_embed.proj.bias)

        self.cls_token = nn.Parameter(vit.cls_token.clone())
        self.pos_embed = nn.Parameter(self.interpolate_pos_embed(vit.pos_embed))

        self.transformer = vit.blocks
        self.norm = vit.norm
        self.head = nn.Linear(self.emb_dim, num_classes)

    def interpolate_pos_embed(self, pos_embed):
        cls = pos_embed[:, :1]
        patch = pos_embed[:, 1:]
        patch = patch.reshape(1, *self.old_grid, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=self.new_grid, mode='bilinear', align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, self.emb_dim)
        return torch.cat((cls, patch), dim=1)

    def forward(self, x):
        # x: (B, 1, T) or (B, 1, 1, T)
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        if x.dim() == 3:
            x = self.db(self.melspec(x.squeeze(1))).unsqueeze(1)

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)
        return torch.sigmoid(self.head(x[:, 0]))
