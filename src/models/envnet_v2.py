import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnvNetV2(nn.Module):
    """
    Minimal EnvNet-v2 for 44.1 kHz mono waveform --> 1 x 44100 x 5s tensor. Layer widths follow Tokozume 2018.
    """

    def __init__(self, num_classes: int = 50, dropout: float = 0.5):
        super().__init__()

        # ---------- 1. Temporal front-end -----------------------------------
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 64), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=(1, 16), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(1, 64), stride=(1, 64)),  # → (B,64,1,260)
        )


        # ---------- 2. 2-D CNN trunk (5 conv-pairs) -------------------------
        # helper to build a "conv-conv-pool" mini-block
        def _block(in_ch, out_ch, k1, k2, pool_ks, pool_stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=k1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=k2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool_ks, stride=pool_stride),
            )

        self.trunk = nn.Sequential(
            _block(1,   32, (8, 8), (8, 8), (5, 3), (5, 3)),   # conv3-4
            _block(32,  64, (1, 4), (1, 4), (1, 2), (1, 2)),   # conv5-6
            _block(64, 128, (1, 2), (1, 2), (1, 2), (1, 2)),   # conv7-8
            _block(128,256, (1, 2), (1, 2), (1, 2), (1, 2)),   # conv9-10
        )
        # After conv10-pool: feature map size → (B, 256, 10, 33)

        # ---------- 3. Classifier head -------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 10 * 33, 4096),  # Updated to match actual trunk output: 256*10*33=84480
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(4096, num_classes),
        )

        self._init_weights()


        # # 1-D conv front-end over raw waveform (kernel=64, stride=2)
        # self.conv1d = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=64, stride=2, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(8, stride=8),
        #     nn.Conv1d(64, 64, kernel_size=16, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(4, stride=4),
        # )

        # # Collapse time dim → treat as pseudo-image (C, T) → (1, C, T)
        # self.conv2d = nn.Sequential(
        #     nn.Conv2d(1, 64, (1, 3), bias=False),  # Use (1, 3) for height=1 input
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d((1, 2)),  # Pool only in width dimension
        #     nn.Conv2d(64, 128, (1, 3), bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d((1, 2)),
        #     nn.Conv2d(128, 256, (1, 3), bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d((1, 2)),
        # )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(256, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, num_classes),
        # )
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, 1, T) or (B, 1, 1, T)
        if x.ndim == 3:
            x = x.unsqueeze(2)          # → (B,1,1,T)

        x = self.frontend(x)            # temporal convs + pool
        x = x.transpose(1, 2)           # swap to (B,1,64,260)

        x = self.trunk(x)               # 2-D conv blocks
        return self.classifier(x)       # FC head → logits


    def replace_head(self, num_classes: int) -> None:
        in_feat = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_feat, num_classes)
