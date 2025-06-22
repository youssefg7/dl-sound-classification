import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvNetV2(nn.Module):
    """
    Minimal EnvNet-v2 for 44.1 kHz mono waveform --> 1 x 44100 x 5s tensor. Layer widths follow Tokozume 2018.
    """

    def __init__(self, num_classes: int = 50, dropout: float = 0.5):
        super().__init__()

        # 1-D conv front-end over raw waveform (kernel=64, stride=2)
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(8, stride=8),
            nn.Conv1d(64, 64, kernel_size=16, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),
        )

        # Collapse time dim → treat as pseudo-image (C, T) → (1, C, T)
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3), bias=False),  # Use (1, 3) for height=1 input
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),  # Pool only in width dimension
            nn.Conv2d(64, 128, (1, 3), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(128, 256, (1, 3), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, T)
        x = self.conv1d(x)  # (B, 64, T')

        # Reshape for 2D conv: treat channels as height, time as width
        B, C, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, T)

        # Apply 2D convolutions
        x = self.conv2d(x)  # (B, 256, H', W')

        # Global average pooling to handle any remaining spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 256)

        return self.classifier(x)

    def replace_head(self, num_classes: int) -> None:
        in_feat = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_feat, num_classes)
