import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaborConv1d(nn.Module):
    def __init__(self, n_filters=40, kernel_size=401, sample_rate=44100, min_freq=60.0, max_freq=7800.0):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Learnable filter parameters
        self.center_freqs = nn.Parameter(torch.linspace(min_freq, max_freq, n_filters) / (sample_rate / 2))
        self.bandwidths = nn.Parameter(torch.full((n_filters,), 1.0))
        self.register_buffer("window", torch.hann_window(kernel_size).unsqueeze(0))

    def forward(self, x):
        device = x.device
        t = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=device) / self.sample_rate
        t = t.view(1, 1, -1)

        center_freqs = self.center_freqs.view(-1, 1, 1)
        bandwidths = self.bandwidths.view(-1, 1, 1)

        envelope = torch.exp(-0.5 * (t * bandwidths * self.sample_rate) ** 2)
        real = torch.cos(2 * math.pi * center_freqs * t) * envelope * self.window.to(device)
        imag = torch.sin(2 * math.pi * center_freqs * t) * envelope * self.window.to(device)

        x_real = F.conv1d(x, real, padding=self.kernel_size // 2)
        x_imag = F.conv1d(x, imag, padding=self.kernel_size // 2)

        return x_real**2 + x_imag**2


class PCEN(nn.Module):
    def __init__(self, num_channels, alpha=0.98, delta=2.0, r=0.5, eps=1e-6, s=0.04):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.full((num_channels,), alpha))
        self.delta = nn.Parameter(torch.full((num_channels,), delta))
        self.r = nn.Parameter(torch.full((num_channels,), r))
        self.s = s

    def forward(self, x):
        # Apply EMA smoothing along time
        M = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        pcen = ((x / (self.eps + M) ** self.r.view(1, -1, 1)) + self.delta.view(1, -1, 1)).log()
        return pcen


class LeafModel(nn.Module):
    def __init__(self, n_filters=40, kernel_size=401, sample_rate=44100, num_classes=50):
        super().__init__()
        self.gabor = GaborConv1d(n_filters=n_filters, kernel_size=kernel_size, sample_rate=sample_rate)
        self.pcen = PCEN(n_filters)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Output shape: [B, C, 1]
        self.classifier = nn.Linear(n_filters, num_classes)

    def forward(self, x):
        x = self.gabor(x)          # [B, C, T]
        x = self.pcen(x)           # [B, C, T]
        x = self.pooling(x)        # [B, C, 1]
        x = x.squeeze(-1)          # [B, C]
        logits = self.classifier(x)  # [B, num_classes]
        return logits
