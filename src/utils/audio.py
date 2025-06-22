"""
Reusable DSP helpers for environmental-sound projects.

* TorchScript-friendly (no global state, no lambdas).
* Follows 44.1 kHz mono convention of ESC-50.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
from torchaudio import transforms as T

# ---------------------------------------------------------------------------- #
# Constants                                                                    #
# ---------------------------------------------------------------------------- #
TARGET_SR = 44_100  # default sample-rate for ESC-50
EPSILON = 1e-9  # avoid log(0)


# ---------------------------------------------------------------------------- #
# Basic I/O                                                                    #
# ---------------------------------------------------------------------------- #
def load_waveform(
    path: str | Path,
    target_sr: int = TARGET_SR,
    mono: bool = True,
    peak_norm: bool = True,
) -> torch.Tensor:
    """
    Load an audio file → (1, samples) float32 tensor in [-1, 1].

    * Resamples on the fly if sample-rate ≠ target_sr.
    * Converts stereo → mono by channel-mean.
    * Optional **peak normalisation** (common in ESC baselines).
    """
    wav, sr = torchaudio.load(str(path))  # float32 / [-1, 1]

    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    if mono and wav.shape[0] > 1:  # (chan, time)
        wav = torch.mean(wav, dim=0, keepdim=True)

    if peak_norm:
        peak = wav.abs().max().clamp_min(EPSILON)
        wav = wav / peak

    return wav


# ---------------------------------------------------------------------------- #
# Time-frequency features                                                      #
# ---------------------------------------------------------------------------- #
def melspectrogram(
    wav: torch.Tensor,
    sr: int = TARGET_SR,
    n_mels: int = 128,
    n_fft: int = 1_024,
    hop_length: int = 512,
    log_scale: bool = True,
) -> torch.Tensor:
    """
    Waveform → (1, n_mels, frames) tensor (dB if log_scale).

    Uses torchaudio’s `MelSpectrogram` + `AmplitudeToDB`.
    """
    spec = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
    )(wav)

    if log_scale:
        spec = T.AmplitudeToDB(top_db=80)(spec)

    return spec


# ---------------------------------------------------------------------------- #
# SpecAugment (time / frequency masking)                                       #
# ---------------------------------------------------------------------------- #
class SpecAugment(torch.nn.Module):
    """
    Minimal SpecAugment block (time + freq masking).
    """

    def __init__(self, time_mask: int = 80, freq_mask: int = 32):
        super().__init__()
        self.time_mask = T.TimeMasking(time_mask_param=time_mask)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        spec = self.time_mask(spec)
        spec = self.freq_mask(spec)
        return spec


# ---------------------------------------------------------------------------- #
# Between-Class (BC) learning mix helper                                       #
# ---------------------------------------------------------------------------- #
def bc_mix(
    wav_a: torch.Tensor,
    wav_b: torch.Tensor,
    *,  # keyword-only for clarity
    ratio: float | None = None,
) -> Tuple[torch.Tensor, float]:
    """
    Mix two waveforms per **Between-Class learning** (Tokozume et al.).

    Returns:
        mixed_wave: (1, time) tensor
        lambda_:    mix ratio applied to wav_a (target is [λ, 1-λ])
    """
    assert wav_a.shape == wav_b.shape
    ratio = random.random() if ratio is None else ratio
    mixed = ratio * wav_a + (1.0 - ratio) * wav_b
    # peak-normalise to avoid clipping
    peak = mixed.abs().max().clamp_min(EPSILON)
    mixed = mixed / peak
    return mixed, ratio


# ---------------------------------------------------------------------------- #
# Utility: pad / crop to fixed length                                          #
# ---------------------------------------------------------------------------- #
def pad_or_trim(
    wav: torch.Tensor,
    length_samples: int,
) -> torch.Tensor:
    """
    Make waveform exactly `length_samples` long (pad wrap or trim).
    Useful for batching models that need fixed-length input.
    """
    cur_len = wav.shape[-1]
    if cur_len == length_samples:
        return wav
    if cur_len < length_samples:
        # Wrap-pad (better than zero pad for ESC-50 short clips)
        repeat = math.ceil(length_samples / cur_len)
        wav = wav.repeat(1, repeat)[:, :length_samples]
    else:  # trim centre
        start = (cur_len - length_samples) // 2
        wav = wav[:, start : start + length_samples]
    return wav
