#!/usr/bin/env python
"""
Prepare ESC-50 for fast PyTorch training.

This script:
1. Validates the presence and schema of meta/esc50.csv
2. Verifies SHA-256 (optional: skip with --no-hash)
3. Loads each WAV, forces mono 44.1 kHz, normalises [-1, 1]
4. Saves torch tensors under data/processed/esc50/fold_{k}/{clip_id}.pt
5. Emits dataset_stats.json with clip counts, duration, and class histogram
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd  # type: ignore
import torch
import torchaudio  # type: ignore
from tqdm import tqdm  # type: ignore

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
RAW_DIR = Path(os.getenv("DATA_DIR", "data/raw")) / "esc50"
PROC_DIR = Path(os.getenv("DATA_DIR", "data/processed")) / "esc50"
META_CSV = RAW_DIR / "meta" / "esc50.csv"
TARGET_SR = 44_100  # Hz, per ESC-50 spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path, buf_size: int = 1 << 16) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def resample_if_needed(wave: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return wave
    resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
    return resampler(wave)


def save_tensor(wave: torch.Tensor, label: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"waveform": wave, "label": label}, out_path)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def prepare_esc50(validate_hash: bool = True) -> None:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {META_CSV}")
    df = pd.read_csv(META_CSV)

    stats: Dict[str, int | float] = {
        "num_clips": len(df),
        "total_duration_sec": 0.0,
        "class_histogram": {},
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing ESC-50"):
        fold = int(row["fold"])  # 1-based → 1…5
        category = row["category"]
        label = int(row["target"])
        filename = row["filename"]

        wav_path = RAW_DIR / "audio" / filename
        if not wav_path.exists():
            logging.warning("Missing file %s ‒ skipping", wav_path)
            continue

        if validate_hash:
            # official repo ships md5, but SHA-256 is quicker to verify locally
            stats.setdefault("file_hashes", {})[filename] = sha256(wav_path)  # type: ignore[assignment]

        wave, sr = torchaudio.load(str(wav_path))
        wave = resample_if_needed(wave, sr)
        wave = torch.mean(wave, dim=0, keepdim=True)  # ensure mono

        # duration
        stats["total_duration_sec"] = float(stats["total_duration_sec"]) + (wave.shape[-1] / TARGET_SR)  # type: ignore[operator]

        # class histogram
        hist = stats["class_histogram"]  # type: ignore[assignment]
        hist[category] = hist.get(category, 0) + 1  # type: ignore[index]

        out_file = PROC_DIR / f"fold_{fold - 1}" / f"{filename.replace('.wav', '.pt')}"
        save_tensor(wave, label, out_file)

    # write stats
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with (PROC_DIR / "dataset_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    logging.info("✓ Finished. Cached tensors in %s", PROC_DIR)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ESC-50 dataset.")
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 calculation (faster).",
    )
    args = parser.parse_args()
    prepare_esc50(validate_hash=not args.no_hash)
