"""
ESC-50 Dataset loader + Lightning DataModule

* Follows the official five-fold CV split (Piczak 2015) - no reshuffle. 1 fold = test, remaining 4 folds = train; a stratified ``val_split``% of *train* is taken for validation.
* Requires cached ``.pt`` tensors created by ``scripts/prepare_esc50.py``.
* Returns **log-Mel spectrograms** suited for CNN input.

Folders expected:
data/processed/esc50/
 ├─ fold_0/ clipA.pt …
 ├─ fold_1/
 ├─ fold_2/
 ├─ fold_3/
 └─ fold_4/
     dataset_stats.json
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import lightning.pytorch as pl

from ..utils.audio import melspectrogram, SpecAugment

# --------------------------------------------------------------------------- #
# Low-level dataset
# --------------------------------------------------------------------------- #
class ESC50Dataset(Dataset):
    """Loads cached tensors and converts to log-Mel spectrogram."""

    def __init__(
        self,
        root: str | Path,
        folds: Sequence[int],
        sample_rate: int = 44_100,
        n_mels: int = 128,
        n_fft: int = 1_024,
        hop_length: int = 512,
        augment: Dict | None = None,
    ) -> None:
        self.root = Path(root)
        self.files: List[Path] = []
        for f in folds:
            self.files += sorted((self.root / f"fold_{f}").glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {self.root}; "
                "did you run scripts/prepare_esc50.py?"
            )

        # Store feature-extraction params
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Set up augmentation
        aug_cfg = augment or {}
        if aug_cfg.get("time_mask") or aug_cfg.get("freq_mask"):
            self.augment = SpecAugment(
                time_mask=aug_cfg.get("time_mask", 80),
                freq_mask=aug_cfg.get("freq_mask", 32),
            )
        else:
            self.augment = None


    # -------------------------------------------- #
    def __len__(self) -> int:
        return len(self.files)

    # -------------------------------------------- #
    def __getitem__(self, idx: int):
        bundle = torch.load(self.files[idx])
        wave: torch.Tensor = bundle["waveform"]  # shape (1, samples)
        label: int = bundle["label"]

        # For EnvNetV2, return raw waveform instead of spectrogram
        # The model expects (1, samples) which is what we have
        return wave, label


# --------------------------------------------------------------------------- #
# Lightning-style DataModule
# --------------------------------------------------------------------------- #

class ESC50DataModule(pl.LightningDataModule):
    """
    Hydrable data module.

    Parameters (all exposed via Hydra config):
    • root: processed dataset dir
    • fold: int in 0-4  → held-out test fold
    • val_split: float 0-1 – share of *train* used for validation
    • batch_size / num_workers : passed to DataLoader
    • augment: dict with {time_mask: bool, freq_mask: bool}
    """

    def __init__(
        self,
        root: str,
        fold: int = 0,
        sample_rate: int = 44_100,
        n_mels: int = 128,
        val_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: Dict | None = None,
    ) -> None:
        super().__init__()
        if not (0 <= fold <= 4):
            raise ValueError("fold must be 0…4 (ESC-50 uses five folds).")
        self.save_hyperparameters(logger=False)

        self._train_set: Dataset | None = None
        self._val_set: Dataset | None = None
        self._test_set: Dataset | None = None

    # -------------------------------------------- #
    def setup(self, stage: str | None = None) -> None:
        root = Path(self.hparams.root)

        test_fold = self.hparams.fold
        train_folds = [f for f in range(5) if f != test_fold]

        # 1) full train (4 folds) – will be split below
        full_train = ESC50Dataset(
            root=root,
            folds=train_folds,
            sample_rate=self.hparams.sample_rate,
            n_mels=self.hparams.n_mels,
            augment=self.hparams.augment,
        )

        # 2) deterministic train/val split (stratified enough because folds
        #    are already balanced). We simply slice by index here.
        val_size = math.ceil(len(full_train) * self.hparams.val_split)
        idxs = list(range(len(full_train)))
        random.Random(42).shuffle(idxs)  # reproducible

        val_idxs = idxs[:val_size]
        train_idxs = idxs[val_size:]

        self._train_set = Subset(full_train, train_idxs)
        self._val_set = Subset(full_train, val_idxs)

        # 3) test set = held-out fold
        self._test_set = ESC50Dataset(
            root=root,
            folds=[test_fold],
            sample_rate=self.hparams.sample_rate,
            n_mels=self.hparams.n_mels,
            augment=None,  # NEVER augment evaluation
        )

    # -------------------------------------------- #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
