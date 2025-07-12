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
from typing import Callable, Dict, List, Sequence, Any, Tuple, Optional, Union

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from ..utils.audio import SpecAugment, melspectrogram
    from .preprocessing import create_preprocessor, BCMixingDataset, EnvNetPreprocessor, ASTPreprocessor, MixupAugmentation, create_one_hot_labels, AST_HOP_LENGTH, AST_N_FFT, create_ast_fallback_spectrogram, load_audio_bundle
except ImportError:
    # Handle direct script execution
    from utils.audio import SpecAugment, melspectrogram
    from preprocessing import create_preprocessor, BCMixingDataset, EnvNetPreprocessor, ASTPreprocessor, MixupAugmentation, create_one_hot_labels, AST_HOP_LENGTH, AST_N_FFT, create_ast_fallback_spectrogram, load_audio_bundle


# --------------------------------------------------------------------------- #
# Low-level dataset
# --------------------------------------------------------------------------- #
class MixupDataset:
    """Mixin class for datasets that support Mixup augmentation."""
    
    def __init__(self, enable_mixup: bool = False, mixup_alpha: float = 0.5, num_classes: int = 50):
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.mixup_augmentation = MixupAugmentation(alpha=mixup_alpha, prob=0.5) if enable_mixup else None
    
    def apply_mixup(self, spec: torch.Tensor, label: int, all_data: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixup augmentation to a spectrogram sample.
        
        Args:
            spec: Current spectrogram
            label: Current label
            all_data: List of all (spectrogram, label) pairs for sampling
            
        Returns:
            Tuple of (mixed_spectrogram, soft_labels)
        """
        if not self.enable_mixup or not self.mixup_augmentation or random.random() > 0.5:
            # Return one-hot labels for non-mixed samples
            soft_labels = create_one_hot_labels(label, self.num_classes)
            return spec, soft_labels
        
        # Sample a different item
        other_idx = random.randint(0, len(all_data) - 1)
        other_spec, other_label = all_data[other_idx]
        
        # Apply mixup
        mixed_spec, soft_labels = self.mixup_augmentation(spec, other_spec, label, other_label, self.num_classes)
        
        return mixed_spec, soft_labels


class ESC50Dataset(Dataset, BCMixingDataset, MixupDataset):
    """Loads cached tensors with model-specific preprocessing."""

    def __init__(
        self,
        root: str | Path,
        folds: Sequence[int],
        sample_rate: int = 44_100,
        n_mels: int = 128,
        n_fft: int = 1_024,
        hop_length: int = 512,
        augment: Dict | None = None,
        preprocessing_mode: str = "envnet_v2",
        preprocessing_config: Dict | None = None,
        enable_bc_mixing: bool = False,
        enable_mixup: bool = False,
        mixup_alpha: float = 0.5,
        num_classes: int = 50,
        training: bool = True,
    ) -> None:
        # Initialize augmentation capabilities
        BCMixingDataset.__init__(self, enable_bc_mixing=enable_bc_mixing, num_classes=num_classes)
        MixupDataset.__init__(self, enable_mixup=enable_mixup, mixup_alpha=mixup_alpha, num_classes=num_classes)
        

        
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
        self.training = training
        self.preprocessing_mode = preprocessing_mode

        # Set up legacy augmentation (for AST mode)
        aug_cfg = augment or {}
        if aug_cfg.get("time_mask") or aug_cfg.get("freq_mask"):
            self.augment = SpecAugment(
                time_mask=aug_cfg.get("time_mask", 80),
                freq_mask=aug_cfg.get("freq_mask", 32),
            )
        else:
            self.augment = None

        # Set up model-specific preprocessor
        self.preprocessor = None
        if preprocessing_config:
            # Merge default config with user config
            default_config = {
                'sample_rate': sample_rate,
                'n_mels': n_mels,
                'bc_mixing': enable_bc_mixing,
            }
            # Convert DictConfig to regular dict to avoid JSON serialization issues
            if hasattr(preprocessing_config, '_content'):
                preprocessing_config = dict(preprocessing_config)
            default_config.update(preprocessing_config)
            
            try:
                cache_dir = self.root.parent / "cache"
                self.preprocessor = create_preprocessor(
                    mode=preprocessing_mode,
                    config_dict=default_config,
                    base_cache_dir=cache_dir,
                    force_rebuild=False
                )
                print(f"✓ Using {preprocessing_mode} preprocessor with caching")
            except Exception as e:
                print(f"⚠ Failed to create preprocessor: {e}, falling back to basic mode")
                self.preprocessor = None

        # Initialize lazy loading for augmentation (memory optimized)
        self._cached_data: Optional[List[Tuple[torch.Tensor, int]]] = None
        self._use_lazy_loading = True
        
        # Pre-load data for augmentation only if needed, small dataset, and training mode
        if (self.enable_bc_mixing or self.enable_mixup) and len(self.files) <= 2000 and training:
            augmentation_type = []
            if self.enable_bc_mixing:
                augmentation_type.append("BC mixing")
            if self.enable_mixup:
                augmentation_type.append("Mixup")
            
            print(f"Pre-loading data for {' and '.join(augmentation_type)}...")
            
            # Load appropriate data based on preprocessing mode
            if self.preprocessing_mode == "ast" and self.enable_mixup:
                # For AST with Mixup, load spectrograms
                self._cached_data = self._load_spectrograms_for_mixup(self.files, show_progress=True)
            else:
                # For EnvNet-v2 with BC mixing, load waveforms
                self._cached_data = self._load_data_for_mixing(self.files, show_progress=True)
            
            if self._cached_data:
                print(f"✓ Pre-loaded {len(self._cached_data)} samples for augmentation")
            else:
                print("⚠ Failed to pre-load data for augmentation")
        elif self.enable_bc_mixing or self.enable_mixup:
            if training:
                print(f"✓ Using lazy loading for augmentation (large dataset: {len(self.files)} files)")
            self._use_lazy_loading = True

    # -------------------------------------------- #
    def __len__(self) -> int:
        return len(self.files)

    # -------------------------------------------- #
    def __getitem__(self, idx: int):
        if self.enable_bc_mixing and self._cached_data:
            # Use cached data for BC mixing
            wave, label = self._cached_data[idx]
        else:
            # Load data normally
            bundle = torch.load(self.files[idx])
            wave: torch.Tensor = bundle["waveform"]  # shape (1, samples)
            label: int = bundle["label"]

        # Apply model-specific preprocessing
        if self.preprocessing_mode == "envnet_v2":
            return self._process_envnet_v2(wave, label, idx)
        elif self.preprocessing_mode == "ast":
            return self._process_ast(wave, label, idx)
        else:
            # Default: return raw waveform
            return wave, label

    def _process_envnet_v2(self, wave: torch.Tensor, label: int, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process sample for EnvNet-v2."""
        # Apply basic preprocessing if not already cached
        if self.preprocessor and not self.enable_bc_mixing:
            wave = self.preprocessor.preprocess(wave, self.sample_rate)
        
        # Apply random cropping and augmentation
        if isinstance(self.preprocessor, EnvNetPreprocessor):
            wave = self.preprocessor.random_crop(wave, training=self.training)
            
            if self.training:
                wave = self.preprocessor.apply_augmentation(wave)
        
        # Apply BC mixing if enabled
        if self.enable_bc_mixing and self.training:
            # Load data for mixing if not already cached
            if self._cached_data is None:
                self._cached_data = self._load_data_for_mixing(self.files, show_progress=False)
            
            wave, soft_labels = self.apply_bc_mixing(wave, label, self._cached_data, self.sample_rate)
            return wave, soft_labels
        else:
            # Return one-hot labels for non-mixed samples
            soft_labels = create_one_hot_labels(label, self.num_classes)
            return wave, soft_labels

    def _process_ast(self, wave: torch.Tensor, label: int, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process sample for AST."""
        # Apply AST preprocessing (spectrogram conversion)
        if self.preprocessor and not (self.enable_mixup and self._cached_data):
            spec = self.preprocessor.preprocess(wave, self.sample_rate)
        elif self.enable_mixup and self._cached_data:
            # Use cached spectrogram for mixup
            spec, _ = self._cached_data[idx]
        else:
            # Fallback to basic log-mel spectrogram (already log-scaled)
            # Use AST-specific parameters to match model expectations
            spec = create_ast_fallback_spectrogram(wave, self.sample_rate, self.n_mels)
            
        # Apply SpecAugment if configured (before mixup)
        if isinstance(self.preprocessor, ASTPreprocessor) and self.training:
            # Get SpecAugment parameters from config - use default AST values
            if self.augment:
                # Extract parameter values from the SpecAugment transform objects
                time_mask_param = getattr(self.augment.time_mask, 'time_mask_param', 192) if hasattr(self.augment, 'time_mask') else 192
                freq_mask_param = getattr(self.augment.freq_mask, 'freq_mask_param', 48) if hasattr(self.augment, 'freq_mask') else 48
                spec = self.preprocessor.apply_specaugment(spec, time_mask=time_mask_param, freq_mask=freq_mask_param)
        elif self.augment and self.training:
            # Legacy SpecAugment
            spec = self.augment(spec)
            
        # Apply Mixup if enabled
        if self.enable_mixup and self.training:
            # Load data for mixing if not already cached
            if self._cached_data is None:
                self._cached_data = self._load_spectrograms_for_mixup(self.files, show_progress=False)
            
            mixed_spec, soft_labels = self.apply_mixup(spec, label, self._cached_data)
            return mixed_spec, soft_labels
        else:
            # Return one-hot labels for non-mixed samples
            soft_labels = create_one_hot_labels(label, self.num_classes)
            return spec, soft_labels

    def set_training(self, training: bool) -> None:
        """Set training mode for the dataset."""
        self.training = training
    
    def _load_spectrograms_for_mixup(self, file_paths: List[Path], show_progress: bool = True) -> List[Tuple[torch.Tensor, int]]:
        """Load spectrograms for Mixup augmentation with AST preprocessing."""
        data = []
        desc = "Loading spectrograms for Mixup"
        
        if show_progress:
            from tqdm import tqdm
            progress_bar = tqdm(file_paths, desc=desc, unit="files")
            file_iterator = progress_bar
        else:
            file_iterator = file_paths
        
        for path in file_iterator:
            audio_data = load_audio_bundle(path)
            if audio_data is None:
                continue
                
            waveform, label = audio_data
            
            # Convert to spectrogram using AST parameters
            if self.preprocessor:
                spec = self.preprocessor.preprocess(waveform, self.sample_rate)
            else:
                # Fallback to basic log-mel spectrogram with AST parameters
                spec = create_ast_fallback_spectrogram(waveform, self.sample_rate, self.n_mels)
            
            data.append((spec, label))
        
        if show_progress:
            progress_bar.close()
        
        return data


# --------------------------------------------------------------------------- #
# Lightning-style DataModule
# --------------------------------------------------------------------------- #


class ESC50DataModule(pl.LightningDataModule):
    """
    Hydrable data module with consolidated configuration support.

    Parameters (all exposed via Hydra config):
    • root: processed dataset dir
    • fold: int in 0-4  → held-out test fold
    • val_split: float 0-1 – share of *train* used for validation
    • batch_size / num_workers : passed to DataLoader
    • is_spectrogram: bool - true for AST mode (spectrograms), false for EnvNet-v2 mode (waveforms)
    • enable_bc_mixing: bool for Between-Class mixing (only when is_spectrogram=false)
    • enable_mixup: bool for Mixup augmentation (only when is_spectrogram=true)
    • time_mask, freq_mask: SpecAugment parameters (only when is_spectrogram=true)
    • preprocessing_mode: automatically set based on is_spectrogram
    • preprocessing_config: dict with preprocessing parameters
    • num_classes: number of classes for soft labels
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
        is_spectrogram: bool = False,
        enable_bc_mixing: bool = False,
        enable_mixup: bool = False,
        mixup_alpha: float = 0.5,
        time_mask: Union[bool, int] = False,
        freq_mask: Union[bool, int] = False,
        preprocessing_mode: str = "envnet_v2",  # Will be overridden by is_spectrogram
        preprocessing_config: Dict | None = None,
        num_classes: int = 50,
        # Legacy parameters for backward compatibility
        augment: Dict | None = None,
    ) -> None:
        super().__init__()
        if not (0 <= fold <= 4):
            raise ValueError("fold must be 0…4 (ESC-50 uses five folds).")
        
        # =============================================================================
        # CONSTRAINT VALIDATION
        # =============================================================================
        self._validate_config_constraints(
            is_spectrogram=is_spectrogram,
            enable_bc_mixing=enable_bc_mixing,
            enable_mixup=enable_mixup,
            time_mask=time_mask,
            freq_mask=freq_mask
        )
        
        # =============================================================================
        # AUTOMATIC PARAMETER DERIVATION
        # =============================================================================
        
        # Override preprocessing_mode based on is_spectrogram
        derived_preprocessing_mode = "ast" if is_spectrogram else "envnet_v2"
        
        # Build augment dict from time_mask and freq_mask
        if augment is None:
            augment = {}
        
        # Override augment with time_mask and freq_mask if they are set
        if time_mask is not False:
            augment['time_mask'] = time_mask
        if freq_mask is not False:
            augment['freq_mask'] = freq_mask
        
        # =============================================================================
        # STORE VALIDATED PARAMETERS
        # =============================================================================
        self.save_hyperparameters(logger=False)

        # Store hyperparameters directly for type safety
        self.root = root
        self.fold = fold
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_spectrogram = is_spectrogram
        self.enable_bc_mixing = enable_bc_mixing
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.augment = augment
        self.preprocessing_mode = derived_preprocessing_mode
        self.preprocessing_config = preprocessing_config
        self.num_classes = num_classes

        self._train_set: Dataset | None = None
        self._val_set: Dataset | None = None
        self._test_set: Dataset | None = None
        
        # Print configuration summary
        self._print_config_summary()

    def _validate_config_constraints(
        self, 
        is_spectrogram: bool, 
        enable_bc_mixing: bool, 
        enable_mixup: bool, 
        time_mask: Union[bool, int], 
        freq_mask: Union[bool, int]
    ) -> None:
        """Validate configuration constraints."""
        errors = []
        
        # Constraint 1: BC mixing cannot be true with spectrograms
        if is_spectrogram and enable_bc_mixing:
            errors.append("enable_bc_mixing cannot be true when is_spectrogram=true (BC mixing is only for waveform mode)")
        
        # Constraint 2: Mixup can only be true with spectrograms
        if not is_spectrogram and enable_mixup:
            errors.append("enable_mixup can only be true when is_spectrogram=true (Mixup is only for spectrogram mode)")
        
        # Constraint 3: time_mask and freq_mask only considered when is_spectrogram=true
        if not is_spectrogram:
            if time_mask is not False and time_mask != 0:
                errors.append("time_mask will be ignored when is_spectrogram=false (SpecAugment is only for spectrogram mode)")
            if freq_mask is not False and freq_mask != 0:
                errors.append("freq_mask will be ignored when is_spectrogram=false (SpecAugment is only for spectrogram mode)")
        
        # Additional validation: time_mask and freq_mask should be positive integers if not False
        if is_spectrogram:
            if time_mask is not False and not isinstance(time_mask, int):
                errors.append("time_mask must be False or a positive integer")
            if freq_mask is not False and not isinstance(freq_mask, int):
                errors.append("freq_mask must be False or a positive integer")
            if isinstance(time_mask, int) and time_mask < 0:
                errors.append("time_mask must be a positive integer")
            if isinstance(freq_mask, int) and freq_mask < 0:
                errors.append("freq_mask must be a positive integer")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  • {error}" for error in errors)
            raise ValueError(error_msg)

    def _print_config_summary(self) -> None:
        """Print a summary of the configuration."""
        mode = "🎵 Spectrogram (AST)" if self.is_spectrogram else "🌊 Waveform (EnvNet-v2)"
        print(f"\n📊 ESC-50 Dataset Configuration")
        print(f"   Mode: {mode}")
        print(f"   Fold: {self.fold} (test), others (train/val)")
        print(f"   Val split: {self.val_split}")
        print(f"   Batch size: {self.batch_size}")
        
        if self.is_spectrogram:
            print(f"   🎵 Spectrogram Settings:")
            print(f"      • Mixup: {'✓' if self.enable_mixup else '✗'}")
            if self.enable_mixup:
                print(f"      • Mixup alpha: {self.mixup_alpha}")
            print(f"      • Time mask: {self.time_mask}")
            print(f"      • Freq mask: {self.freq_mask}")
        else:
            print(f"   🌊 Waveform Settings:")
            print(f"      • BC mixing: {'✓' if self.enable_bc_mixing else '✗'}")
        
        print()

    # -------------------------------------------- #
    def setup(self, stage: str | None = None) -> None:
        # Skip setup if datasets are already created
        if self._train_set is not None and self._val_set is not None and self._test_set is not None:
            return
            
        root = Path(self.root)

        test_fold = self.fold
        train_folds = [f for f in range(5) if f != test_fold]

        # 1) full train (4 folds) – will be split below
        full_train = ESC50Dataset(
            root=root,
            folds=train_folds,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            augment=self.augment,
            preprocessing_mode=self.preprocessing_mode,
            preprocessing_config=self.preprocessing_config,
            enable_bc_mixing=self.enable_bc_mixing,
            enable_mixup=self.enable_mixup,
            mixup_alpha=self.mixup_alpha,
            num_classes=self.num_classes,
            training=True,
        )

        # 2) deterministic train/val split (stratified enough because folds
        #    are already balanced). We simply slice by index here.
        val_size = math.ceil(len(full_train) * self.val_split)
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
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            augment=None,  # NEVER augment evaluation
            preprocessing_mode=self.preprocessing_mode,
            preprocessing_config=self.preprocessing_config,
            enable_bc_mixing=False,  # NEVER mix during evaluation
            enable_mixup=False,  # NEVER mix during evaluation
            mixup_alpha=self.mixup_alpha,
            num_classes=self.num_classes,
            training=False,
        )

    # -------------------------------------------- #
    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            raise RuntimeError("Dataset not set up. Call setup() first.")
        return DataLoader(
            self._train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            raise RuntimeError("Dataset not set up. Call setup() first.")
        return DataLoader(
            self._val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            raise RuntimeError("Dataset not set up. Call setup() first.")
        return DataLoader(
            self._test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
