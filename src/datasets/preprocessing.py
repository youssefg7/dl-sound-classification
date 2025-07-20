"""
Model-specific preprocessing architecture for audio datasets.

This module provides a flexible preprocessing pipeline that supports different
model requirements with efficient caching mechanisms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
import pickle
import gzip
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torchaudio
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)


def create_one_hot_labels(label: int, num_classes: int) -> torch.Tensor:
    """
    Create one-hot encoded soft labels.
    
    Args:
        label: Class label index
        num_classes: Total number of classes
        
    Returns:
        One-hot encoded tensor of shape [num_classes]
    """
    soft_labels = torch.zeros(num_classes, dtype=torch.float32)
    soft_labels[label] = 1.0
    return soft_labels


# AST Model Constants (Audio Spectrogram Transformer)
AST_HOP_LENGTH = 160    # ~3.6ms at 44.1kHz
AST_N_FFT = 1024        # AST model expects n_fft=1024
AST_WIN_LENGTH = 400    # ~9.1ms at 44.1kHz


def resample_waveform(waveform: torch.Tensor, current_rate: int, target_rate: int) -> torch.Tensor:
    """
    Resample waveform to target sample rate if needed.
    
    Args:
        waveform: Input waveform tensor
        current_rate: Current sample rate
        target_rate: Target sample rate
        
    Returns:
        Resampled waveform or original if rates match
    """
    if current_rate != target_rate:
        resampler = torchaudio.transforms.Resample(current_rate, target_rate)
        return resampler(waveform)
    return waveform


def create_ast_fallback_spectrogram(waveform: torch.Tensor, sample_rate: int, n_mels: int = 128) -> torch.Tensor:
    """
    Create AST-compatible log-mel spectrogram using fallback parameters.
    
    Args:
        waveform: Input waveform tensor
        sample_rate: Sample rate of the waveform
        n_mels: Number of mel filter banks
        
    Returns:
        Log-mel spectrogram tensor compatible with AST model
    """
    # Import here to avoid circular imports
    try:
        from ..utils.audio import melspectrogram
    except ImportError:
        from utils.audio import melspectrogram
    
    return melspectrogram(waveform, sample_rate, n_mels, AST_N_FFT, AST_HOP_LENGTH, log_scale=True)


def load_audio_bundle(file_path: Path) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Load audio bundle from file with consistent error handling.
    
    Args:
        file_path: Path to the .pt file containing audio data
        
    Returns:
        Tuple of (waveform, label) or None if loading failed
    """
    try:
        bundle = torch.load(file_path, map_location='cpu')
        waveform = bundle["waveform"]
        label = bundle["label"]
        return waveform, label
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


@dataclass
class CacheStats:
    """Statistics for cache operations."""
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size_mb: float = 0.0
    total_files: int = 0
    compression_ratio: float = 0.0
    avg_load_time_ms: float = 0.0
    avg_save_time_ms: float = 0.0
    last_cleanup_time: Optional[float] = None
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size_mb': self.cache_size_mb,
            'total_files': self.total_files,
            'compression_ratio': self.compression_ratio,
            'avg_load_time_ms': self.avg_load_time_ms,
            'avg_save_time_ms': self.avg_save_time_ms,
            'hit_rate': self.hit_rate(),
            'last_cleanup_time': self.last_cleanup_time
        }


class AdvancedCacheManager:
    """Advanced cache management with compression, invalidation, and optimization."""
    
    def __init__(self, base_cache_dir: Path, max_cache_size_gb: float = 5.0):
        self.base_cache_dir = Path(base_cache_dir)
        self.max_cache_size_gb = max_cache_size_gb
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = CacheStats()
        self.stats_lock = threading.Lock()
        self.load_times = []
        self.save_times = []
        
        # Cache metadata
        self.metadata_path = self.base_cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        # Performance monitoring
        self.performance_log = []
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Invalid cache metadata, creating new")
        
        return {
            'version': '1.0',
            'created_time': time.time(),
            'file_metadata': {},
            'cache_stats': {}
        }
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        self.metadata['cache_stats'] = self.stats.to_dict()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get file hash including modification time."""
        try:
            stat = file_path.stat()
            content = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()[:12]
        except OSError:
            return hashlib.md5(str(file_path).encode()).hexdigest()[:12]
    
    def _get_cache_path(self, original_path: Path, config_hash: str) -> Path:
        """Get cache file path with compression."""
        file_hash = self._get_file_hash(original_path)
        cache_name = f"{original_path.stem}_{file_hash}_{config_hash}.cache.gz"
        return self.base_cache_dir / cache_name
    
    def _compress_and_save(self, data: torch.Tensor, cache_path: Path) -> None:
        """Compress and save tensor data."""
        start_time = time.time()
        
        # Serialize tensor
        serialized = pickle.dumps(data)
        
        # Compress and save
        with gzip.open(cache_path, 'wb', compresslevel=6) as f:
            f.write(serialized)
        
        # Update statistics
        save_time = (time.time() - start_time) * 1000
        with self.stats_lock:
            self.save_times.append(save_time)
            if len(self.save_times) > 1000:
                self.save_times = self.save_times[-1000:]
            self.stats.avg_save_time_ms = sum(self.save_times) / len(self.save_times)
    
    def _load_and_decompress(self, cache_path: Path) -> torch.Tensor:
        """Load and decompress tensor data."""
        start_time = time.time()
        
        # Load and decompress
        with gzip.open(cache_path, 'rb') as f:
            serialized = f.read()
        
        # Deserialize tensor
        data = pickle.loads(serialized)
        
        # Update statistics
        load_time = (time.time() - start_time) * 1000
        with self.stats_lock:
            self.load_times.append(load_time)
            if len(self.load_times) > 1000:
                self.load_times = self.load_times[-1000:]
            self.stats.avg_load_time_ms = sum(self.load_times) / len(self.load_times)
        
        return data
    
    def is_cached(self, original_path: Path, config_hash: str) -> bool:
        """Check if file is cached and valid."""
        cache_path = self._get_cache_path(original_path, config_hash)
        
        if not cache_path.exists():
            return False
        
        # Check if original file is newer than cache
        try:
            original_stat = original_path.stat()
            cache_stat = cache_path.stat()
            
            if original_stat.st_mtime > cache_stat.st_mtime:
                # Original file is newer, cache is invalid
                cache_path.unlink()
                return False
        except OSError:
            return False
        
        return True
    
    def get_cached(self, original_path: Path, config_hash: str) -> Optional[torch.Tensor]:
        """Get cached data if available."""
        if not self.is_cached(original_path, config_hash):
            with self.stats_lock:
                self.stats.cache_misses += 1
            return None
        
        try:
            cache_path = self._get_cache_path(original_path, config_hash)
            data = self._load_and_decompress(cache_path)
            
            with self.stats_lock:
                self.stats.cache_hits += 1
            
            return data
        except Exception as e:
            logger.warning(f"Failed to load cached data for {original_path}: {e}")
            with self.stats_lock:
                self.stats.cache_misses += 1
            return None
    
    def save_cached(self, original_path: Path, config_hash: str, data: torch.Tensor) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(original_path, config_hash)
        
        try:
            self._compress_and_save(data, cache_path)
            
            # Update metadata
            file_key = str(original_path)
            self.metadata['file_metadata'][file_key] = {
                'cache_path': str(cache_path),
                'config_hash': config_hash,
                'cached_time': time.time(),
                'original_size': original_path.stat().st_size if original_path.exists() else 0
            }
            
        except Exception as e:
            logger.warning(f"Failed to save cached data for {original_path}: {e}")
    
    def cleanup_cache(self, max_age_days: int = 30) -> None:
        """Clean up old cache files."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        removed_files = 0
        freed_space = 0
        
        for cache_file in self.base_cache_dir.glob("*.cache.gz"):
            try:
                stat = cache_file.stat()
                if current_time - stat.st_mtime > max_age_seconds:
                    freed_space += stat.st_size
                    cache_file.unlink()
                    removed_files += 1
            except OSError:
                continue
        
        # Update cache size
        self._update_cache_size()
        
        # Enforce size limit
        if self.stats.cache_size_mb * 1024 * 1024 > self.max_cache_size_bytes:
            self._enforce_size_limit()
        
        self.stats.last_cleanup_time = current_time
        self._save_metadata()
        
        logger.info(f"Cache cleanup: removed {removed_files} files, freed {freed_space / 1024 / 1024:.2f} MB")
    
    def _update_cache_size(self) -> None:
        """Update cache size statistics."""
        total_size = 0
        file_count = 0
        
        for cache_file in self.base_cache_dir.glob("*.cache.gz"):
            try:
                total_size += cache_file.stat().st_size
                file_count += 1
            except OSError:
                continue
        
        self.stats.cache_size_mb = total_size / 1024 / 1024
        self.stats.total_files = file_count
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit by removing oldest files."""
        cache_files = []
        
        for cache_file in self.base_cache_dir.glob("*.cache.gz"):
            try:
                stat = cache_file.stat()
                cache_files.append((cache_file, stat.st_mtime, stat.st_size))
            except OSError:
                continue
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        current_size = sum(size for _, _, size in cache_files)
        
        # Remove oldest files until under limit
        for cache_file, mtime, size in cache_files:
            if current_size <= self.max_cache_size_bytes:
                break
            
            try:
                cache_file.unlink()
                current_size -= size
                logger.debug(f"Removed cache file {cache_file} to enforce size limit")
            except OSError:
                continue
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        self._update_cache_size()
        return self.stats


class BCMixingUtils:
    """Utilities for Between-Class (BC) mixing as described in EnvNet-v2 paper."""
    
    @staticmethod
    def a_weighted_spl(waveform: torch.Tensor, sample_rate: int = 44100) -> float:
        """
        Calculate A-weighted Sound Pressure Level (SPL) for perceptual loudness.
        
        Args:
            waveform: Input waveform tensor (shape: [1, samples])
            sample_rate: Sample rate
            
        Returns:
            A-weighted SPL in dB
        """
        # A-weighting filter coefficients (simplified approximation)
        # This is a simplified version - in practice, you'd use a proper A-weighting filter
        
        # Apply A-weighting (simplified high-pass emphasis around 2-4kHz)
        # For now, we'll use RMS energy as a proxy for loudness
        rms = torch.sqrt(torch.mean(waveform**2))
        
        # Convert to dB SPL (assuming 1.0 = full scale = ~94 dB SPL)
        if rms > 0:
            spl_db = 20 * torch.log10(rms) + 94  # Approximate SPL reference
        else:
            spl_db = torch.tensor(-80.0)  # Very quiet
            
        return float(spl_db)
    
    @staticmethod
    def perceptual_mixing_coefficient(r: float, spl1: float, spl2: float) -> float:
        """
        Adjust mixing ratio based on perceptual loudness (A-weighted SPL).
        
        Args:
            r: Random mixing ratio [0, 1]
            spl1: A-weighted SPL of first waveform
            spl2: A-weighted SPL of second waveform
            
        Returns:
            Perceptually-adjusted mixing coefficient p
        """
        # Adjust for equal loudness perception
        # If one sound is much louder, adjust the mixing to compensate
        spl_diff = spl1 - spl2
        
        # Simple perceptual adjustment (can be made more sophisticated)
        if abs(spl_diff) > 10:  # Significant difference in loudness
            # Reduce the contribution of the louder sound slightly
            adjustment = min(abs(spl_diff) / 40.0, 0.3)  # Cap adjustment
            if spl1 > spl2:
                r = r * (1 - adjustment)
            else:
                r = r * (1 + adjustment)
                
        return float(torch.clamp(torch.tensor(r), 0.0, 1.0))
    
    @staticmethod
    def mix_waveforms(wave1: torch.Tensor, wave2: torch.Tensor, p: float) -> torch.Tensor:
        """
        Mix two waveforms using the BC mixing formula.
        
        Args:
            wave1: First waveform tensor
            wave2: Second waveform tensor  
            p: Mixing coefficient [0, 1]
            
        Returns:
            Mixed waveform: (p*x1 + (1-p)*x2) / sqrt(p^2 + (1-p)^2)
        """
        # Ensure same length
        min_len = min(wave1.shape[-1], wave2.shape[-1])
        wave1 = wave1[..., :min_len]
        wave2 = wave2[..., :min_len]
        
        # BC mixing formula
        mixed = p * wave1 + (1 - p) * wave2
        normalization = torch.sqrt(torch.tensor(p**2 + (1 - p)**2))
        
        return mixed / normalization
    
    @staticmethod
    def create_soft_labels(r: float, label1: int, label2: int, num_classes: int) -> torch.Tensor:
        """
        Create soft labels for BC mixing.
        
        Args:
            r: Original mixing ratio
            label1: First class label
            label2: Second class label
            num_classes: Total number of classes
            
        Returns:
            Soft label tensor of shape [num_classes]
        """
        soft_labels = torch.zeros(num_classes, dtype=torch.float32)
        soft_labels[label1] = r
        soft_labels[label2] = 1 - r
        return soft_labels


class LazyDataLoader:
    """Lazy data loader with memory optimization."""
    
    def __init__(self, file_paths: List[Path], batch_size: int = 32):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.current_batch = []
        self.current_index = 0
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.file_paths):
            raise StopIteration
        
        batch_paths = self.file_paths[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        # Load batch data
        batch_data = []
        for path in batch_paths:
            try:
                data = torch.load(path, map_location='cpu')
                batch_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue
        
        return batch_data
    
    def __len__(self):
        return (len(self.file_paths) + self.batch_size - 1) // self.batch_size


class BCMixingDataset:
    """Mixin class for datasets that support BC mixing with memory optimization."""
    
    def __init__(self, enable_bc_mixing: bool = True, num_classes: int = 50):
        self.enable_bc_mixing = enable_bc_mixing
        self.num_classes = num_classes
        self.bc_utils = BCMixingUtils()
        self._cached_data: Optional[List[Tuple[torch.Tensor, int]]] = None
        self._use_lazy_loading = True
    
    def _load_data_for_mixing(self, file_paths: List[Path], show_progress: bool = True) -> List[Tuple[torch.Tensor, int]]:
        """Load data for BC mixing with progress tracking and memory optimization."""
        if self._cached_data is not None:
            return self._cached_data
        
        data = []
        desc = "Loading data for BC mixing"
        
        if show_progress:
            progress_bar = tqdm(file_paths, desc=desc, unit="files")
            file_iterator = progress_bar
        else:
            file_iterator = file_paths
        
        for path in file_iterator:
            audio_data = load_audio_bundle(path)
            if audio_data is not None:
                data.append(audio_data)
        
        if show_progress:
            progress_bar.close()
        
        self._cached_data = data
        return data
    
    def apply_bc_mixing(self, waveform: torch.Tensor, label: int, all_data: List[Tuple[torch.Tensor, int]], 
                       sample_rate: int = 44100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply BC mixing to a waveform sample.
        
        Args:
            waveform: Current waveform
            label: Current label
            all_data: List of all (waveform, label) pairs for sampling different class
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (mixed_waveform, soft_labels)
        """
        if not self.enable_bc_mixing:
            # Return one-hot labels for non-mixed samples
            soft_labels = create_one_hot_labels(label, self.num_classes)
            return waveform, soft_labels
        
        # Find a sample from a different class
        different_class_samples = [(w, l) for w, l in all_data if l != label]
        if not different_class_samples:
            # No different class available, return original
            soft_labels = create_one_hot_labels(label, self.num_classes)
            return waveform, soft_labels
        
        # Sample a different class
        other_waveform, other_label = random.choice(different_class_samples)
        
        # Random mixing ratio
        r = random.random()  # U(0, 1)
        
        # Calculate A-weighted SPL for both waveforms
        spl1 = self.bc_utils.a_weighted_spl(waveform, sample_rate)
        spl2 = self.bc_utils.a_weighted_spl(other_waveform, sample_rate)
        
        # Adjust mixing coefficient for perceptual loudness
        p = self.bc_utils.perceptual_mixing_coefficient(r, spl1, spl2)
        
        # Mix waveforms
        mixed_waveform = self.bc_utils.mix_waveforms(waveform, other_waveform, p)
        
        # Create soft labels
        soft_labels = self.bc_utils.create_soft_labels(r, label, other_label, self.num_classes)
        
        return mixed_waveform, soft_labels


class PreprocessingConfig:
    """Configuration for preprocessing pipelines with enhanced validation."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self._hash = None
        self._validation_errors = []
    
    def get_hash(self) -> str:
        """Get a hash of the configuration for cache validation."""
        if self._hash is None:
            # Include system info in hash for better cache invalidation
            import platform
            import torch
            system_info = {
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'platform': platform.platform()
            }
            # Convert any DictConfig/ListConfig objects to regular dicts/lists for JSON serialization
            def convert_omegaconf(obj):
                if hasattr(obj, '_content'):  # DictConfig check
                    return {k: convert_omegaconf(v) for k, v in obj.items()}
                elif hasattr(obj, '_content') or (hasattr(obj, '__iter__') and hasattr(obj, '__getitem__') and not isinstance(obj, str)):  # ListConfig check
                    try:
                        return [convert_omegaconf(item) for item in obj]
                    except:
                        return obj
                else:
                    return obj
            
            config_dict = convert_omegaconf(self.config)
            
            config_str = json.dumps({
                'config': config_dict,
                'system_info': system_info
            }, sort_keys=True)
            self._hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        return self._hash
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        self._validation_errors = []
        
        # Validate sample rate
        if 'sample_rate' in self.config:
            if not isinstance(self.config['sample_rate'], int) or self.config['sample_rate'] <= 0:
                self._validation_errors.append("sample_rate must be a positive integer")
        
        # Validate n_mels
        if 'n_mels' in self.config:
            if not isinstance(self.config['n_mels'], int) or self.config['n_mels'] <= 0:
                self._validation_errors.append("n_mels must be a positive integer")
        
        # Validate window_length
        if 'window_length' in self.config:
            if not isinstance(self.config['window_length'], (int, float)) or self.config['window_length'] <= 0:
                self._validation_errors.append("window_length must be a positive number")
        
        return len(self._validation_errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()
    
    def __getattr__(self, name: str) -> Any:
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class BasePreprocessor(ABC):
    """Abstract base class for model-specific preprocessors with advanced caching."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.cache_manager: Optional[AdvancedCacheManager] = None
        self._performance_stats = []
    
    @abstractmethod
    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess a waveform for model input.
        
        Args:
            waveform: Input waveform tensor (shape: [1, samples])
            sample_rate: Sample rate of the waveform
            
        Returns:
            Preprocessed tensor ready for model input
        """
        pass
    
    @abstractmethod
    def get_cache_suffix(self) -> str:
        """Return cache suffix for this preprocessor."""
        pass

    def multi_crop_test(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multiple evenly spaced crops for test-time evaluation.
        Default implementation returns single crop - override in subclasses.
        """
        return [self.preprocess(waveform, self.config.config.get('sample_rate', 44100))]
    
    def setup_cache(self, base_cache_dir: Path, force_rebuild: bool = False, max_cache_size_gb: float = 5.0) -> None:
        """
        Set up advanced caching for this preprocessor.
        
        Args:
            base_cache_dir: Base directory for caching
            force_rebuild: Whether to force rebuilding the cache
            max_cache_size_gb: Maximum cache size in GB
        """
        cache_dir = base_cache_dir / self.get_cache_suffix()
        self.cache_manager = AdvancedCacheManager(cache_dir, max_cache_size_gb)
        
        if force_rebuild:
            logger.info(f"Force rebuilding cache for {self.get_cache_suffix()}")
            self.cache_manager.cleanup_cache(max_age_days=0)
    
    def preprocess_with_cache(self, waveform: torch.Tensor, sample_rate: int, 
                            original_path: Optional[Path] = None) -> torch.Tensor:
        """
        Preprocess with caching support.
        
        Args:
            waveform: Input waveform
            sample_rate: Sample rate
            original_path: Original file path for caching
            
        Returns:
            Preprocessed tensor
        """
        if self.cache_manager is None or original_path is None:
            return self.preprocess(waveform, sample_rate)
        
        config_hash = self.config.get_hash()
        
        # Try to get from cache
        cached_data = self.cache_manager.get_cached(original_path, config_hash)
        if cached_data is not None:
            return cached_data
        
        # Process and cache
        start_time = time.time()
        processed_data = self.preprocess(waveform, sample_rate)
        process_time = time.time() - start_time
        
        # Save to cache
        self.cache_manager.save_cached(original_path, config_hash, processed_data)
        
        # Update performance stats
        self._performance_stats.append(process_time)
        if len(self._performance_stats) > 1000:
            self._performance_stats = self._performance_stats[-1000:]
        
        return processed_data
    
    def get_cache_stats(self) -> Optional[CacheStats]:
        """Get cache statistics."""
        if self.cache_manager is None:
            return None
        return self.cache_manager.get_stats()
    
    def cleanup_cache(self, max_age_days: int = 30) -> None:
        """Clean up cache files."""
        if self.cache_manager is not None:
            self.cache_manager.cleanup_cache(max_age_days)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self._performance_stats:
            return {}
        
        return {
            'avg_process_time_ms': (sum(self._performance_stats) / len(self._performance_stats)) * 1000,
            'min_process_time_ms': min(self._performance_stats) * 1000,
            'max_process_time_ms': max(self._performance_stats) * 1000,
            'total_processed': len(self._performance_stats)
        }


class EnvNetPreprocessor(BasePreprocessor):
    """Preprocessor for EnvNet-v2 model."""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self.window_length = config.config.get('window_length', 1.5)  # seconds
        self.padding_ratio = config.config.get('padding_ratio', 0.5)  # T/2 padding
        self.sample_rate = config.config.get('sample_rate', 44100)
        self.bc_mixing = config.config.get('bc_mixing', True)
        self.test_crops = config.config.get('test_crops', 10)
        self.augment_config = config.config.get('augment', {})
        
        # Calculate sample lengths
        self.window_samples = int(self.window_length * self.sample_rate)
        self.padding_samples = int(self.window_samples * self.padding_ratio)
    
    def get_cache_suffix(self) -> str:
        return f"envnet_v2_{self.config.get_hash()}"
    
    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess for EnvNet-v2: pad and potentially crop waveform.
        
        Note: Random cropping and BC mixing are handled at runtime in the dataset.
        This method handles the deterministic preprocessing that can be cached.
        """
        # Resample if needed
        waveform = resample_waveform(waveform, sample_rate, self.sample_rate)
        
        # Add padding for random cropping
        padded_waveform = F.pad(waveform, (self.padding_samples, self.padding_samples), mode='constant', value=0)
        
        return padded_waveform
    
    def random_crop(self, waveform: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply random cropping for training or deterministic cropping for testing.
        
        Args:
            waveform: Padded waveform tensor
            training: Whether in training mode (random crop) or test mode (center crop)
            
        Returns:
            Cropped waveform of length window_samples
        """
        total_length = waveform.shape[-1]
        
        if total_length <= self.window_samples:
            # If waveform is shorter than window, pad it
            needed_padding = self.window_samples - total_length
            return F.pad(waveform, (0, needed_padding), mode='constant', value=0)
        
        if training:
            # Random crop
            max_start = total_length - self.window_samples
            start_idx = random.randint(0, max_start)
        else:
            # Center crop for testing
            start_idx = (total_length - self.window_samples) // 2
            
        return waveform[..., start_idx:start_idx + self.window_samples]
    
    def multi_crop_test(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multiple evenly spaced crops for test-time evaluation.
        
        Args:
            waveform: Padded waveform tensor
            
        Returns:
            List of cropped waveforms
        """
        total_length = waveform.shape[-1]
        
        if total_length <= self.window_samples:
            # If too short, just return padded version
            needed_padding = self.window_samples - total_length
            padded = F.pad(waveform, (0, needed_padding), mode='constant', value=0)
            return [padded]
        
        # Create evenly spaced crops
        max_start = total_length - self.window_samples
        starts = torch.linspace(0, max_start, self.test_crops).long()
        
        crops = []
        for start_idx in starts:
            crop = waveform[..., start_idx:start_idx + self.window_samples]
            crops.append(crop)
            
        return crops
    
    def apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply optional augmentations (time stretch, gain shift).
        
        Args:
            waveform: Input waveform
            
        Returns:
            Augmented waveform
        """
        if not self.augment_config:
            return waveform
            
        # Time stretch augmentation
        if 'time_stretch' in self.augment_config and random.random() < 0.5:
            stretch_range = self.augment_config['time_stretch']
            if isinstance(stretch_range, list) and len(stretch_range) == 2:
                stretch_factor = random.uniform(stretch_range[0], stretch_range[1])
                # Simple time stretch by resampling
                original_length = waveform.shape[-1]
                stretched_length = int(original_length / stretch_factor)
                
                # Resample to simulate time stretch
                if stretched_length != original_length:
                    waveform = F.interpolate(
                        waveform.unsqueeze(0), 
                        size=stretched_length, 
                        mode='linear', 
                        align_corners=False
                    ).squeeze(0)
        
        # Gain shift augmentation
        if 'gain_shift' in self.augment_config and random.random() < 0.5:
            gain_range = self.augment_config['gain_shift']
            if isinstance(gain_range, list) and len(gain_range) == 2:
                gain_db = random.uniform(gain_range[0], gain_range[1])
                gain_linear = 10 ** (gain_db / 20.0)
                waveform = waveform * gain_linear
                
        return waveform


class MixupAugmentation:
    """Mixup augmentation for AST training."""
    
    def __init__(self, alpha: float = 0.5, prob: float = 1.0):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, spec1: torch.Tensor, spec2: torch.Tensor, label1: int, label2: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup to two spectrograms.
        
        Args:
            spec1: First spectrogram
            spec2: Second spectrogram  
            label1: First label
            label2: Second label
            num_classes: Total number of classes
            
        Returns:
            Mixed spectrogram and soft labels
        """
        if random.random() > self.prob:
            # No mixup, return original
            labels = create_one_hot_labels(label1, num_classes)
            return spec1, labels
        
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0
        
        # Mix spectrograms
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        
        # Create soft labels
        soft_labels = torch.zeros(num_classes, dtype=torch.float32)
        soft_labels[label1] = lam
        soft_labels[label2] = 1 - lam
        
        return mixed_spec, soft_labels


class ASTPreprocessor(BasePreprocessor):
    """Preprocessor for Audio Spectrogram Transformer (AST) model."""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        self.n_mels = config.config.get('n_mels', 128)
        self.sample_rate = config.config.get('sample_rate', 44100)
        self.target_mean = config.config.get('target_mean', 0.0)
        self.target_std = config.config.get('target_std', 0.5)
        self.normalize = config.config.get('normalize', True)
        
        # AST paper parameters - match the model exactly
        self.n_fft = AST_N_FFT
        self.hop_length = AST_HOP_LENGTH
        self.win_length = AST_WIN_LENGTH
        
        # Create mel spectrogram transform matching AST model
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        
        # AmplitudeToDB transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        # Mixup augmentation
        mixup_config = config.config.get('mixup', {})
        if mixup_config.get('enabled', False):
            self.mixup = MixupAugmentation(
                alpha=mixup_config.get('alpha', 0.5),
                prob=mixup_config.get('prob', 0.5)
            )
        else:
            self.mixup = None
    
    def get_cache_suffix(self) -> str:
        return f"ast_{self.config.get_hash()}"
    
    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess for AST: convert to log-mel spectrogram matching AST model.
        
        This preprocessing matches exactly what the AST model does internally,
        but allows for caching and external augmentation.
        """
        # Resample if needed
        waveform = resample_waveform(waveform, sample_rate, self.sample_rate)
        
        # Convert to mel spectrogram (matching AST model parameters)
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB scale (matching AST model)
        log_mel = self.amplitude_to_db(mel_spec)
        
        # Normalize if requested (AST paper: mean=0, std=0.5)
        if self.normalize:
            # Global normalization across frequency and time
            current_mean = log_mel.mean()
            current_std = log_mel.std()
            
            if current_std > 0:
                log_mel = (log_mel - current_mean) / current_std
                log_mel = log_mel * self.target_std + self.target_mean
        
        return log_mel
    
    def multi_crop_test(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multiple evenly spaced crops for test-time evaluation.
        For AST, this creates multiple spectrograms from different waveform segments.
        
        Args:
            waveform: Padded waveform tensor
            
        Returns:
            List of preprocessed spectrograms
        """
        # For AST, we create multiple spectrograms from different waveform segments
        # This is less common for AST but provides consistency with EnvNet
        total_length = waveform.shape[-1]
        n_crops = 10  # Default number of crops
        
        if total_length <= self.sample_rate * 5:  # If shorter than 5 seconds
            # Just return the single preprocessed spectrogram
            return [self.preprocess(waveform, self.sample_rate)]
        
        # Create evenly spaced crops
        crop_length = int(self.sample_rate * 5)  # 5-second crops
        max_start = total_length - crop_length
        starts = torch.linspace(0, max_start, n_crops).long()
        
        crops = []
        for start_idx in starts:
            crop = waveform[..., start_idx:start_idx + crop_length]
            # Preprocess each crop to spectrogram
            spec = self.preprocess(crop, self.sample_rate)
            crops.append(spec)
            
        return crops
    
    def apply_specaugment(self, spectrogram: torch.Tensor, time_mask: int = 192, freq_mask: int = 48) -> torch.Tensor:
        """
        Apply SpecAugment (time and frequency masking).
        
        Args:
            spectrogram: Input spectrogram (C, F, T)
            time_mask: Maximum time mask length
            freq_mask: Maximum frequency mask length
            
        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.clone()
        
        # Get dimensions
        channels, n_mels, n_frames = spec.shape
        
        # Time masking
        if time_mask > 0 and n_frames > time_mask:
            mask_length = random.randint(1, min(time_mask, n_frames // 4))
            mask_start = random.randint(0, n_frames - mask_length)
            spec[:, :, mask_start:mask_start + mask_length] = 0
        
        # Frequency masking  
        if freq_mask > 0 and n_mels > freq_mask:
            mask_length = random.randint(1, min(freq_mask, n_mels // 4))
            mask_start = random.randint(0, n_mels - mask_length)
            spec[:, mask_start:mask_start + mask_length, :] = 0
            
        return spec
    
    def apply_mixup(self, spec1: torch.Tensor, spec2: torch.Tensor, label1: int, label2: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation if enabled."""
        if self.mixup:
            return self.mixup(spec1, spec2, label1, label2, num_classes)
        else:
            # No mixup, return original with one-hot label
            labels = create_one_hot_labels(label1, num_classes)
            return spec1, labels


class PreprocessingCache:
    """Enhanced manager for preprocessing cache operations with parallel processing."""
    
    def __init__(self, base_cache_dir: Path, max_cache_size_gb: float = 5.0):
        self.base_cache_dir = Path(base_cache_dir)
        self.max_cache_size_gb = max_cache_size_gb
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Global stats across all preprocessors
        self.global_stats = CacheStats()
        self.preprocessors: Dict[str, BasePreprocessor] = {}
    
    def get_preprocessor(self, mode: str, config: PreprocessingConfig) -> BasePreprocessor:
        """
        Get a preprocessor for the specified mode.
        
        Args:
            mode: Preprocessing mode ('envnet_v2' or 'ast')
            config: Preprocessing configuration
            
        Returns:
            Configured preprocessor instance
        """
        # Validate config first
        if not config.validate():
            errors = config.get_validation_errors()
            raise ValueError(f"Invalid preprocessing config: {errors}")
        
        if mode == 'envnet_v2':
            return EnvNetPreprocessor(config)
        elif mode == 'ast':
            return ASTPreprocessor(config)
        elif mode == 'cnn_esc50':
            return CNNESC50Preprocessor(config)
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
    
    def setup_preprocessor(self, mode: str, config: PreprocessingConfig, force_rebuild: bool = False) -> BasePreprocessor:
        """
        Set up a preprocessor with enhanced caching.
        
        Args:
            mode: Preprocessing mode ('envnet_v2' or 'ast')
            config: Preprocessing configuration
            force_rebuild: Whether to force rebuilding the cache
            
        Returns:
            Configured preprocessor with cache set up
        """
        cache_key = f"{mode}_{config.get_hash()}"
        
        if cache_key in self.preprocessors and not force_rebuild:
            return self.preprocessors[cache_key]
        
        preprocessor = self.get_preprocessor(mode, config)
        preprocessor.setup_cache(self.base_cache_dir, force_rebuild=force_rebuild, max_cache_size_gb=self.max_cache_size_gb)
        
        self.preprocessors[cache_key] = preprocessor
        return preprocessor
    
    def batch_preprocess(self, file_paths: List[Path], mode: str, config: PreprocessingConfig, 
                        num_workers: int = 4, show_progress: bool = True) -> List[torch.Tensor]:
        """
        Batch preprocess multiple files with parallel processing.
        
        Args:
            file_paths: List of file paths to preprocess
            mode: Preprocessing mode
            config: Preprocessing configuration
            num_workers: Number of parallel workers
            show_progress: Whether to show progress bar
            
        Returns:
            List of preprocessed tensors
        """
        preprocessor = self.setup_preprocessor(mode, config)
        results = []
        
        def process_file(file_path: Path) -> Tuple[int, Optional[torch.Tensor]]:
            """Process a single file and return (index, result)."""
            try:
                audio_data = load_audio_bundle(file_path)
                if audio_data is None:
                    return file_paths.index(file_path), None
                
                waveform, _ = audio_data  # We don't need the label for preprocessing
                sample_rate = 44100  # Assume standard sample rate, could be configurable
                
                # Preprocess with caching
                processed = preprocessor.preprocess_with_cache(waveform, sample_rate, file_path)
                return file_paths.index(file_path), processed
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                return file_paths.index(file_path), None
        
        # Process files in parallel
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_file, path) for path in file_paths]
                indexed_results: List[Optional[torch.Tensor]] = [None] * len(file_paths)
                
                # Progress tracking
                if show_progress:
                    progress_bar = tqdm(total=len(file_paths), desc=f"Preprocessing with {mode}", unit="files")
                
                for future in as_completed(futures):
                    try:
                        idx, result = future.result()
                        indexed_results[idx] = result
                        if show_progress:
                            progress_bar.update(1)
                    except Exception as e:
                        logger.warning(f"Processing failed: {e}")
                        if show_progress:
                            progress_bar.update(1)
                        continue
                
                if show_progress:
                    progress_bar.close()
                    
                # Filter out None results
                results = [r for r in indexed_results if r is not None]
        else:
            # Sequential processing
            if show_progress:
                file_iterator = tqdm(file_paths, desc=f"Preprocessing with {mode}", unit="files")
            else:
                file_iterator = file_paths
                
            for path in file_iterator:
                try:
                    _, result = process_file(path)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Processing failed: {e}")
                    continue
        
        return results
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cache statistics across all preprocessors."""
        combined_stats = {
            'total_preprocessors': len(self.preprocessors),
            'cache_size_mb': 0.0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'preprocessor_stats': {}
        }
        
        for key, preprocessor in self.preprocessors.items():
            stats = preprocessor.get_cache_stats()
            if stats:
                combined_stats['cache_size_mb'] += stats.cache_size_mb
                combined_stats['total_cache_hits'] += stats.cache_hits
                combined_stats['total_cache_misses'] += stats.cache_misses
                combined_stats['preprocessor_stats'][key] = stats.to_dict()
        
        # Calculate overall hit rate
        total_requests = combined_stats['total_cache_hits'] + combined_stats['total_cache_misses']
        combined_stats['overall_hit_rate'] = combined_stats['total_cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        return combined_stats
    
    def cleanup_all_caches(self, max_age_days: int = 30) -> None:
        """Clean up all preprocessor caches."""
        for preprocessor in self.preprocessors.values():
            preprocessor.cleanup_cache(max_age_days)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all preprocessors."""
        summary = {
            'total_files_processed': 0,
            'avg_process_time_ms': 0.0,
            'preprocessors': {}
        }
        
        all_times = []
        for key, preprocessor in self.preprocessors.items():
            stats = preprocessor.get_performance_stats()
            if stats:
                summary['preprocessors'][key] = stats
                summary['total_files_processed'] += stats.get('total_processed', 0)
                
                # Collect all processing times for global average
                avg_time = stats.get('avg_process_time_ms', 0.0)
                total_processed = int(stats.get('total_processed', 1))
                if avg_time > 0 and total_processed > 0:
                    # Add the average time for each processed file
                    for _ in range(total_processed):
                        all_times.append(avg_time)
        
        if all_times:
            summary['avg_process_time_ms'] = sum(all_times) / len(all_times)
        
        return summary


# Enhanced factory function with better error handling and logging
def create_preprocessor(mode: str, config_dict: Dict[str, Any], base_cache_dir: Path, 
                       force_rebuild: bool = False, max_cache_size_gb: float = 5.0) -> BasePreprocessor:
    """
    Create and set up a preprocessor with enhanced caching.
    
    Args:
        mode: Preprocessing mode ('envnet_v2' or 'ast')
        config_dict: Configuration dictionary
        base_cache_dir: Base directory for caching
        force_rebuild: Whether to force rebuilding the cache
        max_cache_size_gb: Maximum cache size in GB
        
    Returns:
        Configured preprocessor ready for use
    """
    try:
        config = PreprocessingConfig(**config_dict)
        cache_manager = PreprocessingCache(base_cache_dir, max_cache_size_gb)
        preprocessor = cache_manager.setup_preprocessor(mode, config, force_rebuild=force_rebuild)
        
        logger.info(f"✓ Created {mode} preprocessor with enhanced caching")
        logger.info(f"  - Config hash: {config.get_hash()}")
        logger.info(f"  - Cache directory: {base_cache_dir}")
        logger.info(f"  - Max cache size: {max_cache_size_gb}GB")
        
        return preprocessor
        
    except Exception as e:
        logger.error(f"Failed to create preprocessor: {e}")
        raise


# Utility functions for cache management
def get_cache_usage_report(base_cache_dir: Path) -> Dict[str, Any]:
    """
    Generate a comprehensive cache usage report.
    
    Args:
        base_cache_dir: Base cache directory
        
    Returns:
        Dictionary with cache usage statistics
    """
    cache_dir = Path(base_cache_dir)
    report = {
        'cache_directory': str(cache_dir),
        'total_size_mb': 0.0,
        'total_files': 0,
        'subdirectories': {},
        'file_types': defaultdict(int),
        'largest_files': [],
        'oldest_files': [],
        'newest_files': []
    }
    
    if not cache_dir.exists():
        return report
    
    all_files = []
    
    # Collect all files recursively
    for file_path in cache_dir.rglob('*'):
        if file_path.is_file():
            try:
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                
                file_info = {
                    'path': str(file_path.relative_to(cache_dir)),
                    'size_mb': size_mb,
                    'modified_time': stat.st_mtime,
                    'extension': file_path.suffix
                }
                
                all_files.append(file_info)
                report['total_size_mb'] += size_mb
                report['total_files'] += 1
                report['file_types'][file_path.suffix] += 1
                
                # Track subdirectory sizes
                subdir = str(file_path.parent.relative_to(cache_dir))
                if subdir not in report['subdirectories']:
                    report['subdirectories'][subdir] = {'size_mb': 0.0, 'files': 0}
                report['subdirectories'][subdir]['size_mb'] += size_mb
                report['subdirectories'][subdir]['files'] += 1
                
            except OSError:
                continue
    
    # Sort files for top lists
    if all_files:
        all_files.sort(key=lambda x: x['size_mb'], reverse=True)
        report['largest_files'] = all_files[:10]
        
        all_files.sort(key=lambda x: x['modified_time'])
        report['oldest_files'] = all_files[:5]
        report['newest_files'] = all_files[-5:]
    
    return report


def cleanup_cache_by_age(base_cache_dir: Path, max_age_days: int = 30) -> Dict[str, Any]:
    """
    Clean up cache files older than specified age.
    
    Args:
        base_cache_dir: Base cache directory
        max_age_days: Maximum age in days
        
    Returns:
        Cleanup summary
    """
    cache_dir = Path(base_cache_dir)
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    summary = {
        'files_removed': 0,
        'space_freed_mb': 0.0,
        'errors': []
    }
    
    if not cache_dir.exists():
        return summary
    
    for file_path in cache_dir.rglob('*'):
        if file_path.is_file():
            try:
                stat = file_path.stat()
                if current_time - stat.st_mtime > max_age_seconds:
                    size_mb = stat.st_size / (1024 * 1024)
                    file_path.unlink()
                    summary['files_removed'] += 1
                    summary['space_freed_mb'] += size_mb
            except OSError as e:
                summary['errors'].append(f"Failed to remove {file_path}: {e}")
    
    return summary 


class CNNESC50Preprocessor(BasePreprocessor):
    """Preprocessor for the CNN-ESC50 model from Inik et al. (2023)."""
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
        # parameters
        self.sample_rate = config.config.get('sample_rate', 44100)
        self.n_mels      = config.config.get('n_mels', 128)
        # mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            n_mels=self.n_mels,
            power=2.0
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        # image‐style augmentations (flips + small translation)
        self.augment = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.1,0.1)),
        ])
        # to‐tensor + normalize
        self.post = T.Compose([
            T.ToTensor(),       # yields [C=1,H,W]
            T.Normalize(mean=[0.0], std=[0.5])
        ])

    def get_cache_suffix(self) -> str:
        return f"cnn_esc50_{self.config.get_hash()}"

    def preprocess(self, waveform: Tensor, sample_rate: int) -> Tensor:
        # resample
        wav = resample_waveform(waveform, sample_rate, self.sample_rate)
        # mel → db
        m  = self.mel_transform(wav)      # [1, n_mels, T]
        db = self.to_db(m)                # log-mel
        # to PIL F-mode image for augment
        img = Image.fromarray(db.squeeze(0).numpy().astype('float32'), mode='F')
        # augment & normalize
        img_augmented = self.augment(img)  # PIL Image
        tensor = self.post(img_augmented)   # [1,224,224] tensor
        # replicate to 3 channels for CNN input
        tensor = tensor.repeat(3, 1, 1)  # [3,224,224]
        return tensor