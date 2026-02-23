"""PyTorch Lightning DataModule for Training (vendored from ACE-Step).

Handles data loading and preprocessing for training ACE-Step adapters.
Supports both raw audio loading and preprocessed tensor loading.
"""

import os
import json
import random
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

import torch

from acestep.training_v2.path_utils import normalize_path
from torch.utils.data import Dataset, DataLoader

try:
    from lightning.pytorch import LightningDataModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning("Lightning not installed. Training module will not be available.")
    # Create a dummy class for type hints
    class LightningDataModule:
        pass


# ============================================================================
# Preprocessed Tensor Dataset (Recommended for Training)
# ============================================================================

class PreprocessedTensorDataset(Dataset):
    """Dataset that loads preprocessed tensor files.

    This is the recommended dataset for training as all tensors are pre-computed:
    - target_latents: VAE-encoded audio [T, 64]
    - encoder_hidden_states: Condition encoder output [L, D]
    - encoder_attention_mask: Condition mask [L]
    - context_latents: Source context [T, 65]
    - attention_mask: Audio latent mask [T]

    No VAE/text encoder needed during training - just load tensors directly!

    When ``chunk_duration`` is set (seconds), each ``__getitem__`` call
    extracts a random window of that length from the T-aligned tensors.
    A different random offset is chosen every call, so each epoch sees
    different slices -- this acts as data augmentation and reduces VRAM.
    """

    # ACE-Step DCAE latent rate (frames per second).  Used as a fallback
    # when per-sample duration metadata is unavailable.
    _LATENT_FPS_FALLBACK = 25

    @staticmethod
    def _scan_tensor_dir(tensor_dir: str) -> List[str]:
        """Collect .pt files from tensor_dir when manifest is unavailable."""
        paths: List[str] = []
        for filename in os.listdir(tensor_dir):
            if filename.endswith(".pt") and filename != "manifest.json":
                paths.append(os.path.join(tensor_dir, filename))
        return paths

    @staticmethod
    def _normalize_manifest_path(path_value: Any, tensor_dir: str) -> Optional[str]:
        """Normalize one manifest path entry into an OS-native path string."""
        if not isinstance(path_value, str):
            return None

        raw = path_value.strip()
        if not raw:
            return None

        # Common malformed Windows absolute-path typo: ".C:\\..."
        if re.match(r"^\.[A-Za-z]:[\\/]", raw):
            raw = raw[1:]

        # Windows absolute path / UNC path: keep slashes as-is and normalize.
        is_windows_abs = bool(re.match(r"^[A-Za-z]:[\\/]", raw) or raw.startswith("\\\\"))
        if is_windows_abs:
            return os.path.normpath(raw)

        # Relative paths can arrive with backslashes from Windows-generated manifests.
        normalized_rel = raw.replace("\\", "/")
        try:
            p = Path(normalized_rel).expanduser()
            if not p.is_absolute():
                p = Path(tensor_dir) / p
            return os.path.normpath(str(p.resolve(strict=False)))
        except (OSError, ValueError):
            p = Path(tensor_dir) / normalized_rel
            return os.path.normpath(str(p))

    def __init__(self, tensor_dir: str, chunk_duration: Optional[int] = None):
        """Initialize from a directory of preprocessed .pt files.

        Args:
            tensor_dir: Directory containing preprocessed .pt files and manifest.json
            chunk_duration: Optional random chunk length in seconds.
                ``None`` = disabled (use full sample).
                ``60``  = recommended default.
                Values below 60 (e.g. 30) may reduce training quality for
                full-length inference -- use with caution.
        """
        self.tensor_dir = normalize_path(tensor_dir) or str(tensor_dir).strip()
        self.chunk_duration = chunk_duration
        self.sample_paths = []
        self.manifest_path = os.path.join(self.tensor_dir, "manifest.json")
        self.manifest_error: Optional[str] = None
        self.manifest_loaded: bool = False
        self.manifest_fallback_used: bool = False

        # Load manifest if exists
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                samples = manifest.get("samples", [])
                if not isinstance(samples, list):
                    raise ValueError(
                        f"'samples' must be a list in manifest.json, got {type(samples).__name__}"
                    )
                normalized = [
                    self._normalize_manifest_path(sample, self.tensor_dir) for sample in samples
                ]
                self.sample_paths = [p for p in normalized if p]
                self.manifest_loaded = True
            except json.JSONDecodeError as exc:
                self.manifest_error = (
                    f"manifest.json is invalid JSON: {exc}. "
                    "Windows tip: escape backslashes (\\\\) or use forward slashes (/)."
                )
                logger.error(self.manifest_error)
                self.sample_paths = self._scan_tensor_dir(self.tensor_dir)
                self.manifest_fallback_used = True
            except Exception as exc:
                self.manifest_error = f"Failed to read manifest.json: {exc}"
                logger.error(self.manifest_error)
                self.sample_paths = self._scan_tensor_dir(self.tensor_dir)
                self.manifest_fallback_used = True
        else:
            self.sample_paths = self._scan_tensor_dir(self.tensor_dir)

        # Validate paths
        self.valid_paths = [p for p in self.sample_paths if os.path.exists(p)]

        # Fallback: manifest loaded but all paths invalid (e.g. folder renamed) -- try directory scan
        if (
            self.manifest_loaded
            and len(self.valid_paths) == 0
            and len(self.sample_paths) > 0
        ):
            scanned = self._scan_tensor_dir(self.tensor_dir)
            if scanned:
                logger.warning(
                    "manifest.json paths invalid (e.g. folder renamed); using directory scan (%d .pt files)",
                    len(scanned),
                )
                self.sample_paths = scanned
                self.valid_paths = [p for p in scanned if os.path.exists(p)]
                self.manifest_fallback_used = True

        if len(self.valid_paths) != len(self.sample_paths):
            missing = len(self.sample_paths) - len(self.valid_paths)
            logger.warning(f"Some tensor files not found: {missing} missing")
            if missing > 0 and self.sample_paths:
                preview = [p for p in self.sample_paths if not os.path.exists(p)][:3]
                logger.warning(f"Missing sample preview: {preview}")

        # Auto-detect latent FPS from first sample when chunking is enabled
        self._latent_fps: float = self._LATENT_FPS_FALLBACK
        if self.chunk_duration is not None and self.valid_paths:
            self._latent_fps = self._detect_latent_fps()

        chunk_info = ""
        if self.chunk_duration is not None:
            chunk_frames = int(self.chunk_duration * self._latent_fps)
            chunk_info = f", chunk={self.chunk_duration}s (~{chunk_frames} frames)"
        logger.info(f"PreprocessedTensorDataset: {len(self.valid_paths)} samples from {self.tensor_dir}{chunk_info}")

    def _detect_latent_fps(self) -> float:
        """Probe the first sample to compute latent frames-per-second."""
        try:
            data = torch.load(self.valid_paths[0], map_location='cpu', weights_only=True)
            T = data["target_latents"].shape[0]
            meta = data.get("metadata", {})
            duration = meta.get("duration", 0)
            if isinstance(duration, (int, float)) and duration > 0:
                fps = T / duration
                logger.info(f"Auto-detected latent FPS: {fps:.1f} (from {T} frames / {duration}s)")
                return fps
        except Exception as exc:
            logger.debug(f"Could not auto-detect latent FPS: {exc}")
        logger.info(f"Using fallback latent FPS: {self._LATENT_FPS_FALLBACK}")
        return float(self._LATENT_FPS_FALLBACK)

    def __len__(self) -> int:
        return len(self.valid_paths)

    _REQUIRED_KEYS = frozenset([
        "target_latents", "attention_mask", "encoder_hidden_states",
        "encoder_attention_mask", "context_latents",
    ])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a preprocessed tensor file, optionally chunked."""
        tensor_path = self.valid_paths[idx]
        data = torch.load(tensor_path, map_location='cpu', weights_only=True)

        missing = self._REQUIRED_KEYS - data.keys()
        if missing:
            raise KeyError(
                f"Preprocessed tensor file is missing required keys {sorted(missing)}: "
                f"{tensor_path}"
            )

        target_latents = data["target_latents"]      # [T, 64]
        attention_mask = data["attention_mask"]        # [T]
        context_latents = data["context_latents"]      # [T, 65]
        encoder_hidden_states = data["encoder_hidden_states"]  # [L, D]
        encoder_attention_mask = data["encoder_attention_mask"]  # [L]
        metadata = data.get("metadata", {})
        del data

        # Validate tensors for NaN/Inf (corrupted preprocessing output)
        for _name, _tens in [
            ("target_latents", target_latents),
            ("encoder_hidden_states", encoder_hidden_states),
            ("context_latents", context_latents),
        ]:
            if torch.isnan(_tens).any() or torch.isinf(_tens).any():
                logger.warning(
                    "[Side-Step] NaN/Inf in '%s' of %s -- replacing with zeros",
                    _name, tensor_path,
                )
                _tens.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        # Random chunking: slice a window from T-aligned tensors
        if self.chunk_duration is not None:
            T = target_latents.shape[0]
            chunk_frames = int(self.chunk_duration * self._latent_fps)
            if chunk_frames > 0 and T > chunk_frames:
                start = torch.randint(0, T - chunk_frames, (1,)).item()
                end = start + chunk_frames
                target_latents = target_latents[start:end]
                attention_mask = attention_mask[start:end]
                context_latents = context_latents[start:end]

        return {
            "target_latents": target_latents,           # [T', 64]
            "attention_mask": attention_mask,             # [T']
            "encoder_hidden_states": encoder_hidden_states,  # [L, D]
            "encoder_attention_mask": encoder_attention_mask,  # [L]
            "context_latents": context_latents,          # [T', 65]
            "metadata": metadata,
        }


def collate_preprocessed_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for preprocessed tensor batches.

    Handles variable-length tensors by padding to the longest in the batch.
    """
    # Get max lengths
    max_latent_len = max(s["target_latents"].shape[0] for s in batch)
    max_encoder_len = max(s["encoder_hidden_states"].shape[0] for s in batch)

    # Pad and stack tensors
    target_latents = []
    attention_masks = []
    encoder_hidden_states = []
    encoder_attention_masks = []
    context_latents = []

    for sample in batch:
        # Pad target_latents [T, 64] -> [max_T, 64]
        tl = sample["target_latents"]
        if tl.shape[0] < max_latent_len:
            pad = tl.new_zeros(max_latent_len - tl.shape[0], tl.shape[1])
            tl = torch.cat([tl, pad], dim=0)
        target_latents.append(tl)

        # Pad attention_mask [T] -> [max_T]
        am = sample["attention_mask"]
        if am.shape[0] < max_latent_len:
            pad = am.new_zeros(max_latent_len - am.shape[0])
            am = torch.cat([am, pad], dim=0)
        attention_masks.append(am)

        # Pad context_latents [T, 65] -> [max_T, 65]
        cl = sample["context_latents"]
        if cl.shape[0] < max_latent_len:
            pad = cl.new_zeros(max_latent_len - cl.shape[0], cl.shape[1])
            cl = torch.cat([cl, pad], dim=0)
        context_latents.append(cl)

        # Pad encoder_hidden_states [L, D] -> [max_L, D]
        ehs = sample["encoder_hidden_states"]
        if ehs.shape[0] < max_encoder_len:
            pad = ehs.new_zeros(max_encoder_len - ehs.shape[0], ehs.shape[1])
            ehs = torch.cat([ehs, pad], dim=0)
        encoder_hidden_states.append(ehs)

        # Pad encoder_attention_mask [L] -> [max_L]
        eam = sample["encoder_attention_mask"]
        if eam.shape[0] < max_encoder_len:
            pad = eam.new_zeros(max_encoder_len - eam.shape[0])
            eam = torch.cat([eam, pad], dim=0)
        encoder_attention_masks.append(eam)

    return {
        "target_latents": torch.stack(target_latents),  # [B, T, 64]
        "attention_mask": torch.stack(attention_masks),  # [B, T]
        "encoder_hidden_states": torch.stack(encoder_hidden_states),  # [B, L, D]
        "encoder_attention_mask": torch.stack(encoder_attention_masks),  # [B, L]
        "context_latents": torch.stack(context_latents),  # [B, T, 65]
        "metadata": [s["metadata"] for s in batch],
    }


class PreprocessedDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    """DataModule for preprocessed tensor files.

    This is the recommended DataModule for training. It loads pre-computed tensors
    directly without needing VAE, text encoder, or condition encoder at training time.
    """

    def __init__(
        self,
        tensor_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory_device: str = "",
        val_split: float = 0.0,
        chunk_duration: Optional[int] = None,
    ):
        """Initialize the data module.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            batch_size: Training batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            val_split: Fraction of data for validation (0 = no validation)
            chunk_duration: Random chunk length in seconds (None = disabled)
        """
        if LIGHTNING_AVAILABLE:
            super().__init__()

        self.tensor_dir = normalize_path(tensor_dir) or str(tensor_dir).strip()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device
        self.val_split = val_split
        self.chunk_duration = chunk_duration

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            full_dataset = PreprocessedTensorDataset(
                self.tensor_dir, chunk_duration=self.chunk_duration,
            )

            if self.val_split > 0 and len(full_dataset) > 1:
                n_val = max(1, int(len(full_dataset) * self.val_split))
                n_train = len(full_dataset) - n_val

                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [n_train, n_val]
                )
            else:
                self.train_dataset = full_dataset
                self.val_dataset = None

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        pin_memory_device = self.pin_memory_device if self.pin_memory else ""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=pin_memory_device,
            collate_fn=collate_preprocessed_batch,
            drop_last=False,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        pin_memory_device = self.pin_memory_device if self.pin_memory else ""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=pin_memory_device,
            collate_fn=collate_preprocessed_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

