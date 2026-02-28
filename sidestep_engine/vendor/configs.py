"""Training Configuration Classes (vendored from ACE-Step).

Contains dataclasses for LoRA, LoKR, LoHA, OFT, and training configurations.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.

    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor (alpha/r determines the scaling)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
    """
    r: int = 64
    alpha: int = 128
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    rank_pattern: dict = field(default_factory=dict)
    """Per-module rank overrides for adaptive LoRA (PEFT ``rank_pattern``)."""
    alpha_pattern: dict = field(default_factory=dict)
    """Per-module alpha overrides for adaptive LoRA (PEFT ``alpha_pattern``)."""
    rank_min: int = 16
    """Safety-net rank used as global ``r`` when ``rank_pattern`` is active."""

    def to_dict(self):
        """Convert to dictionary for PEFT config."""
        d = {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }
        if self.rank_pattern:
            d["rank_pattern"] = self.rank_pattern
        if self.alpha_pattern:
            d["alpha_pattern"] = self.alpha_pattern
        return d


@dataclass
class LoKRConfig:
    """Configuration for LoKr (Low-Rank Kronecker) training."""

    linear_dim: int = 64
    linear_alpha: int = 128
    factor: int = -1
    decompose_both: bool = False
    use_tucker: bool = False
    use_scalar: bool = False
    weight_decompose: bool = False
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    full_matrix: bool = False
    bypass_mode: bool = False
    rs_lora: bool = False
    unbalanced_factorization: bool = False

    def to_dict(self):
        """Convert to dictionary for LyCORIS config."""
        return {
            "linear_dim": self.linear_dim,
            "linear_alpha": self.linear_alpha,
            "factor": self.factor,
            "decompose_both": self.decompose_both,
            "use_tucker": self.use_tucker,
            "use_scalar": self.use_scalar,
            "weight_decompose": self.weight_decompose,
            "target_modules": self.target_modules,
            "full_matrix": self.full_matrix,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
            "unbalanced_factorization": self.unbalanced_factorization,
        }


@dataclass
class LoHAConfig:
    """Configuration for LoHA (Low-Rank Hadamard Product) training.

    Uses LyCORIS with ``algo="loha"``.  Richer parameter interactions
    per rank than LoKR — suited for large, diverse datasets.
    """

    linear_dim: int = 64
    linear_alpha: int = 128
    factor: int = -1
    use_tucker: bool = False
    use_scalar: bool = False
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    full_matrix: bool = False
    bypass_mode: bool = False
    rs_lora: bool = False

    def to_dict(self):
        """Convert to dictionary for LyCORIS config."""
        return {
            "linear_dim": self.linear_dim,
            "linear_alpha": self.linear_alpha,
            "factor": self.factor,
            "use_tucker": self.use_tucker,
            "use_scalar": self.use_scalar,
            "target_modules": self.target_modules,
            "full_matrix": self.full_matrix,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
        }


@dataclass
class OFTConfig:
    """Configuration for OFT (Orthogonal Fine-Tuning) training.

    Uses PEFT ``OFTConfig``.  Constrains weight updates to orthogonal
    transformations, preserving pretrained knowledge.  Experimental
    for audio DiT models.
    """

    block_size: int = 64
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    coft: bool = False
    """Constrained OFT — projects updates back to the Cayley manifold."""
    eps: float = 6e-5
    """Epsilon for numerical stability in constrained OFT."""

    def to_dict(self):
        """Convert to dictionary for PEFT OFT config."""
        return {
            "block_size": self.block_size,
            "target_modules": self.target_modules,
            "coft": self.coft,
            "eps": self.eps,
        }


@dataclass
class TrainingConfig:
    """Configuration for training process.

    Training uses:
    - Device-aware mixed precision (bf16 on CUDA/XPU, fp16 on MPS, fp32 on CPU)
    - Discrete timesteps from turbo shift=3.0 schedule (8 steps)
    - Randomly samples one of 8 timesteps per training step:
      [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]
    """
    # Fixed for turbo model
    shift: float = 3.0
    num_inference_steps: int = 8
    learning_rate: float = 3e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_epochs: int = 1000
    save_every_n_epochs: int = 50
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 42
    output_dir: str = "./lora_output"

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory_device: str = ""

    # Logging
    log_every_n_steps: int = 10

    # Validation (for loss curve and best-checkpoint tracking)
    val_split: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError("val_split must be in [0.0, 1.0).")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "shift": self.shift,
            "num_inference_steps": self.num_inference_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "save_every_n_epochs": self.save_every_n_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "pin_memory_device": self.pin_memory_device,
            "log_every_n_steps": self.log_every_n_steps,
            "val_split": self.val_split,
        }
