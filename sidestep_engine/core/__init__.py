"""Core training modules: configs, trainer, optimizer, LoRA/LoKR module.

Also exports shared types (``TrainingUpdate``) so that all layers can
import from ``core`` without depending on the UI layer.
"""

from sidestep_engine.core.types import TrainingUpdate

__all__ = ["TrainingUpdate"]
