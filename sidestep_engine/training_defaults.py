"""
Canonical training defaults shared across entrypoints.

These values are used to keep CLI, Wizard, and GUI behavior aligned.
"""

from __future__ import annotations

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 1000
DEFAULT_SAVE_EVERY = 50
DEFAULT_OPTIMIZER_TYPE = "adamw8bit"

