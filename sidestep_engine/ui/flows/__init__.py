"""
Wizard flow builders -- facade module.

Re-exports ``wizard_train``, ``wizard_preprocess``, and other flow builders
from their dedicated modules so existing ``from .flows import ...`` imports
continue to work after the step-based refactor.
"""

from sidestep_engine.ui.flows.train import wizard_train
from sidestep_engine.ui.flows.preprocess import wizard_preprocess
from sidestep_engine.ui.flows.fisher import wizard_preprocessing_pp, wizard_fisher
from sidestep_engine.ui.flows.resume import wizard_resume

__all__ = [
    "wizard_train",
    "wizard_preprocess",
    "wizard_preprocessing_pp",
    "wizard_fisher",
    "wizard_resume",
]
