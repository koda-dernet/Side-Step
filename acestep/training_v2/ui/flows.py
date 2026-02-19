"""
Wizard flow builders -- facade module.

Re-exports ``wizard_train``, ``wizard_preprocess``, and ``wizard_estimate``
from their dedicated modules so existing ``from .flows import ...`` imports
continue to work after the step-based refactor.
"""

from acestep.training_v2.ui.flows_train import wizard_train
from acestep.training_v2.ui.flows_preprocess import wizard_preprocess
from acestep.training_v2.ui.flows_estimate import wizard_estimate
from acestep.training_v2.ui.flows_fisher import wizard_preprocessing_pp, wizard_fisher
from acestep.training_v2.ui.flows_resume import wizard_resume

__all__ = [
    "wizard_train",
    "wizard_preprocess",
    "wizard_estimate",
    "wizard_preprocessing_pp",
    "wizard_fisher",
    "wizard_resume",
]
