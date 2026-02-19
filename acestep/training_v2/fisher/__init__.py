"""Fisher Information + Spectral Analysis for adaptive LoRA rank assignment.

Public API:
    run_fisher_analysis  -- full pipeline (Fisher + spectral + rank + preview)
    load_fisher_map      -- load a saved fisher_map.json
    save_fisher_map      -- persist analysis results to JSON
"""

from acestep.training_v2.fisher.analysis import run_fisher_analysis
from acestep.training_v2.fisher.io import load_fisher_map, save_fisher_map

__all__ = ["run_fisher_analysis", "load_fisher_map", "save_fisher_map"]
