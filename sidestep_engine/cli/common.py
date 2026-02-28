"""
Backward-compatible re-exports for ACE-Step Training V2 CLI.

Split structure:
    args.py           -- argparse construction (build_root_parser, _add_* helpers)
    validation.py     -- path validation and target-module resolution
    config_builder.py -- LoRA/Training config construction from parsed args
    common.py         -- re-exports for API compatibility (this file)

Usage (unchanged)::

    from sidestep_engine.cli.common import build_root_parser, build_configs
    from sidestep_engine.cli.common import validate_paths, resolve_target_modules
"""

from sidestep_engine.cli.args import (  # noqa: F401
    build_root_parser,
    VARIANT_DIR_MAP,
    _DEFAULT_NUM_WORKERS,
)
from sidestep_engine.cli.validation import (  # noqa: F401
    validate_paths,
    resolve_target_modules,
)
from sidestep_engine.cli.config_builder import (  # noqa: F401
    build_configs,
    build_configs_from_dict,
)
