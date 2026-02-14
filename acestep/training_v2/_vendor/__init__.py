"""Vendored ACE-Step modules for standalone Side-Step operation.

These files are snapshots of ``acestep.training.*`` modules bundled so that
Side-Step's *corrected* (fixed) training loop and preprocessing pipeline can
run without a full ACE-Step installation.

Vanilla training mode still requires base ACE-Step (``acestep.training.trainer``).
"""

# -- Configs ----------------------------------------------------------------
from acestep.training_v2._vendor.configs import (  # noqa: F401
    LoRAConfig,
    LoKRConfig,
    TrainingConfig,
)

# -- LoRA utilities ---------------------------------------------------------
from acestep.training_v2._vendor.lora_utils import (  # noqa: F401
    check_peft_available,
    inject_lora_into_dit,
    load_lora_weights,
    load_training_checkpoint,
    save_lora_weights,
    save_training_checkpoint,
)

# -- LoKR utilities ---------------------------------------------------------
from acestep.training_v2._vendor.lokr_utils import (  # noqa: F401
    check_lycoris_available,
    inject_lokr_into_dit,
    load_lokr_weights,
    save_lokr_training_checkpoint,
    save_lokr_weights,
)

# -- Data module ------------------------------------------------------------
from acestep.training_v2._vendor.data_module import (  # noqa: F401
    PreprocessedDataModule,
)

# -- Preprocessing utilities ------------------------------------------------
from acestep.training_v2._vendor.preprocess_audio import load_audio_stereo  # noqa: F401
from acestep.training_v2._vendor.preprocess_lyrics import encode_lyrics  # noqa: F401
from acestep.training_v2._vendor.preprocess_encoder import run_encoder  # noqa: F401
from acestep.training_v2._vendor.preprocess_context import build_context_latents  # noqa: F401
from acestep.training_v2._vendor.preprocess_text import encode_text  # noqa: F401

# -- Constants --------------------------------------------------------------
from acestep.training_v2._vendor.constants import (  # noqa: F401
    DEFAULT_DIT_INSTRUCTION,
    SFT_GEN_PROMPT,
)
