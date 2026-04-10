"""
Generate audio captions using a local Qwen2.5-Omni-7B model.

Loads the model lazily on first use and caches it for the duration
of the batch.  Supports two VRAM tiers:

* **8-10 GB** — 4-bit NF4 quantisation via ``bitsandbytes``
* **16 GB+** — bf16 (auto-fallback to fp16 on older GPUs)

Call :func:`unload_model` after a batch to free VRAM.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch

from sidestep_engine.data.caption_config import (
    build_user_prompt,
    get_system_prompt,
    resolve_generation_settings,
)

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
_LOCAL_DEFAULT_MAX_NEW_TOKENS = 320
_LOCAL_MAX_NEW_TOKENS_CAP = 640
_LOCAL_DEFAULT_TEMPERATURE = 0.35
_LOCAL_DEFAULT_TOP_P = 0.85
_LOCAL_DEFAULT_REPETITION_PENALTY = 1.08
_LOCAL_AUDIO_WINDOW_SECONDS = 45.0
_LOCAL_FULL_AUDIO_MAX_SECONDS = 60.0

# ── Module-level cache (lazy-loaded) ────────────────────────────────
_model: Any = None
_processor: Any = None
_loaded_tier: Optional[str] = None
_loaded_cpu_offload: Optional[bool] = None


class LocalCaptionOOMError(RuntimeError):
    pass


class _CancelCriteria:
    """transformers-compatible StoppingCriteria that fires when an Event is set."""

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: ARG002
        return self._event.is_set()


# ── Helpers ─────────────────────────────────────────────────────────

def _clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()


def _resolve_local_generation_settings(
    *,
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    repetition_penalty: Optional[float],
) -> dict[str, float | int]:
    generation = resolve_generation_settings(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    if max_new_tokens is None:
        generation["max_tokens"] = _LOCAL_DEFAULT_MAX_NEW_TOKENS
    if temperature is None:
        generation["temperature"] = _LOCAL_DEFAULT_TEMPERATURE
    if top_p is None:
        generation["top_p"] = _LOCAL_DEFAULT_TOP_P
    if repetition_penalty is None:
        generation["repetition_penalty"] = _LOCAL_DEFAULT_REPETITION_PENALTY
    generation["max_tokens"] = max(
        96,
        min(int(generation["max_tokens"]), _LOCAL_MAX_NEW_TOKENS_CAP),
    )
    generation["temperature"] = max(0.0, min(float(generation["temperature"]), 1.2))
    generation["top_p"] = max(0.1, min(float(generation["top_p"]), 1.0))
    generation["repetition_penalty"] = max(1.0, min(float(generation["repetition_penalty"]), 1.3))
    return generation


def _generation_attempts(
    tier: str,
    generation: dict[str, float | int],
) -> list[dict[str, float | int | bool]]:
    attempts: list[dict[str, float | int | bool]] = []
    base_tokens = int(generation["max_tokens"])

    token_candidates = [base_tokens]
    if tier == "16gb":
        token_candidates.extend([288, 224, 160])
    else:
        token_candidates.extend([224, 160, 128])

    seen: set[tuple[int, bool]] = set()
    for idx, tokens in enumerate(token_candidates):
        clamped_tokens = max(96, min(base_tokens, int(tokens)))
        use_cache = idx == 0
        key = (clamped_tokens, use_cache)
        if key in seen:
            continue
        seen.add(key)
        attempts.append(
            {
                "max_tokens": clamped_tokens,
                "temperature": float(generation["temperature"]),
                "top_p": float(generation["top_p"]),
                "repetition_penalty": float(generation["repetition_penalty"]),
                "use_cache": use_cache,
            }
        )

    return attempts


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch._C.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _audio_window(audio_source: Optional[Path]) -> tuple[Optional[float], Optional[float]]:
    if audio_source is None or not audio_source.is_file():
        return None, None
    try:
        from sidestep_engine.data.audio_duration import get_audio_duration

        duration_s = float(get_audio_duration(str(audio_source)) or 0)
    except Exception as exc:
        logger.debug("Failed to resolve audio duration for %s: %s", audio_source.name, exc)
        return None, None

    if duration_s <= 0 or duration_s <= _LOCAL_FULL_AUDIO_MAX_SECONDS:
        return 0.0, None

    window = min(_LOCAL_AUDIO_WINDOW_SECONDS, duration_s)
    center = duration_s * 0.5
    start = max(0.0, min(duration_s - window, center - (window / 2.0)))
    end = start + window
    return round(start, 2), round(end, 2)

def _transcode_audio_for_caption(audio_path: Path) -> Optional[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    fd, tmp_path = tempfile.mkstemp(prefix="sidestep_caption_", suffix=".wav")
    try:
        Path(tmp_path).unlink(missing_ok=True)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(tmp_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        Path(tmp_path).unlink(missing_ok=True)
        logger.warning("Temporary audio transcode failed for %s: %s", audio_path.name, exc)
        return None
    logger.info("Using temporary transcoded audio for captioning: %s", audio_path.name)
    return Path(tmp_path)


def _build_conversation(
    title: str,
    artist: str,
    lyrics_excerpt: str,
    audio_source: Optional[Path],
) -> list[dict[str, Any]]:
    user_prompt = build_user_prompt(
        title,
        artist,
        lyrics_excerpt,
        audio_attached=bool(audio_source and audio_source.is_file()),
    )
    user_content: list[dict[str, str]] = []
    if audio_source and audio_source.is_file():
        audio_start, audio_end = _audio_window(audio_source)
        audio_payload: dict[str, Any] = {"type": "audio", "audio": str(audio_source)}
        if audio_start is not None:
            audio_payload["audio_start"] = audio_start
        if audio_end is not None:
            audio_payload["audio_end"] = audio_end
        user_content.append(audio_payload)
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": get_system_prompt("local") or ""}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _prepare_inputs(conversation: list[dict[str, Any]]) -> tuple[str, Any, Any, Any]:
    from qwen_omni_utils import process_mm_info

    assert _processor is not None
    text_template = _processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=False,
    )
    return text_template, audios, images, videos

def _resolve_model_path() -> str:
    """Return a local checkpoint path if present, else the HF Hub ID.

    Search order:
    1. ``<user_checkpoint_dir>/../Qwen2.5-Omni-7B/`` (settings-based)
    2. ``<project>/checkpoints/Qwen2.5-Omni-7B/`` (installer default)
    3. HF Hub model ID (downloads from the internet)
    """
    from sidestep_engine.settings import get_checkpoint_dir

    _MODEL_SUBDIR = "Qwen2.5-Omni-7B"

    # 1. User-configured checkpoint directory (may be on another drive)
    user_ckpt = get_checkpoint_dir()
    if user_ckpt:
        # Check both <checkpoint_dir>/Qwen2.5-Omni-7B and <checkpoint_dir>/../Qwen2.5-Omni-7B
        for candidate in [
            Path(user_ckpt) / _MODEL_SUBDIR,
            Path(user_ckpt).parent / _MODEL_SUBDIR,
        ]:
            if candidate.is_dir() and any(candidate.iterdir()):
                logger.info("Using local model weights (from settings): %s", candidate)
                return str(candidate)

    # 2. Project-relative default
    project_root = Path(__file__).resolve().parent.parent.parent
    local_dir = project_root / "checkpoints" / _MODEL_SUBDIR
    if local_dir.is_dir() and any(local_dir.iterdir()):
        logger.info("Using local model weights: %s", local_dir)
        return str(local_dir)

    return MODEL_ID


def _pick_dtype() -> torch.dtype:
    """Pick bf16 on Ampere+ GPUs, fall back to fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.mps.is_available():
        return torch.bfloat16
        #return torch.float32
    return torch.float16


def _resolve_input_device(model: Any) -> torch.device:
    """Return the device holding the model's input embeddings.

    ``model.device`` is unreliable with Accelerate ``device_map`` and
    4-bit quantization.  transformers >=4.57 removed the automatic
    ``get_input_embeddings`` for ``Qwen2_5OmniForConditionalGeneration``
    so we walk the ``thinker → model`` chain manually as a fallback.
    """
    # 1. Standard path (works on transformers <4.57 and most other models)
    try:
        return model.get_input_embeddings().weight.device
    except (NotImplementedError, AttributeError):
        pass

    # 2. Qwen2.5-Omni specific: thinker.model.embed_tokens
    thinker = getattr(model, "thinker", None)
    if thinker is not None:
        inner = getattr(thinker, "model", None)
        if inner is not None:
            embed = getattr(inner, "embed_tokens", None)
            if embed is not None:
                return embed.weight.device

    # 3. Last resort
    return next(model.parameters()).device


def _pick_attention_backend() -> Optional[str]:
    """Pick the most memory-efficient supported attention backend available."""
    if not torch.cuda.is_available():
        return None
    if importlib.util.find_spec("flash_attn") is None:
        return None
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return "flash_attention_2" if major >= 8 else None


def _load_model(tier: str, *, allow_cpu_offload: bool = False) -> None:
    """Load model + processor into module-level cache.

    Args:
        tier: ``"8-10gb"`` for 4-bit NF4, ``"12gb"`` for 8-bit INT8,
            or ``"16gb"`` for native bf16/fp16.
        allow_cpu_offload: Whether Accelerate CPU offload may be used when needed.
    """
    global _model, _processor, _loaded_tier, _loaded_cpu_offload  # noqa: PLW0603

    if _model is not None and _loaded_tier == tier and _loaded_cpu_offload == allow_cpu_offload:
        return  # already loaded at the right tier

    # Unload first if switching tiers
    if _model is not None:
        unload_model()

    from transformers import (
        BitsAndBytesConfig,
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    model_path = _resolve_model_path()
    compute_dtype = _pick_dtype()
    attention_backend = _pick_attention_backend()
    load_started = time.perf_counter()
    logger.info(
        "Loading Qwen2.5-Omni-7B (tier=%s, dtype=%s, attn=%s, cpu_offload=%s) from %s …",
        tier, compute_dtype, attention_backend or "sdpa/default", allow_cpu_offload, model_path,
    )

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attention_backend:
        load_kwargs["attn_implementation"] = attention_backend

    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto" if allow_cpu_offload else {"": "cuda:0"}
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        load_kwargs["device_map"] = "mps"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        load_kwargs["device_map"] = "xpu"
    else:
        load_kwargs["device_map"] = "cpu"

    if tier == "8-10gb":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    elif tier == "12gb":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = compute_dtype

    try:
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs,
        )
        _model.disable_talker()
        _processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        _loaded_tier = tier
        _loaded_cpu_offload = allow_cpu_offload
        logger.info(
            "Model loaded successfully (tier=%s, cpu_offload=%s) in %.2fs.",
            tier,
            allow_cpu_offload,
            time.perf_counter() - load_started,
        )
    except Exception as exc:
        _model = None
        _processor = None
        _loaded_tier = None
        _loaded_cpu_offload = None
        if _is_oom_error(exc):
            _clear_cuda_memory()
            raise LocalCaptionOOMError(
                "Local caption model could not be loaded into GPU memory. "
                "Enable CPU offload or switch tiers."
            ) from exc
        raise


def unload_model() -> None:
    """Free GPU memory occupied by the cached model."""
    global _model, _processor, _loaded_tier, _loaded_cpu_offload  # noqa: PLW0603
    if _model is not None:
        del _model
    if _processor is not None:
        del _processor
    _model = None
    _processor = None
    _loaded_tier = None
    _loaded_cpu_offload = None
    _clear_cuda_memory()
    logger.info("Local captioner model unloaded — VRAM freed.")


# ── Public API ──────────────────────────────────────────────────────

def generate_caption(
    title: str,
    artist: str,
    *,
    audio_path: Optional[Path] = None,
    lyrics_excerpt: str = "",
    tier: str = "16gb",
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    allow_cpu_offload: bool = False,
    stop_event: Optional[threading.Event] = None,
) -> Optional[str]:
    """Generate a caption for a song using the local Qwen2.5-Omni model.

    Args:
        title: Song title.
        artist: Artist name.
        audio_path: Path to the audio file.
        lyrics_excerpt: Optional lyrics text.
        tier: ``"8-10gb"`` or ``"16gb"``.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Raw caption string, or ``None`` on failure.
    """
    total_started = time.perf_counter()
    try:
        _load_model(tier, allow_cpu_offload=allow_cpu_offload)
    except ImportError as exc:
        logger.error(
            "Missing dependency for local captioning: %s. "
            "Install with: uv pip install 'side-step[captions,optimizers]' "
            "(local captioner: transformers, torchvision, bitsandbytes, qwen-omni-utils)",
            exc,
        )
        return None
    except LocalCaptionOOMError:
        raise
    except Exception as exc:
        logger.error("Failed to load local captioner model: %s", exc)
        return None

    assert _model is not None and _processor is not None

    generation = _resolve_local_generation_settings(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    attempts = _generation_attempts(tier, generation)
    audio_sources: list[Optional[Path]] = []
    if audio_path and audio_path.is_file():
        audio_sources.append(audio_path)
    else:
        audio_sources.append(None)
    temp_audio: Optional[Path] = None
    conversation: Optional[list[dict[str, Any]]] = None
    text_template: Optional[str] = None
    audios: Any = None
    images: Any = None
    videos: Any = None
    inputs: Any = None
    text_ids: Any = None
    decoded: Optional[list[str]] = None
    build_inputs_started = 0.0
    generate_started = 0.0
    decode_started = 0.0

    inputs = None
    text_ids = None
    decoded = None
    audios = images = videos = None
    try:
        last_error: Optional[Exception] = None
        for idx, source in enumerate(audio_sources):
            try:
                build_inputs_started = time.perf_counter()
                conversation = _build_conversation(title, artist, lyrics_excerpt, source)
                text_template, audios, images, videos = _prepare_inputs(conversation)
                inputs = _processor(
                    text=text_template,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=False,
                )
                input_device = _resolve_input_device(_model)
                inputs = inputs.to(input_device)
                if idx > 0:
                    logger.warning(
                        "Captioning fell back to temporary transcoded audio for: %s",
                        audio_path.name if audio_path else title,
                    )
                break
            except Exception as exc:
                last_error = exc
                if (
                    idx == 0
                    and audio_path
                    and audio_path.is_file()
                    and temp_audio is None
                ):
                    temp_audio = _transcode_audio_for_caption(audio_path)
                    if temp_audio is not None:
                        audio_sources.append(temp_audio)
                if idx == len(audio_sources) - 1:
                    raise
                logger.warning(
                    "Local audio decode failed for %s, retrying with transcoded audio: %s",
                    audio_path.name if audio_path else title,
                    exc,
                )

        if inputs is None:
            if last_error is not None:
                raise last_error
            return None

        last_oom: Optional[torch._C.OutOfMemoryError] = None
        for attempt in attempts:
            try:
                generate_started = time.perf_counter()
                gen_kwargs: dict[str, Any] = dict(
                    **inputs,
                    use_audio_in_video=False,
                    max_new_tokens=int(attempt["max_tokens"]),
                    temperature=float(attempt["temperature"]),
                    top_p=float(attempt["top_p"]),
                    repetition_penalty=float(attempt["repetition_penalty"]),
                    do_sample=float(attempt["temperature"]) > 0,
                    use_cache=bool(attempt["use_cache"]),
                )
                if stop_event is not None:
                    gen_kwargs["stopping_criteria"] = [_CancelCriteria(stop_event)]
                with torch.inference_mode():
                    text_ids = _model.generate(**gen_kwargs)
                break
            except torch._C.OutOfMemoryError as exc:
                last_oom = exc
                text_ids = None
                _clear_cuda_memory()
                logger.warning(
                    "Local caption OOM for '%s - %s'; retrying with max_new_tokens=%s, use_cache=%s",
                    artist,
                    title,
                    attempt["max_tokens"],
                    attempt["use_cache"],
                )
        else:
            if last_oom is not None:
                raise last_oom

        if stop_event is not None and stop_event.is_set():
            logger.info("Local caption generation cancelled for '%s - %s'.", artist, title)
            return None

        prompt_len = int(inputs.input_ids.shape[1])
        if text_ids.shape[1] > prompt_len:
            text_ids = text_ids[:, prompt_len:]

        decode_started = time.perf_counter()
        decoded = _processor.batch_decode(
            text_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = decoded[0].strip() if decoded else ""
        logger.info(
            "Local caption timing for '%s - %s': prep=%.2fs, generate=%.2fs, decode=%.2fs, total=%.2fs",
            artist,
            title,
            max(0.0, generate_started - build_inputs_started),
            max(0.0, decode_started - generate_started),
            time.perf_counter() - decode_started if decode_started else 0.0,
            time.perf_counter() - total_started,
        )
        if result:
            return result

        logger.warning("Local model returned empty for: %s - %s", artist, title)
        return None
    except torch._C.OutOfMemoryError as exc:
        raise LocalCaptionOOMError(
            "Local captioning ran out of GPU memory. Enable CPU offload or switch tiers."
        ) from exc
    except LocalCaptionOOMError:
        raise
    except Exception as exc:
        logger.error("Local caption generation failed for '%s - %s': %s",
                     artist, title, exc)
        return None
    finally:
        conversation = None
        text_template = None
        audios = None
        images = None
        videos = None
        inputs = None
        text_ids = None
        decoded = None
        if temp_audio is not None:
            temp_audio.unlink(missing_ok=True)
        _clear_cuda_memory()
