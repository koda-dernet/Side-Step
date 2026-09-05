"""Exact replication of the ACE-Step 5Hz planner LM prompt format.

Every byte here mirrors the inference-side prompt builder — the Phase-2
prompt whose completion is the audio-code sequence, plus the CoT YAML
block (byte-verified against a production C++ engine implementation).
If the inference prompt builder changes, this file must change with it,
or LM adapters will be trained against a distribution the engine never
queries (a silent failure mode).

Reference constants (engine/src/prompt.h / task-types.h):
    TOKEN_IM_START   151644   <|im_start|>
    TOKEN_IM_END     151645   <|im_end|>
    TOKEN_THINK      151667   <think>
    TOKEN_THINK_END  151668   </think>
    AUDIO_CODE_BASE  151669   <|audio_code_0|>
    AUDIO_CODE_COUNT 65535
    LM_INSTRUCTION   "Generate audio semantic tokens based on the given conditions:"

Inference-time sequence layout (Phase 2, conditional branch):

    <|im_start|>system
    # Instruction
    Generate audio semantic tokens based on the given conditions:

    <|im_end|>
    <|im_start|>user
    # Caption
    {caption}

    # Lyric
    {lyrics}
    <|im_end|>
    <|im_start|>assistant
    <think>
    {cot_yaml}</think>

    {audio code tokens ...}<|im_end|>

The assistant turn is left open after the ``</think>\\n\\n`` — the LM
generates one ``<|audio_code_N|>`` token per 5Hz frame and terminates
with ``<|im_end|>``.  Phase-2 decoding is constrained to exactly that
token set, so training completions are ``codes + <|im_end|>``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

TOKEN_IM_START = 151644
TOKEN_IM_END = 151645
TOKEN_THINK = 151667
TOKEN_THINK_END = 151668
AUDIO_CODE_BASE = 151669
AUDIO_CODE_COUNT = 65535

LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"

LATENT_FPS_25 = 25
CODE_FPS_5 = 5


def yaml_wrap(key: str, val: str) -> str:
    """Port of prompt.h ``yaml_wrap`` (mimics Python yaml.dump wrapping).

    Word-wraps *val* with a line break + 2-space indent whenever the
    current column exceeds 80.  Byte-exact with the C++.
    """
    result = key + ":"
    col = len(key) + 1
    i = 0
    n = len(val)
    while i < n:
        end = val.find(" ", i)
        if end == -1:
            end = n
        word = val[i:end]
        if col > 80:
            result += "\n  "
            col = 2
        else:
            result += " "
            col += 1
        result += word
        col += len(word)
        i = end + 1 if end < n else n
    return result + "\n"


def build_cot_yaml(
    caption: str = "",
    bpm: int = 0,
    duration: int = 0,
    keyscale: str = "",
    language: str = "",
    timesignature: str = "",
) -> str:
    """Port of prompt.h ``build_cot_yaml`` — keys in sorted order, empties omitted."""
    yaml = ""
    if bpm and bpm > 0:
        yaml += f"bpm: {int(bpm)}\n"
    if caption:
        yaml += yaml_wrap("caption", caption)
    if duration and duration > 0:
        yaml += f"duration: {int(duration)}\n"
    if keyscale:
        yaml += f"keyscale: {keyscale}\n"
    if language:
        yaml += f"language: {language}\n"
    if timesignature:
        yaml += f"timesignature: {timesignature}\n"
    return yaml


def build_phase2_prompt_text(caption: str, lyrics: str) -> str:
    """The prompt up to and including ``assistant\\n`` (always loss-masked)."""
    return (
        "<|im_start|>system\n# Instruction\n" + LM_INSTRUCTION + "\n\n"
        + "<|im_end|>\n"
        + "<|im_start|>user\n# Caption\n" + caption + "\n\n# Lyric\n" + lyrics + "\n"
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def build_think_block_text(cot_yaml: str) -> str:
    """``<think>\\n{yaml}</think>\\n\\n`` — the CoT segment of the assistant turn."""
    return "<think>\n" + cot_yaml + "</think>\n\n"


def build_training_texts(
    meta: Dict[str, Any],
    tagged_caption: str,
    lyrics: str,
) -> Tuple[str, str]:
    """Return ``(prompt_text, think_text)`` for one training sample.

    The caller appends audio-code token IDs (``AUDIO_CODE_BASE + idx``)
    plus ``TOKEN_IM_END`` after tokenizing these strings.
    """
    cot = build_cot_yaml(
        caption=tagged_caption,
        bpm=int(meta.get("bpm") or 0),
        duration=int(meta.get("duration") or 0),
        keyscale=str(meta.get("keyscale") or ""),
        language=str(meta.get("language") or ""),
        timesignature=str(meta.get("timesignature") or ""),
    )
    return build_phase2_prompt_text(tagged_caption, lyrics), build_think_block_text(cot)


def apply_trigger_tag(caption: str, custom_tag: str, tag_position: str = "prepend") -> str:
    """Mirror of the caption tagging used for DiT training and generation."""
    tag = (custom_tag or "").strip()
    if not tag:
        return caption
    if tag_position == "append":
        return f"{caption}, {tag}" if caption else tag
    if tag_position == "replace":
        return tag
    return f"{tag}, {caption}" if caption else tag


def codes_to_token_ids(codes: List[int]) -> List[int]:
    """FSQ indices -> LM vocabulary token IDs, with range validation."""
    ids = []
    for c in codes:
        ci = int(c)
        if not (0 <= ci < AUDIO_CODE_COUNT):
            raise ValueError(f"audio code {ci} out of range [0, {AUDIO_CODE_COUNT})")
        ids.append(AUDIO_CODE_BASE + ci)
    return ids


def verify_tokenizer_atomicity(tokenizer) -> Optional[str]:
    """Sanity-check that the HF tokenizer maps our special strings to the
    exact single token IDs the engine uses.  Returns an error string or None.
    """
    checks = [
        ("<|im_start|>", TOKEN_IM_START),
        ("<|im_end|>", TOKEN_IM_END),
        ("<think>", TOKEN_THINK),
        ("</think>", TOKEN_THINK_END),
        ("<|audio_code_0|>", AUDIO_CODE_BASE),
        ("<|audio_code_65534|>", AUDIO_CODE_BASE + 65534),
    ]
    for text, expected in checks:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids != [expected]:
            return f"tokenizer maps {text!r} to {ids}, expected [{expected}]"
    return None
