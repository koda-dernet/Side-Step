"""API key collection step for the setup wizard.

Extracted from ``setup.py`` to stay within the module LOC policy.
"""

from __future__ import annotations

from sidestep_engine.ui.prompt_helpers import ask, print_rich, section


def _mask(value: str | None) -> str:
    """Return a masked version of a secret, or empty string if unset."""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def collect_api_keys(data: dict) -> None:
    """Prompt for optional API keys used by the AI dataset builder.

    All keys are skippable.  Existing values from *data* are shown as
    masked defaults so the user can keep them by pressing Enter.
    """
    section("API Keys (optional — skip any you don't have)")
    print_rich("  [dim]These are used by 'Build dataset from folder' to generate[/]")
    print_rich("  [dim]captions (Gemini/OpenAI) and fetch lyrics (Genius).[/]")
    print_rich("  [dim]You can add or change them later from Settings.[/]\n")

    from sidestep_engine.data.caption_config import (
        DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL, GEMINI_MODEL_SUGGESTIONS,
    )

    # Gemini
    gemini = ask(
        "Gemini API key (leave empty to skip)",
        default=_mask(data.get("gemini_api_key")),
    )
    if gemini and gemini != _mask(data.get("gemini_api_key")):
        data["gemini_api_key"] = gemini
    if data.get("gemini_api_key"):
        print_rich(
            "  [dim]Available models: "
            + ", ".join(GEMINI_MODEL_SUGGESTIONS)
            + "[/]"
        )
        gm = ask("Gemini model", default=data.get("gemini_model") or DEFAULT_GEMINI_MODEL)
        data["gemini_model"] = gm or DEFAULT_GEMINI_MODEL

    # OpenAI
    openai_key = ask(
        "OpenAI API key (leave empty to skip)",
        default=_mask(data.get("openai_api_key")),
    )
    if openai_key and openai_key != _mask(data.get("openai_api_key")):
        data["openai_api_key"] = openai_key
    if data.get("openai_api_key"):
        om = ask("OpenAI model", default=data.get("openai_model") or DEFAULT_OPENAI_MODEL)
        data["openai_model"] = om or DEFAULT_OPENAI_MODEL
        base = ask("OpenAI base URL (empty = api.openai.com)",
                   default=data.get("openai_base_url") or "")
        data["openai_base_url"] = base or None

    # Genius
    genius = ask(
        "Genius API token (leave empty to skip)",
        default=_mask(data.get("genius_api_token")),
    )
    if genius and genius != _mask(data.get("genius_api_token")):
        data["genius_api_token"] = genius
