"""
Submenu implementations for the wizard session loop.

Extracted from ``wizard.py`` to keep modules under the LOC cap.
"""

from __future__ import annotations

from sidestep_engine.ui.prompt_helpers import ask, menu, print_message, section


def _pick_preset_name(presets: list[dict]) -> str:
    """Show a numbered menu of preset names and return the chosen one.

    Args:
        presets: List of preset info dicts (each must have a ``"name"`` key).

    Returns:
        The selected preset name string.
    """
    options = [(p["name"], p.get("description") or "") for p in presets]
    return menu("Select a preset", options, default=1)


def manage_presets_menu() -> None:
    """Submenu for listing, viewing, deleting, importing, and exporting presets."""
    from sidestep_engine.ui.presets import (
        list_presets, load_preset, delete_preset, import_preset, export_preset,
        get_last_preset_error,
    )

    while True:
        action = menu(
            "Manage Presets",
            [
                ("list", "List all presets"),
                ("view", "View preset details"),
                ("delete", "Delete a user preset"),
                ("import", "Import preset from file"),
                ("export", "Export preset to file"),
                ("back", "Back"),
            ],
            default=6,
        )

        if action == "back":
            return

        presets = list_presets()

        if action == "list":
            if not presets:
                print_message("  No presets found.")
                continue
            section("Available Presets")
            for p in presets:
                tag = " (built-in)" if p["builtin"] else ""
                desc = f" -- {p['description']}" if p["description"] else ""
                print_message(f"    {p['name']}{tag}{desc}")
            print_message("")

        elif action == "view":
            if not presets:
                print_message("  No presets found.")
                continue
            name = _pick_preset_name(presets)
            data = load_preset(name)
            if data is None:
                err = get_last_preset_error(clear=True)
                if err:
                    print_message(f"  Could not load preset '{name}': {err}")
                else:
                    print_message(f"  Preset '{name}' not found.")
            else:
                section(f"Preset: {name}")
                for k, v in sorted(data.items()):
                    print_message(f"    {k}: {v}")
                print_message("")

        elif action == "delete":
            user_presets = [p for p in presets if not p["builtin"]]
            if not user_presets:
                print_message("  No user presets to delete.")
                continue
            name = _pick_preset_name(user_presets)
            if delete_preset(name):
                print_message(f"  Deleted preset '{name}'.")
            else:
                err = get_last_preset_error(clear=True)
                if err:
                    print_message(f"  Could not delete preset '{name}': {err}")
                else:
                    print_message(f"  Preset '{name}' not found (or is built-in).")

        elif action == "import":
            path = ask("Path to preset JSON file", required=True)
            imported = import_preset(path)
            if imported:
                print_message(f"  Imported preset '{imported}'.")
            else:
                err = get_last_preset_error(clear=True)
                if err:
                    print_message(f"  Import failed: {err}")
                else:
                    print_message("  Import failed. Check the file path and format.")

        elif action == "export":
            name = _pick_preset_name(presets) if presets else ask("Preset name", required=True)
            dest = ask("Destination path", required=True)
            if export_preset(name, dest):
                print_message(f"  Exported '{name}' to {dest}.")
            else:
                err = get_last_preset_error(clear=True)
                if err:
                    print_message(f"  Export failed: {err}")
                else:
                    print_message(f"  Preset '{name}' not found.")

