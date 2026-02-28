from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidestep_engine.ui import presets


class TestPresetErrorMessages(unittest.TestCase):
    def test_import_missing_file_sets_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "presets"
            with patch("sidestep_engine.ui.presets._local_presets_dir", return_value=local):
                name = presets.import_preset(str(Path(td) / "missing.json"))
                self.assertIsNone(name)
                err = presets.get_last_preset_error(clear=True)
                self.assertIsNotNone(err)
                self.assertIn("File not found", err or "")

    def test_import_invalid_json_sets_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "broken.json"
            src.write_text("{bad json}", encoding="utf-8")
            local = Path(td) / "presets"
            with patch("sidestep_engine.ui.presets._local_presets_dir", return_value=local):
                name = presets.import_preset(str(src))
                self.assertIsNone(name)
                err = presets.get_last_preset_error(clear=True)
                self.assertIsNotNone(err)
                self.assertIn("Invalid preset JSON", err or "")

    def test_import_invalid_name_sets_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "invalid_name.json"
            src.write_text(json.dumps({"name": "bad<name"}), encoding="utf-8")
            local = Path(td) / "presets"
            with patch("sidestep_engine.ui.presets._local_presets_dir", return_value=local):
                name = presets.import_preset(str(src))
                self.assertIsNone(name)
                err = presets.get_last_preset_error(clear=True)
                self.assertIsNotNone(err)
                self.assertIn("Invalid preset name", err or "")


if __name__ == "__main__":
    unittest.main()

