from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.training_v2.ui.prompt_helpers import ask_path


class TestAskPathErrors(unittest.TestCase):
    def test_permission_denied_then_success(self) -> None:
        with (
            patch(
                "acestep.training_v2.ui.prompt_helpers.ask",
                side_effect=["/tmp/denied", "/tmp/ok"],
            ),
            patch(
                "pathlib.Path.exists",
                side_effect=[PermissionError("denied"), True],
            ),
            patch("pathlib.Path.stat", return_value=None),
            patch("pathlib.Path.resolve", side_effect=lambda *args, **kwargs: "/tmp/ok"),
        ):
            out = ask_path("Dataset dir", must_exist=True)
        self.assertIn("/tmp/ok", str(out))

    def test_not_found_then_success(self) -> None:
        with (
            patch(
                "acestep.training_v2.ui.prompt_helpers.ask",
                side_effect=["/tmp/missing", "/tmp/ok2"],
            ),
            patch("pathlib.Path.exists", side_effect=[False, True]),
            patch("pathlib.Path.stat", return_value=None),
            patch("pathlib.Path.resolve", side_effect=lambda *args, **kwargs: "/tmp/ok2"),
        ):
            out = ask_path("Checkpoint dir", must_exist=True)
        self.assertIn("/tmp/ok2", str(out))


if __name__ == "__main__":
    unittest.main()

