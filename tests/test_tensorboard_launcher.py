from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.training_v2.ui.tensorboard_launcher import (
    launch_tensorboard_background,
    should_launch_tensorboard,
    tensorboard_manual_command,
)

_TMP = tempfile.gettempdir()


class TestTensorboardLauncher(unittest.TestCase):
    def test_manual_command_uses_uvx_with_setuptools_constraint(self) -> None:
        log_dir = os.path.join(_TMP, "my runs")
        cmd = tensorboard_manual_command(log_dir)
        self.assertIn('uvx --with "setuptools<70" tensorboard --logdir ', cmd)
        self.assertIn(f'"{log_dir}"', cmd)

    def test_noninteractive_mode_does_not_autolaunch(self) -> None:
        decision = should_launch_tensorboard(
            os.path.join(_TMP, "runs"),
            skip_prompt=True,
            interactive=False,
        )
        self.assertFalse(decision)

    def test_skip_prompt_accepts_default_when_interactive(self) -> None:
        runs_dir = os.path.join(_TMP, "runs")
        self.assertTrue(
            should_launch_tensorboard(
                runs_dir,
                default=True,
                skip_prompt=True,
                interactive=True,
            )
        )
        self.assertFalse(
            should_launch_tensorboard(
                runs_dir,
                default=False,
                skip_prompt=True,
                interactive=True,
            )
        )

    @patch("acestep.training_v2.ui.tensorboard_launcher.subprocess.Popen")
    def test_launch_builds_uvx_command(self, mock_popen) -> None:
        with tempfile.TemporaryDirectory() as td:
            ok, msg = launch_tensorboard_background(Path(td) / "runs")

        self.assertTrue(ok)
        self.assertIn("TensorBoard launched in background", msg)
        self.assertTrue(mock_popen.called)

        cmd = mock_popen.call_args.args[0]
        self.assertEqual(cmd[:4], ["uvx", "--with", "setuptools<70", "tensorboard"])
        self.assertIn("--logdir", cmd)
        self.assertIn("--host", cmd)
        self.assertIn("--port", cmd)

        kwargs = mock_popen.call_args.kwargs
        if sys.platform == "win32":
            self.assertIn("creationflags", kwargs)
        else:
            self.assertTrue(kwargs.get("start_new_session", False))

    @patch(
        "acestep.training_v2.ui.tensorboard_launcher.subprocess.Popen",
        side_effect=FileNotFoundError,
    )
    def test_launch_handles_missing_uvx(self, _mock_popen) -> None:
        ok, msg = launch_tensorboard_background(os.path.join(_TMP, "runs"))
        self.assertFalse(ok)
        self.assertIn("uvx", msg)
        self.assertIn("Run manually", msg)


if __name__ == "__main__":
    unittest.main()
