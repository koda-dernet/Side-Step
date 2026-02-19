from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.training_v2.ui.dependency_check import (
    OptionalDependency,
    ensure_optional_dependencies,
    required_preprocess_optionals,
    required_training_optionals,
)


class TestDependencyPreflightSelection(unittest.TestCase):
    def test_training_selects_bnb_and_tensorboard(self) -> None:
        cfg = SimpleNamespace(optimizer_type="adamw8bit", log_every=10, log_heavy_every=0)
        deps = required_training_optionals(cfg)
        keys = {d.key for d in deps}
        self.assertIn("bitsandbytes", keys)
        self.assertIn("tensorboard", keys)

    def test_training_selects_prodigy_and_tensorboard(self) -> None:
        cfg = SimpleNamespace(optimizer_type="prodigy", log_every=0, log_heavy_every=50)
        deps = required_training_optionals(cfg)
        keys = {d.key for d in deps}
        self.assertIn("prodigyopt", keys)
        self.assertIn("tensorboard", keys)
        self.assertNotIn("bitsandbytes", keys)

    def test_preprocess_selects_lufs_dependency(self) -> None:
        deps = required_preprocess_optionals("lufs")
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].key, "pyloudnorm")


class TestDependencyPromptFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.dep = OptionalDependency(
            key="demo",
            module="demo_pkg",
            install_spec="demo_pkg>=1.0.0",
            reason="demo reason",
            impact_if_missing="demo impact",
        )

    def test_decline_install_keeps_dependency_unresolved(self) -> None:
        with (
            patch("acestep.training_v2.ui.dependency_check._has_module", return_value=False),
            patch("acestep.training_v2.ui.dependency_check._ask_yes_no", return_value=False),
        ):
            unresolved = ensure_optional_dependencies(
                [self.dep], interactive=True, allow_install_prompt=True
            )
        self.assertEqual([d.key for d in unresolved], ["demo"])

    def test_accept_install_resolves_dependency(self) -> None:
        with (
            patch(
                "acestep.training_v2.ui.dependency_check._has_module",
                side_effect=[False, True],
            ),
            patch("acestep.training_v2.ui.dependency_check._ask_yes_no", return_value=True),
            patch("acestep.training_v2.ui.dependency_check._install_dep", return_value=True),
        ):
            unresolved = ensure_optional_dependencies(
                [self.dep], interactive=True, allow_install_prompt=True
            )
        self.assertEqual(unresolved, [])

    def test_noninteractive_mode_does_not_prompt(self) -> None:
        with (
            patch("acestep.training_v2.ui.dependency_check._has_module", return_value=False),
            patch("acestep.training_v2.ui.dependency_check._ask_yes_no") as ask_mock,
        ):
            unresolved = ensure_optional_dependencies(
                [self.dep], interactive=False, allow_install_prompt=True
            )
        ask_mock.assert_not_called()
        self.assertEqual([d.key for d in unresolved], ["demo"])


if __name__ == "__main__":
    unittest.main()

