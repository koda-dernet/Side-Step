"""Regression: LyCORIS adapter params must ALL be unfrozen after injection.

The old code had a post-injection name-matching filter that could
accidentally freeze valid LoKr/LoHA tensors (e.g. lokr_w2_b), causing
silent quality regression with many all-zero saved tensors.

The fix: trust LyCORIS apply_preset + create_lycoris target selection
and enable requires_grad on ALL adapter module params unconditionally.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch


def _make_fake_lycoris_module(name: str, param_shapes: list[tuple]) -> MagicMock:
    """Create a fake LyCORIS module with named parameters."""
    module = MagicMock()
    module.lora_name = name
    params = []
    for shape in param_shapes:
        p = torch.nn.Parameter(torch.randn(*shape))
        p.requires_grad = False  # start frozen
        params.append(p)
    module.parameters = MagicMock(return_value=params)
    return module


def _make_fake_model():
    """Create a minimal fake model with a decoder."""
    model = MagicMock()
    model.decoder = MagicMock()
    # named_parameters yields nothing (already frozen by inject fn)
    model.named_parameters = MagicMock(return_value=[])
    model.parameters = MagicMock(return_value=[
        torch.nn.Parameter(torch.randn(10, 10)),
    ])
    return model


class TestLoKrAllParamsUnfrozen(unittest.TestCase):
    """LoKR: every adapter param must have requires_grad=True after injection."""

    @patch("sidestep_engine.vendor.lokr_utils.create_lycoris")
    @patch("sidestep_engine.vendor.lokr_utils.LycorisNetwork")
    @patch("sidestep_engine.vendor.lokr_utils.LYCORIS_AVAILABLE", True)
    def test_all_lokr_params_unfrozen(self, mock_network_cls, mock_create):
        """Even params with non-matching names must be trainable."""
        from sidestep_engine.vendor.lokr_utils import inject_lokr_into_dit
        from sidestep_engine.vendor.configs import LoKRConfig

        # Two fake modules: one with a matching name, one with an internal name
        mod_match = _make_fake_lycoris_module(
            "layers.0.self_attn.q_proj", [(4, 4)]
        )
        mod_internal = _make_fake_lycoris_module(
            "layers.0.self_attn.q_proj.lokr_w2_b", [(4, 4)]
        )

        fake_net = MagicMock()
        fake_net.loras = [mod_match, mod_internal]
        fake_net.parameters = MagicMock(return_value=[])
        mock_create.return_value = fake_net

        model = _make_fake_model()
        cfg = LoKRConfig(target_modules=["q_proj"])

        _, lycoris_net, info = inject_lokr_into_dit(model, cfg)

        # ALL params from BOTH modules must be unfrozen
        for mod in [mod_match, mod_internal]:
            for p in mod.parameters():
                self.assertTrue(
                    p.requires_grad,
                    f"Param in module '{mod.lora_name}' should be trainable",
                )

        self.assertGreater(info["trainable_params"], 0)


class TestLoHAAllParamsUnfrozen(unittest.TestCase):
    """LoHA: every adapter param must have requires_grad=True after injection."""

    @patch("sidestep_engine.vendor.loha_utils.create_lycoris")
    @patch("sidestep_engine.vendor.loha_utils.LycorisNetwork")
    @patch("sidestep_engine.vendor.loha_utils.LYCORIS_AVAILABLE", True)
    def test_all_loha_params_unfrozen(self, mock_network_cls, mock_create):
        """Even params with non-matching names must be trainable."""
        from sidestep_engine.vendor.loha_utils import inject_loha_into_dit
        from sidestep_engine.vendor.configs import LoHAConfig

        mod_match = _make_fake_lycoris_module(
            "layers.0.self_attn.v_proj", [(4, 4)]
        )
        mod_internal = _make_fake_lycoris_module(
            "layers.0.self_attn.v_proj.hada_w1_b", [(4, 4)]
        )

        fake_net = MagicMock()
        fake_net.loras = [mod_match, mod_internal]
        fake_net.parameters = MagicMock(return_value=[])
        mock_create.return_value = fake_net

        model = _make_fake_model()
        cfg = LoHAConfig(target_modules=["v_proj"])

        _, lycoris_net, info = inject_loha_into_dit(model, cfg)

        for mod in [mod_match, mod_internal]:
            for p in mod.parameters():
                self.assertTrue(
                    p.requires_grad,
                    f"Param in module '{mod.lora_name}' should be trainable",
                )

        self.assertGreater(info["trainable_params"], 0)


class TestLoKrFallbackPath(unittest.TestCase):
    """When lycoris_net.loras is empty, fall back to lycoris_net.parameters()."""

    @patch("sidestep_engine.vendor.lokr_utils.create_lycoris")
    @patch("sidestep_engine.vendor.lokr_utils.LycorisNetwork")
    @patch("sidestep_engine.vendor.lokr_utils.LYCORIS_AVAILABLE", True)
    def test_fallback_enables_all(self, mock_network_cls, mock_create):
        """Empty .loras → all lycoris_net.parameters() must be unfrozen."""
        from sidestep_engine.vendor.lokr_utils import inject_lokr_into_dit
        from sidestep_engine.vendor.configs import LoKRConfig

        p1 = torch.nn.Parameter(torch.randn(4, 4))
        p1.requires_grad = False

        fake_net = MagicMock()
        fake_net.loras = []
        fake_net.parameters = MagicMock(return_value=[p1])
        mock_create.return_value = fake_net

        model = _make_fake_model()
        cfg = LoKRConfig()

        inject_lokr_into_dit(model, cfg)
        self.assertTrue(p1.requires_grad)


if __name__ == "__main__":
    unittest.main()
