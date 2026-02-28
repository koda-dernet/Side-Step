"""Tests for new adapter configuration dataclasses.

Covers DoRA flag on LoRAConfigV2, LoHAConfigV2, and OFTConfigV2
including to_dict, save_json, and from_json round-trips.
"""

import json
import tempfile
import unittest
from pathlib import Path

from sidestep_engine.vendor.configs import LoHAConfig, OFTConfig
from sidestep_engine.core.configs import (
    LoRAConfigV2,
    LoHAConfigV2,
    OFTConfigV2,
)


class TestLoRAConfigV2Dora(unittest.TestCase):
    """Test use_dora field on LoRAConfigV2."""

    def test_dora_default_false(self):
        cfg = LoRAConfigV2()
        self.assertFalse(cfg.use_dora)

    def test_dora_set_true(self):
        cfg = LoRAConfigV2(use_dora=True)
        self.assertTrue(cfg.use_dora)

    def test_dora_in_to_dict(self):
        cfg = LoRAConfigV2(use_dora=True)
        d = cfg.to_dict()
        self.assertTrue(d["use_dora"])

    def test_dora_roundtrip_json(self):
        cfg = LoRAConfigV2(use_dora=True, r=32, alpha=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            cfg.save_json(path)
            loaded = LoRAConfigV2.from_json(path)
        self.assertTrue(loaded.use_dora)
        self.assertEqual(loaded.r, 32)
        self.assertEqual(loaded.alpha, 64)


class TestLoHAConfig(unittest.TestCase):
    """Test base LoHAConfig."""

    def test_defaults(self):
        cfg = LoHAConfig()
        self.assertEqual(cfg.linear_dim, 64)
        self.assertEqual(cfg.linear_alpha, 128)
        self.assertEqual(cfg.factor, -1)
        self.assertFalse(cfg.use_tucker)
        self.assertFalse(cfg.use_scalar)

    def test_to_dict(self):
        cfg = LoHAConfig(linear_dim=32, use_tucker=True)
        d = cfg.to_dict()
        self.assertEqual(d["linear_dim"], 32)
        self.assertTrue(d["use_tucker"])
        self.assertIn("target_modules", d)


class TestLoHAConfigV2(unittest.TestCase):
    """Test extended LoHAConfigV2."""

    def test_extended_fields(self):
        cfg = LoHAConfigV2(attention_type="self", target_mlp=True)
        self.assertEqual(cfg.attention_type, "self")
        self.assertTrue(cfg.target_mlp)

    def test_to_dict_includes_extended(self):
        cfg = LoHAConfigV2(attention_type="cross", target_mlp=True)
        d = cfg.to_dict()
        self.assertEqual(d["attention_type"], "cross")
        self.assertTrue(d["target_mlp"])
        self.assertIn("linear_dim", d)

    def test_json_roundtrip(self):
        cfg = LoHAConfigV2(linear_dim=48, attention_type="self", target_mlp=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loha.json"
            cfg.save_json(path)
            loaded = LoHAConfigV2.from_json(path)
        self.assertEqual(loaded.linear_dim, 48)
        self.assertEqual(loaded.attention_type, "self")
        self.assertTrue(loaded.target_mlp)

    def test_from_json_ignores_unknown_keys(self):
        data = {"linear_dim": 32, "unknown_field": "ignored"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loha.json"
            path.write_text(json.dumps(data))
            loaded = LoHAConfigV2.from_json(path)
        self.assertEqual(loaded.linear_dim, 32)


class TestOFTConfig(unittest.TestCase):
    """Test base OFTConfig."""

    def test_defaults(self):
        cfg = OFTConfig()
        self.assertEqual(cfg.block_size, 64)
        self.assertFalse(cfg.coft)
        self.assertAlmostEqual(cfg.eps, 6e-5)

    def test_to_dict(self):
        cfg = OFTConfig(block_size=32, coft=True)
        d = cfg.to_dict()
        self.assertEqual(d["block_size"], 32)
        self.assertTrue(d["coft"])
        self.assertIn("target_modules", d)


class TestOFTConfigV2(unittest.TestCase):
    """Test extended OFTConfigV2."""

    def test_extended_fields(self):
        cfg = OFTConfigV2(attention_type="self", target_mlp=True)
        self.assertEqual(cfg.attention_type, "self")
        self.assertTrue(cfg.target_mlp)

    def test_to_dict_includes_extended(self):
        cfg = OFTConfigV2(block_size=128, attention_type="cross")
        d = cfg.to_dict()
        self.assertEqual(d["block_size"], 128)
        self.assertEqual(d["attention_type"], "cross")

    def test_json_roundtrip(self):
        cfg = OFTConfigV2(block_size=128, coft=True, eps=1e-4, target_mlp=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oft.json"
            cfg.save_json(path)
            loaded = OFTConfigV2.from_json(path)
        self.assertEqual(loaded.block_size, 128)
        self.assertTrue(loaded.coft)
        self.assertAlmostEqual(loaded.eps, 1e-4)
        self.assertTrue(loaded.target_mlp)

    def test_from_json_ignores_unknown_keys(self):
        data = {"block_size": 32, "unknown_field": "ignored"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oft.json"
            path.write_text(json.dumps(data))
            loaded = OFTConfigV2.from_json(path)
        self.assertEqual(loaded.block_size, 32)


def _peft_oft_available() -> bool:
    try:
        from peft import OFTConfig as _
        return True
    except ImportError:
        return False


class TestOFTPeftConfigCreation(unittest.TestCase):
    """Regression: OFT must use oft_block_size, not r, in PeftOFTConfig."""

    @unittest.skipUnless(
        _peft_oft_available(), "peft >= 0.12.0 required for OFT tests"
    )
    def test_peft_config_uses_oft_block_size_not_r(self):
        """The PeftOFTConfig must have r=0 and oft_block_size=block_size."""
        from unittest.mock import patch, MagicMock

        oft_cfg = OFTConfigV2(block_size=64, target_modules=["q_proj"])

        captured = {}
        _OrigPeftOFTConfig = None
        try:
            from peft import OFTConfig as PeftOFTConfig
            _OrigPeftOFTConfig = PeftOFTConfig
        except ImportError:
            self.skipTest("peft not available")

        def _capture_config(**kwargs):
            captured.update(kwargs)
            return _OrigPeftOFTConfig(**kwargs)

        with patch(
            "sidestep_engine.vendor.oft_utils.PeftOFTConfig",
            side_effect=_capture_config,
        ), patch(
            "sidestep_engine.vendor.oft_utils.get_peft_model",
            return_value=MagicMock(),
        ):
            from sidestep_engine.vendor.oft_utils import inject_oft_into_dit
            model = MagicMock()
            model.decoder = MagicMock(spec=[])
            inject_oft_into_dit(model, oft_cfg)

        self.assertEqual(captured.get("r"), 0, "r must be 0 so PEFT auto-computes from block_size")
        self.assertEqual(captured.get("oft_block_size"), 64)


if __name__ == "__main__":
    unittest.main()
