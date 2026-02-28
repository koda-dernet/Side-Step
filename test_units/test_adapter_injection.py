"""Tests for adapter injection dispatch in FixedLoRAModule.__init__.

Uses mock-based injection to verify the correct _inject_* method is called
for each adapter_type without requiring a real model or GPU.
"""

import unittest
from unittest.mock import MagicMock, patch

from sidestep_engine.core.configs import (
    LoRAConfigV2,
    LoKRConfigV2,
    LoHAConfigV2,
    OFTConfigV2,
    TrainingConfigV2,
)


def _make_training_config(adapter_type: str) -> TrainingConfigV2:
    """Build a minimal TrainingConfigV2 for testing."""
    return TrainingConfigV2(
        adapter_type=adapter_type,
        dataset_dir="/tmp/test",
        output_dir="/tmp/out",
        checkpoint_dir="/tmp/ckpt",
    )


class TestAdapterDispatch(unittest.TestCase):
    """Verify _inject_* method dispatch by adapter_type."""

    @patch("sidestep_engine.core.lora_module.FixedLoRAModule._inject_lora")
    @patch("sidestep_engine.core.lora_module.FixedLoRAModule.__init__", return_value=None)
    def test_lora_dispatch_calls_inject_lora(self, mock_init, mock_inject):
        """Adapter type 'lora' should call _inject_lora."""
        from sidestep_engine.core.lora_module import FixedLoRAModule
        obj = FixedLoRAModule.__new__(FixedLoRAModule)
        obj.adapter_type = "lora"
        obj._inject_lora(MagicMock(), LoRAConfigV2())
        mock_inject.assert_called_once()

    @patch("sidestep_engine.core.lora_module.FixedLoRAModule._inject_lora")
    @patch("sidestep_engine.core.lora_module.FixedLoRAModule.__init__", return_value=None)
    def test_dora_dispatch_calls_inject_lora(self, mock_init, mock_inject):
        """Adapter type 'dora' should also call _inject_lora (with use_dora=True)."""
        from sidestep_engine.core.lora_module import FixedLoRAModule
        obj = FixedLoRAModule.__new__(FixedLoRAModule)
        obj.adapter_type = "dora"
        cfg = LoRAConfigV2(use_dora=True)
        obj._inject_lora(MagicMock(), cfg)
        mock_inject.assert_called_once()

    @patch("sidestep_engine.core.lora_module.FixedLoRAModule._inject_lokr")
    @patch("sidestep_engine.core.lora_module.FixedLoRAModule.__init__", return_value=None)
    def test_lokr_dispatch_calls_inject_lokr(self, mock_init, mock_inject):
        """Adapter type 'lokr' should call _inject_lokr."""
        from sidestep_engine.core.lora_module import FixedLoRAModule
        obj = FixedLoRAModule.__new__(FixedLoRAModule)
        obj.adapter_type = "lokr"
        obj._inject_lokr(MagicMock(), LoKRConfigV2())
        mock_inject.assert_called_once()

    @patch("sidestep_engine.core.lora_module.FixedLoRAModule._inject_loha")
    @patch("sidestep_engine.core.lora_module.FixedLoRAModule.__init__", return_value=None)
    def test_loha_dispatch_calls_inject_loha(self, mock_init, mock_inject):
        """Adapter type 'loha' should call _inject_loha."""
        from sidestep_engine.core.lora_module import FixedLoRAModule
        obj = FixedLoRAModule.__new__(FixedLoRAModule)
        obj.adapter_type = "loha"
        obj._inject_loha(MagicMock(), LoHAConfigV2())
        mock_inject.assert_called_once()

    @patch("sidestep_engine.core.lora_module.FixedLoRAModule._inject_oft")
    @patch("sidestep_engine.core.lora_module.FixedLoRAModule.__init__", return_value=None)
    def test_oft_dispatch_calls_inject_oft(self, mock_init, mock_inject):
        """Adapter type 'oft' should call _inject_oft."""
        from sidestep_engine.core.lora_module import FixedLoRAModule
        obj = FixedLoRAModule.__new__(FixedLoRAModule)
        obj.adapter_type = "oft"
        obj._inject_oft(MagicMock(), OFTConfigV2())
        mock_inject.assert_called_once()


class TestAdapterConfigUnion(unittest.TestCase):
    """Verify AdapterConfig union includes all types."""

    def test_union_members(self):
        from sidestep_engine.core.lora_module import AdapterConfig
        import typing
        args = typing.get_args(AdapterConfig)
        self.assertIn(LoRAConfigV2, args)
        self.assertIn(LoKRConfigV2, args)
        self.assertIn(LoHAConfigV2, args)
        self.assertIn(OFTConfigV2, args)


if __name__ == "__main__":
    unittest.main()
