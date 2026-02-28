"""Tests for Fabric device selection in FixedLoRATrainer.

Regression test for the bug where selecting any CUDA device other than
cuda:0 causes a tensor location mismatch during training because
``Fabric(devices=1)`` always resolves to cuda:0.

The fix uses ``SingleDeviceStrategy(device=...)`` to explicitly target
the device the model is already on.
"""

import unittest
from unittest.mock import MagicMock, patch, call
import torch


class TestFabricDeviceSelection(unittest.TestCase):
    """Verify that _train_fabric targets the correct CUDA device."""

    def _make_trainer(self, device_str: str):
        """Build a minimal FixedLoRATrainer with mocked internals."""
        from sidestep_engine.core.trainer import FixedLoRATrainer

        model = MagicMock()
        adapter_config = MagicMock()
        training_config = MagicMock()
        training_config.device = device_str

        trainer = FixedLoRATrainer(model, adapter_config, training_config)
        return trainer

    @patch("sidestep_engine.core.trainer._FABRIC_AVAILABLE", True)
    @patch("sidestep_engine.core.trainer.torch.cuda.set_device")
    @patch("sidestep_engine.core.trainer.SingleDeviceStrategy")
    @patch("sidestep_engine.core.trainer.Fabric")
    def test_cuda1_uses_single_device_strategy(self, MockFabric, MockStrategy, mock_set_device):
        """Fabric must target cuda:1 when the module lives on cuda:1.

        Regression: Fabric(devices=1) always resolved to cuda:0, causing
        'mat1 is on cuda:1, different from other tensors on cuda:0'.
        """
        trainer = self._make_trainer("cuda:1")

        # Simulate what _train_fabric reads from self.module
        mock_module = MagicMock()
        mock_module.device = torch.device("cuda", 1)
        mock_module.device_type = "cuda"
        mock_module.dtype = torch.bfloat16
        trainer.module = mock_module

        # We only need to verify the Fabric construction, not the full loop.
        mock_fabric_instance = MagicMock()
        MockFabric.return_value = mock_fabric_instance

        mock_strategy_instance = MagicMock()
        MockStrategy.return_value = mock_strategy_instance

        # Provide a minimal training config
        cfg = MagicMock()
        cfg.output_dir = "/tmp/test_output"
        cfg.device = "cuda:1"
        trainer.training_config = cfg

        # Run enough of _train_fabric to reach Fabric init (generator)
        gen = trainer._train_fabric(MagicMock(), None)
        try:
            while not MockFabric.called:
                next(gen)
        except StopIteration:
            pass

        # Verify torch.cuda.set_device was called with the right device
        mock_set_device.assert_called_once_with(torch.device("cuda", 1))

        # Verify SingleDeviceStrategy was created with cuda:1
        MockStrategy.assert_called_once()
        strategy_kwargs = MockStrategy.call_args
        strategy_device = strategy_kwargs.kwargs.get("device") or strategy_kwargs.args[0]
        self.assertEqual(
            strategy_device,
            torch.device("cuda", 1),
            "SingleDeviceStrategy must target cuda:1, not cuda:0",
        )

        # Verify Fabric received the strategy (not accelerator+devices)
        MockFabric.assert_called_once()
        fabric_kwargs = MockFabric.call_args.kwargs
        self.assertIn("strategy", fabric_kwargs)
        self.assertNotIn("accelerator", fabric_kwargs)
        self.assertNotIn("devices", fabric_kwargs)

    @patch("sidestep_engine.core.trainer._FABRIC_AVAILABLE", True)
    @patch("sidestep_engine.core.trainer.SingleDeviceStrategy")
    @patch("sidestep_engine.core.trainer.Fabric")
    def test_cuda0_also_uses_strategy(self, MockFabric, MockStrategy):
        """Even cuda:0 should use SingleDeviceStrategy for consistency."""
        trainer = self._make_trainer("cuda:0")

        mock_module = MagicMock()
        mock_module.device = torch.device("cuda", 0)
        mock_module.device_type = "cuda"
        mock_module.dtype = torch.bfloat16
        trainer.module = mock_module

        mock_fabric_instance = MagicMock()
        MockFabric.return_value = mock_fabric_instance
        MockStrategy.return_value = MagicMock()

        cfg = MagicMock()
        cfg.output_dir = "/tmp/test_output"
        cfg.device = "cuda:0"
        trainer.training_config = cfg

        gen = trainer._train_fabric(MagicMock(), None)
        try:
            while not MockFabric.called:
                next(gen)
        except StopIteration:
            pass

        strategy_kwargs = MockStrategy.call_args
        strategy_device = strategy_kwargs.kwargs.get("device") or strategy_kwargs.args[0]
        self.assertEqual(strategy_device, torch.device("cuda", 0))

    @patch("sidestep_engine.core.trainer._FABRIC_AVAILABLE", True)
    @patch("sidestep_engine.core.trainer.SingleDeviceStrategy")
    @patch("sidestep_engine.core.trainer.Fabric")
    def test_cpu_uses_strategy(self, MockFabric, MockStrategy):
        """CPU device should also route through SingleDeviceStrategy."""
        trainer = self._make_trainer("cpu")

        mock_module = MagicMock()
        mock_module.device = torch.device("cpu")
        mock_module.device_type = "cpu"
        mock_module.dtype = torch.float32
        trainer.module = mock_module

        mock_fabric_instance = MagicMock()
        MockFabric.return_value = mock_fabric_instance
        MockStrategy.return_value = MagicMock()

        cfg = MagicMock()
        cfg.output_dir = "/tmp/test_output"
        cfg.device = "cpu"
        trainer.training_config = cfg

        gen = trainer._train_fabric(MagicMock(), None)
        try:
            while not MockFabric.called:
                next(gen)
        except StopIteration:
            pass

        strategy_kwargs = MockStrategy.call_args
        strategy_device = strategy_kwargs.kwargs.get("device") or strategy_kwargs.args[0]
        self.assertEqual(strategy_device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
