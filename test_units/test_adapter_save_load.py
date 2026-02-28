"""Tests for adapter save/load dispatch in trainer_helpers.py.

Verifies save_adapter_flat routes to the correct backend for each
adapter type, and verify_saved_adapter recognizes all weight file formats.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sidestep_engine.core.trainer_helpers import (
    save_adapter_flat,
    verify_saved_adapter,
)


def _mock_trainer(adapter_type: str, has_lycoris: bool = False) -> MagicMock:
    """Build a minimal mock trainer for save dispatch tests."""
    trainer = MagicMock()
    trainer.adapter_type = adapter_type
    trainer.module = MagicMock()
    trainer.module.lycoris_net = MagicMock() if has_lycoris else None
    trainer.module.adapter_config = MagicMock()
    trainer.module.adapter_config.to_dict.return_value = {"test": True}
    trainer.module.model = MagicMock()
    return trainer


class TestSaveAdapterFlat(unittest.TestCase):
    """Test save_adapter_flat dispatch."""

    @patch("sidestep_engine.core.trainer_helpers.save_lokr_weights")
    def test_lokr_saves_lokr_weights(self, mock_save):
        trainer = _mock_trainer("lokr", has_lycoris=True)
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter_flat(trainer, tmp)
        mock_save.assert_called_once()

    @patch("sidestep_engine.core.trainer_helpers.save_loha_weights")
    def test_loha_saves_loha_weights(self, mock_save):
        trainer = _mock_trainer("loha", has_lycoris=True)
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter_flat(trainer, tmp)
        mock_save.assert_called_once()

    def test_lokr_raises_without_lycoris_net(self):
        trainer = _mock_trainer("lokr", has_lycoris=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                save_adapter_flat(trainer, tmp)

    def test_loha_raises_without_lycoris_net(self):
        trainer = _mock_trainer("loha", has_lycoris=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                save_adapter_flat(trainer, tmp)

    @patch("sidestep_engine.core.trainer_helpers._unwrap_decoder")
    def test_lora_uses_peft_save(self, mock_unwrap):
        mock_decoder = MagicMock()
        mock_unwrap.return_value = mock_decoder
        trainer = _mock_trainer("lora")
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter_flat(trainer, tmp)
        mock_decoder.save_pretrained.assert_called_once()

    @patch("sidestep_engine.core.trainer_helpers._unwrap_decoder")
    def test_dora_uses_peft_save(self, mock_unwrap):
        mock_decoder = MagicMock()
        mock_unwrap.return_value = mock_decoder
        trainer = _mock_trainer("dora")
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter_flat(trainer, tmp)
        mock_decoder.save_pretrained.assert_called_once()

    @patch("sidestep_engine.core.trainer_helpers._unwrap_decoder")
    def test_oft_uses_peft_save(self, mock_unwrap):
        mock_decoder = MagicMock()
        mock_unwrap.return_value = mock_decoder
        trainer = _mock_trainer("oft")
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter_flat(trainer, tmp)
        mock_decoder.save_pretrained.assert_called_once()


class TestVerifySavedAdapter(unittest.TestCase):
    """Test verify_saved_adapter recognizes different weight files."""

    def test_recognizes_lokr_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lokr_weights.safetensors")
            with open(path, "wb") as f:
                f.write(b"dummy")
            # Should not raise -- just logs
            verify_saved_adapter(tmp)

    def test_recognizes_loha_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "loha_weights.safetensors")
            with open(path, "wb") as f:
                f.write(b"dummy")
            verify_saved_adapter(tmp)

    def test_warns_no_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Empty dir -- should not raise
            verify_saved_adapter(tmp)


if __name__ == "__main__":
    unittest.main()
