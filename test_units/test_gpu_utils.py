"""Tests for GPU detection and device normalization."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

_MOD = "sidestep_engine.models.gpu_utils"


def _mock_cuda_props(total_memory: int) -> MagicMock:
    """Create a mock CUDA device properties object."""
    props = MagicMock()
    props.total_memory = total_memory
    return props


class TestDetectGpuDeviceNormalization(unittest.TestCase):
    """Regression: bare 'cuda' / 'xpu' must gain ':0' index."""

    @patch(f"{_MOD}.torch")
    def test_bare_cuda_gets_index(self, mock_torch):
        """detect_gpu('cuda') → device='cuda:0'."""
        from sidestep_engine.models.gpu_utils import detect_gpu

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Mock GPU"
        mock_torch.cuda.get_device_properties.return_value = _mock_cuda_props(8 * 1024 ** 3)
        mock_torch.cuda.synchronize = MagicMock()
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024 ** 3, 8 * 1024 ** 3)

        info = detect_gpu(requested_device="cuda")
        self.assertEqual(info.device, "cuda:0")
        self.assertEqual(info.device_type, "cuda")

    @patch(f"{_MOD}.torch")
    def test_cuda_with_index_preserved(self, mock_torch):
        """detect_gpu('cuda:1') keeps the explicit index."""
        from sidestep_engine.models.gpu_utils import detect_gpu

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Mock GPU"
        mock_torch.cuda.get_device_properties.return_value = _mock_cuda_props(8 * 1024 ** 3)
        mock_torch.cuda.synchronize = MagicMock()
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024 ** 3, 8 * 1024 ** 3)

        info = detect_gpu(requested_device="cuda:1")
        self.assertEqual(info.device, "cuda:1")

    @patch(f"{_MOD}.torch")
    def test_bare_xpu_gets_index(self, mock_torch):
        """detect_gpu('xpu') → device='xpu:0'."""
        from sidestep_engine.models.gpu_utils import detect_gpu

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_xpu = MagicMock()
        mock_xpu.is_available.return_value = True
        mock_xpu.get_device_properties.return_value = MagicMock(total_memory=0)
        mock_torch.xpu = mock_xpu

        info = detect_gpu(requested_device="xpu")
        self.assertEqual(info.device, "xpu:0")

    @patch(f"{_MOD}.torch")
    def test_mps_unchanged(self, mock_torch):
        """detect_gpu('mps') stays as 'mps' (no index needed)."""
        from sidestep_engine.models.gpu_utils import detect_gpu

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        info = detect_gpu(requested_device="mps")
        self.assertEqual(info.device, "mps")

    @patch(f"{_MOD}.torch")
    def test_cpu_unchanged(self, mock_torch):
        """detect_gpu('cpu') stays as 'cpu'."""
        from sidestep_engine.models.gpu_utils import detect_gpu

        mock_torch.cuda.is_available.return_value = False

        info = detect_gpu(requested_device="cpu")
        self.assertEqual(info.device, "cpu")


class TestBestCudaDevice(unittest.TestCase):
    """Multi-CUDA: auto should pick the device with the most VRAM."""

    @patch(f"{_MOD}.torch")
    def test_single_gpu_returns_zero(self, mock_torch):
        """Single CUDA device → always index 0."""
        from sidestep_engine.models.gpu_utils import _best_cuda_device

        mock_torch.cuda.device_count.return_value = 1
        self.assertEqual(_best_cuda_device(), 0)

    @patch(f"{_MOD}.torch")
    def test_two_gpus_picks_larger(self, mock_torch):
        """Two CUDA devices → picks the one with more VRAM."""
        from sidestep_engine.models.gpu_utils import _best_cuda_device

        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_properties.side_effect = [
            _mock_cuda_props(4 * 1024 ** 3),   # cuda:0 = 4 GiB
            _mock_cuda_props(32 * 1024 ** 3),   # cuda:1 = 32 GiB
        ]
        mock_torch.cuda.get_device_name.side_effect = ["Weak GPU", "RTX 5090"]
        self.assertEqual(_best_cuda_device(), 1)

    @patch(f"{_MOD}.torch")
    def test_two_gpus_zero_is_best(self, mock_torch):
        """Two CUDA devices where cuda:0 has more VRAM → stays at 0."""
        from sidestep_engine.models.gpu_utils import _best_cuda_device

        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_properties.side_effect = [
            _mock_cuda_props(32 * 1024 ** 3),   # cuda:0 = 32 GiB
            _mock_cuda_props(4 * 1024 ** 3),    # cuda:1 = 4 GiB
        ]
        self.assertEqual(_best_cuda_device(), 0)

    @patch(f"{_MOD}.torch")
    def test_auto_uses_best_cuda(self, mock_torch):
        """detect_gpu('auto') uses _best_cuda_device for multi-GPU."""
        from sidestep_engine.models.gpu_utils import detect_gpu

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_properties.side_effect = [
            _mock_cuda_props(4 * 1024 ** 3),    # cuda:0 = 4 GiB (scan)
            _mock_cuda_props(32 * 1024 ** 3),   # cuda:1 = 32 GiB (scan)
            _mock_cuda_props(32 * 1024 ** 3),   # cuda:1 again (resolve name+VRAM)
        ]
        mock_torch.cuda.get_device_name.side_effect = [
            "Weak GPU", "RTX 5090",  # _best_cuda_device log
            "RTX 5090",              # detect_gpu name resolve
        ]
        mock_torch.cuda.synchronize = MagicMock()
        mock_torch.cuda.mem_get_info.return_value = (28 * 1024 ** 3, 32 * 1024 ** 3)

        info = detect_gpu(requested_device="auto")
        self.assertEqual(info.device, "cuda:1")
        self.assertEqual(info.name, "RTX 5090")


if __name__ == "__main__":
    unittest.main()
