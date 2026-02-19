"""Tests for training memory feature helpers."""

from __future__ import annotations

import unittest

import torch.nn as nn

from acestep.training_v2.trainer_helpers import force_disable_decoder_cache


class _Cfg:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache


class _Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg(use_cache=True)


class _Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Leaf()


class TestForceDisableDecoderCache(unittest.TestCase):
    """Ensure cache flag is forced off through wrappers."""

    def test_force_disable_on_wrapped_module(self):
        decoder = _Wrapped()
        changed = force_disable_decoder_cache(decoder)
        self.assertTrue(changed)
        self.assertFalse(decoder.model.config.use_cache)

    def test_no_change_when_already_disabled(self):
        decoder = _Leaf()
        decoder.config.use_cache = False
        changed = force_disable_decoder_cache(decoder)
        self.assertFalse(changed)
        self.assertFalse(decoder.config.use_cache)


if __name__ == "__main__":
    unittest.main()

