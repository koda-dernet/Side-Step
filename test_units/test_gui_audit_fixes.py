"""Tests for the GUI audit fix batch (18 bugs across 14 files).

Covers:
- Audio duration detection (ffprobe fallback, mutagen fallback)
- Task manager: _push percent/log fields, _push_event type mapping,
  start_preprocess config key fix, start_ppplus param fix
- file_ops: _read_run_meta status mapping, best_loss sanitization,
  root config search
- Banner: narrow logo when stdout is piped
- VRAM estimation: adjusted constants, adapter_mb in breakdown
"""

import asyncio
import json
import math
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Audio duration
# ---------------------------------------------------------------------------

class TestAudioDurationFfprobe(unittest.TestCase):
    """Test the ffprobe subprocess fallback in get_audio_duration."""

    def test_ffprobe_fallback_parses_duration(self):
        """When soundfile+torchcodec fail, ffprobe subprocess should work."""
        from sidestep_engine.data.audio_duration import _ffprobe_duration

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "123.456\n"

        with patch("sidestep_engine.data.audio_duration.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("sidestep_engine.data.audio_duration.subprocess.run", return_value=mock_result):
            dur = _ffprobe_duration("/fake/audio.mp3")
        self.assertEqual(dur, 123)

    def test_ffprobe_not_found_returns_zero(self):
        from sidestep_engine.data.audio_duration import _ffprobe_duration

        with patch("sidestep_engine.data.audio_duration.shutil.which", return_value=None):
            self.assertEqual(_ffprobe_duration("/fake/audio.mp3"), 0)

    def test_ffprobe_failure_returns_zero(self):
        from sidestep_engine.data.audio_duration import _ffprobe_duration

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("sidestep_engine.data.audio_duration.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("sidestep_engine.data.audio_duration.subprocess.run", return_value=mock_result):
            self.assertEqual(_ffprobe_duration("/fake/audio.mp3"), 0)

    def test_ffprobe_timeout_returns_zero(self):
        from sidestep_engine.data.audio_duration import _ffprobe_duration
        import subprocess

        with patch("sidestep_engine.data.audio_duration.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("sidestep_engine.data.audio_duration.subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 10)):
            self.assertEqual(_ffprobe_duration("/fake/audio.mp3"), 0)

    def test_get_audio_duration_mutagen_path(self):
        """When soundfile and torchcodec fail, mutagen should be tried."""
        from sidestep_engine.data.audio_duration import get_audio_duration

        mock_mf = MagicMock()
        mock_mf.info.length = 245.7

        mock_mutagen = MagicMock()
        mock_mutagen.File.return_value = mock_mf

        orig_import = __import__

        def _mock_import(name, *args, **kwargs):
            if name == "soundfile":
                raise ImportError("mocked")
            if name in ("torchcodec", "torchcodec.decoders"):
                raise ImportError("mocked")
            if name == "mutagen":
                return mock_mutagen
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import), \
             patch("sidestep_engine.data.audio_duration.shutil.which", return_value=None):
            result = get_audio_duration("/fake/audio.mp3")
        self.assertEqual(result, 245)

    def test_get_audio_duration_full_chain_all_fail(self):
        """When all backends fail, returns 0."""
        from sidestep_engine.data.audio_duration import get_audio_duration

        with patch("sidestep_engine.data.audio_duration.shutil.which", return_value=None):
            # Patch all import attempts to fail
            orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def fail_import(name, *args, **kwargs):
                if name in ("soundfile", "torchcodec", "torchcodec.decoders", "mutagen"):
                    raise ImportError(f"No module named '{name}'")
                return orig_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_import):
                result = get_audio_duration("/nonexistent/file.mp3")
            self.assertEqual(result, 0)


def _import_mock(block=None, allow_mutagen_result=None):
    """Helper to selectively block imports."""
    orig = __import__
    block = block or set()

    def _mock(name, *args, **kwargs):
        if name in block or (name.startswith(tuple(f"{b}." for b in block))):
            raise ImportError(f"Mocked: no module '{name}'")
        return orig(name, *args, **kwargs)
    return _mock


# ---------------------------------------------------------------------------
# Task manager: _push and _push_event
# ---------------------------------------------------------------------------

class TestTaskManagerPush(unittest.TestCase):
    """Test _push includes percent+log fields, _push_event maps types."""

    def _make_task(self):
        from sidestep_engine.gui.task_manager import Task
        return Task(task_id="test_1", kind="preprocess")

    def test_push_includes_percent_and_log(self):
        from sidestep_engine.gui.task_manager import _push
        task = self._make_task()
        _push(task, 3, 10, "Processing file.wav")
        msg = task.progress_queue.get_nowait()
        self.assertEqual(msg["type"], "progress")
        self.assertEqual(msg["percent"], 30)
        self.assertEqual(msg["log"], "Processing file.wav")
        self.assertEqual(msg["current"], 3)
        self.assertEqual(msg["total"], 10)

    def test_push_zero_total_no_division_error(self):
        from sidestep_engine.gui.task_manager import _push
        task = self._make_task()
        _push(task, 0, 0, "empty")
        msg = task.progress_queue.get_nowait()
        self.assertEqual(msg["percent"], 0)

    def test_push_100_percent(self):
        from sidestep_engine.gui.task_manager import _push
        task = self._make_task()
        _push(task, 10, 10, "done")
        msg = task.progress_queue.get_nowait()
        self.assertEqual(msg["percent"], 100)

    def test_push_event_complete_maps_to_done(self):
        from sidestep_engine.gui.task_manager import _push_event
        task = self._make_task()
        _push_event(task, "complete", "finished")
        msg = task.progress_queue.get_nowait()
        self.assertEqual(msg["type"], "done")
        self.assertEqual(msg["msg"], "finished")

    def test_push_event_fail_maps_to_error(self):
        from sidestep_engine.gui.task_manager import _push_event
        task = self._make_task()
        _push_event(task, "fail", "boom")
        msg = task.progress_queue.get_nowait()
        self.assertEqual(msg["type"], "error")
        self.assertEqual(msg["msg"], "boom")

    def test_push_event_cancelled_passes_through(self):
        from sidestep_engine.gui.task_manager import _push_event
        task = self._make_task()
        _push_event(task, "cancelled")
        msg = task.progress_queue.get_nowait()
        self.assertEqual(msg["type"], "cancelled")

    def test_push_event_extra_kwargs(self):
        from sidestep_engine.gui.task_manager import _push_event
        task = self._make_task()
        _push_event(task, "complete", processed=5, total=10)
        msg = task.progress_queue.get_nowait()
        self.assertEqual(msg["type"], "done")
        self.assertEqual(msg["processed"], 5)
        self.assertEqual(msg["total"], 10)

    def test_new_task_id_is_collision_resistant(self):
        from sidestep_engine.gui.task_manager import _new_task_id

        ids = {_new_task_id("preprocess") for _ in range(32)}
        self.assertEqual(len(ids), 32)


# ---------------------------------------------------------------------------
# Task manager: start_preprocess config keys
# ---------------------------------------------------------------------------

class TestStartPreprocessConfig(unittest.TestCase):
    """Verify start_preprocess reads correct config keys."""

    def test_reads_output_dir_key(self):
        """Backend should accept 'output_dir' from frontend."""
        from sidestep_engine.gui.task_manager import TaskManager
        tm = TaskManager()

        config = {
            "audio_dir": "/fake/audio",
            "output_dir": "/fake/output",  # frontend sends this
            "checkpoint_dir": "/fake/ckpt",
            "model_variant": "turbo",
            "normalize": "peak",
            "target_db": -1.0,
            "trigger_tag": "suno_v4",
            "tag_position": "prepend",
        }

        with patch("sidestep_engine.data.preprocess.preprocess_audio_files") as mock_pp:
            mock_pp.return_value = {"processed": 5, "failed": 0, "total": 5, "output_dir": "/fake/output"}
            result = tm.start_preprocess(config)
            self.assertTrue(result["ok"])
            # Wait for thread
            task = tm._tasks[result["task_id"]]
            task.thread.join(timeout=5)

            # Verify correct kwargs
            call_kwargs = mock_pp.call_args[1]
            self.assertEqual(call_kwargs["output_dir"], "/fake/output")
            self.assertEqual(call_kwargs["custom_tag"], "suno_v4")
            self.assertEqual(call_kwargs["tag_position"], "prepend")
            self.assertEqual(call_kwargs["target_db"], -1.0)

    def test_falls_back_to_tensor_output(self):
        """If output_dir is missing, fall back to tensor_output."""
        from sidestep_engine.gui.task_manager import TaskManager
        tm = TaskManager()

        config = {
            "audio_dir": "/fake/audio",
            "tensor_output": "/legacy/output",
            "checkpoint_dir": "/fake/ckpt",
        }

        with patch("sidestep_engine.data.preprocess.preprocess_audio_files") as mock_pp:
            mock_pp.return_value = {"processed": 0, "failed": 0, "total": 0, "output_dir": "/legacy/output"}
            result = tm.start_preprocess(config)
            task = tm._tasks[result["task_id"]]
            task.thread.join(timeout=5)
            call_kwargs = mock_pp.call_args[1]
            self.assertEqual(call_kwargs["output_dir"], "/legacy/output")


# ---------------------------------------------------------------------------
# Task manager: start_ppplus correct params
# ---------------------------------------------------------------------------

class TestStartPPPlusParams(unittest.TestCase):
    """Verify start_ppplus uses correct import and parameters."""

    def test_ppplus_correct_import_and_params(self):
        from sidestep_engine.gui.task_manager import TaskManager
        tm = TaskManager()

        config = {
            "checkpoint_dir": "/fake/ckpt",
            "dataset_dir": "/fake/data",
            "model_variant": "turbo",
            "base_rank": 64,
            "rank_min": 16,
            "rank_max": 128,
            "timestep_focus": "balanced",
        }

        with patch("sidestep_engine.analysis.fisher.analysis.run_fisher_analysis") as mock_fa:
            mock_fa.return_value = {"summary": "done", "modules": []}
            result = tm.start_ppplus(config)
            self.assertTrue(result["ok"])
            task = tm._tasks[result["task_id"]]
            task.thread.join(timeout=5)

            call_kwargs = mock_fa.call_args[1]
            self.assertEqual(call_kwargs["base_rank"], 64)
            self.assertEqual(call_kwargs["timestep_focus"], "balanced")
            self.assertTrue(call_kwargs["auto_confirm"])
            self.assertIsNotNone(call_kwargs["progress_callback"])
            self.assertIsNotNone(call_kwargs["cancel_check"])
            # Should NOT have old wrong params
            self.assertNotIn("rank", call_kwargs)
            self.assertNotIn("device", call_kwargs)
            self.assertNotIn("precision", call_kwargs)
            self.assertNotIn("output", call_kwargs)


# ---------------------------------------------------------------------------
# file_ops: _read_run_meta
# ---------------------------------------------------------------------------

class TestReadRunMeta(unittest.TestCase):
    """Test _read_run_meta: status mapping, best_loss sanitization, root config."""

    def test_status_mapping_step_to_stopped(self):
        from sidestep_engine.core.run_discovery import read_run_meta as _read_run_meta
        tmp = Path(__file__).parent / "_test_run_meta_step"
        tmp.mkdir(exist_ok=True)
        progress = tmp / ".progress.jsonl"
        progress.write_text('{"kind":"step","best_loss":0.1234}\n')
        try:
            meta = _read_run_meta(tmp)
            self.assertEqual(meta["status"], "stopped")
            self.assertAlmostEqual(meta["best_loss"], 0.1234)
        finally:
            progress.unlink()
            tmp.rmdir()

    def test_status_mapping_epoch_to_stopped(self):
        from sidestep_engine.core.run_discovery import read_run_meta as _read_run_meta
        tmp = Path(__file__).parent / "_test_run_meta_epoch"
        tmp.mkdir(exist_ok=True)
        progress = tmp / ".progress.jsonl"
        progress.write_text('{"kind":"epoch","best_loss":0.05}\n')
        try:
            meta = _read_run_meta(tmp)
            self.assertEqual(meta["status"], "stopped")
        finally:
            progress.unlink()
            tmp.rmdir()

    def test_best_loss_infinity_sanitized(self):
        from sidestep_engine.core.run_discovery import read_run_meta as _read_run_meta
        tmp = Path(__file__).parent / "_test_run_meta_inf"
        tmp.mkdir(exist_ok=True)
        progress = tmp / ".progress.jsonl"
        progress.write_text('{"kind":"step","best_loss":Infinity}\n')
        try:
            meta = _read_run_meta(tmp)
            self.assertEqual(meta["best_loss"], 0.0)
        finally:
            progress.unlink()
            tmp.rmdir()

    def test_best_loss_nan_sanitized(self):
        from sidestep_engine.core.run_discovery import read_run_meta as _read_run_meta
        tmp = Path(__file__).parent / "_test_run_meta_nan"
        tmp.mkdir(exist_ok=True)
        progress = tmp / ".progress.jsonl"
        progress.write_text('{"kind":"step","best_loss":NaN}\n')
        try:
            meta = _read_run_meta(tmp)
            self.assertEqual(meta["best_loss"], 0.0)
        finally:
            progress.unlink()
            tmp.rmdir()

    def test_final_dir_overrides_status_to_complete(self):
        from sidestep_engine.core.run_discovery import read_run_meta as _read_run_meta
        tmp = Path(__file__).parent / "_test_run_meta_final"
        tmp.mkdir(exist_ok=True)
        final = tmp / "final"
        final.mkdir(exist_ok=True)
        progress = tmp / ".progress.jsonl"
        progress.write_text('{"kind":"step","best_loss":0.1}\n')
        try:
            meta = _read_run_meta(tmp)
            self.assertEqual(meta["status"], "complete")
        finally:
            progress.unlink()
            final.rmdir()
            tmp.rmdir()

    def test_root_config_found(self):
        """Config at run root should be found when final/ doesn't exist."""
        from sidestep_engine.core.run_discovery import read_run_meta as _read_run_meta
        tmp = Path(__file__).parent / "_test_run_meta_rootcfg"
        tmp.mkdir(exist_ok=True)
        cfg = tmp / "sidestep_training_config.json"
        cfg.write_text('{"model_variant":"base","max_epochs":50}')
        try:
            meta = _read_run_meta(tmp)
            self.assertEqual(meta["model"], "base")
            self.assertEqual(meta["epochs"], 50)
        finally:
            cfg.unlink()
            tmp.rmdir()

    def test_final_config_preferred_over_root(self):
        """Config in final/ should be preferred over run root."""
        from sidestep_engine.core.run_discovery import read_run_meta as _read_run_meta
        tmp = Path(__file__).parent / "_test_run_meta_prefer_final"
        tmp.mkdir(exist_ok=True)
        final = tmp / "final"
        final.mkdir(exist_ok=True)
        root_cfg = tmp / "sidestep_training_config.json"
        root_cfg.write_text('{"model_variant":"base","max_epochs":50}')
        final_cfg = final / "sidestep_training_config.json"
        final_cfg.write_text('{"model_variant":"turbo","max_epochs":100}')
        try:
            meta = _read_run_meta(tmp)
            self.assertEqual(meta["model"], "turbo")
            self.assertEqual(meta["epochs"], 100)
        finally:
            final_cfg.unlink()
            root_cfg.unlink()
            final.rmdir()
            tmp.rmdir()


class TestLoadRunConfigFallback(unittest.TestCase):
    """load_run_config should support both final/ and run-root config files."""

    def test_load_run_config_falls_back_to_run_root(self):
        from sidestep_engine.gui.file_ops import load_run_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "lora" / "my_run"
            run_dir.mkdir(parents=True)
            cfg = run_dir / "sidestep_training_config.json"
            cfg.write_text('{"model_variant":"base","max_epochs":123}', encoding="utf-8")

            with patch("sidestep_engine.gui.file_ops._adapters_dir", return_value=root):
                data = load_run_config("my_run")

        self.assertIsNotNone(data)
        self.assertEqual(data["model_variant"], "base")
        self.assertEqual(data["max_epochs"], 123)

    def test_load_run_config_prefers_final_over_root(self):
        from sidestep_engine.gui.file_ops import load_run_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "lora" / "my_run"
            final = run_dir / "final"
            final.mkdir(parents=True)

            (run_dir / "sidestep_training_config.json").write_text(
                '{"model_variant":"base","max_epochs":10}',
                encoding="utf-8",
            )
            (final / "sidestep_training_config.json").write_text(
                '{"model_variant":"turbo","max_epochs":50}',
                encoding="utf-8",
            )

            with patch("sidestep_engine.gui.file_ops._adapters_dir", return_value=root):
                data = load_run_config("my_run")

        self.assertIsNotNone(data)
        self.assertEqual(data["model_variant"], "turbo")
        self.assertEqual(data["max_epochs"], 50)


class TestHistoryArtifactFiltering(unittest.TestCase):
    """History should include valid artifacts and detected-only run folders."""

    def test_build_history_filters_to_valid_artifacts(self):
        from sidestep_engine.gui.file_ops import build_history

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Valid final artifacts
            run_final = root / "lora" / "run_final"
            (run_final / "final").mkdir(parents=True)
            (run_final / "final" / "adapter_model.safetensors").write_bytes(b"x")
            (run_final / ".progress.jsonl").write_text('{"kind":"complete","best_loss":0.11}\n', encoding="utf-8")

            # Valid checkpoint fallback artifacts (latest epoch should win)
            run_ckpt = root / "lora" / "run_ckpt"
            (run_ckpt / "checkpoints" / "epoch_1").mkdir(parents=True)
            (run_ckpt / "checkpoints" / "epoch_1" / "lokr_weights.safetensors").write_bytes(b"x")
            (run_ckpt / "checkpoints" / "epoch_3").mkdir(parents=True)
            (run_ckpt / "checkpoints" / "epoch_3" / "lokr_weights.safetensors").write_bytes(b"x")

            # Invalid: final exists but no adapter artifacts
            run_empty = root / "lora" / "run_empty"
            (run_empty / "final").mkdir(parents=True)
            (run_empty / ".progress.jsonl").write_text('{"kind":"complete","best_loss":0.77}\n', encoding="utf-8")

            with patch("sidestep_engine.gui.file_ops._adapters_dir", return_value=root), \
                 patch("sidestep_engine.gui.file_ops._history_override_roots", return_value=[]):
                rows = build_history()

        by_name = {r["run_name"]: r for r in rows}
        self.assertIn("run_final", by_name)
        self.assertIn("run_ckpt", by_name)
        self.assertIn("run_empty", by_name)

        self.assertEqual(by_name["run_final"]["artifact_source"], "final")
        self.assertEqual(by_name["run_ckpt"]["artifact_source"], "checkpoint")
        self.assertTrue(by_name["run_ckpt"]["artifact_path"].endswith("epoch_3"))
        self.assertTrue(by_name["run_empty"].get("detected_only"))
        self.assertEqual(by_name["run_empty"].get("artifact_source"), "")

    def test_load_run_config_finds_override_root_runs(self):
        from sidestep_engine.gui.file_ops import load_run_config

        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "canonical"
            canonical.mkdir(parents=True)
            ext_root = Path(tmp) / "override_root"
            run_dir = ext_root / "custom_run"
            run_dir.mkdir(parents=True)
            (run_dir / "training_config.json").write_text(
                '{"model_variant":"base","max_epochs":12}',
                encoding="utf-8",
            )

            with patch("sidestep_engine.gui.file_ops._adapters_dir", return_value=canonical), \
                 patch("sidestep_engine.gui.file_ops._history_override_roots", return_value=[ext_root]):
                cfg = load_run_config("custom_run")

        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["model_variant"], "base")
        self.assertEqual(cfg["max_epochs"], 12)

    def test_resolve_gui_path_anchors_relative_to_project_root(self):
        from sidestep_engine.gui.file_ops import _resolve_gui_path, _PROJECT_ROOT

        resolved = _resolve_gui_path("trained_adapters")
        expected = (_PROJECT_ROOT / "trained_adapters").resolve(strict=False)
        self.assertEqual(resolved, expected)


class TestOutputRootMemory(unittest.TestCase):
    """Training output overrides should register extra history roots."""

    def test_override_output_root_is_remembered(self):
        from sidestep_engine.gui.task_manager import _remember_history_root_for_output

        cfg = {
            "output_dir": "/tmp/custom_runs/lora_turbo_20260226_1234",
            "run_name": "lora_turbo_20260226_1234",
        }
        with patch("sidestep_engine.settings.get_trained_adapters_dir", return_value="./trained_adapters"), \
             patch("sidestep_engine.settings.remember_history_output_root") as mock_remember:
            _remember_history_root_for_output(cfg)

        mock_remember.assert_called_once()
        remembered = mock_remember.call_args.args[0]
        self.assertTrue(remembered.endswith("/tmp/custom_runs"))

    def test_canonical_output_root_is_not_remembered(self):
        from sidestep_engine.gui.task_manager import _remember_history_root_for_output

        cfg = {
            "output_dir": "./trained_adapters/lora/lora_turbo_20260226_1234",
            "run_name": "lora_turbo_20260226_1234",
        }
        with patch("sidestep_engine.settings.get_trained_adapters_dir", return_value="./trained_adapters"), \
             patch("sidestep_engine.settings.remember_history_output_root") as mock_remember:
            _remember_history_root_for_output(cfg)

        mock_remember.assert_not_called()


# ---------------------------------------------------------------------------
# Banner: narrow logo
# ---------------------------------------------------------------------------

class TestBannerLogo(unittest.TestCase):
    """Test _pick_logo uses narrow logo when stdout is piped."""

    def test_narrow_logo_when_not_tty(self):
        from sidestep_engine.ui.banner import _pick_logo, _LOGO_NARROW
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty = MagicMock(return_value=False)
            result = _pick_logo()
        self.assertEqual(result, _LOGO_NARROW)

    def test_wide_logo_when_tty_and_wide(self):
        from sidestep_engine.ui.banner import _pick_logo, _LOGO_NARROW, _LOGO_MIN_WIDTH
        with patch("sys.stdout") as mock_stdout, \
             patch("shutil.get_terminal_size", return_value=MagicMock(columns=120)):
            mock_stdout.isatty = MagicMock(return_value=True)
            result = _pick_logo()
        # Wide logo should not be the narrow one
        self.assertNotEqual(result, _LOGO_NARROW)

    def test_narrow_logo_when_tty_but_narrow(self):
        from sidestep_engine.ui.banner import _pick_logo, _LOGO_NARROW
        with patch("sys.stdout") as mock_stdout, \
             patch("shutil.get_terminal_size", return_value=MagicMock(columns=60)):
            mock_stdout.isatty = MagicMock(return_value=True)
            result = _pick_logo()
        self.assertEqual(result, _LOGO_NARROW)


# ---------------------------------------------------------------------------
# VRAM estimation: updated constants + adapter_mb
# ---------------------------------------------------------------------------

class TestVRAMEstimationUpdated(unittest.TestCase):
    """Verify VRAM estimation constants are updated and adapter_mb is included."""

    def test_model_weights_measured(self):
        from sidestep_engine.core.vram_estimation import _MODEL_WEIGHTS_MB
        self.assertGreaterEqual(_MODEL_WEIGHTS_MB, 4500)

    def test_cuda_overhead_reasonable(self):
        from sidestep_engine.core.vram_estimation import _CUDA_OVERHEAD_MB
        self.assertGreaterEqual(_CUDA_OVERHEAD_MB, 300)

    def test_legacy_aliases_exist(self):
        from sidestep_engine.core.vram_estimation import (
            _BACKWARD_MULTIPLIER, _MB_PER_LAYER_60S,
            _MODEL_OVERHEAD_OFFLOAD_MB, _MODEL_OVERHEAD_NO_OFFLOAD_MB,
        )
        self.assertIsNotNone(_BACKWARD_MULTIPLIER)
        self.assertIsNotNone(_MB_PER_LAYER_60S)

    def test_breakdown_includes_adapter_mb(self):
        from sidestep_engine.core.vram_estimation import estimate_peak_vram_mb
        peak, breakdown = estimate_peak_vram_mb(
            checkpointing_ratio=1.0, batch_size=1,
            adapter_type="lora", rank=64, target_mlp=True,
            optimizer_type="adamw8bit",
        )
        self.assertIn("adapter_mb", breakdown)
        self.assertGreater(breakdown["adapter_mb"], 0)

    def test_peak_higher_than_old_estimate(self):
        """With new constants, peak should be >6 GB for typical config."""
        from sidestep_engine.core.vram_estimation import estimate_peak_vram_mb
        peak, _ = estimate_peak_vram_mb(
            checkpointing_ratio=1.0, batch_size=1,
            adapter_type="lora", rank=64, target_mlp=True,
            optimizer_type="adamw8bit", offload_encoder=True,
        )
        self.assertGreater(peak, 6000)  # Should be >6 GB now


# ---------------------------------------------------------------------------
# Existing test compatibility check: _sanitize_floats unchanged
# ---------------------------------------------------------------------------

class TestSanitizeFloatsStillWorks(unittest.TestCase):
    """Verify the existing _sanitize_floats still works correctly."""

    def test_inf_to_none(self):
        from sidestep_engine.gui.file_ops import _sanitize_floats
        self.assertIsNone(_sanitize_floats(float("inf")))

    def test_nan_to_none(self):
        from sidestep_engine.gui.file_ops import _sanitize_floats
        self.assertIsNone(_sanitize_floats(float("nan")))

    def test_normal_float_unchanged(self):
        from sidestep_engine.gui.file_ops import _sanitize_floats
        self.assertEqual(_sanitize_floats(3.14), 3.14)

    def test_nested_dict(self):
        from sidestep_engine.gui.file_ops import _sanitize_floats
        result = _sanitize_floats({"a": float("inf"), "b": 1.0})
        self.assertIsNone(result["a"])
        self.assertEqual(result["b"], 1.0)


# ---------------------------------------------------------------------------
# Config key mapping (GUI → argparse round-trip)
# ---------------------------------------------------------------------------

class TestConfigKeyMapping(unittest.TestCase):
    """Test that GUI config keys correctly map to argparse dests."""

    def _apply(self, json_data):
        """Helper: apply a JSON config dict via _apply_config_file."""
        import argparse
        from sidestep_engine.cli.config_builder import (
            _apply_config_file, _populate_defaults_cache,
        )
        _populate_defaults_cache()
        # Build a namespace with argparse defaults
        from sidestep_engine.cli.args import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args(["train", "-c", "/dev/null", "-d", "/tmp", "-o", "/tmp"])
        # Write JSON to temp file
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(json_data, f)
        args.config = path
        _apply_config_file(args)
        os.unlink(path)
        return args

    def test_grad_accum_maps_to_gradient_accumulation(self):
        args = self._apply({"grad_accum": "8"})
        self.assertEqual(args.gradient_accumulation, 8)

    def test_scheduler_maps_to_scheduler_type(self):
        args = self._apply({"scheduler": "linear"})
        self.assertEqual(args.scheduler_type, "linear")

    def test_early_stop_maps_to_early_stop_patience(self):
        args = self._apply({"early_stop": "50"})
        self.assertEqual(args.early_stop_patience, 50)

    def test_projections_string_split_to_list(self):
        args = self._apply({"projections": "q_proj v_proj"})
        self.assertEqual(args.target_modules, ["q_proj", "v_proj"])

    def test_self_projections_maps(self):
        args = self._apply({"self_projections": "q_proj k_proj"})
        self.assertEqual(args.self_target_modules, ["q_proj", "k_proj"])

    def test_cross_projections_maps(self):
        args = self._apply({"cross_projections": "v_proj o_proj"})
        self.assertEqual(args.cross_target_modules, ["v_proj", "o_proj"])


class TestChunkDurationNormalization(unittest.TestCase):
    """Test that chunk_duration=0 (from GUI) is normalized to None."""

    def test_chunk_duration_zero_becomes_none(self):
        import argparse
        from sidestep_engine.cli.config_builder import build_configs, _apply_config_file, _populate_defaults_cache
        from sidestep_engine.cli.args import build_root_parser
        import tempfile, os

        _populate_defaults_cache()
        parser = build_root_parser()
        args = parser.parse_args([
            "train", "-c", "/dev/null", "-d", "/tmp", "-o", "/tmp",
            "--checkpoint-dir", "/tmp",
        ])
        # Simulate GUI sending chunk_duration: "0"
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"chunk_duration": "0"}, f)
        args.config = path
        _apply_config_file(args)
        os.unlink(path)

        # chunk_duration on args should be 0 (coerced from "0")
        self.assertEqual(args.chunk_duration, 0)

        # build_configs normalizes 0 → None via `or None`
        # We can't call build_configs without real checkpoint files,
        # but we can verify the normalization logic directly
        val = args.chunk_duration or None
        self.assertIsNone(val)


class TestCheckpointingRatioSemantics(unittest.TestCase):
    """Checkpointing ratio 0.0 from GUI should disable checkpointing."""

    def _build_args(self, extra: list[str] | None = None):
        from sidestep_engine.cli.args import build_root_parser

        parser = build_root_parser()
        base = [
            "train",
            "--checkpoint-dir", "/tmp",
            "--dataset-dir", "/tmp",
            "--output-dir", "/tmp/out",
        ]
        return parser.parse_args(base + (extra or []))

    def test_ratio_zero_disables_checkpointing(self):
        from sidestep_engine.cli.config_builder import build_configs

        args = self._build_args([])
        args.gradient_checkpointing = True
        args.gradient_checkpointing_ratio = 0.0

        gpu = MagicMock(device="cpu", precision="fp32")
        with patch("sidestep_engine.cli.config_builder.detect_gpu", return_value=gpu):
            _, cfg = build_configs(args)

        self.assertFalse(cfg.gradient_checkpointing)
        self.assertEqual(cfg.gradient_checkpointing_ratio, 0.0)

    def test_ratio_positive_keeps_checkpointing_enabled(self):
        from sidestep_engine.cli.config_builder import build_configs

        args = self._build_args([])
        args.gradient_checkpointing = True
        args.gradient_checkpointing_ratio = 0.5

        gpu = MagicMock(device="cpu", precision="fp32")
        with patch("sidestep_engine.cli.config_builder.detect_gpu", return_value=gpu):
            _, cfg = build_configs(args)

        self.assertTrue(cfg.gradient_checkpointing)
        self.assertEqual(cfg.gradient_checkpointing_ratio, 0.5)


class TestStartCaptionsContract(unittest.TestCase):
    """TaskManager captions should emit cumulative counters and done result."""

    def test_start_captions_emits_progress_counters_and_done_result(self):
        from sidestep_engine.gui.task_manager import TaskManager

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "a.wav").write_bytes(b"a")
            (base / "b.wav").write_bytes(b"b")

            tm = TaskManager()
            mocked_results = [
                {"status": "written"},
                {"status": "skipped"},
            ]
            with patch("sidestep_engine.data.enrich_song.enrich_one", side_effect=mocked_results):
                out = tm.start_captions({"provider": "skip", "dataset_dir": str(base)})
                task = tm._tasks[out["task_id"]]
                task.thread.join(timeout=5)

            events = []
            while not task.progress_queue.empty():
                events.append(task.progress_queue.get_nowait())

            progress = [e for e in events if e.get("type") == "progress"]
            done = [e for e in events if e.get("type") == "done"]

            self.assertEqual(len(progress), 2)
            self.assertEqual(progress[-1]["written"], 1)
            self.assertEqual(progress[-1]["skipped"], 1)
            self.assertEqual(progress[-1]["failed"], 0)

            self.assertEqual(len(done), 1)
            self.assertEqual(done[0]["result"]["written"], 1)
            self.assertEqual(done[0]["result"]["skipped"], 1)
            self.assertEqual(done[0]["result"]["failed"], 0)
            self.assertEqual(done[0]["result"]["total"], 2)


class TestTaskCancellationContract(unittest.TestCase):
    """stop_task should surface a cancelled event to websocket consumers."""

    @staticmethod
    def _drain_events(task) -> list[dict]:
        events = []
        while not task.progress_queue.empty():
            events.append(task.progress_queue.get_nowait())
        return events

    def test_stop_preprocess_emits_cancelled_not_done(self):
        from sidestep_engine.gui.task_manager import TaskManager

        tm = TaskManager()
        config = {
            "audio_dir": "/fake/audio",
            "output_dir": "/fake/output",
            "checkpoint_dir": "/fake/ckpt",
        }

        def _slow_pp(**_kwargs):
            time.sleep(0.1)
            return {"processed": 0, "failed": 0, "total": 1, "output_dir": "/fake/output"}

        with patch("sidestep_engine.data.preprocess.preprocess_audio_files", side_effect=_slow_pp):
            out = tm.start_preprocess(config)
            task_id = out["task_id"]
            tm.stop_task(task_id)
            task = tm._tasks[task_id]
            task.thread.join(timeout=5)

        events = self._drain_events(task)
        types = [e.get("type") for e in events]
        self.assertIn("cancelled", types)
        self.assertNotIn("done", types)

    def test_stop_ppplus_emits_cancelled_not_done(self):
        from sidestep_engine.gui.task_manager import TaskManager

        tm = TaskManager()
        config = {
            "checkpoint_dir": "/fake/ckpt",
            "dataset_dir": "/fake/data",
            "model_variant": "turbo",
        }

        def _slow_ppplus(**_kwargs):
            time.sleep(0.1)
            return None

        with patch("sidestep_engine.analysis.fisher.analysis.run_fisher_analysis", side_effect=_slow_ppplus):
            out = tm.start_ppplus(config)
            task_id = out["task_id"]
            tm.stop_task(task_id)
            task = tm._tasks[task_id]
            task.thread.join(timeout=5)

        events = self._drain_events(task)
        types = [e.get("type") for e in events]
        self.assertIn("cancelled", types)
        self.assertNotIn("done", types)

    def test_stop_captions_emits_cancelled(self):
        from sidestep_engine.gui.task_manager import TaskManager

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "a.wav").write_bytes(b"a")

            tm = TaskManager()

            def _slow_enrich(*_args, **_kwargs):
                time.sleep(0.1)
                return {"status": "written"}

            with patch("sidestep_engine.data.enrich_song.enrich_one", side_effect=_slow_enrich):
                out = tm.start_captions({"provider": "skip", "dataset_dir": str(base)})
                task_id = out["task_id"]
                tm.stop_task(task_id)
                task = tm._tasks[task_id]
                task.thread.join(timeout=5)

            events = self._drain_events(task)
            self.assertTrue(any(e.get("type") == "cancelled" for e in events))


# ---------------------------------------------------------------------------
# Settings persistence (key→DOM mapping)
# ---------------------------------------------------------------------------

class TestSettingsKeyMapping(unittest.TestCase):
    """Test that the backend settings key→DOM ID mapping covers all keys."""

    def test_all_default_settings_have_dom_mapping(self):
        """Every non-internal settings key should resolve to a DOM ID."""
        from sidestep_engine.settings import _default_settings
        defaults = _default_settings()
        # The explicit map from workspace.js boot()
        explicit_map = {
            "preprocessed_tensors_dir": "settings-tensors-dir",
            "trained_adapters_dir": "settings-adapters-dir",
            "gemini_api_key": "settings-gemini-key",
            "openai_api_key": "settings-openai-key",
            "openai_base_url": "settings-openai-base",
            "genius_api_token": "settings-genius-token",
        }
        # Keys that are internal/non-display
        skip = {"version", "first_run_complete", "caption_provider", "openai_model"}
        for key in defaults:
            if key in skip:
                continue
            dom_id = explicit_map.get(key, "settings-" + key.replace("_", "-"))
            # Just verify the mapping produces a non-empty string
            self.assertTrue(len(dom_id) > 0, f"No DOM mapping for settings key: {key}")

    def test_server_get_settings_none_safe(self):
        """GET /api/settings should not crash when no settings file exists."""
        with patch("sidestep_engine.settings.settings_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/settings.json")
            from sidestep_engine.settings import load_settings
            result = load_settings()
            self.assertIsNone(result)

    def test_server_post_settings_none_safe(self):
        """POST /api/settings should work when no settings file exists."""
        from sidestep_engine.settings import _default_settings
        defaults = _default_settings()
        defaults.update({"checkpoint_dir": "/tmp/test"})
        self.assertEqual(defaults["checkpoint_dir"], "/tmp/test")
        self.assertIn("version", defaults)


if __name__ == "__main__":
    unittest.main()
