"""
Integration tests for the Side-Step GUI server.

Uses httpx AsyncClient with FastAPI's TestClient transport to test
REST endpoints and WebSocket connections without starting a real server.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

# Skip entire module if fastapi is not installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from sidestep_engine.gui.server import create_app

_TEST_TOKEN = "test_token_for_integration_tests_12345"


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the GUI FastAPI app with a known token."""
    app = create_app(token=_TEST_TOKEN, port=8770)
    with TestClient(
        app,
        base_url="http://127.0.0.1:8770",
        headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
    ) as c:
        yield c


@pytest.fixture(scope="module")
def unauth_client():
    """Create a TestClient WITHOUT auth headers."""
    app = create_app(token=_TEST_TOKEN, port=8770)
    with TestClient(app, base_url="http://127.0.0.1:8770") as c:
        yield c


# ======================================================================
# Static / index
# ======================================================================

class TestStaticServing:

    def test_index_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_css_dir_served(self, client):
        """CSS files should be accessible under /css/."""
        r = client.get("/css/tokens.css")
        # 200 if file exists, 404 otherwise — but the route should not 500
        assert r.status_code in (200, 404)

    def test_js_dir_served(self, client):
        """JS files should be accessible under /js/."""
        r = client.get("/js/api.js")
        assert r.status_code in (200, 404)


# ======================================================================
# Settings
# ======================================================================

class TestSettingsEndpoints:

    def test_get_settings(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()
        # Should return a dict with at least checkpoint_dir
        assert isinstance(data, dict)
        assert "checkpoint_dir" in data

    def test_post_settings(self, client):
        r = client.post("/api/settings", json={"data": {"checkpoint_dir": "/tmp/test_ckpts"}})
        assert r.status_code == 200
        assert r.json().get("ok") is True


# ======================================================================
# File browsing
# ======================================================================

class TestBrowseEndpoint:

    def test_browse_home(self, client):
        import pathlib
        home = str(pathlib.Path.home())
        r = client.post("/api/browse", json={"path": home, "dirs_only": False})
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data

    def test_browse_nonexistent(self, client):
        r = client.post("/api/browse", json={"path": "/nonexistent_path_xyz"})
        # Path scoping blocks paths outside allowed roots
        assert r.status_code == 403

    def test_browse_dirs_only(self, client):
        # Use the project root which is always allowed
        import pathlib
        project = str(pathlib.Path(__file__).resolve().parent.parent)
        r = client.post("/api/browse", json={"path": project, "dirs_only": True})
        assert r.status_code == 200
        entries = r.json().get("entries", [])
        names = [e["name"] for e in entries]
        assert "frontend" in names or "sidestep_engine" in names


# ======================================================================
# Models
# ======================================================================

class TestModelsEndpoint:

    def test_models_no_dir(self, client):
        r = client.get("/api/models?checkpoint_dir=")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data

    def test_models_nonexistent_dir(self, client):
        r = client.get("/api/models?checkpoint_dir=/nonexistent_xyz")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data


# ======================================================================
# Datasets
# ======================================================================

class TestDatasetsEndpoint:

    def test_datasets_no_dir(self, client):
        r = client.get("/api/datasets?tensors_dir=")
        assert r.status_code == 200

    def test_scan_audio(self, client):
        r = client.get("/api/dataset/scan?path=/tmp")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data

    def test_scan_audio_includes_sidecar_path(self, client, tmp_path):
        audio = tmp_path / "song.wav"
        sidecar = tmp_path / "song.txt"
        audio.write_bytes(b"RIFF....WAVE")
        sidecar.write_text("caption: test\nlyrics:\n", encoding="utf-8")

        r = client.get(f"/api/dataset/scan?path={tmp_path}")
        assert r.status_code == 200
        data = r.json()
        assert data["files"]
        row = data["files"][0]
        assert row["path"].endswith("song.wav")
        assert row["sidecar_path"].endswith("song.txt")

    def test_scan_audio_includes_folder_tree_metadata(self, client, tmp_path):
        root_file = tmp_path / "root.wav"
        nested_dir = tmp_path / "pack_a"
        nested_dir.mkdir()
        nested_file = nested_dir / "nested.wav"
        root_file.write_bytes(b"RIFF....WAVE")
        nested_file.write_bytes(b"RIFF....WAVE")

        r = client.get(f"/api/dataset/scan?path={tmp_path}")
        assert r.status_code == 200
        data = r.json()
        assert "folders" in data
        assert isinstance(data["folders"], list)
        assert any(f.get("path") == "." for f in data["folders"])
        assert any(f.get("path") == "pack_a" for f in data["folders"])

        files = data.get("files", [])
        assert any(row.get("relative_path") == "root.wav" and row.get("folder_path") == "." for row in files)
        assert any(row.get("relative_path") == "pack_a/nested.wav" and row.get("folder_path") == "pack_a" for row in files)

    def test_create_mix_dataset_symlinks_selected_files(self, client, tmp_path):
        src_root = tmp_path / "loot"
        src_sub = src_root / "drums"
        src_sub.mkdir(parents=True)
        audio = src_sub / "kick.wav"
        sidecar = src_sub / "kick.txt"
        audio.write_bytes(b"RIFF....WAVE")
        sidecar.write_text("caption: kick\n", encoding="utf-8")

        dest_root = tmp_path / "mixes"
        body = {
            "source_root": str(src_root),
            "destination_root": str(dest_root),
            "mix_name": "loot_mix_a",
            "files": [str(audio)],
        }
        r = client.post("/api/dataset/mix", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["created"] == 1

        out_dir = Path(data["path"])
        out_audio = out_dir / "drums" / "kick.wav"
        out_sidecar = out_dir / "drums" / "kick.txt"
        assert out_audio.exists()
        assert out_audio.is_symlink()
        assert out_sidecar.exists()
        assert out_sidecar.is_symlink()

    def test_datasets_all_audio_lists_subfolders(self, client, tmp_path):
        audio_root = tmp_path / "LOOT"
        pack_a = audio_root / "DUMB"
        pack_b = audio_root / "HTIF"
        pack_a.mkdir(parents=True)
        pack_b.mkdir(parents=True)
        (pack_a / "a.wav").write_bytes(b"RIFF....WAVE")
        (pack_b / "b.wav").write_bytes(b"RIFF....WAVE")

        with mock.patch(
            "sidestep_engine.settings.load_settings",
            return_value={
                "audio_dir": str(audio_root),
                "preprocessed_tensors_dir": "",
            },
        ), mock.patch(
            "sidestep_engine.data.audio_duration.get_audio_duration",
            return_value=1.0,
        ):
            r = client.get("/api/datasets/all")

        assert r.status_code == 200
        rows = [d for d in r.json().get("datasets", []) if d.get("type") == "audio"]
        names = {d.get("name") for d in rows}
        assert "DUMB" in names
        assert "HTIF" in names
        assert "LOOT" not in names


# ======================================================================
# Presets
# ======================================================================

class TestPresetsEndpoints:

    def test_list_presets(self, client):
        r = client.get("/api/presets")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_save_and_load_preset(self, client, tmp_path):
        """Save a preset, load it back, delete it."""
        # Patch the presets dir to use tmp_path
        with mock.patch("sidestep_engine.ui.presets._local_presets_dir", return_value=tmp_path), \
             mock.patch("sidestep_engine.ui.presets._global_presets_dir", return_value=tmp_path / "_global"), \
             mock.patch("sidestep_engine.ui.presets._builtin_presets_dir", return_value=tmp_path / "_builtin"):
            # Save
            r = client.post("/api/presets", json={
                "name": "test_preset",
                "data": {"rank": 32, "epochs": 10, "description": "test"},
            })
            assert r.status_code == 200
            assert r.json().get("ok") is True

            # Load
            r = client.get("/api/presets/test_preset")
            assert r.status_code == 200
            data = r.json()
            assert data["rank"] == 32

            # Delete
            r = client.delete("/api/presets/test_preset")
            assert r.status_code == 200

            # Verify deleted
            r = client.get("/api/presets/test_preset")
            assert r.status_code == 404

    def test_load_nonexistent_preset(self, client):
        r = client.get("/api/presets/nonexistent_preset_xyz")
        assert r.status_code == 404


# ======================================================================
# History
# ======================================================================

class TestHistoryEndpoints:

    def test_list_history(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_run_config_nonexistent(self, client):
        r = client.get("/api/history/nonexistent_run_xyz/config")
        assert r.status_code == 404

    def test_run_curve_nonexistent(self, client):
        r = client.get("/api/history/nonexistent_run_xyz/curve")
        assert r.status_code == 200
        assert r.json() == []

    def test_history_detected_folder_visible_and_deletable(self, client, tmp_path):
        adapters_root = tmp_path / "trained_adapters"
        run_dir = adapters_root / "lora" / "detected_run"
        run_dir.mkdir(parents=True)
        (run_dir / "training_config.json").write_text("{}", encoding="utf-8")

        with mock.patch("sidestep_engine.gui.file_ops._adapters_dir", return_value=adapters_root), \
             mock.patch("sidestep_engine.gui.file_ops._history_override_roots", return_value=[]):
            hist = client.get("/api/history")
            assert hist.status_code == 200
            rows = hist.json()
            row = next((r for r in rows if r.get("run_name") == "detected_run"), None)
            assert row is not None
            assert row.get("detected_only") is True
            assert row.get("adapter") == "--"

            deleted = client.delete(f"/api/history/folder?path={run_dir}")
            assert deleted.status_code == 200
            payload = deleted.json()
            assert payload.get("ok") is True
            assert not run_dir.exists()


# ======================================================================
# Checkpoints
# ======================================================================

class TestCheckpointsEndpoint:

    def test_list_checkpoints_nonexistent(self, client):
        r = client.get("/api/checkpoints/nonexistent_run_xyz")
        assert r.status_code == 200
        assert r.json() == {"checkpoints": []}


# ======================================================================
# Fisher map status
# ======================================================================

class TestFisherMapStatusEndpoint:

    def test_fisher_map_status_missing_file(self, client, tmp_path):
        ds = tmp_path / "dataset"
        ds.mkdir()

        r = client.get(f"/api/fisher-map/status?dataset_dir={ds}")
        assert r.status_code == 200
        data = r.json()
        assert data["exists"] is False
        assert data["modules"] == 0

    def test_fisher_map_status_reads_metadata_and_stale_flag(self, client, tmp_path):
        ds = tmp_path / "dataset"
        ds.mkdir()
        fm = ds / "fisher_map.json"
        fm.write_text(json.dumps({
            "model_variant": "turbo",
            "rank_budget": {"min": 16, "max": 128},
            "modules": [{"name": "a"}, {"name": "b"}],
        }), encoding="utf-8")

        ok = client.get(f"/api/fisher-map/status?dataset_dir={ds}&model_variant=turbo")
        assert ok.status_code == 200
        ok_data = ok.json()
        assert ok_data["exists"] is True
        assert ok_data["modules"] == 2
        assert ok_data["rank_min"] == 16
        assert ok_data["rank_max"] == 128
        assert ok_data["stale"] is False

        stale = client.get(f"/api/fisher-map/status?dataset_dir={ds}&model_variant=base")
        assert stale.status_code == 200
        stale_data = stale.json()
        assert stale_data["exists"] is True
        assert stale_data["stale"] is True


# ======================================================================
# GPU
# ======================================================================

class TestGPUEndpoint:

    def test_gpu_snapshot(self, client):
        r = client.get("/api/gpu")
        assert r.status_code == 200
        data = r.json()
        # Should have 'available' and 'name' keys regardless of GPU presence
        assert "name" in data


# ======================================================================
# Training control
# ======================================================================

class TestTrainingControl:

    def test_stop_when_not_running(self, client):
        r = client.post("/api/train/stop")
        assert r.status_code == 200
        data = r.json()
        assert "error" in data  # "No training running"


# ======================================================================
# API key validation
# ======================================================================

class TestValidateKey:

    def test_unknown_provider(self, client):
        r = client.post("/api/validate-key", json={
            "provider": "unknown_provider",
            "key": "test_key",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is False
        assert "Unknown provider" in data.get("error", "")


# ======================================================================
# Sidecar endpoints
# ======================================================================

class TestSidecarEndpoints:

    def test_read_nonexistent_sidecar(self, client):
        r = client.get("/api/sidecar?path=/nonexistent/file.txt")
        assert r.status_code == 200
        data = r.json()
        # sidecar_io returns empty dict for missing files (no exception)
        assert "ok" in data

    @mock.patch("sidestep_engine.gui.file_ops.is_path_allowed", return_value=True)
    def test_write_and_read_sidecar(self, _mock_allowed, client, tmp_path):
        sidecar_path = str(tmp_path / "test.txt")
        # Write
        r = client.post("/api/sidecar", json={
            "path": sidecar_path,
            "data": {"prompt": "A test prompt", "genre": "rock"},
        })
        assert r.status_code == 200

        # Read back
        r = client.get(f"/api/sidecar?path={sidecar_path}")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "test prompt" in data["data"].get("prompt", "").lower() or "test" in str(data["data"])

    @mock.patch("sidestep_engine.gui.file_ops.is_path_allowed", return_value=True)
    def test_write_sidecar_audio_path_coerced_to_txt(self, _mock_allowed, client, tmp_path):
        audio_path = tmp_path / "song.wav"
        sidecar_path = tmp_path / "song.txt"
        original = b"not-real-audio-but-binary"
        audio_path.write_bytes(original)

        r = client.post("/api/sidecar", json={
            "path": str(audio_path),
            "data": {"caption": "safe write", "genre": "rock"},
        })
        assert r.status_code == 200
        assert r.json().get("ok") is True

        # Audio file must remain untouched.
        assert audio_path.read_bytes() == original
        assert sidecar_path.is_file()
        assert "caption: safe write" in sidecar_path.read_text(encoding="utf-8").lower()

        # Reading via audio path should also coerce to sidecar path.
        r = client.get(f"/api/sidecar?path={audio_path}")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["data"].get("caption") == "safe write"


# ======================================================================
# Trigger tag bulk
# ======================================================================

class TestTriggerTagBulk:

    @mock.patch("sidestep_engine.gui.file_ops.is_path_allowed", return_value=True)
    def test_bulk_trigger_tag(self, _mock_allowed, client, tmp_path):
        """Create some sidecar files and apply a bulk trigger tag."""
        # Create two sidecar files
        for name in ("a.txt", "b.txt"):
            p = tmp_path / name
            p.write_text("prompt: existing content\n")
        paths = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
        r = client.post("/api/trigger-tag/bulk", json={
            "paths": paths,
            "tag": "my_trigger",
            "position": "prepend",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["updated"] == 2


# ======================================================================
# WebSocket: GPU
# ======================================================================

class TestGPUWebSocket:

    def test_gpu_ws_sends_data(self, client):
        with client.websocket_connect(f"/ws/gpu?token={_TEST_TOKEN}") as ws:
            data = ws.receive_json()
            assert "name" in data


# ======================================================================
# Security: Token auth
# ======================================================================

class TestTokenAuth:

    def test_api_without_token_returns_401(self, unauth_client):
        r = unauth_client.get("/api/settings")
        assert r.status_code == 401
        assert r.json().get("error") == "Unauthorized"

    def test_api_with_wrong_token_returns_401(self, unauth_client):
        r = unauth_client.get("/api/settings", headers={"Authorization": "Bearer wrong_token"})
        assert r.status_code == 401

    def test_static_index_no_token_ok(self, unauth_client):
        """Index page should be accessible without token."""
        r = unauth_client.get("/")
        assert r.status_code == 200

    def test_static_css_no_token_ok(self, unauth_client):
        """Static CSS should be accessible without token."""
        r = unauth_client.get("/css/tokens.css")
        assert r.status_code in (200, 404)  # 404 if file missing, but not 401

    def test_static_js_no_token_ok(self, unauth_client):
        """Static JS should be accessible without token."""
        r = unauth_client.get("/js/api.js")
        assert r.status_code in (200, 404)

    def test_post_without_token_returns_401(self, unauth_client):
        r = unauth_client.post("/api/browse", json={"path": "/tmp"})
        assert r.status_code == 401


# ======================================================================
# Security: API key masking
# ======================================================================

class TestKeyMasking:

    def test_settings_keys_are_masked(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()
        for key in ("gemini_api_key", "openai_api_key", "genius_api_token"):
            val = data.get(key)
            if val and isinstance(val, str) and len(val) > 4:
                # Should be masked: ••••<last4>
                assert val.startswith("••••"), f"{key} not masked: {val}"


# ======================================================================
# Security: Path scoping
# ======================================================================

class TestPathScoping:

    def test_browse_home_dir_allowed(self, client):
        import pathlib
        home = str(pathlib.Path.home())
        r = client.post("/api/browse", json={"path": home, "dirs_only": True})
        assert r.status_code == 200
        assert "entries" in r.json()

    def test_browse_root_allowed(self, client):
        r = client.post("/api/browse", json={"path": "/", "dirs_only": True})
        assert r.status_code == 200

    def test_browse_etc_blocked(self, client):
        """Browsing /etc should be blocked (outside allowed scope)."""
        r = client.post("/api/browse", json={"path": "/etc"})
        assert r.status_code == 403
