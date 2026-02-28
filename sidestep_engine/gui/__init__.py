"""
Side-Step GUI — pywebview + FastAPI backend.

Launch with ``sidestep --gui`` or::

    from sidestep_engine.gui import launch
    launch(port=8770)
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import webbrowser

# Silence noisy GTK/Qt/Chromium warnings from pywebview
os.environ.setdefault("PYWEBVIEW_LOG", "warning")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu-compositing")

# Qt WebEngine on Linux (Arch/Manjaro/NixOS) needs --no-sandbox for correct rendering/interaction
if os.name != "nt" and "darwin" not in sys.platform.lower():
    existing = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--no-sandbox" not in existing:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{existing} --no-sandbox".strip()

# Hint window manager to use dark titlebar / window decorations
os.environ.setdefault("GTK_THEME", "Adwaita:dark")
os.environ.setdefault("GTK_CSD", "1")  # client-side decorations
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "qt5ct")

logger = logging.getLogger(__name__)


def _resolve_qt_permission_policies(qwebpage_cls: object) -> tuple[object, object]:
    """Return granted/denied policy values for Qt5/Qt6 compatibility."""
    permission_policy = getattr(qwebpage_cls, "PermissionPolicy", None)
    if permission_policy is not None:
        granted = getattr(permission_policy, "PermissionGrantedByUser", None)
        denied = getattr(permission_policy, "PermissionDeniedByUser", None)
        if granted is not None and denied is not None:
            return granted, denied

    # Older bindings expose these constants directly on QWebPage/QWebEnginePage.
    granted = getattr(qwebpage_cls, "PermissionGrantedByUser", 1)
    denied = getattr(qwebpage_cls, "PermissionDeniedByUser", 2)
    return granted, denied


def _patch_pywebview_qt_permissions(qt_module: object) -> bool:
    """Patch pywebview Qt permission callback for strict enum-typed Qt6 APIs."""
    browser_view = getattr(qt_module, "BrowserView", None)
    web_page_cls = getattr(browser_view, "WebPage", None)
    qwebpage_cls = getattr(qt_module, "QWebPage", None)
    handler = getattr(web_page_cls, "onFeaturePermissionRequested", None)

    if web_page_cls is None or qwebpage_cls is None or handler is None:
        return False
    if getattr(handler, "__sidestep_qt_patch__", False):
        return True

    feature_enum = getattr(qwebpage_cls, "Feature", None)
    media_features = tuple(
        getattr(feature_enum, name)
        for name in ("MediaAudioCapture", "MediaVideoCapture", "MediaAudioVideoCapture")
        if feature_enum is not None and hasattr(feature_enum, name)
    )
    granted_policy, denied_policy = _resolve_qt_permission_policies(qwebpage_cls)

    def _patched_on_feature_permission_requested(self, url, feature):
        allow = feature in media_features
        policy = granted_policy if allow else denied_policy
        try:
            self.setFeaturePermission(url, feature, policy)
        except TypeError:
            # Compatibility fallback for older bindings that still expect ints.
            legacy_policy = 1 if allow else 2
            self.setFeaturePermission(url, feature, legacy_policy)

    _patched_on_feature_permission_requested.__sidestep_qt_patch__ = True
    web_page_cls.onFeaturePermissionRequested = _patched_on_feature_permission_requested
    return True


def _apply_qt_permission_patch(webview_module: object) -> bool:
    """Apply runtime pywebview Qt permission patch when Qt backend is available."""
    qt_module = getattr(getattr(webview_module, "platforms", None), "qt", None)
    if qt_module is None:
        try:
            from webview.platforms import qt as qt_module
        except Exception:
            return False

    try:
        return _patch_pywebview_qt_permissions(qt_module)
    except Exception:
        logger.debug("[Side-Step GUI] Qt permission patch failed", exc_info=True)
        return False


def _port_free(host: str, port: int) -> bool:
    """Return True if *port* is available to bind on *host*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def launch(host: str = "127.0.0.1", port: int = 8770) -> None:
    """Start the FastAPI server and open a pywebview window.

    Falls back to the system browser when pywebview is unavailable
    or missing system dependencies (GTK/QT on Linux).
    """
    from sidestep_engine.gui.security import generate_token
    from sidestep_engine.gui.server import create_app

    # Find a free port (try up to 10)
    original_port = port
    for _ in range(10):
        if _port_free(host, port):
            break
        port += 1
    else:
        print(f"[FAIL] Ports {original_port}–{port} all in use. Kill the old server or pick a different port.")
        return

    if port != original_port:
        print(f"[INFO] Port {original_port} in use, using {port} instead.")

    token = generate_token()
    app = create_app(token=token, port=port)

    # Start uvicorn in a daemon thread so the main thread is free
    # for pywebview (which must run on the main thread on macOS).
    def _run_server() -> None:
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level="warning")

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    url = f"http://{host}:{port}"
    auth_url = f"{url}/?token={token}"
    logger.info("[Side-Step GUI] Server starting on %s", url)

    # Wait briefly for uvicorn to bind
    import time
    time.sleep(0.5)

    try:
        import webview  # pywebview
        _apply_qt_permission_patch(webview)

        _maximized = True

        class _WinAPI:
            """Exposed to JS as window.pywebview.api.*"""
            def minimize(self):
                for w in webview.windows:
                    w.minimize()

            def toggle_maximize(self):
                nonlocal _maximized
                for w in webview.windows:
                    if _maximized:
                        w.restore()
                    else:
                        w.maximize()
                _maximized = not _maximized

            def close(self):
                for w in webview.windows:
                    w.destroy()

            def get_position(self):
                for w in webview.windows:
                    return {"x": w.x, "y": w.y}
                return {"x": 0, "y": 0}

            def move_window(self, x, y):
                for w in webview.windows:
                    w.move(int(x), int(y))

            def on_boot_error(self, msg: str):
                logger.error("[Side-Step GUI] Boot failed: %s", msg)
                print(f"[ERROR] GUI boot failed: {msg}")

            def get_token(self) -> str:
                """Return auth token for API calls. Fallback when URL/injection fail (e.g. GTK/WebKit)."""
                return token

        api = _WinAPI()
        win = webview.create_window(
            "Side-Step", auth_url,
            width=1400, height=900,
            background_color="#16161E",
            frameless=False,
            js_api=api,
        )

        def _on_started():
            """Maximize the window once webview is ready."""
            try:
                win.maximize()
            except Exception:
                pass

        webview.start(func=_on_started)
        # Window closed — force exit (daemon server thread + any stray threads)
        import os as _os
        _os._exit(0)
    except ImportError:
        logger.warning("[Side-Step GUI] pywebview not found — opening in browser")
        webbrowser.open(auth_url)
        _keep_alive(server_thread)
    except Exception as exc:
        # pywebview installed but GTK/QT system deps missing
        logger.warning("[Side-Step GUI] Native window failed (%s) — opening in browser", exc)
        print(f"[WARN] Native window unavailable: {exc}")
        print("       Try: uv sync  (to reinstall GUI dependencies)")
        print("       Falling back to browser...")
        webbrowser.open(auth_url)
        _keep_alive(server_thread)


def _keep_alive(server_thread: threading.Thread) -> None:
    """Block until the browser signals shutdown or Ctrl+C."""
    print("[INFO] GUI running in browser. Close the tab or press Ctrl+C to stop.")
    try:
        # server_thread is daemon — it dies when main thread exits.
        # Block in small increments so KeyboardInterrupt is responsive.
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        pass
    print("\n[OK] Server stopped.")
