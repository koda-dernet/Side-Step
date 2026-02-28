"""Regression tests for pywebview Qt permission policy compatibility patch."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import sidestep_engine.gui as gui_module


class TestQtPermissionPatch(unittest.TestCase):
    def test_resolve_prefers_permission_policy_enum(self):
        class PermissionPolicy:
            PermissionGrantedByUser = "enum-granted"
            PermissionDeniedByUser = "enum-denied"

        class QWebPage:
            pass

        QWebPage.PermissionPolicy = PermissionPolicy

        granted, denied = gui_module._resolve_qt_permission_policies(QWebPage)
        self.assertEqual(granted, "enum-granted")
        self.assertEqual(denied, "enum-denied")

    def test_resolve_falls_back_to_direct_constants(self):
        class QWebPage:
            PermissionGrantedByUser = "direct-granted"
            PermissionDeniedByUser = "direct-denied"

        granted, denied = gui_module._resolve_qt_permission_policies(QWebPage)
        self.assertEqual(granted, "direct-granted")
        self.assertEqual(denied, "direct-denied")

    def test_patch_returns_false_when_qt_symbols_missing(self):
        qt_module = SimpleNamespace(BrowserView=SimpleNamespace(WebPage=None), QWebPage=object())
        self.assertFalse(gui_module._patch_pywebview_qt_permissions(qt_module))

    def test_patch_uses_enum_policy_for_media_and_denies_other(self):
        media_audio = object()
        media_video = object()
        media_audio_video = object()
        other_feature = object()

        class Feature:
            MediaAudioCapture = media_audio
            MediaVideoCapture = media_video
            MediaAudioVideoCapture = media_audio_video

        class PermissionPolicy:
            PermissionGrantedByUser = "allow"
            PermissionDeniedByUser = "deny"

        class QWebPage:
            pass

        QWebPage.Feature = Feature
        QWebPage.PermissionPolicy = PermissionPolicy

        class WebPage:
            def __init__(self):
                self.calls = []

            def onFeaturePermissionRequested(self, _url, _feature):
                raise AssertionError("handler should be patched")

            def setFeaturePermission(self, url, feature, policy):
                self.calls.append((url, feature, policy))

        class BrowserView:
            pass

        BrowserView.WebPage = WebPage

        qt_module = SimpleNamespace(BrowserView=BrowserView, QWebPage=QWebPage)

        self.assertTrue(gui_module._patch_pywebview_qt_permissions(qt_module))
        page = WebPage()
        page.onFeaturePermissionRequested("origin", media_audio)
        page.onFeaturePermissionRequested("origin", other_feature)

        self.assertEqual(page.calls[0], ("origin", media_audio, "allow"))
        self.assertEqual(page.calls[1], ("origin", other_feature, "deny"))

    def test_patch_falls_back_to_legacy_int_policy_on_type_error(self):
        media_audio = object()
        other_feature = object()

        class Feature:
            MediaAudioCapture = media_audio
            MediaVideoCapture = object()
            MediaAudioVideoCapture = object()

        class QWebPage:
            pass

        QWebPage.Feature = Feature
        QWebPage.PermissionGrantedByUser = "enum-granted"
        QWebPage.PermissionDeniedByUser = "enum-denied"

        class WebPage:
            def __init__(self):
                self.calls = []

            def onFeaturePermissionRequested(self, _url, _feature):
                raise AssertionError("handler should be patched")

            def setFeaturePermission(self, _url, _feature, policy):
                if not isinstance(policy, int):
                    raise TypeError("legacy API expects int")
                self.calls.append(policy)

        class BrowserView:
            pass

        BrowserView.WebPage = WebPage

        qt_module = SimpleNamespace(BrowserView=BrowserView, QWebPage=QWebPage)

        self.assertTrue(gui_module._patch_pywebview_qt_permissions(qt_module))
        page = WebPage()
        page.onFeaturePermissionRequested("origin", media_audio)
        page.onFeaturePermissionRequested("origin", other_feature)

        self.assertEqual(page.calls, [1, 2])

    def test_apply_patch_uses_webview_platform_qt_module(self):
        media_audio = object()

        class Feature:
            MediaAudioCapture = media_audio
            MediaVideoCapture = object()
            MediaAudioVideoCapture = object()

        class PermissionPolicy:
            PermissionGrantedByUser = "allow"
            PermissionDeniedByUser = "deny"

        class QWebPage:
            pass

        QWebPage.Feature = Feature
        QWebPage.PermissionPolicy = PermissionPolicy

        class WebPage:
            def onFeaturePermissionRequested(self, _url, _feature):
                pass

            def setFeaturePermission(self, _url, _feature, _policy):
                pass

        class BrowserView:
            pass

        BrowserView.WebPage = WebPage

        qt_module = SimpleNamespace(BrowserView=BrowserView, QWebPage=QWebPage)
        webview_module = SimpleNamespace(platforms=SimpleNamespace(qt=qt_module))

        self.assertTrue(gui_module._apply_qt_permission_patch(webview_module))


if __name__ == "__main__":
    unittest.main()
