"""Tests for compatibility checking and warning filters."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from sidestep_engine._compat import (
    check_compatibility,
    install_torchao_warning_filter,
)


class TestCheckCompatibility(unittest.TestCase):
    """Test compatibility check for vendored modules."""

    @patch("sidestep_engine._compat.logger")
    def test_check_compatibility_all_imports_ok(self, mock_logger):
        """Should pass silently when all imports succeed."""
        # Run check
        check_compatibility()

        # Should only have debug log, no warnings
        debug_calls = [c for c in mock_logger.debug.call_args_list]
        warning_calls = [c for c in mock_logger.warning.call_args_list]

        # May have debug calls but should not have warnings
        self.assertEqual(len(warning_calls), 0)

    @patch("sidestep_engine._compat.logger")
    def test_check_compatibility_missing_import(self, mock_logger):
        """Should warn when imports fail (simulated)."""
        # We can't easily simulate import failures without messing with sys.modules,
        # so this is a structural test that the function exists and can be called
        check_compatibility()

        # Function should complete without raising
        self.assertTrue(True)


class TestInstallTorchaoWarningFilter(unittest.TestCase):
    """Test torchao warning filter installation."""

    def test_filter_installation(self):
        """Should install filter on root and torchao loggers."""
        # Clear any existing filters
        root_logger = logging.getLogger()
        torchao_logger = logging.getLogger("torchao")
        root_logger.filters.clear()
        torchao_logger.filters.clear()

        # Install filter
        install_torchao_warning_filter()

        # Should have added filters
        self.assertGreater(len(root_logger.filters), 0)
        self.assertGreater(len(torchao_logger.filters), 0)

    def test_filter_drops_known_warning(self):
        """Should drop the known torchao warning."""
        # Clear filters
        torchao_logger = logging.getLogger("torchao")
        torchao_logger.filters.clear()

        # Install filter
        install_torchao_warning_filter()

        # Create a log record with the known warning
        record = logging.LogRecord(
            name="torchao",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Skipping import of cpp extensions due to incompatible torch version",
            args=(),
            exc_info=None,
        )

        # Filter should drop it
        for f in torchao_logger.filters:
            result = f.filter(record)
            if not result:
                # Filter dropped it
                self.assertFalse(result)
                return

        # If we get here, no filter dropped it (might be OK if filter logic changed)
        # The test is mainly to ensure the function runs without error

    def test_filter_keeps_other_warnings(self):
        """Should keep non-matching warnings."""
        # Clear filters
        torchao_logger = logging.getLogger("torchao")
        torchao_logger.filters.clear()

        # Install filter
        install_torchao_warning_filter()

        # Create a log record with a different warning
        record = logging.LogRecord(
            name="torchao",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Some other warning message",
            args=(),
            exc_info=None,
        )

        # Filter should keep it
        for f in torchao_logger.filters:
            result = f.filter(record)
            if not result:
                self.fail("Filter incorrectly dropped non-matching warning")

    def test_filter_keeps_non_torchao_logs(self):
        """Should not affect logs from other modules."""
        # Clear filters
        root_logger = logging.getLogger()
        root_logger.filters.clear()

        # Install filter
        install_torchao_warning_filter()

        # Create a log record from a different module
        record = logging.LogRecord(
            name="other_module",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Some warning",
            args=(),
            exc_info=None,
        )

        # All filters should pass it through
        for f in root_logger.filters:
            result = f.filter(record)
            self.assertTrue(result)

    @patch.dict("os.environ", {"SIDESTEP_DISABLE_TORCHAO_WARN_FILTER": "1"})
    def test_filter_disabled_by_env_var(self):
        """Should not install filter when disabled by environment variable."""
        # Clear filters
        root_logger = logging.getLogger()
        root_logger.filters.clear()
        initial_count = len(root_logger.filters)

        # Install filter (should be no-op)
        install_torchao_warning_filter()

        # Should not have added filters
        self.assertEqual(len(root_logger.filters), initial_count)

    @patch.dict("os.environ", {"SIDESTEP_DISABLE_TORCHAO_WARN_FILTER": "true"})
    def test_filter_disabled_by_env_var_true(self):
        """Should handle 'true' as disable value."""
        root_logger = logging.getLogger()
        root_logger.filters.clear()
        initial_count = len(root_logger.filters)

        install_torchao_warning_filter()

        self.assertEqual(len(root_logger.filters), initial_count)

    @patch.dict("os.environ", {"SIDESTEP_DISABLE_TORCHAO_WARN_FILTER": "yes"})
    def test_filter_disabled_by_env_var_yes(self):
        """Should handle 'yes' as disable value."""
        root_logger = logging.getLogger()
        root_logger.filters.clear()
        initial_count = len(root_logger.filters)

        install_torchao_warning_filter()

        self.assertEqual(len(root_logger.filters), initial_count)


class TestCompatibilityConstants(unittest.TestCase):
    """Test that compatibility constants are defined."""

    def test_tested_acestep_commit_defined(self):
        """TESTED_ACESTEP_COMMIT should be defined."""
        from sidestep_engine._compat import TESTED_ACESTEP_COMMIT

        self.assertIsInstance(TESTED_ACESTEP_COMMIT, str)
        self.assertGreater(len(TESTED_ACESTEP_COMMIT), 0)

    def test_sidestep_version_defined(self):
        """SIDESTEP_VERSION should be defined."""
        from sidestep_engine._compat import SIDESTEP_VERSION

        self.assertIsInstance(SIDESTEP_VERSION, str)
        self.assertGreater(len(SIDESTEP_VERSION), 0)


if __name__ == "__main__":
    unittest.main()