"""Tests for the native process helper."""

import os
from pathlib import Path

from tools.native_process import helper_path, available, run


def test_helper_path_returns_correct_path() -> None:
    """Test that helper_path returns the expected path."""
    name = "aegis-exec.exe" if os.name == "nt" else "aegis-exec"
    expected = Path(__file__).resolve().parent.parent / "native" / "bin" / name
    assert helper_path() == expected


def test_available_reflects_helper_state() -> None:
    """Test that available() correctly reports whether helper exists and is executable."""
    # This will be True in our test environment since we built the helper
    path = helper_path()
    expected_available = path.is_file() and os.access(path, os.X_OK)
    assert available() == expected_available


def test_run_returns_none_when_helper_unavailable() -> None:
    """Test that run() returns None when helper is not available or not executable."""
    # Temporarily make helper unavailable by setting AEGIS_NATIVE_EXEC to nonexistent path
    import tools.native_process as np
    original_env = os.environ.get("AEGIS_NATIVE_EXEC")

    try:
        os.environ["AEGIS_NATIVE_EXEC"] = "/this/path/definitely/does/not/exist"
        # Clear any cached helper path by forcing a recheck
        result = run(["echo", "test"], cwd=Path("/tmp"), env={}, timeout=1)
        assert result is None
    finally:
        if original_env is None:
            os.environ.pop("AEGIS_NATIVE_EXEC", None)
        else:
            os.environ["AEGIS_NATIVE_EXEC"] = original_env


def test_run_successful_execution() -> None:
    """Test that run() works when helper is available."""
    # Skip if native helper not available in test environment
    if not available():
        return  # Skip test if helper not built

    result = run(["echo", "hello world"], cwd=Path("/tmp"), env=os.environ, timeout=5)
    assert result is not None
    assert isinstance(result, dict)
    assert "returncode" in result
    assert "stdout" in result
    assert "stderr" in result
    assert "timed_out" in result
    assert "native" in result
    assert result["native"] is True
    assert result["returncode"] == 0
    assert "hello world" in result["stdout"]


def test_run_timeout() -> None:
    """Test that run() handles timeout correctly."""
    # Skip if native helper not available in test environment
    if not available():
        return  # Skip test if helper not built

    result = run(["sleep", "2"], cwd=Path("/tmp"), env=os.environ, timeout=1)
    assert result is not None
    assert result["returncode"] == 124  # timeout exit code
    assert result["timed_out"] is True
    assert result["native"] is True
