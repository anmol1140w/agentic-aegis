"""Tests for tools: calculator and file operations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tools.calculator import calculator, safe_eval
from tools import files as file_tools


# ---------------------------------------------------------------------------
# Calculator tests
# ---------------------------------------------------------------------------

class TestCalculator:
    """Test the safe AST-based calculator."""

    def test_addition(self) -> None:
        assert safe_eval("2 + 3") == 5

    def test_multiplication(self) -> None:
        assert safe_eval("125 * 48") == 6000

    def test_division(self) -> None:
        assert safe_eval("10 / 4") == 2.5

    def test_floor_division(self) -> None:
        assert safe_eval("10 // 3") == 3

    def test_modulo(self) -> None:
        assert safe_eval("10 % 3") == 1

    def test_power(self) -> None:
        assert safe_eval("2 ** 10") == 1024

    def test_negative(self) -> None:
        assert safe_eval("-5 + 3") == -2

    def test_sqrt(self) -> None:
        assert safe_eval("sqrt(144)") == 12.0

    def test_complex_expression(self) -> None:
        result = safe_eval("sqrt(144) + 3 ** 2")
        assert result == 21.0

    def test_log10(self) -> None:
        result = safe_eval("log10(1000)")
        assert abs(result - 3.0) < 1e-9

    def test_pi(self) -> None:
        import math
        result = safe_eval("pi")
        assert abs(result - math.pi) < 1e-9

    def test_nested_functions(self) -> None:
        result = safe_eval("abs(-42)")
        assert result == 42

    def test_rejects_imports(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("__import__('os').system('ls')")

    def test_rejects_unknown_names(self) -> None:
        with pytest.raises(ValueError, match="Unknown name"):
            safe_eval("unknown_var + 1")

    def test_rejects_attributes(self) -> None:
        with pytest.raises(ValueError, match="Unsupported expression element"):
            safe_eval("os.system('ls')")

    def test_rejects_string_literals(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("'hello'")

    def test_tool_returns_string(self) -> None:
        # LangChain tool returns string
        result = calculator.invoke({"expression": "2 + 2"})
        assert result == "4"

    def test_tool_handles_error(self) -> None:
        result = calculator.invoke({"expression": "foo()"})
        assert "Error" in result


# ---------------------------------------------------------------------------
# File tool tests
# ---------------------------------------------------------------------------

class TestFileTools:
    """Test file operations with workspace boundary enforcement."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path: Path) -> None:
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        file_tools.set_workspace_root(self.workspace)

    def test_write_and_read(self) -> None:
        result = file_tools.write_file.invoke(
            {"file_path": "test.txt", "content": "hello world"}
        )
        assert "Written" in result

        content = file_tools.read_file.invoke({"file_path": "test.txt"})
        assert content == "hello world"

    def test_list_files(self) -> None:
        (self.workspace / "a.txt").write_text("a")
        (self.workspace / "b.txt").write_text("b")
        result = file_tools.list_files.invoke({"directory": "."})
        assert "a.txt" in result
        assert "b.txt" in result

    def test_create_directory(self) -> None:
        result = file_tools.create_directory.invoke({"directory": "sub/dir"})
        assert "Created" in result
        assert (self.workspace / "sub" / "dir").is_dir()

    def test_read_nonexistent(self) -> None:
        result = file_tools.read_file.invoke({"file_path": "nope.txt"})
        assert "Permission denied" in result or "Error" in result

    def test_path_traversal_blocked(self) -> None:
        result = file_tools.read_file.invoke({"file_path": "../../etc/passwd"})
        assert "Permission denied" in result

    def test_write_blocked_extension(self) -> None:
        result = file_tools.write_file.invoke(
            {"file_path": "evil.sh", "content": "#!/bin/bash\nrm -rf /"}
        )
        assert "Permission denied" in result

    def test_nested_write_creates_parents(self) -> None:
        result = file_tools.write_file.invoke(
            {"file_path": "deep/nested/file.txt", "content": "deep"}
        )
        assert "Written" in result
        assert (self.workspace / "deep" / "nested" / "file.txt").read_text() == "deep"

    def test_read_directory_returns_error(self) -> None:
        (self.workspace / "adir").mkdir()
        result = file_tools.read_file.invoke({"file_path": "adir"})
        assert "directory" in result.lower()
