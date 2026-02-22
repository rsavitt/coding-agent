"""Tests for list_files tool."""

from __future__ import annotations

import os
import tempfile

import pytest
from tools import _list_files, TOOLS, EXPLORER_TOOLS


class TestListFilesFunction:
    def test_lists_python_files(self):
        """Should find .py files in the current directory."""
        result = _list_files("*.py", ".")
        assert "agent.py" in result
        assert "tools.py" in result

    def test_lists_with_glob_pattern(self):
        """Should support ** glob patterns."""
        result = _list_files("sandbox/*.py", ".")
        assert "hello.py" in result or "math_utils.py" in result

    def test_no_matches(self):
        result = _list_files("*.nonexistent_extension_xyz", ".")
        assert "no files matching" in result

    def test_invalid_directory(self):
        result = _list_files("*", "/nonexistent/path/abc123")
        assert "Error" in result or "not a directory" in result

    def test_fallback_to_pathlib(self):
        """In a non-git directory, should still work via pathlib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            for name in ["foo.txt", "bar.py", "baz.py"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("content")

            result = _list_files("*.py", tmpdir)
            assert "bar.py" in result
            assert "baz.py" in result
            # .txt should not match *.py
            assert "foo.txt" not in result

    def test_default_pattern_lists_all(self):
        """Default pattern **/* should list all files."""
        result = _list_files(path=".")
        assert "agent.py" in result
        assert "cli.py" in result


class TestListFilesToolDefinition:
    def test_tool_registered(self):
        """list_files should be in the TOOLS list."""
        names = [t["name"] for t in TOOLS]
        assert "list_files" in names

    def test_tool_in_explorer_tools(self):
        """Explorers should have access to list_files."""
        names = [t["name"] for t in EXPLORER_TOOLS]
        assert "list_files" in names

    def test_tool_has_execute(self):
        tool = next(t for t in TOOLS if t["name"] == "list_files")
        assert callable(tool["execute"])
