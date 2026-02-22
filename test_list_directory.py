"""Tests for the list_directory tool."""

from __future__ import annotations

import os

from tools import _list_directory


class TestListDirectory:
    def test_lists_files_and_dirs(self, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        result = _list_directory(str(tmp_path))
        assert "file.txt" in result
        assert "subdir" in result
        assert "file" in result  # type indicator
        assert "dir" in result   # type indicator

    def test_hides_dotfiles_by_default(self, tmp_path):
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("hi")
        result = _list_directory(str(tmp_path), show_hidden=False)
        assert ".hidden" not in result
        assert "visible.txt" in result

    def test_shows_dotfiles_when_requested(self, tmp_path):
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("hi")
        result = _list_directory(str(tmp_path), show_hidden=True)
        assert ".hidden" in result
        assert "visible.txt" in result

    def test_nonexistent_path(self):
        result = _list_directory("/nonexistent/path/xyz")
        assert "Error" in result
        assert "not a directory" in result

    def test_file_path_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        result = _list_directory(str(f))
        assert "Error" in result
        assert "not a directory" in result

    def test_empty_directory(self, tmp_path):
        result = _list_directory(str(tmp_path))
        assert result == "(empty directory)"

    def test_size_formatting(self, tmp_path):
        (tmp_path / "small.txt").write_text("x" * 100)
        (tmp_path / "medium.txt").write_text("x" * 2048)
        result = _list_directory(str(tmp_path))
        assert "100B" in result
        assert "KB" in result

    def test_header_shows_count(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.txt").write_text("c")
        result = _list_directory(str(tmp_path))
        assert "3 items" in result

    def test_sorted_output(self, tmp_path):
        (tmp_path / "zebra.txt").write_text("z")
        (tmp_path / "alpha.txt").write_text("a")
        result = _list_directory(str(tmp_path))
        lines = result.strip().splitlines()
        # alpha should come before zebra
        alpha_idx = next(i for i, l in enumerate(lines) if "alpha" in l)
        zebra_idx = next(i for i, l in enumerate(lines) if "zebra" in l)
        assert alpha_idx < zebra_idx
