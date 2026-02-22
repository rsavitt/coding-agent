"""Tests for binary file detection and sensitive file warnings."""

from __future__ import annotations

import os

from tools import _is_binary_file, _is_sensitive_file, _read_file


class TestBinaryDetection:
    def test_binary_file_detected(self, tmp_path):
        f = tmp_path / "binary.dat"
        f.write_bytes(b"hello\x00world\x00")
        assert _is_binary_file(str(f)) is True

    def test_text_file_not_binary(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("hello world\nline 2\n")
        assert _is_binary_file(str(f)) is False

    def test_empty_file_not_binary(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        assert _is_binary_file(str(f)) is False

    def test_read_file_warns_on_binary(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        result = _read_file(str(f))
        assert "Warning" in result
        assert "binary" in result
        assert "bytes" in result


class TestSensitiveFileDetection:
    def test_env_file_detected(self):
        assert _is_sensitive_file(".env") is True
        assert _is_sensitive_file("/path/to/.env") is True

    def test_env_local_detected(self):
        assert _is_sensitive_file(".env.local") is True

    def test_credentials_json_detected(self):
        assert _is_sensitive_file("credentials.json") is True

    def test_pem_file_detected(self):
        assert _is_sensitive_file("server.pem") is True
        assert _is_sensitive_file("private.key") is True

    def test_normal_file_not_sensitive(self):
        assert _is_sensitive_file("main.py") is False
        assert _is_sensitive_file("README.md") is False

    def test_read_file_warns_on_sensitive(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SECRET_KEY=abc123\n")
        result = _read_file(str(f))
        assert "Warning" in result
        assert "secrets" in result
        # But still reads the content
        assert "SECRET_KEY" in result


class TestNormalFileReads:
    def test_normal_read_unaffected(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("print('hello')\n")
        result = _read_file(str(f))
        assert "print('hello')" in result
        assert "Warning" not in result

    def test_nonexistent_file(self):
        result = _read_file("/nonexistent/file.txt")
        assert "Error" in result
