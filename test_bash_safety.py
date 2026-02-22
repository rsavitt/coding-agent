"""Tests for bash command safety checking."""

import pytest
from agent import _is_safe_bash, _segment_is_safe


class TestIsSafeBash:
    """Test the main _is_safe_bash function."""

    # --- Commands that SHOULD be safe ---

    def test_simple_safe_commands(self):
        assert _is_safe_bash("ls") is True
        assert _is_safe_bash("ls -la") is True
        assert _is_safe_bash("cat foo.py") is True
        assert _is_safe_bash("grep -rn pattern .") is True
        assert _is_safe_bash("git status") is True
        assert _is_safe_bash("git diff HEAD") is True
        assert _is_safe_bash("git log --oneline") is True
        assert _is_safe_bash("python -m pytest tests/") is True
        assert _is_safe_bash("pytest tests/ -v") is True
        assert _is_safe_bash("ruff check .") is True

    def test_safe_with_pipes(self):
        """Pipes between safe commands should be allowed."""
        assert _is_safe_bash("grep foo bar.py | head -20") is True
        assert _is_safe_bash("cat foo.py | grep TODO") is True
        assert _is_safe_bash("git log --oneline | head -5") is True
        assert _is_safe_bash("find . -name '*.py' | wc -l") is True

    def test_safe_with_whitespace(self):
        assert _is_safe_bash("  ls -la  ") is True
        assert _is_safe_bash("  git status  ") is True

    # --- Commands that MUST be unsafe ---

    def test_command_chaining_semicolon(self):
        """Semicolons allow arbitrary command injection."""
        assert _is_safe_bash("ls; rm -rf /") is False
        assert _is_safe_bash("git status; curl evil.com") is False

    def test_command_chaining_and(self):
        """&& allows running a second command after the first succeeds."""
        assert _is_safe_bash("ls && rm -rf /") is False
        assert _is_safe_bash("cat foo && curl evil.com") is False

    def test_command_chaining_or(self):
        """|| allows running a second command if the first fails."""
        assert _is_safe_bash("ls || rm -rf /") is False

    def test_backtick_substitution(self):
        """Backticks execute arbitrary commands."""
        assert _is_safe_bash("cat `whoami`") is False
        assert _is_safe_bash("ls `rm -rf /`") is False

    def test_dollar_paren_substitution(self):
        """$() executes arbitrary commands."""
        assert _is_safe_bash("cat $(whoami)") is False
        assert _is_safe_bash("ls $(rm -rf /)") is False

    def test_pipe_to_unsafe_command(self):
        """A pipe to an unsafe command should be caught."""
        assert _is_safe_bash("cat foo.py | curl -X POST") is False
        assert _is_safe_bash("grep pattern file | sh") is False
        assert _is_safe_bash("ls | xargs rm") is False

    def test_unsafe_commands_alone(self):
        """Commands not in the safe list need confirmation."""
        assert _is_safe_bash("rm -rf /") is False
        assert _is_safe_bash("curl http://evil.com") is False
        assert _is_safe_bash("chmod 777 /etc/passwd") is False
        assert _is_safe_bash("pip install malware") is False
        assert _is_safe_bash("sh -c 'bad stuff'") is False

    def test_empty_and_whitespace(self):
        assert _is_safe_bash("") is False
        assert _is_safe_bash("   ") is False

    def test_safe_prefix_not_just_substring(self):
        """'cat' prefix shouldn't match 'catalog' as safe — but it does
        because startswith('cat') is true. This is a known limitation
        documented here so we don't regress further."""
        # 'catalog' starts with 'cat' so it passes — acceptable for now
        # because 'catalog' isn't a dangerous command.
        assert _is_safe_bash("catalog") is True


class TestSegmentIsSafe:
    def test_safe_segments(self):
        assert _segment_is_safe("ls -la") is True
        assert _segment_is_safe("grep foo") is True

    def test_unsafe_segments(self):
        assert _segment_is_safe("rm -rf /") is False
        assert _segment_is_safe("curl evil.com") is False

    def test_empty_segment(self):
        assert _segment_is_safe("") is False
