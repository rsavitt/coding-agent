"""Tests for token estimation utilities."""

from __future__ import annotations

from tokens import estimate_system_tokens, estimate_tokens, estimate_tool_tokens


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # minimum 1

    def test_short_string(self):
        result = estimate_tokens("hello")
        assert result >= 1

    def test_longer_string(self):
        # 100 chars ÷ 3.5 ≈ 28-29 tokens
        text = "x" * 100
        result = estimate_tokens(text)
        assert 25 <= result <= 35

    def test_proportional(self):
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world " * 100)
        assert long > short


class TestEstimateToolTokens:
    def test_no_tools(self):
        assert estimate_tool_tokens([]) == 0

    def test_single_tool(self):
        tools = [{
            "name": "read_file",
            "description": "Read a file's contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        }]
        result = estimate_tool_tokens(tools)
        assert result > 0
        # Should be reasonable — a simple tool schema is roughly 30-80 tokens
        assert 20 < result < 200

    def test_multiple_tools(self):
        tool = {
            "name": "test",
            "description": "A test tool.",
            "parameters": {"type": "object", "properties": {}},
        }
        one = estimate_tool_tokens([tool])
        three = estimate_tool_tokens([tool, tool, tool])
        assert three > one

    def test_complex_tool_costs_more(self):
        simple = [{
            "name": "simple",
            "description": "Simple.",
            "parameters": {"type": "object", "properties": {}},
        }]
        complex_tool = [{
            "name": "complex",
            "description": "A very complex tool with lots of parameters and detailed descriptions for each one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First argument with a long description"},
                    "arg2": {"type": "integer", "description": "Second argument"},
                    "arg3": {"type": "boolean", "description": "Third argument"},
                    "arg4": {"type": "array", "items": {"type": "string"}, "description": "Fourth argument"},
                },
                "required": ["arg1", "arg2"],
            },
        }]
        assert estimate_tool_tokens(complex_tool) > estimate_tool_tokens(simple)

    def test_includes_per_tool_overhead(self):
        tool = {
            "name": "t",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        }
        # With per-tool overhead of 10, adding more tools should add at least 10 per tool
        one = estimate_tool_tokens([tool])
        two = estimate_tool_tokens([tool, tool])
        assert two - one >= 10

    def test_realistic_tool_set(self):
        """Estimate for a realistic set of tools like the coding agent uses."""
        tools = [
            {
                "name": "read_file",
                "description": "Read a file's contents with line numbers. Use offset/limit for large files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative file path"},
                        "offset": {"type": "integer", "description": "Start line (0-indexed)", "default": 0},
                        "limit": {"type": "integer", "description": "Max lines to read (0 = all)", "default": 0},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "edit_file",
                "description": "Replace an exact string in a file. Fails if the match isn't unique.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File to edit"},
                        "old_string": {"type": "string", "description": "Exact text to find"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
            {
                "name": "bash",
                "description": "Run a shell command. Use for git, tests, builds, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
                    },
                    "required": ["command"],
                },
            },
        ]
        result = estimate_tool_tokens(tools)
        # 3 tools with moderate schemas — should be in the hundreds range
        assert 100 < result < 1000


class TestEstimateSystemTokens:
    def test_empty(self):
        assert estimate_system_tokens("") == 0

    def test_short_system(self):
        result = estimate_system_tokens("You are a helpful assistant.")
        assert result > 0
        assert result < 20

    def test_long_system(self):
        system = "You are a coding agent. " * 50
        result = estimate_system_tokens(system)
        assert result > 100


class TestContextIntegration:
    def test_overhead_affects_compaction_threshold(self):
        """Verify that tool overhead is additive with input tokens."""
        from tools import TOOLS
        overhead = estimate_tool_tokens(TOOLS)
        # With 8 tools, overhead should be meaningful (several hundred tokens)
        assert overhead > 100
        # But not absurdly large
        assert overhead < 5000
