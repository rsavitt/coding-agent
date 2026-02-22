"""Tests for coder sub-agent type."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from providers import Response, ToolCall
from tools import CODER_TOOLS, EXPLORER_TOOLS, TEST_TOOLS


class TestCoderToolSet:
    def test_coder_has_write_tools(self):
        """Coder should have edit_file and write_file, which others lack."""
        coder_names = {t["name"] for t in CODER_TOOLS}
        assert "edit_file" in coder_names
        assert "write_file" in coder_names
        assert "read_file" in coder_names
        assert "search" in coder_names
        assert "bash" in coder_names

    def test_explorer_lacks_write_tools(self):
        """Explorer should NOT have edit_file or write_file."""
        explorer_names = {t["name"] for t in EXPLORER_TOOLS}
        assert "edit_file" not in explorer_names
        assert "write_file" not in explorer_names

    def test_test_runner_lacks_write_tools(self):
        """Test runner should NOT have edit_file or write_file."""
        runner_names = {t["name"] for t in TEST_TOOLS}
        assert "edit_file" not in runner_names
        assert "write_file" not in runner_names


class TestCoderPrompt:
    def test_coder_prompt_exists(self):
        from prompts import SUB_AGENT_PROMPTS
        assert "coder" in SUB_AGENT_PROMPTS

    def test_coder_prompt_mentions_editing(self):
        from prompts import CODER_SYSTEM
        assert "edit_file" in CODER_SYSTEM
        assert "write_file" in CODER_SYSTEM

    def test_coder_prompt_mentions_verification(self):
        from prompts import CODER_SYSTEM
        assert "verify" in CODER_SYSTEM.lower()


class TestCoderDelegation:
    def test_delegate_tool_accepts_coder_type(self):
        """The delegate tool should accept 'coder' as an agent_type."""
        from sub_agents import get_delegation_tools

        provider = MagicMock()
        provider.call.return_value = Response(
            text="Done, I fixed the bug.", tool_calls=[],
            input_tokens=100, output_tokens=50,
        )

        tools = get_delegation_tools(provider, model="test-model")
        delegate_tool = tools[0]  # delegate (not delegate_parallel)

        # Check that coder is in the enum
        enum_values = delegate_tool["parameters"]["properties"]["agent_type"]["enum"]
        assert "coder" in enum_values

        # Actually call it
        result = delegate_tool["execute"](task="Fix the typo in main.py", agent_type="coder")
        assert "coder" in result
        assert "Done, I fixed the bug." in result

    def test_delegate_parallel_accepts_coder_type(self):
        """The delegate_parallel tool should accept 'coder' as an agent_type."""
        from sub_agents import get_delegation_tools

        provider = MagicMock()
        provider.call.return_value = Response(
            text="Changes made.", tool_calls=[],
            input_tokens=100, output_tokens=50,
        )

        tools = get_delegation_tools(provider, model="test-model")
        parallel_tool = tools[1]  # delegate_parallel

        # Check enum
        items_schema = parallel_tool["parameters"]["properties"]["tasks"]["items"]
        enum_values = items_schema["properties"]["agent_type"]["enum"]
        assert "coder" in enum_values

    def test_coder_subagent_gets_write_tools(self):
        """When delegating to a coder, it should receive write-capable tools."""
        from sub_agents import _AGENT_TOOLS

        coder_tools = _AGENT_TOOLS["coder"]
        tool_names = {t["name"] for t in coder_tools}
        assert "edit_file" in tool_names
        assert "write_file" in tool_names

    def test_coder_subagent_can_use_edit_file(self):
        """A coder sub-agent should be able to call edit_file via tool calls."""
        from sub_agents import SubAgentRunner
        from prompts import CODER_SYSTEM
        from tools import CODER_TOOLS

        # Simulate: LLM calls edit_file, then responds with summary
        resp1 = Response(
            text="I'll fix that.",
            tool_calls=[ToolCall(
                id="tc_1", name="edit_file",
                arguments={"path": "/tmp/test.py", "old_string": "bug", "new_string": "fix"},
            )],
            input_tokens=50, output_tokens=30,
        )
        resp2 = Response(
            text="Fixed the bug in /tmp/test.py by replacing 'bug' with 'fix'.",
            tool_calls=[],
            input_tokens=80, output_tokens=40,
        )

        provider = MagicMock()
        provider.call.side_effect = [resp1, resp2]

        runner = SubAgentRunner(
            provider=provider, tools=CODER_TOOLS,
            system=CODER_SYSTEM, model="test",
        )
        result = runner.run("Fix the bug in /tmp/test.py")

        assert result.status == "completed"
        assert "Fixed the bug" in result.summary
        assert provider.call.call_count == 2


class TestMainAgentPrompt:
    def test_main_prompt_mentions_coder(self):
        from prompts import MAIN_AGENT_SYSTEM
        assert "coder" in MAIN_AGENT_SYSTEM.lower()
