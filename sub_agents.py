"""Sub-agent runner — delegation with tool scoping and budget enforcement."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from prompts import SUB_AGENT_PROMPTS, EXPLORER_SYSTEM
from tools import EXPLORER_TOOLS, TEST_TOOLS, TOOLS


@dataclass
class SubAgentResult:
    summary: str
    status: str  # "completed" | "budget_exceeded" | "error"
    turns_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class SubAgentRunner:
    def __init__(self, provider, tools: list[dict], system: str,
                 model: str = "", max_turns: int = 15, max_tokens: int = 100_000):
        self.provider = provider
        self.tools = tools
        self.tool_map = {t["name"]: t["execute"] for t in tools}
        self.system = system
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def run(self, task: str) -> SubAgentResult:
        messages = [{"role": "user", "content": task}]
        total_in, total_out, turns = 0, 0, 0

        for turns in range(1, self.max_turns + 1):
            if total_in + total_out > self.max_tokens:
                return self._force_summary(messages, total_in, total_out, turns, "budget_exceeded")

            call_kwargs = {"model": self.model} if self.model else {}
            resp = self.provider.call(
                messages=messages, tools=self.tools, system=self.system, **call_kwargs,
            )
            total_in += resp.input_tokens
            total_out += resp.output_tokens

            if not resp.tool_calls:
                return SubAgentResult(
                    summary=resp.text, status="completed",
                    turns_used=turns, input_tokens=total_in, output_tokens=total_out,
                )

            # Build assistant message with tool use
            assistant_content = _build_assistant_content(resp)
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tools and add results
            tool_results = []
            for tc in resp.tool_calls:
                result = self._execute_tool(tc.name, tc.arguments)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})

        return self._force_summary(messages, total_in, total_out, turns, "budget_exceeded")

    def _execute_tool(self, name: str, arguments: dict) -> str:
        fn = self.tool_map.get(name)
        if not fn:
            return f"Error: unknown tool '{name}'"
        try:
            result = fn(**arguments)
            if len(result) > 30000:
                result = result[:15000] + "\n\n... (truncated) ...\n\n" + result[-15000:]
            return result
        except Exception as e:
            return f"Error: {e}"

    def _force_summary(self, messages, total_in, total_out, turns, status):
        """Ask the sub-agent for a final summary without tools."""
        messages.append({"role": "user", "content": (
            "You've reached your budget limit. Summarize your findings so far in a single message. "
            "Do NOT call any tools."
        )})
        call_kwargs = {"model": self.model} if self.model else {}
        resp = self.provider.call(messages=messages, tools=[], system=self.system, **call_kwargs)
        return SubAgentResult(
            summary=resp.text, status=status,
            turns_used=turns, input_tokens=total_in + resp.input_tokens,
            output_tokens=total_out + resp.output_tokens,
        )


def _build_assistant_content(resp) -> list[dict]:
    """Build Anthropic-style content blocks for the assistant turn."""
    content = []
    if resp.text:
        content.append({"type": "text", "text": resp.text})
    for tc in resp.tool_calls:
        content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
    return content


# --- Tools exposed to the main agent ---

def _make_delegate_tool(provider, model: str) -> dict:
    def _delegate(task: str, agent_type: str = "explorer") -> str:
        system = SUB_AGENT_PROMPTS.get(agent_type, EXPLORER_SYSTEM)
        tools = EXPLORER_TOOLS if agent_type == "explorer" else TEST_TOOLS
        runner = SubAgentRunner(provider=provider, tools=tools, system=system, model=model)
        result = runner.run(task)
        header = f"[{agent_type}] status={result.status} turns={result.turns_used}"
        return f"{header}\n\n{result.summary}"

    return {
        "name": "delegate",
        "description": (
            "Delegate a task to a sub-agent. Types: 'explorer' (read-only codebase search), "
            "'test_runner' (run tests and report)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What the sub-agent should do"},
                "agent_type": {
                    "type": "string", "enum": ["explorer", "test_runner"],
                    "description": "Sub-agent type", "default": "explorer",
                },
            },
            "required": ["task"],
        },
        "execute": _delegate,
    }


def _make_delegate_parallel_tool(provider, model: str) -> dict:
    def _delegate_parallel(tasks: list[dict]) -> str:
        """tasks: list of {task: str, agent_type: str}"""
        results = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
            futures = {}
            for i, t in enumerate(tasks):
                agent_type = t.get("agent_type", "explorer")
                system = SUB_AGENT_PROMPTS.get(agent_type, EXPLORER_SYSTEM)
                tools = EXPLORER_TOOLS if agent_type == "explorer" else TEST_TOOLS
                runner = SubAgentRunner(provider=provider, tools=tools, system=system, model=model)
                futures[pool.submit(runner.run, t["task"])] = i

            ordered = [None] * len(tasks)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    ordered[idx] = future.result()
                except Exception as e:
                    ordered[idx] = SubAgentResult(summary=str(e), status="error")

        parts = []
        for i, r in enumerate(ordered):
            parts.append(f"--- Task {i + 1} [{r.status}] ---\n{r.summary}")
        return "\n\n".join(parts)

    return {
        "name": "delegate_parallel",
        "description": "Run multiple sub-agent tasks in parallel. Each item needs 'task' and optional 'agent_type'.",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "agent_type": {"type": "string", "enum": ["explorer", "test_runner"]},
                        },
                        "required": ["task"],
                    },
                    "description": "List of sub-agent tasks to run in parallel",
                },
            },
            "required": ["tasks"],
        },
        "execute": _delegate_parallel,
    }


def get_delegation_tools(provider, model: str = "") -> list[dict]:
    """Return the delegate and delegate_parallel tools wired to the given provider."""
    return [
        _make_delegate_tool(provider, model),
        _make_delegate_parallel_tool(provider, model),
    ]
