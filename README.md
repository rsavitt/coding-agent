# coding-agent

A minimal coding agent in ~760 lines of Python. Reads, writes, and edits code, runs shell commands, delegates to sub-agents, and works with both Anthropic and OpenAI models.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...  # or OPENAI_API_KEY=sk-...
```

## Usage

```bash
# One-shot
python cli.py "Fix the bug in auth.py"

# Interactive REPL
python cli.py

# Provider/model override
python cli.py --provider openai --model gpt-4o "Explain this codebase"
```

## Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read files with line numbers, optional offset/limit |
| `edit_file` | Exact string replacement (fails if match isn't unique) |
| `write_file` | Create or overwrite files |
| `search` | Ripgrep-powered regex search across files |
| `bash` | Shell execution with timeout and permission gating |
| `delegate` | Spawn a sub-agent (explorer or test_runner) |
| `delegate_parallel` | Run multiple sub-agents concurrently |

## Sub-agents

Sub-agents run with scoped tool access and budget limits:

- **explorer** — Read-only. Searches the codebase and reports findings.
- **test_runner** — Read + bash. Runs tests and reports results.

Both have turn limits and token budgets. When the budget is exhausted, the sub-agent is forced to summarize what it found.

## Architecture

```
cli.py          → Entry point, REPL, argument parsing
agent.py        → Core loop: LLM call → tool execution → repeat
tools.py        → Tool definitions and implementations
sub_agents.py   → Sub-agent runner, parallel execution, tool scoping
providers.py    → Anthropic + OpenAI abstraction with format translation
prompts.py      → System prompts for main agent and sub-agent types
```

## Safety

- Bash commands are checked against a safe list (`ls`, `git status`, `pytest`, etc.)
- Unsafe commands (`rm`, `curl`, etc.) prompt for user confirmation
- Tool output is truncated at 30k characters
- Sub-agents get read-only tool access by default
- Context budget warnings at 150k input tokens
