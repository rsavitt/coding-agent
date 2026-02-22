"""Tool definitions — file I/O, search, shell execution."""

from __future__ import annotations

import os
import subprocess


def _read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: {path} does not exist"
    with open(path) as f:
        lines = f.readlines()
    start = max(0, offset)
    end = start + limit if limit > 0 else len(lines)
    numbered = [f"{i + start + 1:>6}\t{line}" for i, line in enumerate(lines[start:end])]
    return "".join(numbered) if numbered else "(empty file)"


def _edit_file(path: str, old_string: str, new_string: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: {path} does not exist"
    with open(path) as f:
        content = f.read()
    count = content.count(old_string)
    if count == 0:
        return "Error: old_string not found in file"
    if count > 1:
        return f"Error: old_string matches {count} locations — provide more context to make it unique"
    with open(path, "w") as f:
        f.write(content.replace(old_string, new_string, 1))
    return "OK"


def _write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return "OK"


def _search(pattern: str, path: str = ".", glob: str = "") -> str:
    path = os.path.expanduser(path)
    cmd = ["rg", "--no-heading", "-n", pattern, path]
    if glob:
        cmd.extend(["--glob", glob])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout[:30000] or "(no matches)"
    except FileNotFoundError:
        # fallback to grep if rg not installed
        cmd = ["grep", "-rn", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout[:30000] or "(no matches)"
    except subprocess.TimeoutExpired:
        return "Error: search timed out after 30s"


def _list_files(pattern: str = "**/*", path: str = ".") -> str:
    """List files matching a glob pattern, respecting .gitignore."""
    import pathlib
    path = os.path.expanduser(path)
    base = pathlib.Path(path)
    if not base.is_dir():
        return f"Error: {path} is not a directory"

    # Try git ls-files first (respects .gitignore)
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", path],
            capture_output=True, text=True, timeout=10, cwd=path,
        )
        if result.returncode == 0:
            all_files = result.stdout.strip().splitlines()
            if pattern == "**/*":
                matched = sorted(all_files)
            else:
                import pathlib as _pl
                matched = sorted(f for f in all_files if _pl.PurePath(f).match(pattern))

            if not matched:
                return f"(no files matching '{pattern}')"
            output = "\n".join(matched)
            if len(output) > 30000:
                output = output[:30000] + "\n... (truncated)"
            return output
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: pathlib glob (no .gitignore filtering)
    try:
        matched = sorted(str(p.relative_to(base)) for p in base.glob(pattern) if p.is_file())
        if not matched:
            return f"(no files matching '{pattern}')"
        output = "\n".join(matched)
        if len(output) > 30000:
            output = output[:30000] + "\n... (truncated)"
        return output
    except Exception as e:
        return f"Error: {e}"


def _bash(command: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if len(output) > 30000:
            output = output[:15000] + "\n\n... (truncated) ...\n\n" + output[-15000:]
        if result.returncode != 0:
            output += f"\n(exit code {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"


# --- Tool registry ---

TOOLS: list[dict] = [
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
        "execute": _read_file,
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
        "execute": _edit_file,
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
        "execute": _write_file,
    },
    {
        "name": "search",
        "description": "Search file contents using ripgrep (regex supported). Returns matching lines with file paths and line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in", "default": "."},
                "glob": {"type": "string", "description": "File glob filter, e.g. '*.py'", "default": ""},
            },
            "required": ["pattern"],
        },
        "execute": _search,
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts'). Respects .gitignore when inside a git repo.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files", "default": "**/*"},
                "path": {"type": "string", "description": "Directory to search in", "default": "."},
            },
        },
        "execute": _list_files,
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
        "execute": _bash,
    },
]

# Read-only subset for explorer sub-agents
EXPLORER_TOOLS = [t for t in TOOLS if t["name"] in ("read_file", "search", "list_files", "bash")]
# Tools for test runner sub-agents (read + bash)
TEST_TOOLS = [t for t in TOOLS if t["name"] in ("read_file", "bash")]
# Full tool access for coder sub-agents
CODER_TOOLS = [t for t in TOOLS if t["name"] in ("read_file", "edit_file", "write_file", "search", "bash")]
