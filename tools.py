"""Tool definitions — file I/O, search, shell execution."""

from __future__ import annotations

import os
import subprocess


# File patterns that commonly contain secrets
_SENSITIVE_PATTERNS = (".env", ".env.local", ".env.production", "credentials.json",
                       ".pem", ".key", ".p12", ".pfx")


def _is_sensitive_file(path: str) -> bool:
    """Check if a file path matches common secret-containing patterns."""
    base = os.path.basename(path)
    return any(base == p or base.endswith(p) for p in _SENSITIVE_PATTERNS)


def _is_binary_file(path: str) -> bool:
    """Check if a file appears to be binary by looking for null bytes in first 8KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return False


def _read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: {path} does not exist"

    # Warn on binary files
    if _is_binary_file(path):
        size = os.path.getsize(path)
        return f"Warning: {path} appears to be a binary file ({size} bytes). Use bash to inspect binary files."

    # Warn on sensitive files
    warning = ""
    if _is_sensitive_file(path):
        warning = f"⚠ Warning: {path} may contain secrets. Be careful not to expose sensitive values.\n\n"

    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error: Unable to read {path}: {e}"
    start = max(0, offset)
    end = start + limit if limit > 0 else len(lines)
    numbered = [f"{i + start + 1:>6}\t{line}" for i, line in enumerate(lines[start:end])]
    content = "".join(numbered) if numbered else "(empty file)"
    return warning + content


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
    try:
        dirname = os.path.dirname(path)
        if dirname:  # Only create directories if there is a directory component
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w", encoding='utf-8') as f:
            f.write(content)
        return "OK"
    except PermissionError:
        return f"Error: Permission denied writing to '{path}'"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, cannot write file"
    except OSError as e:
        return f"Error: Cannot create directory or write file '{path}': {e}"
    except Exception as e:
        return f"Error writing file '{path}': {e}"


def _search(pattern: str, path: str = ".", glob: str = "") -> str:
    path = os.path.expanduser(path)
    cmd = ["rg", "--no-heading", "-n", pattern, path]
    if glob:
        cmd.extend(["--glob", glob])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout[:30000] or "(no matches)"
    except FileNotFoundError:
        # fallback to grep if rg not installed — exclude common ignored dirs
        cmd = ["grep", "-rn",
               "--exclude-dir=.git", "--exclude-dir=node_modules",
               "--exclude-dir=__pycache__", "--exclude-dir=.venv",
               "--exclude-dir=venv", "--exclude-dir=.tox",
               pattern, path]
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


def _list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """List directory contents with file type and size."""
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"

    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return f"Error: permission denied for {path}"

    entries = []
    for name in items:
        if not show_hidden and name.startswith("."):
            continue
        full_path = os.path.join(path, name)
        try:
            stat = os.stat(full_path)
            size = stat.st_size
            is_dir = os.path.isdir(full_path)
            type_indicator = "dir" if is_dir else "file"
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            entries.append(f"  {type_indicator:<5} {size_str:>8}  {name}")
        except OSError:
            entries.append(f"  ?     ?         {name}")

    if not entries:
        return "(empty directory)"

    header = f"{path}/ ({len(entries)} items)"
    return header + "\n" + "\n".join(entries)


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
        "name": "list_directory",
        "description": "List contents of a directory with file types and sizes. More structured than bash('ls').",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list", "default": "."},
                "show_hidden": {"type": "boolean", "description": "Include hidden files (dotfiles)", "default": False},
            },
        },
        "execute": _list_directory,
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
EXPLORER_TOOLS = [t for t in TOOLS if t["name"] in ("read_file", "search", "list_files", "list_directory", "bash")]
# Tools for test runner sub-agents (read + bash)
TEST_TOOLS = [t for t in TOOLS if t["name"] in ("read_file", "bash")]
# Full tool access for coder sub-agents
CODER_TOOLS = [t for t in TOOLS if t["name"] in ("read_file", "edit_file", "write_file", "search", "bash")]
