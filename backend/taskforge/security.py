"""Deterministic safety helpers for built-in local tools.

The functions in this module never execute model-provided shell strings. They are
small enough to audit and deliberately reject ambiguous filesystem targets.
"""

from __future__ import annotations

import ast
import fnmatch
import operator
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolInputError(ValueError):
    """Raised when a tool argument is unsafe or outside its contract."""


_SKIP_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".taskforge",
}
_SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_workspace_path(root: str | Path, relative_path: str) -> Path:
    """Resolve an existing path inside *root* and reject aliases and ADS paths."""

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ToolInputError("path must be a non-empty relative string")
    candidate_text = relative_path.strip().replace("\\", "/")
    if Path(candidate_text).is_absolute() or candidate_text.startswith("/"):
        raise ToolInputError("absolute paths are not allowed")
    if any(part in {"", ".", ".."} for part in candidate_text.split("/")):
        raise ToolInputError("path contains an ambiguous or parent segment")
    if os.name == "nt" and ":" in candidate_text:
        raise ToolInputError("Windows alternate data stream paths are not allowed")

    root_path = Path(root).resolve(strict=True)
    lexical = root_path.joinpath(*candidate_text.split("/"))
    resolved = lexical.resolve(strict=True)
    if not _is_relative_to(resolved, root_path):
        raise ToolInputError("path resolves outside the workspace")
    if lexical.is_symlink():
        raise ToolInputError("symbolic-link targets are not readable by built-in tools")
    try:
        attrs = lexical.lstat().st_file_attributes  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        attrs = 0
    if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise ToolInputError("reparse-point targets are not readable by built-in tools")
    if any(part in _SKIP_PARTS for part in lexical.relative_to(root_path).parts):
        raise ToolInputError("path belongs to an excluded workspace directory")
    if lexical.name.lower() in _SENSITIVE_NAMES:
        raise ToolInputError("sensitive credential files are not readable")
    return lexical


def read_workspace_text(
    root: str | Path,
    relative_path: str,
    *,
    start_line: int = 1,
    max_lines: int = 200,
    max_bytes: int = 256_000,
) -> dict[str, Any]:
    if start_line < 1:
        raise ToolInputError("start_line must be at least 1")
    max_lines = max(1, min(int(max_lines), 200))
    path = resolve_workspace_path(root, relative_path)
    if not path.is_file():
        raise ToolInputError("path is not a regular file")
    size = path.stat().st_size
    if size > max_bytes:
        raise ToolInputError(f"file exceeds {max_bytes} bytes")
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ToolInputError("binary files are not readable")
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = start_line - 1
    selected = lines[start : start + max_lines]
    return {
        "path": path.relative_to(Path(root).resolve()).as_posix(),
        "start_line": start_line,
        "end_line": start + len(selected),
        "total_lines": len(lines),
        "content": "\n".join(
            f"{start + index + 1}: {line}" for index, line in enumerate(selected)
        ),
        "truncated": start + len(selected) < len(lines),
    }


@dataclass(frozen=True)
class GrepMatch:
    path: str
    line: int
    text: str


def grep_workspace(
    root: str | Path,
    pattern: str,
    *,
    include: str = "*",
    regex: bool = False,
    case_sensitive: bool = False,
    limit: int = 20,
    timeout_seconds: float = 2.0,
    max_file_bytes: int = 512_000,
) -> dict[str, Any]:
    """Search text files without a shell or arbitrary command-line flags."""

    if not isinstance(pattern, str) or not pattern:
        raise ToolInputError("pattern must be non-empty")
    if len(pattern) > 256:
        raise ToolInputError("pattern is too long")
    if not isinstance(include, str) or not include or len(include) > 128:
        raise ToolInputError("include must be a short glob")
    if ".." in Path(include).parts or Path(include).is_absolute():
        raise ToolInputError("include glob must stay within the workspace")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern if regex else re.escape(pattern), flags)
    except re.error as exc:
        raise ToolInputError(f"invalid regular expression: {exc}") from exc

    limit = max(1, min(int(limit), 50))
    deadline = time.monotonic() + max(0.05, min(float(timeout_seconds), 5.0))
    root_path = Path(root).resolve(strict=True)
    matches: list[GrepMatch] = []
    scanned_files = 0
    timed_out = False

    for path in root_path.rglob("*"):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        try:
            relative = path.relative_to(root_path)
        except ValueError:
            continue
        if any(part in _SKIP_PARTS for part in relative.parts):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.lower() in _SENSITIVE_NAMES:
            continue
        if not fnmatch.fnmatch(relative.as_posix(), include) and not fnmatch.fnmatch(path.name, include):
            continue
        try:
            safe = resolve_workspace_path(root_path, relative.as_posix())
            if safe.stat().st_size > max_file_bytes:
                continue
            raw = safe.read_bytes()
        except (OSError, ToolInputError):
            continue
        if b"\x00" in raw[:8192]:
            continue
        scanned_files += 1
        text = raw.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                matches.append(GrepMatch(relative.as_posix(), line_number, line[:500]))
                if len(matches) >= limit:
                    return {
                        "matches": [match.__dict__ for match in matches],
                        "scanned_files": scanned_files,
                        "truncated": True,
                        "timed_out": False,
                    }

    return {
        "matches": [match.__dict__ for match in matches],
        "scanned_files": scanned_files,
        "truncated": False,
        "timed_out": timed_out,
    }


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_arithmetic(expression: str) -> int | float:
    """Evaluate a bounded arithmetic expression using an AST allowlist."""

    if not isinstance(expression, str) or not expression.strip():
        raise ToolInputError("expression must be non-empty")
    if len(expression) > 200:
        raise ToolInputError("expression is too long")
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError("invalid arithmetic expression") from exc

    def visit(current: ast.AST, depth: int = 0) -> int | float:
        if depth > 20:
            raise ToolInputError("expression is too deeply nested")
        if isinstance(current, ast.Expression):
            return visit(current.body, depth + 1)
        if isinstance(current, ast.Constant) and type(current.value) in {int, float}:
            if abs(current.value) > 1e100:
                raise ToolInputError("numeric literal is too large")
            return current.value
        if isinstance(current, ast.BinOp) and type(current.op) in _BIN_OPS:
            left = visit(current.left, depth + 1)
            right = visit(current.right, depth + 1)
            if isinstance(current.op, ast.Pow) and abs(right) > 12:
                raise ToolInputError("exponent is too large")
            result = _BIN_OPS[type(current.op)](left, right)
            if abs(result) > 1e100:
                raise ToolInputError("result is too large")
            return result
        if isinstance(current, ast.UnaryOp) and type(current.op) in _UNARY_OPS:
            return _UNARY_OPS[type(current.op)](visit(current.operand, depth + 1))
        raise ToolInputError("only arithmetic operators and numeric literals are allowed")

    return visit(node)

