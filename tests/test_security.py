from pathlib import Path

import pytest

from taskforge.security import (
    ToolInputError,
    evaluate_arithmetic,
    grep_workspace,
    read_workspace_text,
    resolve_workspace_path,
)


def test_read_and_grep_are_bounded_to_workspace(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("first\nneedle here\nlast\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret needle", encoding="utf-8")

    result = read_workspace_text(tmp_path, "src/app.py", start_line=2, max_lines=1)
    assert result["content"] == "2: needle here"
    assert grep_workspace(tmp_path, "needle", include="*.py")["matches"] == [
        {"path": "src/app.py", "line": 2, "text": "needle here"}
    ]
    with pytest.raises(ToolInputError):
        resolve_workspace_path(tmp_path, "../outside.txt")
    with pytest.raises(ToolInputError):
        resolve_workspace_path(tmp_path, str(outside))


def test_grep_skips_credentials_binary_and_caps_results(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("needle=secret", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"needle\x00hidden")
    (tmp_path / "many.txt").write_text("\n".join(["needle"] * 10), encoding="utf-8")

    result = grep_workspace(tmp_path, "needle", limit=3)
    assert len(result["matches"]) == 3
    assert result["truncated"] is True
    assert {match["path"] for match in result["matches"]} == {"many.txt"}


def test_grep_regex_is_explicit_and_invalid_regex_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("task-123\ntask-abc", encoding="utf-8")
    literal = grep_workspace(tmp_path, r"task-\d+", regex=False)
    regex = grep_workspace(tmp_path, r"task-\d+", regex=True)
    assert literal["matches"] == []
    assert regex["matches"][0]["text"] == "task-123"
    with pytest.raises(ToolInputError):
        grep_workspace(tmp_path, "(", regex=True)


def test_calculator_allows_math_but_not_python_execution() -> None:
    assert evaluate_arithmetic("(2 + 3) * 4") == 20
    assert evaluate_arithmetic("2 ** 8") == 256
    with pytest.raises(ToolInputError):
        evaluate_arithmetic("__import__('os').system('whoami')")
    with pytest.raises(ToolInputError):
        evaluate_arithmetic("2 ** 999")

