"""Verify that Python code blocks in the documentation are syntactically valid."""

import ast
import re
import textwrap
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Match fenced Python code blocks in Markdown
CODE_BLOCK_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_python_blocks(md_path: Path) -> list[str]:
    """Extract all Python code blocks from a Markdown file."""
    return CODE_BLOCK_PATTERN.findall(md_path.read_text())


def _is_partial_snippet(code: str) -> bool:
    """Detect code blocks that are intentionally partial snippets.

    These are code fragments shown in the docs to highlight specific
    changes within a larger block of code. They are not meant to be
    valid standalone Python.
    """
    lines = [line for line in code.splitlines() if line.strip()]
    if not lines:
        return True
    # If every non-empty line is indented, it's a snippet from inside
    # a function or class body.
    if all(line[0] in (" ", "\t") for line in lines):
        return True
    # A single-line function/class definition with no body
    if len(lines) == 1 and (lines[0].rstrip().endswith(":")):
        return True
    return False


def _collect_test_cases() -> list[tuple[str, int, str]]:
    """Collect (filename, block_index, code) tuples for parametrization."""
    cases = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        blocks = _extract_python_blocks(md_file)
        for i, block in enumerate(blocks):
            cases.append((md_file.name, i + 1, block))
    return cases


_test_cases = _collect_test_cases()


@pytest.mark.parametrize(
    ("filename", "block_num", "code"),
    _test_cases,
    ids=[f"{name}:block{num}" for name, num, _ in _test_cases],
)
def test_python_block_syntax(filename: str, block_num: int, code: str) -> None:
    """Each Python code block in the docs must be syntactically valid."""
    if _is_partial_snippet(code):
        pytest.skip("partial code snippet")

    try:
        ast.parse(code)
    except SyntaxError:
        # Try again after removing common leading whitespace
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as exc:
            pytest.fail(
                f"{filename} block {block_num}: {exc.msg} (line {exc.lineno})\n\n{code}"
            )
