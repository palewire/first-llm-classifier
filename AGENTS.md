# AGENTS.md

Instructions for AI coding assistants working on this project.

## Project overview

This is an open-source educational tutorial that teaches journalists how to use large-language models to organize and analyze datasets. The primary deliverables are **Sphinx documentation** (in `docs/`) and an **interactive Jupyter notebook** (`notebook.ipynb`). This is not a library or application.

## Tech stack

- **Python 3.13** (pinned in `.python-version`)
- **uv** for dependency management (`uv sync` to install)
- **Sphinx** with **MyST parser** for documentation (Markdown, not reStructuredText)
- **ruff** for linting and formatting
- **pytest** for testing
- **pre-commit** for git hooks

## Setup

```bash
uv sync          # Install all dependencies
make serve       # Start live-preview docs server at localhost:8000
```

## Documentation conventions

Documentation lives in `docs/` as Markdown files parsed by MyST.

### Code blocks

Use fenced Python code blocks:

````
```python
print("hello")
```
````

### Highlighting new lines

When adding new lines of code to an existing block, use the `{emphasize-lines}` directive to highlight them so readers can see what changed:

````
```{emphasize-lines="3,4"}
```python
existing_line = True
new_line_one = True
new_line_two = True
```
````

### MyST extensions

The project enables `attrs_block` and `colon_fence` MyST extensions. Use `{note}`, `{warning}`, and other standard directives as needed.

## Python conventions

- **Keep code as simple as possible.** This is a tutorial for journalists, not a showcase for advanced Python. Favor clarity over cleverness.
- Use modern shorthand type hints (`list[str]`, `dict[str, int]`, `X | None` instead of `Optional[X]`).
- Use `pydantic.BaseModel` for structured data and LLM response schemas.
- Use `stamina` for retry logic.
- Use `tqdm` for console output formatting.
- Handle secrets via environment variables, never hardcode them.

## Notebook sync rule

**Always update `notebook.ipynb` when you change documentation in `docs/`.** The notebook mirrors the tutorial content and must stay in sync with the docs.

## Testing

Run tests with:

```bash
uv run pytest tests/ -v
```

The test suite (`tests/test_docs_python.py`) validates that every Python code block in the documentation is syntactically valid. It parses all ` ```python ` blocks from `docs/*.md` and runs `ast.parse` on each one. Partial snippets (all-indented blocks or single-line definitions) are automatically skipped.

If you add or modify code blocks in the docs, run the tests to verify they parse correctly.

## Pre-commit hooks

Always run pre-commit hooks after making changes:

```bash
uv run pre-commit run --all-files
```

Configured hooks: ruff lint + format, trailing whitespace, end-of-file fixer, YAML/TOML/JSON/AST checks, merge conflict detection, debug statement checks, and large file guards (99 MB limit).

## CI pipeline

GitHub Actions (`.github/workflows/docs.yaml`) runs on every push:

1. **Test** — `uv run pytest tests/ -v`
2. **Build** — `sphinx-build -M html ./docs ./_build/`
3. **Deploy** — uploads to S3 (main branch only)
