# JFBench Agent Guide

This repository implements JFBench, a benchmark for evaluating how well LLMs
extract and format raw input into JSON that matches expected schemas.

Use this file as a practical, project-specific guide for making safe changes
with minimal back-and-forth.

## Runtime and Tooling

- Python version: `>=3.13`
- Dependency manager and runner: `uv`
- Test runner: `pytest`
- Linter/formatter: `ruff`
- Source layout: `src/` (imports in tests use `models.*`)

## Project Structure

- `src/models/case.py`
  - Validates and initializes benchmark directory structure.
  - Expected directories: `cases`, `schemas`, `prompts`, `runs`.
- `src/models/estimator.py`
  - Computes similarity/field/value match metrics for run outputs.
- `src/models/report.py`
  - Builds a `polars.DataFrame` from run rows and exports to Excel.
- `src/models/llm_clients/`
  - `client.py`: provider adapter contract and LM Studio implementation.
  - `__init__.py`: manager/factory that returns provider adapters.
- `src/models/repositories/`
  - `cache.py`: shared `LRUCache`.
  - `schema.py`: schema caching and validator compilation.
  - `prompt.py`: prompt text caching.
  - `__init__.py`: shared file path validation helpers.

## Source of Truth Rules

- For repositories/caching logic, use modules under `src/models/repositories/`.
- Keep public behavior aligned with tests in `tests/`.
- If you change method contracts, update corresponding tests in the same change.

## LLM Client Rules

- `api_host` must be in `host:port` format (no URL scheme).
  - Good: `localhost:1234`
  - Bad: `http://localhost:1234`
- LM Studio adapter is designed as long-lived.
  - Do not silently recreate/close client per request unless explicitly required.
  - Keep explicit close semantics clear and deterministic.
- Ensure model lifecycle cleanup (`unload`) is exception-safe.

## Repository Caching Rules

- File path input must be absolute for repository loaders.
- Cache key versioning should include file metadata (`st_mtime_ns`, `st_size`).
- Repeated access to unchanged files should avoid extra disk reads.
- Changed files should invalidate stale cache entries automatically.

## Report Module Rules

- `Report.generate` accepts row-like data and applies configured schema.
- Excel export requires `xlsxwriter` dependency.
- Preserve report columns defined by `DEFAULT_SCHEMA`.

## Development Rules

- Write clean and readable code.
- Follow PEP 8.
- Prefer explicit, typed function signatures.
- Use Google-style docstrings for public classes/functions.
- Avoid duplication; extract reusable helpers.
- Prefer safe, defensive behavior for I/O, validation, and resource cleanup.

## Verification Workflow (Run After Changes)

1. Format changed files:
   - `ruff format <paths>`
2. Run lint checks:
   - `ruff check <paths>`
3. Run targeted tests first, then full suite:
   - `uv run pytest -q tests/<target_test>.py`
   - `uv run pytest -q tests`
4. If typing-sensitive tests were touched (especially `llm_clients`):
   - `uv run pyright tests/test_llm_clients.py`

## Unit Test Guidelines

- Use `pytest.mark.parametrize` for matrix-like scenarios.
- Use fixtures for setup and isolation.
- Prefer `monkeypatch`/fakes for external SDK behavior in unit tests.
- Do not use `MagicMock` unless testing 3rd-party API interactions is required.
- Cover both happy-path and failure-path behavior.
- Keep tests independent and deterministic.
- Use descriptive test names and clear assertions.
