# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

JFBench is a CLI tool for benchmarking LLM models on JSON extraction tasks. It runs models against test cases, validates their JSON output against schemas, and measures quality metrics (`similarity`, `field_match`, `value_match`).

## Commands

```bash
# Install dependencies
uv sync --dev

# Run all tests
uv run pytest -q tests

# Run a single test file
uv run pytest -q tests/<target_test>.py

# Lint and format
uv run ruff format <changed_paths>
uv run ruff check <changed_paths>

# Type check (LLM clients specifically)
uv run pyright tests/test_llm_clients.py

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files

# Initialize a benchmark directory
uv run jfb --new /path/to/benchmark

# Run a benchmark
uv run jfb /path/to/benchmark lmstudio demo.csv /path/to/results.csv --api-host localhost:1234 -v
```

## Architecture

The CLI pipeline in `src/models/cli.py` orchestrates the full run:

1. `CaseManager` (`case.py`) — validates/creates the benchmark directory (`cases/`, `schemas/`, `prompts/`, `runs/`) and loads cases and run configs (CSV/XLSX).
2. `SchemaRepository` / `PromptRepository` (`repositories/`) — file-backed caches (built on `LRUCache`) for JSON schemas and system prompts. Cache keys include `st_mtime_ns` and `st_size` for automatic invalidation on file changes. File paths must be absolute.
3. `LLMClientManager` / `LMStudioClientAdapter` (`llm_clients/`) — factory and adapter for LLM providers. Currently only `lmstudio` is supported. Adapters are long-lived and reused per unique system prompt. `api_host` must be `host:port` without a URL scheme.
4. `Estimator` (`estimator.py`) — computes flat-key similarity metrics between expected and actual JSON dicts using `deepdiff`. Scores are stored by run ID.
5. `Report` (`report.py`) — builds a `polars.DataFrame` from result rows and writes to `.csv` or `.xlsx` (`xlsxwriter` required for Excel).

**Data flow per benchmark entry**: load case → compile schema → load prompt → get/create adapter → call LLM → validate response → compute scores → append row.

## Key Constraints

- `api_host` format: `host:port` only (e.g. `localhost:1234`), no `http://` prefix.
- Repository file paths must be absolute.
- Run config CSV/XLSX must have columns: `model_id`, `case_name`.
- Case JSON files must have: `raw`, `expected_value`, `schema` (filename in `schemas/`).
- Output path must end in `.csv` or `.xlsx`.
- Adapter lifecycle: always call `adapter.close()` — `LMStudioClientAdapter` unloads the model after each `generate_response` call.

## Code Style

- Line length: 120 (ruff).
- Docstrings: Google style.
- Quotes: double.
- Tests: use `pytest.mark.parametrize` for matrix scenarios, `monkeypatch`/fakes over `MagicMock`, cover both happy and failure paths.
