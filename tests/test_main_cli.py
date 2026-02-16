from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import models.cli as cli_module
import orjson
import polars as pl
import pytest
from click.testing import CliRunner
from models.cli import main


class FakeAdapter:
    """Fake long-lived adapter for CLI benchmark tests."""

    def __init__(self, api_host: str, system_prompt: str) -> None:
        self.api_host = api_host
        self.system_prompt = system_prompt
        self.closed = False

    @contextmanager
    def get_client(self) -> Generator[object]:
        """Return fake client session context."""
        yield object()

    def close(self) -> None:
        """Track explicit close call."""
        self.closed = True

    def generate_response(
        self, client: object, model_id: str, json_schema: dict[str, Any], raw_data: str
    ) -> dict[str, str]:
        """Return deterministic payload compatible with test schema."""
        _ = (client, model_id, json_schema, raw_data)
        return {"answer": "ok"}


class FakeClientManager:
    """Fake manager that returns local fake adapters."""

    def get_client(self, provider: Any, api_host: str, system_prompt: str = "") -> FakeAdapter:
        """Build fake adapter with provider-independent behavior."""
        _ = provider
        return FakeAdapter(api_host=api_host, system_prompt=system_prompt)


@pytest.fixture
def benchmark_root(tmp_path: Path) -> Path:
    """Create benchmark root with one case, one schema, one prompt and run config."""
    root = tmp_path / "benchmark"
    (root / "cases").mkdir(parents=True)
    (root / "schemas").mkdir(parents=True)
    (root / "runs").mkdir(parents=True)
    (root / "prompts").mkdir(parents=True)

    (root / "cases" / "case-1.json").write_bytes(
        orjson.dumps(
            {
                "raw": "raw payload",
                "expected_value": {"answer": "ok"},
                "schema": "answer.schema.json",
            }
        )
    )
    (root / "schemas" / "answer.schema.json").write_bytes(
        orjson.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        )
    )
    (root / "prompts" / "case-1.txt").write_text("You are a strict assistant.", encoding="utf-8")
    (root / "runs" / "demo.csv").write_text("model_id,case_name\nmodel-a,case-1\n", encoding="utf-8")
    return root


class TestMainCli:
    """Unit tests for CLI init and benchmark execution modes."""

    def test_main_cli_new_flag_initializes_case_directory(self, tmp_path: Path) -> None:
        """Create benchmark directory structure in init-only mode."""
        root = tmp_path / "new_benchmark"
        runner = CliRunner()

        result = runner.invoke(main, ["--new", str(root)])

        assert result.exit_code == 0
        assert "Initialized benchmark directory" in result.output
        assert (root / "cases").is_dir()
        assert (root / "schemas").is_dir()
        assert (root / "runs").is_dir()
        assert (root / "prompts").is_dir()

    def test_main_cli_requires_run_arguments_without_new(self, tmp_path: Path) -> None:
        """Return usage error when run arguments are missing."""
        root = tmp_path / "missing_args"
        runner = CliRunner()

        result = runner.invoke(main, [str(root)])

        assert result.exit_code != 0
        assert "Use '--new PATH' to initialize" in result.output

    def test_main_cli_runs_benchmark_and_writes_csv_report(
        self, benchmark_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Run benchmark from config and persist per-model result rows into CSV."""
        output_path = tmp_path / "results.csv"
        runner = CliRunner()
        monkeypatch.setattr(cli_module, "LLMClientManager", FakeClientManager)

        result = runner.invoke(
            main,
            [
                str(benchmark_root),
                "lmstudio",
                "demo.csv",
                str(output_path),
                "--api-host",
                "localhost:1234",
            ],
        )

        assert result.exit_code == 0
        assert "Run 'demo' completed." in result.output
        assert output_path.exists()

        rows = pl.read_csv(output_path).to_dicts()
        assert len(rows) == 1
        row = rows[0]
        assert row["run_name"] == "demo"
        assert row["case"] == "case-1"
        assert row["model"] == "model-a"
        assert row["similarity"] == 100.0
        assert row["field_match"] == 100.0
        assert row["value_match"] == 100.0
        assert row["error"] is None

    def test_main_cli_raises_for_unsupported_output_format(self, benchmark_root: Path) -> None:
        """Fail run when output path extension is not csv/xlsx."""
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                str(benchmark_root),
                "lmstudio",
                "demo.csv",
                str(benchmark_root / "results.json"),
            ],
        )

        assert result.exit_code != 0
        assert "Unsupported output format" in result.output
