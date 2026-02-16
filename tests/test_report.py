from pathlib import Path
from typing import Any

import polars as pl
import pytest
from models.report import DEFAULT_SCHEMA, DEFAULT_WORKSHEET, Report


@pytest.fixture
def report_rows() -> list[dict[str, Any]]:
    """Create minimal report rows suitable for report generation tests."""
    return [{"run_id": 1, "run_name": "baseline"}]


@pytest.fixture
def report_output_file(tmp_path: Path) -> Path:
    """Provide a temporary output file path for report generation."""
    return tmp_path / "report.xlsx"


class TestReport:
    """Unit tests for report writer behavior."""

    @pytest.mark.parametrize("output_as_string", [False, True])
    def test_init_sets_output_file_and_default_worksheet(
        self, report_output_file: Path, output_as_string: bool
    ) -> None:
        """Store output file and use default worksheet name."""
        output_file: str | Path = str(report_output_file) if output_as_string else report_output_file

        report = Report(output_file=output_file)

        assert report.output_file == output_file
        assert report.worksheet == DEFAULT_WORKSHEET
        assert report.schema == DEFAULT_SCHEMA

    def test_init_accepts_custom_worksheet(self, report_output_file: Path) -> None:
        """Store provided custom worksheet name."""
        report = Report(output_file=report_output_file, worksheet="Summary")

        assert report.worksheet == "Summary"

    def test_init_accepts_custom_schema(self, report_output_file: Path) -> None:
        """Store custom schema provided by caller."""
        custom_schema = pl.Schema({"run_id": pl.Int64})

        report = Report(output_file=report_output_file, schema=custom_schema)

        assert report.schema == custom_schema

    def test_generate_calls_write_excel_with_expected_arguments(
        self, report_output_file: Path, report_rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build dataframe with schema and delegate export to write_excel."""
        captured_call: dict[str, Any] = {}

        def fake_write_excel(self: pl.DataFrame, workbook: str | Path, worksheet: str) -> None:
            captured_call["self"] = self
            captured_call["workbook"] = workbook
            captured_call["worksheet"] = worksheet

        monkeypatch.setattr(pl.DataFrame, "write_excel", fake_write_excel)
        report = Report(output_file=report_output_file, worksheet="ResultsSheet")

        report.generate(report_rows)

        assert captured_call["self"].schema == DEFAULT_SCHEMA
        assert captured_call["self"].to_dicts() == [
            {
                "run_id": 1,
                "run_name": "baseline",
                "schema": None,
                "case": None,
                "model": None,
                "similarity": None,
                "field_match": None,
                "field_match_count": None,
                "value_match": None,
                "field_value_match_count": None,
                "llm_response_time_ms": None,
                "etalon": None,
                "llm_result": None,
                "error": None,
                "llm_response": None,
            }
        ]
        assert captured_call["workbook"] == report_output_file
        assert captured_call["worksheet"] == "ResultsSheet"

    def test_generate_raises_when_data_does_not_match_schema(self, report_output_file: Path) -> None:
        """Raise Polars error when row values cannot be cast to schema types."""
        report = Report(output_file=report_output_file)
        invalid_rows = [{"run_id": "not-an-int", "run_name": "baseline"}]

        with pytest.raises(pl.exceptions.ComputeError):
            report.generate(invalid_rows)
