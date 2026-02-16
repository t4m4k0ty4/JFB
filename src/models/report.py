from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_SCHEMA = pl.Schema(
    {
        "run_id": pl.Int64,
        "run_name": pl.Utf8,
        "schema": pl.Utf8,
        "case": pl.Utf8,
        "model": pl.Utf8,
        "similarity": pl.Float64,
        "field_match": pl.Float64,
        "field_match_count": pl.Int64,
        "value_match": pl.Float64,
        "field_value_match_count": pl.Int64,
        "llm_response_time_ms": pl.Int64,
        "etalon": pl.Utf8,
        "llm_result": pl.Utf8,
        "error": pl.Utf8,
        "llm_response": pl.Utf8,
    }
)
DEFAULT_WORKSHEET = "Results"


class Report:
    def __init__(
        self, output_file: str | Path, schema: pl.Schema = DEFAULT_SCHEMA, worksheet: str = DEFAULT_WORKSHEET
    ) -> None:
        self.worksheet = worksheet
        self.output_file = output_file
        self.schema = schema

    def generate(self, data: list[dict[str, Any]]) -> None:
        df = pl.DataFrame(data, schema=self.schema)
        df.write_excel(workbook=self.output_file, worksheet=self.worksheet)
