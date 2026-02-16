import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import orjson
import polars as pl

from models.case import Case, CaseManager, RunConfigEntry
from models.estimator import EstimationRun, Estimator
from models.llm_clients import LLMClientAdapter, LLMClientManager, LLMProvider
from models.logger import configure_logging
from models.report import DEFAULT_SCHEMA, Report
from models.repositories.prompt import PromptRepository
from models.repositories.schema import SchemaRepository

LOGGER = logging.getLogger("jfb.cli")
DEFAULT_API_HOST = "localhost:1234"
OUTPUT_SUFFIXES = {".csv", ".xlsx"}


@click.command()
@click.argument(
    "path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, resolve_path=False),
)
@click.argument(
    "provider",
    required=False,
    type=click.Choice([provider.value for provider in LLMProvider], case_sensitive=False),
)
@click.argument(
    "run_config",
    required=False,
    type=click.Path(path_type=Path, dir_okay=False, resolve_path=False),
)
@click.argument(
    "output_path",
    required=False,
    type=click.Path(path_type=Path, dir_okay=False, resolve_path=False),
)
@click.option(
    "--new",
    "create_new",
    is_flag=True,
    default=False,
    help="Create benchmark directory structure at PATH.",
)
@click.option(
    "--api-host",
    default=DEFAULT_API_HOST,
    show_default=True,
    help="Provider API host in host:port format.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase logging verbosity. Use -vv for debug logs.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Show only errors and critical logs.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default=None,
    help="Set explicit log level. Overrides --verbose/--quiet.",
)
def main(
    path: Path,
    provider: str | None,
    run_config: Path | None,
    output_path: Path | None,
    create_new: bool,
    api_host: str,
    verbose: int,
    quiet: bool,
    log_level: str | None,
) -> None:
    """Run JFBench CLI in init mode or benchmark mode."""
    configure_logging(verbose=verbose, quiet=quiet, log_level=log_level)

    LOGGER.debug(
        (
            "CLI initialized with path=%s provider=%s run_config=%s output_path=%s "
            "create_new=%s api_host=%s verbose=%s quiet=%s log_level=%s"
        ),
        path,
        provider,
        run_config,
        output_path,
        create_new,
        api_host,
        verbose,
        quiet,
        log_level,
    )

    if _is_init_only_mode(provider=provider, run_config=run_config, output_path=output_path):
        if not create_new:
            raise click.UsageError("Use '--new PATH' to initialize a benchmark directory.")
        manager = CaseManager(path, create=True)
        click.echo(f"Initialized benchmark directory: {manager.root_dir}")
        LOGGER.info("Initialization finished for %s", manager.root_dir)
        return

    if provider is None or run_config is None or output_path is None:
        raise click.UsageError(
            "For benchmark run provide all arguments: PATH PROVIDER RUN_CONFIG OUTPUT_PATH.\n"
            "Or use '--new PATH' to initialize a benchmark directory."
        )

    run_name = run_config.stem
    provider_enum = LLMProvider(provider.lower())
    case_manager = CaseManager(path, create=create_new)
    run_config_entries = case_manager.load_run_config(run_config)

    rows = _run_benchmark(
        provider=provider_enum,
        api_host=api_host,
        run_name=run_name,
        case_manager=case_manager,
        run_config_entries=run_config_entries,
    )
    _write_output_report(output_path=output_path.absolute(), rows=rows)

    click.echo(f"Run '{run_name}' completed. Saved {len(rows)} rows to '{output_path.absolute()}'.")
    LOGGER.info("Benchmark run '%s' finished with %s rows", run_name, len(rows))


def _is_init_only_mode(provider: str | None, run_config: Path | None, output_path: Path | None) -> bool:
    return provider is None and run_config is None and output_path is None


def _resolve_schema_path(case_manager: CaseManager, schema_name: str) -> Path:
    schema_path = Path(schema_name)
    if schema_path.suffix == "":
        schema_path = schema_path.with_suffix(".json")
    if not schema_path.is_absolute():
        schema_path = case_manager.schemas_dir / schema_path
    return schema_path.absolute()


def _resolve_prompt_path(case_manager: CaseManager, case_name: str) -> Path:
    return (case_manager.prompts_dir / f"{case_name}.txt").absolute()


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return orjson.dumps(dict(value)).decode("utf-8")
    if isinstance(value, list | tuple):
        return orjson.dumps(list(value)).decode("utf-8")
    return str(value)


def _run_benchmark(
    provider: LLMProvider,
    api_host: str,
    run_name: str,
    case_manager: CaseManager,
    run_config_entries: list[RunConfigEntry],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    estimator = Estimator()
    schema_repository = SchemaRepository()
    prompt_repository = PromptRepository()
    client_manager = LLMClientManager()
    adapters_by_prompt: dict[str, LLMClientAdapter] = {}

    try:
        for index, run_entry in enumerate(run_config_entries, start=1):
            row = _run_single_entry(
                index=index,
                run_name=run_name,
                run_entry=run_entry,
                case_manager=case_manager,
                estimator=estimator,
                schema_repository=schema_repository,
                prompt_repository=prompt_repository,
                client_manager=client_manager,
                provider=provider,
                api_host=api_host,
                adapters_by_prompt=adapters_by_prompt,
            )
            rows.append(row)
    finally:
        for adapter in adapters_by_prompt.values():
            try:
                adapter.close()
            except Exception:
                LOGGER.exception("Failed to close provider adapter cleanly.")

    return rows


def _run_single_entry(
    index: int,
    run_name: str,
    run_entry: RunConfigEntry,
    case_manager: CaseManager,
    estimator: Estimator,
    schema_repository: SchemaRepository,
    prompt_repository: PromptRepository,
    client_manager: LLMClientManager,
    provider: LLMProvider,
    api_host: str,
    adapters_by_prompt: dict[str, LLMClientAdapter],
) -> dict[str, Any]:
    case = case_manager.load_case(run_entry.case_name)
    schema_path = _resolve_schema_path(case_manager, case.schema_)
    schema_repository.add_schema(schema_path)
    schema_entry = schema_repository.get_schema(schema_path)
    if schema_entry is None:
        raise RuntimeError(f"Schema '{schema_path}' is unavailable after cache load.")

    prompt_path = _resolve_prompt_path(case_manager, run_entry.case_name)
    prompt_entry = prompt_repository.get_prompt(prompt_path)

    adapter = adapters_by_prompt.get(prompt_entry.prompt_text)
    if adapter is None:
        adapter = client_manager.get_client(
            provider=provider, api_host=api_host, system_prompt=prompt_entry.prompt_text
        )
        adapters_by_prompt[prompt_entry.prompt_text] = adapter

    llm_response: Mapping[str, Any] | str | None = None
    error: str | None = None
    started_ns = time.perf_counter_ns()

    try:
        with adapter.get_client() as client:
            llm_response = adapter.generate_response(
                client=client,
                model_id=run_entry.model_id,
                json_schema=schema_entry.schema_dict,
                raw_data=case.raw,
            )
        if isinstance(llm_response, Mapping):
            schema_entry.compiled_validator.validate(llm_response)
        else:
            error = "LLM response is not a JSON object."
    except Exception as exc:
        error = str(exc)
        LOGGER.exception("Failed run for model=%s case=%s", run_entry.model_id, run_entry.case_name)

    elapsed_ms = int((time.perf_counter_ns() - started_ns) / 1_000_000)
    actual_payload = dict(llm_response) if isinstance(llm_response, Mapping) else {}
    scores = _estimate_case(
        estimator=estimator,
        run_name=run_name,
        run_index=index,
        case=case,
        actual_payload=actual_payload,
    )

    return {
        "run_id": index,
        "run_name": run_name,
        "schema": case.schema_,
        "case": run_entry.case_name,
        "model": run_entry.model_id,
        "similarity": scores.similarity,
        "field_match": scores.field_match,
        "field_match_count": scores.field_match_count,
        "value_match": scores.value_match,
        "field_value_match_count": scores.field_value_match_count,
        "llm_response_time_ms": elapsed_ms,
        "etalon": orjson.dumps(case.expected_value).decode("utf-8"),
        "llm_result": _as_text(actual_payload) if actual_payload else None,
        "error": error,
        "llm_response": _as_text(llm_response),
    }


def _estimate_case(
    estimator: Estimator, run_name: str, run_index: int, case: Case, actual_payload: dict[str, Any]
) -> Any:
    estimation_id = f"{run_name}:{run_index}"
    estimator.estimate(EstimationRun(id=estimation_id, etalon=case.expected_value, actual=actual_payload))
    return estimator.scores[estimation_id]


def _write_output_report(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix not in OUTPUT_SUFFIXES:
        raise click.BadParameter(
            f"Unsupported output format '{suffix}'. Use .csv or .xlsx file.",
            param_hint="output_path",
        )

    if suffix == ".xlsx":
        report = Report(output_file=output_path)
        report.generate(rows)
        return

    df = pl.DataFrame(rows, schema=DEFAULT_SCHEMA)
    df.write_csv(output_path)
