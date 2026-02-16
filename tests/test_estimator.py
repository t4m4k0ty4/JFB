from typing import Any

import pytest
from models.estimator import EstimationRun, Estimator
from pydantic import ValidationError


@pytest.fixture
def estimator() -> Estimator:
    """Create a fresh estimator instance for each test."""
    return Estimator()


@pytest.fixture
def run_payload() -> dict[str, Any]:
    """Provide a valid estimation run payload."""
    return {"id": "run-1", "etalon": {"name": "Alice"}, "actual": {"name": "Alice"}}


class TestEstimationRun:
    """Unit tests for EstimationRun input validation."""

    def test_creates_estimation_run_from_valid_payload(self, run_payload: dict[str, Any]) -> None:
        """Create EstimationRun when payload has valid types."""
        run = EstimationRun(**run_payload)

        assert run.id == "run-1"
        assert run.etalon == {"name": "Alice"}
        assert run.actual == {"name": "Alice"}

    @pytest.mark.parametrize(
        ("payload", "error_field"),
        [
            ({"id": 1, "etalon": {"a": 1}, "actual": {"a": 1}}, "id"),
            ({"id": "run", "etalon": "invalid", "actual": {"a": 1}}, "etalon"),
            ({"id": "run", "etalon": {"a": 1}, "actual": "invalid"}, "actual"),
        ],
    )
    def test_raises_validation_error_for_invalid_payload_types(self, payload: dict[str, Any], error_field: str) -> None:
        """Raise ValidationError when payload fields have invalid types."""
        with pytest.raises(ValidationError, match=error_field):
            EstimationRun(**payload)


class TestEstimator:
    """Unit tests for Estimator score calculations and batch execution."""

    @pytest.mark.parametrize(
        (
            "run_id",
            "etalon",
            "actual",
            "similarity",
            "field_match",
            "field_match_count",
            "value_match",
            "field_value_match_count",
        ),
        [
            ("same", {"a": 1}, {"a": 1}, 100.0, 100.0, 1, 100.0, 1),
            ("value_changed", {"a": 1}, {"a": 2}, 0.0, 100.0, 1, 0.0, 0),
            (
                "field_added",
                {"a": 1},
                {"a": 1, "b": 2},
                50.0,
                50.0,
                1,
                100.0,
                1,
            ),
            (
                "field_removed",
                {"a": 1, "b": 2},
                {"a": 1},
                50.0,
                50.0,
                1,
                100.0,
                1,
            ),
            ("empty", {}, {}, 100.0, 100.0, 0, 100.0, 0),
            ("disjoint", {"a": 1}, {"b": 1}, 0.0, 0.0, 0, 0.0, 0),
        ],
    )
    def test_estimate_calculates_scores_for_core_scenarios(
        self,
        estimator: Estimator,
        run_id: str,
        etalon: dict[str, Any],
        actual: dict[str, Any],
        similarity: float,
        field_match: float,
        field_match_count: int,
        value_match: float,
        field_value_match_count: int,
    ) -> None:
        """Calculate expected score values for common comparison scenarios."""
        run = EstimationRun(id=run_id, etalon=etalon, actual=actual)
        estimator.estimate(run)
        scores = estimator.scores[run_id]

        assert scores.similarity == pytest.approx(similarity)
        assert scores.field_match == pytest.approx(field_match)
        assert scores.field_match_count == field_match_count
        assert scores.value_match == pytest.approx(value_match)
        assert scores.field_value_match_count == field_value_match_count

    def test_estimate_treats_reordered_list_values_as_equal(self, estimator: Estimator) -> None:
        """Produce full similarity when only list order differs."""
        run = EstimationRun(id="ignore-order", etalon={"a": [1, 2]}, actual={"a": [2, 1]})

        estimator.estimate(run)
        scores = estimator.scores["ignore-order"]

        assert scores.similarity == 100.0
        assert scores.field_match == 100.0
        assert scores.value_match == 100.0

    def test_estimate_detects_iterable_item_added_as_value_mismatch(self, estimator: Estimator) -> None:
        """Decrease similarity when list content changes under the same field."""
        run = EstimationRun(id="iterable-item-added", etalon={"a": [1]}, actual={"a": [1, 2]})

        estimator.estimate(run)
        scores = estimator.scores["iterable-item-added"]

        assert scores.field_match == 100.0
        assert scores.field_match_count == 1
        assert scores.value_match == 0.0
        assert scores.field_value_match_count == 0
        assert scores.similarity == 0.0

    def test_run_processes_multiple_runs_and_returns_scores_dict(self, estimator: Estimator) -> None:
        """Estimate all runs and return internal score mapping."""
        runs = [
            EstimationRun(id="run-1", etalon={"a": 1}, actual={"a": 1}),
            EstimationRun(id="run-2", etalon={"a": 1}, actual={"a": 2}),
        ]

        results = estimator.run(runs)

        assert results is estimator.scores
        assert set(results) == {"run-1", "run-2"}
        assert results["run-1"].similarity == 100.0
        assert results["run-2"].similarity == 0.0

    def test_run_resets_previous_batch_scores(self, estimator: Estimator) -> None:
        """Clear stale scores when run() is called for a new batch."""
        first_batch = [EstimationRun(id="batch-1", etalon={"a": 1}, actual={"a": 1})]
        second_batch = [EstimationRun(id="batch-2", etalon={"a": 1}, actual={"a": 2})]

        estimator.run(first_batch)
        results = estimator.run(second_batch)

        assert set(results) == {"batch-2"}
        assert "batch-1" not in results
        assert results["batch-2"].similarity == 0.0
