import errno
from pathlib import Path

import polars as pl
import pytest
from models.case import CaseManager, RunConfigEntry


@pytest.fixture
def existing_cases_root(tmp_path: Path) -> Path:
    """Create a valid cases root directory with required subdirectories."""
    root_dir = tmp_path / "existing"
    (root_dir / "cases").mkdir(parents=True)
    (root_dir / "schemas").mkdir(parents=True)
    (root_dir / "runs").mkdir(parents=True)
    (root_dir / "prompts").mkdir(parents=True)
    return root_dir


class TestCaseManager:
    """Unit tests for CaseManager directory validation and lifecycle."""

    def test_create_mode_initializes_root_and_structure(self, tmp_path: Path) -> None:
        """Create root directory and required structure when create=True."""
        root_dir = tmp_path / "benchmark_data"
        print(tmp_path)

        manager = CaseManager(root_dir, create=True)

        assert root_dir.is_dir()
        assert (root_dir / "cases").is_dir()
        assert (root_dir / "schemas").is_dir()
        assert manager.root_dir == root_dir.absolute()
        assert manager.is_directory_structure_valid is True
        assert manager.is_directory_valid is True

    def test_validate_existing_structure_without_create(self, existing_cases_root: Path) -> None:
        """Validate existing directory tree when create=False."""
        manager = CaseManager(existing_cases_root, create=False)

        assert manager.root_dir == existing_cases_root.absolute()
        assert manager.is_directory_structure_valid is True
        assert manager.is_directory_valid is True

    def test_missing_root_without_create_raises_file_not_found(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError when root does not exist and create=False."""
        root_dir = tmp_path / "missing_root"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            CaseManager(root_dir, create=False)

    def test_file_path_as_root_raises_not_a_directory(self, tmp_path: Path) -> None:
        """Raise NotADirectoryError when root path points to a file."""
        root_file = tmp_path / "root.txt"
        root_file.write_text("not a directory", encoding="utf-8")

        with pytest.raises(NotADirectoryError, match="not a directory"):
            CaseManager(root_file, create=True)

    @pytest.mark.parametrize(
        ("present_dirs", "expected_error"),
        [
            (("schemas",), "Cases directory"),
            (("cases",), "Schemas directory"),
        ],
    )
    def test_missing_required_structure_raises_when_create_false(
        self, tmp_path: Path, present_dirs: tuple[str, ...], expected_error: str
    ) -> None:
        """Raise FileNotFoundError for missing required subdirectories."""
        root_dir = tmp_path / "broken_structure"
        root_dir.mkdir()

        for directory in present_dirs:
            (root_dir / directory).mkdir()

        with pytest.raises(FileNotFoundError, match=expected_error):
            CaseManager(root_dir, create=False)

    @pytest.mark.parametrize(
        ("path_parts", "expected_error"),
        [
            (("cases_root", "..", "other"), "Invalid directory name component"),
            (("bad\x01name",), "Invalid directory name component"),
            ((("a" * 256),), "exceeds 255 bytes"),
        ],
    )
    def test_invalid_directory_components_raise_value_error(
        self, tmp_path: Path, path_parts: tuple[str, ...], expected_error: str
    ) -> None:
        """Raise ValueError for invalid directory name components."""
        invalid_root = tmp_path.joinpath(*path_parts)

        with pytest.raises(ValueError, match=expected_error):
            CaseManager(invalid_root, create=True)

    def test_invalid_os_error_during_mkdir_is_wrapped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrap invalid mkdir OS errors into a user-facing ValueError."""
        root_dir = tmp_path / "new_root"

        def fake_mkdir(self: Path, parents: bool = False, exist_ok: bool = False) -> None:
            raise OSError(errno.EINVAL, "Invalid argument")

        monkeypatch.setattr(Path, "mkdir", fake_mkdir)

        with pytest.raises(ValueError, match="Invalid directory name"):
            CaseManager(root_dir, create=True)

    def test_remove_cases_directory_deletes_root(self, tmp_path: Path) -> None:
        """Delete root directory recursively via CaseManager helper."""
        root_dir = tmp_path / "to_remove"
        manager = CaseManager(root_dir, create=True)

        manager._remove_cases_directory()

        assert not root_dir.exists()

    def test_load_run_config_from_csv_returns_entries(self, existing_cases_root: Path) -> None:
        """Load run configuration rows from CSV file into typed entries."""
        run_config_path = existing_cases_root / "runs" / "test_run.csv"
        run_config_path.write_text(
            "model_id,case_name\nmodel-a,case-1\nmodel-b,case-2\n",
            encoding="utf-8",
        )
        manager = CaseManager(existing_cases_root, create=False)

        entries = manager.load_run_config("test_run.csv")

        assert len(entries) == 2
        assert all(isinstance(entry, RunConfigEntry) for entry in entries)
        assert [(entry.model_id, entry.case_name) for entry in entries] == [
            ("model-a", "case-1"),
            ("model-b", "case-2"),
        ]

    def test_load_run_config_from_absolute_external_csv_path(self, existing_cases_root: Path, tmp_path: Path) -> None:
        """Load run configuration from absolute CSV path outside default runs directory."""
        external_run_config_path = (tmp_path / "external_run.csv").absolute()
        external_run_config_path.write_text(
            "model_id,case_name\nmodel-a,case-1\n",
            encoding="utf-8",
        )
        manager = CaseManager(existing_cases_root, create=False)

        entries = manager.load_run_config(external_run_config_path)

        assert len(entries) == 1
        assert entries[0].model_id == "model-a"
        assert entries[0].case_name == "case-1"

    def test_load_run_config_from_xlsx_returns_entries(
        self, existing_cases_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load run configuration rows from XLSX file into typed entries."""
        run_config_path = existing_cases_root / "runs" / "test_run.xlsx"
        run_config_path.write_bytes(b"placeholder")
        manager = CaseManager(existing_cases_root, create=False)

        def fake_read_excel(path: Path, schema_overrides: pl.Schema) -> pl.DataFrame:
            assert path == run_config_path
            assert schema_overrides == pl.Schema({"model_id": pl.Utf8, "case_name": pl.Utf8})
            return pl.DataFrame({"model_id": ["model-a"], "case_name": ["case-1"]})

        monkeypatch.setattr(pl, "read_excel", fake_read_excel)

        entries = manager.load_run_config("test_run.xlsx")

        assert len(entries) == 1
        assert entries[0].model_id == "model-a"
        assert entries[0].case_name == "case-1"

    def test_load_run_config_raises_for_missing_file(self, existing_cases_root: Path) -> None:
        """Raise FileNotFoundError when run configuration file is absent."""
        manager = CaseManager(existing_cases_root, create=False)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            manager.load_run_config("missing.csv")

    def test_load_run_config_raises_for_unsupported_format(self, existing_cases_root: Path) -> None:
        """Raise ValueError for unsupported run configuration file extension."""
        run_config_path = existing_cases_root / "runs" / "test_run.json"
        run_config_path.write_text("{}", encoding="utf-8")
        manager = CaseManager(existing_cases_root, create=False)

        with pytest.raises(ValueError, match="Unsupported run configuration file format"):
            manager.load_run_config("test_run.json")
