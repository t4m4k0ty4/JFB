from pathlib import Path

import orjson
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from models.repositories.schema import SchemaEntry, SchemaRepository, validate_file_path


@pytest.fixture
def schema_repository() -> SchemaRepository:
    """Create a fresh schema repository instance for each test."""
    return SchemaRepository()


@pytest.fixture
def valid_schema_file(tmp_path: Path) -> Path:
    """Create a valid JSON schema file and return its absolute path."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    file_path = tmp_path / "person.schema.json"
    file_path.write_bytes(orjson.dumps(schema))
    return file_path.absolute()


class TestSchemaRepository:
    """Unit tests for schema path validation and repository operations."""

    @pytest.mark.parametrize("path_value", [Path("schema.json"), "schema.json"])
    def test_validate_schema_file_path_raises_for_relative_paths(self, path_value: str | Path) -> None:
        """Raise ValueError when schema path is not absolute."""
        with pytest.raises(ValueError, match="must be absolute"):
            validate_file_path(path_value)

    @pytest.mark.parametrize("as_string", [False, True])
    def test_validate_schema_file_path_accepts_absolute_paths(self, tmp_path: Path, as_string: bool) -> None:
        """Accept absolute path provided as Path or string."""
        absolute_path = (tmp_path / "schema.json").absolute()
        path_value = str(absolute_path) if as_string else absolute_path

        assert validate_file_path(path_value) == absolute_path

    def test_add_schema_stores_schema_entry(self, schema_repository: SchemaRepository, valid_schema_file: Path) -> None:
        """Store parsed schema and compiled validator in cache."""
        schema_repository.add_schema(valid_schema_file)
        hash_key = schema_repository.get_hash_key(valid_schema_file)
        schema_entry = schema_repository.schemas.get(hash_key)

        assert isinstance(schema_entry, SchemaEntry)
        assert isinstance(schema_entry.compiled_validator, Draft202012Validator)
        assert schema_entry.schema_dict["type"] == "object"
        assert schema_entry.schema_dict["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    def test_add_schema_raises_for_invalid_json(self, schema_repository: SchemaRepository, tmp_path: Path) -> None:
        """Raise JSONDecodeError when schema file contains invalid JSON."""
        invalid_json_path = (tmp_path / "invalid.schema.json").absolute()
        invalid_json_path.write_bytes(b"{invalid-json")

        with pytest.raises(orjson.JSONDecodeError):
            schema_repository.add_schema(invalid_json_path)

    def test_add_schema_raises_for_invalid_schema(self, schema_repository: SchemaRepository, tmp_path: Path) -> None:
        """Raise SchemaError when JSON document is not a valid JSON Schema."""
        invalid_schema_path = (tmp_path / "invalid-draft.schema.json").absolute()
        invalid_schema_path.write_bytes(orjson.dumps({"type": "unknown_type"}))

        with pytest.raises(SchemaError):
            schema_repository.add_schema(invalid_schema_path)
