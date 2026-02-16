from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import orjson
from jsonschema import Draft202012Validator, FormatChecker

from ..repositories import validate_file_path
from .cache import LRUCache


@dataclass(slots=True)
class SchemaEntry:
    schema_dict: dict[str, Any]
    compiled_validator: Draft202012Validator


class SchemaRepository:
    def __init__(self) -> None:
        self.schemas = LRUCache(100)

    def get_hash_key(self, file_path: str | Path) -> str:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return sha256(
            file_path.as_posix().encode("utf-8")
            + str(file_path.stat().st_mtime_ns).encode("utf-8")
            + str(file_path.stat().st_size).encode("utf-8")
        ).hexdigest()

    def check_schema_exists(self, hash_key: str) -> bool:
        return self.schemas.get(hash_key) is not None

    def check_schema_valid(self, schema: dict[str, Any]) -> None:
        Draft202012Validator.check_schema(schema)

    def add_schema(self, file_path: str | Path) -> None:
        file_path = validate_file_path(file_path)

        hash_key = self.get_hash_key(file_path)
        is_shema_exists = self.check_schema_exists(hash_key)
        if not is_shema_exists:
            schema = orjson.loads(file_path.read_bytes())
            self.check_schema_valid(schema)
            compiled_validator = Draft202012Validator(schema, format_checker=FormatChecker())
            self.schemas.put(hash_key, SchemaEntry(schema_dict=schema, compiled_validator=compiled_validator))

    def get_schema(self, file_path: str | Path) -> SchemaEntry:
        file_path = validate_file_path(file_path)

        hash_key = self.get_hash_key(file_path)
        schema_entry = self.schemas.get(hash_key)

        return schema_entry
