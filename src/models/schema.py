from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import orjson
from jsonschema import Draft202012Validator, FormatChecker


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.__validate_capacity(capacity)
        self.capacity = capacity
        self.cache = {}

    def get(self, key: str) -> Any:
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
        self.cache[key] = value

    def __validate_capacity(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")


@dataclass(slots=True)
class SchemaEntry:
    schema_dict: dict[str, Any]
    compiled_validator: Draft202012Validator


def validate_schema_file_path(file_path: str | Path) -> Path:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.is_absolute():
        raise ValueError(f"Schema file path '{file_path}' must be absolute.")
    return file_path


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
        file_path = validate_schema_file_path(file_path)

        hash_key = self.get_hash_key(file_path)
        is_shema_exists = self.check_schema_exists(hash_key)
        if not is_shema_exists:
            schema = orjson.loads(file_path.read_bytes())
            self.check_schema_valid(schema)
            compiled_validator = Draft202012Validator(schema, format_checker=FormatChecker())
            self.schemas.put(hash_key, SchemaEntry(schema_dict=schema, compiled_validator=compiled_validator))

    def get_schema(self, file_path: str | Path) -> SchemaEntry:
        file_path = validate_schema_file_path(file_path)

        hash_key = self.get_hash_key(file_path)
        schema_entry = self.schemas.get(hash_key)

        return schema_entry
