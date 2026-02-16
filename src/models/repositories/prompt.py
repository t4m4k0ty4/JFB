from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ..repositories import validate_file_path
from .cache import LRUCache


@dataclass(slots=True)
class PromptEntry:
    """Cached prompt payload."""

    prompt_text: str


class PromptRepository:
    """Repository for cached prompt templates loaded from files."""

    def __init__(self, capacity: int = 100) -> None:
        """Create prompt repository with bounded in-memory cache."""
        self.prompts = LRUCache(capacity)

    def get_hash_key(self, file_path: str | Path) -> str:
        """Build cache key based on file path and file version metadata."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return sha256(
            file_path.as_posix().encode("utf-8")
            + str(file_path.stat().st_mtime_ns).encode("utf-8")
            + str(file_path.stat().st_size).encode("utf-8")
        ).hexdigest()

    def check_prompt_exists(self, hash_key: str) -> bool:
        """Check if prompt entry exists in cache."""
        return self.prompts.get(hash_key) is not None

    def add_prompt(self, file_path: str | Path) -> None:
        """Load prompt text from file and cache it when missing."""
        file_path = validate_file_path(file_path)

        hash_key = self.get_hash_key(file_path)
        is_prompt_exists = self.check_prompt_exists(hash_key)
        if not is_prompt_exists:
            prompt_text = file_path.read_text(encoding="utf-8")
            self.prompts.put(hash_key, PromptEntry(prompt_text=prompt_text))

    def get_prompt(self, file_path: str | Path) -> PromptEntry:
        """Get cached prompt entry, loading prompt file on cache miss."""
        file_path = validate_file_path(file_path)

        hash_key = self.get_hash_key(file_path)
        prompt_entry = self.prompts.get(hash_key)
        if prompt_entry is None:
            self.add_prompt(file_path)
            prompt_entry = self.prompts.get(hash_key)

        if prompt_entry is None:
            raise RuntimeError(f"Prompt '{file_path}' was not cached after load.")
        return prompt_entry
