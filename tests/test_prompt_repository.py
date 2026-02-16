from pathlib import Path

import pytest
from models.repositories.prompt import PromptEntry, PromptRepository, validate_file_path


@pytest.fixture
def prompt_repository() -> PromptRepository:
    """Create fresh prompt repository for each test."""
    return PromptRepository()


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    """Create prompt file and return absolute path."""
    file_path = tmp_path / "system_prompt.txt"
    file_path.write_text("You are a strict JSON assistant.", encoding="utf-8")
    return file_path.absolute()


class TestPromptRepository:
    """Unit tests for prompt path validation and caching."""

    @pytest.mark.parametrize("path_value", [Path("prompt.txt"), "prompt.txt"])
    def test_validate_prompt_file_path_raises_for_relative_paths(self, path_value: str | Path) -> None:
        """Raise ValueError when prompt path is not absolute."""
        with pytest.raises(ValueError, match="must be absolute"):
            validate_file_path(path_value)

    @pytest.mark.parametrize("as_string", [False, True])
    def test_validate_prompt_file_path_accepts_absolute_paths(self, tmp_path: Path, as_string: bool) -> None:
        """Accept absolute path provided as Path or string."""
        absolute_path = (tmp_path / "prompt.txt").absolute()
        path_value = str(absolute_path) if as_string else absolute_path

        assert validate_file_path(path_value) == absolute_path

    def test_add_prompt_stores_prompt_entry(self, prompt_repository: PromptRepository, prompt_file: Path) -> None:
        """Store prompt text inside cache entry."""
        prompt_repository.add_prompt(prompt_file)
        hash_key = prompt_repository.get_hash_key(prompt_file)
        prompt_entry = prompt_repository.prompts.get(hash_key)

        assert isinstance(prompt_entry, PromptEntry)
        assert prompt_entry.prompt_text == "You are a strict JSON assistant."

    def test_get_prompt_uses_cache_after_first_load(
        self, prompt_repository: PromptRepository, prompt_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Read prompt file once and reuse cached entry on repeated access."""
        original_read_text = Path.read_text
        read_calls_count = 0

        def spy_read_text(self: Path, encoding: str = "utf-8") -> str:
            nonlocal read_calls_count
            read_calls_count += 1
            return original_read_text(self, encoding=encoding)

        monkeypatch.setattr(Path, "read_text", spy_read_text)

        first_entry = prompt_repository.get_prompt(prompt_file)
        second_entry = prompt_repository.get_prompt(prompt_file)

        assert read_calls_count == 1
        assert first_entry is second_entry

    def test_get_prompt_refreshes_cache_when_file_changes(
        self, prompt_repository: PromptRepository, prompt_file: Path
    ) -> None:
        """Load new prompt contents when file version metadata changes."""
        first_entry = prompt_repository.get_prompt(prompt_file)
        prompt_file.write_text("You are concise.", encoding="utf-8")

        second_entry = prompt_repository.get_prompt(prompt_file)

        assert first_entry.prompt_text == "You are a strict JSON assistant."
        assert second_entry.prompt_text == "You are concise."

    def test_get_prompt_raises_for_missing_file(self, prompt_repository: PromptRepository, tmp_path: Path) -> None:
        """Raise FileNotFoundError when prompt file does not exist."""
        missing_file = (tmp_path / "missing_prompt.txt").absolute()

        with pytest.raises(FileNotFoundError):
            prompt_repository.get_prompt(missing_file)
