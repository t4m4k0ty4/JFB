from pathlib import Path


def validate_file_path(file_path: str | Path) -> Path:
    """Validate that prompt file path is absolute.

    Args:
        file_path: Prompt file path as string or Path.

    Returns:
        Normalized Path instance.

    Raises:
        ValueError: If path is not absolute.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.is_absolute():
        raise ValueError(f"Prompt file path '{file_path}' must be absolute.")
    return file_path
