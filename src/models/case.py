import errno
import re
from pathlib import Path
from shutil import rmtree
from typing import Any

import orjson
from pydantic import BaseModel, Field


class Case(BaseModel):
    raw: str = Field(title="raw source data", description="Raw case data as a string.")
    expected_value: dict[str, Any] = Field(
        title="Expacted valid result of LLM generation",
        description="Expected valid result of LLM generation as a dictionary.",
    )
    schema_: str = Field(
        title="JSON schema name", description="Name of the JSON schema to validate against.", alias="schema"
    )


class CaseManager:
    """Summary.

    Args:
        param1 (type): Description.

    Returns:
        return_type: Description.

    Raises:
        ExceptionType: Description.
    """

    CASE_DIR_NAME = "cases"
    SCHEMAS_DIR_NAME = "schemas"

    def __init__(self, dir: str | Path, create: bool) -> None:
        self.is_directory_valid = False
        self.is_directory_structure_valid = False
        self.init_cases_directory(dir, create)

    def init_cases_directory(self, dir_name: str | Path, create: bool) -> None:
        self.__validate_case_directory(dir_name, create)
        self.cases_dir = self.root_dir / self.CASE_DIR_NAME
        self.schemas_dir = self.root_dir / self.SCHEMAS_DIR_NAME
        self.__validate_cases_directory_structure(create)

    def __validate_cases_directory_structure(self, create: bool) -> None:
        if create:
            self.__init_case_directory_structure()
        else:
            self.__check_case_directory_structure()

    def __init_case_directory_structure(self) -> None:
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.is_directory_structure_valid = True

    def __check_case_directory_structure(self) -> None:
        if not self.cases_dir.exists():
            raise FileNotFoundError(f"Cases directory '{self.cases_dir}' does not exist.")
        if not self.schemas_dir.exists():
            raise FileNotFoundError(f"Schemas directory '{self.schemas_dir}' does not exist.")

        self.is_directory_structure_valid = True

    def _remove_cases_directory(self) -> None:
        rmtree(self.root_dir)

    def __validate_case_directory(self, root_dir: str | Path, create: bool) -> None:
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.__validate_dir_name_components(root_dir)

        if root_dir.exists() and not root_dir.is_dir():
            raise NotADirectoryError(f"Root directory for test cases'{root_dir}' is not a directory.")

        if not root_dir.exists():
            if not create:
                raise FileNotFoundError(f"Root directory for test cases'{root_dir}' does not exist.")
            else:
                try:
                    root_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    if exc.errno in {errno.EINVAL, errno.ENAMETOOLONG, errno.ENOTDIR}:
                        raise ValueError(f"Invalid directory name '{root_dir}'.") from exc
                    raise

        self.root_dir = root_dir.absolute()
        self.is_directory_valid = True

    def __validate_dir_name_components(self, root_dir: Path) -> None:
        MAX_NAME_BYTES = 255

        for part in root_dir.parts:
            if part in {"", root_dir.anchor}:
                continue

            if part in {".", ".."}:
                raise ValueError(f"Invalid directory name component '{part}'.")

            if re.search(r"[\x00-\x1F\x7F]", part):
                raise ValueError(f"Invalid directory name component '{part}'.")

            if len(part.encode("utf-8")) > MAX_NAME_BYTES:
                raise ValueError(f"Directory name component '{part}' exceeds {MAX_NAME_BYTES} bytes.")

    def load_case(self, case_name: str) -> Case:
        case_path = self.cases_dir / f"{case_name}.json"
        if not case_path.exists():
            raise FileNotFoundError(f"Case file '{case_path}' does not exist.")

        case_dict = orjson.loads(case_path.read_bytes())
        return Case.model_validate(case_dict)
