import logging

import pytest
from models.logger import ClickStderrHandler, configure_logging, resolve_log_level


@pytest.fixture(autouse=True)
def restore_root_logger() -> None:
    """Restore root logger handlers and level after each test."""
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level

    yield

    root_logger.handlers.clear()
    root_logger.handlers.extend(previous_handlers)
    root_logger.setLevel(previous_level)


class TestResolveLogLevel:
    """Unit tests for log level resolution from CLI options."""

    @pytest.mark.parametrize(
        ("verbose", "quiet", "log_level", "expected"),
        [
            (0, False, None, logging.WARNING),
            (1, False, None, logging.INFO),
            (2, False, None, logging.DEBUG),
            (10, False, None, logging.DEBUG),
            (0, True, None, logging.ERROR),
            (2, True, None, logging.ERROR),
            (0, False, "warning", logging.WARNING),
            (0, False, "DEBUG", logging.DEBUG),
        ],
    )
    def test_resolve_log_level(self, verbose: int, quiet: bool, log_level: str | None, expected: int) -> None:
        """Resolve expected log level for option combinations."""
        assert resolve_log_level(verbose=verbose, quiet=quiet, log_level=log_level) == expected


class TestConfigureLogging:
    """Unit tests for click-compatible logging configuration."""

    def test_configure_logging_replaces_root_handlers(self) -> None:
        """Install only click handler on root logger."""
        root_logger = logging.getLogger()
        root_logger.addHandler(logging.StreamHandler())

        configure_logging(verbose=1)

        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], ClickStderrHandler)
        assert root_logger.level == logging.INFO

    def test_configure_logging_writes_logs_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Emit formatted logs to stderr through click handler."""
        configure_logging(verbose=1)
        logger = logging.getLogger("tests.app_logger")

        logger.info("test info log")
        captured = capsys.readouterr()

        assert "test info log" in captured.err
        assert "INFO" in captured.err
        assert captured.out == ""
