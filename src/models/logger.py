import logging
from typing import Final

import click

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT: Final[str] = "%H:%M:%S"

LEVEL_COLORS: Final[dict[int, str]] = {
    logging.DEBUG: "bright_black",
    logging.INFO: "blue",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "red",
}


class ClickStderrHandler(logging.Handler):
    """Logging handler that writes messages via click to stderr."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit formatted log record to click stderr stream."""
        try:
            message = self.format(record)
            click.secho(message, err=True, fg=LEVEL_COLORS.get(record.levelno))
        except Exception:
            self.handleError(record)


def resolve_log_level(verbose: int = 0, quiet: bool = False, log_level: str | None = None) -> int:
    """Resolve effective logging level from CLI options."""
    if log_level is not None:
        return logging._nameToLevel[log_level.upper()]
    if quiet:
        return logging.ERROR
    if verbose >= 2:
        return logging.DEBUG
    if verbose == 1:
        return logging.INFO
    return logging.WARNING


def configure_logging(verbose: int = 0, quiet: bool = False, log_level: str | None = None) -> None:
    """Configure root logger for click CLI output."""
    level = resolve_log_level(verbose=verbose, quiet=quiet, log_level=log_level)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handler = ClickStderrHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    root_logger.addHandler(handler)
