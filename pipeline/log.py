"""Centralized loguru logging for the pipeline.

Configures a single file sink under data/logs/ with rotation.
Import `logger` from this module wherever logging is needed.

Intercepts stdlib `logging` so modules using it (e.g. backup.py)
also route through loguru and into the log file.
"""

import logging
import sys

from loguru import logger

from pipeline.config import DATA_DIR

LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove default stderr handler so we control the format
logger.remove()

# Stderr: concise, no timestamp (NiceGUI / terminal already shows context)
logger.add(
    sys.stderr,
    level="INFO",
    format="<level>{level: <8}</level> | {message}",
)

# File: full detail with rotation
logger.add(
    LOG_DIR / "pipeline.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
)


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

# Suppress noisy HTTP request logs from httpx/openai that break tqdm bars
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
