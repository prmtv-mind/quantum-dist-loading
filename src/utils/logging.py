"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Union

def setup_logger(
    name: str,
    log_file: Path = None,
    level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """
    Configure logger with consistent formatting.

    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level (can be string like "INFO" or int like logging.INFO)

    Returns:
        Configured logger instance
    """
    # Convert string level to logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
