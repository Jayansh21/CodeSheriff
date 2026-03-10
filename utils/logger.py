"""
CodeSheriff Logging Module
Provides a reusable logger factory with consistent formatting.
"""

import logging
import sys
from utils.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Format: [TIMESTAMP] [LEVEL] [MODULE] message
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when function is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
