"""Simple CLI logger replacement for removed wisent.cli module."""

import logging


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def bind(logger: logging.Logger, **kwargs) -> logging.Logger:
    """Bind context to logger (simplified - just returns the logger)."""
    # In the original, this probably added context fields
    # For now, just return the logger as-is
    return logger
