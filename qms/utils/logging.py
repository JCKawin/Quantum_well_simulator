from __future__ import annotations

import logging

_BASE_LOGGER_NAME = "qms"
_configured = False


def _ensure_logging() -> None:
    global _configured
    if _configured:
        return

    logger = logging.getLogger(_BASE_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    _ensure_logging()
    if not name:
        return logging.getLogger(_BASE_LOGGER_NAME)

    suffix = name.replace("__main__", "main")
    if suffix.startswith("qms"):
        return logging.getLogger(suffix)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{suffix}")


def set_log_level(level: int | str) -> None:
    _ensure_logging()
    logging.getLogger(_BASE_LOGGER_NAME).setLevel(level)


__all__ = ["get_logger", "set_log_level"]