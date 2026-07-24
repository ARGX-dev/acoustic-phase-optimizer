"""Logging configuration for the Acoustic Phase Optimizer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_loggers: dict[str, logging.Logger] = {}
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    global _initialized
    if _initialized:
        return

    root = logging.getLogger("acoustic_phase_optimizer")
    root.setLevel(logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(f"acoustic_phase_optimizer.{name}")
    _loggers[name] = logger
    return logger
