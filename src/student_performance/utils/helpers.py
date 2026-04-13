"""Shared helper utilities."""
import logging
from typing import List, Tuple


def validate_features(features: dict, required_keys: List[str]) -> Tuple[bool, str]:
    """Check that all required keys are present in features dict."""
    missing = [k for k in required_keys if k not in features]
    if missing:
        return False, f"Missing required feature(s): {', '.join(missing)}"
    return True, ""


def clip_grade(grade: float) -> float:
    """Clip a grade to the valid range [0, 100]."""
    return max(0.0, min(100.0, float(grade)))


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
