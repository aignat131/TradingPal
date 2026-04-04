"""
Helper utilities — formatting and common operations.
"""
from typing import Optional


def format_currency(value: float, decimals: int = 2) -> str:
    """Return a formatted currency string, e.g. '$1,234.56'."""
    return f"${value:,.{decimals}f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Return a formatted percent string, e.g. '+3.45%'."""
    return f"{value:+.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning *default* if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to [low, high]."""
    return max(low, min(high, value))


def score_to_label(score: float) -> str:
    """Convert a [-1, 1] score to a human-readable label."""
    if score > 0.3:
        return "Strongly Bullish"
    if score > 0.1:
        return "Mildly Bullish"
    if score < -0.3:
        return "Strongly Bearish"
    if score < -0.1:
        return "Mildly Bearish"
    return "Neutral"


def truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate *text* to *max_len* characters with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
