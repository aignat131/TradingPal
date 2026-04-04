"""
Input validators for user-supplied parameters.
"""
import re
from datetime import date
from typing import Tuple


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate a stock/crypto ticker symbol.
    Returns (is_valid, message).
    """
    if not ticker or not ticker.strip():
        return False, "Ticker cannot be empty."
    ticker = ticker.strip().upper()
    # Allow letters, digits, hyphens, dots (e.g. BTC-USD, BRK.B)
    if not re.match(r"^[A-Z0-9.\-]{1,12}$", ticker):
        return False, f"'{ticker}' is not a valid ticker (max 12 chars, letters/digits/-/.)."
    return True, ""


def validate_amount(amount: float, min_val: float = 1.0, max_val: float = 10_000_000.0) -> Tuple[bool, str]:
    """
    Validate an investment amount.
    Returns (is_valid, message).
    """
    if amount < min_val:
        return False, f"Amount must be at least ${min_val:,.2f}."
    if amount > max_val:
        return False, f"Amount exceeds maximum (${max_val:,.0f})."
    return True, ""


def validate_date_range(start: date, end: date) -> Tuple[bool, str]:
    """
    Validate that start < end and the range does not exceed 5 years.
    Returns (is_valid, message).
    """
    if start >= end:
        return False, "Start date must be before end date."
    delta_days = (end - start).days
    if delta_days > 365 * 5:
        return False, "Date range cannot exceed 5 years."
    return True, ""
