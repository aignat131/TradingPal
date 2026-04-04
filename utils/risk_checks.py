from typing import Optional


# Thresholds per risk tolerance
_THRESHOLDS = {
    "low":    {"max_investment": 5_000,  "max_drop_pct": 5,  "near_high_pct": 5},
    "medium": {"max_investment": 20_000, "max_drop_pct": 10, "near_high_pct": 10},
    "high":   {"max_investment": 100_000,"max_drop_pct": 20, "near_high_pct": 20},
}


def assess_risk(
    price: float,
    amount: float,
    risk_tolerance: str,
    change_pct: float = 0.0,
    high_52w: Optional[float] = None,
    low_52w: Optional[float] = None,
) -> dict:
    """
    Returns:
        {
          "level":    "LOW" | "MEDIUM" | "HIGH",
          "warnings": [list of warning strings],
          "score":    int (0-100, higher = riskier)
        }
    """
    t = _thresholds(risk_tolerance)
    warnings = []
    score = 0

    # 1. Investment size check
    if amount > t["max_investment"]:
        warnings.append(
            f"Investment of ${amount:,.0f} exceeds the recommended limit "
            f"of ${t['max_investment']:,.0f} for {risk_tolerance} risk tolerance."
        )
        score += 30

    # 2. Large single-day move
    if abs(change_pct) > t["max_drop_pct"]:
        direction = "surge" if change_pct > 0 else "drop"
        warnings.append(
            f"Large intraday {direction} of {change_pct:+.2f}% detected — "
            f"heightened volatility may impact your entry/exit price."
        )
        score += 25

    # 3. Price near 52-week high (overbought risk)
    if high_52w and high_52w > 0:
        pct_from_high = (high_52w - price) / high_52w * 100
        if pct_from_high < t["near_high_pct"]:
            warnings.append(
                f"Price is within {pct_from_high:.1f}% of the 52-week high — "
                f"limited upside headroom."
            )
            score += 20

    # 4. Price near 52-week low (potential distress)
    if low_52w and low_52w > 0 and price > 0:
        pct_from_low = (price - low_52w) / price * 100
        if pct_from_low < 5:
            warnings.append(
                f"Price is within {pct_from_low:.1f}% of the 52-week low — "
                f"may indicate distressed asset."
            )
            score += 20

    # 5. Extreme volatility (day gain > 15%)
    if abs(change_pct) > 15:
        warnings.append(
            "Extreme volatility (>15% daily move). Consider waiting for "
            "price stabilisation before trading."
        )
        score += 25

    level = _score_to_level(score, risk_tolerance)
    return {"level": level, "warnings": warnings, "score": min(score, 100)}


def _thresholds(risk_tolerance: str) -> dict:
    return _THRESHOLDS.get(risk_tolerance.lower(), _THRESHOLDS["medium"])


def _score_to_level(score: int, risk_tolerance: str) -> str:
    if risk_tolerance == "high":
        if score >= 60:
            return "HIGH"
        if score >= 25:
            return "MEDIUM"
        return "LOW"
    if risk_tolerance == "low":
        if score >= 20:
            return "HIGH"
        if score >= 5:
            return "MEDIUM"
        return "LOW"
    # medium
    if score >= 45:
        return "HIGH"
    if score >= 20:
        return "MEDIUM"
    return "LOW"
