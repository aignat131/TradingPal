import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.risk_checks import assess_risk


def test_low_risk_clean():
    result = assess_risk(
        price=100.0,
        amount=1000.0,
        risk_tolerance="medium",
        change_pct=0.5,
        high_52w=120.0,
        low_52w=80.0,
    )
    assert result["level"] == "LOW"
    assert result["warnings"] == []
    assert result["score"] == 0


def test_exceeds_investment_limit():
    result = assess_risk(
        price=100.0,
        amount=50_000.0,
        risk_tolerance="low",
        change_pct=1.0,
        high_52w=110.0,
        low_52w=80.0,
    )
    assert result["level"] == "HIGH"
    assert any("exceeds" in w for w in result["warnings"])


def test_large_intraday_drop_triggers_warning():
    result = assess_risk(
        price=90.0,
        amount=500.0,
        risk_tolerance="medium",
        change_pct=-12.0,
        high_52w=120.0,
        low_52w=80.0,
    )
    assert any("volatility" in w.lower() or "drop" in w.lower() for w in result["warnings"])


def test_near_52w_high_warning():
    result = assess_risk(
        price=198.0,
        amount=1000.0,
        risk_tolerance="medium",
        change_pct=0.5,
        high_52w=200.0,
        low_52w=100.0,
    )
    assert any("52-week high" in w for w in result["warnings"])


def test_near_52w_low_warning():
    result = assess_risk(
        price=101.0,
        amount=1000.0,
        risk_tolerance="medium",
        change_pct=-1.0,
        high_52w=200.0,
        low_52w=100.0,
    )
    assert any("52-week low" in w for w in result["warnings"])


def test_extreme_volatility_warning():
    result = assess_risk(
        price=50.0,
        amount=500.0,
        risk_tolerance="high",
        change_pct=20.0,
        high_52w=60.0,
        low_52w=20.0,
    )
    assert any("Extreme volatility" in w for w in result["warnings"])


def test_high_risk_tolerance_allows_larger_amounts():
    result = assess_risk(
        price=100.0,
        amount=50_000.0,
        risk_tolerance="high",
        change_pct=0.0,
        high_52w=110.0,
        low_52w=80.0,
    )
    assert not any("exceeds" in w for w in result["warnings"])


def test_score_capped_at_100():
    result = assess_risk(
        price=101.0,
        amount=999_999.0,
        risk_tolerance="low",
        change_pct=-25.0,
        high_52w=102.0,
        low_52w=100.0,
    )
    assert result["score"] <= 100


if __name__ == "__main__":
    tests = [
        test_low_risk_clean,
        test_exceeds_investment_limit,
        test_large_intraday_drop_triggers_warning,
        test_near_52w_high_warning,
        test_near_52w_low_warning,
        test_extreme_volatility_warning,
        test_high_risk_tolerance_allows_larger_amounts,
        test_score_capped_at_100,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
