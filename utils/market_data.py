import os
import csv
import random

# Optional: install yfinance for live data.  If unavailable we fall back to
# sample_prices.csv so the app still works without a data API key.
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

_SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_prices.csv")


def get_price_data(ticker: str, asset_type: str = "stock") -> dict:
    """
    Returns a dict with keys: price, change_pct, high_52w, low_52w, volume
    Falls back to sample CSV data when yfinance is unavailable or the ticker
    cannot be resolved.
    """
    if _YF_AVAILABLE:
        try:
            return _fetch_yfinance(ticker)
        except Exception:
            pass  # fall through to sample data

    return _fetch_sample(ticker)


def _fetch_yfinance(ticker: str) -> dict:
    info = yf.Ticker(ticker).info

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    if not price:
        raise ValueError(f"No price data returned for {ticker}")

    prev_close = info.get("previousClose") or price
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

    return {
        "price":      round(float(price), 4),
        "change_pct": round(float(change_pct), 2),
        "high_52w":   round(float(info.get("fiftyTwoWeekHigh") or price), 4),
        "low_52w":    round(float(info.get("fiftyTwoWeekLow")  or price), 4),
        "volume":     info.get("volume") or info.get("regularMarketVolume") or "N/A",
    }


def _fetch_sample(ticker: str) -> dict:
    """Load from sample_prices.csv or generate plausible mock data."""
    rows = _load_csv()
    for row in rows:
        if row.get("ticker", "").upper() == ticker.upper():
            return {
                "price":      float(row["price"]),
                "change_pct": float(row["change_pct"]),
                "high_52w":   float(row["high_52w"]),
                "low_52w":    float(row["low_52w"]),
                "volume":     row.get("volume", "N/A"),
            }

    # Generate plausible mock data so the AI can still respond
    base = random.uniform(10, 500)
    chg  = random.uniform(-5, 5)
    return {
        "price":      round(base, 2),
        "change_pct": round(chg, 2),
        "high_52w":   round(base * random.uniform(1.05, 1.60), 2),
        "low_52w":    round(base * random.uniform(0.40, 0.95), 2),
        "volume":     f"{random.randint(500_000, 50_000_000):,}",
        "_note":      "sample/mock data — install yfinance for live prices",
    }


def _load_csv() -> list[dict]:
    try:
        with open(_SAMPLE_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []
