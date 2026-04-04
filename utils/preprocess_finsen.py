"""
FinSen Pre-processor — run this ONCE before starting the web app.

What it does
------------
1. Reads the raw FinSen CSVs from data/finsen_data/
2. Computes TextBlob sentiment scores for every headline
3. Aggregates to a daily sentiment score per ticker
4. Saves the result to data_cache/finsen_daily_sentiment.parquet
5. Generates an interactive analysis chart → data_cache/finsen_analysis.html

Usage
-----
    python utils/preprocess_finsen.py

Optional flags:
    --tickers AAPL MSFT TSLA   Only process these tickers (faster for testing)
    --limit 200000              Only process the first N rows (faster for testing)

Output
------
    data_cache/finsen_daily_sentiment.parquet
        Columns: date | ticker | avg_score | article_count |
                 positive_count | negative_count | neutral_count

    data_cache/finsen_analysis.html
        Interactive Plotly chart with 4 panels:
          1. Top 30 tickers by article count
          2. Sentiment score distribution
          3. Monthly average sentiment for 6 major stocks
          4. Article volume over time
"""
import argparse
import logging
import os
import sys
import time

# Allow running from repo root OR from utils/ directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Project root = one level up from this file (utils/../)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANDIDATE_FILES = [
    "raw_partner_headlines.csv",
    "analyst_ratings_processed.csv",
    "raw_analyst_ratings.csv",
    #"US_Financial_News.csv",
]

_COLUMN_MAPS = {
    "raw_partner_headlines.csv":     ("date",          "stock",  "headline"),
    "analyst_ratings_processed.csv": ("date",          "ticker", "title"),
    "raw_analyst_ratings.csv":       ("date",          "stock",  "title"),
    "US_Financial_News.csv":         ("date",          "ticker", "headline"),
}

_CHART_TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA"]

OUTPUT_PARQUET = os.path.join(_PROJECT_ROOT, "data_cache", "finsen_daily_sentiment.parquet")
OUTPUT_CHART   = os.path.join(_PROJECT_ROOT, "data_cache", "finsen_analysis.html")


# ---------------------------------------------------------------------------
# Step 1 — Load raw CSV
# ---------------------------------------------------------------------------

def load_raw(data_dir: str, limit: int = 0) -> pd.DataFrame:
    """Load the first available CSV from *data_dir*."""
    for fname in _CANDIDATE_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue

        date_col, ticker_col, headline_col = _COLUMN_MAPS[fname]
        logger.info("Loading %s ...", path)
        t0 = time.time()
        raw = pd.read_csv(path, low_memory=False, nrows=limit if limit > 0 else None)
        logger.info("  → %d rows loaded in %.1fs", len(raw), time.time() - t0)

        df = pd.DataFrame()

        # Date
        for alt in (date_col, "published_utc", "publishedAt", "publish_date"):
            if alt in raw.columns:
                df["date"] = pd.to_datetime(raw[alt], errors="coerce", utc=False)
                if df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)
                break
        else:
            raise ValueError(f"No date column found in {fname}. Columns: {list(raw.columns)}")

        # Ticker
        if ticker_col in raw.columns:
            df["ticker"] = raw[ticker_col].astype(str).str.upper().str.strip()
        else:
            raise ValueError(f"No ticker column in {fname}.")

        # Headline
        if headline_col in raw.columns:
            df["headline"] = raw[headline_col].fillna("").astype(str)
        else:
            df["headline"] = ""

        df.dropna(subset=["date", "ticker"], inplace=True)
        df = df[df["ticker"].str.match(r"^[A-Z]{1,10}$")]  # filter junk rows
        logger.info("  → %d rows after cleaning, %d unique tickers",
                    len(df), df["ticker"].nunique())
        return df.reset_index(drop=True)

    raise FileNotFoundError(
        f"No supported FinSen CSV found in '{data_dir}'.\n"
        f"Expected one of: {_CANDIDATE_FILES}"
    )


# ---------------------------------------------------------------------------
# Step 2 — Compute sentiment
# ---------------------------------------------------------------------------

def compute_sentiment(df: pd.DataFrame, tickers: list = None) -> pd.DataFrame:
    """Add a 'sentiment_score' column using TextBlob (fast, no GPU needed)."""
    if tickers:
        df = df[df["ticker"].isin([t.upper() for t in tickers])].copy()
        logger.info("Filtered to %d tickers: %d rows remain.", len(tickers), len(df))

    logger.info("Computing TextBlob sentiment for %d headlines ...", len(df))
    try:
        from textblob import TextBlob
    except ImportError:
        raise ImportError("textblob is required: pip install textblob")

    t0 = time.time()
    # Process in chunks for progress feedback
    chunk_size = 50_000
    scores = []
    for i in range(0, len(df), chunk_size):
        chunk = df["headline"].iloc[i : i + chunk_size]
        scores.extend(
            round(TextBlob(str(h)).sentiment.polarity, 4) for h in chunk
        )
        logger.info(
            "  Sentiment: %d / %d rows (%.0fs elapsed)",
            min(i + chunk_size, len(df)),
            len(df),
            time.time() - t0,
        )

    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment"] = df["sentiment_score"].apply(
        lambda s: "positive" if s > 0.1 else "negative" if s < -0.1 else "neutral"
    )
    logger.info("Sentiment done in %.1fs.", time.time() - t0)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Aggregate to daily per-ticker
# ---------------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate headline-level rows to one row per (date, ticker).

    Output columns:
      date | ticker | avg_score | article_count |
      positive_count | negative_count | neutral_count
    """
    logger.info("Aggregating to daily sentiment ...")
    df["date_only"] = df["date"].dt.normalize()

    agg = (
        df.groupby(["date_only", "ticker"])
        .agg(
            avg_score=("sentiment_score", "mean"),
            article_count=("headline", "count"),
            positive_count=("sentiment", lambda x: (x == "positive").sum()),
            negative_count=("sentiment", lambda x: (x == "negative").sum()),
            neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
        )
        .reset_index()
        .rename(columns={"date_only": "date"})
    )
    agg["avg_score"] = agg["avg_score"].round(4)
    logger.info(
        "Aggregated: %d daily records, %d unique tickers, date range %s → %s",
        len(agg),
        agg["ticker"].nunique(),
        agg["date"].min().date(),
        agg["date"].max().date(),
    )
    return agg


# ---------------------------------------------------------------------------
# Step 4 — Save parquet
# ---------------------------------------------------------------------------

def save_parquet(agg: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    agg.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / 1_048_576
    logger.info("Saved parquet: %s (%.1f MB)", path, size_mb)


# ---------------------------------------------------------------------------
# Step 5 — Generate analysis chart
# ---------------------------------------------------------------------------

def generate_chart(agg: pd.DataFrame, path: str) -> None:
    """Build a 4-panel interactive Plotly chart and save as HTML."""
    logger.info("Generating analysis chart ...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Top 30 Tickers by Article Count",
            "Sentiment Score Distribution",
            f"Monthly Avg Sentiment — {', '.join(_CHART_TICKERS)}",
            "Total Articles per Month",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    # --- Panel 1: Top 30 tickers by article count ---
    top30 = (
        agg.groupby("ticker")["article_count"].sum()
        .sort_values(ascending=False)
        .head(30)
        .reset_index()
    )
    fig.add_trace(
        go.Bar(
            x=top30["ticker"],
            y=top30["article_count"],
            marker_color="#00d4aa",
            name="Article Count",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # --- Panel 2: Sentiment score distribution ---
    fig.add_trace(
        go.Histogram(
            x=agg["avg_score"],
            nbinsx=60,
            marker_color="#9b59b6",
            name="Sentiment Score",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#ffffff", opacity=0.4, row=1, col=2)

    # --- Panel 3: Monthly avg sentiment for selected tickers ---
    colors = ["#00d4aa", "#ffa500", "#ff4b4b", "#9b59b6", "#3498db", "#e74c3c"]
    agg["month"] = agg["date"].dt.to_period("M").dt.to_timestamp()

    for i, ticker in enumerate(_CHART_TICKERS):
        tkr_df = agg[agg["ticker"] == ticker]
        if tkr_df.empty:
            continue
        monthly = tkr_df.groupby("month")["avg_score"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["avg_score"],
                name=ticker,
                line=dict(color=colors[i % len(colors)], width=2),
                mode="lines",
            ),
            row=2, col=1,
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#ffffff", opacity=0.3, row=2, col=1)

    # --- Panel 4: Total articles per month ---
    monthly_vol = agg.groupby("month")["article_count"].sum().reset_index()
    fig.add_trace(
        go.Bar(
            x=monthly_vol["month"],
            y=monthly_vol["article_count"],
            marker_color="#3498db",
            name="Monthly Volume",
            showlegend=False,
        ),
        row=2, col=2,
    )

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text="FinSen Dataset Analysis",
            font=dict(size=20, color="#ffffff"),
        ),
        height=800,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=11),
        legend=dict(
            bgcolor="rgba(14,17,23,0.8)",
            bordercolor="#333",
            borderwidth=1,
        ),
    )
    for row in range(1, 3):
        for col in range(1, 3):
            fig.update_xaxes(gridcolor="#1a1a2e", linecolor="#333", row=row, col=col)
            fig.update_yaxes(gridcolor="#1a1a2e", linecolor="#333", row=row, col=col)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    logger.info("Chart saved: %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-process FinSen dataset for TradingPal-AI backtesting."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_PROJECT_ROOT, "data/finsen_data"),
        help="Directory containing the raw FinSen CSV files.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PARQUET,
        help="Output path for the parquet file.",
    )
    parser.add_argument(
        "--chart",
        default=OUTPUT_CHART,
        help="Output path for the HTML analysis chart.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Only process these tickers (e.g. --tickers AAPL MSFT TSLA).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows to read from CSV (0 = all). Useful for quick tests.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TradingPal-AI — FinSen Pre-processor")
    logger.info("=" * 60)

    t_start = time.time()

    # 1. Load
    df = load_raw(args.data_dir, limit=args.limit)

    # 2. Sentiment
    df = compute_sentiment(df, tickers=args.tickers)

    # 3. Aggregate
    agg = aggregate_daily(df)

    # 4. Save parquet
    save_parquet(agg, args.output)

    # 5. Chart
    generate_chart(agg, args.chart)

    logger.info("=" * 60)
    logger.info("Done in %.1f seconds.", time.time() - t_start)
    logger.info("Parquet : %s", args.output)
    logger.info("Chart   : %s  ← open in browser", args.chart)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
