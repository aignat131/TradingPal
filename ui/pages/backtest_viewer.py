"""
Backtest Viewer Page — full-year backtesting with professional stock charts.

Chart layout (3 synchronized panels, Plotly):
  Panel 1 (60%): Candlestick price chart + SMA lines + BUY/SELL signal markers
  Panel 2 (20%): Portfolio equity curve (area chart)
  Panel 3 (20%): Drawdown chart (red filled area)

All 3 panels share the same X-axis (synchronized zoom and pan).
Range selector buttons: 1M, 3M, 6M, YTD, 1Y.
"""
import logging
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtesting.accuracy_reporter import AccuracyReporter
from backtesting.backtest_engine import BacktestEngine
from backtesting.sentiment_cache import SentimentCache
import config

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading sentiment cache...")
def _load_sentiment_cache():
    return SentimentCache()


def render() -> None:
    """Render the Backtest Viewer page."""
    st.title("TradingPal-AI — Backtesting Engine")
    st.markdown(
        "Simulate how TradingPal-AI signals would have performed over a historical period. "
        "The engine replays each trading day using only information available at that point "
        "(no look-ahead bias)."
    )

    # Sentiment cache status banner
    cache = _load_sentiment_cache()
    if not cache.available:
        st.warning(
            "**FinSen sentiment cache not found.** Backtests will use technical signals only "
            "(sentiment score = 0).  \n"
            "To enable full sentiment-weighted backtesting, run:  \n"
            "```\npython utils/preprocess_finsen.py\n```"
        )
    else:
        min_d, max_d = cache.get_date_range()
        st.success(
            f"FinSen sentiment cache loaded — "
            f"{len(cache.get_available_tickers())} tickers, "
            f"{min_d.date()} → {max_d.date()}"
        )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    with st.form("backtest_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Ticker", value="AAPL").upper().strip()
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1_000.0,
                value=float(config.INITIAL_CAPITAL),
                step=1_000.0,
            )
        with col2:
            start_date = st.date_input(
                "Start Date", value=date(config.BACKTEST_YEAR, 1, 1)
            )
            risk_pct = st.slider(
                "Risk % per trade",
                min_value=0.5,
                max_value=float(config.MAX_RISK_PERCENT),
                value=float(config.DEFAULT_RISK_PERCENT),
                step=0.5,
            )
        with col3:
            end_date = st.date_input(
                "End Date", value=date(config.BACKTEST_YEAR, 12, 31)
            )
            lookahead = st.slider(
                "Accuracy lookahead (days)",
                min_value=1,
                max_value=20,
                value=5,
            )

        run_btn = st.form_submit_button(
            "Run Backtest", type="primary", use_container_width=True
        )

    if not run_btn:
        st.info("Configure the parameters above and click **Run Backtest**.")
        return

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------
    with st.spinner(f"Running backtest for {ticker} ({start_date} → {end_date})..."):
        engine = BacktestEngine(sentiment_cache=cache)
        result = engine.run_backtest(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=initial_capital,
            risk_percent=risk_pct,
        )

    if result["signals"].empty:
        st.error(
            f"No data available for **{ticker}** in the selected period. "
            "Check the ticker symbol and date range."
        )
        return

    signals_df = result["signals"]
    equity_curve = result["equity_curve"]
    trades = result["trades"]
    metrics = result["metrics"]

    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Total Return",
        f"{metrics['total_return_pct']:+.2f}%",
        delta_color="normal",
    )
    m2.metric("Annualised Return", f"{metrics['annualised_return_pct']:+.2f}%")
    m3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
    m4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%", delta_color="inverse")
    m5.metric(
        "Win Rate",
        f"{metrics['win_rate_pct']:.1f}%",
        f"{metrics['profitable_trades']}/{metrics['total_trades']} trades",
    )

    # ------------------------------------------------------------------
    # Main Chart (3 panels)
    # ------------------------------------------------------------------
    st.subheader("Full-Year Chart")
    fig = _build_chart(signals_df, equity_curve, initial_capital, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Accuracy Analysis
    # ------------------------------------------------------------------
    reporter = AccuracyReporter()
    augmented = reporter.compare_with_reality(signals_df, lookahead_days=lookahead)
    accuracy = reporter.calculate_accuracy_metrics(augmented)

    st.subheader(f"Signal Accuracy ({lookahead}-day lookahead)")
    a1, a2, a3 = st.columns(3)
    a1.metric("Overall Accuracy", f"{accuracy['overall_accuracy_pct']:.1f}%")
    a2.metric(
        "BUY Precision",
        f"{accuracy['buy_precision_pct']:.1f}%",
        f"{accuracy['buy_signals']} signals",
    )
    a3.metric(
        "SELL Precision",
        f"{accuracy['sell_precision_pct']:.1f}%",
        f"{accuracy['sell_signals']} signals",
    )

    # ------------------------------------------------------------------
    # Trade Log
    # ------------------------------------------------------------------
    if trades:
        with st.expander(f"Trade Log ({len(trades)} trades)", expanded=False):
            trades_df = pd.DataFrame(trades)
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(trades_df, use_container_width=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.divider()
    if st.button("Export Report (CSV)", use_container_width=True):
        csv_str, _ = reporter.generate_report(result, lookahead_days=lookahead)
        st.download_button(
            label="Download CSV",
            data=csv_str,
            file_name=f"tradingpal_backtest_{ticker}_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def _build_chart(
    signals_df: pd.DataFrame,
    equity_curve: list,
    initial_capital: float,
    ticker: str,
) -> go.Figure:
    """
    Build a 3-panel synchronized Plotly figure:
      Row 1 (60%): Candlestick + SMA lines + BUY/SELL markers
      Row 2 (20%): Equity curve (filled area)
      Row 3 (20%): Drawdown (red filled area)
    """
    dates = signals_df.index.tolist()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(
            f"{ticker} Price + Signals",
            "Portfolio Equity Curve",
            "Drawdown",
        ),
    )

    # --- Row 1: Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=signals_df["open"],
            high=signals_df["high"],
            low=signals_df["low"],
            close=signals_df["close"],
            name="Price",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4b4b",
        ),
        row=1, col=1,
    )

    # SMA lines
    if "sma_short" in signals_df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=signals_df["sma_short"],
                name=f"SMA {config.SMA_SHORT}",
                line=dict(color="#ffa500", width=1.5, dash="solid"),
                opacity=0.8,
            ),
            row=1, col=1,
        )
    if "sma_long" in signals_df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=signals_df["sma_long"],
                name=f"SMA {config.SMA_LONG}",
                line=dict(color="#9b59b6", width=1.5, dash="dot"),
                opacity=0.8,
            ),
            row=1, col=1,
        )

    # BUY signals — green triangles on price chart
    buy_mask = signals_df["composite_signal"] == "BUY"
    if buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals_df.index[buy_mask],
                y=signals_df["low"][buy_mask] * 0.99,
                mode="markers",
                name="BUY Signal",
                marker=dict(
                    symbol="triangle-up",
                    color="#00d4aa",
                    size=12,
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="<b>BUY</b><br>%{x}<br>Price: $%{text}<extra></extra>",
                text=[f"{p:,.2f}" for p in signals_df["close"][buy_mask]],
            ),
            row=1, col=1,
        )

    # SELL signals — red triangles on price chart
    sell_mask = signals_df["composite_signal"] == "SELL"
    if sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals_df.index[sell_mask],
                y=signals_df["high"][sell_mask] * 1.01,
                mode="markers",
                name="SELL Signal",
                marker=dict(
                    symbol="triangle-down",
                    color="#ff4b4b",
                    size=12,
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="<b>SELL</b><br>%{x}<br>Price: $%{text}<extra></extra>",
                text=[f"{p:,.2f}" for p in signals_df["close"][sell_mask]],
            ),
            row=1, col=1,
        )

    # --- Row 2: Equity Curve ---
    if equity_curve:
        eq_dates = [e["date"] for e in equity_curve]
        eq_values = [e["value"] for e in equity_curve]
        above_colors = ["#00d4aa" if v >= initial_capital else "#ff4b4b" for v in eq_values]

        fig.add_trace(
            go.Scatter(
                x=eq_dates,
                y=eq_values,
                name="Portfolio Value",
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.15)",
                line=dict(color="#00d4aa", width=2),
                hovertemplate="<b>$%{y:,.2f}</b><br>%{x}<extra></extra>",
            ),
            row=2, col=1,
        )
        # Initial capital reference line
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="#888888",
            opacity=0.5,
            row=2, col=1,
        )

    # --- Row 3: Drawdown ---
    if equity_curve:
        import numpy as np

        eq_series = pd.Series(eq_values, index=eq_dates)
        roll_max = eq_series.cummax()
        drawdown = (eq_series - roll_max) / roll_max * 100

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                fill="tozeroy",
                fillcolor="rgba(255,75,75,0.25)",
                line=dict(color="#ff4b4b", width=1.5),
                hovertemplate="<b>%{y:.2f}%</b><br>%{x}<extra></extra>",
            ),
            row=3, col=1,
        )

    # --- Layout ---
    fig.update_layout(
        height=800,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(14,17,23,0.8)",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=20),
    )

    # Range selector on top X-axis
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="1Y"),
            ],
            bgcolor="#1a1a2e",
            activecolor="#00d4aa",
            font=dict(color="#e0e0e0"),
        ),
        row=1, col=1,
    )

    # Style axes
    for row in range(1, 4):
        fig.update_xaxes(
            gridcolor="#1a1a2e",
            showline=True,
            linecolor="#333",
            row=row, col=1,
        )
        fig.update_yaxes(
            gridcolor="#1a1a2e",
            showline=True,
            linecolor="#333",
            row=row, col=1,
        )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

    return fig
