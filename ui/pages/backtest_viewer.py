"""
Backtest Viewer Page — compare three strategies over a historical period.

Strategies:
  RSI + News (AI-Driven): composite signal from RSI + sentiment
  DCA (Time-Based):       buy fixed amount every week/month regardless of price
  Hybrid:                 DCA base buys, doubled on BUY signal, skipped on SELL signal
"""
import logging
from datetime import date
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtesting.accuracy_reporter import AccuracyReporter
from backtesting.backtest_engine import BacktestEngine
from backtesting.sentiment_cache import SentimentCache
import config

logger = logging.getLogger(__name__)

_METRIC_HELP = {
    "Total Return": (
        "Total profit or loss as a % of your starting capital. "
        "Example: +15% means $10,000 grew to $11,500."
    ),
    "Annualised Return": (
        "Projects the return as if the same rate continued for exactly one year. "
        "Useful for comparing strategies with different time periods."
    ),
    "Sharpe Ratio": (
        "Risk-adjusted return score. "
        ">1.0 is good, >2.0 is very good, <0 means worse than holding cash. "
        "Higher = more return per unit of risk taken."
    ),
    "Max Drawdown": (
        "Worst peak-to-bottom loss during the period. "
        "Example: −20% means at some point $10,000 dropped to $8,000. "
        "Closer to 0 is better."
    ),
    "Win Rate": (
        "Percentage of trades that ended in profit. "
        "For DCA/Hybrid this reflects the single summary period."
    ),
}


@st.cache_resource(show_spinner="Loading sentiment cache...")
def _load_sentiment_cache():
    return SentimentCache()


def render(manager_agent=None) -> None:
    """Render the Backtest Viewer page."""
    st.title("TradingPal-AI — Backtesting Engine")
    st.markdown(
        "Compare three investment strategies over any historical period. "
        "Every signal is generated using only information available at that point in time — "
        "no peeking into the future."
    )

    cache = _load_sentiment_cache()
    if not cache.available:
        st.warning(
            "**FinSen sentiment cache not found.** RSI+News and Hybrid will use "
            "technical signals only (sentiment score = 0).  \n"
            "Run `python utils/preprocess_finsen.py` to enable sentiment."
        )
    else:
        min_d, max_d = cache.get_date_range()
        st.success(
            f"FinSen cache loaded — {len(cache.get_available_tickers())} tickers, "
            f"{min_d.date()} → {max_d.date()}"
        )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    with st.form("backtest_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_choice = st.selectbox("Select Stock / Asset", config.STOCK_WATCHLIST, index=0)
            if ticker_choice == "Other (type manually)":
                ticker_manual = st.text_input(
                    "Enter ticker symbol",
                    placeholder="e.g. NVDA, GME, BTC-USD",
                )
                ticker = ticker_manual.upper().strip() if ticker_manual else ""
            else:
                ticker = ticker_choice

            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1_000.0,
                value=float(config.INITIAL_CAPITAL),
                step=1_000.0,
            )

        with col2:
            start_date = st.date_input("Start Date", value=date(config.BACKTEST_YEAR, 1, 1))
            risk_pct = st.slider(
                "Risk % per trade (RSI+News / Hybrid)",
                min_value=0.5,
                max_value=float(config.MAX_RISK_PERCENT),
                value=float(config.DEFAULT_RISK_PERCENT),
                step=0.5,
            )
            dca_frequency = st.selectbox(
                "DCA / Hybrid buy frequency", ["weekly", "monthly"], index=0
            )

        with col3:
            end_date = st.date_input("End Date", value=date(config.BACKTEST_YEAR, 12, 31))
            recurring_amount = st.number_input(
                "Recurring contribution ($)",
                min_value=0.0,
                value=0.0,
                step=50.0,
                help="Amount you add to your account each period (simulates regular deposits)",
            )
            recurring_period = st.selectbox(
                "Contribution period", ["weekly", "monthly"], index=1
            )

        extra_context = st.text_area(
            "Additional Context (optional)",
            placeholder=(
                "Tell the AI about your situation — e.g. 'I already hold 10 shares of AAPL', "
                "'I want to retire in 10 years', 'I am testing which strategy fits my $500/month savings'..."
            ),
            height=80,
        )
        run_btn = st.form_submit_button(
            "Run Backtest", type="primary", use_container_width=True
        )

    if not run_btn:
        st.info(
            "Select a stock, set your capital, and click **Run Backtest** to compare "
            "DCA, RSI+News, and Hybrid strategies side-by-side."
        )
        return

    if not ticker:
        st.error("Please enter a ticker symbol.")
        return

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # ------------------------------------------------------------------
    # Run all three strategies
    # ------------------------------------------------------------------
    with st.spinner(f"Running 3 strategy comparisons for {ticker} ({start_date} → {end_date})..."):
        engine = BacktestEngine(sentiment_cache=cache)
        result_rsi = engine.run_backtest(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=initial_capital,
            risk_percent=risk_pct,
            recurring_amount=recurring_amount,
            recurring_period=recurring_period,
        )
        result_dca = engine.run_dca_backtest(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=initial_capital,
            frequency=dca_frequency,
            recurring_amount=recurring_amount,
            recurring_period=recurring_period,
        )
        result_hybrid = engine.run_hybrid_backtest(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=initial_capital,
            risk_percent=risk_pct,
            frequency=dca_frequency,
            recurring_amount=recurring_amount,
            recurring_period=recurring_period,
        )

    if result_rsi["signals"].empty and result_dca["signals"].empty:
        st.error(
            f"No data available for **{ticker}** in the selected period. "
            "Check the ticker symbol and date range."
        )
        return

    # ------------------------------------------------------------------
    # Tabbed display
    # ------------------------------------------------------------------
    st.divider()
    tab_rsi, tab_dca, tab_hybrid, tab_cmp = st.tabs([
        "RSI + News (AI-Driven)",
        "DCA (Time-Based)",
        "Hybrid",
        "Strategy Comparison",
    ])

    with tab_rsi:
        st.markdown(
            "**RSI + News**: Buys when technical indicators show oversold conditions and "
            "news sentiment is positive. Sells when overbought and sentiment turns negative. "
            "Holds one position at a time — multiple BUY signals while in a position are skipped."
        )
        _render_metrics(result_rsi)
        if not result_rsi["signals"].empty:
            fig = _build_chart(
                result_rsi["signals"], result_rsi["equity_curve"],
                result_rsi["trades"], initial_capital, ticker,
            )
            st.plotly_chart(fig, use_container_width=True)
        _render_rsi_trade_log(result_rsi["trades"])

    with tab_dca:
        st.markdown(
            f"**DCA (Dollar-Cost Averaging)**: Invests a fixed amount every {dca_frequency} "
            "regardless of price or news. Simple, consistent, and removes emotion from investing."
        )
        _render_metrics(result_dca)
        if not result_dca["signals"].empty:
            fig = _build_chart(
                result_dca["signals"], result_dca["equity_curve"],
                result_dca["trades"], initial_capital, ticker,
                buy_log=result_dca.get("buy_log", []),
            )
            st.plotly_chart(fig, use_container_width=True)
        _render_buy_log(result_dca.get("buy_log", []), strategy="DCA")

    with tab_hybrid:
        st.markdown(
            f"**Hybrid**: Combines DCA consistency with signal intelligence. "
            f"Buys double the normal amount when AI signals BUY, skips when it signals SELL."
        )
        _render_metrics(result_hybrid)
        if not result_hybrid["signals"].empty:
            fig = _build_chart(
                result_hybrid["signals"], result_hybrid["equity_curve"],
                result_hybrid["trades"], initial_capital, ticker,
                buy_log=result_hybrid.get("buy_log", []),
            )
            st.plotly_chart(fig, use_container_width=True)
        _render_buy_log(result_hybrid.get("buy_log", []), strategy="Hybrid")

    with tab_cmp:
        st.subheader("Strategy Comparison")
        _render_comparison_table([result_rsi, result_dca, result_hybrid])
        st.subheader("Equity Curves — All Strategies")
        fig_cmp = _build_comparison_chart([result_rsi, result_dca, result_hybrid], ticker, initial_capital)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.divider()
    reporter = AccuracyReporter()
    export_strategy = st.selectbox(
        "Export strategy results",
        ["RSI+News", "DCA", "Hybrid"],
        key="export_strategy_select",
    )
    result_map = {"RSI+News": result_rsi, "DCA": result_dca, "Hybrid": result_hybrid}

    if st.button("Export Report (CSV)", use_container_width=True):
        csv_str = reporter.generate_report_no_accuracy(result_map[export_strategy])
        st.download_button(
            label="Download CSV",
            data=csv_str,
            file_name=f"tradingpal_backtest_{ticker}_{export_strategy}_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # AI Analysis of Backtest Results (context-aware)
    # ------------------------------------------------------------------
    if manager_agent is not None and extra_context and extra_context.strip():
        st.divider()
        st.subheader("AI Analysis of Your Backtest")
        with st.spinner("Asking AI to interpret results for your situation..."):
            best_result = max(
                [result_rsi, result_dca, result_hybrid],
                key=lambda r: r["metrics"]["total_return_pct"],
            )
            strategy_names = {id(result_rsi): "RSI+News", id(result_dca): "DCA", id(result_hybrid): "Hybrid"}
            best_strategy_name = strategy_names[id(best_result)]
            summary_context = (
                f"Backtest results for {ticker} ({start_date} to {end_date}), "
                f"initial capital ${initial_capital:,.0f}. "
                f"RSI+News: {result_rsi['metrics']['total_return_pct']:+.2f}%, "
                f"DCA: {result_dca['metrics']['total_return_pct']:+.2f}%, "
                f"Hybrid: {result_hybrid['metrics']['total_return_pct']:+.2f}%. "
                f"Best strategy: {best_strategy_name}. "
                f"User context: {extra_context.strip()}"
            )
            advice = manager_agent.generate_context_advice(
                ticker=ticker,
                investment_amount=initial_capital,
                risk_tolerance="Medium",
                horizon="Long-term (>3 months)",
                technical_signal={"signal": "NEUTRAL", "rsi": 50.0, "score": 0.0},
                sentiment_signal={"score": 0.0},
                extra_context=summary_context,
                recurring_amount=recurring_amount,
                recurring_period=recurring_period,
            )
        if advice:
            st.markdown(
                f'<div style="background:#1a2a1a; border-left:5px solid #00d4aa; '
                f'padding:14px 18px; border-radius:0 10px 10px 0; margin-bottom:12px;">'
                f'<span style="color:#00d4aa; font-weight:bold;">AI Insight for Your Situation</span><br>'
                f'<span style="color:#e0e0e0;">{advice.get("personal_advice", "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**Things to watch out for**")
                for r in advice.get("context_risks", []):
                    st.markdown(f"- {r}")
            with bc2:
                st.markdown("**Suggested next steps**")
                for i, a in enumerate(advice.get("context_actions", []), 1):
                    st.markdown(f"{i}. {a}")
        else:
            st.info("AI analysis unavailable — Gemini API may not be configured, or no context was provided.")


# ---------------------------------------------------------------------------
# Metrics renderer
# ---------------------------------------------------------------------------

def _render_metrics(result: Dict) -> None:
    """Render the 5-column performance metrics block with info tooltips."""
    metrics = result["metrics"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Total Return",
        f"{metrics['total_return_pct']:+.2f}%",
        delta_color="normal",
        help=_METRIC_HELP["Total Return"],
    )
    m2.metric(
        "Annualised Return",
        f"{metrics['annualised_return_pct']:+.2f}%",
        help=_METRIC_HELP["Annualised Return"],
    )
    m3.metric(
        "Sharpe Ratio",
        f"{metrics['sharpe_ratio']:.3f}",
        help=_METRIC_HELP["Sharpe Ratio"],
    )
    m4.metric(
        "Max Drawdown",
        f"{metrics['max_drawdown_pct']:.2f}%",
        delta_color="inverse",
        help=_METRIC_HELP["Max Drawdown"],
    )
    m5.metric(
        "Win Rate",
        f"{metrics['win_rate_pct']:.1f}%",
        f"{metrics['profitable_trades']}/{metrics['total_trades']} trades",
        help=_METRIC_HELP["Win Rate"],
    )


# ---------------------------------------------------------------------------
# Trade log renderers
# ---------------------------------------------------------------------------

def _render_rsi_trade_log(trades: list) -> None:
    """
    Show RSI+News trade log.
    Explains that the chart may show more signal markers than executed trades
    (multiple BUY signals while already holding a position are skipped).
    """
    if not trades:
        st.caption("No completed trades in this period.")
        return

    with st.expander(
        f"Executed Trades ({len(trades)} completed position(s))",
        expanded=False,
    ):
        st.caption(
            "Each row is a complete trade: one BUY (open) + one SELL (close). "
            "The price chart may show more signal markers — those are all the days the model "
            "wanted to trade, but only the first BUY after closing a position is executed "
            "(one position held at a time)."
        )
        trades_df = pd.DataFrame(trades)
        for col in ["entry_date", "exit_date"]:
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime("%Y-%m-%d")
        st.dataframe(trades_df, use_container_width=True)


def _render_buy_log(buy_log: list, strategy: str = "DCA") -> None:
    """
    Show DCA / Hybrid buy log. Each row is one scheduled buy event.
    """
    if not buy_log:
        st.caption("No buy events recorded.")
        return

    executed = [b for b in buy_log if b.get("shares_bought", 0) > 0]
    skipped = len(buy_log) - len(executed)

    label = f"{strategy} Buy Log ({len(executed)} buys"
    if skipped > 0:
        label += f", {skipped} skipped"
    label += ")"

    with st.expander(label, expanded=False):
        if strategy == "Hybrid":
            st.caption(
                "BUY ×2 = signal was BUY (doubled investment) | "
                "BUY (base) = NEUTRAL signal | "
                "SKIPPED = signal was SELL (no purchase that period)"
            )
        df = pd.DataFrame(buy_log)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _render_comparison_table(results: List[Dict]) -> None:
    """Build a side-by-side strategy comparison table and highlight the winner."""
    labels = ["RSI + News", "DCA", "Hybrid"]
    rows = []
    for r, label in zip(results, labels):
        m = r["metrics"]
        rows.append({
            "Strategy": label,
            "Total Return": f"{m['total_return_pct']:+.2f}%",
            "Ann. Return": f"{m['annualised_return_pct']:+.2f}%",
            "Sharpe": f"{m['sharpe_ratio']:.3f}",
            "Max Drawdown": f"{m['max_drawdown_pct']:.2f}%",
            "Win Rate": f"{m['win_rate_pct']:.1f}%",
            "Final Value": f"${m['final_value']:,.2f}",
            "# Trades": m["total_trades"],
        })

    df = pd.DataFrame(rows).set_index("Strategy")
    st.dataframe(df, use_container_width=True)

    best_idx = max(range(3), key=lambda i: results[i]["metrics"]["total_return_pct"])
    best_label = labels[best_idx]
    best_return = results[best_idx]["metrics"]["total_return_pct"]
    worst_return = min(r["metrics"]["total_return_pct"] for r in results)
    diff = best_return - worst_return

    if best_return > 0:
        st.success(
            f"**Best strategy: {best_label}** — {best_return:+.2f}% total return "
            f"({diff:.2f}pp ahead of the weakest strategy)"
        )
    else:
        st.warning(
            f"All strategies lost value in this period. "
            f"**{best_label}** lost the least ({best_return:+.2f}%)."
        )


def _build_comparison_chart(
    results: List[Dict],
    ticker: str,
    initial_capital: float,
) -> go.Figure:
    """Build an equity curve comparison chart overlaying all 3 strategies."""
    colors = {"RSI+News": "#00d4aa", "DCA": "#ffa500", "Hybrid": "#9b59b6"}
    labels = ["RSI+News", "DCA", "Hybrid"]

    fig = go.Figure()

    show_invested = False
    for result, label in zip(results, labels):
        ec = result["equity_curve"]
        if not ec:
            continue
        invested_vals = [e.get("total_invested", initial_capital) for e in ec]
        if max(invested_vals) > initial_capital:
            show_invested = True
        fig.add_trace(go.Scatter(
            x=[e["date"] for e in ec],
            y=[e["value"] for e in ec],
            name=label,
            line=dict(color=colors[label], width=2),
            hovertemplate=f"<b>{label}</b><br>$%{{y:,.2f}}<br>%{{x}}<extra></extra>",
        ))

    # Show total invested once (using the last result that has it)
    if show_invested:
        last_ec = next((r["equity_curve"] for r in reversed(results) if r["equity_curve"]), None)
        if last_ec:
            fig.add_trace(go.Scatter(
                x=[e["date"] for e in last_ec],
                y=[e.get("total_invested", initial_capital) for e in last_ec],
                name="Total Invested",
                line=dict(color="#ffffff", width=1.5, dash="dot"),
                opacity=0.6,
                hovertemplate="<b>Total Invested</b><br>$%{y:,.2f}<br>%{x}<extra></extra>",
            ))

    fig.add_hline(
        y=initial_capital, line_dash="dash", line_color="#888888", opacity=0.5,
        annotation_text="Starting capital", annotation_position="bottom right",
    )

    fig.update_layout(
        height=400,
        title=f"{ticker} — Portfolio Value by Strategy",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=20),
        xaxis=dict(
            gridcolor="#1a1a2e", showline=True, linecolor="#333",
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#1a1a2e", activecolor="#00d4aa", font=dict(color="#e0e0e0"),
            ),
        ),
        yaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333", title_text="Value ($)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Single-strategy chart builder
# ---------------------------------------------------------------------------

def _build_chart(
    signals_df: pd.DataFrame,
    equity_curve: list,
    trades: list,
    initial_capital: float,
    ticker: str,
    buy_log: list = None,
) -> go.Figure:
    """
    Build a 3-panel synchronized Plotly figure.
    - For RSI+News: marks only EXECUTED trades on the chart (not all signals).
    - For DCA: marks scheduled buy dates from buy_log.
    - For Hybrid: marks both buy_log events and composite signals.
    """
    dates = signals_df.index.tolist()
    buy_log = buy_log or []

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(f"{ticker} Price", "Portfolio Value", "Drawdown"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=signals_df["open"], high=signals_df["high"],
            low=signals_df["low"], close=signals_df["close"],
            name="Price",
            increasing_line_color="#00d4aa", decreasing_line_color="#ff4b4b",
        ),
        row=1, col=1,
    )

    # SMA lines
    if "sma_short" in signals_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=signals_df["sma_short"], name=f"SMA {config.SMA_SHORT}",
            line=dict(color="#ffa500", width=1.5), opacity=0.8,
        ), row=1, col=1)
    if "sma_long" in signals_df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=signals_df["sma_long"], name=f"SMA {config.SMA_LONG}",
            line=dict(color="#9b59b6", width=1.5, dash="dot"), opacity=0.8,
        ), row=1, col=1)

    # Markers: RSI+News — executed trades only
    if "composite_signal" in signals_df.columns and not buy_log:
        executed_buy_dates = {t["entry_date"] for t in trades if t.get("entry_date") is not None}
        executed_sell_dates = {t["exit_date"] for t in trades if t.get("exit_date") is not None}

        buy_exec = signals_df.index.isin(executed_buy_dates)
        if buy_exec.any():
            fig.add_trace(go.Scatter(
                x=signals_df.index[buy_exec],
                y=signals_df["low"][buy_exec] * 0.99,
                mode="markers", name="BUY (executed)",
                marker=dict(symbol="triangle-up", color="#00d4aa", size=13,
                            line=dict(color="#ffffff", width=1)),
                hovertemplate="<b>BUY executed</b><br>%{x}<br>$%{text}<extra></extra>",
                text=[f"{p:,.2f}" for p in signals_df["close"][buy_exec]],
            ), row=1, col=1)

        sell_exec = signals_df.index.isin(executed_sell_dates)
        if sell_exec.any():
            fig.add_trace(go.Scatter(
                x=signals_df.index[sell_exec],
                y=signals_df["high"][sell_exec] * 1.01,
                mode="markers", name="SELL (executed)",
                marker=dict(symbol="triangle-down", color="#ff4b4b", size=13,
                            line=dict(color="#ffffff", width=1)),
                hovertemplate="<b>SELL executed</b><br>%{x}<br>$%{text}<extra></extra>",
                text=[f"{p:,.2f}" for p in signals_df["close"][sell_exec]],
            ), row=1, col=1)

        # Signal-only markers (fired but not executed — smaller, hollow)
        all_buy = signals_df["composite_signal"] == "BUY"
        signal_only_buy = all_buy & ~buy_exec
        if signal_only_buy.any():
            fig.add_trace(go.Scatter(
                x=signals_df.index[signal_only_buy],
                y=signals_df["low"][signal_only_buy] * 0.985,
                mode="markers", name="BUY signal (skipped)",
                marker=dict(symbol="triangle-up-open", color="#00d4aa", size=8, opacity=0.5),
                hovertemplate="<b>BUY signal (already in position)</b><br>%{x}<extra></extra>",
            ), row=1, col=1)

    # Markers: DCA / Hybrid — buy_log events
    if buy_log:
        executed_buys = [b for b in buy_log if b.get("shares_bought", 0) > 0]
        skipped_buys = [b for b in buy_log if b.get("shares_bought", 0) == 0]

        if executed_buys:
            buy_dates = [b["date"] for b in executed_buys]
            buy_prices = [b["price_paid"] for b in executed_buys]
            buy_amounts = [b["amount_spent"] for b in executed_buys]
            buy_idx = signals_df.index.isin(buy_dates)
            if buy_idx.any():
                fig.add_trace(go.Scatter(
                    x=signals_df.index[buy_idx],
                    y=signals_df["low"][buy_idx] * 0.99,
                    mode="markers", name="Scheduled Buy",
                    marker=dict(symbol="diamond", color="#00d4aa", size=10,
                                line=dict(color="#ffffff", width=1)),
                    hovertemplate=(
                        "<b>BUY</b><br>%{x}<br>"
                        "Price: $%{text[0]}<br>Amount: $%{text[1]}<extra></extra>"
                    ),
                    text=[f"{p:.2f},{a:.0f}" for p, a in zip(buy_prices, buy_amounts)],
                ), row=1, col=1)

        if skipped_buys:
            skip_dates = [b["date"] for b in skipped_buys]
            skip_idx = signals_df.index.isin(skip_dates)
            if skip_idx.any():
                fig.add_trace(go.Scatter(
                    x=signals_df.index[skip_idx],
                    y=signals_df["high"][skip_idx] * 1.01,
                    mode="markers", name="Buy Skipped (SELL signal)",
                    marker=dict(symbol="x", color="#ff4b4b", size=9, opacity=0.7),
                    hovertemplate="<b>Buy skipped (SELL signal)</b><br>%{x}<extra></extra>",
                ), row=1, col=1)

    # Equity curve
    if equity_curve:
        eq_dates = [e["date"] for e in equity_curve]
        eq_values = [e["value"] for e in equity_curve]
        invested_values = [e.get("total_invested", initial_capital) for e in equity_curve]
        fig.add_trace(go.Scatter(
            x=eq_dates, y=eq_values, name="Portfolio Value",
            fill="tozeroy", fillcolor="rgba(0,212,170,0.15)",
            line=dict(color="#00d4aa", width=2),
            hovertemplate="<b>Portfolio: $%{y:,.2f}</b><br>%{x}<extra></extra>",
        ), row=2, col=1)
        # Total Invested line — shows how much cash was put in (initial + recurring)
        if max(invested_values) > initial_capital:
            fig.add_trace(go.Scatter(
                x=eq_dates, y=invested_values, name="Total Invested",
                line=dict(color="#ffa500", width=1.5, dash="dot"),
                hovertemplate="<b>Invested: $%{y:,.2f}</b><br>%{x}<extra></extra>",
            ), row=2, col=1)
        fig.add_hline(y=initial_capital, line_dash="dash", line_color="#888888", opacity=0.5, row=2, col=1)

        # Drawdown
        import numpy as np
        eq_series = pd.Series(eq_values, index=eq_dates)
        roll_max = eq_series.cummax()
        drawdown = (eq_series - roll_max) / roll_max * 100
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values, name="Drawdown",
            fill="tozeroy", fillcolor="rgba(255,75,75,0.25)",
            line=dict(color="#ff4b4b", width=1.5),
            hovertemplate="<b>%{y:.2f}%</b><br>%{x}<extra></extra>",
        ), row=3, col=1)

    fig.update_layout(
        height=800, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(14,17,23,0.8)"),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=20),
    )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="1Y"),
            ],
            bgcolor="#1a1a2e", activecolor="#00d4aa", font=dict(color="#e0e0e0"),
        ),
        row=1, col=1,
    )
    for row in range(1, 4):
        fig.update_xaxes(gridcolor="#1a1a2e", showline=True, linecolor="#333", row=row, col=1)
        fig.update_yaxes(gridcolor="#1a1a2e", showline=True, linecolor="#333", row=row, col=1)

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

    return fig
