"""
Math Logic — Rule-Based Advisor

UI layer for the CLIPS expert system.  All inference logic lives in:
    clips/engine.py          — CaptureRouter, run_clips_period, run_simulation
    clips/rules/             — Baza_de_reguli.clp

This file contains only Streamlit UI code: forms, charts, result rendering.

Entry point: render()
"""
import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from clips.engine import run_clips_period as _run_clips_period_fn
from clips.engine import run_simulation as _run_simulation_fn
from ui.components import display_risk_meter, display_signal_gauge, display_verdict_badge

logger = logging.getLogger(__name__)

# Watchlist without the manual-entry sentinel
_WATCHLIST = [t for t in config.STOCK_WATCHLIST if t != "Other (type manually)"]

# Strategy labels → exact CLIPS string values used in defrules
_STRATEGY_MAP = {
    "DCA Fix": "DCA fix",
    "Hibrid": "hibrid",
    "RSI": "RSI",
}

# (frequency, timeframe) → number of periods
_PERIOD_MAP: dict[tuple[str, str], int] = {
    ("Daily",   "1 Month"):   21,
    ("Daily",   "3 Months"):  63,
    ("Daily",   "6 Months"): 126,
    ("Daily",   "1 Year"):   252,
    ("Daily",   "2 Years"):  504,
    ("Daily",   "3 Years"):  756,
    ("Weekly",  "1 Month"):    4,
    ("Weekly",  "3 Months"):  13,
    ("Weekly",  "6 Months"):  26,
    ("Weekly",  "1 Year"):    52,
    ("Weekly",  "2 Years"):  104,
    ("Weekly",  "3 Years"):  156,
    ("Monthly", "1 Month"):    1,
    ("Monthly", "3 Months"):   3,
    ("Monthly", "6 Months"):   6,
    ("Monthly", "1 Year"):    12,
    ("Monthly", "2 Years"):   24,
    ("Monthly", "3 Years"):   36,
}

_TIMEFRAMES = ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "3 Years"]
_FREQUENCIES = ["Daily", "Weekly", "Monthly"]

# yfinance interval per frequency
_YF_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

# ──────────────────────────────────────────────────────────────────────────────
# Data fetching (same pattern as trade_advisor.py)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_data(ticker: str) -> Optional[dict]:
    """
    Fetch current price, RSI(14), and MA200 for a ticker via yfinance.
    Returns dict with keys: price, rsi, ma200  — or None on failure.
    """
    try:
        import yfinance as yf
        from analysis.technical_indicators import TechnicalAnalyzer

        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        close = df["Close"].dropna()
        if len(close) < 14:
            return None

        ma200 = (
            float(close.rolling(200).mean().iloc[-1])
            if len(close) >= 200
            else float(close.mean())
        )

        analyzer = TechnicalAnalyzer()
        rsi = analyzer.calculate_rsi(close.tolist())
        price = float(close.iloc[-1])

        return {"price": round(price, 4), "rsi": round(rsi, 2), "ma200": round(ma200, 4)}

    except Exception as exc:
        logger.warning("Market data fetch failed for %s: %s", ticker, exc)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _forecast_prices(ticker: str, n_periods: int, frequency: str) -> list[float] | None:
    """
    Project future prices for n_periods using the asset's own historical
    average return at the given frequency (Daily / Weekly / Monthly).

    Method: compound the mean periodic return from the last 5 years of data.
    Returns a list of n_periods floats (period 1 … n), or None on failure.
    This is an estimate based on historical averages — not a guarantee.
    """
    try:
        import yfinance as yf
        import numpy as np

        interval = _YF_INTERVAL[frequency]
        df = yf.download(ticker, period="5y", interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        closes = df["Close"].dropna()
        if len(closes) < 10:
            return None

        # Average periodic return and volatility
        returns = closes.pct_change().dropna()
        avg_ret = float(returns.mean())

        current = float(closes.iloc[-1])
        prices = [
            round(current * (1 + avg_ret) ** t, 4)
            for t in range(1, n_periods + 1)
        ]
        return prices

    except Exception as exc:
        logger.warning("Price forecast failed for %s: %s", ticker, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CLIPS engine — delegated to clips/engine.py
# ──────────────────────────────────────────────────────────────────────────────

def _run_clips_period(**kwargs) -> dict:
    """Thin wrapper — see clips.engine.run_clips_period for full docs."""
    return _run_clips_period_fn(**kwargs)


def _run_simulation(
    objective: str,
    strategy_clips: str,
    target_sum: float,
    n_periods: int,
    budget: float,
    ticker: str,
    rsi: float,
    price: float,
    ma200: float,
    qty: float,
    avg_price: float,
    frequency: str = "Monthly",
    predicted_prices: list | None = None,
) -> list[dict]:
    """Thin wrapper — see clips.engine.run_simulation for full docs."""
    def _on_error(period, err):
        period_label = {"Daily": "Day", "Weekly": "Week", "Monthly": "Month"}[frequency]
        st.error(f"CLIPS error on {period_label.lower()} {period}: {err}")

    return _run_simulation_fn(
        objective=objective,
        strategy_clips=strategy_clips,
        target_sum=target_sum,
        n_periods=n_periods,
        budget=budget,
        ticker=ticker,
        rsi=rsi,
        price=price,
        ma200=ma200,
        qty=qty,
        avg_price=avg_price,
        frequency=frequency,
        predicted_prices=predicted_prices,
        on_error=_on_error,
    )


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rsi_zone_label(rsi: float) -> str:
    if rsi < 20:
        return "Supravândut Extrem"
    elif rsi < 40:
        return "Supravândut"
    elif rsi < 60:
        return "Neutru"
    elif rsi < 75:
        return "Supracumpărat"
    else:
        return "Supracumpărat Extrem"


def _rsi_zone_badge(rsi: float) -> None:
    zone = _rsi_zone_label(rsi)
    color = (
        "#00d4aa" if rsi < 40
        else "#ffa500" if rsi < 60
        else "#ff4b4b"
    )
    st.markdown(
        f'<span style="background:{color}22;border:1.5px solid {color};'
        f'border-radius:8px;padding:4px 14px;color:{color};'
        f'font-weight:700;font-size:0.9em;">'
        f'RSI {rsi:.1f} — {zone}</span>',
        unsafe_allow_html=True,
    )


def _render_rsi_gauge(rsi: float) -> None:
    """Reuse display_signal_gauge: map RSI to [-1,1] score."""
    score = (50.0 - rsi) / 50.0
    score = max(-1.0, min(1.0, score))
    if rsi < 40:
        signal = "BUY"
    elif rsi > 60:
        signal = "SELL"
    else:
        signal = "HOLD"
    display_signal_gauge(signal=signal, confidence=abs(score), score=score)


def _render_rsi_gauge_0_100(rsi: float) -> None:
    """Plotly gauge 0–100 with green/yellow/red zones for RSI."""
    zone = _rsi_zone_label(rsi)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi,
        number={"font": {"color": "#f0f0f0", "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#888", "tickfont": {"color": "#888"}},
            "bar": {"color": "#ffffff", "thickness": 0.25},
            "bgcolor": "#1a1a1a",
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 30],  "color": "rgba(0,212,170,0.25)"},
                {"range": [30, 70], "color": "rgba(255,165,0,0.20)"},
                {"range": [70, 100],"color": "rgba(255,75,75,0.25)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.8,
                "value": rsi,
            },
        },
        title={"text": f"RSI — {zone}", "font": {"color": "#aaa", "size": 13}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_strategy_explanation(
    strategy_lbl: str, rsi: float, amount: float,
    budget: float, n_periods: int, objective: str
) -> str:
    """Returns an HTML string explaining the strategy decision in plain Romanian."""
    base_per_period = budget / n_periods if n_periods > 0 else budget

    if objective == "Invest":
        if strategy_lbl == "DCA Fix":
            return (
                f"Ai ales <strong>DCA Fix</strong>. Bugetul tău de <strong>${budget:,.2f}</strong> "
                f"a fost împărțit în mod egal la cele <strong>{n_periods}</strong> perioade. "
                f"Nu ținem cont de fluctuațiile pieței pentru această strategie. "
                f"Suma standard per perioadă: <strong>${base_per_period:,.2f}</strong>."
            )
        elif strategy_lbl == "Hibrid":
            if rsi < 40:
                factor_txt = "crescut agresiv suma cu 50%"
                reason = f"RSI-ul indică o piață favorabilă (subevaluată la {rsi:.1f})"
                tip = " Profităm de prețul mic de azi!"
            elif rsi > 60:
                factor_txt = "redus suma cu 50%"
                reason = f"piața este supraevaluată (RSI: {rsi:.1f})"
                tip = " Protejăm capitalul în fața prețului ridicat."
            else:
                return (
                    f"Ai ales strategia <strong>Hibrid</strong>. Piața este neutră (RSI: {rsi:.1f}). "
                    f"Investim suma standard de <strong>${amount:,.2f}</strong> fără ajustări."
                )
            return (
                f"Ai ales strategia <strong>Hibrid</strong>. Suma standard per perioadă ar fi fost "
                f"<strong>${base_per_period:,.2f}</strong>. Totuși, deoarece {reason}, am {factor_txt}, "
                f"ajungând la <strong>${amount:,.2f}</strong>.{tip}"
            )
        else:  # RSI strategy
            zone = _rsi_zone_label(rsi)
            return (
                f"Ai ales strategia <strong>RSI</strong>. Decizia este declanșată deoarece "
                f"RSI = <strong>{rsi:.1f}</strong> (zona: {zone}). "
                f"Suma investită astăzi: <strong>${amount:,.2f}</strong>."
            )
    else:  # Cash out
        if strategy_lbl == "DCA Fix":
            return (
                f"Ai ales <strong>DCA Fix — Retragere</strong>. Suma de retras per perioadă: "
                f"<strong>${amount:,.2f}</strong> (buget împărțit egal la {n_periods} perioade)."
            )
        elif strategy_lbl == "Hibrid":
            return (
                f"Ai ales strategia <strong>Hibrid — Retragere</strong>. "
                f"RSI: {rsi:.1f} ({_rsi_zone_label(rsi)}). "
                f"Suma retrasă astăzi: <strong>${amount:,.2f}</strong>."
            )
        else:
            return (
                f"Ai ales strategia <strong>RSI — Retragere</strong>. "
                f"Suma retrasă astăzi: <strong>${amount:,.2f}</strong>."
            )


def _next_action_date(frequency: str) -> str:
    """Returns a Romanian-formatted next action date string based on frequency."""
    import datetime
    _RO_MONTHS = [
        "", "Ianuarie", "Februarie", "Martie", "Aprilie", "Mai", "Iunie",
        "Iulie", "August", "Septembrie", "Octombrie", "Noiembrie", "Decembrie",
    ]
    today = datetime.date.today()
    if frequency == "Daily":
        next_date = today + datetime.timedelta(days=1)
        label = "mâine"
    elif frequency == "Weekly":
        next_date = today + datetime.timedelta(weeks=1)
        label = "săptămâna viitoare"
    else:  # Monthly
        month = today.month % 12 + 1
        year = today.year + (1 if today.month == 12 else 0)
        day = min(today.day, [31,28,31,30,31,30,31,31,30,31,30,31][month - 1])
        next_date = today.replace(year=year, month=month, day=day)
        label = "luna viitoare"
    return f"{next_date.day} {_RO_MONTHS[next_date.month]} {next_date.year} ({label})"


def _render_portfolio_chart(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    action_colors = [
        "#00d4aa" if a == "BUY" else "#ff4b4b" if a == "SELL" else "#888888"
        for a in df["Action"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Period"], y=df["Portfolio Value ($)"],
        mode="lines+markers", name="Portfolio Value",
        line=dict(color="#00d4aa", width=2.5),
        marker=dict(size=7, color="#00d4aa"),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
    ))
    fig.add_trace(go.Bar(
        x=df["Period"], y=df["Amount ($)"],
        name="Transaction Amount",
        marker_color=action_colors,
        opacity=0.55,
        yaxis="y2",
    ))
    n = len(rows)
    tick_every = max(1, n // 20)  # show at most ~20 x-axis labels
    fig.update_layout(
        title="Portfolio Value & Transactions per Period",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Period", showgrid=False,
                   tickmode="array",
                   tickvals=df["Period"].tolist()[::tick_every]),
        yaxis=dict(title="Portfolio Value ($)", showgrid=True, gridcolor="#1c2c1c"),
        yaxis2=dict(title="Transaction ($)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_portfolio_chart_with_prediction(rows: list[dict]) -> None:
    """Portfolio chart with an additional dashed predicted portfolio line."""
    df = pd.DataFrame(rows)
    action_colors = [
        "#00d4aa" if a == "BUY" else "#ff4b4b" if a == "SELL" else "#888888"
        for a in df["Action"]
    ]

    fig = go.Figure()

    # Actual portfolio value (solid)
    fig.add_trace(go.Scatter(
        x=df["Period"], y=df["Portfolio Value ($)"],
        mode="lines+markers", name="Portfolio Value (current price)",
        line=dict(color="#00d4aa", width=2.5),
        marker=dict(size=6, color="#00d4aa"),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
    ))

    # Predicted portfolio value (dashed)
    if "Predicted Portfolio ($)" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Period"], y=df["Predicted Portfolio ($)"],
            mode="lines", name="Predicted Portfolio (est.)",
            line=dict(color="#a78bfa", width=2, dash="dash"),
        ))

    # Transaction bars
    fig.add_trace(go.Bar(
        x=df["Period"], y=df["Amount ($)"],
        name="Transaction Amount",
        marker_color=action_colors,
        opacity=0.45,
        yaxis="y2",
    ))

    n = len(rows)
    tick_every = max(1, n // 20)
    fig.update_layout(
        title="Portfolio Value & Predicted Growth",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Period", showgrid=False,
                   tickmode="array",
                   tickvals=df["Period"].tolist()[::tick_every]),
        yaxis=dict(title="Portfolio Value ($)", showgrid=True, gridcolor="#1c2c1c"),
        yaxis2=dict(title="Transaction ($)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main render
# ──────────────────────────────────────────────────────────────────────────────

def render() -> None:
    """Math Logic page — entry point called from app.py."""
    st.title("Math Logic — Rule-Based Advisor")
    st.markdown(
        "Tell us your budget and time frame. "
        "We'll fetch live market data and use proven investment rules to tell you "
        "**exactly how much to act on today**."
    )

    # ── Parameter form ────────────────────────────────────────────────────────
    with st.form("rule_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker_choice = st.selectbox("Asset", _WATCHLIST, index=0)
            objective     = st.selectbox("I want to…", ["Invest", "Cash out"])
            strategy_lbl  = st.selectbox("Strategy", list(_STRATEGY_MAP.keys()))

        with col2:
            budget        = st.number_input(
                "My Budget / Target ($)",
                min_value=100.0,
                value=10_000.0,
                step=100.0,
                help="Invest: money to invest. Cash out: total amount you want to extract.",
            )
            timeframe_lbl = st.selectbox("Over how long?", _TIMEFRAMES, index=3)
            qty_held      = st.number_input(
                "Units I currently hold (Cash out only)",
                min_value=0.0,
                value=0.0,
                step=0.001,
                format="%.6f",
                help="How many units of this asset you currently own. Required for Cash out.",
            )
            avg_cost      = st.number_input(
                "My avg. purchase price ($, Cash out only)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Your average cost per unit. Used to check if selling is profitable.",
            )

        with col3:
            frequency = st.selectbox(
                "How often do you want to act?",
                _FREQUENCIES,
                index=2,   # default: Monthly
                help=(
                    "Daily — check RSI every trading day (best for RSI/Hybrid strategy).\n"
                    "Weekly — act once a week.\n"
                    "Monthly — act once a month (recommended for DCA Fix)."
                ),
            )
            st.markdown(" ")  # spacer
            st.info(
                "**Tip:** DCA Fix works best Monthly. "
                "RSI & Hybrid strategies can benefit from Daily or Weekly timing.",
                icon="💡",
            )

        run_btn = st.form_submit_button(
            "Get My Recommendation", type="primary", use_container_width=True
        )

    strategy_clips = _STRATEGY_MAP[strategy_lbl]
    n_periods      = _PERIOD_MAP[(frequency, timeframe_lbl)]

    # ── Auto-fetch market data ─────────────────────────────────────────────────
    mkt = _fetch_market_data(ticker_choice)
    if mkt is None:
        st.warning(
            f"Could not fetch live data for **{ticker_choice}**. "
            "Check your internet connection or try another asset."
        )
        current_price, current_rsi, ma200 = 100.0, 50.0, 95.0
    else:
        current_price = mkt["price"]
        current_rsi   = float(mkt["rsi"])
        ma200         = mkt["ma200"]

    # ── Pre-run placeholder ───────────────────────────────────────────────────
    if not run_btn and "rule_sim_rows" not in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Current RSI")
            _render_rsi_gauge(current_rsi)
            _rsi_zone_badge(current_rsi)
        with col2:
            st.markdown("#### Risk Level")
            risk_score = (current_rsi / 100.0) if objective == "Invest" else (1.0 - current_rsi / 100.0)
            display_risk_meter(min(1.0, max(0.0, risk_score)))
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    if run_btn:
        clips_ticker = ticker_choice.replace("-", "").replace(".", "").upper()

        # Warn user if period count is very high (slow CLIPS loop)
        if n_periods > 100:
            st.info(
                f"Running {n_periods} periods ({frequency.lower()} over {timeframe_lbl}). "
                "This may take a few seconds…"
            )

        if objective == "Cash out" and qty_held <= 0.0:
            st.warning(
                "For **Cash out**, please enter how many units you currently hold "
                "(\"Units I currently hold\"). Without holdings, no sell rule can trigger."
            )
            return

        with st.spinner("Analysing market conditions…"):
            predicted_prices = _forecast_prices(ticker_choice, n_periods, frequency)

            is_cashout = (objective == "Cash out")
            first = _run_clips_period(
                objective=objective,
                strategy_clips=strategy_clips,
                target_sum=budget,
                periods=n_periods,
                budget=0.0 if is_cashout else budget,
                ticker=clips_ticker,
                rsi=current_rsi,
                price=current_price,
                ma200=ma200,
                qty=qty_held if is_cashout else 0.0,
                avg_price=avg_cost if is_cashout else 0.0,
            )

            if first["error"]:
                st.error(f"Rules engine error: {first['error']}")
                return

            sim_rows = _run_simulation(
                objective=objective,
                strategy_clips=strategy_clips,
                target_sum=budget,
                n_periods=n_periods,
                budget=0.0 if is_cashout else budget,
                ticker=clips_ticker,
                rsi=current_rsi,
                price=current_price,
                ma200=ma200,
                qty=qty_held if is_cashout else 0.0,
                avg_price=avg_cost if is_cashout else 0.0,
                frequency=frequency,
                predicted_prices=predicted_prices,
            )

        st.session_state["rule_first"]    = first
        st.session_state["rule_sim_rows"] = sim_rows
        st.session_state["rule_params"]   = {
            "ticker":          ticker_choice,
            "rsi":             current_rsi,
            "price":           current_price,
            "ma200":           ma200,
            "objective":       objective,
            "budget":          budget,
            "timeframe":       timeframe_lbl,
            "n_periods":       n_periods,
            "frequency":       frequency,
            "strategy_lbl":    strategy_lbl,
            "has_prediction":  predicted_prices is not None,
        }

    # ── Display results ───────────────────────────────────────────────────────
    if "rule_first" not in st.session_state:
        return

    first: dict    = st.session_state["rule_first"]
    sim_rows: list = st.session_state["rule_sim_rows"]
    params: dict   = st.session_state["rule_params"]

    action      = first["action"]
    amount      = first["amount"]
    ticker_disp = params["ticker"]
    rsi         = params["rsi"]
    price       = params["price"]
    ma200       = params["ma200"]
    budget      = params["budget"]
    n_periods   = params["n_periods"]
    frequency   = params["frequency"]
    objective   = params["objective"]
    strategy_lbl = params["strategy_lbl"]

    action_color = "#00d4aa" if action == "BUY" else "#ff4b4b" if action == "SELL" else "#ffa500"

    df_full  = pd.DataFrame(sim_rows)
    has_pred = "Predicted Portfolio ($)" in df_full.columns

    # ── SECTION 1: Recomandarea Principală ───────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="color:#888;font-size:0.78em;font-weight:700;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">'
        '1. Recomandarea Principală</p>',
        unsafe_allow_html=True,
    )

    if action == "HOLD":
        action_ro = "AȘTEAPTĂ"
        headline_html = f"Piața este neutră — nu acționa astăzi"
        units_txt = f"RSI: {rsi:.1f} ({_rsi_zone_label(rsi)}). Revino la următoarea perioadă."
    else:
        action_ro = "CUMPĂRĂ" if action == "BUY" else "VINDE"
        verb_ro   = "Investește" if objective == "Invest" else "Vinde"
        units     = amount / price if price > 0 else 0
        headline_html = f"{verb_ro} <strong>${amount:,.2f}</strong> în {ticker_disp} astăzi"
        units_txt = f"Asta înseamnă ~{units:.4f} unități la prețul curent de ${price:,.2f}."

    st.markdown(
        f"""
        <div style="background:#0d1f0d;border:1.5px solid {action_color};
                    border-radius:16px;padding:28px 32px;margin-bottom:12px;">
            <div style="color:{action_color};font-size:2.2em;font-weight:900;
                        letter-spacing:3px;margin-bottom:12px;">
                {action_ro}
            </div>
            <div style="color:#f0f0f0;font-size:1.6em;font-weight:700;margin-bottom:10px;">
                {headline_html}
            </div>
            <div style="color:#aaa;font-size:1em;line-height:1.7;">{units_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SECTION 2: Radiografia Pieței ────────────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="color:#888;font-size:0.78em;font-weight:700;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">'
        '2. Radiografia Pieței</p>',
        unsafe_allow_html=True,
    )

    col_price, col_ma, col_rsi = st.columns(3)

    with col_price:
        st.markdown(
            f"""
            <div style="background:#111;border-radius:12px;padding:20px 24px;text-align:center;">
                <div style="color:#888;font-size:0.8em;font-weight:600;
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                    {ticker_disp} — Preț Curent
                </div>
                <div style="color:#f0f0f0;font-size:2.2em;font-weight:800;">
                    ${price:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ma:
        above = price > ma200
        trend_color = "#00d4aa" if above else "#ff4b4b"
        trend_arrow = "↑" if above else "↓"
        trend_text  = "Trend Crescător" if above else "Trend Descrescător"
        st.markdown(
            f"""
            <div style="background:#111;border-radius:12px;padding:20px 24px;text-align:center;">
                <div style="color:#888;font-size:0.8em;font-weight:600;
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                    MA200 (Media Mobilă 200)
                </div>
                <div style="color:#f0f0f0;font-size:2.2em;font-weight:800;margin-bottom:10px;">
                    ${ma200:,.2f}
                </div>
                <div style="display:inline-block;background:{trend_color}22;
                            border:1.5px solid {trend_color};border-radius:20px;
                            padding:5px 16px;color:{trend_color};
                            font-weight:700;font-size:0.9em;">
                    {trend_arrow} {trend_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_rsi:
        _render_rsi_gauge_0_100(rsi)

    # ── SECTION 3: Logica din Spatele Deciziei ────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="color:#888;font-size:0.78em;font-weight:700;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">'
        '3. Logica din Spatele Deciziei</p>',
        unsafe_allow_html=True,
    )

    explanation = _build_strategy_explanation(
        strategy_lbl=strategy_lbl,
        rsi=rsi,
        amount=amount,
        budget=budget,
        n_periods=n_periods,
        objective=objective,
    )
    st.markdown(
        f"""
        <div style="background:#111;border-radius:12px;padding:20px 28px;
                    color:#ccc;font-size:1.05em;line-height:1.8;">
            {explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SECTION 4: Progresul Planului ────────────────────────────────────────
    st.divider()
    st.markdown(
        '<p style="color:#888;font-size:0.78em;font-weight:700;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">'
        '4. Progresul Planului</p>',
        unsafe_allow_html=True,
    )

    freq_singular = {"Daily": "Zi", "Weekly": "Săptămână", "Monthly": "Lună"}.get(frequency, frequency)
    freq_plural   = {"Daily": "Zile", "Weekly": "Săptămâni", "Monthly": "Luni"}.get(frequency, frequency)
    budget_left   = budget - amount
    next_date_str = _next_action_date(frequency)

    p1, p2, p3 = st.columns(3)
    p1.metric(
        f"Status Plan ({freq_singular} 1 din {n_periods})",
        f"1 / {n_periods} {freq_plural}",
    )
    p2.metric(
        "Buget Rămas",
        f"${budget_left:,.2f}",
        delta=f"-${amount:,.2f}",
        delta_color="inverse",
    )
    p3.metric(
        "Buget Total",
        f"${budget:,.2f}",
    )

    st.markdown(
        f"""
        <div style="background:#111;border-radius:10px;padding:14px 22px;
                    margin-top:8px;color:#aaa;font-size:0.95em;line-height:1.7;">
            <strong style="color:#f0f0f0;">Sfat:</strong>
            Revino pe <strong style="color:#f0f0f0;">{next_date_str}</strong>
            pentru următoarea analiză și recomandare.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not sim_rows:
        st.warning("No simulation results. Try a different asset or time frame.")
        return

    # ── Expanded details (collapsed by default) ───────────────────────────────
    st.divider()
    with st.expander(f"Planul complet perioadă cu perioadă ({params['frequency']} / {params['timeframe']})", expanded=False):
        if has_pred:
            st.caption(
                "**Portfolio Value ($)** folosește prețul de azi. "
                "**Predicted Portfolio ($)** folosește prețul estimat per perioadă "
                "bazat pe randamentul mediu istoric al activului."
            )
        else:
            st.caption("RSI este menținut constant la valoarea de azi pentru toate perioadele.")

        display_cols = ["Period", "Action", "Amount ($)", "Cash Left ($)", "Portfolio Value ($)"]
        if has_pred:
            display_cols += ["Predicted Price ($)", "Predicted Portfolio ($)"]
        df_display = df_full[display_cols].copy()

        def _style_action(val: str) -> str:
            if val == "BUY":
                return "color:#00d4aa;font-weight:bold;"
            if val == "SELL":
                return "color:#ff4b4b;font-weight:bold;"
            return "color:#ffa500;"

        styled = df_display.style.map(_style_action, subset=["Action"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.subheader("Valoarea portofoliului în timp")
        if has_pred:
            _render_portfolio_chart_with_prediction(sim_rows)
        else:
            _render_portfolio_chart(sim_rows)

    with st.expander("Cum a fost calculat? (Log Motor Reguli)", expanded=False):
        triggered = first.get("rules_triggered", [])
        if triggered:
            st.markdown("**Reguli declanșate:**")
            for r in triggered:
                st.markdown(f"- `{r}`")
        else:
            st.info("Nicio regulă declanșată (fallback HOLD).")
        st.caption("Output brut din sistemul expert CLIPS pentru prima perioadă.")
        if first["clips_log"].strip():
            st.code(first["clips_log"], language=None)
        else:
            st.info("Niciun printout din motorul de reguli.")

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_bytes = df_full.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descarcă planul (CSV)",
        data=csv_bytes,
        file_name=(
            f"tradingpal_{params['ticker']}_{params['objective']}_"
            f"{params['frequency']}_{params['timeframe'].replace(' ','')}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )
