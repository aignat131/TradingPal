"""
Math Logic — Rule-Based Advisor

Runs the actual CLIPS expert system defined in assets/Baza_de_reguli.clp
via the clipspy library. No rules are reimplemented in Python; the .clp
file is loaded verbatim and executed by the CLIPS inference engine.

Uses clipspy 1.0.6 — a Python 3.10 pre-built wheel is available on PyPI.

Entry point: render()
"""
import logging
import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from ui.components import display_risk_meter, display_signal_gauge, display_verdict_badge

logger = logging.getLogger(__name__)

# Path to the CLIPS rule base (relative to project root where streamlit runs)
_CLP_PATH = os.path.join("assets", "Baza_de_reguli.clp")

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
# CLIPS engine
# ──────────────────────────────────────────────────────────────────────────────

def _make_capture_router():
    """
    Dynamically build a clips.Router subclass that captures CLIPS printout.
    Defined at call-time so clips import errors surface cleanly.
    """
    import clips

    class CaptureRouter(clips.Router):
        # clips.Router uses __slots__; we store the buffer in a class-level
        # dict keyed by id(self) to avoid slot conflicts.
        _buffers: dict = {}

        def __init__(self):
            super().__init__("capture", 100)
            CaptureRouter._buffers[id(self)] = []

        def query(self, name: str) -> bool:
            return name in ("stdout", "stderr", "stdin", "t",
                            "wdisplay", "wdialog", "wwarning", "werror")

        def write(self, name: str, message: str) -> None:
            CaptureRouter._buffers[id(self)].append(message)

        def read(self, name: str) -> int:
            return 0

        def unread(self, name: str, char: int) -> int:
            return 0

        def exit(self, code: int) -> None:
            pass

        def get_output(self) -> str:
            return "".join(CaptureRouter._buffers.get(id(self), []))

        def __del__(self):
            CaptureRouter._buffers.pop(id(self), None)

    return CaptureRouter()


def _run_clips_period(
    objective: str,
    strategy_clips: str,
    target_sum: float,
    periods: int,
    budget: float,
    ticker: str,
    rsi: float,
    price: float,
    ma200: float,
    qty: float,
    avg_price: float,
) -> dict:
    """
    Run one CLIPS inference cycle.

    Loads Baza_de_reguli.clp, asserts user-supplied facts (WITHOUT calling
    env.reset() so the test deffacts are never loaded), runs the engine, and
    returns the updated slot values extracted from the working memory.

    Returns a dict with keys:
        clips_log, action, amount, new_budget, new_qty, new_avg_price,
        new_target_sum, new_periods, rule_fired, error
    """
    result = {
        "clips_log": "",
        "action": "HOLD",
        "amount": 0.0,
        "new_budget": budget,
        "new_qty": qty,
        "new_avg_price": avg_price,
        "new_target_sum": target_sum,
        "new_periods": max(0, periods - 1),
        "rule_fired": "Nicio regulă aplicabilă",
        "error": None,
    }

    try:
        import clips  # clipspy 1.0.6

        env = clips.Environment()

        # Attach capture router BEFORE load so printout during load is also caught
        capture = _make_capture_router()
        env.add_router(capture)

        env.load(_CLP_PATH)
        # Do NOT call env.reset() — avoids loading the test deffacts

        # Assert initial facts using exact CLIPS template slot names from the .clp file
        env.assert_string(
            f'(stare-sistem '
            f'(faza initializare) '
            f'(obiectiv "{objective}") '
            f'(strategie "{strategy_clips}") '
            f'(suma-tinta {target_sum:.4f}) '
            f'(perioade-ramase {periods}) '
            f'(exista-fisier da) '
            f'(eof nu))'
        )
        env.assert_string(
            f'(portofoliu (buget-disponibil {budget:.4f}))'
        )
        # CLIPS SYMBOL type for (nume) — no quotes around the ticker symbol
        env.assert_string(
            f'(activ-piata (nume {ticker}) (rsi {rsi:.4f}) '
            f'(pret {price:.4f}) (ma200 {ma200:.4f}))'
        )
        env.assert_string(
            f'(detinere-activ (nume-activ {ticker}) '
            f'(cantitate {qty:.6f}) (pret-mediu {avg_price:.4f}))'
        )

        env.run()
        result["clips_log"] = capture.get_output()

        # Read back the final working memory
        stare = {}
        portof = {}
        detinere = {}

        for fact in env.facts():
            tmpl_name = fact.template.name
            slots = dict(fact)

            if tmpl_name == "stare-sistem":
                stare = slots
            elif tmpl_name == "portofoliu":
                portof = slots
            elif tmpl_name == "detinere-activ":
                detinere = slots

        # Extract updated values
        new_budget = float(portof.get("buget-disponibil", budget))
        new_qty = float(detinere.get("cantitate", qty))
        new_avg = float(detinere.get("pret-mediu", avg_price))
        new_target = float(stare.get("suma-tinta", target_sum))
        new_periods = int(stare.get("perioade-ramase", max(0, periods - 1)))

        # Infer action and amount from changes
        budget_delta = new_budget - budget
        qty_delta = new_qty - qty

        if qty_delta > 1e-9:
            action = "BUY"
            amount = abs(budget_delta)
        elif qty_delta < -1e-9:
            action = "SELL"
            amount = abs(budget_delta)
        else:
            action = "HOLD"
            amount = 0.0

        # Infer which rule fired from the CLIPS printout
        log = result["clips_log"]
        rule_fired = "Nicio regulă aplicabilă (HOLD)"
        if "EXEC:" in log:
            for line in log.splitlines():
                if "EXEC:" in line:
                    rule_fired = line.strip()
                    break
        elif "ALERTĂ:" in log:
            for line in log.splitlines():
                if "ALERTĂ:" in line:
                    rule_fired = line.strip()
                    break

        result.update({
            "action": action,
            "amount": amount,
            "new_budget": new_budget,
            "new_qty": new_qty,
            "new_avg_price": new_avg,
            "new_target_sum": new_target,
            "new_periods": new_periods,
            "rule_fired": rule_fired,
        })

    except ImportError:
        result["error"] = (
            "clipspy is not installed. Run: pip install clipspy==0.3.3"
        )
    except Exception as exc:
        logger.exception("CLIPS engine error: %s", exc)
        result["error"] = str(exc)

    return result


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
    predicted_prices: list[float] | None = None,
) -> list[dict]:
    """
    Simulate all N periods sequentially using the CLIPS engine.
    RSI / price / MA200 stay constant for decision logic (live snapshot).
    If predicted_prices is provided, each row also shows the forecasted price
    and the portfolio value recalculated at that forecasted price.
    Returns a list of row dicts for DataFrame display.
    """
    period_label = {"Daily": "Day", "Weekly": "Week", "Monthly": "Month"}[frequency]

    rows = []
    cur_budget = budget
    cur_qty = qty
    cur_avg = avg_price
    cur_target = target_sum
    cur_periods = n_periods

    for period in range(1, n_periods + 1):
        if cur_periods <= 0:
            break

        r = _run_clips_period(
            objective=objective,
            strategy_clips=strategy_clips,
            target_sum=cur_target,
            periods=cur_periods,
            budget=cur_budget,
            ticker=ticker,
            rsi=rsi,
            price=price,
            ma200=ma200,
            qty=cur_qty,
            avg_price=cur_avg,
        )

        if r["error"]:
            st.error(f"CLIPS error on {period_label.lower()} {period}: {r['error']}")
            break

        portfolio_value = r["new_budget"] + r["new_qty"] * price

        pred_price = (
            predicted_prices[period - 1]
            if predicted_prices and period - 1 < len(predicted_prices)
            else None
        )
        pred_portfolio = (
            round(r["new_budget"] + r["new_qty"] * pred_price, 2)
            if pred_price is not None
            else None
        )

        row = {
            "Period": f"{period_label} {period}",
            "Action": r["action"],
            "Amount ($)": round(r["amount"], 2),
            "Cash Left ($)": round(r["new_budget"], 2),
            "Portfolio Value ($)": round(portfolio_value, 2),
        }
        if pred_price is not None:
            row["Predicted Price ($)"] = pred_price
            row["Predicted Portfolio ($)"] = pred_portfolio

        rows.append(row)

        # Advance state for next period
        cur_budget = r["new_budget"]
        cur_qty = r["new_qty"]
        cur_avg = r["new_avg_price"]
        cur_target = r["new_target_sum"]
        cur_periods = r["new_periods"]

        # Stop if nothing left to do
        if r["action"] == "HOLD" and cur_target <= 0:
            break

    return rows


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
            budget        = st.number_input("My Budget ($)", min_value=100.0,
                                            value=10_000.0, step=100.0)
            timeframe_lbl = st.selectbox("Over how long?", _TIMEFRAMES, index=3)

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

        with st.spinner("Analysing market conditions…"):
            predicted_prices = _forecast_prices(ticker_choice, n_periods, frequency)

            first = _run_clips_period(
                objective=objective,
                strategy_clips=strategy_clips,
                target_sum=budget,
                periods=n_periods,
                budget=budget,
                ticker=clips_ticker,
                rsi=current_rsi,
                price=current_price,
                ma200=ma200,
                qty=0.0,
                avg_price=0.0,
            )

            if first["error"]:
                st.error(f"Rules engine error: {first['error']}")
                return

            sim_rows = _run_simulation(
                objective=objective,
                strategy_clips=strategy_clips,
                target_sum=budget,
                n_periods=n_periods,
                budget=budget,
                ticker=clips_ticker,
                rsi=current_rsi,
                price=current_price,
                ma200=ma200,
                qty=0.0,
                avg_price=0.0,
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

    action     = first["action"]
    amount     = first["amount"]
    ticker_disp = params["ticker"]
    verb       = "invest" if params["objective"] == "Invest" else "sell"
    action_color = "#00d4aa" if action == "BUY" else "#ff4b4b" if action == "SELL" else "#ffa500"

    # ── Today's recommendation card ───────────────────────────────────────────
    st.divider()
    above = params["price"] > params["ma200"]
    trend_label = "above MA200 — bullish trend" if above else "below MA200 — caution"

    if action == "HOLD":
        headline = f"Hold for now — market conditions are neutral"
        sub = (
            f"RSI is {params['rsi']:.1f} ({_rsi_zone_label(params['rsi'])}) "
            f"and {ticker_disp} is {trend_label}. "
            f"Wait for a better entry point."
        )
    else:
        headline = f"Today's move: {verb} **${amount:,.2f}** in {ticker_disp}"
        units = amount / params["price"] if params["price"] > 0 else 0
        sub = (
            f"That buys approximately **{units:.4f} units** at the current price of "
            f"${params['price']:,.2f}. "
            f"RSI is {params['rsi']:.1f} ({_rsi_zone_label(params['rsi'])}) "
            f"and price is {trend_label}."
        )

    st.markdown(
        f"""
        <div style="background:#111;border-radius:14px;padding:24px 28px;margin-bottom:8px;">
            <div style="color:{action_color};font-size:0.8em;font-weight:700;
                        text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">
                {action} — {params['objective'].upper()}
            </div>
            <div style="color:#f0f0f0;font-size:1.5em;font-weight:700;margin-bottom:10px;">
                {headline}
            </div>
            <div style="color:#999;font-size:0.95em;line-height:1.6;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Market snapshot + verdict ─────────────────────────────────────────────
    col_badge, col_gauge, col_risk = st.columns(3)

    with col_badge:
        st.markdown("#### Decision")
        confidence = min(1.0, amount / max(params["budget"], 1))
        display_verdict_badge(
            decision=action if action in ("BUY", "SELL", "HOLD") else "HOLD",
            confidence=confidence,
        )

    with col_gauge:
        st.markdown("#### RSI Signal")
        _render_rsi_gauge(params["rsi"])
        _rsi_zone_badge(params["rsi"])

    with col_risk:
        st.markdown("#### Risk Level")
        risk_score = (params["rsi"] / 100.0) if params["objective"] == "Invest" else (1.0 - params["rsi"] / 100.0)
        display_risk_meter(min(1.0, max(0.0, risk_score)))

    # ── Plan overview metrics ─────────────────────────────────────────────────
    st.divider()
    df_full = pd.DataFrame(sim_rows)
    has_pred = "Predicted Portfolio ($)" in df_full.columns

    total_invested  = sum(r["Amount ($)"] for r in sim_rows)
    final_actual    = sim_rows[-1]["Portfolio Value ($)"] if sim_rows else 0.0
    final_predicted = (
        sim_rows[-1].get("Predicted Portfolio ($)", final_actual)
        if sim_rows else 0.0
    ) or final_actual
    gain_predicted  = final_predicted - total_invested

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Budget", f"${params['budget']:,.0f}")
    m2.metric("Time Frame", params["timeframe"])
    m3.metric("Frequency", params["frequency"])
    m4.metric(
        f"Today's {'Investment' if params['objective'] == 'Invest' else 'Sale'}",
        f"${amount:,.2f}",
    )
    m5.metric(
        "Projected Final Value" + (" (est.)" if has_pred else ""),
        f"${final_predicted:,.2f}",
        delta=f"${gain_predicted:+,.2f}",
        help=(
            "Estimated using the asset's historical average return per period. "
            "Not a guarantee — actual results will vary."
        ) if has_pred else None,
    )

    if not sim_rows:
        st.warning("No simulation results. Try a different asset or time frame.")
        return

    # ── Period-by-period plan ─────────────────────────────────────────────────
    st.divider()
    freq_label = params["frequency"].lower()
    st.subheader(f"{params['frequency']} plan over {params['timeframe']}")
    if has_pred:
        st.caption(
            "**Portfolio Value ($)** uses today's price. "
            "**Predicted Portfolio ($)** uses the estimated price for that period, "
            "based on this asset's historical average return. "
            "RSI is held constant at today's value for all decision periods."
        )
    else:
        st.caption(
            "RSI is held constant at today's value across all periods. "
            "Price prediction unavailable for this asset."
        )

    # Build display columns — always show core cols, add prediction cols if available
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

    # ── Portfolio chart (add predicted line if available) ─────────────────────
    st.divider()
    st.subheader("Portfolio value over time")
    if has_pred:
        _render_portfolio_chart_with_prediction(sim_rows)
    else:
        _render_portfolio_chart(sim_rows)

    # ── CLIPS log (collapsed — for transparency) ──────────────────────────────
    with st.expander("How was this calculated? (Rules engine log)", expanded=False):
        st.caption("Raw output from the CLIPS expert system for the first period.")
        if first["clips_log"].strip():
            st.code(first["clips_log"], language=None)
        else:
            st.info("No printout from the rules engine.")

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_bytes = df_full.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download plan (CSV)",
        data=csv_bytes,
        file_name=(
            f"tradingpal_{params['ticker']}_{params['objective']}_"
            f"{params['frequency']}_{params['timeframe'].replace(' ','')}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )
