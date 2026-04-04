"""
Trade Advisor Page — portfolio-first recommendation interface.

Flow:
1. User enters their profile (budget, risk, horizon)
2. App analyses every asset in the watchlist
3. Top 10 opportunities are displayed as scrollable cards (ranked by profit potential)
"""
import html
import logging

import streamlit as st

from agents.manager_agent import ManagerAgent
from agents.technical_agent import TechnicalAgent
import config

logger = logging.getLogger(__name__)

WATCHLIST = [t for t in config.STOCK_WATCHLIST if t != "Other (type manually)"]

# Color / background maps reused across card helpers
_COLOR = {"BUY": "#00d4aa", "SELL": "#ff4b4b", "HOLD": "#ffa500"}
_BG    = {"BUY": "#001f18",  "SELL": "#1f0000",  "HOLD": "#1a1000"}


def render(
    sentiment_analyzer,
    manager_agent: ManagerAgent,
) -> None:
    """Render the Trade Advisor page."""
    st.title("TradingPal-AI — Portfolio Advisor")
    st.markdown(
        "Tell us about your investment profile and we'll scan **every asset in our database**, "
        "rank them by profit potential, and show you the top 10 opportunities right now."
    )

    # ------------------------------------------------------------------
    # User Profile Form
    # ------------------------------------------------------------------
    with st.form("profile_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            budget = st.number_input(
                "Investment Budget ($)",
                min_value=100.0,
                max_value=1_000_000.0,
                value=1_000.0,
                step=100.0,
            )
            risk = st.selectbox(
                "Risk Tolerance",
                ["Low", "Medium", "High"],
                index=1,
            )

        with col2:
            horizon = st.selectbox(
                "Time Horizon",
                [
                    "Intraday",
                    "Short-term (1-4 weeks)",
                    "Medium-term (1-3 months)",
                    "Long-term (>3 months)",
                ],
                index=1,
            )
            trade_frequency = st.selectbox(
                "How often do you want to trade?",
                [
                    "Daily (Active Trader)",
                    "Weekly (Regular Investor)",
                    "Monthly (Passive Investor)",
                ],
                index=1,
            )

        with col3:
            recurring_amount = st.number_input(
                "Recurring Savings ($, optional)",
                min_value=0.0,
                max_value=100_000.0,
                value=0.0,
                step=50.0,
                help="Regular amount you plan to add each period (e.g. monthly savings)",
            )
            recurring_period = st.selectbox(
                "Recurring Period",
                ["weekly", "monthly"],
                index=1,
            )

        submitted = st.form_submit_button(
            "Generate My Portfolio",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        _render_placeholder()
        return

    # ------------------------------------------------------------------
    # Analyse every asset in the watchlist
    # ------------------------------------------------------------------
    progress_bar = st.progress(0, text="Scanning markets…")
    tech_agent = TechnicalAgent()
    signals: dict = {}
    total = len(WATCHLIST)

    for i, ticker in enumerate(WATCHLIST):
        progress_bar.progress(
            int((i + 1) / total * 100),
            text=f"Analysing {ticker}… ({i + 1}/{total})",
        )
        try:
            signals[ticker] = tech_agent.analyze(ticker)
        except Exception as exc:
            logger.warning("Failed to analyse %s: %s", ticker, exc)

    progress_bar.empty()

    if not signals:
        st.error("Could not retrieve market data. Please try again later.")
        return

    # ------------------------------------------------------------------
    # Rank by profit potential via Gemini
    # ------------------------------------------------------------------
    with st.spinner("Ranking opportunities with AI…"):
        portfolio = manager_agent.generate_portfolio_recommendations(
            budget=budget,
            risk=risk,
            horizon=horizon,
            signals=signals,
            recurring_amount=recurring_amount,
            recurring_period=recurring_period,
        )

    # ------------------------------------------------------------------
    # Results header
    # ------------------------------------------------------------------
    st.divider()
    st.subheader(f"Top 10 Investment Opportunities for You")
    st.caption(
        f"Ranked by profit potential · Budget **${budget:,.0f}** · "
        f"Risk **{risk}** · Horizon **{horizon}**"
    )

    if not portfolio:
        st.info("No ranked results available. Please try again.")
        return

    # ------------------------------------------------------------------
    # Instagram-style scrollable cards
    # ------------------------------------------------------------------
    for rank, item in enumerate(portfolio, start=1):
        _render_card(rank, item, budget)

    # ------------------------------------------------------------------
    # Recurring Savings Projection (if set)
    # ------------------------------------------------------------------
    if recurring_amount > 0:
        st.divider()
        _render_recurring_summary(recurring_amount, recurring_period, budget, horizon)


# ---------------------------------------------------------------------------
# Card renderer
# ---------------------------------------------------------------------------

def _render_card(rank: int, item: dict, budget: float) -> None:
    """Render a single investment opportunity card."""
    ticker    = item.get("ticker", "")
    signal    = item.get("signal", "HOLD").upper()
    score     = max(0.0, min(1.0, item.get("score", 0.5)))
    reasoning = html.escape(item.get("reasoning", ""))
    alloc_pct = item.get("suggested_allocation", 0.0)
    alloc_amt = budget * alloc_pct / 100.0

    color = _COLOR.get(signal, "#888888")
    bg    = _BG.get(signal, "#0e1117")
    bar_w = int(score * 100)
    stars = "★" * min(5, max(1, round(score * 5)))
    stars_empty = "☆" * (5 - len(stars))

    # --- Header: rank + ticker + signal badge ---
    st.markdown(
        f'<div style="background:{bg};border:1px solid {color}33;border-left:6px solid {color};'
        f'border-radius:14px 14px 0 0;padding:18px 24px 10px 24px;margin-top:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<span style="color:#444;font-size:1.5em;font-weight:700;">#{rank}</span>'
        f'<div><div style="color:#fff;font-size:1.9em;font-weight:800;letter-spacing:3px;line-height:1;">{ticker}</div>'
        f'<div style="color:{color};font-size:0.9em;margin-top:3px;">{stars}'
        f'<span style="color:#333;">{stars_empty}</span></div></div></div>'
        f'<div style="background:{color}22;border:2px solid {color};border-radius:10px;'
        f'padding:8px 20px;color:{color};font-weight:800;font-size:1.2em;letter-spacing:3px;">'
        f'{signal}</div></div></div>',
        unsafe_allow_html=True,
    )

    # --- Profit potential bar ---
    st.markdown(
        f'<div style="background:{bg};border-left:6px solid {color};border-right:1px solid {color}33;'
        f'padding:10px 24px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<span style="color:#888;font-size:0.78em;text-transform:uppercase;letter-spacing:1px;">Profit Potential</span>'
        f'<span style="color:{color};font-size:0.88em;font-weight:700;">{score*100:.0f}%</span></div>'
        f'<div style="background:#1a1a2e;border-radius:6px;height:7px;overflow:hidden;">'
        f'<div style="background:linear-gradient(90deg,{color}88,{color});width:{bar_w}%;height:100%;border-radius:6px;"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # --- Reasoning (plain text via st.markdown, no HTML injection risk) ---
    st.markdown(
        f'<div style="background:{bg};border-left:6px solid {color};border-right:1px solid {color}33;'
        f'padding:10px 24px 4px 24px;">'
        f'<p style="color:#d0d0d0;font-size:0.95em;line-height:1.6;margin:0;">{reasoning}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Allocation footer (BUY only) ---
    if signal == "BUY" and alloc_pct > 0:
        st.markdown(
            f'<div style="background:{bg};border-left:6px solid {color};border-right:1px solid {color}33;'
            f'border-bottom:1px solid {color}33;border-radius:0 0 14px 14px;padding:8px 24px 14px 24px;">'
            f'<span style="color:#aaa;font-size:0.82em;">Suggested allocation: </span>'
            f'<span style="color:{color};font-weight:bold;">${alloc_amt:,.0f} &nbsp;&middot;&nbsp; {alloc_pct:.0f}% of budget</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:{bg};border-left:6px solid {color};border-right:1px solid {color}33;'
            f'border-bottom:1px solid {color}33;border-radius:0 0 14px 14px;height:10px;"></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Recurring savings projection
# ---------------------------------------------------------------------------

def _render_recurring_summary(
    recurring_amount: float,
    recurring_period: str,
    budget: float,
    horizon: str,
) -> None:
    import plotly.graph_objects as go

    st.subheader("Recurring Savings Projection")
    st.caption(
        f"Investing **${recurring_amount:,.0f} {recurring_period}** on top of your "
        f"initial **${budget:,.0f}** — here's how your total invested capital grows."
    )

    periods_per_year = 52 if recurring_period == "weekly" else 12
    horizon_map = {
        "Intraday": 0.08,
        "Short-term (1-4 weeks)": 0.25,
        "Medium-term (1-3 months)": 0.5,
        "Long-term (>3 months)": 3.0,
    }
    years = horizon_map.get(horizon, 1.0)
    total_periods = max(int(years * periods_per_year), 1)

    cumulative = []
    running = budget
    for i in range(total_periods + 1):
        cumulative.append({"period": i, "total_invested": round(running, 2)})
        running += recurring_amount

    prefix = "W" if recurring_period == "weekly" else "M"
    labels = [f"{prefix}{c['period']}" for c in cumulative]
    values = [c["total_invested"] for c in cumulative]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values,
        mode="lines+markers",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        hovertemplate="Period %{x}<br>Total Invested: $%{y:,.0f}<extra></extra>",
        marker=dict(size=4),
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333", title_text="Period"),
        yaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333", title_text="Total Invested ($)"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    final_invested = budget + recurring_amount * total_periods
    c1, c2, c3 = st.columns(3)
    c1.metric("Initial Investment", f"${budget:,.0f}")
    c2.metric(f"Total Added ({recurring_period.title()})", f"${recurring_amount * total_periods:,.0f}")
    c3.metric("Total Invested by End", f"${final_invested:,.0f}")


# ---------------------------------------------------------------------------
# Placeholder (before first submit)
# ---------------------------------------------------------------------------

def _render_placeholder() -> None:
    st.markdown(
        """
        <div style="
            background: #0e1117;
            border: 1px solid #00d4aa33;
            border-radius: 14px;
            padding: 32px;
            text-align: center;
            margin: 24px 0;
        ">
            <div style="color:#00d4aa; font-size:3em;">📊</div>
            <div style="color:#ffffff; font-size:1.4em; font-weight:700; margin:12px 0 8px;">
                Your personalised portfolio is one click away
            </div>
            <div style="color:#aaa; font-size:1em; line-height:1.6;">
                Fill in your budget, risk tolerance, and time horizon above,<br>
                then click <b style="color:#00d4aa;">Generate My Portfolio</b> to scan every asset we track
                and get your top 10 ranked investment opportunities.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Stocks & ETFs")
        st.markdown("AAPL, MSFT, TSLA, NVDA, SPY, QQQ and more")
    with c2:
        st.markdown("#### Crypto")
        st.markdown("BTC, ETH, SOL ranked alongside traditional assets")
    with c3:
        st.markdown("#### AI Ranking")
        st.markdown("Gemini scores each asset specifically for your profile")
