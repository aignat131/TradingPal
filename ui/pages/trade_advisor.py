"""
Trade Advisor Page — beginner-friendly recommendation interface.

Sections:
1. Input form (ticker dropdown, amount, trading frequency, horizon)
2. Technical Analysis panel (plain English)
3. News Sentiment panel (plain English)
4. AI Recommendation panel (Gemini)
5. Today's Trade Opportunities
6. Other Stocks Worth Watching
"""
import logging

import streamlit as st

from agents.manager_agent import ManagerAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
import config
from ui.components import (
    display_news_sentiment,
    display_signal_gauge,
    display_verdict_badge,
)

POPULAR_TICKERS = config.STOCK_WATCHLIST
from utils.validators import validate_amount, validate_ticker

logger = logging.getLogger(__name__)



def render(
    sentiment_analyzer,
    manager_agent: ManagerAgent,
) -> None:
    """Render the Trade Advisor page."""
    st.title("TradingPal-AI — Trade Advisor")
    st.markdown(
        "Select a stock below and click **Analyze** to get a plain-English recommendation "
        "powered by AI, technical analysis, and news sentiment."
    )

    # ------------------------------------------------------------------
    # Input Form
    # ------------------------------------------------------------------
    with st.form("trade_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_choice = st.selectbox("Select Stock / Asset", POPULAR_TICKERS, index=0)
            if ticker_choice == "Other (type manually)":
                ticker_manual = st.text_input(
                    "Enter ticker symbol",
                    placeholder="e.g. NVDA, GME, BTC-USD",
                )
                ticker = ticker_manual.upper().strip() if ticker_manual else ""
            else:
                ticker = ticker_choice

            asset_type = st.selectbox("Asset Type", ["Stock", "ETF", "Crypto", "Forex"])

        with col2:
            investment_amount = st.number_input(
                "Investment Amount ($)",
                min_value=100.0,
                max_value=1_000_000.0,
                value=1_000.0,
                step=100.0,
            )
            trade_frequency = st.selectbox(
                "How often do you want to trade?",
                ["Daily (Active Trader)", "Weekly (Regular Investor)", "Monthly (Passive Investor)"],
                index=1,
            )
            recurring_amount = st.number_input(
                "Recurring Income / Savings ($)",
                min_value=0.0,
                max_value=100_000.0,
                value=0.0,
                step=50.0,
                help="Regular amount you add to investments each period (e.g. monthly salary savings)",
            )
            recurring_period = st.selectbox(
                "Recurring Period",
                ["weekly", "monthly"],
                index=1,
            )

        with col3:
            risk_tolerance = st.selectbox(
                "Risk Tolerance", ["Low", "Medium", "High"], index=1
            )
            horizon = st.selectbox(
                "Time Horizon",
                ["Intraday", "Short-term (1-4 weeks)", "Medium-term (1-3 months)", "Long-term (>3 months)"],
                index=1,
            )

        extra_context = st.text_area(
            "Additional Context (optional)",
            placeholder=(
                "Tell the AI about your situation — e.g. 'I already have $500 worth of AMZN', "
                "'I heard there is an earnings report next week', 'I am saving for a house in 2 years'..."
            ),
            height=100,
        )
        submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)

    if not submitted:
        _render_placeholder()
        return

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    ticker_ok, ticker_msg = validate_ticker(ticker)
    amount_ok, amount_msg = validate_amount(investment_amount)
    if not ticker_ok:
        st.error(f"Invalid ticker: {ticker_msg}")
        return
    if not amount_ok:
        st.error(f"Invalid amount: {amount_msg}")
        return

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    with st.spinner(f"Analysing {ticker}..."):
        tech_agent = TechnicalAgent()
        sentiment_agent = SentimentAgent(sentiment_analyzer)
        risk_agent = RiskAgent()

        technical_signal = tech_agent.analyze(ticker)
        sentiment_signal = sentiment_agent.analyze(ticker)
        risk_plan = risk_agent.analyze(ticker, investment_amount, risk_tolerance)
        recommendation = manager_agent.generate_recommendation(
            ticker=ticker,
            investment_amount=investment_amount,
            risk_tolerance=risk_tolerance,
            technical_signal=technical_signal,
            sentiment_signal=sentiment_signal,
            risk_plan=risk_plan,
            extra_context=extra_context,
            recurring_amount=recurring_amount,
            recurring_period=recurring_period,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    st.divider()

    # Panel 1: Technical Analysis — plain English
    with st.expander("Technical Analysis", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            display_signal_gauge(
                technical_signal["signal"],
                confidence=abs(technical_signal["score"]),
                score=technical_signal["score"],
            )
        with c2:
            rsi = technical_signal["rsi"]
            if rsi < 30:
                rsi_label = f"The stock looks oversold — this may be a good time to consider buying (RSI: {rsi:.0f})"
                rsi_color = "#00d4aa"
            elif rsi > 70:
                rsi_label = f"The stock looks overpriced right now — be cautious before buying more (RSI: {rsi:.0f})"
                rsi_color = "#ff4b4b"
            else:
                rsi_label = f"The stock is trading in a normal range — no extreme signals (RSI: {rsi:.0f})"
                rsi_color = "#ffa500"
            st.markdown(
                f'<p style="color:{rsi_color}; font-size:1.1em; font-weight:500;">{rsi_label}</p>',
                unsafe_allow_html=True,
            )

            sma_trend = technical_signal.get("sma_trend", "neutral")
            trend_text = {
                "bullish": "Price trend is pointing upward",
                "bearish": "Price trend is pointing downward",
                "neutral": "No clear price trend right now",
            }.get(sma_trend, "No clear price trend right now")
            st.markdown(f"**Trend:** {trend_text}")

            with st.expander("Technical details (advanced)", expanded=False):
                st.metric("RSI (14)", f"{rsi:.1f}")
                st.metric("Tech Score", f"{technical_signal['score']:+.3f}")
                st.metric("Volatility", f"{technical_signal['volatility']:.1f}%")
                if technical_signal.get("sma_short"):
                    st.metric("SMA 20", f"${technical_signal['sma_short']:,.2f}")
                if technical_signal.get("sma_long"):
                    st.metric("SMA 50", f"${technical_signal['sma_long']:,.2f}")

    # Panel 2: News Sentiment — plain English
    with st.expander("News Sentiment", expanded=True):
        score = sentiment_signal["score"]
        if score > 0.1:
            sent_text = "News about this stock is mostly positive"
            sent_color = "#00d4aa"
        elif score < -0.1:
            sent_text = "News about this stock is mostly negative"
            sent_color = "#ff4b4b"
        else:
            sent_text = "News about this stock is mixed"
            sent_color = "#ffa500"

        st.markdown(
            f'<p style="color:{sent_color}; font-size:1.2em; font-weight:bold;">{sent_text}</p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Based on {sentiment_signal['news_count']} recent headlines")

        headlines = sentiment_signal.get("live_headlines", [])
        if headlines:
            for h in headlines[:3]:
                st.markdown(f"- {h}")
        else:
            st.info("No recent headlines found.")

        with st.expander("Sentiment details (advanced)", expanded=False):
            st.metric("Sentiment Score", f"{score:+.3f}")
            st.metric("Trend", sentiment_signal.get("trend", "stable").title())
            st.caption(sentiment_signal.get("historical_context", ""))
            scored = [{"text": h, "label": "neutral", "score": 0.0} for h in headlines]
            if scored:
                display_news_sentiment(scored)

    # Panel 3: AI Recommendation
    with st.expander("AI Recommendation", expanded=True):
        decision = recommendation.get("decision", "HOLD")
        confidence = recommendation.get("confidence", 0.5)

        r1, r2 = st.columns([1, 2])
        with r1:
            display_verdict_badge(decision, confidence)
            st.markdown("")
            display_signal_gauge(decision, confidence)

        with r2:
            st.subheader("What does this mean?")
            st.write(recommendation.get("explanation", ""))

            st.subheader("Things to watch out for")
            for risk in recommendation.get("key_risks", []):
                st.markdown(f"- {risk}")

            st.subheader("Suggested next steps")
            for step in recommendation.get("next_steps", []):
                st.markdown(f"1. {step}")

    # ------------------------------------------------------------------
    # Personalized AI Advice (only when extra_context provided)
    # ------------------------------------------------------------------
    if extra_context and extra_context.strip():
        with st.expander("Personalized Advice Based on Your Situation", expanded=True):
            with st.spinner("Generating personalized advice..."):
                context_advice = manager_agent.generate_context_advice(
                    ticker=ticker,
                    investment_amount=investment_amount,
                    risk_tolerance=risk_tolerance,
                    horizon=horizon,
                    technical_signal=technical_signal,
                    sentiment_signal=sentiment_signal,
                    extra_context=extra_context,
                    recurring_amount=recurring_amount,
                    recurring_period=recurring_period,
                )
            if context_advice:
                st.markdown(
                    f'<div style="background:#1a2a1a; border-left:5px solid #00d4aa; '
                    f'padding:14px 18px; border-radius:0 10px 10px 0; margin-bottom:12px;">'
                    f'<span style="color:#00d4aa; font-weight:bold;">AI Advice for Your Situation</span><br>'
                    f'<span style="color:#e0e0e0;">{context_advice.get("personal_advice", "")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                ca1, ca2 = st.columns(2)
                with ca1:
                    st.markdown("**Risks specific to your situation**")
                    for r in context_advice.get("context_risks", []):
                        st.markdown(f"- {r}")
                with ca2:
                    st.markdown("**Actions tailored to your situation**")
                    for i, a in enumerate(context_advice.get("context_actions", []), 1):
                        st.markdown(f"{i}. {a}")
            else:
                st.info("Personalized advice unavailable — Gemini API may not be configured.")

    # ------------------------------------------------------------------
    # Recurring Income Summary (when set)
    # ------------------------------------------------------------------
    if recurring_amount > 0:
        st.divider()
        _render_recurring_summary(recurring_amount, recurring_period, investment_amount, horizon)

    # ------------------------------------------------------------------
    # Trade Opportunities
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Today's Trade Opportunities")
    st.caption("Specific actions you could take to grow your portfolio — in plain English.")

    with st.spinner("Finding opportunities..."):
        opportunities = manager_agent.generate_trade_opportunities(
            ticker=ticker,
            investment_amount=investment_amount,
            trade_frequency=trade_frequency,
            horizon=horizon,
            technical_signal=technical_signal,
            sentiment_signal=sentiment_signal,
            extra_context=extra_context,
            recurring_amount=recurring_amount,
            recurring_period=recurring_period,
        )
    _render_trade_opportunities(opportunities, investment_amount)

    # ------------------------------------------------------------------
    # Stock Discovery
    # ------------------------------------------------------------------
    st.subheader("Other Stocks Worth Watching")
    st.caption(
        "Consider spreading your investment across multiple assets. "
        "These suggestions could complement your current position."
    )

    with st.spinner("Discovering growth opportunities..."):
        suggestions = manager_agent.suggest_new_stocks(
            current_ticker=ticker,
            investment_amount=investment_amount,
            technical_signal=technical_signal,
            sentiment_signal=sentiment_signal,
        )
    _render_stock_suggestions(suggestions)


# ---------------------------------------------------------------------------
# Renderer helpers
# ---------------------------------------------------------------------------

def _render_trade_opportunities(opportunities: list, total_capital: float = 0.0) -> None:
    """Render actionable trade cards with colored action badges and money allocations."""
    if not opportunities:
        st.info("No specific trade opportunities identified at this time.")
        return

    color_map = {"BUY": "#00d4aa", "SELL": "#ff4b4b", "HOLD": "#ffa500"}
    bg_map = {"BUY": "#003d30", "SELL": "#3d0000", "HOLD": "#3d2000"}

    # Compute suggested allocation: split capital proportionally among BUY actions
    buy_opps = [o for o in opportunities if o.get("action", "HOLD").upper() == "BUY"]
    n_buys = len(buy_opps) or 1
    per_buy = total_capital / n_buys if total_capital > 0 else 0.0

    for opp in opportunities:
        action = opp.get("action", "HOLD").upper()
        color = color_map.get(action, "#888888")
        bg = bg_map.get(action, "#1a1a2e")
        stock = opp.get("stock", "")
        reason = opp.get("reason", "")
        price_range = opp.get("price_range", "current market price")

        # Money suggestion line
        if action == "BUY" and per_buy > 0:
            low = round(per_buy * 0.8, 2)
            high = round(per_buy * 1.2, 2)
            money_line = f"Suggested amount: <b>${low:,.0f} – ${high:,.0f}</b>"
        elif action == "SELL":
            money_line = "Close or reduce your current position"
        else:
            money_line = "No new money needed — monitor and wait"

        st.markdown(
            f"""<div style="
                border-left: 5px solid {color};
                padding: 12px 16px;
                margin: 8px 0;
                background: {bg};
                border-radius: 0 10px 10px 0;
            ">
                <span style="color:{color}; font-weight:bold; font-size:1.1em;">{action}: {stock}</span><br>
                <span style="color:#e0e0e0; font-size:0.95em;">{reason}</span><br>
                <span style="color:#aaaaaa; font-size:0.82em;">Price range: {price_range} &nbsp;|&nbsp; {money_line}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_stock_suggestions(suggestions: list) -> None:
    """Render stock discovery cards in a column grid."""
    if not suggestions:
        st.info("No additional stock suggestions available right now.")
        return

    cols = st.columns(min(len(suggestions), 3))
    for col, s in zip(cols, suggestions):
        with col:
            st.markdown(
                f"""<div style="
                    background: #1a1a2e;
                    border: 1px solid #00d4aa55;
                    border-radius: 12px;
                    padding: 20px 16px;
                    text-align: center;
                    min-height: 110px;
                ">
                    <div style="color:#00d4aa; font-size:1.5em; font-weight:bold; letter-spacing:2px;">
                        {s.get('ticker', '')}
                    </div>
                    <div style="color:#e0e0e0; font-size:0.88em; margin-top:10px; line-height:1.4;">
                        {s.get('reason', '')}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_recurring_summary(
    recurring_amount: float,
    recurring_period: str,
    investment_amount: float,
    horizon: str,
) -> None:
    """Show a projected growth summary for recurring contributions."""
    import plotly.graph_objects as go

    st.subheader("Recurring Savings Projection")
    st.caption(
        f"If you invest **${recurring_amount:,.0f} {recurring_period}** on top of your "
        f"initial **${investment_amount:,.0f}**, here's how your total invested capital grows over time."
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
    running = investment_amount
    for i in range(total_periods + 1):
        cumulative.append({"period": i, "total_invested": round(running, 2)})
        running += recurring_amount

    period_labels = [f"{'W' if recurring_period == 'weekly' else 'M'}{c['period']}" for c in cumulative]
    values = [c["total_invested"] for c in cumulative]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=period_labels, y=values,
        mode="lines+markers",
        name="Total Invested",
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

    final_invested = investment_amount + recurring_amount * total_periods
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Investment", f"${investment_amount:,.0f}")
    col2.metric(f"Total Added ({recurring_period.title()})", f"${recurring_amount * total_periods:,.0f}")
    col3.metric("Total Invested by End", f"${final_invested:,.0f}")


def _render_placeholder() -> None:
    st.info(
        "Choose a stock from the dropdown above and click **Analyze** to get your personalised "
        "AI-powered recommendation in plain English."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Is it a good time to buy?")
        st.markdown("We check RSI, price trends, and momentum")
    with c2:
        st.markdown("#### What is the news saying?")
        st.markdown("We scan recent headlines and measure sentiment")
    with c3:
        st.markdown("#### What should I do?")
        st.markdown("Gemini AI gives you a plain-English recommendation")
