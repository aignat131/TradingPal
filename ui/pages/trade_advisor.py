"""
Trade Advisor Page — main recommendation interface.

Sections:
1. Input form (ticker, amount, risk tolerance)
2. Technical Analysis panel
3. Sentiment Analysis panel
4. AI Recommendation panel (Gemini)
"""
import logging

import streamlit as st

from agents.manager_agent import ManagerAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from ui.components import (
    display_news_sentiment,
    display_position_card,
    display_signal_gauge,
    display_verdict_badge,
)
from utils.validators import validate_amount, validate_ticker

logger = logging.getLogger(__name__)


def render(
    sentiment_analyzer,
    manager_agent: ManagerAgent,
) -> None:
    """Render the Trade Advisor page."""
    st.title("TradingPal-AI — Trade Advisor")
    st.markdown(
        "Enter your trade details below and click **Analyze Trade** to receive "
        "an AI-powered recommendation backed by technical indicators and sentiment analysis."
    )

    # ------------------------------------------------------------------
    # Input Form
    # ------------------------------------------------------------------
    with st.form("trade_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input(
                "Ticker Symbol",
                value="AAPL",
                placeholder="e.g. AAPL, TSLA, BTC-USD",
            ).upper().strip()
            asset_type = st.selectbox("Asset Type", ["Stock", "ETF", "Crypto", "Forex"])

        with col2:
            investment_amount = st.number_input(
                "Investment Amount ($)",
                min_value=100.0,
                max_value=1_000_000.0,
                value=1_000.0,
                step=100.0,
            )
            direction = st.selectbox("Direction", ["Buy", "Sell"])

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
            placeholder="e.g. Earnings report next week, sector rotation thesis...",
            height=80,
        )
        submitted = st.form_submit_button("Analyze Trade", type="primary", use_container_width=True)

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
        )

    # ------------------------------------------------------------------
    # Results — 3 panels
    # ------------------------------------------------------------------
    st.divider()

    # Panel 1: Technical Analysis
    with st.expander("Technical Analysis", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            display_signal_gauge(
                technical_signal["signal"],
                confidence=abs(technical_signal["score"]),
            )
        with c2:
            st.metric("RSI (14)", f"{technical_signal['rsi']:.1f}")
            st.metric("SMA Trend", technical_signal["sma_trend"].title())
            st.metric("Volatility", f"{technical_signal['volatility']:.1f}%")
        with c3:
            st.metric("Tech Score", f"{technical_signal['score']:+.3f}")
            st.metric("SMA 20", f"${technical_signal['sma_short']:,.2f}" if technical_signal['sma_short'] else "N/A")
            st.metric("SMA 50", f"${technical_signal['sma_long']:,.2f}" if technical_signal['sma_long'] else "N/A")

    # Panel 2: Sentiment Analysis
    with st.expander("Sentiment Analysis", expanded=True):
        s1, s2 = st.columns([1, 2])
        with s1:
            score = sentiment_signal["score"]
            if score > 0.1:
                sentiment_label = "Positive"
            elif score < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            st.metric("Aggregate Sentiment", sentiment_label)
            st.metric("Sentiment Score", f"{score:+.3f}")
            st.metric("Trend", sentiment_signal["trend"].title())
            st.metric("News Analysed", sentiment_signal["news_count"])
            st.caption(sentiment_signal["historical_context"])

        with s2:
            headlines = sentiment_signal.get("live_headlines", [])
            if headlines:
                # Show with neutral label since we have only text here
                items = [{"text": h, "label": "neutral", "score": 0.0} for h in headlines]
                display_news_sentiment(items)
            else:
                st.info("No live news available.")

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
            st.subheader("Explanation")
            st.write(recommendation.get("explanation", ""))

            st.subheader("Key Risks")
            for risk in recommendation.get("key_risks", []):
                st.markdown(f"- {risk}")

            st.subheader("Next Steps")
            for step in recommendation.get("next_steps", []):
                st.markdown(f"1. {step}")

    # Risk Plan
    with st.expander("Risk Management Plan", expanded=True):
        display_position_card(risk_plan)


def _render_placeholder() -> None:
    st.info(
        "Fill in the form above and click **Analyze Trade** to get your personalised "
        "AI-powered recommendation."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Technical Signals")
        st.markdown("RSI · SMA · Volatility")
    with c2:
        st.markdown("#### Sentiment Signals")
        st.markdown("FinBERT · NewsAPI · FinSen")
    with c3:
        st.markdown("#### AI Decision")
        st.markdown("Gemini · BUY / HOLD / SELL")
