"""
Reusable Streamlit / Plotly UI components for TradingPal-AI.
"""
from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Signal Gauge
# ---------------------------------------------------------------------------

def display_signal_gauge(signal: str, confidence: float, score: float = None) -> None:
    """
    Render a Plotly gauge showing the trading signal and confidence.

    *score* (optional): raw technical score in [-1, 1]. When provided, drives the
    gauge needle continuously so NEUTRAL values still show where in the range
    the asset sits (e.g. score=-0.15 → gauge at ~42, not stuck at 50).
    """
    color_map = {
        "BUY": "#00d4aa",
        "SELL": "#ff4b4b",
        "HOLD": "#ffa500",
        "NEUTRAL": "#888888",
    }

    if score is not None:
        # Continuous mapping: score -1 → 0, score 0 → 50, score +1 → 100
        value = round((score + 1) * 50, 1)
        value = max(5.0, min(95.0, value))  # keep needle visible
    else:
        # Fallback: use confidence to widen from centre
        if signal.upper() == "BUY":
            value = round(50 + confidence * 45, 1)
        elif signal.upper() == "SELL":
            value = round(50 - confidence * 45, 1)
        else:
            value = 50.0

    # Derive colour from final value even if signal is NEUTRAL
    if value >= 62:
        color = "#00d4aa"
    elif value <= 38:
        color = "#ff4b4b"
    else:
        color = color_map.get(signal.upper(), "#888888")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={"text": f"Signal: <b>{signal}</b>", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(255,75,75,0.2)"},
                    {"range": [30, 70], "color": "rgba(136,136,136,0.2)"},
                    {"range": [70, 100], "color": "rgba(0,212,170,0.2)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
            number={"suffix": "%", "font": {"size": 24}},
            delta={"reference": 50, "valueformat": ".0f"},
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Risk Meter
# ---------------------------------------------------------------------------

def display_risk_meter(risk_score: float) -> None:
    """
    Render a risk level meter.
    *risk_score* is in [0, 1] where 1 = maximum risk.
    """
    pct = round(risk_score * 100, 1)
    if risk_score < 0.33:
        label, color = "LOW RISK", "#00d4aa"
    elif risk_score < 0.66:
        label, color = "MEDIUM RISK", "#ffa500"
    else:
        label, color = "HIGH RISK", "#ff4b4b"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": label, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "rgba(0,212,170,0.13)"},
                    {"range": [33, 66], "color": "rgba(255,165,0,0.13)"},
                    {"range": [66, 100], "color": "rgba(255,75,75,0.13)"},
                ],
            },
            number={"suffix": "%"},
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Position Card
# ---------------------------------------------------------------------------

def display_position_card(position_data: Dict) -> None:
    """Render a risk-plan card with entry, stop-loss, and take-profit levels."""
    col1, col2, col3 = st.columns(3)
    entry = position_data.get("entry_price", 0)
    stop = position_data.get("stop_loss", 0)
    tp1 = position_data.get("take_profit_1", 0)
    tp2 = position_data.get("take_profit_2", 0)
    rrr = position_data.get("risk_reward_ratio", 0)
    max_loss = position_data.get("max_loss", 0)
    position_size = position_data.get("position_size", 0)

    with col1:
        st.metric("Entry Price", f"${entry:,.2f}")
        st.metric("Position Size", f"{position_size:.4f} shares")

    with col2:
        st.metric(
            "Stop-Loss",
            f"${stop:,.2f}",
            delta=f"{(stop - entry) / entry * 100:.1f}%" if entry else None,
            delta_color="inverse",
        )
        st.metric("Max Loss", f"${max_loss:,.2f}", delta_color="inverse")

    with col3:
        st.metric(
            "Take-Profit 1",
            f"${tp1:,.2f}",
            delta=f"+{(tp1 - entry) / entry * 100:.1f}%" if entry else None,
        )
        st.metric(
            "Take-Profit 2",
            f"${tp2:,.2f}",
            delta=f"+{(tp2 - entry) / entry * 100:.1f}%" if entry else None,
        )

    st.caption(f"Risk/Reward Ratio: **{rrr:.2f}** | Risk %: {position_data.get('risk_percent', 0):.1f}%")


# ---------------------------------------------------------------------------
# News Sentiment Display
# ---------------------------------------------------------------------------

def display_news_sentiment(news_items: List[Dict]) -> None:
    """Render a list of news items with colour-coded sentiment badges."""
    if not news_items:
        st.info("No news items to display.")
        return

    badge_colors = {
        "positive": "#00d4aa",
        "negative": "#ff4b4b",
        "neutral": "#888888",
    }
    for item in news_items:
        label = item.get("label", "neutral").lower()
        score = item.get("score", 0.0)
        text = item.get("text", item.get("title", ""))
        color = badge_colors.get(label, "#888888")
        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {color};
                padding: 6px 12px;
                margin: 4px 0;
                background: #1a1a2e;
                border-radius: 0 6px 6px 0;
            ">
                <span style="color:{color}; font-weight:bold; font-size:0.8em;">
                    {label.upper()} ({score:+.2f})
                </span><br>
                <span style="color:#e0e0e0; font-size:0.9em;">{text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Verdict Badge
# ---------------------------------------------------------------------------

def display_verdict_badge(decision: str, confidence: float) -> None:
    """Render a large coloured verdict badge."""
    colors = {
        "BUY": ("#00d4aa", "#003d30"),
        "SELL": ("#ff4b4b", "#3d0000"),
        "HOLD": ("#ffa500", "#3d2000"),
    }
    fg, bg = colors.get(decision.upper(), ("#ffffff", "#333333"))
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background: {bg};
            border: 2px solid {fg};
            border-radius: 12px;
            padding: 12px 32px;
            text-align: center;
        ">
            <div style="color:{fg}; font-size: 2.2em; font-weight: 900; letter-spacing: 4px;">
                {decision.upper()}
            </div>
            <div style="color:{fg}; font-size: 0.9em; margin-top: 4px;">
                Confidence: {confidence * 100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
