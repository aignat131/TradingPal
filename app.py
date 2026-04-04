"""
TradingPal-AI — Main Streamlit Application.

Entry point: streamlit run app.py

Architecture:
  Sidebar        → navigation + API status
  Trade Advisor  → live recommendation (FinBERT + NewsAPI + Gemini)
  Backtest Viewer→ historical simulation (pre-processed FinSen parquet + yfinance)

FinSen data is NOT loaded here. The Backtest page loads its own
SentimentCache from data_cache/finsen_daily_sentiment.parquet
(built once by: python utils/preprocess_finsen.py).
"""
import logging

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TradingPal-AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resource initialisation
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FinBERT model (first run may take a minute)...")
def _load_sentiment_analyzer():
    from analysis.sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer()


@st.cache_resource
def _load_manager_agent():
    from agents.manager_agent import ManagerAgent
    return ManagerAgent()


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --accent: #00d4aa;
                --danger: #ff4b4b;
                --warning: #ffa500;
            }
            section[data-testid="stSidebar"] > div:first-child {
                background: #0e1117;
            }
            [data-testid="stMetricDelta"] svg { display: none; }
            [data-testid="stExpander"] summary {
                font-weight: 600;
                font-size: 1.05em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(manager_agent) -> str:
    import config

    with st.sidebar:
        st.markdown("## 📈 TradingPal-AI")
        st.markdown("*Multi-agent AI Investment Advisor*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["Trade Advisor", "Backtest Viewer"],
            label_visibility="collapsed",
        )
        st.divider()

        st.markdown("### System Status")
        _status("Gemini API", bool(config.GEMINI_API_KEY))
        _status("NewsAPI", bool(config.NEWS_API_KEY))
        _status("FinBERT Model", manager_agent._gemini_available is not None)

        import os
        cache_ok = os.path.isfile(
            os.path.join("data_cache", "finsen_daily_sentiment.parquet")
        )
        _status("FinSen Cache (backtest)", cache_ok)

        st.divider()
        st.markdown("### About")
        st.markdown(
            "TradingPal-AI combines FinBERT sentiment analysis, "
            "technical indicators (RSI/SMA), and Google Gemini "
            "for personalised investment recommendations."
        )
        if not cache_ok:
            st.info(
                "Run `python utils/preprocess_finsen.py` "
                "to enable sentiment-weighted backtesting."
            )
        st.markdown("**Disclaimer:** Educational use only. Not financial advice.")

    return page


def _status(label: str, ok: bool) -> None:
    icon = "🟢" if ok else "🔴"
    msg = "Ready" if ok else "Not configured"
    st.markdown(f"{icon} **{label}** — {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()

    try:
        sentiment_analyzer = _load_sentiment_analyzer()
    except Exception as exc:
        logger.error("Sentiment analyzer failed: %s", exc)
        from analysis.sentiment_analyzer import SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()

    try:
        manager_agent = _load_manager_agent()
    except Exception as exc:
        logger.error("Manager agent failed: %s", exc)
        from agents.manager_agent import ManagerAgent
        manager_agent = ManagerAgent()

    page = _render_sidebar(manager_agent)

    if page == "Trade Advisor":
        from ui.pages.trade_advisor import render as render_trade
        render_trade(sentiment_analyzer, manager_agent)
    else:
        from ui.pages.backtest_viewer import render as render_backtest
        render_backtest(manager_agent=manager_agent)


def _inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _inside_streamlit():
    main()
elif __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:]
        ).returncode
    )
