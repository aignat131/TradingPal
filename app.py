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
    from datetime import datetime, timezone, time as dt_time
    import os

    with st.sidebar:
        st.markdown("## 📈 TradingPal-AI")
        st.markdown("*Multi-agent AI Investment Advisor*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["AI Trade Advisor", "Backtest Viewer", "Math Logic"],
            label_visibility="collapsed",
        )
        st.divider()

        # ── Market Clock ──────────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        # NYSE hours: 09:30–16:00 ET (UTC-4 in summer, UTC-5 in winter)
        # Simple approximation using UTC offset
        et_offset = -4  # EDT (close enough year-round for display)
        now_et = now_utc.hour + et_offset
        market_open_et  = 9.5    # 9:30
        market_close_et = 16.0   # 16:00
        is_weekday = now_utc.weekday() < 5
        market_hour_et = (now_utc.hour + et_offset) + now_utc.minute / 60
        market_is_open = is_weekday and market_open_et <= market_hour_et < market_close_et

        market_color  = "#00d4aa" if market_is_open else "#ff4b4b"
        market_status = "OPEN" if market_is_open else "CLOSED"
        market_dot    = "🟢" if market_is_open else "🔴"

        if market_is_open:
            mins_left = int((market_close_et - market_hour_et) * 60)
            sub = f"Closes in {mins_left // 60}h {mins_left % 60}m"
        elif is_weekday and market_hour_et < market_open_et:
            mins_to_open = int((market_open_et - market_hour_et) * 60)
            sub = f"Opens in {mins_to_open // 60}h {mins_to_open % 60}m"
        else:
            sub = "Opens Mon 9:30 AM ET"

        st.markdown(
            f"""
            <div style="background:#111;border-radius:12px;padding:14px 16px;margin-bottom:4px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="color:#888;font-size:0.7em;text-transform:uppercase;letter-spacing:1px;">NYSE Market</div>
                        <div style="color:{market_color};font-size:1.4em;font-weight:800;letter-spacing:2px;margin-top:2px;">
                            {market_dot} {market_status}
                        </div>
                        <div style="color:#666;font-size:0.72em;margin-top:3px;">{sub}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#888;font-size:0.7em;text-transform:uppercase;letter-spacing:1px;">Local time</div>
                        <div style="color:#e0e0e0;font-size:1em;font-weight:600;margin-top:2px;">
                            {now_utc.strftime("%H:%M UTC")}
                        </div>
                        <div style="color:#666;font-size:0.7em;margin-top:3px;">
                            {now_utc.strftime("%d %b %Y")}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Watchlist Stats ───────────────────────────────────────────
        n_stocks = len([t for t in config.STOCK_WATCHLIST
                        if t != "Other (type manually)" and "-USD" not in t])
        n_crypto = len([t for t in config.STOCK_WATCHLIST if "-USD" in t])

        st.markdown(
            f"""
            <div style="background:#111;border-radius:12px;padding:12px 16px;margin-top:8px;">
                <div style="color:#888;font-size:0.7em;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:8px;">Tracking</div>
                <div style="display:flex;gap:12px;">
                    <div style="flex:1;background:#0e1117;border-radius:8px;padding:8px;text-align:center;">
                        <div style="color:#00d4aa;font-size:1.4em;font-weight:800;">{n_stocks}</div>
                        <div style="color:#666;font-size:0.68em;margin-top:2px;">Stocks & ETFs</div>
                    </div>
                    <div style="flex:1;background:#0e1117;border-radius:8px;padding:8px;text-align:center;">
                        <div style="color:#f7931a;font-size:1.4em;font-weight:800;">{n_crypto}</div>
                        <div style="color:#666;font-size:0.68em;margin-top:2px;">Crypto</div>
                    </div>
                    <div style="flex:1;background:#0e1117;border-radius:8px;padding:8px;text-align:center;">
                        <div style="color:#a78bfa;font-size:1.4em;font-weight:800;">AI</div>
                        <div style="color:#666;font-size:0.68em;margin-top:2px;">Powered</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── About ─────────────────────────────────────────────────────
        st.markdown("### About")
        st.markdown(
            "TradingPal-AI combines **FinBERT** sentiment analysis, "
            "**RSI/SMA** technical indicators, and **Google Gemini** "
            "for personalised investment recommendations."
        )
        cache_ok = os.path.isfile(
            os.path.join("data_cache", "finsen_daily_sentiment.parquet")
        )
        if not cache_ok:
            st.info(
                "Run `python utils/preprocess_finsen.py` "
                "to enable sentiment-weighted backtesting."
            )
        st.markdown(
            "<div style='color:#555;font-size:0.75em;margin-top:8px;'>"
            "Educational use only. Not financial advice.</div>",
            unsafe_allow_html=True,
        )

    return page


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

    if page == "AI Trade Advisor":
        from ui.pages.trade_advisor import render as render_trade
        render_trade(sentiment_analyzer, manager_agent)
    elif page == "Backtest Viewer":
        from ui.pages.backtest_viewer import render as render_backtest
        render_backtest(manager_agent=manager_agent)
    else:  # Math Logic
        from ui.pages.rule_advisor import render as render_rules
        render_rules()


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
