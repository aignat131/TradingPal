"""
Trade Advisor Page — Instagram-style scrollable card feed.

Flow:
1. User enters their profile (budget, risk, horizon)
2. App analyses every asset in the watchlist
3. Top 10 opportunities are displayed as full-screen snap-scrolling cards
   Each card shows: signal badge · live chart + projected candles · reasoning · allocation
"""
import base64
import html as _html
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.utils import PlotlyJSONEncoder

from agents.manager_agent import ManagerAgent
from agents.technical_agent import TechnicalAgent
import config

logger = logging.getLogger(__name__)

WATCHLIST = [t for t in config.STOCK_WATCHLIST if t != "Other (type manually)"]

_COLOR = {"BUY": "#00d4aa", "SELL": "#ff4b4b"}
_BG    = {"BUY": "#00120d",  "SELL": "#120000"}

# Height of each snap card in the feed (px)
_CARD_H = 540


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price_data(ticker: str) -> dict | None:
    """Fetch last 35 trading days of OHLCV via yfinance. Cached 5 min."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="2mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.tail(30)
        return {
            "dates": [str(d.date()) for d in df.index],
            "open":  [round(float(v), 4) for v in df["Open"]],
            "high":  [round(float(v), 4) for v in df["High"]],
            "low":   [round(float(v), 4) for v in df["Low"]],
            "close": [round(float(v), 4) for v in df["Close"]],
        }
    except Exception as exc:
        logger.debug("Price fetch failed for %s: %s", ticker, exc)
        return None


def _project_candles(chart_data: dict, signal: str, n: int = 8) -> dict:
    """Generate n simulated future candles with directional bias."""
    closes = chart_data["close"]
    last_close = float(closes[-1])
    last_date = pd.Timestamp(chart_data["dates"][-1])

    if len(closes) >= 6:
        log_ret = np.diff(np.log(np.maximum(closes[-15:], 0.01)))
        vol = float(np.std(log_ret)) * last_close
    else:
        vol = last_close * 0.015
    vol = max(vol, last_close * 0.005)

    direction = 1.0 if signal == "BUY" else -1.0
    drift = direction * vol * 0.45

    # Use a fixed seed so projections are stable across rerenders
    rng = np.random.default_rng(seed=int(abs(hash(chart_data["dates"][-1])) % 2**32))

    proj: dict = {"dates": [], "open": [], "high": [], "low": [], "close": []}
    current = last_close
    day = last_date
    added = 0
    while added < n:
        day += pd.Timedelta(days=1)
        if day.weekday() >= 5:
            continue
        o = current
        c = current + drift + float(rng.normal(0, vol * 0.25))
        c = max(c, 0.01)
        h = max(o, c) + abs(float(rng.normal(0, vol * 0.18)))
        l = min(o, c) - abs(float(rng.normal(0, vol * 0.18)))
        proj["dates"].append(str(day.date()))
        proj["open"].append(round(o, 4))
        proj["high"].append(round(h, 4))
        proj["low"].append(round(l, 4))
        proj["close"].append(round(c, 4))
        current = c
        added += 1
    return proj


def _build_chart_json(chart_data: dict, proj_data: dict | None, signal: str) -> str:
    """Return a JSON string {data, layout} for Plotly.newPlot."""
    color = _COLOR.get(signal, "#888888")
    fig = go.Figure()

    # Historical candles
    fig.add_trace(go.Candlestick(
        x=chart_data["dates"],
        open=chart_data["open"],
        high=chart_data["high"],
        low=chart_data["low"],
        close=chart_data["close"],
        name="Price",
        increasing=dict(line=dict(color="#00d4aa", width=1.2), fillcolor="#00d4aa"),
        decreasing=dict(line=dict(color="#ff4b4b", width=1.2), fillcolor="#ff4b4b"),
    ))

    # Projected candles (lighter opacity via separate trace color)
    if proj_data and proj_data["dates"]:
        proj_inc = color if signal == "BUY" else "#ff8888"
        proj_dec = "#88ffdd" if signal == "BUY" else color
        fig.add_trace(go.Candlestick(
            x=proj_data["dates"],
            open=proj_data["open"],
            high=proj_data["high"],
            low=proj_data["low"],
            close=proj_data["close"],
            name="Projected",
            increasing=dict(line=dict(color=proj_inc, width=1), fillcolor=proj_inc),
            decreasing=dict(line=dict(color=proj_dec, width=1), fillcolor=proj_dec),
            opacity=0.45,
        ))
        # Dashed separator
        fig.add_shape(
            type="line",
            x0=chart_data["dates"][-1], x1=chart_data["dates"][-1],
            y0=0, y1=1, yref="paper",
            line=dict(color="rgba(255,255,255,34)", width=1, dash="dot"),
        )
        # "PROJECTED" label
        mid_idx = len(proj_data["dates"]) // 2
        all_prices = proj_data["high"] + proj_data["low"] + chart_data["high"] + chart_data["low"]
        y_top = max(all_prices) * 1.005
        fig.add_annotation(
            x=proj_data["dates"][mid_idx], y=y_top,
            text="PROJECTED", showarrow=False,
            font=dict(color=color, size=8), opacity=0.65,
        )

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(
            showgrid=False, rangeslider=dict(visible=False),
            tickfont=dict(color="#666", size=8), showline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1c2c1c" if signal == "BUY" else "#2c1c1c",
            tickfont=dict(color="#666", size=8), side="right",
        ),
        showlegend=False,
        font=dict(color="#e0e0e0"),
    )

    payload = {"data": fig.data, "layout": fig.layout}
    return json.dumps(payload, cls=PlotlyJSONEncoder)


# ---------------------------------------------------------------------------
# Feed HTML builder
# ---------------------------------------------------------------------------

def _build_feed_html(portfolio: list, budget: float, price_cache: dict) -> str:
    """Build the full HTML for the snap-scroll feed."""

    cards_html = ""
    chart_js_blocks = ""

    for rank, item in enumerate(portfolio, start=1):
        ticker    = item.get("ticker", "")
        signal    = item.get("signal", "BUY").upper()
        if signal not in ("BUY", "SELL"):
            signal = "BUY"
        score     = max(0.0, min(1.0, item.get("score", 0.5)))
        reasoning = _html.escape(item.get("reasoning", ""))
        alloc_pct = item.get("suggested_allocation", 0.0)
        alloc_amt = budget * alloc_pct / 100.0
        news_url  = f"https://finance.yahoo.com/quote/{ticker}/news/"

        color = _COLOR.get(signal, "#888888")
        bg    = _BG.get(signal, "#0e1117")
        bar_w = int(score * 100)
        n_stars = min(5, max(1, round(score * 5)))
        stars_html = (
            f'<span style="color:{color}">{"★" * n_stars}</span>'
            f'<span style="color:#333">{"☆" * (5 - n_stars)}</span>'
        )

        chart_data = price_cache.get(ticker)
        chart_id   = f"chart_{rank}"

        if chart_data:
            proj_data  = _project_candles(chart_data, signal)
            chart_json = _build_chart_json(chart_data, proj_data, signal)
            chart_block = f'<div id="{chart_id}" class="chart-area"></div>'
            chart_js_blocks += f"""
            (function() {{
                var payload = {chart_json};
                var layout = Object.assign({{}}, payload.layout, {{
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor:  'rgba(0,0,0,0)',
                    height: 295,
                    margin: {{l:4, r:4, t:4, b:4}}
                }});
                Plotly.newPlot('{chart_id}', payload.data, layout,
                    {{displayModeBar: false, responsive: true, staticPlot: false}});
            }})();
            """
        else:
            chart_block = f"""
            <div class="chart-area chart-placeholder" style="border:1px dashed #333;">
                <span style="color:#555;font-size:0.8em;">Chart unavailable</span>
            </div>
            """

        if signal == "BUY" and alloc_pct > 0:
            footer_right = (
                f'<span class="alloc-pill" style="border-color:{color};color:{color};">'
                f'💰 ${alloc_amt:,.0f} &nbsp;·&nbsp; {alloc_pct:.0f}%</span>'
            )
        else:
            footer_right = ""

        cards_html += f"""
        <div class="card" style="background: linear-gradient(160deg, {bg} 0%, #0e1117 60%);">
            <div class="card-stripe" style="background:{color};"></div>

            <div class="card-header">
                <div class="header-left">
                    <span class="rank-num" style="color:#444;">#{rank}</span>
                    <span class="ticker-label" style="color:#fff;">{ticker}</span>
                    {stars_html}
                </div>
                <div class="signal-badge" style="background:{color}18;border:1.5px solid {color};color:{color};">
                    {signal}
                </div>
            </div>

            {chart_block}

            <div class="card-body">
                <div class="score-row">
                    <span class="score-label">Profit potential</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{bar_w}%;
                            background:linear-gradient(90deg,{color}55,{color});"></div>
                    </div>
                    <span class="score-pct" style="color:{color};">{bar_w}%</span>
                </div>
                <p class="reasoning">{reasoning}</p>
                <div class="card-footer-row">
                    <a href="{news_url}" target="_blank" class="news-link"
                       style="color:{color};">Latest news ↗</a>
                    {footer_right}
                </div>
            </div>
        </div>
        """

    total_h = _CARD_H * len(portfolio)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html {{
    height: {_CARD_H}px;
    overflow: hidden;
  }}
  body {{
    height: {_CARD_H}px;
    overflow-y: scroll;
    scroll-snap-type: y mandatory;
    -webkit-overflow-scrolling: touch;
    background: #0e1117;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #e0e0e0;
    scrollbar-width: none;
  }}
  body::-webkit-scrollbar {{ display: none; }}

  /* ── card ── */
  .card {{
    height: {_CARD_H}px;
    scroll-snap-align: start;
    scroll-snap-stop: always;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    border-bottom: 1px solid #1a1a2e;
  }}
  .card-stripe {{
    height: 3px;
    width: 100%;
    flex-shrink: 0;
  }}

  /* ── header ── */
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 16px 6px;
    flex-shrink: 0;
  }}
  .header-left {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .rank-num {{
    font-size: 1.2em;
    font-weight: 700;
  }}
  .ticker-label {{
    font-size: 1.75em;
    font-weight: 900;
    letter-spacing: 2px;
    line-height: 1;
  }}
  .signal-badge {{
    padding: 5px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 1em;
    letter-spacing: 2px;
  }}

  /* ── chart ── */
  .chart-area {{
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
    padding: 0 4px;
  }}
  .chart-placeholder {{
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
  }}

  /* ── body / footer ── */
  .card-body {{
    padding: 6px 16px 12px;
    flex-shrink: 0;
  }}
  .score-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }}
  .score-label {{
    font-size: 0.7em;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
  }}
  .bar-track {{
    flex: 1;
    height: 5px;
    background: #1a1a2e;
    border-radius: 4px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
  }}
  .score-pct {{
    font-size: 0.78em;
    font-weight: 700;
    white-space: nowrap;
  }}
  .reasoning {{
    font-size: 0.83em;
    color: #ccc;
    line-height: 1.5;
    margin-bottom: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .card-footer-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .news-link {{
    font-size: 0.75em;
    text-decoration: none;
    opacity: 0.85;
  }}
  .news-link:hover {{ opacity: 1; text-decoration: underline; }}
  .alloc-pill {{
    font-size: 0.75em;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid;
  }}

  /* ── scroll indicator dots ── */
  #dots {{
    position: fixed;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    gap: 5px;
    z-index: 99;
  }}
  .dot {{
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #333;
    transition: background 0.2s, transform 0.2s;
    cursor: pointer;
  }}
  .dot.active {{
    background: #00d4aa;
    transform: scale(1.5);
  }}
</style>
</head>
<body>

{cards_html}

<!-- Scroll indicator dots -->
<div id="dots">
  {"".join(f'<div class="dot" id="dot_{i}" onclick="scrollToCard({i})"></div>' for i in range(len(portfolio)))}
</div>

<script>
  var totalCards = {len(portfolio)};
  var cardH = {_CARD_H};

  // Init charts
  window.addEventListener('load', function() {{
    {chart_js_blocks}
    updateDots(0);
  }});

  // Dot navigation
  function scrollToCard(idx) {{
    document.body.scrollTo({{top: idx * cardH, behavior: 'smooth'}});
  }}

  // Update active dot on scroll
  function updateDots(idx) {{
    for (var i = 0; i < totalCards; i++) {{
      var d = document.getElementById('dot_' + i);
      if (d) d.className = 'dot' + (i === idx ? ' active' : '');
    }}
  }}

  document.body.addEventListener('scroll', function() {{
    var idx = Math.round(document.body.scrollTop / cardH);
    updateDots(idx);
  }}, {{passive: true}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(
    sentiment_analyzer,
    manager_agent: ManagerAgent,
) -> None:
    """Render the Trade Advisor page."""
    st.title("TradingPal-AI — Portfolio Advisor")
    st.markdown(
        "Fill in your profile and we'll scan **every asset in our database**, "
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
            width='stretch',
        )

    if submitted:
        # New submission — reset dismissed tickers
        st.session_state["dismissed_tickers"] = set()
        st.session_state["last_trade_frequency"] = trade_frequency

    if not submitted and "last_portfolio" not in st.session_state:
        _render_placeholder()
        return

    # Restore values from session state if not re-submitted
    if not submitted:
        budget           = st.session_state.get("last_budget", budget)
        risk             = st.session_state.get("last_risk", risk)
        horizon          = st.session_state.get("last_horizon", horizon)
        trade_frequency  = st.session_state.get("last_trade_frequency", trade_frequency)
        recurring_amount = st.session_state.get("last_recurring_amount", recurring_amount)
        recurring_period = st.session_state.get("last_recurring_period", recurring_period)

    if submitted:
        st.session_state.update({
            "last_budget": budget,
            "last_risk": risk,
            "last_horizon": horizon,
            "last_trade_frequency": trade_frequency,
            "last_recurring_amount": recurring_amount,
            "last_recurring_period": recurring_period,
        })

    if not submitted and "last_portfolio" in st.session_state:
        _render_results(
            st.session_state["last_portfolio"],
            st.session_state.get("last_price_cache", {}),
            budget, risk, horizon, trade_frequency, recurring_amount, recurring_period,
            manager_agent=manager_agent,
        )
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
    # Rank by profit potential via AI
    # ------------------------------------------------------------------
    with st.spinner("Ranking opportunities with AI…"):
        portfolio = manager_agent.generate_portfolio_recommendations(
            budget=budget,
            risk=risk,
            horizon=horizon,
            signals=signals,
            trade_frequency=trade_frequency,
            recurring_amount=recurring_amount,
            recurring_period=recurring_period,
        )

    if not portfolio:
        st.info("No ranked results available. Please try again.")
        return

    # ------------------------------------------------------------------
    # Prefetch chart data for all tickers (parallel via cache)
    # ------------------------------------------------------------------
    chart_fetch_bar = st.progress(0, text="Loading charts…")
    price_cache: dict = {}
    for i, item in enumerate(portfolio):
        tkr = item.get("ticker", "")
        if tkr:
            price_cache[tkr] = _fetch_price_data(tkr)
        chart_fetch_bar.progress(int((i + 1) / len(portfolio) * 100))
    chart_fetch_bar.empty()

    # Persist for dismiss-button reruns
    st.session_state["last_portfolio"]   = portfolio
    st.session_state["last_price_cache"] = price_cache

    _render_results(portfolio, price_cache, budget, risk, horizon, trade_frequency,
                    recurring_amount, recurring_period, manager_agent=manager_agent)


# ---------------------------------------------------------------------------
# Results renderer (called on fresh submit AND on dismiss-button reruns)
# ---------------------------------------------------------------------------

def _render_results(
    portfolio: list,
    price_cache: dict,
    budget: float,
    risk: str,
    horizon: str,
    trade_frequency: str,
    recurring_amount: float,
    recurring_period: str,
    manager_agent=None,
) -> None:
    dismissed = st.session_state.get("dismissed_tickers", set())
    visible = [item for item in portfolio if item.get("ticker", "") not in dismissed]

    if not visible:
        st.info("You dismissed all picks. Regenerate a new portfolio to start fresh.")
        return

    # Frequency-aware budget banner
    buy_visible = [item for item in visible if item.get("signal") == "BUY"]
    deployed_pct = sum(item.get("suggested_allocation", 0) for item in buy_visible)
    reserved_pct = max(0, 100 - deployed_pct)
    reserved_amt = budget * reserved_pct / 100
    is_good_day  = getattr(manager_agent, "last_is_good_day", False)
    avg_score    = getattr(manager_agent, "last_avg_score", 0.0)

    freq_lower = trade_frequency.lower()
    if "daily" in freq_lower:
        tier_label  = "35% (strong signal day)" if is_good_day else "20% (normal day)"
        reserve_tip = f"keeping **${reserved_amt:,.0f}** ({reserved_pct:.0f}%) free for tomorrow's trades"
    elif "weekly" in freq_lower:
        tier_label  = "50% (strong signal day)" if is_good_day else "40% (normal day)"
        reserve_tip = f"keeping **${reserved_amt:,.0f}** ({reserved_pct:.0f}%) in reserve"
    else:
        tier_label  = "65% (strong signal day)" if is_good_day else "55% (normal day)"
        reserve_tip = f"**${reserved_amt:,.0f}** ({reserved_pct:.0f}%) held back"

    opportunity_tag = " 🟢 Strong market opportunity detected." if is_good_day else ""
    st.info(
        f"**{trade_frequency}** — deploying **{deployed_pct:.0f}%** of your budget "
        f"(limit: {tier_label}), {reserve_tip}.{opportunity_tag}"
    )

    # ------------------------------------------------------------------
    # Results header
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Top Investment Opportunities for You")
    st.caption(
        f"Ranked by profit potential · Budget **${budget:,.0f}** · "
        f"Risk **{risk}** · Horizon **{horizon}** · "
        f"Trade frequency **{trade_frequency}** · Scroll through each card below"
    )

    # ------------------------------------------------------------------
    # Instagram-style snap-scroll feed
    # ------------------------------------------------------------------
    feed_html = _build_feed_html(visible, budget, price_cache)
    _feed_src = "data:text/html;charset=utf-8;base64," + base64.b64encode(feed_html.encode()).decode()
    st.iframe(_feed_src, height=_CARD_H)

    # ------------------------------------------------------------------
    # Dismiss buttons — one per visible card
    # ------------------------------------------------------------------
    st.caption("Remove a pick you don't like:")
    dismiss_cols = st.columns(len(visible))
    for col, item in zip(dismiss_cols, visible):
        ticker = item.get("ticker", "")
        signal = item.get("signal", "BUY")
        color  = "#00d4aa" if signal == "BUY" else "#ff4b4b"
        if col.button(f"✕ {ticker}", key=f"dismiss_{ticker}",
                      help=f"Remove {ticker} from your picks"):
            st.session_state.setdefault("dismissed_tickers", set()).add(ticker)
            st.rerun()

    # ------------------------------------------------------------------
    # Prediction chart — projected returns for BUY picks
    # ------------------------------------------------------------------
    buy_items = [item for item in visible if item.get("signal") == "BUY"]
    if buy_items:
        st.divider()
        _render_prediction_chart(
            buy_items, price_cache, budget, recurring_amount, recurring_period
        )
        st.divider()
        _render_wealth_journey(
            buy_items, price_cache, budget,
            recurring_amount, recurring_period, horizon, trade_frequency,
        )


# ---------------------------------------------------------------------------
# Prediction chart — normalised projected returns + portfolio value projection
# ---------------------------------------------------------------------------

_CHART_COLORS = [
    "#00d4aa", "#7b61ff", "#ff9f43", "#54a0ff", "#ff6b9d",
    "#26de81", "#fd9644", "#45aaf2", "#a29bfe", "#fdcb6e",
]


def _render_prediction_chart(
    buy_items: list,
    price_cache: dict,
    budget: float,
    recurring_amount: float = 0.0,
    recurring_period: str = "monthly",
) -> None:
    st.subheader("Projected Returns — BUY Picks")
    st.caption(
        "Solid = historical · Dashed = projection · "
        "Rank #1 (highest score) gets the largest allocation. "
        "Past performance is not indicative of future results."
    )

    # ------------------------------------------------------------------
    # Build per-stock projection data
    # ------------------------------------------------------------------
    projections = []   # list of dicts with all needed data per stock
    today_date  = None

    for idx, item in enumerate(buy_items):
        ticker     = item.get("ticker", "")
        chart_data = price_cache.get(ticker)
        if not chart_data:
            continue
        base_price = chart_data["close"][-1]
        if base_price <= 0:
            continue

        alloc_pct  = item.get("suggested_allocation", 0.0)
        alloc_amt  = budget * alloc_pct / 100.0
        proj_data  = _project_candles(chart_data, "BUY", n=10)
        color      = _CHART_COLORS[idx % len(_CHART_COLORS)]

        hist_dates = chart_data["dates"][-10:]
        hist_pct   = [round((p / base_price - 1) * 100, 3) for p in chart_data["close"][-10:]]
        proj_dates = proj_data["dates"]
        proj_pct   = [round((p / base_price - 1) * 100, 3) for p in proj_data["close"]]

        if today_date is None:
            today_date = hist_dates[-1]

        projections.append({
            "ticker":     ticker,
            "color":      color,
            "alloc_amt":  alloc_amt,
            "alloc_pct":  alloc_pct,
            "hist_dates": hist_dates,
            "hist_pct":   hist_pct,
            "proj_dates": proj_dates,
            "proj_pct":   proj_pct,
        })

    if not projections:
        st.info("No chart data available for BUY picks.")
        return

    # ------------------------------------------------------------------
    # Chart 1: per-stock % return (historical + projected)
    # ------------------------------------------------------------------
    fig = go.Figure()

    for sp in projections:
        # Solid historical
        fig.add_trace(go.Scatter(
            x=sp["hist_dates"], y=sp["hist_pct"],
            mode="lines", name=sp["ticker"],
            line=dict(color=sp["color"], width=2),
            legendgroup=sp["ticker"],
            hovertemplate=f"<b>{sp['ticker']}</b><br>%{{x}}<br>%{{y:+.2f}}%"
                          f"<br>Allocated: ${sp['alloc_amt']:,.0f} ({sp['alloc_pct']:.1f}%)<extra></extra>",
        ))
        # Dashed projection (continues from last historical point)
        fig.add_trace(go.Scatter(
            x=[sp["hist_dates"][-1]] + sp["proj_dates"],
            y=[sp["hist_pct"][-1]]   + sp["proj_pct"],
            mode="lines", name=sp["ticker"],
            line=dict(color=sp["color"], width=2, dash="dash"),
            legendgroup=sp["ticker"], showlegend=False,
            hovertemplate=f"<b>{sp['ticker']} (proj)</b><br>%{{x}}<br>%{{y:+.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#333", line_width=1)
    if today_date:
        fig.add_shape(
            type="line",
            x0=today_date, x1=today_date,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="rgba(255,255,255,0.15)"),
        )
        fig.add_annotation(
            x=today_date, y=1, yref="paper", text="TODAY",
            showarrow=False, font=dict(color="#666", size=10),
            xanchor="left", yanchor="top",
        )

    fig.update_layout(
        height=360, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333", title_text="Date"),
        yaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333",
                   title_text="Return from today (%)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333", borderwidth=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Horizon config — maps horizon string → chart/schedule parameters
# ---------------------------------------------------------------------------

# ── TUNE ME ──────────────────────────────────────────────────────────────────
# Horizon config: n = number of periods in chart/table, td = trading days per
# period (used to compound the daily rate), unit = row label in the roadmap.
# Change "n" to show more/fewer rows, change "td" to adjust how aggressively
# returns compound (larger td = bigger per-period growth).
_HORIZON_CFG = {
    "Intraday":                 {"n": 5,  "unit": "Day",   "td": 1,  "label": "5 trading days"},
    "Short-term (1-4 weeks)":   {"n": 4,  "unit": "Week",  "td": 5,  "label": "4 weeks"},
    "Medium-term (1-3 months)": {"n": 12, "unit": "Week",  "td": 5,  "label": "12 weeks"},
    "Long-term (>3 months)":    {"n": 12, "unit": "Month", "td": 21, "label": "12 months"},
}
# ─────────────────────────────────────────────────────────────────────────────


def _model_daily_return(buy_items: list, price_cache: dict, budget: float) -> tuple:
    """Returns (daily_rate, total_deployed_$) derived from the 10-day model projection."""
    total_deployed = sum(budget * item.get("suggested_allocation", 0) / 100 for item in buy_items)
    if total_deployed <= 0:
        return 0.001, 0.0
    w_pct = 0.0
    for item in buy_items:
        cd = price_cache.get(item.get("ticker", ""))
        if not cd or cd["close"][-1] <= 0:
            continue
        base = cd["close"][-1]
        proj = _project_candles(cd, "BUY", n=10)
        pct  = (proj["close"][-1] / base - 1) * 100
        w_pct += pct * (budget * item.get("suggested_allocation", 0) / 100) / total_deployed
    daily_rate = (1 + w_pct / 100) ** (1 / 10) - 1
    # ── TUNE ME ──────────────────────────────────────────────────────────────
    # These caps translate directly to the projected returns in the chart.
    # 0.001 ≈ +28 % annual max  |  -0.0009 ≈ -20 % annual floor
    # Raise them for more aggressive projections, lower for conservative ones.
    daily_rate = min(max(daily_rate, -0.0009), 0.001)
    # ─────────────────────────────────────────────────────────────────────────
    return daily_rate, total_deployed


# ---------------------------------------------------------------------------
# Wealth journey: combined long-term projection + investment roadmap
# ---------------------------------------------------------------------------

def _render_wealth_journey(
    buy_items: list,
    price_cache: dict,
    budget: float,
    recurring_amount: float,
    recurring_period: str,
    horizon: str,
    trade_frequency: str,
) -> None:
    cfg            = _HORIZON_CFG.get(horizon, _HORIZON_CFG["Long-term (>3 months)"])
    daily_rate, total_deployed = _model_daily_return(buy_items, price_cache, budget)
    if total_deployed <= 0:
        return

    # Per-period return compounded from the model's implied daily rate.
    # ── TUNE ME: change ±0.25 to allow higher/lower per-period swings ────────
    period_ret = min(max((1 + daily_rate) ** cfg["td"] - 1, -0.25), 0.25)

    # Per-period recurring contribution (pro-rated to period length)
    if recurring_amount > 0:
        td_per_recur = 5.0 if recurring_period == "weekly" else 21.0
        period_recur = recurring_amount * cfg["td"] / td_per_recur
    else:
        period_recur = 0.0

    n      = cfg["n"]
    unit   = cfg["unit"]
    labels = [f"{unit} 0\n(Today)"] + [f"{unit} {i}" for i in range(1, n + 1)]

    # Savings account baseline: 3 % annual
    savings_r = (1.03) ** (cfg["td"] / 252) - 1
    savings = [budget]
    for _ in range(n):
        savings.append(round(savings[-1] * (1 + savings_r), 2))

    # Capital-only: deployed amount + recurring, no returns
    capital = [total_deployed]
    for _ in range(n):
        capital.append(round(capital[-1] + period_recur, 2))

    # TradingPal: compound returns + recurring contributions
    tp = [total_deployed]
    for _ in range(n):
        tp.append(round(tp[-1] * (1 + period_ret) + period_recur, 2))

    # Milestones
    def _first_at(threshold):
        return next((i for i, v in enumerate(tp) if v >= threshold), None)

    m25 = _first_at(total_deployed * 1.25)
    m50 = _first_at(total_deployed * 1.50)
    m2x = _first_at(total_deployed * 2.00)

    final_val  = tp[-1]
    final_gain = tp[-1] - capital[-1]
    gain_color = "#00d4aa" if final_gain >= 0 else "#ff4b4b"

    # ---- Header ----
    total_contributed = capital[-1]
    gain_pct = (final_val / total_contributed - 1) * 100 if total_contributed > 0 else 0.0
    st.subheader(f"Your Wealth Journey — Next {cfg['label']}")
    recur_note = f" + ${recurring_amount:,.0f}/{recurring_period} added regularly" if recurring_amount > 0 else ""
    st.write(
        f"You're deploying ${total_deployed:,.0f} today{recur_note}. "
        f"Based on current market signals, your portfolio could grow to "
        f"${final_val:,.0f} — "
        f"a projected gain of ${final_gain:+,.0f} ({gain_pct:+.1f}%) on total capital invested."
    )

    # ---- Chart ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=savings,
        mode="lines", name="Savings account (3%/yr)",
        line=dict(color="#3a3a4a", width=1.5, dash="dot"),
        hovertemplate="%{x}<br>Savings: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=capital,
        mode="lines", name="Capital invested (no returns)",
        line=dict(color="#556", width=1.5),
        fill="tozeroy", fillcolor="rgba(80,80,100,0.06)",
        hovertemplate="%{x}<br>Capital: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=tp,
        mode="lines+markers", name="With TradingPal",
        line=dict(color="#00d4aa", width=3),
        fill="tonexty", fillcolor="rgba(0,212,170,0.09)",
        marker=dict(size=5, color="#00d4aa"),
        hovertemplate="%{x}<br><b>Portfolio: $%{y:,.0f}</b><extra></extra>",
    ))
    for ms_idx, ms_label, ms_icon in [(m25, "+25%", "⭐"), (m50, "+50%", "🌟"), (m2x, "2× 🏆", "")]:
        if ms_idx is not None and ms_idx <= n:
            fig.add_annotation(
                x=labels[ms_idx], y=tp[ms_idx],
                text=f"{ms_icon} {ms_label}",
                showarrow=True, arrowhead=2, arrowcolor="#00d4aa",
                font=dict(color="#00d4aa", size=11),
                bgcolor="rgba(0,18,13,0.85)", bordercolor="#00d4aa",
                ax=0, ay=-38,
            )
    fig.update_layout(
        height=400, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        margin=dict(l=70, r=20, t=20, b=60),
        xaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333",
                   tickangle=-30),
        yaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333",
                   title_text="Portfolio value ($)", tickprefix="$", tickformat=",.0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a1a2e",
                    orientation="h", y=-0.22),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    # ---- Key numbers ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deployed today",   f"${total_deployed:,.0f}")
    c2.metric("Total contributed", f"${total_contributed:,.0f}",
              delta=f"+${total_contributed - total_deployed:,.0f} recurring" if period_recur > 0 else None)
    c3.metric("Projected value",  f"${final_val:,.0f}",
              delta=f"{gain_pct:+.1f}%")
    c4.metric("Projected gains",  f"${final_gain:+,.0f}",
              delta="vs. savings account" if final_val > savings[-1] else "below savings")

    # ---- Investment schedule ----
    st.divider()
    _render_investment_schedule(
        buy_items, budget, total_deployed, period_recur,
        cfg, period_ret, price_cache, trade_frequency, tp,
    )


# ---------------------------------------------------------------------------
# Investment roadmap: step-by-step action plan
# ---------------------------------------------------------------------------

def _render_investment_schedule(
    buy_items: list,
    budget: float,
    total_deployed: float,
    period_recur: float,
    cfg: dict,
    period_ret: float,
    price_cache: dict,
    trade_frequency: str,
    tp_values: list,
) -> None:
    unit      = cfg["unit"]
    n         = cfg["n"]
    freq      = trade_frequency.lower()

    # ── TUNE ME ──────────────────────────────────────────────────────────────
    # _action decides what action label each period row gets.
    # Logic: initial invest → rebalance checkpoints → recurring every period → hold.
    # If you want recurring only every 2nd period, change `if period_recur > 0`
    # to `if period_recur > 0 and i % 2 == 0`.
    def _action(i: int):  # i = 1-indexed period number
        if i == 1:
            return "🟢", "ENTER POSITIONS"
        if i == n:
            return "📊", "REVIEW & TAKE PROFIT"
        # Periodic rebalance checkpoints (independent of recurring)
        if "monthly" in unit.lower() and i % 3 == 0:
            return "⚖️", "REBALANCE"
        if "week" in unit.lower() and i % 4 == 0:
            return "⚖️", "REBALANCE"
        if "day" in unit.lower() and i % 2 == 0:
            return "👀", "REVIEW & HOLD"
        # Recurring contribution — every period once there's recurring income
        if period_recur > 0:
            return "💰", "ADD RECURRING"
        return "📈", "HOLD & MONITOR"
    # ─────────────────────────────────────────────────────────────────────────

    # ── TUNE ME ──────────────────────────────────────────────────────────────
    # Edit the tips below to change the advice shown in the roadmap table.
    _tips = {
        "ENTER POSITIONS":      "Buy all listed picks at or near market price. Hold through volatility.",
        "HOLD & MONITOR":       "Hold all positions. Check if price is above your entry — no panic selling.",
        "ADD RECURRING":        "Add your recurring contribution split across your top 2 performers.",
        "REBALANCE":            "Trim any position up >20%, reinvest into laggards showing recovery.",
        "REVIEW & TAKE PROFIT": "Lock in 30–50% of gains on your biggest winners. Let the rest compound.",
        "REVIEW & HOLD":        "Check each position — ensure it's still on the right trend.",
    }
    # ─────────────────────────────────────────────────────────────────────────

    import datetime
    today = datetime.date(2026, 4, 5)
    td_per_period = cfg["td"]
    # Calendar days per trading day ≈ 1.4 (accounts for weekends)
    calendar_days = int(td_per_period * 1.4)

    rows = []
    pv = total_deployed

    for i in range(1, n + 1):
        emoji, act_label = _action(i)
        tip = _tips.get(act_label, "")
        period_start = today + datetime.timedelta(days=(i - 1) * calendar_days)
        period_end   = today + datetime.timedelta(days=i * calendar_days - 1)
        date_str = (
            period_start.strftime("%b %d")
            if td_per_period == 1
            else f"{period_start.strftime('%b %d')} – {period_end.strftime('%b %d')}"
        )
        pv = tp_values[i] if i < len(tp_values) else tp_values[-1]

        if i == 1:
            # One row per stock on entry day
            for item in buy_items:
                ticker    = item.get("ticker", "")
                alloc_pct = item.get("suggested_allocation", 0)
                alloc_amt = budget * alloc_pct / 100
                cd        = price_cache.get(ticker, {})
                cp        = cd.get("close", [0])[-1] if cd else 0
                rows.append({
                    f"{unit}":          f"{unit} 1",
                    "Date":             date_str,
                    "Action":           f"{emoji} {act_label}",
                    "Asset":            ticker,
                    "Amount":           f"${alloc_amt:,.0f}  ({alloc_pct:.1f}%)",
                    "Entry":            f"~${cp:.2f}" if cp > 0 else "Market",
                    "Est. Portfolio":   f"${pv:,.0f}",
                    "Tip":              tip,
                })
        else:
            asset  = "-"
            amount = "-"
            if "ADD" in act_label and period_recur > 0:
                top    = buy_items[:2]
                asset  = " + ".join(t.get("ticker", "") for t in top)
                amount = f"${period_recur:,.0f}"
            elif "REBALANCE" in act_label or "PROFIT" in act_label:
                asset = "All positions"
            rows.append({
                f"{unit}":         f"{unit} {i}",
                "Date":            date_str,
                "Action":          f"{emoji} {act_label}",
                "Asset":           asset,
                "Amount":          amount,
                "Entry":           "-",
                "Est. Portfolio":  f"${pv:,.0f}",
                "Tip":             tip,
            })

    df = pd.DataFrame(rows)

    st.subheader("Your Investment Roadmap")
    st.caption(
        f"Step-by-step plan for the next {cfg['label']} based on your "
        f"**{trade_frequency}** trading style and today's signals. "
        "Portfolio estimates use the model's projected return rate — actual results will vary."
    )
    st.dataframe(df, width="stretch", hide_index=True)


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
        st.markdown("AAPL, MSFT, TSLA, NVDA, SPY, QQQ and 35 more")
    with c2:
        st.markdown("#### Crypto")
        st.markdown("BTC, ETH, SOL, BNB, XRP ranked alongside traditional assets")
    with c3:
        st.markdown("#### AI Ranking")
        st.markdown("Gemini scores each asset specifically for your profile")
