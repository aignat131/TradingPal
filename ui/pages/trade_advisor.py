"""
Trade Advisor Page — Instagram-style scrollable card feed.

Flow:
1. User enters their profile (budget, risk, horizon)
2. App analyses every asset in the watchlist
3. Top 10 opportunities are displayed as full-screen snap-scrolling cards
   Each card shows: signal badge · live chart + projected candles · reasoning · allocation
"""
import html as _html
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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

    # ------------------------------------------------------------------
    # Results header
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Top 10 Investment Opportunities for You")
    st.caption(
        f"Ranked by profit potential · Budget **${budget:,.0f}** · "
        f"Risk **{risk}** · Horizon **{horizon}** · Scroll through each card below"
    )

    # ------------------------------------------------------------------
    # Instagram-style snap-scroll feed
    # ------------------------------------------------------------------
    feed_html = _build_feed_html(portfolio, budget, price_cache)
    components.html(feed_html, height=_CARD_H, scrolling=False)

    # ------------------------------------------------------------------
    # Recurring Savings Projection (if set)
    # ------------------------------------------------------------------
    if recurring_amount > 0:
        st.divider()
        _render_recurring_summary(recurring_amount, recurring_period, budget, horizon)


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

    period_label = "per week" if recurring_period == "weekly" else "per month"
    st.subheader("Recurring Savings Projection")
    st.caption(
        f"Investing **${recurring_amount:,.0f} {period_label}** on top of your "
        f"initial **${budget:,.0f}** — here's how your total invested capital grows over time."
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

    prefix = "Wk" if recurring_period == "weekly" else "Mo"
    labels = [f"{prefix} {c['period']}" for c in cumulative]
    values = [c["total_invested"] for c in cumulative]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values,
        mode="lines+markers",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        hovertemplate="%{x}<br>Total invested: $%{y:,.0f}<extra></extra>",
        marker=dict(size=4),
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333",
                   title_text=f"Period ({period_label})"),
        yaxis=dict(gridcolor="#1a1a2e", showline=True, linecolor="#333",
                   title_text="Total Invested ($)"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    final_invested = budget + recurring_amount * total_periods
    c1, c2, c3 = st.columns(3)
    c1.metric("Initial Investment", f"${budget:,.0f}")
    c2.metric(f"Total Added ({period_label.title()})", f"${recurring_amount * total_periods:,.0f}")
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
        st.markdown("AAPL, MSFT, TSLA, NVDA, SPY, QQQ and 35 more")
    with c2:
        st.markdown("#### Crypto")
        st.markdown("BTC, ETH, SOL, BNB, XRP ranked alongside traditional assets")
    with c3:
        st.markdown("#### AI Ranking")
        st.markdown("Gemini scores each asset specifically for your profile")
