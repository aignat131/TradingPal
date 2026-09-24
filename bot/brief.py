"""
Daily brief: market moves + news recap + buy/sell ideas, formatted for Telegram.

Pipeline: prices (bot.market) + headlines (bot.news) + headline sentiment
(SentimentAnalyzer) → Gemini/Groq writes the recap as JSON → validated and
rendered as Telegram HTML. If no AI is available a rule-based brief is built
from the same numbers.
"""
import html
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from bot import market, news
from bot.storage import WatchlistStore

logger = logging.getLogger(__name__)

TECH_WEIGHT = 0.75
SENTIMENT_WEIGHT = 0.25
WATCH_MOVE_PCT = 5.0

_BRIEF_PROMPT = """\
You are TradingPal, writing a short morning market recap for one retail investor.
It is {date}. Use ONLY the data below — never invent news, prices or events.

=== WATCHLIST (prices, technicals, headline sentiment) ===
{assets_json}

=== BENCHMARKS ===
{benchmarks_json}

=== HEADLINES (id: [source] title) ===
{headlines}

=== RULE-BASED IDEA CANDIDATES (composite score: + bullish, - bearish) ===
{candidates}

Tasks:
1. "mood": one sentence on the overall mood across stocks and crypto.
2. "crypto_news": the 3-4 most important crypto headlines for this investor.
3. "market_news": the 3-4 most important stock-market headlines for this investor.
   For each news item give the headline "id" exactly as listed and a "summary" of
   max 20 words explaining why it matters (to the watchlist if relevant).
4. "ideas": 1-2 BUY ideas and 1-2 SELL ideas, only for watchlist symbols.
   SELL means "consider selling / avoid" — the investor may not own it.
   Base each on the numbers and headlines above; "reason" max 25 words, citing
   the concrete signal (e.g. RSI, trend, move, news). Respect any asset note.
   If nothing is convincing on one side, return fewer ideas rather than forcing one.

Respond with ONLY this JSON (no markdown):
{{"mood": "...",
  "crypto_news": [{{"id": "c1", "summary": "..."}}],
  "market_news": [{{"id": "m1", "summary": "..."}}],
  "ideas": [{{"symbol": "NVDA", "action": "BUY", "reason": "..."}}]}}
"""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def headline_sentiment(by_symbol: Dict[str, List[Dict]]) -> Dict[str, Optional[float]]:
    """Average TextBlob/FinBERT score of each symbol's headlines (None if no news)."""
    if not any(by_symbol.values()):
        return {s: None for s in by_symbol}
    from analysis.sentiment_analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    return {
        sym: (analyzer.get_aggregate_sentiment([i["title"] for i in items]) if items else None)
        for sym, items in by_symbol.items()
    }


def composite_score(tech: Optional[Dict], sentiment: Optional[float]) -> Optional[float]:
    if not tech:
        return None
    return round(TECH_WEIGHT * tech.get("score", 0.0) + SENTIMENT_WEIGHT * (sentiment or 0.0), 3)


def strength(score: float) -> str:
    s = abs(score)
    return "strong" if s >= 0.35 else "moderate" if s >= 0.15 else "weak"


def _watch_asset_is_notable(tech: Optional[Dict], score: Optional[float]) -> bool:
    """'watch' assets only appear on a strong signal or a big daily move."""
    if not tech:
        return False
    big_move = abs(tech.get("change_1d") or 0) >= WATCH_MOVE_PCT
    return big_move or (score is not None and strength(score) == "strong")


def rank_candidates(scores: Dict[str, Optional[float]]) -> Tuple[List[str], List[str]]:
    """(bullish symbols best-first, bearish symbols worst-first)."""
    valid = [(s, v) for s, v in scores.items() if v is not None]
    buys = [s for s, v in sorted(valid, key=lambda x: -x[1]) if v > 0]
    sells = [s for s, v in sorted(valid, key=lambda x: x[1]) if v < 0]
    return buys, sells


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return html.escape(str(text or ""), quote=False)


def fmt_price(price: float) -> str:
    if price >= 1000:
        return f"${price:,.0f}"
    if price >= 1:
        return f"${price:,.2f}"
    return f"${price:.4f}"


def fmt_pct(pct: Optional[float]) -> str:
    if pct is None:
        return "n/a"
    arrow = "▲" if pct > 0 else "▼" if pct < 0 else "•"
    return f"{arrow}{abs(pct):.1f}%"


def _headline_line(item: Dict, summary: str = "") -> str:
    title = _esc(item["title"])
    link = f'<a href="{html.escape(item["url"], quote=True)}">{title}</a>' if item.get("url") else title
    line = f"• {link} <i>({_esc(item['source'])})</i>"
    if summary:
        line += f"\n   ↳ {_esc(summary)}"
    return line


def _index_headlines(news_data: Dict) -> Dict[str, Dict]:
    """Give every headline a short stable id the AI can reference."""
    index = {}
    for i, item in enumerate(news_data.get("crypto", [])[:15], 1):
        index[f"c{i}"] = item
    for i, item in enumerate(news_data.get("markets", [])[:15], 1):
        index[f"m{i}"] = item
    n = 1
    for items in news_data.get("by_symbol", {}).values():
        for item in items:
            index[f"t{n}"] = item
            n += 1
    return index


# ---------------------------------------------------------------------------
# AI + fallback
# ---------------------------------------------------------------------------

def _validate_ai(result: Optional[Dict], headlines: Dict[str, Dict], symbols: List[str]) -> Optional[Dict]:
    """Keep only references to real headlines / watchlist symbols."""
    if not isinstance(result, dict) or "ideas" not in result:
        return None

    def news_list(key: str) -> List[Tuple[Dict, str]]:
        out, used = [], set()
        for n in result.get(key) or []:
            if isinstance(n, dict) and n.get("id") in headlines and n["id"] not in used:
                used.add(n["id"])
                out.append((headlines[n["id"]], str(n.get("summary", ""))[:200]))
        return out[:4]

    ideas = []
    for idea in result.get("ideas") or []:
        if not isinstance(idea, dict):
            continue
        sym = str(idea.get("symbol", "")).upper()
        action = str(idea.get("action", "")).upper()
        if sym in symbols and action in ("BUY", "SELL") and all(i["symbol"] != sym for i in ideas):
            ideas.append({"symbol": sym, "action": action, "reason": str(idea.get("reason", ""))[:250]})
    return {
        "mood": str(result.get("mood", ""))[:300],
        "crypto_news": news_list("crypto_news"),
        "market_news": news_list("market_news"),
        "ideas": ideas[:4],
    }


def _rule_based_ideas(store: WatchlistStore, snapshot: Dict, scores: Dict) -> List[Dict]:
    buys, sells = rank_candidates(scores)
    ideas = []
    for sym, action in ([(buys[0], "BUY")] if buys else []) + ([(sells[0], "SELL")] if sells else []):
        t = snapshot[sym]
        rsi = f"RSI {t['rsi']:.0f}" if t.get("rsi") is not None else "RSI n/a"
        ideas.append({
            "symbol": sym,
            "action": action,
            "reason": f"{rsi}, {t['sma_trend']} trend, 5-day move {fmt_pct(t.get('change_5d'))}.",
        })
    return ideas


def _rule_based_mood(snapshot: Dict) -> str:
    spx, btc = snapshot.get("^GSPC"), snapshot.get("BTC-USD")
    parts = []
    if spx and spx.get("change_1d") is not None:
        parts.append(f"S&P 500 {'up' if spx['change_1d'] >= 0 else 'down'} {abs(spx['change_1d']):.1f}% last session")
    if btc and btc.get("change_1d") is not None:
        parts.append(f"Bitcoin {'up' if btc['change_1d'] >= 0 else 'down'} {abs(btc['change_1d']):.1f}% on the day")
    return (", ".join(parts) + ".") if parts else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(
    store: WatchlistStore,
    snapshot: Dict,
    scores: Dict,
    content: Dict,
    now_local: datetime,
    ai_used: bool,
    news_available: bool,
) -> str:
    lines = [f"📊 <b>TradingPal Daily</b> — {now_local.strftime('%a %d %b %Y')}"]
    if content.get("mood"):
        lines.append(f"<i>{_esc(content['mood'])}</i>")

    bench = [
        f"{name} {fmt_pct(snapshot[sym]['change_1d'])}"
        for sym, name in market.BENCHMARKS.items()
        if snapshot.get(sym)
    ]
    if bench:
        lines += ["", "🌍 <b>Markets</b>", " · ".join(bench)]

    for title, types in (("📈 <b>Stocks &amp; ETFs</b>", ("stock", "etf", "index", "commodity", "other")),
                         ("🪙 <b>Crypto</b>", ("crypto",))):
        rows = []
        for a in store.assets:
            if a["type"] not in types:
                continue
            t = snapshot.get(a["symbol"])
            if a.get("focus") == "watch" and not _watch_asset_is_notable(t, scores.get(a["symbol"])):
                continue
            if not t:
                rows.append(f"<b>{_esc(a['symbol'])}</b> — no data")
                continue
            rsi = f" · RSI {t['rsi']:.0f}" if t.get("rsi") is not None else ""
            rows.append(
                f"<b>{_esc(a['symbol'])}</b> {fmt_price(t['price'])} "
                f"{fmt_pct(t['change_1d'])} (5d {fmt_pct(t['change_5d'])}){rsi}"
            )
        if rows:
            lines += ["", title] + rows

    for title, key in (("📰 <b>Crypto news</b>", "crypto_news"), ("📰 <b>Stock news</b>", "market_news")):
        items = content.get(key) or []
        if items:
            lines += ["", title] + [_headline_line(item, summary) for item, summary in items]
    if not news_available:
        lines += ["", "📰 <i>News sources were unavailable this morning.</i>"]

    ideas = content.get("ideas") or []
    lines += ["", "💡 <b>Ideas</b>"]
    if ideas:
        for idea in ideas:
            icon = "🟢 Consider buying" if idea["action"] == "BUY" else "🔴 Consider selling/avoiding"
            sc = scores.get(idea["symbol"])
            tag = f" <i>[{strength(sc)} signal]</i>" if sc is not None else ""
            lines.append(f"{icon} <b>{_esc(idea['symbol'])}</b>{tag}\n   {_esc(idea['reason'])}")
    else:
        lines.append("No clear buy or sell setups today — holding pattern.")

    lines += [
        "",
        f"<i>{'AI recap' if ai_used else 'Rule-based recap (AI unavailable)'} from RSI/SMA + headlines. "
        "Not financial advice.</i>",
        "<i>Message me to change your watchlist (e.g. “add AMD”, “stop tracking Tesla”) — "
        "I apply it before the next recap.</i>",
    ]
    return "\n".join(lines)


def build_brief(store: WatchlistStore, now_local: datetime, manager=None) -> str:
    symbols = store.symbols()
    snapshot = market.get_snapshot(symbols)
    news_data = news.collect_news(symbols)
    sentiment = headline_sentiment(news_data["by_symbol"])
    scores = {s: composite_score(snapshot.get(s), sentiment.get(s)) for s in symbols}

    headlines = _index_headlines(news_data)
    news_available = bool(headlines)
    if not news_available and not any(snapshot.get(s) for s in symbols):
        # Nothing real to report — fail so the next run in the window retries.
        raise RuntimeError("No market data or news available.")

    content = None
    if manager is not None:
        assets_payload = []
        for a in store.assets:
            t = snapshot.get(a["symbol"])
            assets_payload.append({
                "symbol": a["symbol"], "name": a["name"], "type": a["type"],
                "focus": a["focus"], "note": a.get("note", ""),
                "data": {k: t[k] for k in ("price", "change_1d", "change_5d", "change_1m", "rsi",
                                             "sma_trend", "signal", "volatility")} if t else "unavailable",
                "headline_sentiment": sentiment.get(a["symbol"]),
                "composite": scores.get(a["symbol"]),
            })
        buys, sells = rank_candidates(scores)
        prompt = _BRIEF_PROMPT.format(
            date=now_local.strftime("%A %d %B %Y"),
            assets_json=json.dumps(assets_payload, indent=1),
            benchmarks_json=json.dumps({name: snapshot.get(sym) and snapshot[sym]["change_1d"]
                                        for sym, name in market.BENCHMARKS.items()}),
            headlines="\n".join(
                f"{hid}: [{h['source']}{' / ' + h['symbol'] if h.get('symbol') else ''}] {h['title']}"
                for hid, h in headlines.items()
            ) or "(no headlines available)",
            candidates=(
                "Bullish: " + (", ".join(f"{s} {scores[s]:+.2f}" for s in buys[:4]) or "none")
                + "\nBearish: " + (", ".join(f"{s} {scores[s]:+.2f}" for s in sells[:4]) or "none")
            ),
        )
        content = _validate_ai(manager.ask_json(prompt), headlines, symbols)
        if content is None:
            logger.warning("AI brief unavailable or invalid — using rule-based brief.")

    ai_used = content is not None
    if content is None:
        content = {
            "mood": _rule_based_mood(snapshot),
            "crypto_news": [(i, "") for i in news_data["crypto"][:4]],
            "market_news": [(i, "") for i in news_data["markets"][:4]],
            "ideas": [],
        }
    if not content["ideas"]:
        content["ideas"] = _rule_based_ideas(store, snapshot, scores)

    return render(store, snapshot, scores, content, now_local, ai_used, news_available)
