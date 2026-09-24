"""
Real news for the daily brief: public RSS feeds + Yahoo Finance ticker news.

There is deliberately NO simulated fallback here (unlike data.news_fetcher):
if every source fails the brief says "news unavailable" instead of inventing
headlines.
"""
import calendar
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

CRYPTO_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}
MARKET_FEEDS = {
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
}

_USER_AGENT = "Mozilla/5.0 (compatible; TradingPalBot/1.0)"
_MAX_AGE = timedelta(hours=36)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _dedupe(items: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for item in items:
        key = _norm_title(item["title"])[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _is_recent(published: Optional[datetime], now: datetime) -> bool:
    # Items without a timestamp are kept; feeds are newest-first anyway.
    return published is None or now - published <= _MAX_AGE


def parse_feed(content: bytes, source: str, now: Optional[datetime] = None) -> List[Dict]:
    import feedparser

    now = now or datetime.now(timezone.utc)
    items = []
    for entry in feedparser.parse(content).entries:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        published = None
        ts = entry.get("published_parsed") or entry.get("updated_parsed")
        if ts:
            published = datetime.fromtimestamp(calendar.timegm(ts), tz=timezone.utc)
        if not _is_recent(published, now):
            continue
        items.append({
            "title": title,
            "url": entry.get("link", ""),
            "source": source,
            "published": published.isoformat() if published else "",
        })
    return items


def fetch_feed(source: str, url: str, limit: int = 15) -> List[Dict]:
    import requests

    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        return parse_feed(resp.content, source)[:limit]
    except Exception as exc:
        logger.warning("Feed %s failed: %s", source, exc)
        return []


def _parse_yf_item(item: Dict, symbol: str) -> Optional[Dict]:
    """yfinance has shipped two news formats; accept both."""
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content:
        title = content.get("title", "")
        url = ((content.get("clickThroughUrl") or {}).get("url")
               or (content.get("canonicalUrl") or {}).get("url", ""))
        source = (content.get("provider") or {}).get("displayName", "Yahoo Finance")
        published = None
        if content.get("pubDate"):
            try:
                published = datetime.fromisoformat(content["pubDate"].replace("Z", "+00:00"))
            except ValueError:
                pass
    else:
        title = item.get("title", "")
        url = item.get("link", "")
        source = item.get("publisher", "Yahoo Finance")
        ts = item.get("providerPublishTime")
        published = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
    if not title:
        return None
    return {
        "title": title.strip(),
        "url": url,
        "source": source,
        "published": published.isoformat() if published else "",
        "symbol": symbol,
        "_dt": published,
    }


def fetch_ticker_news(symbols: Iterable[str], per_symbol: int = 4) -> Dict[str, List[Dict]]:
    import yfinance as yf

    now = datetime.now(timezone.utc)
    result: Dict[str, List[Dict]] = {}
    for sym in symbols:
        try:
            raw = yf.Ticker(sym).news or []
        except Exception as exc:
            logger.warning("Ticker news failed for %s: %s", sym, exc)
            raw = []
        parsed = []
        for item in raw:
            p = _parse_yf_item(item, sym)
            if p and _is_recent(p.pop("_dt"), now):
                parsed.append(p)
        result[sym] = _dedupe(parsed)[:per_symbol]
    return result


def collect_news(symbols: Iterable[str]) -> Dict[str, object]:
    """
    Gather everything the brief needs:
    {"crypto": [...], "markets": [...], "by_symbol": {sym: [...]}}
    """
    crypto, markets = [], []
    for name, url in CRYPTO_FEEDS.items():
        crypto.extend(fetch_feed(name, url, limit=8))
    for name, url in MARKET_FEEDS.items():
        markets.extend(fetch_feed(name, url, limit=8))
    return {
        "crypto": _dedupe(crypto),
        "markets": _dedupe(markets),
        "by_symbol": fetch_ticker_news(symbols),
    }
