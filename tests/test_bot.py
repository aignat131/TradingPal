"""Unit tests for the Telegram bot (all network calls mocked)."""
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from bot import brief, commands, main, market, news, schedule
from bot.storage import BotState, WatchlistStore
from bot.telegram import MAX_MESSAGE_LEN, split_message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text(json.dumps({
        "settings": {"send_time": "08:00", "timezone": "Europe/Bucharest"},
        "assets": [
            {"symbol": "NVDA", "name": "NVIDIA", "type": "stock", "focus": "core", "note": ""},
            {"symbol": "TSLA", "name": "Tesla", "type": "stock", "focus": "core", "note": ""},
            {"symbol": "BTC-USD", "name": "Bitcoin", "type": "crypto", "focus": "core", "note": ""},
        ],
    }))
    return WatchlistStore(str(path))


class FakeManager:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def ask_json(self, prompt):
        self.prompts.append(prompt)
        return self.response


def _series(values):
    idx = pd.date_range("2026-03-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def test_store_add_remove_roundtrip(store, tmp_path):
    store.add("amd", name="AMD", asset_type="stock")
    store.remove("TSLA")
    store.save()
    reloaded = WatchlistStore(str(tmp_path / "watchlist.json"))
    assert reloaded.symbols() == ["NVDA", "BTC-USD", "AMD"]


def test_store_rejects_bad_input(store):
    with pytest.raises(ValueError):
        store.add("not a ticker!")
    with pytest.raises(ValueError):
        store.remove("AAPL")
    with pytest.raises(ValueError):
        store.set_focus("NVDA", "sometimes")


def test_state_defaults_when_missing(tmp_path):
    state = BotState(str(tmp_path / "missing.json"))
    assert state.telegram_offset == 0 and state.last_brief_date == ""


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

def test_analyze_series_rising_market():
    result = market.analyze_series(_series(np.linspace(100, 160, 120)))
    assert result["price"] == pytest.approx(160)
    assert result["change_1d"] > 0 and result["rsi"] > 60
    assert result["sma_trend"] == "bullish"


def test_analyze_series_short_history_has_no_technicals():
    result = market.analyze_series(_series([10, 11, 12]))
    assert result["rsi"] is None and result["signal"] == "NEUTRAL"
    assert market.analyze_series(_series([10])) is None


def test_snapshot_marks_missing_symbols_unavailable(monkeypatch):
    cols = pd.MultiIndex.from_product([["NVDA"], ["Close"]])
    df = pd.DataFrame(np.linspace(100, 120, 80).reshape(-1, 1), columns=cols,
                      index=pd.date_range("2026-01-01", periods=80))
    monkeypatch.setattr(market, "_download", lambda symbols, period="6mo": df)
    snap = market.get_snapshot(["NVDA", "FAKE"])
    assert snap["NVDA"]["price"] == pytest.approx(120)
    assert snap["FAKE"] is None  # never invented


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

_RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Bitcoin jumps on ETF inflows</title><link>https://x.test/a</link>
<pubDate>Wed, 23 Sep 2026 20:00:00 GMT</pubDate></item>
<item><title>Old story</title><link>https://x.test/b</link>
<pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_parse_feed_drops_stale_items():
    now = datetime(2026, 9, 24, 6, 0, tzinfo=timezone.utc)
    items = news.parse_feed(_RSS, "CoinDesk", now=now)
    assert [i["title"] for i in items] == ["Bitcoin jumps on ETF inflows"]


def test_parse_yf_item_both_formats():
    new = {"content": {"title": "NVDA beats", "pubDate": "2026-09-24T01:00:00Z",
                       "provider": {"displayName": "Reuters"},
                       "canonicalUrl": {"url": "https://r.test/1"}}}
    old = {"title": "TSLA recall", "link": "https://y.test/2", "publisher": "AP",
           "providerPublishTime": 1790000000}
    assert news._parse_yf_item(new, "NVDA")["source"] == "Reuters"
    assert news._parse_yf_item(old, "TSLA")["url"] == "https://y.test/2"


def test_collect_news_has_no_simulated_fallback(monkeypatch):
    monkeypatch.setattr(news, "fetch_feed", lambda *a, **k: [])
    monkeypatch.setattr(news, "fetch_ticker_news", lambda symbols: {s: [] for s in symbols})
    data = news.collect_news(["NVDA"])
    assert data == {"crypto": [], "markets": [], "by_symbol": {"NVDA": []}}


# ---------------------------------------------------------------------------
# Brief
# ---------------------------------------------------------------------------

def _mock_brief_inputs(monkeypatch):
    def snap(symbols):
        base = {"price": 100.0, "change_1d": 1.0, "change_5d": 2.0, "change_1m": 3.0,
                "rsi": 50.0, "sma_trend": "neutral", "signal": "NEUTRAL", "volatility": 30.0}
        out = {s: dict(base, score=0.0) for s in symbols}
        out["NVDA"].update(score=0.6, rsi=28.0, sma_trend="bullish")
        out["TSLA"].update(score=-0.5, rsi=78.0, sma_trend="bearish")
        out["^VIX"] = None
        return out

    monkeypatch.setattr(market, "get_snapshot", lambda symbols: snap(list(symbols) + list(market.BENCHMARKS)))
    monkeypatch.setattr(news, "collect_news", lambda symbols: {
        "crypto": [{"title": "BTC <rallies> & more", "url": "https://c.test/1", "source": "CoinDesk"}],
        "markets": [{"title": "Stocks mixed", "url": "https://m.test/1", "source": "CNBC"}],
        "by_symbol": {s: [] for s in symbols},
    })


_NOW = datetime(2026, 9, 24, 8, 5, tzinfo=ZoneInfo("Europe/Bucharest"))


def test_brief_with_ai_drops_invented_references(monkeypatch, store):
    _mock_brief_inputs(monkeypatch)
    ai = FakeManager({
        "mood": "Cautiously positive.",
        "crypto_news": [{"id": "c1", "summary": "Momentum returns"}, {"id": "c99", "summary": "made up"}],
        "market_news": [{"id": "m1", "summary": "Flat open"}],
        "ideas": [
            {"symbol": "NVDA", "action": "BUY", "reason": "Oversold in uptrend"},
            {"symbol": "TSLA", "action": "SELL", "reason": "Overbought"},
            {"symbol": "GME", "action": "BUY", "reason": "not on watchlist"},
        ],
    })
    text = brief.build_brief(store, _NOW, ai)
    assert "Cautiously positive." in text
    assert "BTC &lt;rallies&gt; &amp; more" in text  # HTML-escaped
    assert "made up" not in text and "GME" not in text
    assert "Consider buying</b>" not in text  # sanity: bold only wraps symbol
    assert "🟢 Consider buying <b>NVDA</b>" in text
    assert "🔴 Consider selling/avoiding <b>TSLA</b>" in text
    assert "AI recap" in text
    assert len(text) <= MAX_MESSAGE_LEN


def test_brief_falls_back_to_rules_without_ai(monkeypatch, store):
    _mock_brief_inputs(monkeypatch)
    text = brief.build_brief(store, _NOW, manager=None)
    assert "Rule-based recap" in text
    assert "<b>NVDA</b>" in text and "<b>TSLA</b>" in text
    assert "Stocks mixed" in text


def test_brief_invalid_ai_output_uses_fallback(monkeypatch, store):
    _mock_brief_inputs(monkeypatch)
    text = brief.build_brief(store, _NOW, FakeManager({"decision": "BUY"}))
    assert "Rule-based recap" in text


def test_brief_reports_missing_news(monkeypatch, store):
    _mock_brief_inputs(monkeypatch)
    monkeypatch.setattr(news, "collect_news", lambda s: {"crypto": [], "markets": [], "by_symbol": {}})
    assert "News sources were unavailable" in brief.build_brief(store, _NOW)


def test_rank_candidates():
    buys, sells = brief.rank_candidates({"A": 0.4, "B": -0.3, "C": 0.1, "D": None, "E": -0.6})
    assert buys == ["A", "C"] and sells == ["E", "B"]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def test_slash_commands_work_without_ai(store):
    proc = commands.CommandProcessor(store, manager=None, symbol_exists=lambda s: True)
    reply = proc.handle("/add AMD SOL-USD")
    assert "Added AMD" in reply and store.get("SOL-USD")["type"] == "crypto"
    assert "Removed TSLA" in proc.handle("/remove TSLA")
    assert "once a day" in proc.handle("/start")


def test_free_text_via_ai(store):
    ai = FakeManager({"actions": [
        {"type": "add", "symbol": "mstr", "name": "Strategy", "asset_type": "stock"},
        {"type": "remove", "symbol": "TSLA"},
        {"type": "set_focus", "symbol": "BTC-USD", "focus": "watch"},
    ], "reply": ""})
    proc = commands.CommandProcessor(store, manager=ai, symbol_exists=lambda s: True)
    reply = proc.handle("add strategy, drop tesla, bitcoin only on strong signals")
    assert store.get("MSTR") and not store.get("TSLA")
    assert store.get("BTC-USD")["focus"] == "watch"
    assert "Removed TSLA" in reply and store.dirty


def test_unknown_symbol_not_added(store):
    ai = FakeManager({"actions": [{"type": "add", "symbol": "ZZZZ"}], "reply": ""})
    proc = commands.CommandProcessor(store, manager=ai, symbol_exists=lambda s: False)
    reply = proc.handle("add zzzz")
    assert "Couldn't find" in reply and not store.get("ZZZZ")


def test_ai_clarifying_reply_is_escaped(store):
    ai = FakeManager({"actions": [], "reply": "Which <one>?"})
    proc = commands.CommandProcessor(store, manager=ai, symbol_exists=lambda s: True)
    reply = proc.handle("add that thing")
    assert reply == "Which &lt;one&gt;?"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hhmm,last,expected", [
    ("07:59", "", False),
    ("08:00", "", True),
    ("11:30", "", True),
    ("13:30", "", True),                # late GitHub cron still delivers
    ("14:30", "", False),               # window missed
    ("09:00", "2026-09-24", False),     # already sent today
    ("09:00", "2026-09-23", True),
])
def test_brief_is_due(hhmm, last, expected):
    h, m = map(int, hhmm.split(":"))
    now = datetime(2026, 9, 24, h, m, tzinfo=ZoneInfo("Europe/Bucharest"))
    assert schedule.brief_is_due(now, "08:00", last) is expected


class FakeTelegram:
    def __init__(self, updates):
        self.chat_id = "42"
        self.updates = updates
        self.sent = []

    def get_updates(self, offset):
        return [u for u in self.updates if u["update_id"] >= offset]

    def send_message(self, text, chat_id=""):
        self.sent.append(text)


def test_process_messages_ignores_strangers(store, tmp_path):
    tg = FakeTelegram([
        {"update_id": 10, "message": {"chat": {"id": 999}, "text": "/remove NVDA"}},
        {"update_id": 11, "message": {"chat": {"id": 42}, "text": "/list"}},
    ])
    state = BotState(str(tmp_path / "state.json"))
    proc = commands.CommandProcessor(store, symbol_exists=lambda s: True)
    main.process_messages(tg, state, proc)
    assert store.get("NVDA")  # stranger's command ignored
    assert state.telegram_offset == 12
    assert len(tg.sent) == 1 and "Your watchlist" in tg.sent[0]


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

def test_split_message_respects_limit():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(300))
    chunks = split_message(text)
    assert len(chunks) > 1 and all(len(c) <= MAX_MESSAGE_LEN for c in chunks)
    assert "\n".join(chunks) == text


def test_watch_assets_only_shown_when_notable(monkeypatch, store):
    _mock_brief_inputs(monkeypatch)
    store.set_focus("BTC-USD", "watch")  # quiet: score 0, move 1%
    store.set_focus("NVDA", "watch")     # strong score 0.6 → still shown
    text = brief.build_brief(store, _NOW)
    assert "<b>BTC-USD</b> $" not in text
    assert "<b>NVDA</b> $" in text


def test_brief_refuses_when_nothing_is_available(monkeypatch, store):
    monkeypatch.setattr(market, "get_snapshot", lambda symbols: {s: None for s in symbols})
    monkeypatch.setattr(news, "collect_news", lambda s: {"crypto": [], "markets": [], "by_symbol": {}})
    with pytest.raises(RuntimeError):
        brief.build_brief(store, _NOW)


@pytest.mark.parametrize("utc,expected", [
    (datetime(2026, 7, 1, 5, 0, tzinfo=timezone.utc), True),    # summer: 08:00 local
    (datetime(2026, 7, 1, 6, 0, tzinfo=timezone.utc), True),    # (skipped in practice: already sent)
    (datetime(2026, 1, 15, 5, 0, tzinfo=timezone.utc), False),  # winter: 07:00 local
    (datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc), True),   # winter: 08:00 local
])
def test_utc_crons_cover_8am_bucharest_all_year(utc, expected):
    local = utc.astimezone(ZoneInfo("Europe/Bucharest"))
    assert schedule.brief_is_due(local, "08:00", "") is expected


def test_gate_writes_github_output(monkeypatch, tmp_path, store):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(schedule, "WatchlistStore", lambda: store)
    monkeypatch.setattr(schedule, "BotState", lambda: BotState(str(tmp_path / "state.json")))
    monkeypatch.setattr(schedule, "local_now",
                        lambda s: datetime(2026, 9, 24, 8, 3, tzinfo=ZoneInfo("Europe/Bucharest")))
    schedule.main()
    assert out.read_text() == "due=true\n"


@pytest.mark.parametrize("month", [1, 7])  # winter (UTC+2) and summer (UTC+3)
def test_workflow_cron_attempts_reach_8am_bucharest(month):
    import yaml

    with open(".github/workflows/telegram-bot.yml") as fh:
        cron = yaml.safe_load(fh)[True]["schedule"][0]["cron"]
    minutes, hours = (list(map(int, f.split(","))) for f in cron.split()[:2])
    attempts = sorted(datetime(2026, month, 15, h, m, tzinfo=timezone.utc) for h in hours for m in minutes)
    due = [t for t in attempts
           if schedule.brief_is_due(t.astimezone(ZoneInfo("Europe/Bucharest")), "08:00", "")]
    assert len(due) >= 3  # several retries if GitHub drops a run
    first_local = due[0].astimezone(ZoneInfo("Europe/Bucharest"))
    assert (first_local.hour, first_local.minute) <= (8, 30)
