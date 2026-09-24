"""
Watchlist + bot state persistence.

Both live as JSON files under bot/data/ so the GitHub Actions workflow can
commit them back to the repo after each run — the repo *is* the database.
Keep all reads/writes behind these classes so the backend can later be
swapped (SQLite, Supabase, ...) without touching the rest of the bot.
"""
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WATCHLIST_PATH = os.path.join(_DATA_DIR, "watchlist.json")
STATE_PATH = os.path.join(_DATA_DIR, "state.json")

ASSET_TYPES = ("stock", "etf", "crypto", "index", "commodity", "other")
FOCUS_LEVELS = ("core", "watch")
MAX_ASSETS = 40

_SYMBOL_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

_DEFAULT_SETTINGS = {"send_time": "08:00", "timezone": "Europe/Bucharest"}


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def is_valid_symbol_format(symbol: str) -> bool:
    return bool(_SYMBOL_RE.match(normalize_symbol(symbol)))


def _read_json(path: str, default: Dict) -> Dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        logger.error("Corrupt JSON in %s (%s) — using defaults.", path, exc)
        return default


def _write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


class WatchlistStore:
    """The user's assets and bot settings."""

    def __init__(self, path: str = WATCHLIST_PATH) -> None:
        self._path = path
        data = _read_json(path, {"settings": {}, "assets": []})
        self.settings: Dict = {**_DEFAULT_SETTINGS, **data.get("settings", {})}
        self.assets: List[Dict] = data.get("assets", [])
        self.dirty = False

    # --- queries -------------------------------------------------------

    def get(self, symbol: str) -> Optional[Dict]:
        sym = normalize_symbol(symbol)
        return next((a for a in self.assets if a["symbol"] == sym), None)

    def symbols(self) -> List[str]:
        return [a["symbol"] for a in self.assets]

    # --- mutations (each returns a short human-readable result) -------

    def add(
        self,
        symbol: str,
        name: str = "",
        asset_type: str = "other",
        focus: str = "core",
        note: str = "",
    ) -> str:
        sym = normalize_symbol(symbol)
        if not is_valid_symbol_format(sym):
            raise ValueError(f"'{symbol}' is not a valid ticker format.")
        if self.get(sym):
            return f"{sym} is already on your watchlist."
        if len(self.assets) >= MAX_ASSETS:
            raise ValueError(f"Watchlist is full ({MAX_ASSETS} assets). Remove something first.")
        self.assets.append({
            "symbol": sym,
            "name": (name or sym).strip()[:60],
            "type": asset_type if asset_type in ASSET_TYPES else "other",
            "focus": focus if focus in FOCUS_LEVELS else "core",
            "note": (note or "").strip()[:200],
            "added_at": datetime.utcnow().strftime("%Y-%m-%d"),
        })
        self.dirty = True
        return f"Added {sym} ({(name or sym).strip()})."

    def remove(self, symbol: str) -> str:
        asset = self.get(symbol)
        if not asset:
            raise ValueError(f"{normalize_symbol(symbol)} is not on your watchlist.")
        self.assets.remove(asset)
        self.dirty = True
        return f"Removed {asset['symbol']} ({asset['name']})."

    def set_focus(self, symbol: str, focus: str) -> str:
        asset = self.get(symbol)
        if not asset:
            raise ValueError(f"{normalize_symbol(symbol)} is not on your watchlist.")
        if focus not in FOCUS_LEVELS:
            raise ValueError(f"Focus must be one of: {', '.join(FOCUS_LEVELS)}.")
        asset["focus"] = focus
        self.dirty = True
        return f"{asset['symbol']} focus set to {focus}."

    def set_note(self, symbol: str, note: str) -> str:
        asset = self.get(symbol)
        if not asset:
            raise ValueError(f"{normalize_symbol(symbol)} is not on your watchlist.")
        asset["note"] = (note or "").strip()[:200]
        self.dirty = True
        return f"Note saved for {asset['symbol']}." if asset["note"] else f"Note cleared for {asset['symbol']}."

    def set_send_time(self, hhmm: str) -> str:
        hhmm = (hhmm or "").strip()
        if len(hhmm) == 4 and hhmm[1] == ":":
            hhmm = "0" + hhmm
        if not _TIME_RE.match(hhmm):
            raise ValueError("Send time must look like HH:MM (24h), e.g. 08:00.")
        self.settings["send_time"] = hhmm
        self.dirty = True
        return f"Daily brief time set to {hhmm} ({self.settings['timezone']})."

    # --- persistence ---------------------------------------------------

    def save(self) -> None:
        _write_json(self._path, {"settings": self.settings, "assets": self.assets})
        self.dirty = False

    def describe(self) -> str:
        """Plain-text listing used in bot replies and AI prompts."""
        if not self.assets:
            return "(empty)"
        lines = []
        for a in self.assets:
            line = f"{a['symbol']} — {a['name']} [{a['type']}, {a['focus']}]"
            if a.get("note"):
                line += f" — note: {a['note']}"
            lines.append(line)
        return "\n".join(lines)


class BotState:
    """Bookkeeping between runs: Telegram update offset + last brief date."""

    def __init__(self, path: str = STATE_PATH) -> None:
        self._path = path
        data = _read_json(path, {})
        self.telegram_offset: int = int(data.get("telegram_offset", 0))
        self.last_brief_date: str = data.get("last_brief_date", "")

    def save(self) -> None:
        _write_json(self._path, {
            "telegram_offset": self.telegram_offset,
            "last_brief_date": self.last_brief_date,
        })
