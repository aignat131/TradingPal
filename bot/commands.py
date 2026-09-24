"""
Turn the user's Telegram messages into watchlist changes.

Free-text messages ("start following AMD and drop Tesla") are interpreted by
the AI into a list of structured actions; slash commands (/add AMD) work even
when no AI is available. Every action is validated before touching the store,
and the reply is built from what actually happened — not from the AI's text.
"""
import html
import logging
from typing import Callable, Dict, List, Optional, Tuple

from bot.storage import ASSET_TYPES, FOCUS_LEVELS, WatchlistStore, normalize_symbol

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "I send you a market recap every morning and can manage your watchlist.\n\n"
    "Just write naturally, e.g.:\n"
    "• <i>add AMD and Solana</i>\n"
    "• <i>stop tracking Tesla</i>\n"
    "• <i>only mention XRP when there's a strong signal</i>\n"
    "• <i>note on BTC: long-term hold</i>\n"
    "• <i>show my list</i>\n\n"
    "Shortcuts: /list, /add SYMBOL, /remove SYMBOL, /help\n"
    "<i>I read your messages once a day, just before the morning recap, so changes "
    "apply to the next recap. Need it sooner? GitHub → Actions → Telegram bot → Run workflow.</i>"
)

_COMMAND_PROMPT = """\
You manage the watchlist of a market-recap Telegram bot. Convert the user's
message into actions. Symbols must be Yahoo Finance tickers: US stocks like
"AAPL", ETFs like "SPY", crypto as "<COIN>-USD" (e.g. "SOL-USD"), indices like
"^GSPC", futures like "GC=F". Map company/coin names to tickers yourself
(e.g. "Strategy"/"MicroStrategy" → "MSTR", "gold" → "GLD").

Current watchlist:
{watchlist}

Allowed actions (JSON objects):
- {{"type": "add", "symbol": "AMD", "name": "AMD", "asset_type": one of {asset_types}, "focus": "core"|"watch", "note": ""}}
- {{"type": "remove", "symbol": "TSLA"}}
- {{"type": "set_focus", "symbol": "XRP-USD", "focus": "core"|"watch"}}   ("watch" = mention only on strong signals)
- {{"type": "set_note", "symbol": "BTC-USD", "note": "..."}}
- {{"type": "list"}}
- {{"type": "help"}}

If the message is unclear or unrelated, return no actions and put a short
clarifying question or answer in "reply". Otherwise leave "reply" empty.

User message: {message}

Respond with ONLY JSON: {{"actions": [...], "reply": ""}}
"""


def parse_slash_command(text: str) -> Optional[List[Dict]]:
    """Deterministic parser for /commands. Returns None for free text."""
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text.split()
    cmd = parts[0][1:].split("@")[0].lower()
    args = parts[1:]
    if cmd in ("start", "help"):
        return [{"type": "help"}]
    if cmd in ("list", "watchlist"):
        return [{"type": "list"}]
    if cmd == "add" and args:
        return [{"type": "add", "symbol": a} for a in args]
    if cmd in ("remove", "rm", "delete") and args:
        return [{"type": "remove", "symbol": a} for a in args]
    if cmd == "focus" and len(args) == 2:
        return [{"type": "set_focus", "symbol": args[0], "focus": args[1].lower()}]
    return [{"type": "help"}]


def _guess_type(symbol: str) -> str:
    if symbol.endswith("-USD"):
        return "crypto"
    if symbol.startswith("^"):
        return "index"
    if symbol.endswith("=F"):
        return "commodity"
    return "stock"


class CommandProcessor:
    def __init__(
        self,
        store: WatchlistStore,
        manager=None,
        symbol_exists: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.store = store
        self.manager = manager
        if symbol_exists is None:
            from bot.market import symbol_exists as _exists
            symbol_exists = _exists
        self._symbol_exists = symbol_exists

    def interpret(self, text: str) -> Tuple[List[Dict], str]:
        slash = parse_slash_command(text)
        if slash is not None:
            return slash, ""
        if self.manager is None:
            return [{"type": "help"}], "I can't understand free text right now (AI unavailable) — use the shortcuts."
        result = self.manager.ask_json(_COMMAND_PROMPT.format(
            watchlist=self.store.describe(),
            asset_types="|".join(ASSET_TYPES),
            message=text[:1000],
        ))
        if not isinstance(result, dict) or not isinstance(result.get("actions"), list):
            return [], "Sorry, I couldn't process that right now. Try a shortcut like /add AMD."
        actions = [a for a in result["actions"] if isinstance(a, dict)]
        return actions, str(result.get("reply") or "")[:500]

    def apply(self, action: Dict) -> str:
        """Execute one action → reply line ("" if ignored)."""
        kind = action.get("type")
        try:
            if kind == "add":
                sym = normalize_symbol(action.get("symbol", ""))
                if self.store.get(sym):
                    return f"ℹ️ {sym} is already on your watchlist."
                if not self._symbol_exists(sym):
                    return f"❌ Couldn't find prices for “{sym}” on Yahoo Finance — not added."
                focus = action.get("focus", "core")
                msg = self.store.add(
                    sym,
                    name=action.get("name", ""),
                    asset_type=action.get("asset_type") or _guess_type(sym),
                    focus=focus if focus in FOCUS_LEVELS else "core",
                    note=action.get("note", ""),
                )
                return f"✅ {msg}"
            if kind == "remove":
                return f"🗑 {self.store.remove(action.get('symbol', ''))}"
            if kind == "set_focus":
                return f"✅ {self.store.set_focus(action.get('symbol', ''), action.get('focus', ''))}"
            if kind == "set_note":
                return f"✅ {self.store.set_note(action.get('symbol', ''), action.get('note', ''))}"
            if kind == "list":
                return "📋 <b>Your watchlist</b>\n" + html.escape(self.store.describe(), quote=False)
            if kind == "help":
                return HELP_TEXT
        except ValueError as exc:
            return f"❌ {html.escape(str(exc), quote=False)}"
        logger.warning("Ignoring unknown action: %s", action)
        return ""

    def handle(self, text: str) -> str:
        """Process one user message → reply text."""
        actions, ai_reply = self.interpret(text)
        lines = []
        for action in actions[:10]:
            line = self.apply(action)
            if line:
                lines.append(line)
        if ai_reply and not lines:
            lines.append(html.escape(ai_reply, quote=False))
        if not lines:
            lines.append("I didn't catch a watchlist change there. Send /help to see what I can do.")
        return "\n".join(lines)
