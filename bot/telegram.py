"""
Minimal Telegram Bot API client (sendMessage + getUpdates) over plain requests.
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

MAX_MESSAGE_LEN = 4096


def split_message(text: str, limit: int = MAX_MESSAGE_LEN) -> List[str]:
    """Split on blank lines / newlines so HTML tags are never cut mid-line."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for block in text.split("\n"):
        candidate = f"{current}\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # A single line longer than the limit — hard cut as last resort.
        while len(block) > limit:
            chunks.append(block[:limit])
            block = block[limit:]
        current = block
    if current:
        chunks.append(current)
    return chunks


class TelegramClient:
    def __init__(self, token: str, chat_id: str, dry_run: bool = False) -> None:
        self._base = f"https://api.telegram.org/bot{token}"
        self.chat_id = str(chat_id)
        self.dry_run = dry_run

    def _call(self, method: str, **params) -> Dict:
        import requests

        resp = requests.post(f"{self._base}/{method}", json=params, timeout=30)
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {data.get('description', resp.status_code)}")
        return data

    def send_message(self, text: str, chat_id: str = "") -> None:
        for chunk in split_message(text):
            if self.dry_run:
                print("----- [dry-run] Telegram message -----")
                print(chunk)
                continue
            self._call(
                "sendMessage",
                chat_id=chat_id or self.chat_id,
                text=chunk,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )

    def get_updates(self, offset: int) -> List[Dict]:
        """Fetch pending updates; passing offset also confirms older ones."""
        data = self._call("getUpdates", offset=offset, timeout=0, allowed_updates=["message"])
        return data.get("result", [])
