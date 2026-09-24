"""
TradingPal Telegram bot — one run = handle new messages, then send the daily
brief if it's due. Designed to be invoked every ~30 minutes by GitHub Actions.

Usage:
    python -m bot.main                 # normal run
    python -m bot.main --dry-run       # print messages instead of sending, don't save
    python -m bot.main --force-brief   # send the brief now regardless of schedule
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from bot.storage import BotState, WatchlistStore
from bot.telegram import TelegramClient

logger = logging.getLogger("bot")

# Give up on a missed morning brief after this long (e.g. Actions outage).
BRIEF_WINDOW = timedelta(hours=4)


def brief_is_due(now_local: datetime, send_time: str, last_brief_date: str) -> bool:
    hour, minute = (int(x) for x in send_time.split(":"))
    scheduled = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    today = now_local.strftime("%Y-%m-%d")
    return last_brief_date != today and scheduled <= now_local < scheduled + BRIEF_WINDOW


def process_messages(tg: TelegramClient, state: BotState, processor) -> bool:
    """Handle pending messages from the owner. Returns True if a brief was requested."""
    wants_brief = False
    updates = tg.get_updates(state.telegram_offset)
    for update in updates:
        state.telegram_offset = max(state.telegram_offset, update["update_id"] + 1)
        message = update.get("message") or {}
        chat_id = str((message.get("chat") or {}).get("id", ""))
        text = message.get("text") or ""
        if chat_id != tg.chat_id:
            logger.warning("Ignoring message from unauthorised chat %s.", chat_id)
            continue
        if not text:
            continue
        logger.info("Handling message: %r", text[:100])
        reply, brief = processor.handle(text)
        wants_brief = wants_brief or brief
        tg.send_message(reply)
    return wants_brief


def _load_manager():
    from agents.manager_agent import ManagerAgent

    manager = ManagerAgent()
    if not (manager._gemini_available or manager._groq_available):
        logger.warning("No AI key configured — running rule-based only.")
        return None
    return manager


def run(dry_run: bool = False, force_brief: bool = False) -> int:
    token, chat_id = config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        if not dry_run:
            logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
            return 1
        logger.warning("Telegram not configured — dry run will skip reading messages.")

    store = WatchlistStore()
    state = BotState()
    tg = TelegramClient(token, chat_id, dry_run=dry_run)
    manager = _load_manager()
    exit_code = 0

    try:
        wants_brief = False
        if token and chat_id:
            from bot.commands import CommandProcessor

            try:
                wants_brief = process_messages(tg, state, CommandProcessor(store, manager))
            except Exception:
                logger.exception("Processing Telegram messages failed.")
                exit_code = 1

        now_local = datetime.now(ZoneInfo(store.settings["timezone"]))
        scheduled = brief_is_due(now_local, store.settings["send_time"], state.last_brief_date)
        if scheduled or force_brief or wants_brief:
            from bot.brief import build_brief

            try:
                tg.send_message(build_brief(store, now_local, manager))
                if scheduled:
                    state.last_brief_date = now_local.strftime("%Y-%m-%d")
                logger.info("Brief sent (%s).", "scheduled" if scheduled else "on demand")
            except Exception:
                # Not marked as sent → the next run inside the window retries.
                logger.exception("Building/sending the brief failed.")
                exit_code = 1
        else:
            logger.info("Brief not due (now %s, send time %s, last sent %s).",
                        now_local.strftime("%H:%M"), store.settings["send_time"],
                        state.last_brief_date or "never")
    finally:
        if not dry_run:
            if store.dirty:
                store.save()
            state.save()
    return exit_code


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="TradingPal Telegram bot")
    parser.add_argument("--dry-run", action="store_true", help="print instead of sending; don't save state")
    parser.add_argument("--force-brief", action="store_true", help="send the brief regardless of schedule")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run, force_brief=args.force_brief))


if __name__ == "__main__":
    main()
