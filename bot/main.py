"""
TradingPal Telegram bot — one run = read the messages received since the last
run (watchlist changes), then send the daily brief. GitHub Actions runs it once
each morning; extra runs only happen when started manually.

Usage:
    python -m bot.main                 # scheduled run: brief only if it's due
    python -m bot.main --force-brief   # manual run: send the brief now
    python -m bot.main --dry-run       # print messages instead of sending, don't save
"""
import argparse
import logging
import sys

import config
from bot.schedule import brief_is_due, local_now
from bot.storage import BotState, WatchlistStore
from bot.telegram import TelegramClient

logger = logging.getLogger("bot")


def process_messages(tg: TelegramClient, state: BotState, processor) -> None:
    """Handle pending messages from the owner (Telegram keeps them for 24h)."""
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
        tg.send_message(processor.handle(text))


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
        if token and chat_id:
            from bot.commands import CommandProcessor

            try:
                process_messages(tg, state, CommandProcessor(store, manager))
            except Exception:
                logger.exception("Processing Telegram messages failed.")
                exit_code = 1

        now_local = local_now(store)
        scheduled = brief_is_due(now_local, store.settings["send_time"], state.last_brief_date)
        if scheduled or force_brief:
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
