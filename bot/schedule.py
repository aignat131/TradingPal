"""
When is the daily brief due?

Stdlib-only so the workflow can run it as a cheap gate *before* installing
dependencies:  python3 -m bot.schedule  → prints "due=true|false"
(appended to $GITHUB_OUTPUT when set).

GitHub cron is UTC-only, so the workflow fires at both 05:00 and 06:00 UTC to
hit 08:00 Europe/Bucharest in summer (UTC+3) and winter (UTC+2); this gate
lets exactly one of them do the work.
"""
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from bot.storage import BotState, WatchlistStore

# Give up on a missed morning brief after this long (e.g. an Actions outage).
BRIEF_WINDOW = timedelta(hours=4)


def brief_is_due(now_local: datetime, send_time: str, last_brief_date: str) -> bool:
    hour, minute = (int(x) for x in send_time.split(":"))
    scheduled = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    today = now_local.strftime("%Y-%m-%d")
    return last_brief_date != today and scheduled <= now_local < scheduled + BRIEF_WINDOW


def local_now(store: WatchlistStore) -> datetime:
    return datetime.now(ZoneInfo(store.settings["timezone"]))


def main() -> None:
    store = WatchlistStore()
    due = brief_is_due(local_now(store), store.settings["send_time"], BotState().last_brief_date)
    line = f"due={'true' if due else 'false'}"
    print(line)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
