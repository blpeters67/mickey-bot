#!/usr/bin/env python3
"""
dbcheck.py — diagnose "no such table" errors.

Run this on the box the bot actually runs on, with the same interpreter:

    venv/bin/python dbcheck.py

It answers three questions:
  1. Which file does the bot's settings point at? (absolute path + cwd)
  2. Does init_db() actually succeed, or does executescript abort partway?
  3. After init_db(), which tables really exist in that file?

Plus it scans your home dir for stray bot.db files, since a relative
`./bot.db` launched from the wrong cwd is a classic cause of this.
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

import aiosqlite

from core.config import settings
from core import db

# Tables the current schema is supposed to create. Anything missing after a
# successful init_db() means executescript didn't run that statement.
EXPECTED = [
    "guild_prefix", "guild_settings", "reminders", "playlists",
    "playlist_links", "emoji_usage", "sticker_usage", "message_stats",
    "voice_settings", "voice_users", "app_settings", "emoji_user_usage",
    "liked_songs", "track_plays", "quiz_excludes", "quiz_scores",
    "lastfm_users", "starboard_settings", "starboard_posts",
]


async def tables_in(path: str) -> set[str]:
    async with aiosqlite.connect(path) as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return {row[0] for row in await cur.fetchall()}


# Tables whose row counts reveal whether a file holds the bot's *real* data —
# used to tell the live database apart from an empty stray one.
_DATA_TABLES = ("guild_prefix", "playlists", "liked_songs", "app_settings")


async def data_signal(path: str) -> str:
    """A short 'how much real data is in here' summary for a bot.db file."""
    parts = []
    try:
        async with aiosqlite.connect(path) as conn:
            have = {r[0] for r in await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            for t in _DATA_TABLES:
                if t not in have:
                    parts.append(f"{t}=∅")
                    continue
                cur = await conn.execute(f"SELECT COUNT(*) FROM {t}")
                parts.append(f"{t}={(await cur.fetchone())[0]}")
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e}"
    return "  ".join(parts)


async def main() -> int:
    raw = settings.db_path
    absolute = os.path.abspath(raw)
    print("=" * 70)
    print(f"cwd                : {os.getcwd()}")
    print(f"settings.db_path   : {raw!r}")
    print(f"  -> absolute      : {absolute}")
    print(f"  -> exists        : {os.path.exists(absolute)}")
    print("=" * 70)

    # 1. Does init_db() succeed? This is the critical test — if executescript
    #    aborts on a bad statement, the exception shows up right here with the
    #    exact SQL that failed.
    print("\n[1] Running init_db() ...")
    try:
        await db.init_db()
        print("    init_db() returned WITHOUT error.")
    except Exception:
        print("    init_db() RAISED — this is the bug. Traceback:\n")
        traceback.print_exc()
        print("\n    ^ executescript aborted here; every table defined AFTER the")
        print("      failing statement was never created. Fix that statement in")
        print("      core/db.py's _SCHEMA (look for a stray char / bad quote).")
        return 1

    # 2. What actually exists now?
    have = await tables_in(absolute)
    missing = [t for t in EXPECTED if t not in have]
    print(f"\n[2] Tables in {absolute}:")
    print(f"    present ({len(have)}): {', '.join(sorted(have)) or '(none)'}")
    if missing:
        print(f"    MISSING ({len(missing)}): {', '.join(missing)}")
        print("\n    init_db() succeeded but these are still missing — that means")
        print("    the bot process reads a DIFFERENT file than this one. Compare")
        print("    the absolute path above with the stray files below.")
    else:
        print("    All expected tables present. ✅")
    print(f"    data: {await data_signal(absolute)}")

    # 3. Any other bot.db files lurking? (relative-path / wrong-cwd trap.)
    print("\n[3] Scanning for other bot.db files under your home dir ...")
    print("    (The file with the biggest row counts / newest mtime is the one")
    print("     the bot has really been writing to — point DB_PATH at it, or")
    print("     copy it over the anchored path above, if they differ.)")
    home = Path.home()
    found = 0
    import datetime
    for p in home.rglob("bot.db"):
        found += 1
        mtime = p.stat().st_mtime
        when = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        same = "  <- anchored path" if os.path.abspath(str(p)) == absolute else ""
        print(f"    {p}{same}")
        print(f"        mtime={when}   {await data_signal(str(p))}")
    if found <= 1:
        print("    (only the expected one — no wrong-cwd duplicate)")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
