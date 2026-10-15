"""
Async SQLite layer. Replaces the two JSON files (uploads.json, playlists.json)
plus adds per-guild prefix and settings tables.

Why SQLite over JSON:
  * concurrent writes don't corrupt the file
  * never blocks the event loop (aiosqlite runs queries in a thread)
  * trivial migrations as the schema grows
  * indexes make the lookups fast at any size

Call `init_db()` once at startup. After that, every public function in here
opens its own short-lived connection — simple, safe, and good enough for
a Discord bot's traffic level.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import aiosqlite

from . import recurrence
from .config import settings
from .logging_setup import get_logger

log = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS guild_prefix (
    guild_id INTEGER PRIMARY KEY,
    prefix   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS guild_settings (
    guild_id           INTEGER PRIMARY KEY,
    idle_seconds       INTEGER,                       -- NULL = use default; 0 = disabled
    leave_when_alone   INTEGER,                       -- NULL = use default; 0/1
    autoplay           INTEGER,                       -- NULL = default (off); 0/1
    sponsorblock       INTEGER,                       -- NULL = default (on);  0/1
    autojoin           INTEGER                        -- NULL = default (off); 0/1
);

CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER,                              -- NULL for DM reminders
    channel_id  INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    message     TEXT    NOT NULL,
    due_at      REAL    NOT NULL,
    created_at  REAL    NOT NULL,
    repeat_kind TEXT,                                 -- NULL = one-shot; else interval/clock
    repeat_spec TEXT,                                 -- seconds, or days:HH:MM
    fire_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at);

-- Per-user, cross-guild preferences. Timezone drives calendar reminders
-- ("every weekday 8:30" is meaningless without one) and defaults to UTC.
-- tts_voice is the voice -tts speaks in for that person (NULL = default).
CREATE TABLE IF NOT EXISTS user_settings (
    user_id   INTEGER PRIMARY KEY,
    timezone  TEXT,
    tts_voice TEXT
);

-- Accrued voice-channel time, one row per (guild, user, channel). Only
-- durations are stored: no audio, no timestamps of individual joins.
CREATE TABLE IF NOT EXISTS voice_time (
    guild_id    INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    channel_id  INTEGER NOT NULL,
    seconds     REAL    NOT NULL DEFAULT 0,
    sessions    INTEGER NOT NULL DEFAULT 0,
    last_seen   REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id, channel_id)
);
CREATE INDEX IF NOT EXISTS idx_voice_time_guild ON voice_time(guild_id, seconds);

-- Sessions still open, so a restart mid-call doesn't lose the time. Rows are
-- written on join and consumed on leave (or on the next startup sweep).
CREATE TABLE IF NOT EXISTS voice_open_sessions (
    guild_id   INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    joined_at  REAL    NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);

-- Users who asked not to be counted. Presence here also blocks new writes.
CREATE TABLE IF NOT EXISTS voice_stats_optout (
    user_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS playlists (
    guild_id   INTEGER NOT NULL,
    name_key   TEXT    NOT NULL,           -- lowercased name, used for lookup
    name       TEXT    NOT NULL,           -- display name
    creator    TEXT    NOT NULL,
    created_at REAL    NOT NULL,
    source_url TEXT,                       -- original .txt URL if imported
    PRIMARY KEY (guild_id, name_key)
);

CREATE TABLE IF NOT EXISTS playlist_links (
    guild_id INTEGER NOT NULL,
    name_key TEXT    NOT NULL,
    position INTEGER NOT NULL,
    url      TEXT    NOT NULL,
    PRIMARY KEY (guild_id, name_key, position),
    FOREIGN KEY (guild_id, name_key)
        REFERENCES playlists (guild_id, name_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS emoji_usage (
    guild_id      INTEGER NOT NULL,
    emoji_id      INTEGER NOT NULL,
    name          TEXT    NOT NULL,
    is_native     INTEGER NOT NULL DEFAULT 0,    -- 1 if from this guild
    is_animated   INTEGER NOT NULL DEFAULT 0,
    in_messages   INTEGER NOT NULL DEFAULT 0,
    in_reactions  INTEGER NOT NULL DEFAULT 0,
    last_used     REAL    NOT NULL,
    PRIMARY KEY (guild_id, emoji_id)
);
CREATE INDEX IF NOT EXISTS idx_emoji_use_total
    ON emoji_usage(guild_id, in_messages, in_reactions);

CREATE TABLE IF NOT EXISTS sticker_usage (
    guild_id    INTEGER NOT NULL,
    sticker_id  INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    is_native   INTEGER NOT NULL DEFAULT 0,
    used_count  INTEGER NOT NULL DEFAULT 0,
    last_used   REAL    NOT NULL,
    PRIMARY KEY (guild_id, sticker_id)
);

CREATE TABLE IF NOT EXISTS message_stats (
    guild_id        INTEGER NOT NULL,
    channel_id      INTEGER NOT NULL,
    user_id         INTEGER NOT NULL,
    msg_count       INTEGER NOT NULL DEFAULT 0,
    last_message_at REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, channel_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_message_stats_guild ON message_stats(guild_id);

CREATE TABLE IF NOT EXISTS voice_settings (
    guild_id         INTEGER PRIMARY KEY,
    enabled          INTEGER NOT NULL DEFAULT 0,    -- voice control on for this guild?
    wake_word        TEXT,                          -- NULL = use the configured default
    feedback_default TEXT NOT NULL DEFAULT 'text',  -- fallback reply style for users who haven't chosen
    feedback_channel_id INTEGER                     -- where voice-command replies go (NULL = summon channel)
);

CREATE TABLE IF NOT EXISTS voice_users (
    guild_id  INTEGER NOT NULL,
    user_id   INTEGER NOT NULL,
    opted_in  INTEGER NOT NULL DEFAULT 0,
    feedback  TEXT,                                 -- NULL = use the guild default
    PRIMARY KEY (guild_id, user_id)
);

CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,                         -- host-wide settings (not per-guild)
    value TEXT NOT NULL
);

-- Per-(guild, user, emoji) usage. The aggregate `emoji_usage` above answers
-- "which emoji is used most"; this answers "who uses which emoji". Counts
-- emoji a user *sends* (in messages, plus their own reactions live). History
-- scans only attribute message emoji to authors — reactions can't be mapped to
-- users without an API call per reaction.
CREATE TABLE IF NOT EXISTS emoji_user_usage (
    guild_id    INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    emoji_id    INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    is_animated INTEGER NOT NULL DEFAULT 0,
    count       INTEGER NOT NULL DEFAULT 0,
    last_used   REAL    NOT NULL,
    PRIMARY KEY (guild_id, user_id, emoji_id)
);
CREATE INDEX IF NOT EXISTS idx_emoji_user ON emoji_user_usage(guild_id, user_id);

-- Personal, global liked songs (not per-guild — a user's likes follow them).
CREATE TABLE IF NOT EXISTS liked_songs (
    user_id   INTEGER NOT NULL,
    url       TEXT    NOT NULL,          -- webpage_url, dedupe key
    title     TEXT    NOT NULL,
    artist    TEXT,
    duration  INTEGER NOT NULL DEFAULT 0,
    liked_at  REAL    NOT NULL,
    PRIMARY KEY (user_id, url)
);
CREATE INDEX IF NOT EXISTS idx_liked_user ON liked_songs(user_id, liked_at);

-- One row per *fully played* track (natural completion only). Drives the
-- "wrapped" leaderboards and the music-quiz pool.
CREATE TABLE IF NOT EXISTS track_plays (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id     INTEGER NOT NULL,
    requester_id INTEGER,                -- NULL when unknown (e.g. autoplay)
    url          TEXT,
    title        TEXT    NOT NULL,
    artist       TEXT,
    duration     INTEGER NOT NULL DEFAULT 0,
    is_music     INTEGER NOT NULL DEFAULT 0,   -- passed the "real music" heuristic
    played_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_track_plays_guild ON track_plays(guild_id, played_at);

-- Per-guild manual "not a song" exclusions for the quiz, plus running scores.
CREATE TABLE IF NOT EXISTS quiz_excludes (
    guild_id INTEGER NOT NULL,
    url      TEXT    NOT NULL,
    PRIMARY KEY (guild_id, url)
);
CREATE TABLE IF NOT EXISTS quiz_scores (
    guild_id INTEGER NOT NULL,
    user_id  INTEGER NOT NULL,
    points   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
);

-- Per-user Last.fm link (global, like likes). pending_tok holds the request
-- token between `-lastfm link` and `-lastfm confirm`.
CREATE TABLE IF NOT EXISTS lastfm_users (
    user_id     INTEGER PRIMARY KEY,
    session_key TEXT,
    username    TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    pending_tok TEXT
);

-- Starboard: per-guild config + a map of source message -> board message so
-- counts can be updated and posts removed when stars drop.
CREATE TABLE IF NOT EXISTS starboard_settings (
    guild_id   INTEGER PRIMARY KEY,
    channel_id INTEGER,                  -- NULL = disabled
    threshold  INTEGER NOT NULL DEFAULT 1,
    emoji      TEXT    NOT NULL DEFAULT '⭐'
);
CREATE TABLE IF NOT EXISTS starboard_posts (
    guild_id      INTEGER NOT NULL,
    source_msg_id INTEGER NOT NULL,
    board_msg_id  INTEGER NOT NULL,
    star_count    INTEGER NOT NULL,
    PRIMARY KEY (guild_id, source_msg_id)
);

-- Learned lyric timing corrections, keyed by the upload itself (watch URL),
-- not by song. Two uploads of the same track need different offsets — that's
-- the whole point: one has an intro card, one doesn't.
--
-- Global rather than per-guild on purpose. The correct offset is a property of
-- the audio file, so a fix found in one server is right in every server, and
-- the alternative is every guild rediscovering the same number.
CREATE TABLE IF NOT EXISTS lyric_offsets (
    source_id  TEXT PRIMARY KEY,          -- webpage_url of the upload
    offset_s   REAL    NOT NULL,          -- seconds; + shows lyrics earlier
    votes      INTEGER NOT NULL DEFAULT 1,
    title      TEXT,                      -- for debugging by a human
    updated_at INTEGER NOT NULL
);

-- Index of separated instrumentals on disk. Separation costs minutes of CPU
-- per track, so the cache is the difference between karaoke being usable and
-- being a novelty. Keyed by upload AND model: switching models must not serve
-- the previous model's output.
--
-- `last_used_at` drives LRU eviction once the directory exceeds its size cap.
CREATE TABLE IF NOT EXISTS stem_cache (
    source_id    TEXT NOT NULL,
    model_file   TEXT NOT NULL,
    path         TEXT NOT NULL,
    bytes        INTEGER NOT NULL DEFAULT 0,
    duration     REAL    NOT NULL DEFAULT 0,
    title        TEXT,
    created_at   INTEGER NOT NULL,
    last_used_at INTEGER NOT NULL,
    PRIMARY KEY (source_id, model_file)
);
CREATE INDEX IF NOT EXISTS idx_stem_cache_lru ON stem_cache(last_used_at);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

async def _ensure_column(db, table: str, column: str, decl: str) -> None:
    """ALTER TABLE ... ADD COLUMN if it doesn't exist yet (poor-man's migration
    — CREATE TABLE IF NOT EXISTS won't touch tables that already exist)."""
    cur  = await db.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in await cur.fetchall()}
    if column not in cols:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        log.info("Migrated: added %s.%s", table, column)


def _iter_statements(script: str):
    """Yield individual SQL statements from a `;`-separated schema script.

    Strips `-- comments` before splitting: several comments in _SCHEMA contain
    a `;` ("NULL = default; 0/1"), which would otherwise chop a statement in
    half. No `--` appears inside a string literal in this schema, so removing
    everything from `--` to end-of-line per line is safe.
    """
    cleaned = "\n".join(
        (ln[:ln.find("--")] if "--" in ln else ln)
        for ln in script.splitlines()
    )
    for chunk in cleaned.split(";"):
        if chunk.strip():
            yield chunk.strip() + ";"


async def init_db() -> None:
    """Create tables if missing. Safe to call repeatedly.

    Each statement runs independently: a single malformed statement logs an
    error and is skipped rather than aborting the whole script and silently
    leaving every table after it uncreated (which is exactly how a stray
    character in _SCHEMA turns into a cascade of "no such table" errors).
    """
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")    # better concurrency
        await db.execute("PRAGMA foreign_keys=ON")
        failed = 0
        for stmt in _iter_statements(_SCHEMA):
            try:
                await db.execute(stmt)
            except Exception:
                failed += 1
                log.exception("Schema statement failed (skipping):\n%s",
                              stmt[:300])
        await _ensure_column(db, "guild_settings", "autoplay", "INTEGER")
        await _ensure_column(db, "guild_settings", "sponsorblock", "INTEGER")
        await _ensure_column(db, "guild_settings", "autojoin", "INTEGER")
        await _ensure_column(db, "reminders", "repeat_kind", "TEXT")
        await _ensure_column(db, "reminders", "repeat_spec", "TEXT")
        await _ensure_column(db, "reminders", "fire_count",
                             "INTEGER NOT NULL DEFAULT 0")
        await _ensure_column(db, "user_settings", "tts_voice", "TEXT")
        await _ensure_column(db, "voice_settings", "feedback_channel_id", "INTEGER")
        # Sweep link rows orphaned by the old delete path (which relied on a
        # CASCADE that never fired — see delete_playlist).
        await db.execute(
            "DELETE FROM playlist_links WHERE NOT EXISTS ("
            "  SELECT 1 FROM playlists p"
            "  WHERE p.guild_id = playlist_links.guild_id"
            "    AND p.name_key = playlist_links.name_key)"
        )
        await db.commit()
    if failed:
        log.error("init_db: %d schema statement(s) failed — see tracebacks above. "
                  "Those tables/indexes are missing.", failed)
    # Log the *absolute* path so a wrong-cwd relative `./bot.db` is obvious.
    log.info("Database ready at %s (abs: %s)",
             settings.db_path, os.path.abspath(settings.db_path))


# ──────────────────────────────────────────────────────────────────────────────
# Prefixes
# ──────────────────────────────────────────────────────────────────────────────

# get_prefix runs for every single message the bot can see, so it's cached
# in memory. The cache is only ever written through set_prefix, which keeps
# it coherent for the lifetime of the process.
_prefix_cache: dict[int, str] = {}


async def get_prefix(guild_id: int) -> str:
    """Return the prefix for this guild, or the default if none set."""
    cached = _prefix_cache.get(guild_id)
    if cached is not None:
        return cached
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT prefix FROM guild_prefix WHERE guild_id = ?",
            (guild_id,),
        )
        row = await cur.fetchone()
    prefix = row[0] if row else settings.default_prefix
    _prefix_cache[guild_id] = prefix
    return prefix


async def set_prefix(guild_id: int, prefix: str) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO guild_prefix (guild_id, prefix) VALUES (?, ?) "
            "ON CONFLICT(guild_id) DO UPDATE SET prefix = excluded.prefix",
            (guild_id, prefix),
        )
        await db.commit()
    _prefix_cache[guild_id] = prefix


# ──────────────────────────────────────────────────────────────────────────────
# Per-guild settings (auto-leave behaviour, etc.)
# ──────────────────────────────────────────────────────────────────────────────

async def get_guild_settings(guild_id: int) -> dict[str, Any]:
    """Return raw values (idle_seconds, leave_when_alone) for a guild.

    Either field may be None — that means the guild hasn't set it and the
    caller should apply its own default. We deliberately keep NULL as a
    distinct state from 0 so we can tell "disabled by user" apart from
    "never configured".
    """
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT idle_seconds, leave_when_alone, autoplay, sponsorblock, autojoin "
            "FROM guild_settings WHERE guild_id = ?",
            (guild_id,),
        )
        row = await cur.fetchone()
    if not row:
        return {"idle_seconds": None, "leave_when_alone": None,
                "autoplay": None, "sponsorblock": None, "autojoin": None}
    return {"idle_seconds": row[0], "leave_when_alone": row[1],
            "autoplay": row[2], "sponsorblock": row[3], "autojoin": row[4]}


async def _set_guild_flag(guild_id: int, column: str, enabled: bool | None) -> None:
    """Upsert one nullable 0/1 column on guild_settings. `column` is always a
    literal from this module, never user input."""
    val = None if enabled is None else (1 if enabled else 0)
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            f"""INSERT INTO guild_settings (guild_id, {column}) VALUES (?, ?)
                ON CONFLICT(guild_id) DO UPDATE SET {column} = excluded.{column}""",
            (guild_id, val),
        )
        await db.commit()


async def set_autoplay(guild_id: int, enabled: bool | None) -> None:
    await _set_guild_flag(guild_id, "autoplay", enabled)


async def set_sponsorblock(guild_id: int, enabled: bool | None) -> None:
    await _set_guild_flag(guild_id, "sponsorblock", enabled)


async def set_autojoin(guild_id: int, enabled: bool | None) -> None:
    await _set_guild_flag(guild_id, "autojoin", enabled)


async def set_idle_seconds(guild_id: int, seconds: int | None) -> None:
    """`seconds=None` clears the override (use default again).
    `seconds=0` disables idle-disconnect entirely for the guild."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO guild_settings (guild_id, idle_seconds) VALUES (?, ?)
               ON CONFLICT(guild_id) DO UPDATE SET idle_seconds = excluded.idle_seconds""",
            (guild_id, seconds),
        )
        await db.commit()


async def set_leave_when_alone(guild_id: int, enabled: bool | None) -> None:
    """`enabled=None` clears the override; `True`/`False` set explicitly."""
    val = None if enabled is None else (1 if enabled else 0)
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO guild_settings (guild_id, leave_when_alone) VALUES (?, ?)
               ON CONFLICT(guild_id) DO UPDATE SET leave_when_alone = excluded.leave_when_alone""",
            (guild_id, val),
        )
        await db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Playlists
# ──────────────────────────────────────────────────────────────────────────────

async def create_playlist(
    guild_id: int,
    name: str,
    *,
    creator: str,
    source_url: str | None = None,
) -> bool:
    """Returns True if created, False if a playlist with that key existed."""
    name_key = name.lower().strip()
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT 1 FROM playlists WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        if await cur.fetchone():
            return False
        await db.execute(
            "INSERT INTO playlists (guild_id, name_key, name, creator, "
            "created_at, source_url) VALUES (?, ?, ?, ?, ?, ?)",
            (guild_id, name_key, name, creator, time.time(), source_url),
        )
        await db.commit()
    return True


async def overwrite_playlist(
    guild_id: int,
    name: str,
    *,
    creator: str,
    source_url: str | None = None,
) -> None:
    """Replace (or create) a playlist's metadata, wiping existing links."""
    name_key = name.lower().strip()
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO playlists (guild_id, name_key, name, creator, "
            "created_at, source_url) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(guild_id, name_key) DO UPDATE SET "
            "name = excluded.name, creator = excluded.creator, "
            "created_at = excluded.created_at, source_url = excluded.source_url",
            (guild_id, name_key, name, creator, time.time(), source_url),
        )
        await db.execute(
            "DELETE FROM playlist_links WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        await db.commit()


async def delete_playlist(guild_id: int, name_key: str) -> bool:
    name_key = name_key.lower().strip()
    async with aiosqlite.connect(settings.db_path) as db:
        # Delete the links explicitly. The schema's ON DELETE CASCADE never
        # fires here: PRAGMA foreign_keys is per-connection and only init_db's
        # connection turned it on, so relying on it left orphaned link rows
        # that silently resurrected when a same-named playlist was recreated.
        await db.execute(
            "DELETE FROM playlist_links WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        cur = await db.execute(
            "DELETE FROM playlists WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        await db.commit()
        return cur.rowcount > 0


async def rename_playlist(guild_id: int, old_name: str, new_name: str) -> str:
    """Returns 'ok' | 'missing' | 'exists'."""
    old_key = old_name.lower().strip()
    new_key = new_name.lower().strip()
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT 1 FROM playlists WHERE guild_id = ? AND name_key = ?",
            (guild_id, old_key),
        )
        if not await cur.fetchone():
            return "missing"
        if old_key != new_key:
            cur = await db.execute(
                "SELECT 1 FROM playlists WHERE guild_id = ? AND name_key = ?",
                (guild_id, new_key),
            )
            if await cur.fetchone():
                return "exists"
        await db.execute(
            "UPDATE playlists SET name = ?, name_key = ? "
            "WHERE guild_id = ? AND name_key = ?",
            (new_name, new_key, guild_id, old_key),
        )
        await db.execute(
            "UPDATE playlist_links SET name_key = ? "
            "WHERE guild_id = ? AND name_key = ?",
            (new_key, guild_id, old_key),
        )
        await db.commit()
    return "ok"


async def get_playlist(guild_id: int, name: str) -> dict[str, Any] | None:
    """Returns {name, creator, created_at, source_url, links: [str, ...]} or None."""
    name_key = name.lower().strip()
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM playlists WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        meta = await cur.fetchone()
        if not meta:
            return None
        cur = await db.execute(
            "SELECT url FROM playlist_links WHERE guild_id = ? AND name_key = ? "
            "ORDER BY position",
            (guild_id, name_key),
        )
        links = [r[0] for r in await cur.fetchall()]
    out = dict(meta)
    out["links"] = links
    return out


async def list_playlists(guild_id: int) -> list[dict[str, Any]]:
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT p.*, COUNT(l.url) AS link_count FROM playlists p "
            "LEFT JOIN playlist_links l ON l.guild_id = p.guild_id "
            "AND l.name_key = p.name_key "
            "WHERE p.guild_id = ? GROUP BY p.guild_id, p.name_key "
            "ORDER BY p.created_at",
            (guild_id,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def add_playlist_links(
    guild_id: int, name: str, urls: list[str],
) -> tuple[int, list[str]]:
    """Append URLs (skipping duplicates) → (added_count, skipped_dupes)."""
    name_key = name.lower().strip()
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT MAX(position), COUNT(*) FROM playlist_links "
            "WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        row     = await cur.fetchone()
        next_pos = (row[0] or -1) + 1

        cur = await db.execute(
            "SELECT url FROM playlist_links WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        existing = {r[0] for r in await cur.fetchall()}

        added: list[str] = []
        dupes: list[str] = []
        for url in urls:
            if url in existing:
                dupes.append(url)
                continue
            existing.add(url)
            added.append(url)

        await db.executemany(
            "INSERT INTO playlist_links (guild_id, name_key, position, url) "
            "VALUES (?, ?, ?, ?)",
            [(guild_id, name_key, next_pos + i, u) for i, u in enumerate(added)],
        )
        await db.commit()
    return len(added), dupes


async def remove_playlist_links(
    guild_id: int, name: str, indices: set[int],
) -> list[str]:
    """1-based indices. Returns removed URLs and re-numbers positions."""
    name_key = name.lower().strip()
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT url FROM playlist_links WHERE guild_id = ? AND name_key = ? "
            "ORDER BY position",
            (guild_id, name_key),
        )
        urls = [r[0] for r in await cur.fetchall()]

        keep: list[str]    = []
        removed: list[str] = []
        for i, url in enumerate(urls, 1):
            (removed if i in indices else keep).append(url)

        await db.execute(
            "DELETE FROM playlist_links WHERE guild_id = ? AND name_key = ?",
            (guild_id, name_key),
        )
        await db.executemany(
            "INSERT INTO playlist_links (guild_id, name_key, position, url) "
            "VALUES (?, ?, ?, ?)",
            [(guild_id, name_key, i, u) for i, u in enumerate(keep)],
        )
        await db.commit()
    return removed


# ──────────────────────────────────────────────────────────────────────────────
# Reminders
# ──────────────────────────────────────────────────────────────────────────────

async def add_reminder(
    *, guild_id: int | None, channel_id: int, user_id: int,
    message: str, due_at: float,
    repeat_kind: str | None = None, repeat_spec: str | None = None,
) -> int:
    """Insert a reminder, returns its id.

    `repeat_kind`/`repeat_spec` come from a `recurrence.Schedule`; leaving
    them None makes an ordinary one-shot reminder.
    """
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "INSERT INTO reminders (guild_id, channel_id, user_id, message, "
            "due_at, created_at, repeat_kind, repeat_spec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (guild_id, channel_id, user_id, message, due_at, time.time(),
             repeat_kind, repeat_spec),
        )
        await db.commit()
        return cur.lastrowid


async def count_reminders_for(user_id: int, *, repeating: bool | None = None) -> int:
    """Pending reminder count, optionally restricted to repeating/one-shot."""
    sql    = "SELECT COUNT(*) FROM reminders WHERE user_id = ?"
    params: tuple = (user_id,)
    if repeating is True:
        sql += " AND repeat_kind IS NOT NULL"
    elif repeating is False:
        sql += " AND repeat_kind IS NULL"
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(sql, params)
        return (await cur.fetchone())[0]


# ── per-user settings ────────────────────────────────────────────────────────

async def get_user_timezone(user_id: int) -> str | None:
    """IANA zone name for a user, or None if they've never set one."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT timezone FROM user_settings WHERE user_id = ?", (user_id,),
        )
        row = await cur.fetchone()
    return row[0] if row and row[0] else None


async def set_user_timezone(user_id: int, tz: str | None) -> None:
    """Store (or with tz=None, clear) a user's timezone."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO user_settings (user_id, timezone) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET timezone = excluded.timezone",
            (user_id, tz),
        )
        await db.commit()


async def get_user_tts_voice(user_id: int) -> str | None:
    """The voice id a user picked with -ttsvoice, or None for the default."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT tts_voice FROM user_settings WHERE user_id = ?", (user_id,),
        )
        row = await cur.fetchone()
    return row[0] if row and row[0] else None


async def set_user_tts_voice(user_id: int, voice: str | None) -> None:
    """Store (or with voice=None, clear) a user's TTS voice. Leaves their
    other settings (timezone) alone."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO user_settings (user_id, tts_voice) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET tts_voice = excluded.tts_voice",
            (user_id, voice),
        )
        await db.commit()


async def count_reminders(user_id: int) -> int:
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM reminders WHERE user_id = ?", (user_id,),
        )
        return (await cur.fetchone())[0]


async def list_reminders(user_id: int) -> list[dict[str, Any]]:
    """All pending reminders for one user, soonest first."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM reminders WHERE user_id = ? ORDER BY due_at",
            (user_id,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def delete_reminder(reminder_id: int, user_id: int) -> bool:
    """Delete one reminder — only if it belongs to `user_id`."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "DELETE FROM reminders WHERE id = ? AND user_id = ?",
            (reminder_id, user_id),
        )
        await db.commit()
        return cur.rowcount > 0


# ──────────────────────────────────────────────────────────────────────────────
# Voice time tracking
# ──────────────────────────────────────────────────────────────────────────────

async def voice_optout_ids() -> set[int]:
    """Every user who has opted out of voice-time tracking."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute("SELECT user_id FROM voice_stats_optout")
        return {r[0] for r in await cur.fetchall()}


async def set_voice_optout(user_id: int, opted_out: bool) -> None:
    """Opt in or out. Opting out also erases the user's existing rows —
    an opt-out that leaves the data behind isn't really one."""
    async with aiosqlite.connect(settings.db_path) as db:
        if opted_out:
            await db.execute(
                "INSERT OR IGNORE INTO voice_stats_optout (user_id) VALUES (?)",
                (user_id,),
            )
            await db.execute("DELETE FROM voice_time WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM voice_open_sessions WHERE user_id = ?",
                             (user_id,))
        else:
            await db.execute("DELETE FROM voice_stats_optout WHERE user_id = ?",
                             (user_id,))
        await db.commit()


async def voice_session_open(guild_id: int, user_id: int, channel_id: int,
                             at: float) -> None:
    """Record that someone is in a voice channel as of `at`.

    Persisted rather than held in memory so a restart mid-call doesn't throw
    the session away — `voice_sweep_open` reconciles on the next startup.
    """
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO voice_open_sessions (guild_id, user_id, channel_id, "
            "joined_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(guild_id, user_id) DO UPDATE SET "
            "channel_id = excluded.channel_id, joined_at = excluded.joined_at",
            (guild_id, user_id, channel_id, at),
        )
        await db.commit()


async def voice_session_close(guild_id: int, user_id: int, at: float,
                              *, max_seconds: float | None = None) -> float:
    """Close an open session and bank the elapsed time. Returns the seconds
    added (0 if there was no open session).

    `max_seconds` clamps the credit — used by the startup sweep, where a
    session left open by a crash could otherwise bank days of phantom time.
    """
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT channel_id, joined_at FROM voice_open_sessions "
            "WHERE guild_id = ? AND user_id = ?", (guild_id, user_id),
        )
        row = await cur.fetchone()
        if row is None:
            return 0.0
        elapsed = max(0.0, at - row["joined_at"])
        if max_seconds is not None:
            elapsed = min(elapsed, max_seconds)
        await db.execute(
            "DELETE FROM voice_open_sessions WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )
        if elapsed > 0:
            await db.execute(
                "INSERT INTO voice_time (guild_id, user_id, channel_id, seconds, "
                "sessions, last_seen) VALUES (?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(guild_id, user_id, channel_id) DO UPDATE SET "
                "seconds = seconds + excluded.seconds, "
                "sessions = sessions + 1, last_seen = excluded.last_seen",
                (guild_id, user_id, row["channel_id"], elapsed, at),
            )
        await db.commit()
        return elapsed


async def voice_open_sessions() -> list[dict[str, Any]]:
    """Every session the DB still believes is running."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM voice_open_sessions")
        return [dict(r) for r in await cur.fetchall()]


async def voice_leaderboard(guild_id: int, limit: int = 100) -> list[dict[str, Any]]:
    """Total voice seconds per user in a guild, busiest first."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT user_id, SUM(seconds) AS seconds, SUM(sessions) AS sessions, "
            "MAX(last_seen) AS last_seen FROM voice_time WHERE guild_id = ? "
            "GROUP BY user_id ORDER BY seconds DESC LIMIT ?",
            (guild_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]


async def voice_channel_totals(guild_id: int, limit: int = 25) -> list[dict[str, Any]]:
    """Total voice seconds per channel in a guild, busiest first."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT channel_id, SUM(seconds) AS seconds, SUM(sessions) AS sessions "
            "FROM voice_time WHERE guild_id = ? GROUP BY channel_id "
            "ORDER BY seconds DESC LIMIT ?",
            (guild_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]


async def voice_user_detail(guild_id: int, user_id: int) -> list[dict[str, Any]]:
    """One user's voice time broken down by channel."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT channel_id, seconds, sessions, last_seen FROM voice_time "
            "WHERE guild_id = ? AND user_id = ? ORDER BY seconds DESC",
            (guild_id, user_id),
        )
        return [dict(r) for r in await cur.fetchall()]


async def voice_guild_total(guild_id: int) -> tuple[float, int]:
    """(total seconds, distinct users) for a guild."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT COALESCE(SUM(seconds), 0), COUNT(DISTINCT user_id) "
            "FROM voice_time WHERE guild_id = ?", (guild_id,),
        )
        row = await cur.fetchone()
    return float(row[0]), int(row[1])


async def pop_due_reminders(now: float) -> list[dict[str, Any]]:
    """Fetch every reminder due by `now`, retiring or rescheduling each.

    One-shots are deleted; repeating ones have `due_at` advanced to their next
    occurrence and `fire_count` bumped. Both happen in the same transaction as
    the read, so a crash mid-delivery loses at most the current batch rather
    than replaying it forever on restart.

    The returned rows carry the `due_at` that just fired, plus a
    `next_due` key (None for one-shots) so the caller can tell the user when
    a repeating reminder will come round again.
    """
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM reminders WHERE due_at <= ? ORDER BY due_at", (now,),
        )
        rows = [dict(r) for r in await cur.fetchall()]
        if not rows:
            return rows

        # Timezones are needed to advance clock schedules; fetch the ones we
        # actually need in a single pass rather than per row.
        user_ids = {r["user_id"] for r in rows}
        cur = await db.execute(
            "SELECT user_id, timezone FROM user_settings WHERE user_id IN "
            f"({','.join('?' * len(user_ids))})", tuple(user_ids),
        )
        tzs = {r["user_id"]: r["timezone"] for r in await cur.fetchall()}

        drop: list[int] = []
        for r in rows:
            sched = recurrence.Schedule.from_db(r["repeat_kind"], r["repeat_spec"])
            if sched is None:
                r["next_due"] = None
                drop.append(r["id"])
                continue
            try:
                nxt = sched.next_after(r["due_at"], tzs.get(r["user_id"]) or "UTC")
            except Exception:
                # A corrupt spec must not wedge the delivery loop in a retry
                # storm — deliver this one, then retire it.
                log.exception("Reminder #%s has an unusable repeat spec %r",
                              r["id"], r["repeat_spec"])
                r["next_due"] = None
                drop.append(r["id"])
                continue
            r["next_due"] = nxt
            await db.execute(
                "UPDATE reminders SET due_at = ?, fire_count = fire_count + 1 "
                "WHERE id = ?", (nxt, r["id"]),
            )
        if drop:
            await db.execute(
                f"DELETE FROM reminders WHERE id IN ({','.join('?' * len(drop))})",
                tuple(drop),
            )
        await db.commit()
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Emoji / sticker usage tracking
# ──────────────────────────────────────────────────────────────────────────────

async def record_emoji_use(
    *, guild_id: int, emoji_id: int, name: str,
    is_native: bool, is_animated: bool,
    kind: str, now: float,
) -> None:
    """Bump in_messages or in_reactions for one (guild, emoji) row.

    Upserts so the very first time an emoji is seen creates the row.
    `kind` is 'message' or 'reaction'.
    """
    msg_inc = 1 if kind == "message"  else 0
    rxn_inc = 1 if kind == "reaction" else 0
    column  = "in_messages" if kind == "message" else "in_reactions"
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            f"""INSERT INTO emoji_usage
                  (guild_id, emoji_id, name, is_native, is_animated,
                   in_messages, in_reactions, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(guild_id, emoji_id) DO UPDATE SET
                  {column} = {column} + 1,
                  name = excluded.name,
                  is_native = excluded.is_native,
                  is_animated = excluded.is_animated,
                  last_used = excluded.last_used""",
            (guild_id, emoji_id, name, int(is_native), int(is_animated),
             msg_inc, rxn_inc, now),
        )
        await db.commit()


async def record_sticker_use(
    *, guild_id: int, sticker_id: int, name: str,
    is_native: bool, now: float,
) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO sticker_usage
                  (guild_id, sticker_id, name, is_native, used_count, last_used)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(guild_id, sticker_id) DO UPDATE SET
                  used_count = used_count + 1,
                  name = excluded.name,
                  is_native = excluded.is_native,
                  last_used = excluded.last_used""",
            (guild_id, sticker_id, name, int(is_native), now),
        )
        await db.commit()


async def list_emoji_usage(
    guild_id: int, *,
    filter_mode: str = "all",     # 'all' | 'native' | 'external'
    sort_mode:   str = "used",    # 'used' | 'alpha' | 'recent'
) -> list[dict]:
    where = "WHERE guild_id = ?"
    if filter_mode == "native":   where += " AND is_native = 1"
    elif filter_mode == "external": where += " AND is_native = 0"

    if sort_mode == "alpha":
        order = "ORDER BY name COLLATE NOCASE"
    elif sort_mode == "recent":
        order = "ORDER BY last_used DESC"
    else:
        order = "ORDER BY (in_messages + in_reactions) DESC, last_used DESC"

    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"SELECT * FROM emoji_usage {where} {order}",
            (guild_id,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_sticker_usage(
    guild_id: int, *,
    filter_mode: str = "all",
    sort_mode:   str = "used",
) -> list[dict]:
    where = "WHERE guild_id = ?"
    if filter_mode == "native":   where += " AND is_native = 1"
    elif filter_mode == "external": where += " AND is_native = 0"

    if sort_mode == "alpha":
        order = "ORDER BY name COLLATE NOCASE"
    elif sort_mode == "recent":
        order = "ORDER BY last_used DESC"
    else:
        order = "ORDER BY used_count DESC, last_used DESC"

    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"SELECT * FROM sticker_usage {where} {order}",
            (guild_id,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def bulk_record_scan(
    *, guild_id: int,
    emojis:   dict[int, dict],
    stickers: dict[int, dict],
) -> None:
    """One-shot upsert for a historical scan.

    Adds the in-memory counts onto whatever's already in the table.
    Re-running a scan double-counts — there's no de-dupe by message id;
    the assumption is you run this once for backfill and let the live
    listeners handle everything afterward.

    Both dicts map id → {name, is_native, is_animated?, msg_count|used_count,
    rxn_count?, last_used}. One transaction, fast even for big servers.
    """
    async with aiosqlite.connect(settings.db_path) as db:
        for emoji_id, e in emojis.items():
            await db.execute(
                """INSERT INTO emoji_usage
                      (guild_id, emoji_id, name, is_native, is_animated,
                       in_messages, in_reactions, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(guild_id, emoji_id) DO UPDATE SET
                      in_messages  = in_messages  + excluded.in_messages,
                      in_reactions = in_reactions + excluded.in_reactions,
                      name         = excluded.name,
                      is_native    = excluded.is_native,
                      is_animated  = excluded.is_animated,
                      last_used    = MAX(emoji_usage.last_used, excluded.last_used)""",
                (guild_id, emoji_id, e["name"],
                 int(e["is_native"]), int(e["is_animated"]),
                 e["msg_count"], e["rxn_count"], e["last_used"]),
            )
        for sticker_id, s in stickers.items():
            await db.execute(
                """INSERT INTO sticker_usage
                      (guild_id, sticker_id, name, is_native,
                       used_count, last_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(guild_id, sticker_id) DO UPDATE SET
                      used_count = used_count + excluded.used_count,
                      name       = excluded.name,
                      is_native  = excluded.is_native,
                      last_used  = MAX(sticker_usage.last_used, excluded.last_used)""",
                (guild_id, sticker_id, s["name"],
                 int(s["is_native"]), s["used_count"], s["last_used"]),
            )
        await db.commit()


async def record_user_emoji(
    *, guild_id: int, user_id: int, emoji_id: int, name: str,
    is_animated: bool, now: float,
) -> None:
    """Bump one (guild, user, emoji) tally — how many times this person used
    this emoji. Called live for message emoji (author) and reactions (reactor)."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO emoji_user_usage
                  (guild_id, user_id, emoji_id, name, is_animated, count, last_used)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(guild_id, user_id, emoji_id) DO UPDATE SET
                  count       = count + 1,
                  name        = excluded.name,
                  is_animated = excluded.is_animated,
                  last_used   = MAX(last_used, excluded.last_used)""",
            (guild_id, user_id, emoji_id, name, int(is_animated), now),
        )
        await db.commit()


async def user_emoji_stats(guild_id: int, user_id: int) -> dict:
    """One member's emoji usage: {total, distinct, last_used,
    emojis: [{emoji_id, name, is_animated, count}, ...] most-used first}."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT emoji_id, name, is_animated, count, last_used
               FROM emoji_user_usage WHERE guild_id = ? AND user_id = ?
               ORDER BY count DESC, last_used DESC""",
            (guild_id, user_id),
        )
        rows = [dict(r) for r in await cur.fetchall()]
    return {
        "total": sum(r["count"] for r in rows),
        "distinct": len(rows),
        "last_used": max((r["last_used"] for r in rows), default=0.0),
        "emojis": rows,
    }


async def bulk_record_user_emoji(
    guild_id: int, user_emojis: dict[tuple[int, int], dict],
) -> None:
    """One-transaction upsert of a history scan's per-user emoji counts. Adds
    onto existing counts (re-running double-counts — run once for backfill).
    `user_emojis` maps (user_id, emoji_id) -> {name, is_animated, count, last_used}."""
    async with aiosqlite.connect(settings.db_path) as db:
        for (user_id, emoji_id), e in user_emojis.items():
            await db.execute(
                """INSERT INTO emoji_user_usage
                      (guild_id, user_id, emoji_id, name, is_animated, count, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(guild_id, user_id, emoji_id) DO UPDATE SET
                      count       = count + excluded.count,
                      name        = excluded.name,
                      is_animated = excluded.is_animated,
                      last_used   = MAX(emoji_user_usage.last_used, excluded.last_used)""",
                (guild_id, user_id, emoji_id, e["name"],
                 int(e["is_animated"]), e["count"], e["last_used"]),
            )
        await db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Message statistics — who's talking, and where
# ──────────────────────────────────────────────────────────────────────────────

async def record_message(
    *, guild_id: int, channel_id: int, user_id: int, now: float,
) -> None:
    """Bump one (guild, channel, user) message tally. Called from the live
    on_message listener for every human message in a guild text channel."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO message_stats
                  (guild_id, channel_id, user_id, msg_count, last_message_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(guild_id, channel_id, user_id) DO UPDATE SET
                  msg_count       = msg_count + 1,
                  last_message_at = MAX(last_message_at, excluded.last_message_at)""",
            (guild_id, channel_id, user_id, now),
        )
        await db.commit()


async def top_senders(
    guild_id: int, *, channel_id: int | None = None, limit: int = 10,
) -> list[dict]:
    """Top users by message count — server-wide, or within one channel.
    Returns [{user_id, total, last_message_at}, ...], highest first."""
    where  = "WHERE guild_id = ?"
    params: list[Any] = [guild_id]
    if channel_id is not None:
        where += " AND channel_id = ?"
        params.append(channel_id)
    params.append(limit)
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT user_id,
                       SUM(msg_count)       AS total,
                       MAX(last_message_at) AS last_message_at
                FROM message_stats {where}
                GROUP BY user_id
                ORDER BY total DESC, last_message_at DESC
                LIMIT ?""",
            params,
        )
        return [dict(r) for r in await cur.fetchall()]


async def top_channels(guild_id: int, *, limit: int = 10) -> list[dict]:
    """Busiest channels by message count. [{channel_id, total}, ...]."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT channel_id, SUM(msg_count) AS total
               FROM message_stats WHERE guild_id = ?
               GROUP BY channel_id ORDER BY total DESC LIMIT ?""",
            (guild_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]


async def guild_message_totals(guild_id: int) -> dict:
    """Whole-guild rollup: {total, users, channels}."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            """SELECT COALESCE(SUM(msg_count), 0),
                      COUNT(DISTINCT user_id),
                      COUNT(DISTINCT channel_id)
               FROM message_stats WHERE guild_id = ?""",
            (guild_id,),
        )
        total, users, channels = await cur.fetchone()
    return {"total": total, "users": users, "channels": channels}


async def channel_message_total(guild_id: int, channel_id: int) -> int:
    """Total messages tracked in one channel."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT COALESCE(SUM(msg_count), 0) FROM message_stats "
            "WHERE guild_id = ? AND channel_id = ?",
            (guild_id, channel_id),
        )
        return (await cur.fetchone())[0]


async def user_message_stats(guild_id: int, user_id: int) -> dict:
    """One user's totals, server rank, and per-channel breakdown.
    {total, last_message_at, rank, channels: [{channel_id, count}, ...]}.
    rank is 0 when the user has no tracked messages."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT channel_id, msg_count, last_message_at
               FROM message_stats WHERE guild_id = ? AND user_id = ?
               ORDER BY msg_count DESC""",
            (guild_id, user_id),
        )
        rows  = [dict(r) for r in await cur.fetchall()]
        total = sum(r["msg_count"] for r in rows)
        last  = max((r["last_message_at"] for r in rows), default=0.0)
        # Rank = 1 + how many users out-posted this one.
        rank = 0
        if total > 0:
            cur = await db.execute(
                """SELECT COUNT(*) FROM (
                       SELECT user_id, SUM(msg_count) AS t
                       FROM message_stats WHERE guild_id = ?
                       GROUP BY user_id HAVING t > ?
                   )""",
                (guild_id, total),
            )
            rank = (await cur.fetchone())[0] + 1
    channels = [{"channel_id": r["channel_id"], "count": r["msg_count"]}
                for r in rows]
    return {"total": total, "last_message_at": last,
            "rank": rank, "channels": channels}


async def replace_message_stats(
    guild_id: int, *, scanned_channel_ids: set[int],
    counts: dict[tuple[int, int], dict],
) -> None:
    """Overwrite tallies for the scanned channels with authoritative counts
    from a full history walk. Only rows for `scanned_channel_ids` are cleared,
    so channels the bot couldn't read keep their live-tracked data. `counts`
    maps (channel_id, user_id) -> {'count': int, 'last': float}. One
    transaction, so a re-run can't double-count."""
    async with aiosqlite.connect(settings.db_path) as db:
        for cid in scanned_channel_ids:
            await db.execute(
                "DELETE FROM message_stats WHERE guild_id = ? AND channel_id = ?",
                (guild_id, cid),
            )
        await db.executemany(
            """INSERT INTO message_stats
                  (guild_id, channel_id, user_id, msg_count, last_message_at)
                VALUES (?, ?, ?, ?, ?)""",
            [(guild_id, cid, uid, c["count"], c["last"])
             for (cid, uid), c in counts.items()],
        )
        await db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Voice control — per-guild settings + per-user opt-in
# ──────────────────────────────────────────────────────────────────────────────

async def get_voice_settings(guild_id: int) -> dict[str, Any]:
    """{enabled, wake_word, feedback_default, feedback_channel_id}. wake_word is
    None when the guild hasn't overridden it (caller applies its default);
    feedback_channel_id None means "reply where the bot was summoned"."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT enabled, wake_word, feedback_default, feedback_channel_id "
            "FROM voice_settings WHERE guild_id = ?",
            (guild_id,),
        )
        row = await cur.fetchone()
    if not row:
        return {"enabled": False, "wake_word": None, "feedback_default": "text",
                "feedback_channel_id": None}
    return {"enabled": bool(row[0]), "wake_word": row[1],
            "feedback_default": row[2], "feedback_channel_id": row[3]}


async def set_voice_feedback_channel(guild_id: int, channel_id: int | None) -> None:
    await _set_voice_field(guild_id, "feedback_channel_id", channel_id)


async def list_voice_enabled_guilds() -> set[int]:
    """Guild ids with voice control switched on (restored at startup)."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute("SELECT guild_id FROM voice_settings WHERE enabled = 1")
        return {r[0] for r in await cur.fetchall()}


async def _set_voice_field(guild_id: int, column: str, value: Any) -> None:
    """Upsert one column on voice_settings. `column` is always a literal."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            f"""INSERT INTO voice_settings (guild_id, {column}) VALUES (?, ?)
                ON CONFLICT(guild_id) DO UPDATE SET {column} = excluded.{column}""",
            (guild_id, value),
        )
        await db.commit()


async def set_voice_enabled(guild_id: int, enabled: bool) -> None:
    await _set_voice_field(guild_id, "enabled", int(enabled))


async def set_voice_wake(guild_id: int, wake_word: str | None) -> None:
    await _set_voice_field(guild_id, "wake_word", wake_word)


async def set_voice_feedback_default(guild_id: int, mode: str) -> None:
    await _set_voice_field(guild_id, "feedback_default", mode)


async def get_voice_user(guild_id: int, user_id: int) -> dict[str, Any]:
    """{opted_in: bool, feedback: str|None} for one member. feedback None means
    'use the guild default'."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT opted_in, feedback FROM voice_users "
            "WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )
        row = await cur.fetchone()
    if not row:
        return {"opted_in": False, "feedback": None}
    return {"opted_in": bool(row[0]), "feedback": row[1]}


async def set_voice_optin(guild_id: int, user_id: int, opted_in: bool) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO voice_users (guild_id, user_id, opted_in)
                VALUES (?, ?, ?)
                ON CONFLICT(guild_id, user_id)
                DO UPDATE SET opted_in = excluded.opted_in""",
            (guild_id, user_id, int(opted_in)),
        )
        await db.commit()


async def set_voice_user_feedback(
    guild_id: int, user_id: int, mode: str | None,
) -> None:
    """Set one member's personal feedback style (None → fall back to guild
    default). Leaves their opt-in state untouched."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO voice_users (guild_id, user_id, opted_in, feedback)
                VALUES (?, ?, 0, ?)
                ON CONFLICT(guild_id, user_id)
                DO UPDATE SET feedback = excluded.feedback""",
            (guild_id, user_id, mode),
        )
        await db.commit()


async def list_voice_optins(guild_id: int) -> set[int]:
    """User ids who've opted into voice control in this guild. The recorder uses
    this to transcribe only opted-in speakers (privacy + CPU)."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT user_id FROM voice_users WHERE guild_id = ? AND opted_in = 1",
            (guild_id,),
        )
        return {r[0] for r in await cur.fetchall()}


# ──────────────────────────────────────────────────────────────────────────────
# App settings — host-wide key/value (not per-guild)
# ──────────────────────────────────────────────────────────────────────────────

async def get_app_setting(key: str, default: str | None = None) -> str | None:
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,),
        )
        row = await cur.fetchone()
    return row[0] if row else default


async def set_app_setting(key: str, value: str) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await db.commit()


async def list_app_settings(prefix: str) -> dict[str, str]:
    """Every app setting whose key starts with `prefix`."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT key, value FROM app_settings WHERE substr(key, 1, ?) = ?",
            (len(prefix), prefix),
        )
        rows = await cur.fetchall()
    return {k: v for k, v in rows}


# ──────────────────────────────────────────────────────────────────────────────
# Liked songs — personal, global (a user's likes are not tied to a guild)
# ──────────────────────────────────────────────────────────────────────────────

async def add_like(
    user_id: int, *, url: str, title: str,
    artist: str | None = None, duration: int = 0,
) -> bool:
    """Like a song for a user. Returns False if it was already liked."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            """INSERT OR IGNORE INTO liked_songs
                  (user_id, url, title, artist, duration, liked_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, url, title, artist, int(duration or 0), time.time()),
        )
        await db.commit()
        return cur.rowcount > 0


async def remove_like(user_id: int, url: str) -> bool:
    """Unlike a song. Returns True if a row was actually removed."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "DELETE FROM liked_songs WHERE user_id = ? AND url = ?",
            (user_id, url),
        )
        await db.commit()
        return cur.rowcount > 0


async def is_liked(user_id: int, url: str) -> bool:
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT 1 FROM liked_songs WHERE user_id = ? AND url = ?",
            (user_id, url),
        )
        return await cur.fetchone() is not None


async def count_likes(user_id: int) -> int:
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM liked_songs WHERE user_id = ?", (user_id,),
        )
        return (await cur.fetchone())[0]


async def list_likes(
    user_id: int, *, limit: int = 100, offset: int = 0,
) -> list[dict[str, Any]]:
    """A user's liked songs, newest first."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT url, title, artist, duration, liked_at
               FROM liked_songs WHERE user_id = ?
               ORDER BY liked_at DESC LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        )
        return [dict(r) for r in await cur.fetchall()]


# ──────────────────────────────────────────────────────────────────────────────
# Play stats — one row per fully-played track; powers "wrapped" + quiz pool
# ──────────────────────────────────────────────────────────────────────────────

async def record_play(
    *, guild_id: int, requester_id: int | None, url: str | None,
    title: str, artist: str | None, duration: int, is_music: bool,
) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO track_plays
                  (guild_id, requester_id, url, title, artist,
                   duration, is_music, played_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (guild_id, requester_id, url, title, artist,
             int(duration or 0), int(bool(is_music)), time.time()),
        )
        await db.commit()


def _since_clause(since: float | None) -> tuple[str, list[Any]]:
    """Optional `played_at >= since` fragment + params."""
    if since is None:
        return "", []
    return " AND played_at >= ?", [since]


async def top_tracks(
    guild_id: int, *, since: float | None = None, limit: int = 10,
) -> list[dict[str, Any]]:
    """Most-played tracks in a guild. [{url, title, artist, plays}, ...]."""
    extra, params = _since_clause(since)
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT url,
                       MAX(title)  AS title,
                       MAX(artist) AS artist,
                       COUNT(*)    AS plays
                FROM track_plays
                WHERE guild_id = ?{extra}
                GROUP BY COALESCE(url, title)
                ORDER BY plays DESC, title ASC
                LIMIT ?""",
            [guild_id, *params, limit],
        )
        return [dict(r) for r in await cur.fetchall()]


async def top_artists(
    guild_id: int, *, since: float | None = None, limit: int = 10,
) -> list[dict[str, Any]]:
    """Most-played artists. [{artist, plays}, ...]. Skips unknown-artist rows."""
    extra, params = _since_clause(since)
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT artist, COUNT(*) AS plays
                FROM track_plays
                WHERE guild_id = ? AND artist IS NOT NULL AND artist != ''{extra}
                GROUP BY artist
                ORDER BY plays DESC, artist ASC
                LIMIT ?""",
            [guild_id, *params, limit],
        )
        return [dict(r) for r in await cur.fetchall()]


async def top_listeners(
    guild_id: int, *, since: float | None = None, limit: int = 10,
) -> list[dict[str, Any]]:
    """Who queued the most fully-played tracks. [{user_id, plays, seconds}, ...]."""
    extra, params = _since_clause(since)
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT requester_id      AS user_id,
                       COUNT(*)          AS plays,
                       SUM(duration)     AS seconds
                FROM track_plays
                WHERE guild_id = ? AND requester_id IS NOT NULL{extra}
                GROUP BY requester_id
                ORDER BY plays DESC
                LIMIT ?""",
            [guild_id, *params, limit],
        )
        return [dict(r) for r in await cur.fetchall()]


async def user_top_tracks(
    guild_id: int, user_id: int, *, since: float | None = None, limit: int = 10,
) -> list[dict[str, Any]]:
    extra, params = _since_clause(since)
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT url, MAX(title) AS title, MAX(artist) AS artist,
                       COUNT(*) AS plays
                FROM track_plays
                WHERE guild_id = ? AND requester_id = ?{extra}
                GROUP BY COALESCE(url, title)
                ORDER BY plays DESC, title ASC
                LIMIT ?""",
            [guild_id, user_id, *params, limit],
        )
        return [dict(r) for r in await cur.fetchall()]


async def user_top_artists(
    guild_id: int, user_id: int, *, since: float | None = None, limit: int = 10,
) -> list[dict[str, Any]]:
    extra, params = _since_clause(since)
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT artist, COUNT(*) AS plays
                FROM track_plays
                WHERE guild_id = ? AND requester_id = ?
                  AND artist IS NOT NULL AND artist != ''{extra}
                GROUP BY artist
                ORDER BY plays DESC, artist ASC
                LIMIT ?""",
            [guild_id, user_id, *params, limit],
        )
        return [dict(r) for r in await cur.fetchall()]


async def user_play_totals(
    guild_id: int, user_id: int, *, since: float | None = None,
) -> dict[str, Any]:
    """{plays, seconds} of fully-played tracks this user queued."""
    extra, params = _since_clause(since)
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            f"""SELECT COUNT(*), COALESCE(SUM(duration), 0)
                FROM track_plays
                WHERE guild_id = ? AND requester_id = ?{extra}""",
            [guild_id, user_id, *params],
        )
        plays, seconds = await cur.fetchone()
    return {"plays": plays, "seconds": seconds}


# ──────────────────────────────────────────────────────────────────────────────
# Music quiz — eligible pool, manual excludes, scores
# ──────────────────────────────────────────────────────────────────────────────

async def quiz_pool(
    guild_id: int, *, min_dur: int = 60, max_dur: int = 600, limit: int = 500,
) -> list[dict[str, Any]]:
    """Distinct quiz-eligible tracks: flagged is_music, sane duration, and not
    manually excluded. [{url, title, artist, duration}, ...]."""
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT url,
                      MAX(title)    AS title,
                      MAX(artist)   AS artist,
                      MAX(duration) AS duration
               FROM track_plays
               WHERE guild_id = ? AND is_music = 1
                 AND duration BETWEEN ? AND ?
                 AND url IS NOT NULL
                 AND url NOT IN (
                     SELECT url FROM quiz_excludes WHERE guild_id = ?)
               GROUP BY url
               ORDER BY RANDOM()
               LIMIT ?""",
            (guild_id, min_dur, max_dur, guild_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]


async def quiz_exclude(guild_id: int, url: str) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO quiz_excludes (guild_id, url) VALUES (?, ?)",
            (guild_id, url),
        )
        await db.commit()


async def quiz_include(guild_id: int, url: str) -> bool:
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "DELETE FROM quiz_excludes WHERE guild_id = ? AND url = ?",
            (guild_id, url),
        )
        await db.commit()
        return cur.rowcount > 0


async def quiz_add_points(guild_id: int, user_id: int, points: int) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO quiz_scores (guild_id, user_id, points)
                VALUES (?, ?, ?)
                ON CONFLICT(guild_id, user_id)
                DO UPDATE SET points = points + excluded.points""",
            (guild_id, user_id, points),
        )
        await db.commit()


async def quiz_leaderboard(guild_id: int, *, limit: int = 10) -> list[dict[str, Any]]:
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT user_id, points FROM quiz_scores
               WHERE guild_id = ? AND points > 0
               ORDER BY points DESC LIMIT ?""",
            (guild_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]


# ──────────────────────────────────────────────────────────────────────────────
# Last.fm — per-user account links (global)
# ──────────────────────────────────────────────────────────────────────────────

async def lastfm_get(user_id: int) -> dict[str, Any] | None:
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT user_id, session_key, username, enabled, pending_tok "
            "FROM lastfm_users WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
    return dict(row) if row else None


async def lastfm_set_pending(user_id: int, token: str) -> None:
    """Stash the request token between `link` and `confirm`."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO lastfm_users (user_id, pending_tok)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET pending_tok = excluded.pending_tok""",
            (user_id, token),
        )
        await db.commit()


async def lastfm_complete(user_id: int, *, session_key: str, username: str) -> None:
    """Finish linking: store the session key, clear the pending token, enable."""
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO lastfm_users
                  (user_id, session_key, username, enabled, pending_tok)
                VALUES (?, ?, ?, 1, NULL)
                ON CONFLICT(user_id) DO UPDATE SET
                  session_key = excluded.session_key,
                  username    = excluded.username,
                  enabled     = 1,
                  pending_tok = NULL""",
            (user_id, session_key, username),
        )
        await db.commit()


async def lastfm_set_enabled(user_id: int, enabled: bool) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "UPDATE lastfm_users SET enabled = ? WHERE user_id = ?",
            (int(enabled), user_id),
        )
        await db.commit()


async def lastfm_unlink(user_id: int) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute("DELETE FROM lastfm_users WHERE user_id = ?", (user_id,))
        await db.commit()


async def lastfm_active_links(user_ids: list[int]) -> list[dict[str, Any]]:
    """Among `user_ids`, those with scrobbling enabled and a session key.
    [{user_id, session_key}, ...]. Empty list for empty input."""
    if not user_ids:
        return []
    placeholders = ",".join("?" * len(user_ids))
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"""SELECT user_id, session_key FROM lastfm_users
                WHERE enabled = 1 AND session_key IS NOT NULL
                  AND user_id IN ({placeholders})""",
            user_ids,
        )
        return [dict(r) for r in await cur.fetchall()]


# ──────────────────────────────────────────────────────────────────────────────
# Starboard — per-guild config + source→board message map
# ──────────────────────────────────────────────────────────────────────────────

async def get_starboard(guild_id: int) -> dict[str, Any]:
    """{channel_id: int|None, threshold: int, emoji: str}. Defaults when unset."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT channel_id, threshold, emoji FROM starboard_settings "
            "WHERE guild_id = ?",
            (guild_id,),
        )
        row = await cur.fetchone()
    if not row:
        return {"channel_id": None, "threshold": 1, "emoji": "⭐"}
    return {"channel_id": row[0], "threshold": row[1], "emoji": row[2]}


async def _set_starboard_field(guild_id: int, column: str, value: Any) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            f"""INSERT INTO starboard_settings (guild_id, {column})
                VALUES (?, ?)
                ON CONFLICT(guild_id) DO UPDATE SET {column} = excluded.{column}""",
            (guild_id, value),
        )
        await db.commit()


async def set_starboard_channel(guild_id: int, channel_id: int | None) -> None:
    await _set_starboard_field(guild_id, "channel_id", channel_id)


async def set_starboard_threshold(guild_id: int, threshold: int) -> None:
    await _set_starboard_field(guild_id, "threshold", threshold)


async def set_starboard_emoji(guild_id: int, emoji: str) -> None:
    await _set_starboard_field(guild_id, "emoji", emoji)


async def starboard_get_post(guild_id: int, source_msg_id: int) -> dict[str, Any] | None:
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT board_msg_id, star_count FROM starboard_posts "
            "WHERE guild_id = ? AND source_msg_id = ?",
            (guild_id, source_msg_id),
        )
        row = await cur.fetchone()
    return {"board_msg_id": row[0], "star_count": row[1]} if row else None


async def starboard_upsert_post(
    guild_id: int, source_msg_id: int, board_msg_id: int, star_count: int,
) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """INSERT INTO starboard_posts
                  (guild_id, source_msg_id, board_msg_id, star_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(guild_id, source_msg_id) DO UPDATE SET
                  board_msg_id = excluded.board_msg_id,
                  star_count   = excluded.star_count""",
            (guild_id, source_msg_id, board_msg_id, star_count),
        )
        await db.commit()


async def starboard_delete_post(guild_id: int, source_msg_id: int) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "DELETE FROM starboard_posts WHERE guild_id = ? AND source_msg_id = ?",
            (guild_id, source_msg_id),
        )
        await db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Voice wake words — stored as a JSON list in voice_settings.wake_word.
# Backward compatible: a bare string (old value) is read as a one-element list.
# ──────────────────────────────────────────────────────────────────────────────

def _decode_wakes(raw: str | None) -> list[str]:
    """Parse the wake_word column into a list. Tolerates the legacy single
    string and NULL."""
    if not raw:
        return []
    try:
        val = json.loads(raw)
    except (ValueError, TypeError):
        return [raw]                       # legacy: a plain phrase
    if isinstance(val, list):
        return [str(w) for w in val if str(w).strip()]
    return [str(val)] if str(val).strip() else []


async def get_voice_wakes(guild_id: int) -> list[str]:
    """All wake phrases configured for a guild ([] if none → caller defaults)."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT wake_word FROM voice_settings WHERE guild_id = ?", (guild_id,),
        )
        row = await cur.fetchone()
    return _decode_wakes(row[0] if row else None)


async def set_voice_wakes(guild_id: int, wakes: list[str]) -> None:
    """Replace the whole wake-word list (empty list → NULL = use default)."""
    cleaned = [" ".join(w.lower().split()) for w in wakes if w.strip()]
    value = json.dumps(cleaned) if cleaned else None
    await _set_voice_field(guild_id, "wake_word", value)


async def add_voice_wake(guild_id: int, phrase: str) -> list[str]:
    """Add one wake phrase (no-op if already present). Returns the new list."""
    wakes = await get_voice_wakes(guild_id)
    norm = " ".join(phrase.lower().split())
    if norm and norm not in wakes:
        wakes.append(norm)
        await set_voice_wakes(guild_id, wakes)
    return wakes


async def remove_voice_wake(guild_id: int, phrase: str) -> list[str]:
    """Remove one wake phrase if present. Returns the new list."""
    wakes = await get_voice_wakes(guild_id)
    norm = " ".join(phrase.lower().split())
    if norm in wakes:
        wakes.remove(norm)
        await set_voice_wakes(guild_id, wakes)
    return wakes


# ──────────────────────────────────────────────────────────────────────────────
# Lyric timing offsets
# ──────────────────────────────────────────────────────────────────────────────

# Keyed by upload URL rather than by song: the right correction depends on the
# particular file being streamed (intro cards, silence, re-encodes), so two
# uploads of the same song legitimately need different numbers.

async def get_lyric_offset(source_id: str) -> float | None:
    """The saved timing correction for this upload, or None if never set."""
    if not source_id:
        return None
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT offset_s FROM lyric_offsets WHERE source_id = ?",
            (source_id,),
        )
        row = await cur.fetchone()
    return float(row[0]) if row else None


async def save_lyric_offset(source_id: str, offset_s: float,
                            title: str | None = None) -> None:
    """Record a correction. Re-saving the same upload averages the new value
    into the old one rather than replacing it outright.

    Averaging matters because these come from humans eyeballing a scrolling
    message: any single nudge carries a few hundred milliseconds of reaction
    time, in a direction that varies. Repeated corrections converge on the
    truth instead of ping-ponging around it. The running mean is weighted by
    vote count, so an established value resists a single sloppy nudge.
    """
    if not source_id:
        return
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT offset_s, votes FROM lyric_offsets WHERE source_id = ?",
            (source_id,),
        )
        row = await cur.fetchone()
        if row:
            prev, votes = float(row[0]), int(row[1])
            # Cap the weight so a long-established wrong value can still be
            # corrected within a few nudges.
            w = min(votes, 5)
            merged = (prev * w + float(offset_s)) / (w + 1)
            await db.execute(
                "UPDATE lyric_offsets SET offset_s = ?, votes = ?, "
                "title = COALESCE(?, title), updated_at = ? WHERE source_id = ?",
                (merged, votes + 1, title, int(time.time()), source_id),
            )
        else:
            await db.execute(
                "INSERT INTO lyric_offsets (source_id, offset_s, votes, title, "
                "updated_at) VALUES (?, ?, 1, ?, ?)",
                (source_id, float(offset_s), title, int(time.time())),
            )
        await db.commit()


async def clear_lyric_offset(source_id: str) -> bool:
    """Forget the correction for this upload. True if one existed."""
    if not source_id:
        return False
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "DELETE FROM lyric_offsets WHERE source_id = ?", (source_id,))
        await db.commit()
        return cur.rowcount > 0


# ──────────────────────────────────────────────────────────────────────────────
# Separated-stem cache
# ──────────────────────────────────────────────────────────────────────────────

# The DB rows are an *index* over files on disk, and the two can disagree — a
# user can delete the cache directory, and a crash can leave a file with no
# row. Every read verifies the file still exists; the sweep handles the other
# direction. Treating the DB as authoritative on its own would mean handing
# ffmpeg a path to nothing.

async def get_stem(source_id: str, model_file: str) -> str | None:
    """Path to the cached instrumental, or None. Refreshes its LRU stamp."""
    if not source_id:
        return None
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT path FROM stem_cache WHERE source_id = ? AND model_file = ?",
            (source_id, model_file),
        )
        row = await cur.fetchone()
        if not row:
            return None
        path = row[0]
        if not (path and os.path.exists(path) and os.path.getsize(path) > 0):
            # The file went away underneath us; drop the stale row so the
            # caller re-separates instead of failing on a dead path.
            await db.execute(
                "DELETE FROM stem_cache WHERE source_id = ? AND model_file = ?",
                (source_id, model_file),
            )
            await db.commit()
            return None
        await db.execute(
            "UPDATE stem_cache SET last_used_at = ? "
            "WHERE source_id = ? AND model_file = ?",
            (int(time.time()), source_id, model_file),
        )
        await db.commit()
        return path


async def put_stem(source_id: str, model_file: str, path: str, *,
                   duration: float = 0.0, title: str | None = None) -> None:
    """Record a freshly separated instrumental."""
    if not source_id or not path:
        return
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    now = int(time.time())
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO stem_cache (source_id, model_file, path, bytes, "
            "duration, title, created_at, last_used_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id, model_file) DO UPDATE SET "
            "path=excluded.path, bytes=excluded.bytes, "
            "duration=excluded.duration, title=COALESCE(excluded.title, title), "
            "last_used_at=excluded.last_used_at",
            (source_id, model_file, path, size, duration, title, now, now),
        )
        await db.commit()


async def stem_cache_stats() -> dict:
    """Row count and total bytes, for the `karaoke cache` command."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(SUM(duration), 0) "
            "FROM stem_cache")
        row = await cur.fetchone()
    return {"count": row[0] or 0, "bytes": row[1] or 0, "seconds": row[2] or 0.0}


async def evict_stems(max_bytes: int) -> tuple[int, int]:
    """Delete least-recently-used stems until the cache fits. (files, bytes).

    Evicting by *last use* rather than age matters: a server with ten karaoke
    regulars should keep those ten instrumentals forever and throw away the
    one-off that somebody tried in March.
    """
    if max_bytes <= 0:
        return (0, 0)
    freed_files = freed_bytes = 0
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute("SELECT COALESCE(SUM(bytes), 0) FROM stem_cache")
        total = (await cur.fetchone())[0] or 0
        if total <= max_bytes:
            return (0, 0)
        cur = await db.execute(
            "SELECT source_id, model_file, path, bytes FROM stem_cache "
            "ORDER BY last_used_at ASC")
        for source_id, model_file, path, size in await cur.fetchall():
            if total <= max_bytes:
                break
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                log.debug("could not remove cached stem %s", path, exc_info=True)
            await db.execute(
                "DELETE FROM stem_cache WHERE source_id = ? AND model_file = ?",
                (source_id, model_file))
            total -= (size or 0)
            freed_files += 1
            freed_bytes += (size or 0)
        await db.commit()
    if freed_files:
        log.info("Evicted %d cached stem(s), freed %.1f MB",
                 freed_files, freed_bytes / 1e6)
    return (freed_files, freed_bytes)


async def clear_stem_cache() -> tuple[int, int]:
    """Delete every cached stem. Returns (files, bytes)."""
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute("SELECT path, bytes FROM stem_cache")
        rows = await cur.fetchall()
        n = total = 0
        for path, size in rows:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
            n += 1
            total += (size or 0)
        await db.execute("DELETE FROM stem_cache")
        await db.commit()
    return (n, total)
