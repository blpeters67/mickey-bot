"""
Bot entry point. Loads cogs, syncs slash commands, runs forever.

Run with:  python bot.py
"""
from __future__ import annotations

import asyncio
import os
import sys

import discord
from discord.ext import commands

from core import audio, db
from core.config import settings
from core.console import start_console
from core.logging_setup import get_logger, setup_logging

setup_logging()
log = get_logger("bot")


COG_MODULES = [
    "cogs.admin",
    "cogs.maintenance",
    "cogs.music",
    "cogs.playlists",
    "cogs.favorites",
    "cogs.musicstats",
    "cogs.downloads",
    "cogs.images",
    "cogs.recording",
    "cogs.serverstats",
    "cogs.starboard",
    "cogs.voicedj",
    "cogs.tts",
    "cogs.lastfm",
    "cogs.lyrics",
    "cogs.karaoke",
    "cogs.emoji",
    "cogs.reminders",
    "cogs.games.poker",
    "cogs.games.blackjack",
    "cogs.games.yahtzee",
    "cogs.games.musicquiz",
]


async def _prefix(bot: commands.Bot, message: discord.Message):
    """Per-guild prefix from the DB, falls back to settings.default_prefix."""
    if message.guild is None:
        return settings.default_prefix
    try:
        return await db.get_prefix(message.guild.id)
    except Exception:
        log.exception("get_prefix failed for guild %s", message.guild.id)
        return settings.default_prefix


def _build_bot() -> commands.Bot:
    intents                 = discord.Intents.default()
    intents.message_content = True
    intents.voice_states    = True
    intents.members         = True
    bot = commands.Bot(
        command_prefix=_prefix,
        intents=intents,
        help_command=None,
    )

    @bot.event
    async def setup_hook():
        failed: list[str] = []
        for cog in COG_MODULES:
            try:
                await bot.load_extension(cog)
                log.info("Loaded %s", cog)
            except Exception:
                log.exception("Failed to load %s", cog)
                failed.append(cog)
        if failed:
            # A failed cog load is otherwise easy to miss: the bot starts fine
            # and just silently lacks those commands. Surface a scannable summary
            # after the per-cog tracebacks so a missing command is obvious.
            log.error("%d cog(s) FAILED to load: %s — their commands will be "
                      "missing. See the tracebacks above.",
                      len(failed), ", ".join(failed))
        # Stash for the restart notice (on_ready reads it). Same process as
        # this hook, so plain attributes are fine — no DB round-trip needed.
        bot._failed_cogs = failed
        bot._slash_sync_failed = False
        try:
            synced = await bot.tree.sync()
            log.info("Synced %d slash command(s)", len(synced))
        except Exception:
            log.exception("Slash sync failed")
            bot._slash_sync_failed = True

    @bot.event
    async def on_ready():
        log.info("Logged in as %s (id=%s) — serving %d guild(s)",
                 bot.user, bot.user.id, len(bot.guilds))
        try:
            await bot.change_presence(activity=discord.Game(name="music · type -help"))
        except Exception:
            pass
        # Console is started on first on_ready only — reconnects shouldn't
        # spawn additional reader threads.
        if not getattr(bot, "_console_running", False):
            bot._console_running = True
            start_console(bot)
        # If we just came back from a Discord-triggered restart, announce it
        # in the channel it was issued from.
        await _announce_restart(bot)

    return bot


def _fmt_elapsed(seconds: float) -> str:
    """Compact restart duration: '38s' or '1m 38s'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


async def _announce_restart(bot: commands.Bot) -> None:
    """Tell the channel a restart was issued from that the bot is back.

    The restart marker is written to the DB by the `-console restart` handler
    just before the process re-execs (memory doesn't survive `os.execv`, the
    DB does). We clear it *before* sending so this only fires once even though
    `on_ready` also runs on every plain gateway reconnect.
    """
    import json
    import time

    raw = await db.get_app_setting("restart_notify")
    if not raw:
        return
    await db.set_app_setting("restart_notify", "")  # consume it (fire once)
    try:
        data = json.loads(raw)
    except Exception:
        return
    # Ignore stale markers — e.g. a crash that only came back hours later.
    elapsed = time.time() - data.get("at", 0)
    if elapsed > 600:
        return

    channel = bot.get_channel(data["channel_id"])
    if channel is None:
        try:
            channel = await bot.fetch_channel(data["channel_id"])
        except Exception:
            log.warning("Restart notice: channel %s is gone", data["channel_id"])
            return

    # Headline: back online + how long the whole restart took.
    lines = [f"✅ Back online in {_fmt_elapsed(elapsed)}."]
    # Surface anything that went wrong during startup so a broken restart
    # isn't silently reported as a clean one.
    failed = getattr(bot, "_failed_cogs", [])
    if failed:
        pretty = ", ".join(c.removeprefix("cogs.") for c in failed)
        lines.append(f"⚠️ {len(failed)} cog(s) failed to load: {pretty} — "
                     "their commands are missing. Check the logs.")
    if getattr(bot, "_slash_sync_failed", False):
        lines.append("⚠️ Slash-command sync failed — slash commands may be stale.")
    text = "\n".join(lines)
    msg_id = data.get("message_id")
    try:
        if msg_id:
            # Turn the original "Restarting…" message into the all-clear.
            original = await channel.fetch_message(msg_id)
            await original.edit(content=text)
        else:
            await channel.send(text)
    except discord.HTTPException:
        # Original message was deleted, or the edit failed — just post fresh.
        try:
            await channel.send(text)
        except Exception:
            log.exception("Restart notice: could not send back-online message")


_thread_dump_file = None   # kept alive so faulthandler's fd stays open


def _enable_thread_dumps() -> None:
    """Dump every thread's stack to a file on SIGUSR1, so a wedged process can
    be inspected even when stderr is blocked. Trigger with `kill -USR1 <pid>`;
    the dump lands in the temp dir. Linux/macOS only — a no-op on Windows.

    This is the tool for the transcribe-while-talking hang: reproduce the
    freeze, send the signal, and the file shows exactly where each thread is
    stuck (which reveals the shared lock that's deadlocking)."""
    global _thread_dump_file
    import faulthandler
    import signal
    import tempfile

    faulthandler.enable()                       # also dumps on a hard crash
    if not hasattr(signal, "SIGUSR1"):
        return
    path = os.path.join(tempfile.gettempdir(), "musicbot-threads.txt")
    try:
        _thread_dump_file = open(path, "w")
        faulthandler.register(
            signal.SIGUSR1, file=_thread_dump_file,
            all_threads=True, chain=True,
        )
        log.info("Thread-dump armed: kill -USR1 %d → %s", os.getpid(), path)
    except Exception:
        log.debug("Could not arm SIGUSR1 thread dump", exc_info=True)


async def _startup_preflight() -> None:
    """Log anything this interpreter is missing for voice (quick — no model
    load). `python bot.py --preflight` runs the full check."""
    from core import preflight
    try:
        checks = await asyncio.to_thread(preflight.run)
    except Exception:
        log.exception("Startup preflight failed to run")
        return
    for c in checks:
        if c.name in ("interpreter", "platform"):
            log.info("Preflight: %s — %s", c.name, c.detail)
    for c in preflight.problems(checks):
        (log.error if c.status == preflight.FAIL else log.warning)(
            "Preflight: %s — %s", c.name, c.detail)


async def _main() -> commands.Bot | None:
    _enable_thread_dumps()
    await db.init_db()
    audio.init_spotify()
    _load_opus()
    await _startup_preflight()
    # NB: we deliberately do NOT disable DAVE. Discord enforces DAVE E2EE on all
    # non-stage voice calls (since 2026-03-02) and rejects opt-outs with voice
    # close code 4017 — disabling it breaks ALL voice. The -clip recorder
    # decrypts received audio instead (cogs/recording.py _maybe_decrypt).
    bot = _build_bot()
    try:
        await bot.start(settings.discord_token)
    except discord.LoginFailure:
        log.error("Login failed — your Discord token is wrong or revoked. "
                  "Generate a new one at discord.com/developers/applications.")
        sys.exit(1)
    return bot


def _load_opus() -> None:
    """Explicitly load libopus so voice works.

    discord.py won't auto-load opus on Linux — you get the unhelpful error
    "Opus library needed in order to use voice" the first time you try to
    play a track. We try a few common names because distros disagree on
    what the unversioned symlink is called (often only `libopus.so.0`
    exists; the dev package is what adds `libopus.so`).
    """
    if discord.opus.is_loaded():
        return
    import ctypes.util
    candidates = []
    found = ctypes.util.find_library("opus")
    if found:
        candidates.append(found)
    candidates += ["libopus.so.0", "libopus.so", "opus"]
    for name in candidates:
        try:
            discord.opus.load_opus(name)
            log.info("Loaded opus from %s", name)
            return
        except Exception:
            continue
    log.warning(
        "Could not load libopus. Voice commands will fail. "
        "Try: sudo apt install libopus0 libopus-dev"
    )


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        # Full host check (including loading the voice model and timing an
        # inference), printed; exit status 1 if anything required is missing.
        from core import preflight
        checks = preflight.run(load_model_name=settings.voice_whisper_model)
        print(preflight.format_report(checks))
        sys.exit(1 if any(c.status == preflight.FAIL for c in checks) else 0)
    try:
        bot = asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("Shutting down.")
        sys.exit(0)

    # If the console's `restart` command ran, asyncio.run() returns
    # normally and the bot object carries the flag. Re-exec to swap in a
    # fresh interpreter that re-reads every module from disk.
    if bot is not None and getattr(bot, "_restart_requested", False):
        # Resolve everything to absolute paths so the exec doesn't depend on
        # the current working directory or PATH lookup. argv[0] in particular
        # is often a relative `bot.py` which would break after cwd changes.
        py     = os.path.abspath(sys.executable)
        script = os.path.abspath(sys.argv[0])
        rest   = sys.argv[1:]
        log.info("Restarting: %s %s %s", py, script, " ".join(rest))
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            os.execv(py, [py, script] + rest)
        except Exception as e:
            # If we got here, execv itself failed (bad path, permissions, etc).
            # The original process is still alive — print and exit so the user
            # at least knows what went wrong instead of silently disappearing.
            log.error("Restart failed: %s", e)
            sys.exit(1)
