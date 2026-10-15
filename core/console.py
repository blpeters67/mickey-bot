"""
Admin console — read commands from stdin while the bot runs.

You attach to the bot's screen (or run it in the foreground), type a
command, and get a printout to the console. No Discord interaction.

Usage from your shell:

    > guilds
    > guild 12345
    > voice
    > stats

Type `help` for the full list. Commands run in the bot's event loop, so
they see live state and never race with command handlers.

Design notes
============

* Reads stdin in a background thread (blocking `input()` would jam the
  event loop). Each line is fed back into asyncio via `run_coroutine_threadsafe`.
* No auth needed — if you can write to the bot's stdin, you're already
  the bot owner. Don't expose the screen socket publicly.
* When stdin is not a TTY (systemd unit, nohup, redirected stdin), the
  reader exits cleanly and the bot keeps running without it.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import io
import os
import platform
import sys
import threading
import time
from typing import Awaitable, Callable

import discord
from discord.ext import commands

from .logging_setup import get_logger
from .state import _states

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing helpers — these write to stdout, not the logger, because
# this output is for the human at the console, not for log files.
# ──────────────────────────────────────────────────────────────────────────────

# When a capture buffer is set (per async task), console output goes there
# instead of stdout — that's how the Discord `-console` owner bridge grabs a
# command's printout. contextvars keep it task-local, so capturing one
# invocation never steals another task's (or the real console's) output.
_capture: contextvars.ContextVar = contextvars.ContextVar("console_capture", default=None)


def _print(*args) -> None:
    """Print console output: to the active capture buffer for this task (the
    Discord bridge), else to the real stdout, flushed so it shows up live."""
    buf = _capture.get()
    if buf is not None:
        print(*args, file=buf)
    else:
        print(*args, flush=True)


def _table(rows: list[list[str]], *, headers: list[str] | None = None) -> None:
    """Print an aligned text table. Falls back gracefully on empty input."""
    if not rows and not headers:
        return
    all_rows = ([headers] if headers else []) + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(all_rows[0]))]
    for i, row in enumerate(all_rows):
        cells = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(row))
        _print(cells)
        if headers and i == 0:
            _print("  ".join("─" * w for w in widths))


def _fmt_uptime(seconds: float) -> str:
    s    = int(seconds)
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Command implementations — each receives (bot, args list).
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_help(bot: commands.Bot, args: list[str]) -> None:
    _print("Available console commands:")
    for name, (_fn, desc) in sorted(COMMANDS.items()):
        _print(f"  {name:14}  {desc}")
    # Handled synchronously in the reader thread (see _reader_loop), so it's
    # not in COMMANDS — list it by hand.
    _print(f"  {'threads':14}  Dump every thread's stack (works even if the bot is hung)")


async def cmd_guilds(bot: commands.Bot, args: list[str]) -> None:
    """List every server the bot is in."""
    guilds = sorted(bot.guilds, key=lambda g: g.member_count or 0, reverse=True)
    if not guilds:
        _print("Not in any guilds yet — still connecting?")
        return
    rows = []
    for g in guilds:
        owner = str(g.owner) if g.owner else "?"
        members = g.member_count or 0
        rows.append([str(g.id), g.name, owner, f"{members} members"])
    _table(rows, headers=["ID", "Name", "Owner", "Size"])
    _print(f"\nTotal: {len(guilds)} guild(s)")


async def cmd_guild(bot: commands.Bot, args: list[str]) -> None:
    """Show details for one server — channels, member count, owner, prefix."""
    if not args:
        _print("Usage: guild <id_or_name_substring>")
        return
    target = " ".join(args).lower()

    # Try ID first, then case-insensitive name substring match.
    g = None
    if target.isdigit():
        g = bot.get_guild(int(target))
    if g is None:
        matches = [x for x in bot.guilds if target in x.name.lower()]
        if len(matches) > 1:
            _print(f"Ambiguous — {len(matches)} guilds match {target!r}:")
            for m in matches:
                _print(f"  {m.id}  {m.name}")
            return
        if matches:
            g = matches[0]
    if g is None:
        _print(f"No guild matched {target!r}.")
        return

    from . import db
    try:
        prefix = await db.get_prefix(g.id)
    except Exception:
        prefix = "?"

    _print(f"Name:        {g.name}")
    _print(f"ID:          {g.id}")
    _print(f"Owner:       {g.owner} (id={g.owner_id})")
    _print(f"Members:     {g.member_count}")
    _print(f"Created:     {g.created_at:%Y-%m-%d}")
    _print(f"Prefix:      {prefix}")
    _print(f"Boost tier:  {g.premium_tier}  ({g.premium_subscription_count or 0} boosts)")
    if g.description:
        _print(f"Description: {g.description}")

    voice = [c for c in g.channels if isinstance(c, discord.VoiceChannel)]
    text  = [c for c in g.channels if isinstance(c, discord.TextChannel)]
    _print(f"\nChannels:    {len(text)} text, {len(voice)} voice")
    for c in voice[:15]:
        connected = [m.display_name for m in c.members if not m.bot]
        bot_in = " 🤖 (bot here)" if any(m.id == bot.user.id for m in c.members) else ""
        people = f"  [{', '.join(connected)}]" if connected else ""
        _print(f"  🔊 {c.name}{bot_in}{people}")


async def cmd_voice(bot: commands.Bot, args: list[str]) -> None:
    """Show every voice channel the bot is currently connected to."""
    if not bot.voice_clients:
        _print("Not connected to any voice channels.")
        return
    rows = []
    for vc in bot.voice_clients:
        playing = "▶ playing" if vc.is_playing() else ("⏸ paused" if vc.is_paused() else "💤 idle")
        state   = _states.get(vc.guild.id)
        current = state.current["title"] if (state and state.current) else "—"
        qlen    = len(state.queue) if state else 0
        rows.append([
            vc.guild.name, vc.channel.name, playing,
            current[:40] + ("…" if len(current) > 40 else ""),
            f"q={qlen}",
        ])
    _table(rows, headers=["Guild", "Channel", "State", "Now playing", "Queue"])


async def cmd_stats(bot: commands.Bot, args: list[str]) -> None:
    """Overall bot stats — uptime, guilds, voice connections, latency, memory."""
    started = getattr(bot, "_console_started_at", time.time())

    # Memory: read /proc/self/status — cross-distro on Linux, no extra dep.
    mem_mb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_mb = int(line.split()[1]) / 1024
                    break
    except OSError:
        pass

    total_queue = sum(len(s.queue) for s in _states.values())
    playing     = sum(1 for vc in bot.voice_clients if vc.is_playing())

    _print(f"User:           {bot.user} (id={bot.user.id})")
    _print(f"Uptime:         {_fmt_uptime(time.time() - started)}")
    _print(f"Guilds:         {len(bot.guilds)}")
    _print(f"Total members:  {sum(g.member_count or 0 for g in bot.guilds)}")
    _print(f"Voice clients:  {len(bot.voice_clients)} ({playing} playing)")
    _print(f"Queued tracks:  {total_queue} (across all guilds)")
    _print(f"Gateway ping:   {bot.latency * 1000:.0f} ms")
    if mem_mb is not None:
        _print(f"Memory (RSS):   {mem_mb:.1f} MB")
    _print(f"Python:         {platform.python_version()}")
    _print(f"discord.py:     {discord.__version__}")


async def cmd_invites(bot: commands.Bot, args: list[str]) -> None:
    """For each guild, try to fetch an invite — useful when you've forgotten
    which private servers the bot is in and want to actually join them."""
    _print("Fetching invites — this may take a moment…")
    for g in bot.guilds:
        link = None
        # Look for existing invites first (cheaper, doesn't pollute the server)
        try:
            invites = await g.invites()
            if invites:
                link = invites[0].url
        except discord.Forbidden:
            pass
        except Exception as e:
            log.warning("invites() failed for %s: %s", g.name, e)

        # Fall back to creating a new permanent invite in the first writable channel
        if not link:
            for c in g.text_channels:
                try:
                    inv = await c.create_invite(max_age=0, max_uses=0, unique=False,
                                                 reason="console: invites command")
                    link = inv.url
                    break
                except discord.Forbidden:
                    continue
                except Exception:
                    continue

        _print(f"  {g.name:30}  {link or '(no permission to create invite)'}")


async def cmd_leave(bot: commands.Bot, args: list[str]) -> None:
    """Make the bot leave a guild. Use with care — this kicks the bot."""
    if not args:
        _print("Usage: leave <guild_id>")
        return
    if not args[0].isdigit():
        _print("Guild ID must be numeric. Use `guilds` to see IDs.")
        return
    g = bot.get_guild(int(args[0]))
    if g is None:
        _print(f"Not in a guild with id {args[0]}.")
        return
    _print(f"Leaving {g.name!r} ({g.id})…")
    try:
        await g.leave()
        _print("Left.")
    except Exception as e:
        _print(f"Failed: {e}")


async def cmd_reload(bot: commands.Bot, args: list[str]) -> None:
    """Reload one or all cogs without restarting the bot."""
    extension_map = {
        "music":    "cogs.music",
        "admin":    "cogs.admin",
        "playlists":"cogs.playlists",
        "downloads":"cogs.downloads",
        "lyrics":   "cogs.lyrics",
        "emoji":    "cogs.emoji",
        "reminders":"cogs.reminders",
        "poker":    "cogs.games.poker",
        "blackjack":"cogs.games.blackjack",
        "yahtzee":  "cogs.games.yahtzee",
    }

    # No arg or "all" → reload everything in dependency order. Admin first
    # so the help/error handlers are fresh; music before playlists/downloads
    # since the latter call music._play_next.
    if not args or args[0].lower() == "all":
        order = ["admin", "music", "playlists", "downloads", "lyrics",
                 "emoji", "reminders", "poker", "blackjack", "yahtzee"]
        ok = fail = 0
        for name in order:
            ext = extension_map[name]
            try:
                await bot.reload_extension(ext)
                _print(f"  ✓ {ext}")
                ok += 1
            except Exception as e:
                _print(f"  ✗ {ext}: {e}")
                log.exception("Reload of %s failed", ext)
                fail += 1
        _print(f"\n{ok} reloaded, {fail} failed.")
        return

    # Single-cog reload.
    target = args[0].lower()
    if target not in extension_map and target not in extension_map.values():
        _print(f"Unknown cog: {target!r}")
        _print("Available: " + ", ".join(sorted(extension_map)) + ", all")
        return
    ext = extension_map.get(target, target)
    try:
        await bot.reload_extension(ext)
        _print(f"Reloaded {ext}.")
    except Exception as e:
        _print(f"Reload failed: {e}")
        log.exception("Reload of %s failed", ext)


async def cmd_quit(bot: commands.Bot, args: list[str]) -> None:
    """Shut the bot down cleanly.

    Calls `bot.close()` which signals the gateway to disconnect. The main
    `bot.start()` call in bot.py returns once that's done, which lets
    `asyncio.run(_main())` exit and the process terminate.
    """
    _print("Shutting down… (gateway disconnect can take a few seconds)")
    await bot.close()


async def cmd_restart(bot: commands.Bot, args: list[str]) -> None:
    """Restart the bot in place.

    Closes the Discord connection cleanly, then re-execs the Python
    interpreter with the same argv. `os.execv` replaces the current
    process — same PID, same stdin/stdout (so this works inside `screen`
    and you keep your console), but every module is reloaded from disk.

    Use this when you've changed something in `core/` that a `reload all`
    can't pick up.
    """
    _print("")
    _print("┌─ Restarting ──────────────────────────────────────────────")
    _print("│ This may take 30–90 seconds. Here's what happens:")
    _print("│   1. Gateway disconnect (a few seconds)")
    _print("│   2. New Python process starts, reloads every module")
    _print("│   3. Cogs load, slash commands re-sync with Discord")
    _print("│   4. Bot reconnects to voice channels it was in (if any)")
    _print("│")
    _print("│ If you see arrow keys printing as ^[[A, type 'stty sane'")
    _print("│ and press Enter — the terminal will snap back to normal.")
    _print("│")
    _print("│ If the bot doesn't return within 2 minutes:")
    _print("│   • Discord may be rate-limiting reconnects — just wait.")
    _print("│   • Or Ctrl+C and run `python bot.py` manually.")
    _print("└───────────────────────────────────────────────────────────")
    _print("")

    # Reset the terminal to a clean cooked-mode state before exec. Without
    # this, the new process's input() can see raw key bytes (^[[A for arrow
    # keys, etc.) because tty settings persisted from this process. `stty
    # sane` is the universal "make the tty behave normally" command.
    if sys.stdin.isatty():
        try:
            import subprocess
            subprocess.run(["stty", "sane"], check=False)
        except Exception:
            pass  # Not critical — worst case is a slightly weird prompt.

    # Set the flag BEFORE close. `await bot.close()` can interleave with the
    # loop's own teardown such that control never returns from the await,
    # so any statement after it might not run.
    bot._restart_requested = True
    await bot.close()


# Each command: name -> (async fn, short description)
COMMANDS: dict[str, tuple[Callable[..., Awaitable[None]], str]] = {
    "help":    (cmd_help,    "Show this list"),
    "guilds":  (cmd_guilds,  "List every server the bot is in"),
    "guild":   (cmd_guild,   "Detailed info for one guild (by id or name)"),
    "voice":   (cmd_voice,   "List active voice connections + what's playing"),
    "stats":   (cmd_stats,   "Uptime, memory, latency, totals"),
    "invites": (cmd_invites, "Get an invite URL for each guild"),
    "leave":   (cmd_leave,   "Leave a guild by id"),
    "reload":  (cmd_reload,  "Reload one cog, or `reload all` for everything"),
    "restart": (cmd_restart, "Restart the bot process (re-reads every file from disk)"),
    "quit":    (cmd_quit,    "Shut the bot down cleanly"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Thread dump — the diagnostic for a hung bot. Built and printed entirely in
# the caller's (reader) thread, touching nothing on the event loop, so it works
# precisely when the loop is deadlocked.
# ──────────────────────────────────────────────────────────────────────────────

def _format_thread_dump() -> str:
    import traceback
    frames = sys._current_frames()
    names  = {t.ident: t.name for t in threading.enumerate()}
    lines  = [f"===== Thread dump: {len(frames)} threads ====="]
    for tid, frame in frames.items():
        lines.append(f"\n--- {names.get(tid, '?')} (tid={tid}) ---")
        for entry in traceback.format_stack(frame):
            lines.append(entry.rstrip())
    return "\n".join(lines)


def _format_task_dump(loop: asyncio.AbstractEventLoop | None) -> str:
    """Pending asyncio tasks — the blind spot of a thread dump. A coroutine
    suspended on an `await` that never completes (a stuck command) shows up
    here, not in the OS threads. Read-only and best-effort: enumerating the
    loop's tasks from this thread can race, so we just report and move on."""
    if loop is None:
        return "\n(no event loop reference — skipping async tasks)"
    import io
    try:
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
    except Exception as e:
        return f"\n(could not enumerate asyncio tasks: {e!r})"
    if not tasks:
        return "\n===== Async tasks: none pending ====="
    out = [f"\n===== Async tasks: {len(tasks)} pending ====="]
    for t in tasks:
        sio = io.StringIO()
        try:
            t.print_stack(file=sio)
        except Exception as e:
            sio.write(f"(stack unavailable: {e!r})")
        out.append(f"\n--- TASK {t.get_name()} ---\n{sio.getvalue().rstrip()}")
    return "\n".join(out)


def _dump_all_threads(loop: asyncio.AbstractEventLoop | None = None) -> None:
    dump = _format_thread_dump() + "\n" + _format_task_dump(loop)
    _print(dump)
    try:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "musicbot-threads.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(dump + "\n")
        _print(f"\n(also saved to {path} — paste either this output or that file)")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# stdin reader — runs in a daemon thread, dispatches into the asyncio loop.
# ──────────────────────────────────────────────────────────────────────────────

def _reader_thread(bot: commands.Bot, loop: asyncio.AbstractEventLoop) -> None:
    if not sys.stdin or not sys.stdin.isatty():
        log.info("stdin is not a TTY — console disabled. "
                 "Run `python bot.py` interactively to get the admin console.")
        return

    _print("\n  Admin console ready. Type 'help' for commands, 'quit' to shut down.\n")

    reason = _reader_loop(bot, loop)
    # The console dying must never be silent — when the prompt seems dead
    # after re-attaching to screen, this log line is the evidence of what
    # actually happened (vs. a frozen terminal, which logs nothing).
    log.warning(
        "Console reader exited: %s. The bot itself keeps running — "
        "Ctrl+C here (or `kill %d` from another shell) to stop it.",
        reason, os.getpid(),
    )


def _reader_loop(bot: commands.Bot, loop: asyncio.AbstractEventLoop) -> str:
    """The prompt loop. Returns a human-readable reason when it exits —
    every exit path is deliberate and reported, never a bare `return`."""
    while True:
        try:
            line = input("> ")
        except EOFError:
            return "stdin closed (EOF)"
        except KeyboardInterrupt:
            return "KeyboardInterrupt"
        except Exception as e:
            return f"input() failed: {e!r}"

        try:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            name, args = parts[0].lower(), parts[1:]

            # `threads` runs RIGHT HERE in the reader thread, not via the event
            # loop — so it still works when the loop is deadlocked, which is the
            # only time you need it.
            if name == "threads":
                _dump_all_threads(loop)
                continue

            entry = COMMANDS.get(name)
            if entry is None:
                _print(f"Unknown command: {name!r}. Type 'help' for the list.")
                continue

            fn, _desc = entry

            # `quit` and `restart` are special: schedule and return immediately.
            # Waiting on the future would race with the bot's own loop teardown.
            if name in ("quit", "restart"):
                asyncio.run_coroutine_threadsafe(fn(bot, args), loop)
                return f"'{name}' issued"

            fut = asyncio.run_coroutine_threadsafe(fn(bot, args), loop)
            try:
                fut.result(timeout=60)
            except concurrent.futures.TimeoutError:
                # TimeoutError stringifies to "" — the old generic handler
                # printed a baffling "Command failed: " with nothing after it.
                _print("…taking longer than 60s. It keeps running in the "
                       "background; output will appear when it finishes.")
            except concurrent.futures.CancelledError:
                return "command cancelled — event loop shutting down"
            except RuntimeError as e:
                if "loop" in str(e).lower():
                    return f"event loop unavailable ({e})"
                _print(f"Command failed: {e}")
                log.exception("Console command %r failed", name)
            except Exception as e:
                _print(f"Command failed: {e}")
                log.exception("Console command %r failed", name)
        except Exception:
            # Belt and braces: an unexpected error in the dispatch machinery
            # itself must not kill the console. Log it and keep prompting.
            log.exception("Console dispatch error for line %r — still listening", line)
            try:
                _print("Console hiccup — details in the log. Still listening.")
            except Exception:
                return "stdout broken"


async def run_console_command(bot, name: str, args: list[str]) -> str:
    """Run one console command and return its captured output as text — for the
    Discord owner bridge (`-console`). Output the command would `_print` to
    stdout is captured task-locally instead. `restart`/`quit` are NOT handled
    here (the caller fires those directly — they tear down the process rather
    than producing output)."""
    buf = io.StringIO()
    token = _capture.set(buf)
    try:
        if name == "threads":
            buf.write(_format_thread_dump() + "\n" + _format_task_dump(bot.loop))
        else:
            entry = COMMANDS.get(name)
            if entry is None:
                buf.write(f"Unknown command: {name!r}. Type 'help' for the list.")
            else:
                await entry[0](bot, args)
    finally:
        _capture.reset(token)
    return buf.getvalue()


def start_console(bot: commands.Bot) -> None:
    """Spawn the stdin reader thread. Call once after the bot is built."""
    bot._console_started_at = time.time()       # used by `stats`
    loop = asyncio.get_event_loop()
    t = threading.Thread(
        target=_reader_thread, args=(bot, loop),
        name="console-reader", daemon=True,
    )
    t.start()
