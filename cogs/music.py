"""
Music cog: play / queue / seek / skip / pause / volume / etc.

Notable fixes vs the original bot.py:
  * Stream URLs resolved at play-time (lazy), not at queue-time.
  * Per-guild prefix (the prefix command now actually persists, per guild).
  * h:m:s seek parsing supports tracks longer than 60 minutes.
  * Permission gate: most commands require the user be in the bot's voice channel.
  * Idle disconnect: handled in cogs/admin.py via on_voice_state_update.
  * Logging instead of print(); errors carry tracebacks.
  * No mutable globals; per-guild state lives in core.state.
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import discord
from discord.ext import commands
try:
    from discord.ext import voice_recv
except ImportError:  # optional extension; without it -clip recording is unavailable
    voice_recv = None

from core import audio, db, queue_ops
from core.ffmpeg_source import FailureInfo, classify_failure
from core.formatters import (
    fmt_duration, make_progress_bar, parse_indices, parse_seek,
    redact_secrets, unwrap_url,
)
from core.logging_setup import get_logger
from core.playback_clock import FRAME_SECONDS, PlaybackClock, rate_from_filter
from core.state import clear_queue, get_state, playback_position, requeue_copy, reset_state
from core.voice_mix import DuckingSource

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# How a track's playback ended
#
# A track moves queued → starting (handed to FFmpeg/the player) → playing (the
# first frame of actual music went out) → finished | failed. "Queued" and a
# successful vc.play() are NOT evidence of sound: FFmpeg can die on its first
# read (a 403, an expired link, a dead connection), and that used to look
# exactly like a song ending — the track was consumed, counted as played, and
# nobody was told.
# ──────────────────────────────────────────────────────────────────────────────

#: Statuses of a `state.current` whose source is (or is about to be) live.
#: While the current track is in one of these, its own end handling decides
#: what plays next — a concurrent kickstart must not advance past it.
_LIVE = frozenset({"starting", "playing", "retrying"})

#: An error-free end this far short of the stream's length is premature.
_EARLY_END_MIN_SLACK = 10.0
_EARLY_END_FRACTION = 0.05
#: One refresh-and-retry per failure; the allowance comes back only after this
#: much music played (two drops an hour apart in a long mix are both retried;
#: a stream that keeps dying can't loop).
_RETRY_HEALTHY_SECONDS = 60.0
#: Resume a premature failure this far before where the audio stopped.
_RESUME_BACKOFF = 2.0
#: `queue` mentions a failed track for this long.
_FAILURE_SHOWN_FOR = 600


@dataclass
class PlaybackOutcome:
    """What one FFmpeg run actually did, captured on the audio thread the
    moment the player stops (before discord.py cleans the source up)."""
    error: BaseException | None
    music_frames: int | None          # None: a source that can't tell us
    start_position: float
    rate: float
    returncode: int | None
    stderr: str

    @classmethod
    def capture(cls, source, err, *, position: float, rate: float) -> "PlaybackOutcome":
        # discord.py ≥2.6 passes the source's error to `after`; older versions
        # don't, so read it ourselves (DuckingSource forwards FFmpeg's).
        if err is None:
            err = getattr(source, "_current_error", None)
        inner = getattr(source, "original", None)
        stderr = ""
        get = getattr(inner, "stderr_text", None)
        if callable(get):
            try:
                stderr = get()
            except Exception:
                stderr = ""
        frames = getattr(source, "music_frames", None)
        return cls(err, frames if isinstance(frames, int) else None, position, rate,
                   getattr(inner, "returncode", None), stderr)

    @property
    def music_seconds(self) -> float:
        return (self.music_frames or 0) * FRAME_SECONDS

    @property
    def reached(self) -> float:
        """Position in the track where the audio stopped."""
        return self.start_position + self.music_seconds * self.rate


@dataclass(frozen=True)
class Verdict:
    kind: str                        # completed | startup_failure | premature
    info: FailureInfo | None = None
    resume_at: float = 0.0


def judge_playback(track: dict, outcome: PlaybackOutcome) -> Verdict:
    """Completed, failed before any music, or cut short — from the evidence,
    not from the fact that the player stopped."""
    frames = outcome.music_frames
    if outcome.error is not None:
        info = classify_failure(outcome.error, outcome.stderr)
        if not frames:
            return Verdict("startup_failure", info, outcome.start_position)
        return Verdict("premature", info, max(0.0, outcome.reached - _RESUME_BACKOFF))
    if frames is None:
        return Verdict("completed")          # no evidence either way
    if frames == 0:
        return Verdict("startup_failure", classify_failure(None, outcome.stderr),
                       outcome.start_position)
    expected = float(track.get("_stream_duration") or track.get("duration") or 0)
    if expected > 0:
        slack = max(_EARLY_END_MIN_SLACK, _EARLY_END_FRACTION * expected)
        if outcome.reached < expected - slack:
            info = classify_failure(None, outcome.stderr)
            if info.category == "no_audio_produced":
                info = FailureInfo("ended_early", True,
                                   "the stream ended early without an error",
                                   "Try playing it again.")
            return Verdict("premature", info, max(0.0, outcome.reached - _RESUME_BACKOFF))
    return Verdict("completed")


def _startup_problem(exc: BaseException, stage: str) -> tuple[str, str]:
    """A failure to even start playback → ("systemic", msg) when nothing
    could play right now (missing FFmpeg/Opus, no voice connection), else
    ("track", reason) for a problem with this one track."""
    windows = sys.platform.startswith("win")
    if isinstance(exc, discord.opus.OpusNotLoaded):
        fix = ("discord.py's bundled Opus library didn't load — reinstall discord.py "
               "in the bot's environment and restart me" if windows else
               "install libopus (`sudo apt install libopus0`) and restart me")
        return "systemic", f"❌ I can't produce sound: the Opus audio library isn't available. Owner: {fix}."
    text = str(exc)
    if stage == "source":
        if isinstance(exc, discord.ClientException) and "was not found" in text:
            fix = ("install FFmpeg (e.g. `winget install Gyan.FFmpeg`) so `ffmpeg.exe` "
                   "is on PATH, then restart me" if windows else
                   "install it (`sudo apt install ffmpeg`) and restart me")
            return "systemic", f"❌ I can't play anything: FFmpeg isn't installed or isn't on PATH. Owner: {fix}."
        if isinstance(exc, (discord.ClientException, OSError)):
            return "systemic", (f"❌ I couldn't start FFmpeg ({type(exc).__name__}). "
                                "Try again in a moment; the details are in my log.")
        return "track", f"I couldn't open its audio ({redact_secrets(text)[:150] or type(exc).__name__})"
    if isinstance(exc, discord.ClientException):
        return "systemic", ("⚠️ Couldn't start playback — I seem to have lost the voice "
                            "connection. Run `rejoin` and I'll carry on.")
    return "track", f"the player rejected it ({type(exc).__name__})"


def _log_future(what: str, fut) -> None:
    """Done-callback for background playback work: an exception must reach
    the log, not die unobserved in a Future nobody awaits."""
    try:
        if fut.cancelled():
            return
        exc = fut.exception()
    except Exception:                     # pragma: no cover — defensive
        return
    if exc is not None:
        log.error("Background playback task failed (%s)", what, exc_info=exc)


# ──────────────────────────────────────────────────────────────────────────────
# Permission helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _ensure_voice(ctx: commands.Context) -> bool:
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("❌ You must be in a voice channel first.")
        return False
    return True


async def _ensure_same_channel(ctx: commands.Context) -> bool:
    vc = ctx.voice_client
    if vc is None:
        return True
    if not ctx.author.voice or ctx.author.voice.channel != vc.channel:
        await ctx.send(f"❌ You need to be in **{vc.channel.name}** to control playback.")
        return False
    return True


async def _start_recording(ctx: commands.Context, vc) -> None:
    """Hand the (just-connected) voice client to the Recording cog so the
    rolling buffer for `-clip` starts. Idempotent and best-effort — playback
    must never fail just because recording couldn't start."""
    rec = ctx.bot.get_cog("Recording")
    if rec is not None:
        try:
            await rec.attach(vc, ctx.channel)
        except Exception:
            log.exception("Failed to attach voice recorder")


async def _connect_to_author(ctx: commands.Context):
    """Connect (or move) to the author's channel. Returns None and tells the
    user off if the bot is mid-playback in a different channel — otherwise
    anyone in the server could yank the bot away from active listeners."""
    # An autojoin may be mid-handshake for this server: let it finish rather
    # than treating its half-built client as a zombie and tearing it down.
    music = ctx.bot.get_cog("Music")
    joining = getattr(music, "_joining", {}).get(ctx.guild.id) if music else None
    if joining is not None:
        try:
            await asyncio.wait_for(joining.wait(), 15)
        except asyncio.TimeoutError:
            pass
    vc = ctx.voice_client
    # Zombie client: the object exists but the underlying session is dead
    # (failed handshake, unre-covered gateway drop). Reusing it makes every
    # vc.play raise "Not connected" — which used to be swallowed, so every
    # play from then on silently did nothing until a manual `leave`. Scrap
    # it and connect fresh instead.
    if vc is not None and not vc.is_connected():
        log.warning("Zombie voice client in guild %s — rebuilding the connection.",
                    ctx.guild.id)
        try:
            await vc.disconnect(force=True)
        except Exception:
            log.debug("zombie disconnect grumbled", exc_info=True)
        vc = None
    if vc is None:
        cls = voice_recv.VoiceRecvClient if voice_recv else discord.VoiceClient
        try:
            vc = await ctx.author.voice.channel.connect(cls=cls)
        except (asyncio.TimeoutError, discord.ClientException) as e:
            live = ctx.guild.voice_client
            if (isinstance(e, discord.ClientException) and live is not None
                    and live.is_connected() and live.channel == ctx.author.voice.channel):
                vc = live          # someone else (autojoin) connected us meanwhile
            else:
                # Handshake timed out or a half-open session is in the way.
                # Surface it — the old behaviour was a generic traceback.
                log.warning("Voice connect failed in guild %s: %s", ctx.guild.id, e)
                await ctx.send("❌ Couldn't establish the voice connection — Discord "
                               "may be having a moment. Try again in a few seconds.")
                return None
        await _start_recording(ctx, vc)
        return vc
    if vc.channel != ctx.author.voice.channel:
        if vc.is_playing() or vc.is_paused():
            await ctx.send(
                f"❌ I'm already playing in **{vc.channel.name}** — "
                "join that channel, or stop playback there first."
            )
            return None
        await vc.move_to(ctx.author.voice.channel)
    await _start_recording(ctx, vc)
    return vc


#: app_settings key for the channel music was last controlled from, per guild.
#: Kept across restarts: after one, the bot can be in voice (autojoin) before
#: anyone types a command, and its messages must still go where people use it.
_MUSIC_CHANNEL_KEY = "music_channel:"
_bg_tasks: set = set()


def _remember_text_channel(ctx: commands.Context) -> None:
    """Stash the channel a music command came from so the playback engine can
    report skips/stalls there. Without this, play-time failures had no route
    back to users ("added ✅" … then nothing, with only a log line)."""
    if ctx.guild is None:
        return
    state = get_state(ctx.guild.id)
    if state.text_channel_id == ctx.channel.id:
        return
    state.text_channel_id = ctx.channel.id

    async def _save(gid: int, cid: int) -> None:
        try:
            await db.set_app_setting(f"{_MUSIC_CHANNEL_KEY}{gid}", str(cid))
        except Exception:
            log.debug("couldn't save the music channel", exc_info=True)
    try:
        task = asyncio.get_running_loop().create_task(_save(ctx.guild.id, ctx.channel.id))
    except RuntimeError:
        return
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _is_voice_chat(channel) -> bool:
    """A voice/stage channel's built-in text chat — never a place for the bot
    to post on its own initiative."""
    return isinstance(channel, (discord.VoiceChannel, discord.StageChannel))


# ──────────────────────────────────────────────────────────────────────────────
# Now-playing embed
# ──────────────────────────────────────────────────────────────────────────────

def _now_label(track: dict) -> str:
    """"Now Playing" only once audio has actually gone out."""
    status = track.get("_status")
    if status == "retrying":
        return "⏳ Starting (retrying with a fresh link)…"
    if status == "starting":
        return "⏳ Starting…"
    return "🎵 Now Playing"


def _recent_failure(state) -> str | None:
    """The latest playback failure, if recent, as one line."""
    if not state.failures:
        return None
    at, title, reason = state.failures[-1]
    if time.time() - at > _FAILURE_SHOWN_FOR:
        return None
    ago = max(1, int(time.time() - at))
    when = f"{ago}s ago" if ago < 90 else f"{ago // 60}m ago"
    return f"⚠️ **{title}** failed to play {when}: {reason}."


def _build_np_embed(state, vc=None) -> discord.Embed:
    t        = state.current
    duration = t.get("duration") or 0
    # Frame-accurate, so the bar no longer runs ahead of the audio by however
    # long the track has spent paused.
    elapsed  = max(0, int(playback_position(state, vc)))
    if duration:
        elapsed = min(elapsed, int(duration))
    vol      = int(state.volume * 100)

    bar = make_progress_bar(elapsed, duration)
    paused = "⏸️ " if (state.clock is not None and state.clock.is_paused) else ""
    progress = f"{paused}`{fmt_duration(elapsed)}` `[{bar}]` `{fmt_duration(duration)}`"

    embed = discord.Embed(
        title=_now_label(t),
        description=f"**[{t['title']}]({t.get('webpage_url') or '#'})**",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="Progress",     value=progress,                            inline=False)
    embed.add_field(name="Requested by", value=t.get("requester") or "—",          inline=True)
    embed.add_field(name="Volume",       value=f"{vol}%",                          inline=True)
    embed.add_field(name="Loop",         value=("🔂 Song" if state.loop else "🔁 Queue"
                                                if state.loop_queue else "➡️ Off"), inline=True)
    if state.filter_label:
        embed.add_field(name="Filter", value=f"🎛️ {state.filter_label}", inline=True)
    if state.queue:
        embed.add_field(name="Up next", value=f"{len(state.queue)} track(s)", inline=True)
    if t.get("thumbnail"):
        embed.set_thumbnail(url=t["thumbnail"])
    return embed


# ──────────────────────────────────────────────────────────────────────────────
# Attachment handling
#
# Users can attach audio (or video) files directly to a `-play` / `-playnext`
# message. The bot then treats the attachments as the source instead of any
# accompanying text query. Single-file = queue immediately; multi-file = same
# "First only / Whole / Cancel" prompt we use for URL playlists.
# ──────────────────────────────────────────────────────────────────────────────

_AUDIO_VIDEO_EXTS = {
    # Audio
    ".mp3", ".wav", ".ogg", ".m4a", ".flac", ".opus", ".aac", ".wma",
    # Video (audio gets extracted by ffmpeg automatically)
    ".mp4", ".webm", ".mov", ".mkv", ".m4v",
}


def _audio_attachments(message: discord.Message) -> list[discord.Attachment]:
    """Filter message.attachments down to things that look playable.

    We trust `content_type` when Discord set it (audio/* and video/*), and
    fall back to file extension when it's missing or unhelpful ("application/
    octet-stream" is common from some clients). Anything else is skipped —
    no need to queue someone's .png and have ffmpeg choke on it later.
    """
    out: list[discord.Attachment] = []
    for att in message.attachments:
        ct = (att.content_type or "").lower()
        if ct.startswith(("audio/", "video/")):
            out.append(att)
            continue
        ext = os.path.splitext(att.filename)[1].lower()
        if ext in _AUDIO_VIDEO_EXTS:
            out.append(att)
    return out


async def _attachment_to_track(att: discord.Attachment) -> dict:
    """Build a queue-ready track dict from a Discord attachment.

    `att.url` is a Discord CDN link, so we can lean on the existing
    fetch_discord_track helper — it grabs duration via ffprobe and derives
    a title from the filename. The attachment URL embeds the filename in
    the path, so the title comes out human-readable automatically.
    """
    return await audio.fetch_discord_track(att.url)


def _extract_shuffle_flag(query: str | None) -> tuple[bool, str | None]:
    """Pull `--shuffle` / `--random` tokens out of a query string.

    Returns `(shuffle_requested, cleaned_query)`. If the cleaned query is
    empty, returns None for the query (so callers can check truthiness
    consistently). Token-level matching — a literal `--shuffle` somewhere
    inside a URL won't get hit (URLs don't contain spaces).
    """
    if not query:
        return False, query
    shuffle = False
    kept: list[str] = []
    for tok in query.split():
        if tok.lower() in ("--shuffle", "--random"):
            shuffle = True
        else:
            kept.append(tok)
    cleaned = " ".join(kept).strip()
    return shuffle, (cleaned or None)


# ──────────────────────────────────────────────────────────────────────────────
# Cog
# ──────────────────────────────────────────────────────────────────────────────

class MusicCog(commands.Cog, name="Music"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Per-guild marker of the last play we recorded — {guild_id: (key, ts)}.
        # Lets _on_track_finished suppress loop-mode laps so an overnight loop
        # can't inflate one song's play count.
        self._last_recorded_play: dict[int, tuple[str, float]] = {}
        # Strong references to fire-and-forget tasks: asyncio keeps only weak
        # ones, so an unreferenced task can be collected mid-flight.
        self._tasks: set[asyncio.Task] = set()
        # guild → set when an autojoin connect for it finishes (see autojoin).
        self._joining: dict[int, asyncio.Event] = {}

    async def cog_check(self, ctx: commands.Context) -> bool:
        # Everything here touches ctx.guild / voice state; DMs would traceback.
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    async def cog_load(self) -> None:
        """Restore where each server last used music commands."""
        try:
            saved = await db.list_app_settings(_MUSIC_CHANNEL_KEY)
        except Exception:
            log.debug("couldn't load saved music channels", exc_info=True)
            return
        for key, value in saved.items():
            try:
                gid, cid = int(key[len(_MUSIC_CHANNEL_KEY):]), int(value)
            except ValueError:
                continue
            state = get_state(gid)
            if state.text_channel_id is None:
                state.text_channel_id = cid

    # ── background work that must not fail silently ──────────────────────────

    def _spawn(self, coro, what: str) -> asyncio.Task:
        """create_task with a strong reference and logged exceptions (event
        loop thread only)."""
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(partial(_log_future, what))
        return task

    def _from_thread(self, coro, what: str):
        """Schedule `coro` on the bot's loop from any thread (the audio
        thread's after-callback included); its exception is logged."""
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, self.bot.loop)
        except RuntimeError:              # loop closed: shutting down
            coro.close()
            return None
        fut.add_done_callback(partial(_log_future, what))
        return fut

    def _dispatch(self, event: str, *args) -> None:
        dispatch = getattr(self.bot, "dispatch", None)
        if dispatch is not None:
            try:
                dispatch(event, *args)
            except Exception:
                log.debug("dispatch %s failed", event, exc_info=True)

    # ── playback engine ──────────────────────────────────────────────────────
    async def _play_next(
        self, guild: discord.Guild, vc: discord.VoiceClient, expected_gen: int,
    ) -> None:
        state = get_state(guild.id)
        # Serialise advancement per guild. Without the lock, two concurrent
        # kickstarts (two users queueing into an idle bot at once) both popped
        # a track; whichever resolved second failed its gen check and its
        # track simply vanished — "added ✅" but never played, never re-queued.
        # With the lock, the loser re-checks the gen before touching the queue
        # and bails with its track still safely queued.
        async with state.advance_lock:
            # Callers can hold a voice client from before a `rejoin`; always
            # drive the guild's current one.
            vc = getattr(guild, "voice_client", None) or vc
            await self._advance_locked(guild, vc, state, expected_gen)

    async def _advance_locked(
        self, guild: discord.Guild, vc: discord.VoiceClient,
        state, expected_gen: int,
    ) -> None:
        if state.playback_gen != expected_gen:
            return
        # Something is already audibly playing — the after-callback chain owns
        # advancement; running here anyway would stomp the active track.
        if vc.is_playing() or vc.is_paused():
            return
        # The current track's source has ended (or is being retried) but its
        # end handling hasn't run yet: that handler decides what's next. A
        # kickstart landing in that gap used to replace the track before its
        # failure could be reported.
        cur = state.current
        if cur is not None and cur.get("_status") in _LIVE:
            return

        if state.loop and cur is not None:
            state.queue.appendleft(cur)
        elif state.loop_queue and cur is not None and cur.get("_status") == "finished":
            # Queue loop: the song that just finished goes round again, last.
            state.queue.append(requeue_copy(cur))

        # Iterate (not recurse) past unresolvable tracks — a long playlist of
        # dead links shouldn't build a call stack, and a failed track must not
        # stay in `state.current` or loop mode would re-queue and retry it
        # forever. Failures are collected and reported to the channel — they
        # used to vanish into the log, which read as "queued but never played".
        skipped: list[str] = []
        while True:
            if state.playback_gen != expected_gen:
                return
            if not state.queue:
                seed = state.current
                state.current          = None
                state.track_started_at = None
                state.clock            = None
                # Queue ran dry — autoplay (if enabled) finds a related track
                # and keeps the music going.
                if seed is not None and not state.loop:
                    if await self._try_autoplay(guild, state, seed):
                        if state.playback_gen != expected_gen:
                            return
                        continue
                self._report_skips(guild, skipped, queue_ended=True)
                return

            track = state.queue.popleft()
            state.current = track
            # A fresh start of this queue item (a loop lap counts as one).
            track["_status"] = "starting"
            track["_retries"] = 0

            try:
                await audio.resolve_stream_url(track)
            except Exception as e:
                if state.playback_gen != expected_gen or state.current is not track:
                    return
                log.warning("Failed to resolve %r — skipping: %s",
                            track.get("title"), redact_secrets(str(e)))
                reason = audio.friendly_error_msg(e)
                skipped.append(f"**{track.get('title') or 'Unknown'}** — {reason}")
                self._record_failure(state, track, reason)
                track["_status"] = "failed"
                state.current = None
                continue

            # The resolve awaited network I/O; a stop/seek/skip may have landed
            # meanwhile and bumped the gen. Starting playback now would override it.
            if state.playback_gen != expected_gen or state.current is not track:
                return
            break

        self._report_skips(guild, skipped, queue_ended=False)

        # SponsorBlock: look up skip segments (cached per video, never fatal).
        track["_sb_segments"] = []
        try:
            gs = await db.get_guild_settings(guild.id)
            if gs["sponsorblock"] != 0:   # NULL (never configured) defaults to on
                track["_sb_segments"] = await audio.fetch_sponsor_segments(
                    track.get("webpage_url"))
        except Exception:
            log.exception("SponsorBlock lookup failed")
        if state.playback_gen != expected_gen or state.current is not track:
            return

        # A track interrupted by a lost connection resumes where it stopped;
        # anything else is a fresh play (history + Last.fm once audio starts).
        position = float(track.pop("_resume_at", 0.0) or 0.0)
        if position <= 0:
            track["_fresh"] = True
            # If the video opens inside a skippable segment, start right past it.
            for s, e in track["_sb_segments"]:
                if s <= 1.0 and e > position:
                    position = e + 0.05
        self._begin_playback(guild, vc, state, track, position=position)

    # ── engine → channel feedback ────────────────────────────────────────────

    def _notify_channel(self, guild: discord.Guild, preferred_id: int | None = None):
        """Where engine messages go: the channel the track was requested from,
        else where music was last controlled, else where the bot was last
        summoned, else the server's system channel. Never a voice channel's
        own chat unless that's exactly where the request came from."""
        state = get_state(guild.id)
        for cid in (preferred_id, state.text_channel_id):
            ch = self.bot.get_channel(cid) if cid else None
            if ch is not None:
                return ch
        rec = self.bot.get_cog("Recording")
        ch = getattr(rec, "text_channels", {}).get(guild.id) if rec else None
        if ch is not None and not _is_voice_chat(ch):
            return ch
        sysch = getattr(guild, "system_channel", None)
        me = getattr(guild, "me", None)
        if sysch is not None and me is not None:
            try:
                if sysch.permissions_for(me).send_messages:
                    return sysch
            except Exception:
                pass
        return None

    def _notify(self, guild: discord.Guild, text: str, *, channel_id: int | None = None) -> None:
        """Best-effort message about playback. Fire-and-forget: engine
        feedback must never break playback."""
        channel = self._notify_channel(guild, channel_id)
        if channel is None:
            log.info("music notice for guild %s had nowhere to go: %s", guild.id, text[:200])
            return

        async def _send():
            try:
                await channel.send(
                    text, allowed_mentions=discord.AllowedMentions.none())
            except Exception:
                log.debug("music notify failed", exc_info=True)

        self._spawn(_send(), "music notice")

    def _report_skips(self, guild: discord.Guild, skipped: list[str],
                      *, queue_ended: bool) -> None:
        """Tell the channel which queued tracks were dropped and why."""
        if not skipped:
            return
        if len(skipped) == 1:
            msg = f"⚠️ Skipping {skipped[0]}"
        else:
            shown = "\n".join(f"• {s}" for s in skipped[:5])
            extra = f"\n…and {len(skipped) - 5} more" if len(skipped) > 5 else ""
            msg = f"⚠️ Skipped **{len(skipped)}** unplayable track(s):\n{shown}{extra}"
        if queue_ended:
            msg += "\nThe queue is empty now."
        self._notify(guild, msg[:1900])

    @staticmethod
    def _record_failure(state, track: dict, reason: str) -> None:
        state.failures.append((time.time(), track.get("title") or "Unknown", reason))

    def _fail_track(self, guild: discord.Guild, state, track: dict,
                    info: FailureInfo, *, stopped_at: float | None, retried: bool) -> None:
        """Final failure: say what happened where it was requested, record it
        (so `queue` can explain an empty queue), and make sure it's neither
        counted as played nor re-queued by loop mode. The caller advances."""
        track["_status"] = "failed"
        if state.current is track:
            state.current          = None
            state.track_started_at = None
            state.clock            = None
        title = track.get("title") or "that track"
        where = (f"stopped at `{fmt_duration(int(stopped_at))}` — "
                 if stopped_at else "")
        retry_note = " I retried once with a fresh link." if retried else ""
        nxt = (" Moving on to the next track." if state.queue
               else " Nothing else is queued.")
        self._record_failure(state, track, info.reason)
        self._notify(
            guild,
            f"❌ Couldn't play **{title}** — {where}{info.reason}.{retry_note} "
            f"{info.advice}{nxt}"[:1900],
            channel_id=track.get("request_channel_id"),
        )
        self._dispatch("track_failed", guild, track, info.category)

    def _begin_playback(
        self, guild: discord.Guild, vc: discord.VoiceClient,
        state, track: dict, *, position: float = 0.0,
    ) -> bool:
        """Start (or restart) the ffmpeg source for `track` at `position`.

        The single entry point for everything that makes sound: _play_next,
        seek, filter changes, retries, rejoin and SponsorBlock hops all come
        through here so the gen-bump ordering, filter application, and watcher
        spawning stay consistent. Expects `track["stream_url"]` to be resolved.

        Returns True once the player has the source. That is NOT proof of
        sound — the track becomes "playing" only when its first music frame
        goes out (`_on_first_audio`). On False the failure has been handled:
        a problem with this track is reported and the queue moves on; a
        problem that would stop *any* track (no FFmpeg/Opus/voice connection)
        is reported and the track waits at the front of the queue.
        """
        state.playback_gen += 1
        my_gen = state.playback_gen
        if vc.is_playing() or vc.is_paused():
            vc.stop()   # the old after-callback sees the bumped gen and bails
        track["_status"] = "starting"

        # ffmpeg is told to start at `position`, so that's exactly where the
        # clock is anchored. `rate` accounts for speed filters: under nightcore
        # one second of delivered audio covers 1.25 seconds of the track, and
        # without this the position ran 25% slow for the whole song.
        rate = rate_from_filter(state.filter_af)
        if state.clock is None:
            state.clock = PlaybackClock()
        state.clock.rebase(position, rate=rate)
        state.track_started_at = time.time() - position

        # Source construction spawns FFmpeg — it belongs inside the startup
        # error handling (a missing ffmpeg used to escape from here, leaving a
        # dead track marked as current).
        try:
            source = audio.create_audio_source(
                track["stream_url"], volume=state.volume,
                seek=position, filter_af=state.filter_af,
                http_headers=track.get("_http_headers"),
            )
        except Exception as e:
            return self._startup_failed(guild, vc, state, track, my_gen, e, stage="source")

        if isinstance(source, DuckingSource):
            source.on_first_music = partial(self._first_audio_from_thread,
                                            guild, track, my_gen)

        def after(err):
            # Audio thread. Snapshot the evidence now — discord.py cleans the
            # source up as soon as this returns — and decide on the loop.
            outcome = PlaybackOutcome.capture(source, err, position=position, rate=rate)
            self._from_thread(
                self._on_playback_ended(guild, vc, track, my_gen, outcome),
                "playback end handling")

        try:
            vc.play(source, after=after)
        except Exception as e:
            try:
                source.cleanup()          # kill the FFmpeg we just spawned
            except Exception:
                log.debug("source cleanup after a failed play()", exc_info=True)
            return self._startup_failed(guild, vc, state, track, my_gen, e, stage="play")

        if track.get("_sb_segments"):
            self._spawn(self._sb_watch(guild, vc, state, track, my_gen), "SponsorBlock watcher")
        return True

    def _startup_failed(self, guild, vc, state, track: dict, my_gen: int,
                        exc: BaseException, *, stage: str) -> bool:
        kind, text = _startup_problem(exc, stage)
        state.track_started_at = None
        state.clock            = None
        if kind == "systemic":
            log.error("Playback can't start in guild %s (%s): %r", guild.id, stage, exc)
            # Nothing could play right now, so don't burn through the queue:
            # this track waits at the front until the cause is fixed.
            track["_status"] = "queued"
            if state.current is track:
                state.current = None
            if not (state.queue and state.queue[0] is track):
                state.queue.appendleft(track)
            self._notify(guild, f"{text} **{track.get('title') or 'Your track'}** is still queued.",
                         channel_id=track.get("request_channel_id"))
            return False
        log.warning("Couldn't start %r (%s): %s", track.get("title"), stage,
                    redact_secrets(repr(exc)))
        self._fail_track(guild, state, track,
                         FailureInfo("start_failed", False, text, "Try a different upload or link."),
                         stopped_at=None, retried=False)
        self._from_thread(self._play_next(guild, vc, my_gen), "advance after a failed start")
        return False

    def _first_audio_from_thread(self, guild, track: dict, gen: int) -> None:
        try:
            self.bot.loop.call_soon_threadsafe(self._on_first_audio, guild, track, gen)
        except RuntimeError:
            pass                                   # loop closed: shutting down

    def _on_first_audio(self, guild, track: dict, gen: int) -> None:
        """The first frame of real music went out: the track is playing now.
        Only now does a fresh start count as played history / Last.fm "now
        playing" — a lookup or a vc.play() call proves nothing."""
        state = get_state(guild.id)
        if state.playback_gen != gen or state.current is not track:
            return
        track["_status"] = "playing"
        if track.pop("_fresh", False):
            self._record_history(state, track)
            self._announce_now_playing(guild, track)
            self._dispatch("track_audible", guild, track)

    async def _on_playback_ended(self, guild, vc, track: dict, my_gen: int,
                                 outcome: PlaybackOutcome) -> None:
        try:
            await self._handle_playback_end(guild, vc, track, my_gen, outcome)
        except Exception:
            # A bug here must not leave the track "live" forever (that would
            # block every later advance). Fail it visibly and move on.
            log.exception("Playback end handling crashed for %r", track.get("title"))
            state = get_state(guild.id)
            if state.current is track and state.playback_gen == my_gen:
                self._fail_track(guild, state, track,
                                 FailureInfo("internal", False, "something went wrong on my side",
                                             "The details are in my log."),
                                 stopped_at=None, retried=False)
                await self._play_next(guild, vc, my_gen)

    async def _handle_playback_end(self, guild, vc, track: dict, my_gen: int,
                                   outcome: PlaybackOutcome) -> None:
        """Every FFmpeg run ends here: superseded, skipped, completed, or
        failed — and each is handled differently."""
        state = get_state(guild.id)
        if state.playback_gen != my_gen:
            return       # stop / seek / filter / rejoin / retry already took over
        if state.current is not track:
            await self._play_next(guild, vc, my_gen)   # skipped: move on
            return
        verdict = judge_playback(track, outcome)
        if verdict.kind == "completed":
            track["_status"] = "finished"
            # The one moment a play counts toward stats/scrobbling.
            self._spawn(self._on_track_finished(guild, track), "track-finished bookkeeping")
            await self._play_next(guild, vc, my_gen)
            return
        tail = redact_secrets(outcome.stderr.strip())[-600:]
        log.warning(
            "Playback failed (%s, %s) for %r: music %.1fs from %.1fs, rc=%s, error=%s%s",
            verdict.kind, verdict.info.category, track.get("title"), outcome.music_seconds,
            outcome.start_position, outcome.returncode,
            redact_secrets(repr(outcome.error)) if outcome.error else None,
            f"\nffmpeg said:\n{tail}" if tail else "")
        await self._recover_or_fail(guild, track, my_gen, verdict, outcome)

    async def _recover_or_fail(self, guild, track: dict, my_gen: int,
                               verdict: Verdict, outcome: PlaybackOutcome) -> None:
        state = get_state(guild.id)
        async with state.advance_lock:
            if state.playback_gen != my_gen or state.current is not track:
                return
            vc = getattr(guild, "voice_client", None)
            if vc is None or not vc.is_connected():
                self._interrupted(guild, state, track, verdict.resume_at)
                return
            if outcome.music_seconds >= _RETRY_HEALTHY_SECONDS:
                track["_retries"] = 0
            info = verdict.info
            retried = track.get("_retries", 0) > 0
            if info.retryable and not retried:
                track["_retries"] = track.get("_retries", 0) + 1
                track["_status"] = "retrying"
                log.info("Retrying %r once with a fresh stream URL (from %.1fs)",
                         track.get("title"), verdict.resume_at)
                try:
                    await audio.resolve_stream_url(track, force=True)
                    refresh_error = None
                except Exception as e:
                    refresh_error = e
                # The refresh awaited the network: a stop, skip, seek or rejoin
                # meanwhile wins — never resurrect a track someone stopped.
                if state.playback_gen != my_gen or state.current is not track:
                    return
                vc = getattr(guild, "voice_client", None)
                if vc is None or not vc.is_connected():
                    self._interrupted(guild, state, track, verdict.resume_at)
                    return
                if refresh_error is None:
                    self._begin_playback(guild, vc, state, track, position=verdict.resume_at)
                    return
                retried = False          # the reason below already says the refresh failed
                info = FailureInfo(
                    "refresh_failed", False,
                    f"{info.reason}, and I couldn't get a fresh link "
                    f"({audio.friendly_error_msg(refresh_error)})",
                    "Try again later or pick a different upload.")
            self._fail_track(guild, state, track, info,
                             stopped_at=outcome.reached if verdict.kind == "premature" else None,
                             retried=retried)
            await self._advance_locked(guild, vc, state, my_gen)

    def _interrupted(self, guild, state, track: dict, resume_at: float) -> None:
        """The voice connection went away under the track: not the track's
        fault. It goes back to the front of the queue and resumes from where
        it stopped once the bot is back (`rejoin` or `play`)."""
        track["_status"] = "queued"
        track["_resume_at"] = resume_at
        if state.current is track:
            state.current          = None
            state.track_started_at = None
            state.clock            = None
        if not (state.queue and state.queue[0] is track):
            state.queue.appendleft(track)
        self._notify(guild, f"⚠️ I lost the voice connection during **{track.get('title')}**. "
                            "Run `rejoin` and I'll pick it up where it stopped.",
                     channel_id=track.get("request_channel_id"))

    async def _sb_watch(
        self, guild: discord.Guild, vc: discord.VoiceClient,
        state, track: dict, my_gen: int,
    ) -> None:
        """Poll playback position and hop over SponsorBlock segments.

        Each hop restarts playback through _begin_playback, which bumps the
        gen and spawns a fresh watcher — so this instance exits after one skip.
        """
        segments = track.get("_sb_segments") or []
        while (state.playback_gen == my_gen and state.current is track
               and vc.is_connected()):
            if vc.is_playing():
                elapsed = playback_position(state, vc)
                for s, e in segments:
                    # Skip only if enough of the segment is left to be worth
                    # the ffmpeg restart hiccup.
                    if s <= elapsed < e - 1.5:
                        log.info("SponsorBlock: skipping %.0fs–%.0fs in %r",
                                 s, e, track.get("title"))
                        self._begin_playback(guild, vc, state, track,
                                             position=e + 0.05)
                        return
            await asyncio.sleep(1)

    async def _try_autoplay(self, guild: discord.Guild, state, seed: dict) -> bool:
        """Queue one related track when the queue empties, if the guild opted
        into autoplay. Returns True if something was queued."""
        generation, epoch = state.playback_gen, state.queue_epoch
        try:
            gs = await db.get_guild_settings(guild.id)
            if gs["autoplay"] != 1:    # opt-in: NULL/0 = off
                return False
            exclude = {t.get("webpage_url") for t in state.history
                       if t.get("webpage_url")}
            exclude.add(seed.get("webpage_url"))
            nxt = await audio.fetch_autoplay_track(seed, exclude_urls=exclude)
        except Exception:
            log.exception("Autoplay failed")
            return False
        if state.playback_gen != generation or state.queue_epoch != epoch:
            return False
        if nxt is None:
            log.info("Autoplay: nothing related to %r found", seed.get("title"))
            # Autoplay is opt-in, so a user is expecting the music to keep
            # going — tell them why it stopped instead of going silent.
            self._notify(guild, "📻 Autoplay couldn't find a related track — "
                                "the queue has ended.")
            return False
        state.queue.append(nxt)
        log.info("Autoplay queued %r", nxt.get("title"))
        return True

    def _record_history(self, state, track: dict) -> None:
        """Append a lightweight stub to the guild's play history. Stream URLs
        aren't kept — `back` re-resolves so replays never hit expired links."""
        h = state.history
        if h and h[-1].get("webpage_url") == track.get("webpage_url"):
            return   # loop replays / SB hops shouldn't fill history with one song
        h.append({
            "title":          track.get("title"),
            "webpage_url":    track.get("webpage_url"),
            "duration":       track.get("duration") or 0,
            "thumbnail":      track.get("thumbnail"),
            "requester":      track.get("requester"),
            "stream_url":     None,
            "_needs_resolve": True,
            "_search_query":  track.get("webpage_url") or track.get("_search_query"),
        })

    async def _on_track_finished(self, guild: discord.Guild, track: dict) -> None:
        """A track played to its natural end. Record it for the "wrapped" play
        stats, then hand off to Last.fm scrobbling. Loop-mode laps of the same
        song are recorded once, not every lap."""
        url      = track.get("webpage_url")
        title    = track.get("title") or url or "Unknown"
        duration = int(track.get("duration") or 0)
        if duration <= 0:
            return   # unresolved stub / live stream — not a real, countable play

        gid = guild.id
        now = time.time()
        key = url or title
        last = self._last_recorded_play.get(gid)
        # A loop lap = same song as the last record, while loop is on, arriving
        # within roughly its own length. Advance the marker either way so a long
        # loop keeps suppressing.
        is_loop_lap = (
            get_state(gid).loop and last and last[0] == key
            and (now - last[1]) < duration + 30
        )
        self._last_recorded_play[gid] = (key, now)
        if is_loop_lap:
            return

        try:
            await db.record_play(
                guild_id=gid, requester_id=track.get("requester_id"),
                url=url, title=title, artist=track.get("artist"),
                duration=duration, is_music=bool(track.get("is_music_meta")),
            )
        except Exception:
            log.exception("record_play failed for %r", title)

        # Last.fm scrobble for opted-in listeners (no-op if unconfigured).
        await self._scrobble_finished(guild, track)

    async def _scrobble_finished(self, guild: discord.Guild, track: dict) -> None:
        """Hand a finished track to the Last.fm cog, if it's loaded. Kept as a
        thin seam so the music engine doesn't hard-depend on scrobbling."""
        cog = self.bot.get_cog("LastFM")
        if cog is None:
            return
        try:
            await cog.scrobble_track(guild, track)
        except Exception:
            log.exception("Last.fm scrobble hand-off failed")

    def _announce_now_playing(self, guild: discord.Guild, track: dict) -> None:
        """Fire-and-forget Last.fm 'now playing' for a freshly started track.
        Called when its first audio frame goes out — not on seek/SB-hop, and
        never for a track that failed before making a sound."""
        cog = self.bot.get_cog("LastFM")
        if cog is None:
            return
        self._spawn(cog.now_playing(guild, track), "Last.fm now-playing")

    def _kickstart(self, guild: discord.Guild, vc: discord.VoiceClient) -> None:
        state = get_state(guild.id)
        self._from_thread(self._play_next(guild, vc, state.playback_gen), "queue advance")

    def _busy(self, guild: discord.Guild) -> bool:
        """Playing, paused, or a track is starting/being retried."""
        vc = getattr(guild, "voice_client", None)
        if vc is not None and (vc.is_playing() or vc.is_paused()):
            return True
        cur = get_state(guild.id).current
        return cur is not None and cur.get("_status") in _LIVE

    @staticmethod
    def _abandon_starting(state) -> bool:
        """Drop a current track that hasn't made a sound yet (starting, or
        waiting on a retry) so the next one can go. Its pending start/retry
        sees `current` change and stands down."""
        cur = state.current
        if cur is not None and cur.get("_status") in ("starting", "retrying"):
            cur["_status"] = "skipped"
            if state.loop_queue:
                state.queue.append(requeue_copy(cur))   # queue loop: it comes round again
            state.current = None
            return True
        return False

    async def _admit_after_lookup(self, ctx: commands.Context, epoch: int,
                                  title: str):
        """A typed request's lookup has finished: may its result still be
        queued? Returns the live voice client, or None after telling the user
        why not (stopped or disconnected while we were searching)."""
        state = get_state(ctx.guild.id)
        if state.queue_epoch != epoch:
            changed = "The queue was cleared" if state.queue_change_reason == "cleared" else "Playback was stopped"
            await ctx.send(f"⏹️ {changed} while I was looking that up, "
                           f"so I didn't queue **{title}**.")
            return None
        vc = ctx.guild.voice_client
        if vc is None or not vc.is_connected():
            await ctx.send(f"❌ I left the voice channel while looking that up, so "
                           f"**{title}** wasn't queued. Run the command again.")
            return None
        return vc

    # ── autojoin (driven by cogs/admin.py's voice-state listener) ────────────

    async def _autojoin_notice_channel(self, guild: discord.Guild):
        """Where the recorder's one-time notice goes after an unsummoned join:
        where music was last controlled (remembered across restarts), else the
        voice-control reply channel if one is set, else the system channel.
        Never a voice channel's own chat."""
        me = guild.me
        candidates = []
        state = get_state(guild.id)
        if state.text_channel_id:
            candidates.append(self.bot.get_channel(state.text_channel_id))
        try:
            cid = (await db.get_voice_settings(guild.id))["feedback_channel_id"]
            if cid:
                candidates.append(guild.get_channel(cid))
        except Exception:
            pass
        candidates.append(guild.system_channel)
        for cand in candidates:
            if cand is None or _is_voice_chat(cand):
                continue
            try:
                if cand.permissions_for(me).send_messages:
                    return cand
            except Exception:
                continue
        return None

    async def autojoin(self, member: discord.Member, channel) -> bool:
        """Join `channel` because `member` just entered it in an otherwise
        empty server. Same connection setup as being summoned (receive client
        + recorder, so voice control works straight away); nothing plays."""
        guild = channel.guild
        if guild.voice_client is not None or guild.id in self._joining:
            return False
        done = self._joining[guild.id] = asyncio.Event()
        cls = voice_recv.VoiceRecvClient if voice_recv else discord.VoiceClient
        try:
            try:
                vc = await channel.connect(cls=cls)
            except Exception as e:
                log.warning("Autojoin into %s (guild %s) failed: %s", channel, guild.id, e)
                return False
            rec = self.bot.get_cog("Recording")
            if rec is not None:
                try:
                    # Nobody typed a command, so this join doesn't decide where
                    # replies go (that stays "where music was last used").
                    await rec.attach(vc, await self._autojoin_notice_channel(guild),
                                     reply_here=False)
                except Exception:
                    log.exception("Failed to attach voice recorder after autojoin")
            log.info("Autojoined %s in %s for %s", channel, guild.name, member.display_name)
            return True
        finally:
            done.set()
            self._joining.pop(guild.id, None)

    # ── play ─────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="play", aliases=["p"], description="Play a song, playlist, or attached file",
                              help=(
                                  "Search or paste a URL — YouTube, SoundCloud, Spotify, "
                                  "Bandcamp, audio.com, or a direct file link. Playlists "
                                  "(and audio.com collections) prompt you to "
                                  "choose 'Queue all', 'Shuffle all', or 'First only'.\n\n"
                                  "**Also accepts file attachments.** Drag an .mp3/.wav/.mp4 "
                                  "(etc.) into a message with `-play` and the bot plays the "
                                  "attached file. Multiple attached files trigger the same "
                                  "prompt as URL playlists. Attached file always wins over "
                                  "any text query in the same message.\n\n"
                                  "**Shuffle flag.** Add `--shuffle` (or `--random`) anywhere "
                                  "to skip the prompt and immediately shuffle-queue all items. "
                                  "Shuffle only affects the new batch — existing queue items "
                                  "stay where they are.\n\n"
                                  "**Examples**\n"
                                  "`play never gonna give you up`\n"
                                  "`play https://youtu.be/dQw4w9WgXcQ`\n"
                                  "`play https://youtube.com/playlist?list=... --shuffle`\n"
                                  "`play` *(with multiple .mp3s attached)*"
                              ))
    async def play(
        self, ctx: commands.Context,
        attachment: discord.Attachment | None = None,
        *, query: str | None = None,
    ):
        if not await _ensure_voice(ctx):
            return
        _remember_text_channel(ctx)

        # Pull --shuffle out of the query first — it's valid alongside a URL,
        # a search, or just attachments.
        shuffle, query = _extract_shuffle_flag(query)
        query = unwrap_url(query) if query else query   # tolerate <url> wrapping

        # Attachment wins. Gather candidates from both sources: the slash
        # `attachment:` param (for slash invocations) and message.attachments
        # (for prefix invocations — Discord lets users drop files alongside
        # a prefix command in the same message).
        candidates: list[discord.Attachment] = []
        if attachment is not None:
            candidates.append(attachment)
        # ctx.message.attachments is a list; safe to use even if empty.
        candidates.extend(_audio_attachments(ctx.message))
        # De-dupe by id in case the slash attachment also showed up in the
        # message attachments list.
        seen_ids: set[int] = set()
        unique: list[discord.Attachment] = []
        for a in candidates:
            if a.id not in seen_ids:
                seen_ids.add(a.id)
                unique.append(a)

        # No query AND no attachments → can't do anything.
        if not unique and not query:
            return await ctx.send(
                "❌ Provide a search query, a URL, or attach an audio/video file."
            )

        # If we have attachments, ignore any query.
        if unique:
            if query:
                log.info("play: attachment(s) present, ignoring text query %r", query)
            vc = await _connect_to_author(ctx)
            if vc is None:
                return
            await self._handle_attachments(ctx, vc, unique,
                                            at_front=False, preset_shuffle=shuffle)
            return

        if audio.is_spotify_url(query) and not audio.spotify_ready():
            return await ctx.send(
                "❌ Spotify isn't configured. Add `SPOTIFY_CLIENT_ID` and "
                "`SPOTIFY_CLIENT_SECRET` to `.env` and restart."
            )

        vc = await _connect_to_author(ctx)
        if vc is None:
            return
        state = get_state(ctx.guild.id)
        epoch = state.queue_epoch

        if audio.is_any_playlist(query):
            if not await self._handle_playlist_query(ctx, vc, query, preset_shuffle=shuffle,
                                                     epoch=epoch):
                return
            vc = ctx.guild.voice_client or vc
        else:
            # Single track — shuffle is a no-op, so just ignore the flag.
            async with ctx.typing():
                try:
                    track = await audio.fetch_track(query)
                except ValueError as e:
                    return await ctx.send(f"❌ {audio.friendly_error_msg(e)}")
                except Exception as e:
                    log.exception("fetch_track failed")
                    return await ctx.send(f"❌ {audio.friendly_error_msg(e)}")
                if not track:
                    return await ctx.send("❌ Could not find anything for that query.")
            # The search awaited the network: a stop or disconnect meanwhile
            # means this result is obsolete — don't restart the music.
            vc = await self._admit_after_lookup(ctx, epoch, track["title"])
            if vc is None:
                return
            track["requester"] = ctx.author.display_name
            track["requester_id"] = ctx.author.id
            track["request_channel_id"] = ctx.channel.id
            idle = not self._busy(ctx.guild)
            state.queue.append(track)
            # "Added" is what we know. Whether it actually plays is reported
            # separately — a failure gets its own message.
            await ctx.send(
                f"✅ Added **{track['title']}** "
                f"`[{fmt_duration(track['duration'])}]` to the queue."
                + (" Starting…" if idle else "")
            )

        if not vc.is_playing() and not vc.is_paused():
            self._kickstart(ctx.guild, vc)

    async def _handle_attachments(
        self, ctx: commands.Context, vc: discord.VoiceClient,
        attachments: list[discord.Attachment], *,
        at_front: bool, preset_shuffle: bool = False,
    ) -> None:
        """Queue one or more attached files. Multiple files trigger the same
        First/All/Shuffle/Cancel prompt as URL playlists. `at_front=True`
        mirrors playnext semantics (insert at queue head). `preset_shuffle=True`
        skips the prompt and goes directly to 'shuffle all'."""
        state = get_state(ctx.guild.id)
        epoch = state.queue_epoch

        # Single file: queue it directly. Shuffle is a no-op on one item.
        if len(attachments) == 1:
            async with ctx.typing():
                try:
                    track = await _attachment_to_track(attachments[0])
                except Exception as e:
                    log.exception("attachment fetch failed")
                    return await ctx.send(f"❌ Couldn't read attachment: "
                                          f"{audio.friendly_error_msg(e)}")
            vc = await self._admit_after_lookup(ctx, epoch, track["title"])
            if vc is None:
                return
            track["requester"] = ctx.author.display_name
            track["requester_id"] = ctx.author.id
            track["request_channel_id"] = ctx.channel.id
            if at_front:
                state.queue.appendleft(track)
            else:
                state.queue.append(track)
            verb = "will play next" if at_front else "added to the queue"
            await ctx.send(
                f"✅ **{track['title']}** "
                f"`[{fmt_duration(track['duration'])}]` {verb}."
            )
            if not vc.is_playing() and not vc.is_paused():
                self._kickstart(ctx.guild, vc)
            return

        # Multiple files. If --shuffle was passed, skip the prompt.
        if preset_shuffle:
            choice = "shuffle"
            prompt = await ctx.send("🎲 Loading & shuffling…")
        else:
            class AttachmentPrompt(discord.ui.View):
                def __init__(self):
                    super().__init__(timeout=30)
                    self.choice = None

                async def interaction_check(self, interaction: discord.Interaction) -> bool:
                    if interaction.user != ctx.author:
                        await interaction.response.send_message(
                            "❌ Only the requester can choose.", ephemeral=True,
                        )
                        return False
                    return True

                @discord.ui.button(label="All", style=discord.ButtonStyle.success, emoji="📋")
                async def all_btn(self, i, _b):
                    self.choice = "all"; self.stop(); await i.response.defer()

                @discord.ui.button(label="Shuffle all", style=discord.ButtonStyle.success, emoji="🎲")
                async def shuffle_btn(self, i, _b):
                    self.choice = "shuffle"; self.stop(); await i.response.defer()

                @discord.ui.button(label="First only", style=discord.ButtonStyle.primary, emoji="🎵")
                async def one_btn(self, i, _b):
                    self.choice = "one"; self.stop(); await i.response.defer()

                @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary, emoji="❌")
                async def cancel_btn(self, i, _b):
                    self.choice = "cancel"; self.stop(); await i.response.defer()

            view = AttachmentPrompt()
            action_word = "play next" if at_front else "queue"
            prompt = await ctx.send(
                f"📋 {len(attachments)} files attached — what should I {action_word}?",
                view=view,
            )
            await view.wait()
            choice = view.choice

            if choice in (None, "cancel"):
                return await prompt.edit(content="❌ Cancelled.", view=None)

            await prompt.edit(content="⏳ Loading…", view=None)

        # Resolve. "First only" is one ffprobe; the rest are sequential
        # (parallel would hammer the same CDN host and risk rate-limiting).
        targets = attachments if choice in ("all", "shuffle") else attachments[:1]
        tracks: list[dict] = []
        for att in targets:
            try:
                t = await _attachment_to_track(att)
            except Exception as e:
                log.exception("attachment fetch failed for %s", att.filename)
                await ctx.send(f"⚠️ Skipping `{att.filename}`: {e}")
                continue
            t["requester"] = ctx.author.display_name
            t["requester_id"] = ctx.author.id
            tracks.append(t)

        if not tracks:
            return await prompt.edit(content="❌ None of the attachments were playable.")
        vc = await self._admit_after_lookup(ctx, epoch, f"{len(tracks)} file(s)")
        if vc is None:
            return await prompt.edit(content="❌ Not queued.")
        for t in tracks:
            t["request_channel_id"] = ctx.channel.id

        # Shuffle only the new batch — existing queue stays put.
        if choice == "shuffle":
            random.shuffle(tracks)

        if at_front:
            # Reverse so the first track in `tracks` lands at queue position 0.
            for t in reversed(tracks):
                state.queue.appendleft(t)
        else:
            for t in tracks:
                state.queue.append(t)

        if choice == "shuffle":
            verb = "will play next" if at_front else "queued"
            await prompt.edit(content=f"🎲 Shuffled & {verb}: **{len(tracks)} files**.")
        elif choice == "all":
            verb = "will play next" if at_front else "added to the queue"
            await prompt.edit(content=f"📋 **{len(tracks)} files** {verb}.")
        else:
            t = tracks[0]
            verb = "will play next" if at_front else "added"
            await prompt.edit(
                content=f"✅ **{t['title']}** `[{fmt_duration(t['duration'])}]` {verb}."
            )

        if not vc.is_playing() and not vc.is_paused():
            self._kickstart(ctx.guild, vc)


    async def _handle_playlist_query(
        self, ctx: commands.Context, vc: discord.VoiceClient, query: str,
        *, preset_shuffle: bool = False, epoch: int | None = None,
    ) -> bool:
        """Prompt + fetch + queue a playlist. Returns True if anything was
        queued (the caller then kickstarts)."""
        state = get_state(ctx.guild.id)
        if epoch is None:
            epoch = state.queue_epoch

        # If --shuffle was passed on the command, skip the prompt entirely
        # and go straight to "shuffle and queue all". Power-user shortcut.
        if preset_shuffle:
            choice = "shuffle"
        else:
            class PlaylistPrompt(discord.ui.View):
                def __init__(self):
                    super().__init__(timeout=30)
                    self.choice = None

                async def interaction_check(self, interaction: discord.Interaction) -> bool:
                    if interaction.user != ctx.author:
                        await interaction.response.send_message(
                            "❌ Only the requester can choose.", ephemeral=True
                        )
                        return False
                    return True

                @discord.ui.button(label="Queue all", style=discord.ButtonStyle.success, emoji="📋")
                async def all_btn(self, i, _b):
                    self.choice = "all"; self.stop(); await i.response.defer()

                @discord.ui.button(label="Shuffle all", style=discord.ButtonStyle.success, emoji="🎲")
                async def shuffle_btn(self, i, _b):
                    self.choice = "shuffle"; self.stop(); await i.response.defer()

                @discord.ui.button(label="First only", style=discord.ButtonStyle.primary, emoji="🎵")
                async def one_btn(self, i, _b):
                    self.choice = "one"; self.stop(); await i.response.defer()

                @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary, emoji="❌")
                async def cancel_btn(self, i, _b):
                    self.choice = "cancel"; self.stop(); await i.response.defer()

            view = PlaylistPrompt()
            prompt = await ctx.send("📋 That looks like a playlist — what should I do?", view=view)
            await view.wait()
            choice = view.choice

            if choice in (None, "cancel"):
                await prompt.edit(content="❌ Cancelled.", view=None)
                return False

            await prompt.edit(content="⏳ Loading…", view=None)

        # For the preset-shuffle case we never made `prompt`. Make a status
        # message now so the user sees progress while we fetch.
        if preset_shuffle:
            prompt = await ctx.send("🎲 Loading & shuffling…")

        try:
            if choice in ("all", "shuffle"):
                tracks = await audio.fetch_playlist(query)
            else:
                # "First only" path. We can't just call fetch_track(playlist_url)
                # because yt-dlp will sometimes pull the entire playlist's
                # metadata anyway, which hangs on long playlists. Fetch the
                # playlist's flat index (fast — one HTTP request), then resolve
                # only the first entry to a real track.
                tracks = []
                first_entries = await audio.fetch_playlist(query)
                if first_entries:
                    head = first_entries[0]
                    real = await audio.fetch_track(
                        head.get("webpage_url") or head.get("_search_query") or query
                    )
                    if real:
                        tracks = [real]
        except ValueError as e:
            await prompt.edit(content=f"❌ {audio.friendly_error_msg(e)}")
            return False
        except Exception as e:
            log.exception("fetch failed for %s", query)
            await prompt.edit(content=f"❌ {audio.friendly_error_msg(e)}")
            return False

        if not tracks:
            await prompt.edit(content="❌ No tracks found — playlist may be private or empty.")
            return False
        if await self._admit_after_lookup(ctx, epoch, f"{len(tracks)} track(s)") is None:
            await prompt.edit(content="❌ Not queued.")
            return False

        # Shuffle the new batch only — don't touch what's already queued.
        if choice == "shuffle":
            random.shuffle(tracks)

        for t in tracks:
            t["requester"] = ctx.author.display_name
            t["requester_id"] = ctx.author.id
            t["request_channel_id"] = ctx.channel.id
            state.queue.append(t)

        if choice == "shuffle":
            label = audio.playlist_source_label(query)
            await prompt.edit(content=f"🎲 Shuffled & queued **{len(tracks)} tracks** from {label}.")
        elif choice == "all":
            label = audio.playlist_source_label(query)
            await prompt.edit(content=f"📋 Added **{len(tracks)} tracks** from {label}.")
        else:
            t = tracks[0]
            await prompt.edit(
                content=f"✅ Added **{t['title']}** `[{fmt_duration(t['duration'])}]`."
            )
        return True

    # ── playnext ─────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="playnext", aliases=["pn"], description="Add a song to the top of the queue (URL, search, or file)",
                              help=(
                                  "Like `play`, but inserts the track at the front of the queue "
                                  "so it plays right after the current one.\n\n"
                                  "Also accepts attached files — same rules as `play`: attachment "
                                  "wins over text query, and multiple files give you a First/All "
                                  "prompt. With 'All', files land at the front in the order they "
                                  "were attached.\n\n"
                                  "Add `--shuffle` to randomize the new batch before it gets "
                                  "inserted at the front.\n\n"
                                  "**Example**\n`playnext lofi hip hop`\n"
                                  "`playnext` *(with a .wav attached)*"
                              ))
    async def playnext(
        self, ctx: commands.Context,
        attachment: discord.Attachment | None = None,
        *, query: str | None = None,
    ):
        if not await _ensure_voice(ctx):
            return
        _remember_text_channel(ctx)

        shuffle, query = _extract_shuffle_flag(query)
        query = unwrap_url(query) if query else query

        # Same attachment-collection pattern as `play`.
        candidates: list[discord.Attachment] = []
        if attachment is not None:
            candidates.append(attachment)
        candidates.extend(_audio_attachments(ctx.message))
        seen_ids: set[int] = set()
        unique: list[discord.Attachment] = []
        for a in candidates:
            if a.id not in seen_ids:
                seen_ids.add(a.id)
                unique.append(a)

        if not unique and not query:
            return await ctx.send(
                "❌ Provide a search query, a URL, or attach an audio/video file."
            )

        vc = await _connect_to_author(ctx)
        if vc is None:
            return

        if unique:
            if query:
                log.info("playnext: attachment(s) present, ignoring text query %r", query)
            await self._handle_attachments(ctx, vc, unique,
                                            at_front=True, preset_shuffle=shuffle)
            return

        # URL playlist + playnext: route through _handle_playlist_query so
        # shuffle and the prompt work consistently. The handler does append,
        # not insert-at-front, but playnext over a playlist URL is an edge
        # case — the user almost always means a single track here. We could
        # special-case if it matters later.
        if audio.is_any_playlist(query):
            if await self._handle_playlist_query(ctx, vc, query, preset_shuffle=shuffle):
                vc = ctx.guild.voice_client or vc
                if not vc.is_playing() and not vc.is_paused():
                    self._kickstart(ctx.guild, vc)
            return

        state = get_state(ctx.guild.id)
        epoch = state.queue_epoch
        async with ctx.typing():
            try:
                track = await audio.fetch_track(query)
            except Exception as e:
                log.exception("fetch_track failed")
                return await ctx.send(f"❌ {audio.friendly_error_msg(e)}")
            if not track:
                return await ctx.send("❌ Could not find anything for that query.")
        vc = await self._admit_after_lookup(ctx, epoch, track["title"])
        if vc is None:
            return
        track["requester"] = ctx.author.display_name
        track["requester_id"] = ctx.author.id
        track["request_channel_id"] = ctx.channel.id
        state.queue.appendleft(track)
        await ctx.send(f"⏫ **{track['title']}** will play next.")
        if not vc.is_playing() and not vc.is_paused():
            self._kickstart(ctx.guild, vc)

    # ── playback controls ────────────────────────────────────────────────────

    @commands.hybrid_command(name="skip", aliases=["s"], description="Skip the current track",
                              help="Stops the current song and moves to the next one in the queue.")
    async def skip(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        vc = ctx.voice_client
        state = get_state(ctx.guild.id)
        # Shared with voice control (core/queue_ops.py): loop mode would
        # re-queue the current song when it stops, so it's dropped first; with
        # the whole queue looping it stays in the rotation; a song still
        # starting (no sound yet) is skipped too.
        if queue_ops.skip_current(state, vc, self, ctx.guild):
            await ctx.send("⏭️ Skipped.")
        else:
            await ctx.send("❌ Nothing is playing.")

    @commands.hybrid_command(name="pause", description="Pause the current track",
                              help="Pauses playback. Use `resume` to continue from where it left off.")
    async def pause(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        vc = ctx.voice_client
        # music_is_playing (inside): paused music that -tts is talking over is
        # still paused — pausing it "again" would freeze the speech mid-word.
        if queue_ops.pause(get_state(ctx.guild.id), vc):
            await ctx.send("⏸️ Paused.")
        else:
            await ctx.send("❌ Nothing to pause.")

    @commands.hybrid_command(name="resume", aliases=["r"], description="Resume playback",
                              help="Resumes a paused track from where it left off.")
    async def resume(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        vc = ctx.voice_client
        # Also true while -tts talks over a pause; the track comes back in
        # underneath the speech.
        if queue_ops.resume(get_state(ctx.guild.id), vc):
            await ctx.send("▶️ Resumed.")
        else:
            await ctx.send("❌ Nothing is paused.")

    @commands.hybrid_command(name="stop", description="Stop playback and disconnect",
                              help="Clears the queue, stops the current track, and disconnects from voice. "
                                   "Different from `pause` — this is destructive.")
    async def stop(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        reset_state(ctx.guild.id)
        vc = ctx.voice_client
        if vc:
            vc.stop()
            await vc.disconnect()
        await ctx.send("⏹️ Stopped and disconnected.")

    @commands.hybrid_command(name="join", description="Join your voice channel",
                              help="Joins the voice channel you're currently in. You must be in one already.")
    async def join(self, ctx: commands.Context):
        if not await _ensure_voice(ctx):
            return
        _remember_text_channel(ctx)
        if await _connect_to_author(ctx) is None:
            return
        await ctx.send(f"✅ Joined **{ctx.author.voice.channel.name}**.")

    @commands.hybrid_command(name="leave", aliases=["disconnect", "dc"], description="Disconnect from voice",
                              help="Disconnects from voice without clearing the queue immediately. "
                                   "(The queue clears anyway since playback stops.)")
    async def leave(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        vc = ctx.voice_client
        if vc:
            reset_state(ctx.guild.id)
            await vc.disconnect()
            await ctx.send("👋 Disconnected.")
        else:
            await ctx.send("❌ I'm not in a voice channel.")

    @commands.hybrid_command(name="rejoin", aliases=["reconnect", "fixaudio"],
                              description="Rebuild the voice connection (fixes 'playing but silent')",
                              help=(
                                  "Tears the voice connection down and builds a fresh one, "
                                  "resuming the current track where it left off — the queue "
                                  "is kept.\n\n"
                                  "Use this when the bot *claims* to be playing but nobody "
                                  "hears anything: a voice session can wedge after Discord "
                                  "hiccups (voice-server failover, encryption re-keys, "
                                  "gateway blips), and no command short of a full "
                                  "reconnect recovers it."
                              ))
    async def rejoin(self, ctx: commands.Context):
        if not await _ensure_voice(ctx):
            return
        if not await _ensure_same_channel(ctx):
            return
        _remember_text_channel(ctx)
        state = get_state(ctx.guild.id)
        epoch = state.queue_epoch
        cur = state.current
        pos = 0.0
        if cur and (state.clock is not None or state.track_started_at):
            pos = playback_position(state, ctx.voice_client)

        # Invalidate pending after-callbacks BEFORE tearing down, so the old
        # session's stop can't advance/consume the queue behind our back.
        state.playback_gen += 1
        vc = ctx.voice_client
        if vc is not None:
            try:
                await vc.disconnect(force=True)
            except Exception:
                log.debug("rejoin: disconnect grumbled", exc_info=True)

        vc = await _connect_to_author(ctx)
        if vc is None:
            # _connect_to_author already told the user why. Keep the track:
            # it waits at the front of the queue, resuming where it was.
            if cur is not None and state.current is cur:
                cur["_status"] = "queued"
                cur["_resume_at"] = pos
                state.current = None
                state.queue.appendleft(cur)
            return

        if cur:
            try:
                await audio.resolve_stream_url(cur)
            except Exception as e:
                state.current = None
                self._record_failure(state, cur, audio.friendly_error_msg(e))
                return await ctx.send(
                    "🔄 Reconnected, but couldn't resume the track: "
                    f"{audio.friendly_error_msg(e)}")
            # The resolve awaited the network: don't resurrect a track that
            # was stopped meanwhile.
            if state.queue_epoch != epoch or state.current is not cur:
                return await ctx.send("🔄 Rebuilt the voice connection.")
            if self._begin_playback(ctx.guild, vc, state, cur, position=pos):
                await ctx.send(f"🔄 Rebuilt the voice connection — resuming "
                               f"**{cur['title']}** at `{fmt_duration(int(pos))}`.")
        else:
            await ctx.send("🔄 Rebuilt the voice connection.")
            if state.queue:
                self._kickstart(ctx.guild, vc)

    # ── seek ─────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="seek", aliases=["ff", "rw"],
                              description="Seek (1:30 / 90 / +30 / -15 / 1:23:45)",
                              help=(
                                  "Jump to a position in the current track. Many formats accepted:\n"
                                  "**Examples**\n"
                                  "`seek 1:30` — to 1 minute 30 seconds\n"
                                  "`seek 1:23:45` — to 1 hour 23 minutes 45 seconds\n"
                                  "`seek 90` — to 90 seconds\n"
                                  "`seek +30` — forward 30 seconds\n"
                                  "`seek -15` — back 15 seconds"
                              ))
    async def seek(self, ctx: commands.Context, *, position: str):
        if not await _ensure_same_channel(ctx):
            return
        vc = ctx.voice_client
        if not vc or (not vc.is_playing() and not vc.is_paused()):
            return await ctx.send("❌ Nothing is playing.")

        state = get_state(ctx.guild.id)
        cur = state.current
        if not cur:
            return await ctx.send("❌ Nothing is playing.")

        elapsed = playback_position(state, vc)
        target  = parse_seek(position, elapsed)
        if target is None:
            return await ctx.send("❌ Invalid format. Examples: `1:30` · `1:23:45` · `90` · `+30` · `-15`")

        duration = cur.get("duration") or 0
        if duration and target >= duration:
            return await ctx.send(f"❌ Past end of track ({fmt_duration(duration)}).")

        gen = state.playback_gen
        try:
            await audio.resolve_stream_url(cur)
        except Exception as e:
            log.warning("resolve failed during seek: %s", redact_secrets(str(e)))
            return await ctx.send(f"❌ Couldn't re-resolve track: {audio.friendly_error_msg(e)}")
        if state.current is not cur or state.playback_gen != gen:
            return await ctx.send("❌ The track changed while I was seeking — try again.")

        if not self._begin_playback(ctx.guild, vc, state, cur, position=target):
            return
        m, s = divmod(int(target), 60)
        h, m = divmod(m, 60)
        ts = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        await ctx.send(f"⏩ Seeked to `{ts}`.")

    # ── volume / loop / shuffle / balance ────────────────────────────────────

    @commands.hybrid_command(name="volume", aliases=["vol"],
                              description="Set volume 0-1000 (100 = normal)",
                              help=(
                                  "0 = silent, 100 = normal, 200+ = boosted (may distort).\n\n"
                                  "**Examples**\n"
                                  "`volume 50` — half volume\n"
                                  "`volume 100` — normal\n"
                                  "`volume 200` — boosted"
                              ))
    async def volume(self, ctx: commands.Context, vol: int):
        if not await _ensure_same_channel(ctx):
            return
        vc = ctx.voice_client
        if not vc or (not vc.is_playing() and not vc.is_paused()):
            return await ctx.send("❌ Nothing is playing.")
        if not 0 <= vol <= 1000:
            return await ctx.send("❌ Volume must be between 0 and 1000.")
        queue_ops.set_volume(get_state(ctx.guild.id), vc, vol)
        warning = " ⚠️ Boosted — may distort." if vol > 200 else ""
        emoji   = "🔊" if vol <= 100 else "📢"
        await ctx.send(f"{emoji} Volume set to **{vol}%**.{warning}")

    @commands.hybrid_command(name="loop", aliases=["l"], description="Loop the song or the queue",
                              help=("`loop` — toggle looping the current song\n"
                                    "`loop queue` — loop the whole queue: finished and skipped "
                                    "songs go round again, last\n"
                                    "`loop song` · `loop off`"))
    async def loop(self, ctx: commands.Context, mode: Optional[str] = None):
        if not await _ensure_same_channel(ctx):
            return
        state = get_state(ctx.guild.id)
        mode = (mode or "").strip().lower()
        if mode in ("queue", "all", "q", "playlist"):
            queue_ops.set_loop(state, "queue")
            n = len(state.queue) + (1 if state.current else 0)
            return await ctx.send(f"🔁 Looping the **queue** ({n} song{'s' if n != 1 else ''}).")
        if mode in ("song", "track", "one", "single", "on"):
            queue_ops.set_loop(state, "song")
        elif mode in ("off", "none", "stop", "disable"):
            queue_ops.set_loop(state, "off")
        elif mode:
            return await ctx.send(f"Usage: `{ctx.prefix}loop [song|queue|off]`")
        else:
            queue_ops.set_loop(state, "off" if (state.loop or state.loop_queue) else "song")
        await ctx.send("🔂 Looping this **song**." if state.loop else "➡️ Loop **disabled**.")

    @commands.hybrid_command(name="shuffle", description="Shuffle the queue",
                              help="Randomly reorders the upcoming queue. Doesn't affect the current track.")
    async def shuffle(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        queue_ops.shuffle(get_state(ctx.guild.id))
        await ctx.send("🔀 Queue shuffled.")

    @commands.hybrid_command(name="balance",
                              description="Alternate queue fairly between requesters",
                              help=(
                                  "Round-robin reorders the queue so it alternates between people who "
                                  "requested songs. If you queued 5 and your friend queued 2, the order "
                                  "becomes you-friend-you-friend-you-you-you instead of you-you-you-you-you-friend-friend."
                              ))
    async def balance(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        state = get_state(ctx.guild.id)
        if not state.queue:
            return await ctx.send("❌ Queue is empty.")
        queue_ops.balance(state)
        await ctx.send("⚖️ Queue rebalanced.")

    # ── search ───────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="search", aliases=["find"],
                              description="Search YouTube and pick from the results",
                              help=(
                                  "Searches YouTube and shows the top 15 results, 5 per page, "
                                  "with ◀ ▶ buttons to flip pages and a dropdown to pick — "
                                  "instead of blindly queueing the first match like `play` does.\n\n"
                                  "Only you (the requester) can pick. The menu expires after "
                                  "2 minutes.\n\n"
                                  "**Example**\n`search never gonna give you up`"
                              ))
    async def search(self, ctx: commands.Context, *, query: str):
        if not await _ensure_voice(ctx):
            return
        async with ctx.typing():
            try:
                results = await audio.search_tracks(query, limit=15)
            except Exception as e:
                log.exception("search failed")
                return await ctx.send(f"❌ {audio.friendly_error_msg(e)}")
        if not results:
            return await ctx.send("❌ No results for that search.")

        cog   = self
        PER   = 5
        pages = (len(results) + PER - 1) // PER

        def page_content(page: int) -> str:
            lo = page * PER
            lines = "\n".join(
                f"`{lo + j + 1}.` **{r['title']}** `[{fmt_duration(r['duration'])}]`"
                + (f" — {r['uploader']}" if r.get("uploader") else "")
                for j, r in enumerate(results[lo:lo + PER])
            )
            note = f" — page {page + 1}/{pages}" if pages > 1 else ""
            return f"🔍 Results for **{query}**{note}:\n{lines}"

        class SearchView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=120)
                self.page = 0
                self._refresh()

            def _refresh(self):
                lo = self.page * PER
                self.pick.options = [
                    discord.SelectOption(
                        label=f"{lo + j + 1}. {r['title']}"[:100],
                        description=(f"{fmt_duration(r['duration'])} · "
                                     f"{r.get('uploader') or 'YouTube'}")[:100],
                        value=str(lo + j),
                    )
                    for j, r in enumerate(results[lo:lo + PER])
                ]
                self.prev_btn.disabled = self.page == 0
                self.next_btn.disabled = self.page >= pages - 1

            async def interaction_check(self, interaction: discord.Interaction) -> bool:
                if interaction.user != ctx.author:
                    await interaction.response.send_message(
                        "❌ Only the requester can pick.", ephemeral=True,
                    )
                    return False
                return True

            async def on_timeout(self):
                try:
                    await prompt.edit(content="⌛ Search expired — nothing queued.",
                                      view=None)
                except discord.HTTPException:
                    pass

            @discord.ui.select(placeholder="Pick a track…",
                               options=[discord.SelectOption(label="…", value="0")])
            async def pick(self, interaction: discord.Interaction,
                           select: discord.ui.Select):
                self.stop()
                track = dict(results[int(select.values[0])])
                await interaction.response.edit_message(
                    content=f"⏳ Adding **{track['title']}**…", view=None,
                )
                await cog._queue_search_pick(ctx, prompt, track)

            @discord.ui.button(label="◀ Prev", style=discord.ButtonStyle.secondary, row=1)
            async def prev_btn(self, interaction: discord.Interaction, _b):
                self.page -= 1
                self._refresh()
                await interaction.response.edit_message(
                    content=page_content(self.page), view=self,
                )

            @discord.ui.button(label="Next ▶", style=discord.ButtonStyle.secondary, row=1)
            async def next_btn(self, interaction: discord.Interaction, _b):
                self.page += 1
                self._refresh()
                await interaction.response.edit_message(
                    content=page_content(self.page), view=self,
                )

            @discord.ui.button(label="Cancel", emoji="✖",
                               style=discord.ButtonStyle.secondary, row=1)
            async def cancel_btn(self, interaction: discord.Interaction, _b):
                self.stop()
                await interaction.response.edit_message(
                    content="❌ Search cancelled.", view=None,
                )

        prompt = await ctx.send(page_content(0), view=SearchView())

    async def _queue_search_pick(
        self, ctx: commands.Context, prompt: discord.Message, track: dict,
    ) -> None:
        # The author may have left voice while deciding.
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await prompt.edit(content="❌ You're not in a voice channel anymore.")
        _remember_text_channel(ctx)
        vc = await _connect_to_author(ctx)
        if vc is None:
            return await prompt.edit(content="❌ Cancelled.")
        track["requester"] = ctx.author.display_name
        track["requester_id"] = ctx.author.id
        track["request_channel_id"] = ctx.channel.id
        get_state(ctx.guild.id).queue.append(track)
        await prompt.edit(
            content=f"✅ Added **{track['title']}** "
                    f"`[{fmt_duration(track['duration'])}]` to the queue."
        )
        if not vc.is_playing() and not vc.is_paused():
            self._kickstart(ctx.guild, vc)

    # ── audio filters ────────────────────────────────────────────────────────

    @commands.hybrid_command(name="filter", aliases=["fx", "filters"],
                              description="Audio filters: bassboost, nightcore, 8d, speed …",
                              help=(
                                  "Applies an ffmpeg audio filter to playback. Takes effect "
                                  "immediately on the current track and stays on for "
                                  "everything after, until you turn it off.\n\n"
                                  "**Filters**\n"
                                  "`bassboost` `nightcore` `slowed` `8d` `tremolo` `vibrato` "
                                  "`karaoke`\n"
                                  "`speed <0.25-4>` — playback speed without pitch correction\n"
                                  "`off` — back to normal\n\n"
                                  "The `karaoke` filter is crude: it cancels whatever sits in "
                                  "the centre of the stereo image, which takes the kick, snare "
                                  "and bass out along with the vocal. Use the **`karaoke` "
                                  "command** instead — it runs a real separation model.\n\n"
                                  "**Examples**\n"
                                  "`filter bassboost`\n"
                                  "`filter speed 1.5`\n"
                                  "`filter off`"
                              ))
    async def filter_cmd(self, ctx: commands.Context, *, name: str | None = None):
        state = get_state(ctx.guild.id)
        names = ", ".join(f"`{n}`" for n in audio.AUDIO_FILTERS)
        if not name:
            return await ctx.send(
                f"🎛️ Current filter: **{state.filter_label or 'off'}**\n"
                f"Available: {names}, `speed <0.25-4>`, `off`"
            )
        if not await _ensure_same_channel(ctx):
            return

        name = name.lower().strip()
        if name in ("off", "none", "clear", "reset", "normal"):
            state.filter_af = state.filter_label = None
            label = None
        elif name.split()[0] == "speed":
            try:
                x = float(name.split()[1])
            except (IndexError, ValueError):
                return await ctx.send("❌ Usage: `filter speed 1.5` (0.25–4).")
            if not 0.25 <= x <= 4.0:
                return await ctx.send("❌ Speed must be between 0.25 and 4.")
            state.filter_af    = audio.speed_filter(x)
            state.filter_label = label = f"speed ×{x:g}"
        elif name in audio.AUDIO_FILTERS:
            state.filter_af    = audio.AUDIO_FILTERS[name]
            state.filter_label = label = name
            if name == "karaoke":
                # Nobody reaches for this because they wanted centre-channel
                # cancellation; they wanted karaoke. Point them at the thing
                # that actually does it.
                await ctx.send(
                    "-# Heads up: this filter just cancels the centre of the "
                    "stereo image, so the drums and bass go with the vocal. "
                    "The `karaoke` **command** uses a real separation model.")
        else:
            return await ctx.send(
                f"❌ Unknown filter `{name}`. "
                f"Available: {names}, `speed <0.25-4>`, `off`"
            )

        # Restart the current track in place so the filter kicks in now.
        vc = ctx.voice_client
        applied = "Will apply to the next track."
        if vc and (vc.is_playing() or vc.is_paused()) and state.current:
            elapsed = max(0.0, playback_position(state, vc))
            cur, gen = state.current, state.playback_gen
            try:
                # Refresh the stream URL if it's gone stale (cheap while fresh
                # thanks to the TTL check) — restarting ffmpeg against an
                # expired URL used to kill the track instead of filtering it.
                await audio.resolve_stream_url(cur)
            except Exception as e:
                log.warning("resolve failed during filter change: %s", redact_secrets(str(e)))
                return await ctx.send(
                    f"❌ Couldn't refresh the stream: {audio.friendly_error_msg(e)}")
            if state.current is cur and state.playback_gen == gen:
                self._begin_playback(ctx.guild, vc, state, cur, position=elapsed)
                applied = "Applied to the current track."
        if label is None:
            await ctx.send(f"🎛️ Filters **off**. {applied}")
        else:
            await ctx.send(f"🎛️ Filter set to **{label}**. {applied}")

    # ── autoplay / sponsorblock toggles ──────────────────────────────────────

    @commands.hybrid_command(name="autoplay",
                              description="Keep playing related tracks when the queue runs out (on/off)",
                              help=(
                                  "Radio mode. When the queue runs out, the bot picks a track "
                                  "related to the last one (via YouTube's mix radio) and keeps "
                                  "going — skipping anything it recently played.\n\n"
                                  "Off by default; per-server setting, persists across restarts.\n\n"
                                  "**Examples**\n`autoplay on` · `autoplay off` · `autoplay` *(show)*"
                              ))
    async def autoplay(self, ctx: commands.Context, value: str | None = None):
        if value is None:
            gs = await db.get_guild_settings(ctx.guild.id)
            on = gs["autoplay"] == 1
            return await ctx.send(
                f"📻 Autoplay is **{'on' if on else 'off'}**. "
                f"`{ctx.prefix}autoplay on|off` to change."
            )
        v = value.strip().lower()
        if   v in ("on", "yes", "true", "1", "enable", "enabled"):    enabled = True
        elif v in ("off", "no", "false", "0", "disable", "disabled"): enabled = False
        else:
            return await ctx.send("❌ Use `on` or `off`.")
        await db.set_autoplay(ctx.guild.id, enabled)
        if enabled:
            await ctx.send("📻 Autoplay **on** — when the queue runs out I'll "
                           "keep going with related tracks.")
        else:
            await ctx.send("📻 Autoplay **off**.")

    @commands.hybrid_command(name="sponsorblock", aliases=["sb"],
                              description="Auto-skip sponsor segments in YouTube videos (on/off)",
                              help=(
                                  "Uses the crowd-sourced [SponsorBlock](https://sponsor.ajay.app) "
                                  "database to hop over sponsor reads, self-promo, and "
                                  "like/subscribe reminders in YouTube tracks.\n\n"
                                  "On by default; per-server setting. There's a small audio "
                                  "hiccup at each skip (ffmpeg restarts at the new position).\n\n"
                                  "**Examples**\n`sponsorblock off` · `sponsorblock on` · "
                                  "`sponsorblock` *(show)*"
                              ))
    async def sponsorblock(self, ctx: commands.Context, value: str | None = None):
        if value is None:
            gs = await db.get_guild_settings(ctx.guild.id)
            on = gs["sponsorblock"] != 0
            return await ctx.send(
                f"⏭️ SponsorBlock is **{'on' if on else 'off'}**. "
                f"`{ctx.prefix}sponsorblock on|off` to change."
            )
        v = value.strip().lower()
        if   v in ("on", "yes", "true", "1", "enable", "enabled"):    enabled = True
        elif v in ("off", "no", "false", "0", "disable", "disabled"): enabled = False
        else:
            return await ctx.send("❌ Use `on` or `off`.")
        await db.set_sponsorblock(ctx.guild.id, enabled)
        await ctx.send(f"⏭️ SponsorBlock is now **{'on' if enabled else 'off'}** "
                       "(applies from the next track).")

    # ── history / back ───────────────────────────────────────────────────────

    @commands.hybrid_command(name="history", aliases=["played"],
                              description="Show recently played tracks",
                              help="The last 10 tracks this server played, newest first. "
                                   "Use `back` to replay the most recent one.")
    async def history(self, ctx: commands.Context):
        state   = get_state(ctx.guild.id)
        cur_url = state.current.get("webpage_url") if state.current else None
        entries = [t for t in state.history if t.get("webpage_url") != cur_url]
        if not entries:
            return await ctx.send("📜 No playback history yet.")
        recent = list(reversed(entries))[:10]
        lines  = "\n".join(
            f"`{i}.` {t['title']} `[{fmt_duration(t.get('duration', 0))}]` "
            f"— *{t.get('requester') or '—'}*"
            for i, t in enumerate(recent, 1)
        )
        embed = discord.Embed(
            title="📜 Recently played", description=lines,
            color=discord.Color.blurple(),
        )
        embed.set_footer(text=f"{ctx.prefix}back replays #1")
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="back", aliases=["previous", "prev"],
                              description="Replay the previous track",
                              help=(
                                  "Plays the most recent track from history. If something is "
                                  "playing now, it gets skipped (it stays in history, so `back` "
                                  "again bounces between the two)."
                              ))
    async def back(self, ctx: commands.Context):
        if not await _ensure_voice(ctx):
            return
        if not await _ensure_same_channel(ctx):
            return
        _remember_text_channel(ctx)
        state   = get_state(ctx.guild.id)
        cur_url = state.current.get("webpage_url") if state.current else None
        prev = next(
            (t for t in reversed(state.history) if t.get("webpage_url") != cur_url),
            None,
        )
        if prev is None:
            return await ctx.send("❌ No previous track in history.")
        replay = dict(prev)
        replay["request_channel_id"] = ctx.channel.id
        state.queue.appendleft(replay)
        vc = ctx.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            state.current = None   # explicit jump — don't let loop re-queue
            vc.stop()
        else:
            vc = await _connect_to_author(ctx)
            if vc is None:
                return
            self._abandon_starting(state)
            self._kickstart(ctx.guild, vc)
        await ctx.send(f"⏮️ Replaying **{prev['title']}**.")

    # ── queue manipulation ───────────────────────────────────────────────────

    @commands.hybrid_command(name="move",
                              description="Move a queued track to another position",
                              help="Positions come from `queue`.\n\n**Example**\n"
                                   "`move 5 1` — move track #5 to the front")
    async def move(self, ctx: commands.Context, source: int, destination: int):
        if not await _ensure_same_channel(ctx):
            return
        state = get_state(ctx.guild.id)
        items = list(state.queue)
        n = len(items)
        if n < 2:
            return await ctx.send("❌ Need at least 2 queued tracks to move things.")
        if not (1 <= source <= n and 1 <= destination <= n):
            return await ctx.send(f"❌ Positions must be between 1 and {n}.")
        if source == destination:
            return await ctx.send("❌ That's already its position.")
        (entry_id,) = queue_ops.ids_at(state, [source])
        t, _from, _to = queue_ops.move_id(state, entry_id, destination - 1)
        await ctx.send(f"↕️ Moved **{t['title']}** to position {destination}.")

    @commands.hybrid_command(name="swap",
                              description="Swap two queued tracks",
                              help="**Example**\n`swap 1 4` — exchange tracks #1 and #4")
    async def swap(self, ctx: commands.Context, first: int, second: int):
        if not await _ensure_same_channel(ctx):
            return
        state = get_state(ctx.guild.id)
        items = list(state.queue)
        n = len(items)
        if not (1 <= first <= n and 1 <= second <= n):
            return await ctx.send(f"❌ Positions must be between 1 and {n}." if n
                                  else "❌ Queue is empty.")
        if first == second:
            return await ctx.send("❌ Those are the same position.")
        a, b = queue_ops.swap_ids(state, *queue_ops.ids_at(state, [first, second]))
        await ctx.send(f"🔄 Swapped **{a['title']}** and **{b['title']}**.")

    @commands.hybrid_command(name="skipto", aliases=["jump"],
                              description="Skip ahead to a specific queued track",
                              help="Drops everything before that track and plays it now.\n\n"
                                   "**Example**\n`skipto 4` — play track #4, dropping #1–3")
    async def skipto(self, ctx: commands.Context, position: int):
        if not await _ensure_voice(ctx):
            return
        if not await _ensure_same_channel(ctx):
            return
        _remember_text_channel(ctx)
        state = get_state(ctx.guild.id)
        n = len(state.queue)
        if n == 0:
            return await ctx.send("❌ Queue is empty.")
        if not 1 <= position <= n:
            return await ctx.send(f"❌ Position must be between 1 and {n}.")
        (entry_id,) = queue_ops.ids_at(state, [position])
        queue_ops.drop_before(state, entry_id)
        target = state.queue[0]
        vc = ctx.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            state.current = None
            vc.stop()
        else:
            vc = await _connect_to_author(ctx)
            if vc is None:
                return
            self._abandon_starting(state)
            self._kickstart(ctx.guild, vc)
        dropped = f" (dropped {position - 1} before it)" if position > 1 else ""
        await ctx.send(f"⏭️ Jumping to **{target['title']}**{dropped}.")

    @commands.hybrid_command(name="reverse", description="Reverse the queue order",
                              help="Last queued plays first. Doesn't affect the current track.")
    async def reverse(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        state = get_state(ctx.guild.id)
        if len(state.queue) < 2:
            return await ctx.send("❌ Need at least 2 queued tracks.")
        queue_ops.reverse(state)
        await ctx.send("🔃 Queue reversed.")

    @commands.hybrid_command(name="dedupe", aliases=["unique"],
                              description="Remove duplicate tracks from the queue",
                              help="Keeps the first copy of each track (including the one "
                                   "currently playing) and removes the rest.")
    async def dedupe(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        state = get_state(ctx.guild.id)
        if not state.queue:
            return await ctx.send("❌ Queue is empty.")
        removed = len(queue_ops.dedupe(state))
        if removed:
            await ctx.send(f"🧹 Removed **{removed}** duplicate(s); {len(state.queue)} track(s) left.")
        else:
            await ctx.send("✨ No duplicates found.")

    # ── queue / clear / remove / nowplaying ──────────────────────────────────

    @commands.hybrid_command(name="clear", description="Clear the queue",
                              help="Removes every upcoming track. Doesn't stop the current one — use `stop` for that.")
    async def clear(self, ctx: commands.Context):
        if not await _ensure_same_channel(ctx):
            return
        clear_queue(ctx.guild.id)
        voice = self.bot.get_cog("Voice DJ")
        if voice is not None:
            voice._cancel_plays(ctx.guild.id, reason="clear")
        await ctx.send("🧹 Queue cleared.")

    @commands.hybrid_command(name="remove",
                              description="Remove tracks (e.g. 3 / 1,3,5 / 2-6)",
                              help=(
                                  "Removes specific tracks from the queue. Get the numbers from `queue`.\n\n"
                                  "**Examples**\n"
                                  "`remove 3` — just track #3\n"
                                  "`remove 1,3,5` — tracks 1, 3, and 5\n"
                                  "`remove 2-6` — tracks 2 through 6\n"
                                  "`remove 1,3,5-8` — mixed"
                              ))
    async def remove(self, ctx: commands.Context, *, args: str):
        if not await _ensure_same_channel(ctx):
            return
        state = get_state(ctx.guild.id)
        items = list(state.queue)
        if not items:
            return await ctx.send("❌ Queue is empty.")
        idxs = parse_indices(args)
        if not idxs:
            return await ctx.send("❌ Invalid. Examples: `3` · `1,3,5` · `2-6`.")
        bad = [i for i in idxs if not 1 <= i <= len(items)]
        if bad:
            return await ctx.send(f"❌ Out of range: {sorted(bad)}. Queue has {len(items)}.")
        removed = [t["title"] for t in queue_ops.remove_ids(
            state, queue_ops.ids_at(state, sorted(idxs)))]
        if len(removed) == 1:
            await ctx.send(f"🗑️ Removed **{removed[0]}**.")
        else:
            body = "\n".join(f"  • {t}" for t in removed)
            await ctx.send(f"🗑️ Removed **{len(removed)} tracks**:\n{body}")

    @commands.hybrid_command(name="queue", aliases=["q"], description="Show the queue",
                              help="Shows the current track and the next 10 in line.")
    async def queue(self, ctx: commands.Context):
        state = get_state(ctx.guild.id)
        items = list(state.queue)
        failure = _recent_failure(state)
        if not state.current and not items:
            # Right after a request, "empty" alone reads as "your song
            # vanished" — say what happened to it.
            return await ctx.send("Queue is empty." + (f"\n{failure}" if failure else ""))
        embed = discord.Embed(title="📋 Queue", color=discord.Color.blurple())
        if state.current:
            t = state.current
            embed.add_field(
                name=_now_label(t),
                value=f"**{t['title']}** `[{fmt_duration(t.get('duration', 0))}]` "
                      f"— *{t.get('requester', '—')}*",
                inline=False,
            )
        if items:
            preview = "\n".join(
                f"`{i}.` {t['title']} `[{fmt_duration(t.get('duration', 0))}]` "
                f"— *{t.get('requester', '—')}*"
                for i, t in enumerate(items[:10], 1)
            )
            extra = f"\n…and {len(items) - 10} more" if len(items) > 10 else ""
            embed.add_field(name=f"Up Next ({len(items)})", value=preview + extra, inline=False)
        if failure:
            embed.set_footer(text=failure.replace("**", "")[:200])
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="nowplaying", aliases=["np"],
                              description="Show the current track",
                              help="Detailed view of what's playing right now — title, progress bar, "
                                   "requester, volume, loop status.")
    async def nowplaying(self, ctx: commands.Context):
        state = get_state(ctx.guild.id)
        if not state.current:
            return await ctx.send("❌ Nothing is playing.")
        await ctx.send(embed=_build_np_embed(state, ctx.voice_client))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MusicCog(bot))
