"""
Voice-clip cog — record voice chat, clip the last few minutes on demand.

While the bot is in a voice channel it keeps a *rolling* buffer of everyone's
voice (raw Opus frames — only ~2-3 MB per talker per 5 min, nothing touches
disk). `clip` then renders the buffered audio and posts it:

    clip                 → last 5 min (the whole buffer), mixed to one mp3
    clip 2m              → last 2 minutes
    clip 2m 23s          → compound durations work (same parser as remindme)
    clip 90s             → seconds
    clip 3m split        → one .mkv with a separate audio track per person

Receiving needs `discord-ext-voice-recv`, and the bot must connect with
`VoiceRecvClient` (see cogs/music.py). Discord enforces DAVE end-to-end voice
encryption on all non-stage calls (since 2026-03-02), so each received frame is
E2EE — voice-recv only undoes the transport layer. We decrypt with the bot's
own DAVE session (it's a member of the call's MLS group) before buffering; see
`RollingRecorder._maybe_decrypt`.
"""
from __future__ import annotations

import asyncio
import functools
import os
import shutil
import tempfile
import threading
import time
import wave
from collections import defaultdict, deque

import discord
from discord.ext import commands, tasks
from discord.ext import voice_recv
from discord.opus import Decoder

try:
    import davey
    _MEDIA_AUDIO = davey.MediaType.audio
except Exception:  # davey is mandatory for voice on discord.py 2.7, but be safe
    davey = None
    _MEDIA_AUDIO = None

from core.config import settings
from core.formatters import fmt_duration_long, parse_duration
from core.logging_setup import get_logger

log = get_logger("recording")

# How far back the buffer reaches. Also the cap on any `clip` request.
BUFFER_SECONDS = int(os.getenv("CLIP_BUFFER_SECONDS", "300"))

# Opus/PCM frame geometry — pulled from discord's decoder so we stay in sync
# with whatever it actually emits (48 kHz, stereo, 16-bit, 20 ms frames).
FRAME_SECONDS = Decoder.FRAME_LENGTH / 1000.0          # 0.02
FRAME_BYTES   = Decoder.FRAME_SIZE                      # 3840
SAMPLE_RATE   = Decoder.SAMPLING_RATE                   # 48000
CHANNELS      = Decoder.CHANNELS                        # 2
SAMPLE_WIDTH  = Decoder.SAMPLE_SIZE // Decoder.CHANNELS # 2
SILENCE_FRAME = b"\x00" * FRAME_BYTES

SPLIT_WORDS = {"split", "tracks", "separate", "multitrack", "mkv"}


class RollingRecorder(voice_recv.AudioSink):
    """An AudioSink that keeps the last ``window`` seconds of Opus per user.

    ``write`` is called from voice-recv's receive thread, so all access to the
    buffers is guarded by a lock. We keep raw Opus (cheap) and only decode at
    clip time.
    """

    def __init__(self, window: int = BUFFER_SECONDS, *, tap=None):
        super().__init__()
        self.window = window
        self._lock = threading.Lock()
        # user_id -> deque[(arrival_monotonic: float, opus: bytes)]
        self._buffers: dict[int, deque] = defaultdict(deque)
        self._names: dict[int, str] = {}
        # Live feed for voice control: tap(guild_id, user_id, opus, seq, ts,
        # ssrc, arrival), called on the receive thread for every decrypted
        # frame. Voice control reads frames as they arrive instead of polling
        # this clip buffer — so it never sees audio from before someone opted in.
        self.tap = tap
        # Health counters (read by `voice diag`): these used to be invisible
        # (DAVE failures were debug-only).
        self.frames = 0
        self.dave_not_ready = 0
        self.dave_failures = 0
        self.tap_errors = 0

    # voice-recv: True means we receive raw Opus frames (data.opus) rather than
    # decoded PCM — far less memory for an always-on buffer.
    def wants_opus(self) -> bool:
        return True

    def write(self, user, data) -> None:
        if user is None:
            return
        opus = data.opus
        if not opus:
            return
        # Under DAVE (E2EE, mandatory on Discord since 2026-03-02) this frame is
        # still end-to-end encrypted here — voice-recv only undid the transport
        # layer. Decrypt with the bot's DAVE session before buffering.
        opus = self._maybe_decrypt(user.id, opus)
        if not opus:
            return
        now = time.monotonic()
        self.frames += 1
        with self._lock:
            buf = self._buffers[user.id]
            buf.append((now, opus))
            self._names[user.id] = getattr(user, "display_name", str(user.id))
            cutoff = now - self.window
            while buf and buf[0][0] < cutoff:
                buf.popleft()
        tap = self.tap
        if tap is not None:
            # Keep the RTP sequence/timestamp/SSRC: voice control uses them to
            # conceal lost packets and to size pauses, instead of gluing
            # whatever arrived together.
            pkt = data.packet
            try:
                tap(self.voice_client.guild.id, user.id, opus,
                    getattr(pkt, "sequence", None), getattr(pkt, "timestamp", None),
                    getattr(pkt, "ssrc", None), now)
            except Exception:
                self.tap_errors += 1
                if self.tap_errors in (1, 10, 100) or self.tap_errors % 1000 == 0:
                    log.exception("voice tap failed (%d so far)", self.tap_errors)

    def _maybe_decrypt(self, user_id: int, opus: bytes):
        """Return the plaintext Opus frame, decrypting via DAVE when the call is
        end-to-end encrypted. Returns None to drop the frame (session not ready,
        or decrypt failed). Called on voice-recv's receive thread."""
        vc = self.voice_client
        sess = getattr(getattr(vc, "_connection", None), "dave_session", None)
        if sess is None:
            return opus            # not an E2EE session — frame is already plaintext
        if _MEDIA_AUDIO is None or not getattr(sess, "ready", False):
            self.dave_not_ready += 1
            return None            # mid-negotiation / no davey — can't decrypt, skip
        try:
            return sess.decrypt(user_id, _MEDIA_AUDIO, opus)
        except Exception:
            # Counted for `voice diag`; logged at debug only (one per frame
            # would flood), with an occasional summary line.
            self.dave_failures += 1
            if self.dave_failures in (1, 50) or self.dave_failures % 1000 == 0:
                log.warning("DAVE decrypt failures: %d (latest user %s)",
                            self.dave_failures, user_id)
            log.debug("DAVE decrypt failed (user %s)", user_id, exc_info=True)
            return None

    def prune(self) -> None:
        """Drop expired frames for everyone (called periodically so the buffer
        of someone who spoke once and went quiet doesn't linger forever)."""
        cutoff = time.monotonic() - self.window
        with self._lock:
            for uid in list(self._buffers):
                buf = self._buffers[uid]
                while buf and buf[0][0] < cutoff:
                    buf.popleft()
                if not buf:
                    del self._buffers[uid]

    def snapshot(self, seconds: float):
        """Return ``(end_t, start_t, {uid: (name, [(t, opus), ...])})`` for the
        last ``seconds`` — a copy, so rendering can run without the lock."""
        now = time.monotonic()
        start = now - seconds
        out: dict[int, tuple[str, list]] = {}
        with self._lock:
            for uid, buf in self._buffers.items():
                frames = [(t, o) for (t, o) in buf if t >= start]
                if frames:
                    out[uid] = (self._names.get(uid, str(uid)), frames)
        return now, start, out

    def frames_after(self, user_id: int, after_t: float):
        """Return ``(opus_frames, first_t, last_t)`` for this user's buffered
        frames that arrived strictly after ``after_t`` (monotonic). Used by the
        voice-command listener to pull a fresh utterance. ``([], None, None)``
        when there's nothing new."""
        with self._lock:
            buf = self._buffers.get(user_id)
            sel = [(t, o) for (t, o) in buf if t > after_t] if buf else []
        if not sel:
            return [], None, None
        return [o for _, o in sel], sel[0][0], sel[-1][0]

    def cleanup(self) -> None:
        # Called by voice-recv when listening stops (e.g. the bot disconnects).
        with self._lock:
            self._buffers.clear()
            self._names.clear()


def _decode_to_wavs(snapshot, workdir: str) -> list[tuple[int, str, str]]:
    """Decode each user's buffered Opus into a silence-aligned WAV.

    All users share one wall-clock timeline (frame arrival time), so the WAVs
    line up when mixed or layered. Gaps between talk-bursts are padded with
    silence. Returns ``[(user_id, display_name, wav_path), ...]``. CPU-bound —
    call via ``asyncio.to_thread``.
    """
    end_t, start_t, users = snapshot
    results: list[tuple[int, str, str]] = []

    for idx, (uid, (name, frames)) in enumerate(users.items()):
        decoder = Decoder()
        pcm = bytearray()
        cursor = start_t
        for arrival, opus in frames:
            gap = arrival - cursor
            if gap >= FRAME_SECONDS:
                pcm += SILENCE_FRAME * int(gap / FRAME_SECONDS)
            try:
                pcm += decoder.decode(opus, fec=False)
            except Exception:
                pcm += SILENCE_FRAME
            cursor = arrival + FRAME_SECONDS

        tail = end_t - cursor
        if tail >= FRAME_SECONDS:
            pcm += SILENCE_FRAME * int(tail / FRAME_SECONDS)

        if not pcm:
            continue

        path = os.path.join(workdir, f"track{idx}.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(pcm))
        results.append((uid, name, path))

    return results


def _safe_title(name: str) -> str:
    """Strip characters that would break ffmpeg's `key=value` metadata."""
    cleaned = name.replace("=", " ").replace("\n", " ").replace("\r", " ").strip()
    return cleaned[:64] or "unknown"


async def _encode(wavs: list[tuple[int, str, str]], split: bool, workdir: str) -> str:
    """Combine the per-user WAVs into the final file via ffmpeg.

    Mixed → one mp3 (amix). Split → one .mkv with one Opus audio track per
    person, each titled with their name.
    """
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error"]
    for _uid, _name, path in wavs:
        cmd += ["-i", path]

    if split:
        out = os.path.join(workdir, "clip.mkv")
        for i in range(len(wavs)):
            cmd += ["-map", str(i)]
        cmd += ["-c:a", "libopus", "-b:a", "96k"]
        for i, (_uid, name, _path) in enumerate(wavs):
            cmd += [f"-metadata:s:a:{i}", f"title={_safe_title(name)}"]
        cmd += [out]
    else:
        out = os.path.join(workdir, "clip.mp3")
        if len(wavs) > 1:
            cmd += ["-filter_complex",
                    f"amix=inputs={len(wavs)}:duration=longest:normalize=0"]
        cmd += ["-c:a", "libmp3lame", "-q:a", "2", out]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        msg = (stderr or b"").decode(errors="replace").strip().splitlines()
        raise RuntimeError(msg[-1][:300] if msg else "ffmpeg failed")
    return out


# Receiver recovery: retry delays (seconds) after receive stops while the bot
# is still in voice.
_RECOVERY_DELAYS = (1, 2, 5, 10, 30)


class Recording(commands.Cog):
    """Always-on voice buffer + the `clip` command."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.recorders: dict[int, RollingRecorder] = {}
        # Text channel the bot was summoned from per guild — reused by the voice
        # command listener (cogs/voicedj.py) to echo what it heard.
        self.text_channels: dict[int, discord.abc.Messageable] = {}
        self._notified: set[int] = set()
        # Live-frame subscribers (voice control) and receiver-session watchers.
        self._subscribers: list = []
        self._session_watchers: list = []
        # Per guild: which listen() session is current (so a stale "stopped"
        # callback from an old session is ignored), and its health for diag.
        self._sessions: dict[int, object] = {}
        self.health: dict[int, dict] = {}
        self._recovering: dict[int, asyncio.Task] = {}
        self._prune_loop.start()

    def cog_unload(self) -> None:
        self._prune_loop.cancel()
        for task in self._recovering.values():
            task.cancel()

    # ── live tap ─────────────────────────────────────────────────────────────

    def subscribe(self, frame_cb, session_cb=None) -> None:
        """frame_cb(guild_id, user_id, opus, seq, ts, ssrc, arrival) runs on the
        RECEIVE THREAD for every decrypted frame — it must be cheap and never
        block. session_cb(guild_id, event) runs on the event loop when a
        guild's receiver starts, stops or is recovered."""
        if frame_cb not in self._subscribers:
            self._subscribers.append(frame_cb)
        if session_cb is not None and session_cb not in self._session_watchers:
            self._session_watchers.append(session_cb)

    def unsubscribe(self, frame_cb, session_cb=None) -> None:
        if frame_cb in self._subscribers:
            self._subscribers.remove(frame_cb)
        if session_cb in self._session_watchers:
            self._session_watchers.remove(session_cb)

    def _tap(self, guild_id, user_id, opus, seq, ts, ssrc, arrival) -> None:
        for cb in tuple(self._subscribers):
            cb(guild_id, user_id, opus, seq, ts, ssrc, arrival)

    def _notify_session(self, guild_id: int, event: str) -> None:
        for cb in tuple(self._session_watchers):
            try:
                cb(guild_id, event)
            except Exception:
                log.exception("receiver session watcher failed")

    @tasks.loop(seconds=60)
    async def _prune_loop(self) -> None:
        for rec in list(self.recorders.values()):
            rec.prune()

    async def attach(self, vc, channel: discord.abc.Messageable, *,
                     reply_here: bool = True) -> None:
        """Start (or confirm) the rolling buffer on a voice client. Idempotent
        — safe to call on every join/move. Called from cogs/music.py after a
        connect, with the text channel the command came from for the notice.
        `reply_here=False` (autojoin: nobody typed anything) posts the notice
        there but leaves "where voice-control replies go" alone."""
        if not isinstance(vc, voice_recv.VoiceRecvClient):
            log.warning("Voice client in guild %s isn't a VoiceRecvClient — "
                        "clip recording unavailable. Is discord-ext-voice-recv "
                        "installed and is connect() using it?",
                        getattr(getattr(vc, "guild", None), "id", "?"))
            return
        gid = vc.guild.id
        if channel is not None and reply_here:
            # Refresh even when already listening: voice-command replies go to
            # where the bot was most recently summoned, not wherever it first was.
            self.text_channels[gid] = channel
        if vc.is_listening():
            return
        recorder = RollingRecorder(tap=self._tap)
        token = object()
        try:
            vc.listen(recorder, after=functools.partial(self._on_listen_done, gid, token))
        except Exception as e:
            log.exception("Failed to start voice receive in guild %s", gid)
            self.health[gid] = {"state": "failed", "since": time.monotonic(),
                                "error": repr(e)[:200],
                                "restarts": self.health.get(gid, {}).get("restarts", 0)}
            return
        self._sessions[gid] = token
        self.recorders[gid] = recorder
        prev = self.health.get(gid, {})
        restarted = prev.get("state") in ("stopped", "recovering", "failed")
        self.health[gid] = {"state": "listening", "since": time.monotonic(),
                            "error": None,
                            "restarts": prev.get("restarts", 0) + (1 if restarted else 0)}
        self._notify_session(gid, "recovered" if restarted else "started")
        if channel is None:
            return

        if vc.guild.id not in self._notified:
            self._notified.add(vc.guild.id)
            mins = BUFFER_SECONDS // 60
            try:
                await channel.send(
                    f"🔴 Heads up — I keep a rolling **{mins}-minute** buffer of voice "
                    f"in here so anyone can `clip` it. Nothing is written to disk until "
                    f"someone runs the command, and the buffer is wiped when I leave."
                )
            except Exception:
                pass

    def _on_listen_done(self, guild_id: int, token: object,
                        exc: Exception | None) -> None:
        # Runs on voice-recv's "audioreader-stopper" thread — hop to the loop.
        try:
            self.bot.loop.call_soon_threadsafe(self._listen_stopped, guild_id, token, exc)
        except RuntimeError:
            pass   # loop closed: shutting down

    def _listen_stopped(self, guild_id: int, token: object,
                        exc: Exception | None) -> None:
        """Receive stopped. If the bot left voice, that's expected. If it's
        still connected, receiving died under it — the "transcription just
        died" symptom — so recover with backoff instead of only logging."""
        if self._sessions.get(guild_id) is not token:
            return                                    # an older session's callback
        self._sessions.pop(guild_id, None)
        self.recorders.pop(guild_id, None)            # invalidate the dead recorder
        guild = self.bot.get_guild(guild_id)
        vc = guild.voice_client if guild else None
        restarts = self.health.get(guild_id, {}).get("restarts", 0)
        if vc is None or not vc.is_connected():
            log.info("Voice receive stopped (bot left voice) in guild %s.", guild_id)
            self.health[guild_id] = {"state": "idle", "since": time.monotonic(),
                                     "error": None, "restarts": restarts}
            self._notify_session(guild_id, "stopped")
            return
        log.warning("Voice receive stopped while still connected in guild %s "
                    "(error=%r) — recovering.", guild_id, exc)
        self.health[guild_id] = {"state": "stopped", "since": time.monotonic(),
                                 "error": repr(exc)[:200] if exc else None,
                                 "restarts": restarts}
        self._notify_session(guild_id, "stopped")
        old = self._recovering.get(guild_id)
        if old is None or old.done():
            self._recovering[guild_id] = asyncio.create_task(self._recover(guild_id))

    async def _recover(self, guild_id: int) -> None:
        for attempt, delay in enumerate(_RECOVERY_DELAYS, 1):
            await asyncio.sleep(delay)
            guild = self.bot.get_guild(guild_id)
            vc = guild.voice_client if guild else None
            if vc is None or not vc.is_connected():
                return                                # left voice meanwhile
            if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
                return                                # something else re-attached
            h = self.health.setdefault(guild_id, {})
            h.update(state="recovering", attempt=attempt)
            try:
                await self.attach(vc, None)
            except Exception as e:
                h["error"] = repr(e)[:200]
                log.warning("Voice receive recovery attempt %d failed in guild %s: %r",
                            attempt, guild_id, e)
            if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
                log.info("Voice receive recovered in guild %s (attempt %d).",
                         guild_id, attempt)
                return
        log.error("Voice receive could not be recovered in guild %s — voice "
                  "control and clip are down until the bot rejoins.", guild_id)
        self.health.setdefault(guild_id, {}).update(state="failed")

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after) -> None:
        # When the bot itself leaves voice, forget its buffer + notice state.
        if member.id != self.bot.user.id:
            return
        if after.channel is None and before.channel is not None:
            gid = before.channel.guild.id
            self.recorders.pop(gid, None)
            self.text_channels.pop(gid, None)
            self._notified.discard(gid)
            self._sessions.pop(gid, None)
            task = self._recovering.pop(gid, None)
            if task is not None:
                task.cancel()
            self.health[gid] = {"state": "idle", "since": time.monotonic(),
                                "error": None,
                                "restarts": self.health.get(gid, {}).get("restarts", 0)}
            self._notify_session(gid, "stopped")

    @commands.hybrid_command(
        name="clip",
        description="Clip the last few minutes of voice chat to an audio file",
        help=(
            "Posts the recent voice chat as audio. I have to be in the voice "
            "channel (start me with `play`, or just have me join).\n\n"
            "Give a duration to choose how far back — same formats as "
            "`remindme`. Omit it for the whole buffer "
            f"({BUFFER_SECONDS // 60} min).\n\n"
            "Add `split` to get one `.mkv` with a **separate audio track per "
            "person** (great for editing) instead of a single mixed mp3.\n\n"
            "**Examples**\n"
            "`clip`            — last 5 min, mixed\n"
            "`clip 2m`         — last 2 minutes\n"
            "`clip 2m 23s`     — compound durations\n"
            "`clip 90s split`  — last 90s, one track per person\n\n"
            "Heads up: I record everyone in the channel; people are told when "
            "I start buffering."
        ),
    )
    async def clip(self, ctx: commands.Context, *, options: str = "") -> None:
        vc = ctx.voice_client
        if not isinstance(vc, voice_recv.VoiceRecvClient) or not vc.is_listening():
            return await ctx.send(
                "❌ I'm not recording any voice chat right now. Get me into a "
                "voice channel first (e.g. `play` something)."
            )
        recorder = self.recorders.get(ctx.guild.id)
        if not isinstance(recorder, RollingRecorder):
            sink = getattr(vc, "sink", None)
            recorder = sink if isinstance(sink, RollingRecorder) else None
        if recorder is None:
            return await ctx.send("❌ Voice buffer isn't available right now.")

        # Pull the `split` keyword out of the args; the rest is the duration.
        split = False
        dur_tokens: list[str] = []
        for tok in options.split():
            if tok.lower() in SPLIT_WORDS:
                split = True
            else:
                dur_tokens.append(tok)

        if dur_tokens:
            seconds = parse_duration("".join(dur_tokens))
            if seconds is None:
                return await ctx.send(
                    "❌ Couldn't read that duration. Try `clip 5m`, `clip 2m 23s`, "
                    "or `clip 90s`. Add `split` for separate tracks."
                )
            if seconds <= 0:
                return await ctx.send("❌ Duration must be positive.")
        else:
            seconds = BUFFER_SECONDS
        seconds = min(seconds, BUFFER_SECONDS)

        snap = recorder.snapshot(seconds)
        _end, _start, users = snap
        if not users:
            return await ctx.send(
                f"❌ Nothing's been said in the last {fmt_duration_long(int(seconds))}."
            )

        people = f"{len(users)} {'person' if len(users) == 1 else 'people'}"
        status = await ctx.send(
            f"✂️ Rendering the last **{fmt_duration_long(int(seconds))}** of voice "
            f"({people})…"
        )
        workdir = tempfile.mkdtemp(prefix="clip-", dir=settings.download_tmp)
        try:
            wavs = await asyncio.to_thread(_decode_to_wavs, snap, workdir)
            if not wavs:
                return await status.edit(content="❌ No decodable audio in that window.")
            outfile = await _encode(wavs, split, workdir)
            await self._deliver(ctx, status, outfile, seconds, split, len(users))
        except Exception as e:
            log.exception("clip render failed in guild %s", ctx.guild.id)
            await status.edit(content=f"❌ Clip failed: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _deliver(self, ctx, status, outfile, seconds, split, n_users) -> None:
        """Send the clip — Discord attachment if it fits, else catbox/litterbox
        (reusing the downloads cog's upload helpers)."""
        size = os.path.getsize(outfile)
        label = fmt_duration_long(int(seconds))
        fname = "clip.mkv" if split else "clip.mp3"
        if split:
            desc = (f"🎙️ Last **{label}** of voice — {n_users} separate "
                    f"track{'s' if n_users != 1 else ''} (.mkv).")
        else:
            desc = f"🎙️ Last **{label}** of voice — mixed."

        limit = ctx.guild.filesize_limit if ctx.guild else 8 * 1024 * 1024
        if size <= max(limit - 256 * 1024, 0):
            await status.edit(content=desc)
            await ctx.send(file=discord.File(outfile, filename=fname))
            return

        # Too big for a Discord attachment — fall back to the external hosts.
        from cogs import downloads as dl
        try:
            if size <= dl.CATBOX_MAX:
                url = await dl._upload_to_catbox(outfile)
            elif size <= dl.LITTERBOX_MAX:
                url = await dl._upload_to_litterbox(outfile)
            else:
                return await status.edit(content="❌ Clip is too large to upload.")
            await status.edit(content=f"{desc}\n{url}")
        except Exception as e:
            await status.edit(content=f"❌ Upload failed: {e}")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Recording(bot))
