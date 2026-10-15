"""
Text-to-speech in voice chat — for people without a microphone.

`-tts brb, getting food` and the bot says it in your voice channel, in a
realistic neural voice (core/tts.py). Everyone can pick their own voice with
`-ttsvoice`, so listeners can tell people apart, and the speaker's name is
announced whenever it changes ("Alex says: …").

Speech never fights the music (core/voice_mix.py): over a playing track the
music ducks under the voice and swells back after; over paused music the bot
talks and the track stays exactly where it was; with nothing playing it just
speaks. Lines queue per server and play in order, each synthesised as soon as
it's sent so there's no dead air between them.
"""
from __future__ import annotations

import array
import asyncio
import difflib
import functools
import math
import time
import wave
from collections import deque
from dataclasses import dataclass, field

import discord
from discord import app_commands
from discord.ext import commands

from core import db, tts
from core.config import settings
from core.logging_setup import get_logger
from core.state import get_state
from core.voice_mix import CueTiming, DuckingSource, EffectClip, SpeechClip, SpeechSource

log = get_logger(__name__)

_MAX_PENDING          = 6       # per server — past this, someone's spamming
_MAX_PENDING_PER_USER = 3
_NAME_GAP             = 120.0   # re-announce the speaker after a lull this long
_POLL                 = 0.05    # how often a playing line is checked on
_FX_POLL              = 0.02    # …and a sound effect (one audio frame)
_RESET_WORDS          = {"reset", "default", "clear", "off"}


@dataclass(eq=False)
class _Line:
    author_id: int
    job: asyncio.Task                  # → decoded PCM
    message: discord.Message | None    # None for slash invocations
    channel: discord.abc.Messageable
    #: False for a sound effect (the wake-word cue): it plays like a line but
    #: is not "the bot talking", so voice control must not treat it as speech
    #: to ignore (a 0.6 s blip would otherwise swallow the command said next).
    speech: bool = True


@dataclass(eq=False)
class _GuildSpeech:
    pending: deque = field(default_factory=deque)
    worker: asyncio.Task | None = None
    rendering: _Line | None = None     # taken off the queue, audio not ready yet
    clip: SpeechClip | None = None     # being spoken right now
    last_author: int | None = None
    last_at: float = 0.0
    # When the bot was audibly talking (monotonic start, end) — voice control
    # ignores no-wake-word replies captured during these (speaker echo).
    spoke: deque = field(default_factory=lambda: deque(maxlen=16))


def _chime_pcm(kind: str) -> bytes:
    """A short, soft confirmation sound as 48 kHz stereo s16le."""
    notes = {"ok": [(880, 0.07), (1320, 0.11)],
             "listening": [(990, 0.09)],
             "cancel": [(660, 0.08), (440, 0.12)]}.get(kind, [(880, 0.1)])
    out = array.array("h")
    for freq, dur in notes:
        n = int(48_000 * dur)
        fade = int(48_000 * 0.012)
        for i in range(n):
            env = min(1.0, i / fade, (n - i) / fade)
            v = int(0.22 * 32767 * env * math.sin(2 * math.pi * freq * i / 48_000))
            out.append(v)
            out.append(v)
    out.extend([0] * (2 * int(48_000 * 0.05)))
    return out.tobytes()


#: path+volume → PCM. A sound effect is read once, not per play.
_cue_cache: dict[str, bytes] = {}


def _to_mixer_pcm(raw: bytes, channels: int, rate: int, volume: float,
                  path: str) -> bytes:
    """48 kHz stereo s16le, converting only when the file isn't already that.
    The bundled cue is converted once at build time, so the usual path is a
    straight read; anything else needs audioop (still bundled, and discord.py
    imports it anyway — it is what its deprecation warning is about)."""
    if channels == 2 and rate == 48_000 and volume == 1.0:
        return raw
    try:
        import audioop
    except ImportError:                       # Python 3.13+ without the backport
        log.warning("Sound %s isn't 48 kHz/16-bit/stereo and audioop is gone — "
                    "convert it once: ffmpeg -i in.wav -ar 48000 -ac 2 "
                    "-sample_fmt s16 out.wav", path)
        return b""
    if channels == 1:
        raw = audioop.tostereo(raw, 2, 1.0, 1.0)
    if rate != 48_000:
        raw, _ = audioop.ratecv(raw, 2, 2, rate, 48_000, None)
    if volume != 1.0:
        raw = audioop.mul(raw, 2, volume)
    return raw


def _load_cue_pcm(path: str, volume: float = 1.0) -> bytes:
    """A WAV sound effect as the mixer's format, cached. b"" if unplayable."""
    key = f"{path}|{volume:.2f}"
    cached = _cue_cache.get(key)
    if cached is not None:
        return cached
    pcm = b""
    try:
        with wave.open(path, "rb") as w:
            width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
            raw = w.readframes(w.getnframes())
        if width != 2 or channels not in (1, 2) or not raw:
            log.warning("Sound %s isn't a playable 16-bit mono/stereo WAV "
                        "(%d-bit, %d channel(s))", path, width * 8, channels)
        else:
            pcm = _to_mixer_pcm(raw, channels, rate, volume, path)
    except Exception as e:
        log.warning("Sound %s couldn't be read: %s", path, e)
    _cue_cache[key] = pcm
    return pcm


async def _connect_to_author(ctx: commands.Context):
    """Join/move to the caller's channel via the music cog's connector, so the
    session gets the receive-capable client that -clip and voice control need
    (and is refused if the bot is busy playing music elsewhere). Imported
    lazily so `reload music` is honoured."""
    from cogs.music import _connect_to_author as _music_connect
    return await _music_connect(ctx)


class TTSCog(commands.Cog, name="TTS"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._guilds: dict[int, _GuildSpeech] = {}
        self._fx_tasks: set[asyncio.Task] = set()

    async def cog_check(self, ctx: commands.Context) -> bool:
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    async def cog_load(self) -> None:
        # The wake cue's sound, decoded now: the first cue after a restart
        # shouldn't read the disk on the way out.
        if settings.voice_cue and settings.voice_cue_sound and settings.voice_cue_file:
            try:
                await asyncio.to_thread(_load_cue_pcm, settings.voice_cue_file,
                                        settings.voice_cue_volume)
            except Exception:
                log.debug("cue sound preload failed", exc_info=True)

    async def cog_unload(self) -> None:
        for gs in self._guilds.values():
            self._halt(gs)
            if gs.worker is not None:
                gs.worker.cancel()

    # ── API for other cogs (voice-control feedback) ──────────────────────────

    def _bot_in_voice(self, guild: discord.Guild) -> bool:
        vc = guild.voice_client
        return vc is not None and vc.is_connected()

    async def say(self, guild: discord.Guild, text: str, *, voice: str | None = None) -> bool:
        """Speak `text` in the bot's current voice channel (no name prefix).
        Returns False if TTS isn't available or the bot isn't in voice."""
        if not tts.available() or not self._bot_in_voice(guild):
            return False
        speech = tts.clean_for_speech(text, expand=False)
        if not speech:
            return False
        gs = self._guilds.setdefault(guild.id, _GuildSpeech())
        if len(gs.pending) >= _MAX_PENDING:
            return False
        job = asyncio.create_task(tts.render(speech, voice or tts.default_voice()))
        gs.pending.append(_Line(0, job, None, None))
        self._ensure_worker(guild.id, gs)
        return True

    def chime(self, guild: discord.Guild, kind: str = "ok") -> bool:
        """Play a short confirmation sound over whatever is playing."""
        if not self._bot_in_voice(guild):
            return False
        gs = self._guilds.setdefault(guild.id, _GuildSpeech())
        if len(gs.pending) >= _MAX_PENDING:
            return False
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(_chime_pcm(kind))
        gs.pending.append(_Line(0, fut, None, None))
        self._ensure_worker(guild.id, gs)
        return True

    def cue(self, guild: discord.Guild, path: str, *, volume: float = 1.0,
            timing: CueTiming | None = None) -> bool:
        """Play a sound-effect file right now — the wake-word "I'm listening"
        cue. It goes on the mixer's effect layer (core/voice_mix.py), not the
        speech queue: over music, over a line being spoken, over a pause (the
        track stays where it was), or on its own — from the next 20 ms audio
        frame, never behind a line still being rendered or spoken, and never
        refused because the speech queue is full. Not counted as the bot
        speaking, so voice control still hears the command said right after
        it. False if the bot isn't in voice or the file is unusable."""
        if not self._bot_in_voice(guild):
            return False
        pcm = _load_cue_pcm(path, volume)          # cached: no disk read after the first
        if not pcm:
            return False
        fx = EffectClip(pcm, timing=timing)
        if timing is not None:
            timing.scheduled = time.monotonic()
        mode = self._attach_effect(guild, guild.voice_client, fx)
        if mode is None:
            return False
        task = asyncio.get_running_loop().create_task(self._ride_effect(guild, fx, mode))
        self._fx_tasks.add(task)
        task.add_done_callback(self._fx_tasks.discard)
        return True

    def _attach_effect(self, guild: discord.Guild, vc, fx: EffectClip) -> str | None:
        """Lay `fx` on the player, synchronously. 'solo' (nothing was
        playing), 'overlay', or 'held' (over paused music, which this resumed
        in hold mode — see _ride_effect). None if it can't be placed."""
        if vc is None or not vc.is_connected():
            return None
        if not vc.is_playing() and not vc.is_paused():
            try:
                vc.play(SpeechSource(effect=fx),
                        after=functools.partial(self._solo_finished, guild.id))
            except Exception as e:
                log.warning("cue: couldn't start playback in guild %s: %s", guild.id, e)
                return None
            return "solo"
        src = vc.source
        if isinstance(src, (DuckingSource, SpeechSource)):
            src.attach_effect(fx)
            if vc.is_paused() and isinstance(src, DuckingSource):
                src.hold = True                    # the track stays where it is
                vc.resume()
                return "held"
            return "held" if getattr(src, "hold", False) else "overlay"
        return None

    async def _ride_effect(self, guild: discord.Guild, fx: EffectClip, mode: str) -> None:
        """Until the effect has played: if its source goes away under it (a
        skip, a song starting where it played on its own), the rest carries
        on wherever playback is now. Afterwards a pause it lifted goes back
        on — unless speech is talking over the pause, or someone resumed."""
        give_up = time.monotonic() + fx.seconds + 5.0
        held = mode == "held"
        try:
            while not fx.done and time.monotonic() < give_up:
                await asyncio.sleep(_FX_POLL)
                if not fx.done and not fx.hosted:
                    again = self._attach_effect(guild, guild.voice_client, fx)
                    if again is None:
                        fx.cancel()
                        return
                    held = held or again == "held"
        finally:
            vc = guild.voice_client
            src = getattr(vc, "source", None) if vc is not None else None
            if (held and isinstance(src, DuckingSource) and src.hold and not src.speaking
                    and vc.is_playing()):
                vc.pause()

    def speaking_windows(self, guild_id: int) -> list[tuple[float, float]]:
        """Recent intervals (monotonic) when the bot was talking; an open one
        ends "now"."""
        gs = self._guilds.get(guild_id)
        if gs is None:
            return []
        now = time.monotonic()
        return [(a, b if b is not None else now) for a, b in gs.spoke]

    # ── -tts ─────────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="tts", aliases=["say"],
        description="Say something out loud in voice chat — for when you've got no mic",
        help=(
            "The bot says your message in your voice channel, in a realistic "
            "voice. Made for anyone without a mic.\n\n"
            "Over music, the music dips while it talks and comes back after; "
            "paused music stays paused. Mentions are read as names, links as "
            "\"a YouTube link\", and chat shorthand in full (idk → I don't know). "
            "Your name is announced when the speaker changes, so people know "
            "who's talking.\n\n"
            "Pick your own voice with `ttsvoice`; cut it off with `ttsstop`.\n\n"
            "**Examples**\n"
            "`tts brb, getting food`\n"
            "`say gg everyone`"
        ))
    @app_commands.describe(text="What to say out loud")
    async def tts_cmd(self, ctx: commands.Context, *, text: str):
        if not tts.available():
            return await ctx.send(
                "❌ TTS isn't installed on the host. Owner: run `./update.sh` "
                "(or `pip install edge-tts`) and restart.")
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ Join a voice channel first — I'll say it there.")
        await ctx.defer()   # a voice connect can outlast the 3 s slash window

        guild = ctx.guild
        speech = tts.clean_for_speech(
            text,
            member_name=lambda i: m.display_name if (m := guild.get_member(i)) else None,
            role_name=lambda i: r.name if (r := guild.get_role(i)) else None,
            channel_name=lambda i: c.name if (c := guild.get_channel(i)) else None,
        )
        if not speech:
            return await ctx.send("❌ There's nothing in that I can say out loud.")
        speech, trimmed = tts.trim_for_speech(speech, settings.tts_max_chars)

        gs = self._guilds.setdefault(guild.id, _GuildSpeech())
        if len(gs.pending) >= _MAX_PENDING:
            return await ctx.send("⏳ I'm a few messages behind — give me a second.")
        if sum(ln.author_id == ctx.author.id for ln in gs.pending) >= _MAX_PENDING_PER_USER:
            return await ctx.send("⏳ Still saying your last few — hang on.")

        # Say who's talking whenever the speaker changes, or after a lull.
        now = time.monotonic()
        if gs.last_author != ctx.author.id or now - gs.last_at > _NAME_GAP:
            name = tts.clean_for_speech(ctx.author.display_name, expand=False) or "Someone"
            speech = f"{name} says: {speech}"
        voice = await db.get_user_tts_voice(ctx.author.id) or tts.default_voice()

        # Synthesise while connecting: the first line of a session is the one
        # people sit waiting on.
        job = asyncio.create_task(tts.render(speech, voice))
        vc = await _connect_to_author(ctx)
        if vc is None:
            job.cancel()
            return                      # the connector already said why
        gs.last_author, gs.last_at = ctx.author.id, now
        gs.pending.append(_Line(ctx.author.id, job,
                                None if ctx.interaction else ctx.message, ctx.channel))
        self._ensure_worker(guild.id, gs)
        await self._acknowledge(ctx, text, trimmed)

    async def _acknowledge(self, ctx: commands.Context, text: str, trimmed: bool) -> None:
        """Quiet confirmation. A prefix command gets a reaction — anyone
        chatting via TTS sends a lot of these, and a reply each time would bury
        the channel. A slash command must answer, so it echoes the text, which
        leaves the same record in chat that a prefix message does."""
        if ctx.interaction is not None:
            shown = text if len(text) <= 1800 else text[:1800] + "…"
            note = " *(trimmed — too long to read in full)*" if trimmed else ""
            await ctx.send(f"🗣️ {shown}{note}",
                           allowed_mentions=discord.AllowedMentions.none())
            return
        for emoji in ("🗣️", "✂️") if trimmed else ("🗣️",):
            try:
                await ctx.message.add_reaction(emoji)
            except discord.HTTPException:
                break

    # ── -ttsstop ─────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="ttsstop", aliases=["ttsskip", "shush"],
        description="Stop the bot talking (and drop any queued TTS)",
        help="Cuts off whatever `tts` is saying and clears the lines waiting "
             "to be said. Music keeps playing.")
    async def ttsstop(self, ctx: commands.Context):
        vc = ctx.voice_client
        if vc is not None and (not ctx.author.voice or ctx.author.voice.channel != vc.channel):
            return await ctx.send(f"❌ You need to be in **{vc.channel.name}** to do that.")
        gs = self._guilds.get(ctx.guild.id)
        spoken, queued = self._halt(gs) if gs else (0, 0)
        if not spoken and not queued:
            return await ctx.send("❌ I'm not saying anything.")
        extra = f" — and dropped {queued} queued line{'s' if queued != 1 else ''}" if queued else ""
        await ctx.send(f"🤐 Stopped{extra}.")

    # ── -ttsvoice ────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="ttsvoice", aliases=["ttsvoices"],
        description="Pick the voice tts uses for you (no argument: list them)",
        help=(
            "Everyone gets their own TTS voice, so people can tell who's "
            "talking. Your pick follows you to every server.\n\n"
            "`ttsvoice` — list the voices\n"
            "`ttsvoice ava` — switch to Ava\n"
            "`ttsvoice fr-FR-DeniseNeural` — any Microsoft neural voice id "
            "works too, other languages included\n"
            "`ttsvoice reset` — back to the default"
        ))
    @app_commands.describe(voice="A name from the list (e.g. ava), a full voice id, or 'reset'")
    async def ttsvoice(self, ctx: commands.Context, *, voice: str | None = None):
        await ctx.defer()   # the live voice list is a network call
        mine = await db.get_user_tts_voice(ctx.author.id) or tts.default_voice()
        if not voice:
            return await ctx.send(embed=await self._voices_embed(ctx, mine))

        if voice.strip().lower() in _RESET_WORDS:
            await db.set_user_tts_voice(ctx.author.id, None)
            return await ctx.send("🗣️ Back to the default voice — "
                                  f"**{tts.voice_label(tts.default_voice())}**.")

        catalog = await tts.fetch_voice_catalog()
        chosen = tts.resolve_voice(voice, catalog)
        if chosen is None:
            shown = voice.replace("`", "")[:60]
            close = difflib.get_close_matches(voice.strip().lower(), list(tts.VOICES),
                                              n=3, cutoff=0.6)
            hint = (" Did you mean " + ", ".join(f"`{c}`" for c in close) + "?") if close else ""
            return await ctx.send(f"❌ I don't know a voice called `{shown}`.{hint} "
                                  f"`{ctx.clean_prefix}ttsvoice` lists them.")
        await db.set_user_tts_voice(ctx.author.id, chosen)
        await ctx.send(f"🗣️ Your TTS voice is now **{tts.voice_label(chosen, catalog)}**.")

    @ttsvoice.autocomplete("voice")
    async def _voice_autocomplete(self, interaction: discord.Interaction, current: str):
        cur = current.strip().lower()
        choices = [app_commands.Choice(name=f"{nick} — {blurb}", value=nick)
                   for nick, (vid, blurb) in tts.VOICES.items()
                   if not cur or cur in nick or cur in vid.lower()]
        if not cur or "reset".startswith(cur):
            choices.append(app_commands.Choice(name="reset — back to the default", value="reset"))
        return choices[:25]

    async def _voices_embed(self, ctx: commands.Context, mine: str) -> discord.Embed:
        catalog = await tts.fetch_voice_catalog()
        lines = [f"`{nick}` — {blurb}" + ("  ◀ yours" if vid == mine else "")
                 for nick, vid, blurb in tts.curated_voices(catalog)]
        p = ctx.clean_prefix
        embed = discord.Embed(
            title="🗣️ TTS voices",
            description=(
                f"Yours: **{tts.voice_label(mine, catalog)}**\n"
                f"Switch with `{p}ttsvoice <name>` — everyone can have their own, "
                "so people can tell who's talking.\n\n" + "\n".join(lines)
            ),
            color=discord.Color.blurple(),
        )
        embed.set_footer(text="Any Microsoft neural voice id works too, e.g. "
                              f"fr-FR-DeniseNeural · {p}ttsvoice reset for the default")
        return embed

    # ── the speaking loop ────────────────────────────────────────────────────

    def _ensure_worker(self, guild_id: int, gs: _GuildSpeech) -> None:
        if gs.worker is None or gs.worker.done():
            gs.worker = asyncio.create_task(self._speak_queue(guild_id, gs),
                                            name=f"tts:{guild_id}")

    async def _speak_queue(self, guild_id: int, gs: _GuildSpeech) -> None:
        while gs.pending:
            line = gs.pending.popleft()
            gs.rendering = line
            try:
                await asyncio.wait({line.job})   # never raises for the job's own errors
            finally:
                gs.rendering = None
            if line.job.cancelled():
                continue
            exc = line.job.exception()
            if exc is not None:
                await self._report_failure(line, exc)
                continue

            guild = self.bot.get_guild(guild_id)
            vc = guild.voice_client if guild else None
            if vc is None or not vc.is_connected():
                self._halt(gs)      # we left voice; nothing queued should play later
                return
            clip = SpeechClip(line.job.result())
            gs.clip = clip
            window = [time.monotonic(), None]
            if line.speech:
                gs.spoke.append(window)
            try:
                await self._deliver(guild, clip)
            except Exception:
                log.exception("TTS delivery failed in guild %s", guild_id)
                clip.cancel()
            finally:
                gs.clip = None
                window[1] = time.monotonic()

    def _halt(self, gs: _GuildSpeech) -> tuple[int, int]:
        """Cut off the current line and drop everything queued.
        Returns (lines cut off, queued lines dropped)."""
        queued = len(gs.pending)
        for line in gs.pending:
            line.job.cancel()
        gs.pending.clear()
        spoken = 0
        if gs.rendering is not None:
            gs.rendering.job.cancel()
            spoken = 1
        if gs.clip is not None and not gs.clip.done:
            gs.clip.cancel()
            spoken = 1
        return spoken, queued

    async def _report_failure(self, line: _Line, exc: BaseException) -> None:
        if isinstance(exc, tts.TTSError):
            msg = f"❌ Couldn't say that — {exc}"
        else:
            log.error("TTS render failed", exc_info=exc)
            msg = "❌ Couldn't say that — something went wrong (details are in my logs)."
        try:
            if line.message is not None:
                await line.message.reply(msg, mention_author=False)
            elif line.channel is not None:
                await line.channel.send(msg)
            else:
                log.warning("TTS line (no channel) failed: %s", msg)
        except discord.HTTPException:
            pass

    # ── getting a line heard, whatever the player is doing ───────────────────

    async def _deliver(self, guild: discord.Guild, clip: SpeechClip) -> None:
        """Play `clip` to the end. If its host goes away mid-sentence (skip,
        seek, track change), the rest is re-attached to whatever is playing
        now, so the sentence carries on rather than dying."""
        give_up = time.monotonic() + clip.seconds + 45
        while not clip.done:
            vc = guild.voice_client
            if vc is None or not vc.is_connected() or time.monotonic() > give_up:
                clip.cancel()
                return
            mode = self._attach(guild, vc, clip)
            if mode is None:
                # The previous line's player is still winding down, or something
                # we can't mix into is playing — check again shortly.
                await asyncio.sleep(_POLL)
                continue
            await self._ride(vc, clip, mode)

    def _attach(self, guild: discord.Guild, vc, clip: SpeechClip) -> str | None:
        """Put `clip` on the player. Deliberately synchronous: no other code on
        the event loop can touch the player between these checks and the
        attach. Returns how it's playing — 'solo', 'overlay' or 'held' — or
        None if it couldn't be placed yet."""
        if not vc.is_playing() and not vc.is_paused():
            try:
                vc.play(SpeechSource(clip),
                        after=functools.partial(self._solo_finished, guild.id))
            except Exception as e:
                log.warning("TTS: couldn't start playback in guild %s: %s", guild.id, e)
                clip.cancel()
                return None
            return "solo"
        src = vc.source
        if isinstance(src, SpeechSource) and src.clip is None:
            src.attach(clip)                 # a cue playing on its own: talk alongside it
            return "solo"
        if isinstance(src, DuckingSource):
            src.attach(clip)
            if vc.is_paused():
                # Talk over the pause: the source stops pulling music, so the
                # track and its playback clock stay exactly where they were.
                src.hold = True
                vc.resume()
                return "held"
            return "held" if src.hold else "overlay"
        return None

    async def _ride(self, vc, clip: SpeechClip, mode: str) -> None:
        """Wait while `clip` plays. Returns when it finishes, when its host
        lets go of it (skip/seek/stop), or when the music is paused under it —
        the caller then re-attaches it as 'held' and it keeps talking."""
        src = vc.source if mode == "held" else None
        try:
            while not clip.done and clip.hosted:
                if mode != "solo" and vc.is_paused():
                    return
                await asyncio.sleep(_POLL)
        except asyncio.CancelledError:
            clip.cancel()
            if mode == "held":
                self._end_hold(vc, src)
            raise
        if mode == "held":
            self._end_hold(vc, src)

    @staticmethod
    def _end_hold(vc, src) -> None:
        """Done talking over a pause: pause the player again, unless someone
        resumed the music meanwhile. `hold` stays set while paused (resume_music
        lifts it), so a read already in flight can't leak a frame of music."""
        if (isinstance(src, DuckingSource) and src.hold
                and vc.source is src and vc.is_playing()):
            vc.pause()

    def _solo_finished(self, guild_id: int, err: Exception | None) -> None:
        # Audio thread — hop back onto the event loop.
        if err:
            log.warning("TTS playback error: %s", err)
        try:
            self.bot.loop.call_soon_threadsafe(self._start_waiting_music, guild_id)
        except RuntimeError:
            pass   # loop closed — shutting down

    def _start_waiting_music(self, guild_id: int) -> None:
        """A spoken line just ended with nothing else playing. `play` doesn't
        start the queue while the bot is talking, so if songs were queued
        meanwhile, start them now."""
        guild = self.bot.get_guild(guild_id)
        vc = guild.voice_client if guild else None
        if vc is None or not vc.is_connected() or vc.is_playing() or vc.is_paused():
            return
        state = get_state(guild_id)
        if not state.queue and state.current is None:
            return
        music = self.bot.get_cog("Music")
        if music is not None:
            music._kickstart(guild, vc)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(TTSCog(bot))
