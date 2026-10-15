"""
Lyrics cog — karaoke-style synced display.

The timing work lives in two other modules; this file is the presentation
layer plus the human feedback loop:

  * core/playback_clock.py — where we actually are in the track.
  * core/lyrics_sync.py    — which lyrics, and how to align them.

What this file contributes to staying in sync:

  * **Scheduling instead of polling.** The old loop woke every 2.5 s and asked
    "has the line changed?", so a line could sit on screen up to 2.5 s late.
    This one computes when the next line is due and sleeps exactly that long.
  * **Lead compensation.** An edit takes time to reach the API and fan out to
    clients. We fire it `EDIT_LEAD` early so the line lands on the beat rather
    than after it.
  * **Adaptive rate limiting.** Discord will throttle a message edited too
    often. Rather than pick a pessimistic constant, we start responsive and
    back off only if the API actually pushes back.
  * **Surviving pause and seek.** Position comes from the clock, so pausing
    freezes the display and resuming picks up in the right place — the old
    loop simply exited on pause and left stale lyrics on screen forever.
"""
from __future__ import annotations

import asyncio
import bisect
import time

import aiohttp
import discord
from discord.ext import commands

from core import db
from core.logging_setup import get_logger
from core.lyrics_sync import (
    Alignment, LyricLine, TrackQuery, align, build_query, confidence_ok,
    detect_leading_silence, fetch_lyrics,
)
from core.state import get_state, playback_position

log = get_logger(__name__)

#: Fire edits this far ahead of the timestamp to cover API + gateway latency.
EDIT_LEAD = 0.45
#: Never edit more often than this, even if lines come faster.
MIN_EDIT_INTERVAL = 1.6
#: Ceiling on a single sleep, so track changes are noticed promptly.
MAX_SLEEP = 3.0
#: Lines closer together than this are shown together rather than flickering.
MERGE_GAP = 1.2
#: Lines of context either side of the current one.
WINDOW = 4
#: Nudges beyond this are almost certainly a mis-match, not a timing error.
MAX_NUDGE = 30.0


def _merge(lines: list[LyricLine]) -> list[LyricLine]:
    """Collapse near-simultaneous lines so the display doesn't strobe.

    Also drops the empty lines LRC files use as instrumental markers *as
    separate entries* while keeping their timestamps: an instrumental break
    should leave the last sung line on screen, not blank the display.
    """
    out: list[LyricLine] = []
    for ln in lines:
        if not ln.text.strip():
            continue
        if out and ln.time - out[-1].time < MERGE_GAP:
            out[-1] = LyricLine(out[-1].time, f"{out[-1].text} / {ln.text}")
        else:
            out.append(ln)
    return out


class LyricsCog(commands.Cog, name="Lyrics"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._session: aiohttp.ClientSession | None = None
        #: guild_id -> the live karaoke session, so `lyrics offset` can find it
        #: and so a second invocation replaces the first instead of racing it.
        self._live: dict[int, "_LiveSession"] = {}

    async def cog_unload(self) -> None:
        for sess in list(self._live.values()):
            sess.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _http(self) -> aiohttp.ClientSession:
        # One session for the cog's lifetime — the old code built and tore down
        # a ClientSession per lookup, which throws away connection reuse and
        # leaks warnings on shutdown.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def cog_check(self, ctx: commands.Context) -> bool:
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    # ── main command ─────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="lyrics", aliases=["ly"],
        description="Show lyrics for the current song, synced to playback",
        help=("Shows lyrics for whatever's playing, scrolling in time with the "
              "music. Give a song name to look up lyrics for something else.\n\n"
              "If the timing is off, `lyrics offset +1.5` nudges it later and "
              "`lyrics offset -1.5` earlier. Corrections are saved per track, "
              "so a song you've fixed once stays fixed.\n\n"
              "`lyrics offset` — show the current correction\n"
              "`lyrics offset reset` — forget it"),
    )
    async def lyrics(self, ctx: commands.Context, *, query: str | None = None):
        # `lyrics offset ...` is handled here rather than as a subcommand so
        # that plain `lyrics <song>` keeps working unchanged for both prefix
        # and slash users.
        if query and query.split()[0].lower() in ("offset", "sync", "nudge"):
            return await self._handle_offset(ctx, query.split(maxsplit=1)[1:])

        state = get_state(ctx.guild.id)
        cur = state.current

        if query:
            q = TrackQuery(title=query.strip())
        elif cur:
            q = build_query(cur)
        else:
            return await ctx.send("❌ Nothing playing. Try `lyrics <song name>`.")

        status = await ctx.send(f"🔍 Searching lyrics for **{q.title}**…")

        try:
            lyr = await fetch_lyrics(q, session=await self._http())
        except Exception:
            log.exception("lyrics lookup failed")
            return await status.edit(content="❌ Lyrics lookup failed — check my logs.")

        if lyr is None or (not lyr.has_sync and not lyr.plain):
            return await status.edit(content=f"❌ No lyrics found for **{q.title}**.")

        header = (f"🎵 **{lyr.artist_name} — {lyr.track_name}**\n"
                  if lyr.artist_name else f"🎵 **{lyr.track_name}**\n")

        live_possible = (
            not query and cur is not None and cur is state.current
            and ctx.voice_client is not None and confidence_ok(lyr, q)
        )
        if not live_possible:
            note = ""
            if not query and lyr.has_sync and not confidence_ok(lyr, q):
                # Say *why* rather than silently dumping a static wall of text —
                # "the lyrics didn't scroll" with no explanation is the thing
                # that made this feel broken.
                why = lyr.note or "couldn't confirm this is the same recording"
                note = f"*Showing static lyrics — {why}.*\n"
            body = "\n".join(ln.text for ln in lyr.lines if ln.text) \
                if lyr.has_sync else lyr.plain
            return await self._post_chunked(status, header + note, body)

        # Cancel any karaoke already running in this guild.
        prev = self._live.pop(ctx.guild.id, None)
        if prev:
            prev.cancel()

        session = _LiveSession(self, ctx, status, header, lyr, q, cur)
        self._live[ctx.guild.id] = session
        try:
            await session.run()
        finally:
            if self._live.get(ctx.guild.id) is session:
                self._live.pop(ctx.guild.id, None)

    # ── offset subcommand ────────────────────────────────────────────────────

    async def _handle_offset(self, ctx: commands.Context, rest: list[str]) -> None:
        state = get_state(ctx.guild.id)
        cur = state.current
        if not cur:
            return await ctx.send("❌ Nothing playing — nothing to re-time.")
        source_id = (cur.get("webpage_url") or "").strip()
        if not source_id:
            return await ctx.send("❌ This track has no stable ID, so I can't "
                                  "save a correction for it.")

        arg = (rest[0] if rest else "").strip().lower()
        session = self._live.get(ctx.guild.id)

        if not arg:
            saved = await db.get_lyric_offset(source_id)
            live = f"{session.alignment.offset:+.2f}s" if session else "—"
            how = "; ".join(session.alignment.reasons) if session else ""
            saved_txt = f"`{saved:+.2f}s`" if saved is not None else "*none yet*"
            msg = (f"🎚️ Timing for **{cur.get('title', 'this track')}**\n"
                   f"Applied now: `{live}`\n"
                   f"Saved: {saved_txt}")
            if how:
                msg += f"\n-# {how}"
            return await ctx.send(msg)

        if arg in ("reset", "clear", "forget", "off"):
            existed = await db.clear_lyric_offset(source_id)
            if session:
                session.set_offset(0.0)
            return await ctx.send("🎚️ Correction cleared." if existed
                                  else "🎚️ There was no saved correction.")

        try:
            delta = float(arg.replace("s", ""))
        except ValueError:
            return await ctx.send(
                "❌ Usage: `lyrics offset +1.5` (later) · `lyrics offset -1.5` "
                "(earlier) · `lyrics offset reset`"
            )
        if abs(delta) > MAX_NUDGE:
            return await ctx.send(
                f"❌ That's more than {MAX_NUDGE:.0f}s — if it's off by that "
                f"much I've probably matched the wrong recording entirely, and "
                f"an offset won't save it."
            )

        # Nudging is relative to what's applied right now, so repeated small
        # corrections converge instead of fighting each other.
        base = session.alignment.offset if session else \
            (await db.get_lyric_offset(source_id) or 0.0)
        new = base + delta
        if session:
            session.set_offset(new)
        await db.save_lyric_offset(source_id, new, cur.get("title"))
        stored = await db.get_lyric_offset(source_id)
        await ctx.send(
            f"🎚️ Shifted lyrics **{delta:+.2f}s** → now `{new:+.2f}s`.\n"
            f"-# Saved for this track (stored average `{stored:+.2f}s`). "
            f"It'll be applied automatically next time."
        )

    # ── static output ────────────────────────────────────────────────────────

    async def _post_chunked(self, status: discord.Message, header: str,
                            text: str) -> None:
        text = text or "*(no lyrics text)*"
        # Split on line boundaries so a chunk never cuts a lyric in half.
        chunks, buf = [], header
        for line in text.splitlines():
            if len(buf) + len(line) + 1 > 1900:
                chunks.append(buf)
                buf = ""
            buf += line + "\n"
        if buf.strip():
            chunks.append(buf)
        await status.edit(content=chunks[0][:2000])
        for c in chunks[1:]:
            await status.channel.send(c[:2000])


class _LiveSession:
    """One running karaoke display."""

    def __init__(self, cog: LyricsCog, ctx: commands.Context,
                 message: discord.Message, header: str, lyr, q: TrackQuery,
                 track: dict):
        self.cog = cog
        self.ctx = ctx
        self.message = message
        self.header = header
        self.lyr = lyr
        self.q = q
        self.track = track
        self.state = get_state(ctx.guild.id)
        self.gen = self.state.playback_gen
        # A real Alignment from the start, not None: `lyrics offset` can arrive
        # while run() is still awaiting the silence probe, and it reads this.
        self.alignment = Alignment()
        self.lines: list[LyricLine] = []
        self.times: list[float] = []
        self._min_interval = MIN_EDIT_INTERVAL
        self._cancelled = False
        self._dirty = asyncio.Event()

    def cancel(self) -> None:
        self._cancelled = True
        self._dirty.set()

    def set_offset(self, offset: float) -> None:
        """Re-align live, without refetching. Wakes the loop so the change is
        visible immediately rather than at the next scheduled line."""
        self.alignment.offset = offset
        self.alignment.learned = True
        self._rebuild()
        self._dirty.set()

    def _rebuild(self) -> None:
        shifted = self.alignment.apply(self.lyr.lines)
        self.lines = _merge(shifted)
        self.times = [ln.time for ln in self.lines]

    async def _resolve_alignment(self):
        source_id = self.q.source_id
        learned = await db.get_lyric_offset(source_id) if source_id else None

        lead = None
        if learned is None:
            # Only worth probing the audio when we don't already have a human's
            # answer. Bounded and best-effort: a slow or unprobeable stream
            # falls through to metadata reasoning rather than delaying the
            # display.
            try:
                lead = await asyncio.wait_for(
                    detect_leading_silence(self.q.stream_url,
                                           cache_key=source_id or ""),
                    timeout=12,
                )
            except Exception:
                lead = None

        return align(self.lyr, self.q, leading_silence=lead,
                     learned_offset=learned)

    def _render(self, idx: int, *, paused: bool = False) -> str:
        lo = max(0, idx - WINDOW)
        hi = min(len(self.lines), idx + WINDOW + 1)
        body = []
        for k in range(lo, hi):
            text = self.lines[k].text
            if k == idx:
                body.append(f"**▶ {text}**")
            else:
                body.append(f"​  {text}")
        out = self.header + "\n".join(body)
        if paused:
            out += "\n​\n-# ⏸️ paused"
        return out[:2000]

    async def _edit(self, content: str) -> bool:
        """Edit the message, absorbing rate limits. False means give up."""
        try:
            await self.message.edit(content=content)
            return True
        except discord.NotFound:
            return False                      # someone deleted the message
        except discord.Forbidden:
            return False
        except discord.HTTPException as e:
            if e.status == 429:
                # discord.py already slept through the retry-after; widen our
                # own spacing so we stop provoking it.
                self._min_interval = min(5.0, self._min_interval * 1.5)
                log.info("lyrics: rate limited, backing off to %.1fs",
                         self._min_interval)
                return True
            log.debug("lyrics edit failed: %s", e)
            return False

    def _still_current(self) -> bool:
        return (not self._cancelled
                and self.state.current is self.track
                and self.state.playback_gen == self.gen)

    async def run(self) -> None:
        self.alignment = await self._resolve_alignment()
        self._rebuild()
        if not self.lines:
            await self._edit(self.header + (self.lyr.plain[:1800] or "*(empty)*"))
            return

        vc = self.ctx.voice_client
        duration = self.q.duration or 0.0
        last_idx = -2
        last_edit = 0.0
        was_paused = False

        while True:
            if not self._still_current():
                break
            vc = self.ctx.voice_client
            if vc is None or not vc.is_connected():
                break

            clock = self.state.clock
            paused = bool(clock and clock.is_paused)
            pos = playback_position(self.state, vc)
            if duration and pos >= duration - 0.25:
                break

            # bisect over the aligned timestamps: O(log n) per tick, and it
            # lands on the right line after a seek just as readily as during
            # normal playback (the old linear scan assumed monotonic progress).
            idx = bisect.bisect_right(self.times, pos + EDIT_LEAD) - 1
            idx = max(0, idx)

            now = time.monotonic()
            if (idx != last_idx or paused != was_paused) and \
                    now - last_edit >= self._min_interval:
                if not await self._edit(self._render(idx, paused=paused)):
                    return
                last_idx, was_paused, last_edit = idx, paused, now

            # Sleep until the next line is due, in wall-clock terms.
            self._dirty.clear()
            if paused:
                sleep = 1.0                      # cheap poll, waiting to resume
            else:
                rate = (clock.rate if clock else 1.0) or 1.0
                nxt = self.times[idx + 1] if idx + 1 < len(self.times) else None
                if nxt is None:
                    sleep = MAX_SLEEP
                else:
                    sleep = (nxt - EDIT_LEAD - pos) / rate
                # Never faster than the rate limit allows, never so slow that a
                # skip goes unnoticed.
                floor = max(0.25, self._min_interval - (now - last_edit))
                sleep = min(MAX_SLEEP, max(floor, sleep))
            try:
                await asyncio.wait_for(self._dirty.wait(), timeout=sleep)
            except asyncio.TimeoutError:
                pass
            if self._cancelled:
                return

        if self._cancelled:
            return
        try:
            tail = self.header + "\n".join(f"​  {ln.text}"
                                           for ln in self.lines[-WINDOW:])
            await self._edit(tail + "\n​\n-# ✅ finished")
        except Exception:
            pass


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LyricsCog(bot))
