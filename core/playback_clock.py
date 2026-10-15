"""
Playback clock — the single source of truth for "how far into this track are we?"

Why this exists
---------------
Position used to be `time.time() - state.track_started_at`. That's a wall
clock, and a wall clock lies in four separate ways:

  * **Pause.** `vc.pause()` stops the audio but not the clock, so a 30-second
    pause shifts every subsequent position read 30 seconds late — permanently.
    Synced lyrics and the `nowplaying` progress bar never recovered, and the
    only cure was skipping the track.
  * **Buffering and stalls.** When the CDN throttles, ffmpeg starves and the
    voice socket sends silence, but wall-clock time marches on regardless.
  * **Seek granularity.** ffmpeg was started with `-ss {int(seek)}`, truncating
    to whole seconds, while the clock was rebased to the exact float. Every
    seek introduced up to a second of error.
  * **Speed filters.** `nightcore` / `slowed` / `speed ×N` change how fast the
    track is consumed. One wall-clock second is 1.25 track-seconds under
    nightcore, so position drifted 25% off over the length of a song.

The fix is to prefer the *audio* clock over the wall clock. discord.py's
AudioPlayer thread increments `loops` once per 20 ms frame that it actually
hands to the voice socket, so `loops * 0.02` is how much audio has genuinely
been delivered. It doesn't advance while paused, and it doesn't advance while
ffmpeg is starved — which is exactly the behaviour we want.

Two caveats drive the design:

  * `AudioPlayer.resume()` resets `loops` to 0, so the counter measures time
    since the last start-or-resume, not since the track began. We keep our own
    `anchor` (position in the track at that moment) and add the counter to it.
  * `loops` is a private attribute. Every read is defended, sanity-checked
    against the wall clock, and falls back to pause-aware wall-clock timing if
    discord.py ever changes shape. A wrong-but-close clock beats a crash.

Everything here is pure Python and takes the voice client as a parameter, so
it's testable without discord.py — see tests/test_playback_clock.py.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

__all__ = [
    "PlaybackClock",
    "rate_from_filter",
    "FRAME_SECONDS",
]

# discord.py sends one 20 ms Opus frame per player loop.
FRAME_SECONDS = 0.02

# The frame counter can legitimately fall *behind* the wall clock by any
# amount — that's exactly what a buffering stall looks like, and honouring it
# is the point of this module. It can never legitimately run *ahead*: you
# cannot deliver more audio than time has passed. So the sanity check is
# deliberately one-sided. Running ahead means our assumption about discord.py's
# internals is wrong (most likely we read a stale player from before the
# current vc.play()), and the wall clock is the safer answer.
_MAX_AHEAD = 5.0

_ASETRATE = re.compile(r"asetrate\s*=\s*(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)")
_ATEMPO = re.compile(r"atempo\s*=\s*(\d+(?:\.\d+)?)")


def rate_from_filter(filter_af: str | None) -> float:
    """How many track-seconds pass per second of delivered audio.

    `nightcore` is `asetrate=48000*1.25,aresample=48000`: resampling upward
    consumes the source 1.25× faster, so one output second covers 1.25 seconds
    of the track. `speed ×N` chains `atempo=` factors, which multiply.

    Returns 1.0 for no filter, unknown filters, or anything that doesn't parse
    — an unknown filter is assumed not to change tempo, which is true of every
    filter in AUDIO_FILTERS except the speed ones.
    """
    if not filter_af:
        return 1.0
    rate = 1.0
    m = _ASETRATE.search(filter_af)
    if m:
        try:
            base, mult = float(m.group(1)), float(m.group(2))
            if base > 0 and mult > 0:
                rate *= mult
        except (TypeError, ValueError):
            pass
    for chunk in _ATEMPO.findall(filter_af):
        try:
            v = float(chunk)
        except (TypeError, ValueError):
            continue
        if v > 0:
            rate *= v
    # Clamp to the range the `filter speed` command already enforces, so a
    # malformed chain can't produce an absurd multiplier.
    return min(8.0, max(0.125, rate))


@dataclass
class PlaybackClock:
    """Tracks true position within a single track.

    A new clock is created every time ffmpeg is (re)started — new track, seek,
    filter change, SponsorBlock hop. `anchor` is where in the track that
    restart began; the frame counter or the wall clock measures forward from
    there.
    """

    #: Position in the track, in seconds, at the last start-or-resume.
    anchor: float = 0.0
    #: `time.monotonic()` at that same moment. Monotonic, not `time.time()`,
    #: so an NTP step or a DST change can't jerk playback position sideways.
    anchor_mono: float = field(default_factory=time.monotonic)
    #: Track-seconds consumed per second of delivered audio (speed filters).
    rate: float = 1.0
    #: `time.monotonic()` when paused, or None while running.
    paused_mono: float | None = None
    #: Frame count at the moment we paused, so position freezes exactly.
    paused_frames: int | None = None
    #: Manual correction, in seconds, applied on top (used by lyrics nudging).
    #: Positive means "the clock reads later", i.e. lyrics advance sooner.
    offset: float = 0.0
    #: Set once the frame counter proves unreadable, to stop retrying it.
    _frames_unavailable: bool = False

    # ── reading ──────────────────────────────────────────────────────────────

    def position(self, vc=None) -> float:
        """Current position within the track, in seconds, never negative.

        Pass the voice client to get frame-accurate timing. Without it, this
        degrades to pause-aware wall-clock timing, which is still strictly
        better than the old `time.time() - started_at`.
        """
        return max(0.0, self._raw_position(vc) + self.offset)

    def _raw_position(self, vc=None) -> float:
        if self.paused_mono is not None:
            # Frozen. Prefer the frame count captured at pause time.
            if self.paused_frames is not None:
                return self.anchor + self.paused_frames * FRAME_SECONDS * self.rate
            return self.anchor + (self.paused_mono - self.anchor_mono) * self.rate

        wall = self.anchor + (time.monotonic() - self.anchor_mono) * self.rate

        frames = self._read_frames(vc)
        if frames is None:
            return wall

        audio = self.anchor + frames * FRAME_SECONDS * self.rate
        # Behind the wall clock is normal and meaningful (a stall). Ahead of it
        # is impossible, so treat it as a broken read.
        if audio > wall + _MAX_AHEAD:
            return wall
        return audio

    def _read_frames(self, vc) -> int | None:
        """discord.py's delivered-frame count, or None if unreadable."""
        if vc is None or self._frames_unavailable:
            return None
        try:
            player = getattr(vc, "_player", None)
            if player is None:
                return None
            loops = getattr(player, "loops", None)
            if isinstance(loops, bool) or not isinstance(loops, int):
                # bool is an int subclass; a True here would mean 0.02s.
                self._frames_unavailable = True
                return None
            if loops < 0:
                return None
            return loops
        except Exception:
            # Never let an internals change take playback position down with it.
            self._frames_unavailable = True
            return None

    # ── transitions ──────────────────────────────────────────────────────────

    def pause(self, vc=None) -> None:
        """Freeze the clock. Idempotent."""
        if self.paused_mono is not None:
            return
        self.paused_frames = self._read_frames(vc)
        self.paused_mono = time.monotonic()

    def resume(self, vc=None) -> None:
        """Unfreeze, rebasing onto wherever we froze. Idempotent.

        Called *after* `vc.resume()`, because that's what resets discord.py's
        frame counter to zero — and zero is what our new anchor assumes.
        """
        if self.paused_mono is None:
            return
        self.anchor = self._raw_position(None) if self.paused_frames is None \
            else self.anchor + self.paused_frames * FRAME_SECONDS * self.rate
        self.paused_mono = None
        self.paused_frames = None
        self.anchor_mono = time.monotonic()

    def rebase(self, position: float, *, rate: float | None = None) -> None:
        """Hard-set the position, e.g. after a seek restarts ffmpeg."""
        self.anchor = max(0.0, float(position))
        self.anchor_mono = time.monotonic()
        self.paused_mono = None
        self.paused_frames = None
        self._frames_unavailable = False
        if rate is not None:
            self.rate = rate

    def nudge(self, delta: float) -> float:
        """Shift reported position by `delta` seconds. Returns the new offset."""
        self.offset += float(delta)
        return self.offset

    @property
    def is_paused(self) -> bool:
        return self.paused_mono is not None

    # ── interop ──────────────────────────────────────────────────────────────

    def as_started_at(self, vc=None) -> float:
        """A `time.time()`-domain value equivalent to the legacy
        `state.track_started_at`, so older call sites keep working.

        Kept in sync (rather than deleted) deliberately: any consumer this
        refactor missed still gets a *pause-corrected* answer instead of a
        drifting one.
        """
        return time.time() - self.position(vc)
