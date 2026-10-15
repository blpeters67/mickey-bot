"""
Per-guild playback state — queue, current track, loop, volume, etc.

Kept deliberately minimal: just a dataclass and a registry. Every command
mutates it via `get_state(guild_id)`. Nothing here knows about Discord;
the music cog wires it up.

The `playback_gen` counter is the trick that lets seek/skip cancel the
old `after` callback without races. Bumping it before `vc.stop()` makes
the stale callback's scheduled `_play_next` see a mismatched gen and bail.
"""
from __future__ import annotations

import asyncio
import itertools
from collections import deque
from dataclasses import dataclass, field

from .playback_clock import PlaybackClock

# ──────────────────────────────────────────────────────────────────────────────
# Queue entries with an identity
# ──────────────────────────────────────────────────────────────────────────────

_qids = itertools.count(1)


def new_qid() -> str:
    """A fresh occurrence id ("q17"). Process-wide and never reused, so an id
    seen by the voice planner can't come to mean a different entry later."""
    return f"q{next(_qids)}"


def qid(entry) -> str | None:
    """The occurrence id of a queue entry (or of the current track)."""
    return entry.get("_qid") if isinstance(entry, dict) else None


class TrackQueue(deque):
    """The upcoming queue: a deque whose entries each carry an occurrence id
    (``_qid``), stamped the first time an entry goes into a queue, and which
    counts its own changes (``rev``).

    Five copies of one song queued on purpose are five dicts, so five ids — a
    catalogue id or URL can't tell them apart, an id can. An entry that comes
    back (loop mode, a track put back at the front after a failed start or a
    reconnect) keeps its id: it is the same occurrence. A *copy* made for a
    queue loop (requeue_copy drops every "_" key) is a new occurrence and
    gets a new id. The same dict appended while it is still queued gets a
    shallow copy with its own id, so two positions never share one.

    `rev` lets a voice request notice that someone else changed the queue
    between its steps (core/voice_exec.py). It is not a lock: everything that
    touches the queue runs on the event loop, and a synchronous read-check-
    modify is atomic there."""

    def __init__(self, iterable=(), maxlen=None, *, rev: int = 0):
        super().__init__(maxlen=maxlen)
        self.rev = rev
        for e in iterable:
            super().append(self._stamp(e))

    # The id goes on when an entry enters; nothing else about it is touched.
    def _stamp(self, entry):
        if isinstance(entry, dict):
            if entry.get("_qid") and any(x is entry for x in self):
                entry = dict(entry)
                entry["_qid"] = new_qid()
            elif not entry.get("_qid"):
                entry["_qid"] = new_qid()
        return entry

    def _bump(self) -> None:
        self.rev += 1

    def append(self, entry) -> None:
        super().append(self._stamp(entry))
        self._bump()

    def appendleft(self, entry) -> None:
        super().appendleft(self._stamp(entry))
        self._bump()

    def extend(self, entries) -> None:
        for e in list(entries):
            super().append(self._stamp(e))
        self._bump()

    def extendleft(self, entries) -> None:
        for e in list(entries):
            super().appendleft(self._stamp(e))
        self._bump()

    def insert(self, index: int, entry) -> None:
        super().insert(index, self._stamp(entry))
        self._bump()

    def pop(self):
        out = super().pop()
        self._bump()
        return out

    def popleft(self):
        out = super().popleft()
        self._bump()
        return out

    def remove(self, value) -> None:
        super().remove(value)
        self._bump()

    def clear(self) -> None:
        super().clear()
        self._bump()

    def rotate(self, n: int = 1) -> None:
        super().rotate(n)
        self._bump()

    def reverse(self) -> None:
        super().reverse()
        self._bump()

    def __setitem__(self, index, entry) -> None:
        super().__setitem__(index, self._stamp(entry))
        self._bump()

    def __delitem__(self, index) -> None:
        super().__delitem__(index)
        self._bump()

    def __iadd__(self, other):
        self.extend(other)
        return self

    def ids(self) -> list[str | None]:
        return [qid(e) for e in self]

    def index_of(self, entry_id: str) -> int:
        """Position (0-based) of the entry with this id, or -1."""
        for i, e in enumerate(self):
            if qid(e) == entry_id:
                return i
        return -1

    def __copy__(self):
        return TrackQueue(self, self.maxlen, rev=self.rev)

    def __reduce__(self):
        return (TrackQueue, (list(self), self.maxlen))


@dataclass
class GuildState:
    queue:            deque = field(default_factory=TrackQueue)
    current:          dict | None = None
    loop:             bool = False  # repeat the current song
    # Repeat the whole queue: a finished (or skipped) song goes to the back
    # of the queue instead of away. Never on together with `loop`.
    loop_queue:       bool = False
    volume:           float = 1.0   # 1.0 = 100 %
    # Legacy wall-clock start time. Still maintained so any consumer this
    # refactor missed keeps working, but it is now *derived* from `clock` and
    # refreshed on pause/resume, so it no longer drifts. Prefer `clock`.
    track_started_at: float | None = None  # unix time of current play
    # Authoritative playback position. Counts delivered audio frames rather
    # than wall-clock seconds, so pausing, buffering, seeking and speed
    # filters can't desync it. See core/playback_clock.py for why that matters.
    clock:            PlaybackClock | None = None
    playback_gen:     int = 0
    # Active ffmpeg audio filter: the raw -af chain and a human label for
    # display. Sticky per guild until cleared — survives track changes.
    filter_af:        str | None = None
    filter_label:     str | None = None
    # Recently played tracks, oldest→newest. Drives `history`, `back`, and
    # autoplay's don't-repeat logic. Entries are resolve-on-demand stubs.
    history:          deque = field(default_factory=lambda: deque(maxlen=50))
    # Text channel the last music command came from — where the engine posts
    # skip/stall notices. Playback failures used to be invisible: a track
    # that failed to resolve was dropped with only a log line, which read as
    # "queued it, said ✅, never played".
    text_channel_id:  int | None = None
    # Serialises queue advancement (MusicCog._play_next). Two concurrent
    # kickstarts — e.g. two users queueing into an idle bot — used to both
    # pop a track, and the loser's track silently vanished.
    advance_lock:     asyncio.Lock = field(default_factory=asyncio.Lock)
    # Bumped by reset_state (stop, leave, disconnects). A song lookup that
    # started before a stop compares it after its network wait and refuses to
    # queue its result — otherwise a slow search could restart the music
    # after someone stopped it.
    queue_epoch:      int = 0
    queue_change_reason: str = "stopped"
    # Recent tracks that failed to play: (unix time, title, reason). `queue`
    # shows the latest, so "Queue is empty" right after a request comes with
    # the reason instead of nothing.
    failures:         deque = field(default_factory=lambda: deque(maxlen=5))

    def __setattr__(self, name, value) -> None:
        # Commands rebuild the queue (`state.queue = deque(shuffled)`): keep
        # it a TrackQueue, its ids, and a revision count that moves on.
        if name == "queue" and not isinstance(value, TrackQueue):
            old = self.__dict__.get("queue")
            rev = (old.rev + 1) if isinstance(old, TrackQueue) else 0
            value = TrackQueue(value if value is not None else (), rev=rev)
        elif name == "queue" and value is not self.__dict__.get("queue"):
            old = self.__dict__.get("queue")
            if isinstance(old, TrackQueue) and value.rev <= old.rev:
                value.rev = old.rev + 1
        object.__setattr__(self, name, value)


_states: dict[int, GuildState] = {}


def get_state(guild_id: int) -> GuildState:
    """Lazily create and return the GuildState for this guild."""
    state = _states.get(guild_id)
    if state is None:
        state = GuildState()
        _states[guild_id] = state
    return state


def requeue_copy(track: dict) -> dict:
    """A queue entry for a song coming round again (queue loop): the song,
    not its last run — no stale stream URL, status, retries or position, so
    it resolves fresh from its page like any other entry, and nothing still
    holding the old dict can mistake the new one for itself."""
    fresh = {k: v for k, v in track.items() if not k.startswith("_")}
    fresh["stream_url"] = None
    fresh["_needs_resolve"] = True
    page = track.get("webpage_url") or ""
    # A song that was resolved has its real page (the same upload comes back);
    # one that never got that far still has what it was going to search for.
    resolved = not track.get("_needs_resolve") and "://" in page
    fresh["_search_query"] = (page if resolved else
                              track.get("_search_query") or page or track.get("title"))
    return fresh


def clear_queue(guild_id: int) -> int:
    """Clear upcoming tracks and invalidate older admissions, keeping playback."""
    s = get_state(guild_id)
    count = len(s.queue)
    s.queue.clear()
    s.queue_epoch += 1
    s.queue_change_reason = "cleared"
    return count


def reset_state(guild_id: int) -> None:
    """Drop the queue and current track, e.g. after stop or kick."""
    s = get_state(guild_id)
    s.queue.clear()
    s.current          = None
    s.track_started_at = None
    s.clock            = None
    s.playback_gen    += 1   # invalidate any pending after-callbacks
    s.queue_epoch     += 1   # …and any song lookups still in flight
    s.queue_change_reason = "stopped"


def playback_position(state: "GuildState", vc=None) -> float:
    """Seconds into the current track. The one function everything should use.

    Falls back to the legacy wall-clock field when no clock exists yet (e.g. a
    guild whose playback started before this module was reloaded), so callers
    never have to branch.
    """
    if state.clock is not None:
        return state.clock.position(vc)
    if state.track_started_at is None:
        return 0.0
    import time
    return max(0.0, time.time() - state.track_started_at)
