"""
Queue and player operations shared by the typed commands (cogs/music.py) and
voice control (core/voice_exec.py) — one implementation, so "move", "remove",
"shuffle", "skip"… can't behave differently depending on how they were asked
for.

They take the GuildState (and, for the player, the voice client), change it,
and say what they did — including the occurrence ids (core/state.py) of the
entries they touched. Synchronous on purpose: on the event loop a
find-check-change with no await in between is atomic, so nothing else can
move the queue between finding an entry and acting on it.

Positions: the typed commands number the upcoming queue from 1 (`move 5 1`);
the song playing now is not in the queue. Voice control works with ids and
turns its references into ids first (core/voice_plan.py).
"""
from __future__ import annotations

import random
from collections import deque
from typing import Iterable

from .formatters import interleave_queue
from .state import GuildState, qid, requeue_copy


class QueueOpError(ValueError):
    """An operation that can't be done as asked; the message says why."""


# ── ids and positions ─────────────────────────────────────────────────────────

def ids_at(state: GuildState, positions: Iterable[int]) -> list[str]:
    """Occurrence ids of the entries at these 1-based queue positions, in the
    order given. Raises QueueOpError for a position the queue doesn't have."""
    items = list(state.queue)
    out = []
    for p in positions:
        if not 1 <= p <= len(items):
            raise QueueOpError(f"position {p} isn't in the queue (it has {len(items)})")
        out.append(qid(items[p - 1]))
    return out


def find(state: GuildState, entry_id: str) -> tuple[int, dict] | None:
    """(0-based position, entry) of an id in the upcoming queue, or None."""
    for i, e in enumerate(state.queue):
        if qid(e) == entry_id:
            return i, e
    return None


# ── the queue ─────────────────────────────────────────────────────────────────

def remove_ids(state: GuildState, entry_ids: Iterable[str]) -> list[dict]:
    """Take these entries out of the queue; returns them in queue order.
    Ids not (or no longer) queued are ignored — the caller checks first when
    that matters."""
    gone = set(entry_ids)
    kept, removed = [], []
    for e in state.queue:
        (removed if qid(e) in gone else kept).append(e)
    if removed:
        state.queue = deque(kept)
    return removed


def move_id(state: GuildState, entry_id: str, index: int) -> tuple[dict, int, int]:
    """Move one entry so it ends up at 0-based `index` (counted with the entry
    already taken out; clamped to the ends). Returns (entry, from, to)."""
    hit = find(state, entry_id)
    if hit is None:
        raise QueueOpError("that song isn't in the queue any more")
    i, entry = hit
    items = list(state.queue)
    items.pop(i)
    to = max(0, min(index, len(items)))
    items.insert(to, entry)
    state.queue = deque(items)
    return entry, i, to


def swap_ids(state: GuildState, a: str, b: str) -> tuple[dict, dict]:
    hit_a, hit_b = find(state, a), find(state, b)
    if hit_a is None or hit_b is None:
        raise QueueOpError("that song isn't in the queue any more")
    if a == b:
        raise QueueOpError("those are the same song")
    items = list(state.queue)
    (i, ea), (j, eb) = hit_a, hit_b
    items[i], items[j] = eb, ea
    state.queue = deque(items)
    return ea, eb


def shuffle(state: GuildState, rng: random.Random | None = None) -> int:
    """Shuffle the upcoming queue (the current song is untouched)."""
    items = list(state.queue)
    (rng or random).shuffle(items)
    state.queue = deque(items)
    return len(items)


def reverse(state: GuildState) -> int:
    state.queue.reverse()
    return len(state.queue)


def balance(state: GuildState) -> int:
    """Round-robin between requesters (core/formatters.interleave_queue)."""
    state.queue = interleave_queue(list(state.queue))
    return len(state.queue)


def dedupe(state: GuildState) -> list[dict]:
    """Remove later copies of songs already queued or playing (same link, or
    the same title when there is no link yet). Returns what was removed."""
    seen: set[str] = set()
    if state.current:
        cur_key = state.current.get("webpage_url") or state.current.get("title")
        if cur_key:
            seen.add(cur_key)
    kept, removed = [], []
    for t in state.queue:
        key = t.get("webpage_url") or t.get("title") or ""
        if key in seen:
            removed.append(t)
            continue
        seen.add(key)
        kept.append(t)
    if removed:
        state.queue = deque(kept)
    return removed


def drop_before(state: GuildState, entry_id: str) -> list[dict]:
    """Everything queued before this entry is dropped (`skipto`): it becomes
    the next song. Returns what was dropped."""
    hit = find(state, entry_id)
    if hit is None:
        raise QueueOpError("that song isn't in the queue any more")
    dropped = [state.queue.popleft() for _ in range(hit[0])]
    return dropped


def insert_entries(state: GuildState, entries: list[dict], at: int | None = None) -> list[str]:
    """Put new entries in the queue (at the end, or before 0-based `at`), and
    return their occurrence ids — stamped as they go in."""
    if at is None:
        state.queue.extend(entries)
        start = len(state.queue) - len(entries)
    else:
        at = max(0, min(at, len(state.queue)))
        for n, e in enumerate(entries):
            state.queue.insert(at + n, e)
        start = at
    return [qid(e) for e in list(state.queue)[start:start + len(entries)]]


# ── the player ────────────────────────────────────────────────────────────────

def skip_current(state: GuildState, vc, music, guild) -> bool:
    """Stop the song playing now so the queue moves on. With the whole queue
    looping it goes round again (last); the single-song loop doesn't hold it.
    A song still starting (no sound yet) is dropped the same way and the
    next one kicked off. False if nothing was playing or starting."""
    if vc is not None and (vc.is_playing() or vc.is_paused()):
        if state.loop_queue and state.current is not None:
            state.queue.append(requeue_copy(state.current))
        state.current = None            # don't let loop re-queue it
        vc.stop()                       # its end advances the queue
        return True
    if vc is not None and music is not None and music._abandon_starting(state):
        music._kickstart(guild, vc)     # it hadn't made a sound yet
        return True
    return False


def pause(state: GuildState, vc) -> bool:
    """Pause the music (not speech talking over a pause). False if nothing
    was playing."""
    from . import voice_mix
    if vc is None or not voice_mix.music_is_playing(vc):
        return False
    # Freeze the clock *before* the audio stops, so the captured frame count
    # still reflects a running player.
    if state.clock is not None:
        state.clock.pause(vc)
    vc.pause()
    if state.clock is not None:
        state.track_started_at = state.clock.as_started_at(vc)
    return True


def resume(state: GuildState, vc) -> bool:
    """Resume paused music — also from under speech talking over the pause."""
    from . import voice_mix
    if vc is None or not voice_mix.music_is_paused(vc):
        return False
    voice_mix.resume_music(vc)
    # After the resume, because that's what zeroes discord.py's frame
    # counter — and a zeroed counter is what the new anchor assumes.
    if state.clock is not None:
        state.clock.resume(vc)
        state.track_started_at = state.clock.as_started_at(vc)
    return True


def set_volume(state: GuildState, vc, percent: int) -> int:
    """Set the volume (percent, 0-1000) on the state and the playing source."""
    percent = max(0, min(1000, int(percent)))
    state.volume = percent / 100
    if vc is not None and getattr(vc, "source", None) is not None:
        vc.source.volume = state.volume
    return percent


LOOP_MODES = ("song", "queue", "off")


def loop_mode(state: GuildState) -> str:
    return "song" if state.loop else "queue" if state.loop_queue else "off"


def set_loop(state: GuildState, mode: str) -> tuple[str, str]:
    """Set the loop to song / queue / off. Returns (before, after)."""
    if mode not in LOOP_MODES:
        raise QueueOpError(f"unknown loop mode {mode!r}")
    before = loop_mode(state)
    state.loop, state.loop_queue = mode == "song", mode == "queue"
    return before, mode
