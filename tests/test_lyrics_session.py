"""Integration tests for the live karaoke loop and the offset store.

The unit tests cover the maths; these cover the parts that only go wrong when
things actually run — that the scheduler advances, terminates, doesn't spin,
and reacts to a mid-song correction.
"""
from __future__ import annotations

import asyncio

import pytest

import cogs.lyrics as lyrics_mod
from core import db
from core.lyrics_sync import LyricLine, SyncedLyrics, TrackQuery
from core.playback_clock import PlaybackClock
from core.state import get_state, reset_state

pytestmark = pytest.mark.asyncio

GUILD = 918273645


# ── fakes ─────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self):
        self.edits: list[str] = []
        self.channel = self

    async def edit(self, *, content=None, **_):
        self.edits.append(content or "")

    async def send(self, content=None, **_):
        self.edits.append(content or "")


class FakeVC:
    """No `_player`, so the clock falls back to wall time — which is what makes
    the position advance in real seconds during the test."""

    def is_connected(self):
        return True


class FakeGuild:
    id = GUILD


class FakeCtx:
    def __init__(self):
        self.guild = FakeGuild()
        self.voice_client = FakeVC()
        self.sent: list[str] = []

    async def send(self, content=None, **_):
        self.sent.append(content or "")
        return FakeMessage()


def _lyrics(times):
    return SyncedLyrics(
        lines=[LyricLine(t, f"line {i}") for i, t in enumerate(times)],
        plain="\n".join(f"line {i}" for i in range(len(times))),
        track_name="Song", artist_name="Artist",
        lrc_duration=1.0, confidence=0.95,
    )


@pytest.fixture
def fast(monkeypatch):
    """Compress the scheduler's constants so a whole song runs in <1s."""
    monkeypatch.setattr(lyrics_mod, "EDIT_LEAD", 0.0)
    monkeypatch.setattr(lyrics_mod, "MIN_EDIT_INTERVAL", 0.01)
    monkeypatch.setattr(lyrics_mod, "MAX_SLEEP", 0.05)
    monkeypatch.setattr(lyrics_mod, "MERGE_GAP", 0.0)


@pytest.fixture
def session(fast):
    reset_state(GUILD)
    track = {"title": "Artist - Song", "duration": 1.2,
             "webpage_url": "", "stream_url": ""}
    state = get_state(GUILD)
    state.current = track
    state.clock = PlaybackClock()
    state.clock.rebase(0.0)

    ctx = FakeCtx()
    msg = FakeMessage()
    q = TrackQuery(title="Song", artist="Artist", duration=1.2)
    cog = lyrics_mod.LyricsCog.__new__(lyrics_mod.LyricsCog)
    cog._live = {}
    s = lyrics_mod._LiveSession(cog, ctx, msg, "HDR\n",
                                _lyrics([0.0, 0.15, 0.3, 0.45]), q, track)
    yield s, msg, state, track
    reset_state(GUILD)


# ── the loop ──────────────────────────────────────────────────────────────────

async def test_loop_advances_through_lines_and_finishes(session):
    s, msg, state, _ = session
    await asyncio.wait_for(s.run(), timeout=5)

    assert msg.edits, "the display never rendered"
    assert "finished" in msg.edits[-1]
    # The highlight must actually reach the last line, not just show it as
    # context while stalling on an earlier one.
    assert any("**▶ line 3**" in e for e in msg.edits)
    # And it must not have hammered the API: far fewer edits than loop turns.
    assert len(msg.edits) <= 12, f"too many edits: {len(msg.edits)}"


async def test_highlight_only_moves_forward(session):
    s, msg, _, _ = session
    await asyncio.wait_for(s.run(), timeout=5)

    seen = []
    for e in msg.edits:
        for i in range(4):
            if f"**▶ line {i}**" in e:
                seen.append(i)
    assert seen == sorted(seen), f"highlight jumped backwards: {seen}"


async def test_loop_stops_when_the_track_changes(session):
    s, msg, state, _ = session

    async def swap():
        await asyncio.sleep(0.05)
        state.current = {"title": "something else"}

    await asyncio.wait_for(asyncio.gather(s.run(), swap()), timeout=5)
    # It exited early, so it never got to the last line.
    assert not any("**▶ line 3**" in e for e in msg.edits)


async def test_cancel_stops_promptly(session):
    s, msg, _, _ = session

    async def stop():
        await asyncio.sleep(0.03)
        s.cancel()

    await asyncio.wait_for(asyncio.gather(s.run(), stop()), timeout=5)
    assert "finished" not in (msg.edits[-1] if msg.edits else "")


async def test_loop_survives_a_deleted_message(session):
    """A user deleting the lyrics message must end the session, not spin."""
    import discord

    s, msg, _, _ = session

    async def boom(*_, **__):
        raise discord.NotFound(_FakeResp(), "gone")

    msg.edit = boom
    await asyncio.wait_for(s.run(), timeout=5)


class _FakeResp:
    status = 404
    reason = "Not Found"


# ── live re-alignment ─────────────────────────────────────────────────────────

async def test_set_offset_shifts_timings_immediately(session):
    s, _, _, _ = session
    s.alignment = lyrics_mod.align(s.lyr, s.q)
    s._rebuild()
    before = list(s.times)

    s.set_offset(2.5)
    assert s.times == pytest.approx([t + 2.5 for t in before])
    assert s._dirty.is_set(), "the loop wasn't woken to redraw"


async def test_render_marks_paused(session):
    s, _, _, _ = session
    s.alignment = lyrics_mod.align(s.lyr, s.q)
    s._rebuild()
    assert "paused" in s._render(0, paused=True)
    assert "paused" not in s._render(0, paused=False)


async def test_render_highlights_exactly_one_line(session):
    s, _, _, _ = session
    s.alignment = lyrics_mod.align(s.lyr, s.q)
    s._rebuild()
    out = s._render(2)
    assert out.count("**▶ ") == 1
    assert "**▶ line 2**" in out


async def test_merge_collapses_near_simultaneous_lines():
    merged = lyrics_mod._merge([
        LyricLine(10.0, "a"), LyricLine(10.3, "b"), LyricLine(30.0, "c"),
    ])
    assert [m.text for m in merged] == ["a / b", "c"]


async def test_merge_drops_blank_marker_lines():
    merged = lyrics_mod._merge([
        LyricLine(0.0, ""), LyricLine(50.0, "  "), LyricLine(60.0, "real"),
    ])
    assert [m.text for m in merged] == ["real"]


# ── learned offsets ───────────────────────────────────────────────────────────

async def test_offset_round_trips():
    await db.init_db()
    url = "https://example.test/watch?v=roundtrip"
    await db.clear_lyric_offset(url)
    assert await db.get_lyric_offset(url) is None

    await db.save_lyric_offset(url, 1.5, "Some Song")
    assert await db.get_lyric_offset(url) == pytest.approx(1.5)

    assert await db.clear_lyric_offset(url) is True
    assert await db.get_lyric_offset(url) is None


async def test_repeat_corrections_are_averaged_not_replaced():
    """Nudges carry human reaction time, so they're averaged. A second opinion
    should move the stored value toward itself without overshooting to it."""
    await db.init_db()
    url = "https://example.test/watch?v=average"
    await db.clear_lyric_offset(url)

    await db.save_lyric_offset(url, 2.0)
    await db.save_lyric_offset(url, 3.0)
    val = await db.get_lyric_offset(url)
    assert 2.0 < val < 3.0
    await db.clear_lyric_offset(url)


async def test_established_value_resists_one_bad_nudge():
    await db.init_db()
    url = "https://example.test/watch?v=stable"
    await db.clear_lyric_offset(url)

    for _ in range(8):
        await db.save_lyric_offset(url, 1.0)
    settled = await db.get_lyric_offset(url)
    assert settled == pytest.approx(1.0, abs=0.05)

    await db.save_lyric_offset(url, 20.0)      # someone fat-fingers it
    assert await db.get_lyric_offset(url) < 5.0
    await db.clear_lyric_offset(url)


async def test_empty_source_id_is_ignored():
    await db.init_db()
    assert await db.get_lyric_offset("") is None
    await db.save_lyric_offset("", 1.0)        # must not raise
    assert await db.clear_lyric_offset("") is False
