"""Tests for the playback clock.

The whole point of this module is that it stays right when the wall clock is
wrong, so most of these tests deliberately move time and the frame counter
independently and check the clock follows the *audio*.
"""
from __future__ import annotations

import time

import pytest

from core.playback_clock import FRAME_SECONDS, PlaybackClock, rate_from_filter


class FakePlayer:
    def __init__(self, loops: int = 0):
        self.loops = loops


class FakeVC:
    """Stands in for discord.VoiceClient — only `_player.loops` is read."""

    def __init__(self, loops: int = 0):
        self._player = FakePlayer(loops)

    def advance(self, seconds: float) -> None:
        self._player.loops += int(round(seconds / FRAME_SECONDS))


def play(clock: PlaybackClock, vc: FakeVC, seconds: float) -> None:
    """Simulate `seconds` of normal playback: frames delivered *and* real time
    passing, in step. Advancing only one of the two is what a stall or a
    broken-internals read looks like, and those are tested separately."""
    vc.advance(seconds)
    clock.anchor_mono -= seconds
    if clock.paused_mono is not None:
        clock.paused_mono -= seconds


def test_position_tracks_frames_not_wall_clock():
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    # 10 seconds of wall time pass but only 4 seconds of audio was delivered —
    # a stalled/buffering stream. Position must report the audio, not the wall.
    c.anchor_mono = time.monotonic() - 10.0
    vc.advance(4.0)
    assert c.position(vc) == pytest.approx(4.0, abs=0.05)


def test_falls_back_to_wall_clock_without_voice_client():
    c = PlaybackClock()
    c.rebase(30.0)
    c.anchor_mono = time.monotonic() - 5.0
    assert c.position(None) == pytest.approx(35.0, abs=0.05)


def test_pause_freezes_position():
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    play(c, vc, 12.0)
    c.pause(vc)
    frozen = c.position(vc)
    assert frozen == pytest.approx(12.0, abs=0.05)
    # Wall clock keeps moving during the pause; position must not.
    c.anchor_mono -= 60.0
    assert c.position(vc) == pytest.approx(frozen, abs=0.001)
    assert c.is_paused


def test_resume_continues_from_pause_point():
    """The regression that caused most 'lyrics are out of sync' reports:
    a pause used to shift every later reading by the pause duration."""
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    play(c, vc, 12.0)
    c.pause(vc)

    # 45 seconds of real time go by while paused.
    c.anchor_mono -= 45.0
    c.paused_mono -= 45.0

    # discord.py zeroes the frame counter on resume.
    vc._player.loops = 0
    c.resume(vc)
    assert c.position(vc) == pytest.approx(12.0, abs=0.05)

    play(c, vc, 3.0)
    assert c.position(vc) == pytest.approx(15.0, abs=0.05)


def test_repeated_pause_resume_does_not_accumulate_error():
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    played = 0.0
    for _ in range(6):
        play(c, vc, 5.0)
        played += 5.0
        c.pause(vc)
        c.anchor_mono -= 20.0        # long pause in wall-clock terms
        c.paused_mono -= 20.0
        vc._player.loops = 0
        c.resume(vc)
    assert c.position(vc) == pytest.approx(played, abs=0.1)


def test_pause_is_idempotent():
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    play(c, vc, 8.0)
    c.pause(vc)
    first = c.position(vc)
    c.pause(vc)
    assert c.position(vc) == pytest.approx(first, abs=0.001)


def test_resume_without_pause_is_noop():
    vc = FakeVC(loops=100)
    c = PlaybackClock()
    c.rebase(5.0)
    before = c.position(vc)
    c.resume(vc)
    assert c.position(vc) == pytest.approx(before, abs=0.05)


def test_rebase_after_seek():
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    play(c, vc, 30.0)
    # Seek to 2:00 restarts ffmpeg, so discord.py builds a fresh player.
    vc = FakeVC()
    c.rebase(120.0)
    assert c.position(vc) == pytest.approx(120.0, abs=0.05)
    play(c, vc, 7.0)
    assert c.position(vc) == pytest.approx(127.0, abs=0.05)


def test_speed_filter_scales_position():
    """Under nightcore the track is consumed 1.25x faster than it's played,
    so 40s of delivered audio is 50s of the track."""
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0, rate=1.25)
    play(c, vc, 40.0)                # 40s of audio delivered over 40s of time
    assert c.position(vc) == pytest.approx(50.0, abs=0.1)


def test_stall_is_honoured_not_rejected():
    """A long buffering stall leaves the frame counter far behind the wall
    clock. That gap is the truth, not an error — the audio really is behind."""
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    vc.advance(10.0)                           # only 10s of audio delivered
    c.anchor_mono = time.monotonic() - 120.0   # over 2 minutes of real time
    assert c.position(vc) == pytest.approx(10.0, abs=0.1)


def test_offset_nudge():
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    play(c, vc, 10.0)
    c.nudge(1.5)
    assert c.position(vc) == pytest.approx(11.5, abs=0.05)
    c.nudge(-3.0)
    assert c.position(vc) == pytest.approx(8.5, abs=0.05)


def test_position_never_negative():
    c = PlaybackClock()
    c.rebase(0.0)
    c.nudge(-50.0)
    assert c.position(None) >= 0.0


def test_missing_player_attribute_falls_back():
    class Bare:
        pass

    c = PlaybackClock()
    c.rebase(0.0)
    c.anchor_mono = time.monotonic() - 6.0
    assert c.position(Bare()) == pytest.approx(6.0, abs=0.05)


def test_broken_loops_attribute_is_survived():
    class Weird:
        class _P:
            loops = "not a number"
        _player = _P()

    c = PlaybackClock()
    c.rebase(0.0)
    c.anchor_mono = time.monotonic() - 6.0
    assert c.position(Weird()) == pytest.approx(6.0, abs=0.05)


def test_absurd_frame_count_is_rejected():
    """If our assumptions about discord.py break, prefer plausible over wild."""
    vc = FakeVC(loops=10_000_000)      # ~55 hours of frames
    c = PlaybackClock()
    c.rebase(0.0)
    c.anchor_mono = time.monotonic() - 5.0
    assert c.position(vc) == pytest.approx(5.0, abs=0.1)


def test_as_started_at_round_trips():
    vc = FakeVC()
    c = PlaybackClock()
    c.rebase(0.0)
    play(c, vc, 20.0)
    started = c.as_started_at(vc)
    assert time.time() - started == pytest.approx(20.0, abs=0.1)


@pytest.mark.parametrize("chain,expected", [
    (None, 1.0),
    ("", 1.0),
    ("bass=g=10", 1.0),
    ("asetrate=48000*1.25,aresample=48000", 1.25),
    ("asetrate=48000*0.8,aresample=48000", 0.8),
    ("atempo=1.5", 1.5),
    ("atempo=2.0,atempo=1.5", 3.0),
    ("atempo=0.5,atempo=0.5", 0.25),
    ("garbage=", 1.0),
])
def test_rate_from_filter(chain, expected):
    assert rate_from_filter(chain) == pytest.approx(expected, rel=1e-6)


def test_rate_from_filter_is_clamped():
    assert rate_from_filter("atempo=2.0,atempo=2.0,atempo=2.0,atempo=2.0") <= 8.0
    assert rate_from_filter("atempo=0.5,atempo=0.5,atempo=0.5,atempo=0.5") >= 0.125
