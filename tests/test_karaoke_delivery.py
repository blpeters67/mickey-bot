"""Tests for karaoke's file-delivery path.

The old version swapped the instrumental into live playback, which could never
work on a CPU host: separation is slower than realtime, so the track always
finished before the swap was ready. These tests pin the replacement — resolve,
separate, hand back a file — and the details that make it pleasant: a readable
attachment name, a cache hit that skips the work, and no interference with the
player.
"""
from __future__ import annotations

import os

import pytest

import cogs.karaoke as karaoke_mod
from cogs.karaoke import _safe_filename


# ── attachment naming ─────────────────────────────────────────────────────────

def test_safe_filename_is_readable():
    """The cache stores files under a hash; nobody wants a3f9c1….mp3 in their
    Downloads folder."""
    assert _safe_filename("Daft Punk - Get Lucky") == \
        "Daft Punk - Get Lucky (Instrumental).mp3"


@pytest.mark.parametrize("bad", [
    'song/with/slashes', 'a:colon', 'quo"tes', 'pipe|char',
    'star*', 'question?', 'back\\slash', 'angle<brackets>',
])
def test_safe_filename_strips_path_and_reserved_characters(bad):
    out = _safe_filename(bad)
    for ch in '<>:"/\\|?*':
        assert ch not in out, f"{ch!r} survived in {out!r}"


def test_safe_filename_rejects_control_characters():
    out = _safe_filename("song\x00with\x1fcontrol")
    assert "\x00" not in out and "\x1f" not in out


def test_safe_filename_cannot_traverse_directories():
    out = _safe_filename("../../etc/passwd")
    assert "/" not in out and not out.startswith("..")


def test_safe_filename_truncates_long_titles():
    out = _safe_filename("x" * 500)
    assert len(out) < 130
    assert out.endswith(" (Instrumental).mp3")


def test_safe_filename_handles_empty_and_junk():
    assert _safe_filename("").endswith(".mp3")
    assert _safe_filename("...").endswith(".mp3")
    assert _safe_filename(None).endswith(".mp3")


def test_safe_filename_collapses_whitespace():
    assert _safe_filename("too    many\t\tspaces") == \
        "too many spaces (Instrumental).mp3"


def test_safe_filename_keeps_unicode():
    """Non-ASCII titles are fine in an attachment name and mangling them
    would make half a music library unrecognisable."""
    out = _safe_filename("Sigur Rós — Hoppípolla")
    assert "Sigur Rós" in out and "Hoppípolla" in out


# ── the command no longer touches playback ────────────────────────────────────

def test_module_does_not_drive_the_player():
    """The whole point of the rewrite: karaoke must not start, stop, swap or
    seek playback. If any of these reappear, the race is back."""
    src = open(karaoke_mod.__file__, encoding="utf-8").read()
    for forbidden in ("_begin_playback", "_local_path", "vc.play",
                      "voice_client.play", "state.queue.append"):
        assert forbidden not in src, f"karaoke reaches into playback via {forbidden}"


def test_bitrate_default_fits_a_normal_song_under_discords_limit():
    """192k over ~4 minutes is ~5.8 MB; 320k would be ~9.6 MB and half of all
    songs would spill past Discord's 10 MB attachment limit into a link."""
    from core.config import settings
    kbps = int(str(settings.stem_bitrate).lower().rstrip("k") or 192)
    four_minutes_bytes = kbps * 1000 / 8 * 240
    assert four_minutes_bytes < 10 * 1024 * 1024, (
        f"{kbps}k puts a 4-minute track at "
        f"{four_minutes_bytes/1e6:.1f} MB, over Discord's limit")
