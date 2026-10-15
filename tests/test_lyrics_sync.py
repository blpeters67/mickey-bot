"""Tests for lyrics matching and alignment."""
from __future__ import annotations

import pytest

from core.lyrics_sync import (
    Alignment, LyricLine, SyncedLyrics, TrackQuery, _score, align,
    build_query, clean_title, confidence_ok, parse_lrc,
)


# ── LRC parsing ───────────────────────────────────────────────────────────────

def test_parse_basic():
    lines, off = parse_lrc("[00:12.50]hello\n[01:05.00]world")
    assert off == 0.0
    assert [(l.time, l.text) for l in lines] == [(12.5, "hello"), (65.0, "world")]


def test_parse_multiple_timestamps_per_line():
    """A repeated chorus is written once with several timestamps."""
    lines, _ = parse_lrc("[00:30.00][01:30.00][02:30.00]chorus")
    assert [l.time for l in lines] == [30.0, 90.0, 150.0]
    assert all(l.text == "chorus" for l in lines)


def test_parse_without_decimals():
    lines, _ = parse_lrc("[01:07]no fraction")
    assert lines[0].time == pytest.approx(67.0)


def test_parse_colon_centiseconds():
    lines, _ = parse_lrc("[01:07:50]colon form")
    assert lines[0].time == pytest.approx(67.5)


def test_parse_skips_metadata():
    lines, _ = parse_lrc("[ar:Artist]\n[ti:Title]\n[00:01.00]real")
    assert len(lines) == 1 and lines[0].text == "real"


def test_parse_offset_tag_shifts_earlier():
    """LRC convention: a positive offset makes lyrics appear earlier."""
    lines, off = parse_lrc("[offset:+500]\n[00:10.00]line")
    assert off == pytest.approx(0.5)
    assert lines[0].time == pytest.approx(10.0)   # raw; caller applies offset


def test_parse_output_is_sorted():
    lines, _ = parse_lrc("[02:00.00]second\n[00:10.00]first")
    assert [l.text for l in lines] == ["first", "second"]


def test_parse_handles_junk():
    lines, _ = parse_lrc("not a lyric line\n\n[00:05.00]good\n[bad]")
    assert [l.text for l in lines] == ["good"]


def test_parse_empty():
    assert parse_lrc("") == ([], 0.0)
    assert parse_lrc(None) == ([], 0.0)


# ── title cleaning ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("Song Name (Official Video)", "Song Name"),
    ("Song Name [Official Music Video]", "Song Name"),
    ("Song Name (Lyrics)", "Song Name"),
    ("Song Name (2011 Remaster)", "Song Name"),
    ("Song Name | Official Audio", "Song Name"),
    ("Song Name (HD)", "Song Name"),
])
def test_clean_title_strips_upload_noise(raw, expect):
    assert clean_title(raw) == expect


@pytest.mark.parametrize("raw", [
    "Song Name (Live at Wembley)",
    "Song Name (Extended Version)",
    "Song Name (Acoustic)",
    "Song Name (Remix)",
    "Song Name (Sped Up)",
])
def test_clean_title_preserves_version_markers(raw):
    """These change the timing, so stripping them causes the exact desync bug
    this work exists to fix."""
    assert clean_title(raw) == raw


# ── query building ────────────────────────────────────────────────────────────

def test_build_query_prefers_structured_fields():
    q = build_query({
        "title": "Whatever The Uploader Called It (Official Video)",
        "artist": "Real Artist", "track": "Real Track",
        "duration": 210, "webpage_url": "https://y/watch?v=abc",
    })
    assert (q.artist, q.title) == ("Real Artist", "Real Track")
    assert q.duration == 210
    assert q.source_id == "https://y/watch?v=abc"


def test_build_query_splits_dashed_title():
    q = build_query({"title": "Artist Name - Song Title (Official Video)"})
    assert (q.artist, q.title) == ("Artist Name", "Song Title")


def test_build_query_handles_en_dash():
    q = build_query({"title": "Artist – Song"})
    assert (q.artist, q.title) == ("Artist", "Song")


def test_build_query_uses_topic_channel():
    q = build_query({"title": "Song Title", "uploader": "Some Artist - Topic"})
    assert q.artist == "Some Artist"


def test_build_query_strips_featured_artists():
    q = build_query({"title": "Artist - Song (feat. Someone Else)"})
    assert "feat" not in q.title.lower()
    assert q.title.strip() == "Song"


def test_build_query_no_dash():
    q = build_query({"title": "Just A Title"})
    assert q.title == "Just A Title" and q.artist == ""


# ── candidate scoring ─────────────────────────────────────────────────────────

def _cand(**kw):
    base = {"trackName": "Song", "artistName": "Artist", "duration": 200,
            "syncedLyrics": "[00:01.00]x"}
    base.update(kw)
    return base


def test_exact_duration_scores_highest():
    q = TrackQuery(title="Song", artist="Artist", duration=200)
    good, _ = _score(_cand(), q)
    off, _ = _score(_cand(duration=214), q)
    assert good > off
    assert good > 0.9


def test_wildly_wrong_duration_is_rejected():
    q = TrackQuery(title="Song", artist="Artist", duration=200)
    score, note = _score(_cand(duration=400), q)
    assert score < 0.55
    assert "length differs" in note


def test_version_mismatch_penalised():
    """A live recording must not win against a studio query even when the
    title words and the length happen to agree."""
    q = TrackQuery(title="Song", artist="Artist", duration=200)
    studio, _ = _score(_cand(trackName="Song"), q)
    live, note = _score(_cand(trackName="Song (Live)"), q)
    assert live < studio
    assert "version mismatch" in note


def test_instrumental_heavily_penalised():
    q = TrackQuery(title="Song", artist="Artist", duration=200)
    score, note = _score(_cand(instrumental=True), q)
    assert score < 0.4
    assert "instrumental" in note


def test_unsynced_candidate_ranked_below_synced():
    q = TrackQuery(title="Song", artist="Artist", duration=200)
    synced, _ = _score(_cand(), q)
    plain, _ = _score(_cand(syncedLyrics=""), q)
    assert plain < synced


# ── alignment ─────────────────────────────────────────────────────────────────

_DEFAULT_LINES = object()


def _lyr(lrc_duration=200.0, confidence=0.9, lines=_DEFAULT_LINES):
    if lines is _DEFAULT_LINES:
        lines = [LyricLine(10.0, "a"), LyricLine(20.0, "b")]
    return SyncedLyrics(
        lines=lines,
        plain="a\nb", track_name="Song", artist_name="Artist",
        lrc_duration=lrc_duration, confidence=confidence,
    )


def test_learned_offset_wins_over_everything():
    a = align(_lyr(), TrackQuery(title="Song", duration=260),
              leading_silence=3.0, learned_offset=-1.25)
    assert a.offset == pytest.approx(-1.25)
    assert a.scale == 1.0
    assert a.learned


def test_leading_silence_becomes_offset():
    a = align(_lyr(200.0), TrackQuery(title="Song", duration=203),
              leading_silence=2.8)
    assert a.offset == pytest.approx(2.8)
    assert a.scale == 1.0
    assert any("silence" in r for r in a.reasons)


def test_tempo_shifted_upload_gets_a_scale():
    """A 3% faster re-upload: the fix is multiplicative, not an offset."""
    a = align(_lyr(200.0), TrackQuery(title="Song", duration=194.0),
              leading_silence=0.0)
    assert a.scale == pytest.approx(194.0 / 200.0, rel=1e-6)
    assert any("faster" in r for r in a.reasons)


def test_measured_silence_beats_tempo_hypothesis():
    """When we've *observed* front padding that explains the length gap, don't
    also invent a tempo difference."""
    a = align(_lyr(200.0), TrackQuery(title="Song", duration=206.0),
              leading_silence=5.5)
    assert a.scale == 1.0
    assert a.offset == pytest.approx(5.5)


def test_identical_durations_need_no_correction():
    a = align(_lyr(200.0), TrackQuery(title="Song", duration=200.5))
    assert a.offset == 0.0 and a.scale == 1.0


def test_huge_length_difference_warns():
    a = align(_lyr(200.0), TrackQuery(title="Song", duration=340.0))
    assert any("different" in r for r in a.reasons)


def test_alignment_apply_transforms_times():
    a = Alignment(offset=2.0, scale=1.01)
    out = a.apply([LyricLine(100.0, "x")])
    assert out[0].time == pytest.approx(103.0)
    assert out[0].text == "x"


def test_alignment_describe():
    assert "no correction" in Alignment().describe()
    assert "+1.50s" in Alignment(offset=1.5).describe()
    assert "×1.0300" in Alignment(scale=1.03).describe()


# ── confidence gate ───────────────────────────────────────────────────────────

def test_confidence_ok_happy_path():
    assert confidence_ok(_lyr(200.0), TrackQuery(title="Song", duration=200))


def test_confidence_rejects_low_score():
    assert not confidence_ok(_lyr(200.0, confidence=0.2),
                             TrackQuery(title="Song", duration=200))


def test_confidence_rejects_wrong_length():
    """The extended mix must fall back to a static dump rather than pretending
    to scroll in time."""
    assert not confidence_ok(_lyr(200.0), TrackQuery(title="Song", duration=420))


def test_confidence_rejects_unsynced():
    assert not confidence_ok(_lyr(200.0, lines=[]),
                             TrackQuery(title="Song", duration=200))
