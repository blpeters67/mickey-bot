"""End-to-end tests for the lrclib fetch path, against a local stub server.

These exercise the part that unit tests can't: real aiohttp requests, real
JSON, real candidate pooling across the `/get` and `/search` endpoints, and the
ranking decisions that determine whether the karaoke display runs at all.

The stub server is stdlib `http.server` on a background thread rather than
`pytest-aiohttp`, so these tests need nothing beyond what requirements.txt
already pins.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

from core import lyrics_sync
from core.lyrics_sync import (
    build_query, clean_title, confidence_ok, fetch_lyrics,
)


LRC_STUDIO = "\n".join(
    f"[{m:02d}:{s:05.2f}]studio line {i}"
    for i, (m, s) in enumerate([(0, 12.0), (0, 30.5), (1, 2.0), (2, 40.0)])
)
LRC_LIVE = "[00:20.00]live line 0\n[01:10.00]live line 1"


def _entry(**kw):
    base = {
        "id": 1, "trackName": "Get Lucky", "artistName": "Daft Punk",
        "albumName": "Random Access Memories", "duration": 369.0,
        "instrumental": False, "plainLyrics": "studio words",
        "syncedLyrics": LRC_STUDIO,
    }
    base.update(kw)
    return base


@pytest.fixture
def stub(monkeypatch):
    """A stand-in lrclib. `state` lets each test choose what it returns."""
    state = {"get": None, "search": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):        # keep pytest output clean
            pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path.endswith("/get"):
                payload = state["get"]
            elif path.endswith("/search"):
                payload = state["search"]
            else:
                payload = None
            if payload is None:
                body, code = b'{"code":404}', 404
            else:
                body, code = json.dumps(payload).encode(), 200
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        lyrics_sync, "LRCLIB_BASE",
        f"http://127.0.0.1:{server.server_address[1]}/api")
    lyrics_sync.clear_cache()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        lyrics_sync.clear_cache()


# ── happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exact_match_produces_synced_lines(stub):
    stub["get"] = _entry()
    q = build_query({"title": "Daft Punk - Get Lucky (Official Audio)",
                     "duration": 369, "webpage_url": "https://y/x"})
    assert (q.artist, q.title) == ("Daft Punk", "Get Lucky")

    lyr = await fetch_lyrics(q)
    assert lyr is not None
    assert lyr.track_name == "Get Lucky"
    assert len(lyr.lines) == 4
    assert lyr.lines[0].time == pytest.approx(12.0)
    assert lyr.lines[-1].time == pytest.approx(160.0)
    assert confidence_ok(lyr, q)


@pytest.mark.asyncio
async def test_falls_back_to_search_when_get_404s(stub):
    stub["get"] = None
    stub["search"] = [_entry()]
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    lyr = await fetch_lyrics(q)
    assert lyr is not None and len(lyr.lines) == 4


@pytest.mark.asyncio
async def test_no_results_returns_none(stub):
    stub["get"] = None
    stub["search"] = []
    q = build_query({"title": "Nonexistent - Nothing", "duration": 100})
    assert await fetch_lyrics(q) is None


# ── ranking ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_correct_duration_beats_wrong_duration(stub):
    """The decisive fix: pick by how long the recording actually is."""
    stub["get"] = None
    stub["search"] = [
        _entry(id=1, trackName="Get Lucky (Radio Edit)", duration=248.0,
               syncedLyrics="[00:05.00]radio"),
        _entry(id=2, duration=369.0),
    ]
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    lyr = await fetch_lyrics(q)
    assert lyr.lines[0].text == "studio line 0"


@pytest.mark.asyncio
async def test_live_upload_does_not_confidently_sync_to_studio(stub):
    """A Live Aid recording is 100s longer than the studio cut. Better to show
    static lyrics than to scroll confidently wrong ones."""
    stub["get"] = None
    stub["search"] = [_entry(trackName="Bohemian Rhapsody", duration=354.0)]
    q = build_query({"title": "Queen - Bohemian Rhapsody (Live Aid 1985)",
                     "duration": 466})
    lyr = await fetch_lyrics(q)
    assert lyr is not None
    assert not confidence_ok(lyr, q), "would have scrolled the wrong timings"


@pytest.mark.asyncio
async def test_instrumental_entry_is_not_preferred(stub):
    stub["get"] = None
    stub["search"] = [
        _entry(id=1, instrumental=True, syncedLyrics="[00:01.00]~"),
        _entry(id=2),
    ]
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    lyr = await fetch_lyrics(q)
    assert lyr.lines[0].text == "studio line 0"


@pytest.mark.asyncio
async def test_synced_entry_preferred_over_plain_only(stub):
    stub["get"] = None
    stub["search"] = [
        _entry(id=1, syncedLyrics="", plainLyrics="just words"),
        _entry(id=2),
    ]
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    lyr = await fetch_lyrics(q)
    assert lyr.has_sync


@pytest.mark.asyncio
async def test_plain_only_still_returned_when_nothing_synced(stub):
    stub["get"] = None
    stub["search"] = [_entry(syncedLyrics="", plainLyrics="just words")]
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    lyr = await fetch_lyrics(q)
    assert not lyr.has_sync
    assert lyr.plain == "just words"
    assert not confidence_ok(lyr, q)


@pytest.mark.asyncio
async def test_duplicate_entries_deduplicated(stub):
    stub["get"] = _entry(id=7)
    stub["search"] = [_entry(id=7), _entry(id=7)]
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    lyr = await fetch_lyrics(q)
    assert len(lyr.lines) == 4


# ── robustness ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_offset_tag_is_applied(stub):
    stub["get"] = _entry(syncedLyrics="[offset:+500]\n[00:10.00]shifted")
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    lyr = await fetch_lyrics(q)
    # +500ms means "show earlier", so 10.0 becomes 9.5.
    assert lyr.lines[0].time == pytest.approx(9.5)


@pytest.mark.asyncio
async def test_server_error_is_survived(stub, monkeypatch):
    monkeypatch.setattr(lyrics_sync, "LRCLIB_BASE", "http://127.0.0.1:1/api")
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    assert await fetch_lyrics(q) is None       # no exception escapes


@pytest.mark.asyncio
async def test_results_are_cached(stub):
    stub["get"] = _entry()
    q = build_query({"title": "Daft Punk - Get Lucky", "duration": 369})
    first = await fetch_lyrics(q)
    stub["get"] = None
    stub["search"] = []
    second = await fetch_lyrics(q)
    assert second is first, "second lookup hit the network instead of the cache"


# ── the regressions that made real titles unmatchable ─────────────────────────

def test_ft_inside_a_word_is_not_treated_as_featuring():
    """`ft` matched inside "Daft Punk" and truncated the artist to "Da"."""
    q = build_query({"title": "Daft Punk - Get Lucky"})
    assert q.artist == "Daft Punk"
    assert q.title == "Get Lucky"


@pytest.mark.parametrize("raw,artist", [
    ("Daft Punk - Get Lucky", "Daft Punk"),
    ("Swift - Style", "Swift"),
    ("Drift - Motion", "Drift"),
    ("Left Boy - Song", "Left Boy"),
])
def test_word_boundary_artists_survive(raw, artist):
    assert build_query({"title": raw}).artist == artist


def test_genuine_featuring_is_still_stripped():
    assert build_query({"title": "Artist - Song feat. Someone"}).title == "Song"
    assert build_query({"title": "Artist - Song (feat. Someone)"}).title == "Song"
    assert build_query({"title": "Artist - Song ft. Someone"}).title == "Song"


def test_with_in_a_title_is_preserved():
    """"with" only means "featuring" inside brackets."""
    assert build_query({"title": "Billy Idol - Dancing With Myself"}).title \
        == "Dancing With Myself"


@pytest.mark.parametrize("raw,expect", [
    ("Take On Me [Remastered in 4K]", "Take On Me"),
    ("Song (2011 Remaster)", "Song"),
    ("Song (Remastered 2009)", "Song"),
    ("Song [Remaster]", "Song"),
])
def test_remaster_tags_fully_stripped(raw, expect):
    """"[Remastered in 4K]" used to survive into the search string and stop the
    track matching anything at all."""
    assert clean_title(raw) == expect
