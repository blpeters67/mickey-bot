"""YouTube's stream links only work a few seconds after they're issued.

Since late 2026 YouTube hands out links that answer 403 until a moment set in
the player response; yt-dlp reports it as the format's `available_at` and its
own downloader sleeps until then. The bot handed the link to FFmpeg at once,
so every YouTube song failed to start with "403 Forbidden" — and the retry
fetched another fresh link and hit the same wall. Measured on the server
(2026-11-01): 403 until ~2 s before `available_at`, 206 after.
"""
from __future__ import annotations

import time

import pytest

from core import audio


@pytest.fixture
def slept(monkeypatch):
    waits = []

    async def fake_sleep(s):
        waits.append(s)
    monkeypatch.setattr(audio.asyncio, "sleep", fake_sleep)
    return waits


def _resolving_to(monkeypatch, info):
    async def fake_retry(fn, **kw):
        return info
    monkeypatch.setattr(audio, "_with_retry", fake_retry)


@pytest.mark.asyncio
async def test_a_new_link_is_not_played_before_it_works(monkeypatch, slept):
    _resolving_to(monkeypatch, {"url": "https://gv/media", "webpage_url": "https://y/x",
                                "available_at": time.time() + 4})
    track = {"webpage_url": "https://y/x", "_needs_resolve": True}
    await audio.resolve_stream_url(track)
    assert track["stream_url"] == "https://gv/media"
    assert len(slept) == 1 and 3.0 < slept[0] <= 4.5


@pytest.mark.asyncio
async def test_a_link_fetched_at_queue_time_still_waits_its_turn(monkeypatch, slept):
    """`play` resolves when queueing; the song can start a second later."""
    track = {"webpage_url": "https://y/x", "stream_url": "https://gv/media",
             "_resolved_at": time.time(), "_available_at": time.time() + 2}
    await audio.resolve_stream_url(track)
    assert len(slept) == 1 and 1.0 < slept[0] <= 2.5


@pytest.mark.asyncio
async def test_a_link_that_already_works_is_played_at_once(monkeypatch, slept):
    _resolving_to(monkeypatch, {"url": "https://gv/media", "webpage_url": "https://y/x",
                                "available_at": time.time() - 30})
    await audio.resolve_stream_url({"webpage_url": "https://y/x", "_needs_resolve": True})
    assert slept == []


@pytest.mark.asyncio
async def test_a_silly_wait_is_capped(monkeypatch, slept):
    _resolving_to(monkeypatch, {"url": "https://gv/media", "webpage_url": "https://y/x",
                                "available_at": time.time() + 3600})
    await audio.resolve_stream_url({"webpage_url": "https://y/x", "_needs_resolve": True})
    assert slept and slept[0] <= audio.MAX_AVAILABILITY_WAIT
