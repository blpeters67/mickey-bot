"""DB-backed tests for the new features: play stats / wrapped aggregation, the
quiz pool filter, liked songs, and the multi-wake-word storage.

These exercise core.db against the throwaway SQLite file pinned in conftest.py.
Each test uses its own guild/user ids so they don't interfere. Async functions
are driven with asyncio.run() so no pytest-asyncio dependency is needed.
"""
from __future__ import annotations

import asyncio

import pytest

from core import db


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module", autouse=True)
def _init_db():
    run(db.init_db())


# ── Play stats / wrapped ─────────────────────────────────────────────────────

def test_top_tracks_artists_and_user_stats():
    async def main():
        g = 70001
        # Song A played twice by user 1, Song B once by user 2.
        for _ in range(2):
            await db.record_play(guild_id=g, requester_id=1, url="a",
                                 title="Song A", artist="Artist X",
                                 duration=200, is_music=True)
        await db.record_play(guild_id=g, requester_id=2, url="b",
                             title="Song B", artist="Artist Y",
                             duration=100, is_music=True)

        top = await db.top_tracks(g)
        assert top[0]["url"] == "a"
        assert top[0]["plays"] == 2

        artists = await db.top_artists(g)
        assert artists[0]["artist"] == "Artist X"
        assert artists[0]["plays"] == 2

        user1 = await db.user_top_tracks(g, 1)
        assert user1[0]["plays"] == 2
        totals = await db.user_play_totals(g, 1)
        assert totals["plays"] == 2
        assert totals["seconds"] == 400

        listeners = await db.top_listeners(g)
        assert listeners[0]["user_id"] == 1
        assert listeners[0]["plays"] == 2

    run(main())


def test_window_filtering_excludes_out_of_range():
    async def main():
        g = 70002
        await db.record_play(guild_id=g, requester_id=1, url="x", title="X",
                             artist=None, duration=120, is_music=True)
        # A `since` in the future hides everything just recorded.
        import time
        assert await db.top_tracks(g, since=time.time() + 1000) == []
        # No window → it's there.
        assert len(await db.top_tracks(g)) == 1

    run(main())


def test_top_artists_skips_unknown_artist():
    async def main():
        g = 70003
        await db.record_play(guild_id=g, requester_id=1, url="n", title="No Artist",
                             artist=None, duration=120, is_music=False)
        assert await db.top_artists(g) == []

    run(main())


# ── Quiz pool filter ─────────────────────────────────────────────────────────

def test_quiz_pool_filters_and_excludes():
    async def main():
        g = 70004
        await db.record_play(guild_id=g, requester_id=1, url="song", title="Real Song",
                             artist="A", duration=200, is_music=True)
        await db.record_play(guild_id=g, requester_id=1, url="meme", title="lol meme",
                             artist=None, duration=200, is_music=False)
        await db.record_play(guild_id=g, requester_id=1, url="short", title="Clip",
                             artist="A", duration=20, is_music=True)

        pool = await db.quiz_pool(g)
        urls = {t["url"] for t in pool}
        assert urls == {"song"}          # meme not music, clip too short

        await db.quiz_exclude(g, "song")
        assert await db.quiz_pool(g) == []
        assert await db.quiz_include(g, "song") is True
        assert {t["url"] for t in await db.quiz_pool(g)} == {"song"}

    run(main())


def test_quiz_scores():
    async def main():
        g = 70005
        await db.quiz_add_points(g, 11, 3)
        await db.quiz_add_points(g, 11, 2)
        await db.quiz_add_points(g, 12, 1)
        board = await db.quiz_leaderboard(g)
        assert board[0] == {"user_id": 11, "points": 5}
        assert board[1] == {"user_id": 12, "points": 1}

    run(main())


# ── Liked songs ──────────────────────────────────────────────────────────────

def test_likes_add_dedupe_remove():
    async def main():
        u = 80001
        assert await db.add_like(u, url="l1", title="Liked", artist="A", duration=100)
        # Second add of same url is a no-op (already liked).
        assert await db.add_like(u, url="l1", title="Liked", artist="A", duration=100) is False
        assert await db.is_liked(u, "l1") is True
        assert await db.count_likes(u) == 1

        await db.add_like(u, url="l2", title="Liked 2", artist=None, duration=50)
        rows = await db.list_likes(u)
        assert {r["url"] for r in rows} == {"l1", "l2"}

        assert await db.remove_like(u, "l1") is True
        assert await db.is_liked(u, "l1") is False
        assert await db.count_likes(u) == 1

    run(main())


# ── Multi-wake-word storage ──────────────────────────────────────────────────

def test_wake_word_list_storage():
    async def main():
        g = 90001
        assert await db.get_voice_wakes(g) == []       # none configured
        await db.add_voice_wake(g, "Hey DJ")
        await db.add_voice_wake(g, "computer")
        await db.add_voice_wake(g, "computer")          # dedupe
        assert await db.get_voice_wakes(g) == ["hey dj", "computer"]

        await db.remove_voice_wake(g, "hey dj")
        assert await db.get_voice_wakes(g) == ["computer"]

        await db.set_voice_wakes(g, [])                 # reset to default
        assert await db.get_voice_wakes(g) == []

    run(main())


def test_legacy_single_wake_word_is_read_as_list():
    async def main():
        g = 90002
        # Simulate an old row that stored a bare phrase (pre-JSON migration).
        await db.set_voice_wake(g, "jarvis")
        assert await db.get_voice_wakes(g) == ["jarvis"]

    run(main())
