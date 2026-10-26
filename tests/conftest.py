"""Shared test setup.

Runs before any test module is imported, so it's the right place to pin the
environment that `core.config` reads *once* at import time:

  * a dummy Discord token, so importing core.config never sys.exit()s on a box
    without token.txt / DISCORD_TOKEN, and
  * a throwaway DB_PATH, so DB-backed tests never touch the real bot.db.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("DISCORD_TOKEN", "test-token-not-real")

_TEST_DIR = tempfile.TemporaryDirectory(prefix="musicbot-pytest-")
_TEST_DB = os.path.join(_TEST_DIR.name, "bot.db")
os.environ["DB_PATH"] = _TEST_DB

# The music catalogue (Deezer/iTunes) must never be asked for real by a test:
# with this set, the real hosts answer like a network that's down, so a voice
# "play" falls back to the (mocked) YouTube search as it would offline.
# tests/test_music_catalog.py runs its own local stand-in server instead.
from core import music_catalog as _music_catalog  # noqa: E402

_music_catalog.BLOCK_REAL_HOSTS = True

# Nor YouTube: a test that needs a search stubs core.audio.search_tracks /
# fetch_track; without that, the lookup fails like a network that's down.
from core import audio as _audio  # noqa: E402

_audio.BLOCK_REAL_SEARCH = True

# Nor audio.com's API: tests/test_audiocom.py runs a local stand-in for it.
from core import audiocom as _audiocom  # noqa: E402

_audiocom.BLOCK_REAL_HOST = True
