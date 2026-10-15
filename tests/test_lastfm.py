"""Unit tests for the pure Last.fm helpers: API-signature construction (must
match the spec exactly or Last.fm rejects the call) and the artist/title
extraction used when scrobbling."""
from __future__ import annotations

import hashlib

from core import lastfm
from cogs.lastfm import _artist_title


def test_sign_sorts_keys_and_excludes_format():
    params = {
        "method": "auth.getToken",
        "api_key": "K",
        "format": "json",     # must be excluded from the signature
        "b": "2",
        "a": "1",
    }
    # sorted keys minus 'format': a, api_key, b, method
    raw = "a1api_keyKb2methodauth.getToken"
    expected = hashlib.md5((raw + lastfm.settings.lastfm_secret).encode()).hexdigest()
    assert lastfm._sign(params) == expected


def test_signed_attaches_api_key_sig_and_format():
    out = lastfm._signed({"method": "auth.getToken"})
    assert out["format"] == "json"
    assert "api_sig" in out
    assert out["api_key"] == lastfm.settings.lastfm_key


def test_artist_title_prefers_tags():
    assert _artist_title({"artist": "Daft Punk", "title": "whatever"}) == \
        ("Daft Punk", "whatever")
    assert _artist_title(
        {"artist": "Daft Punk", "song_title": "One More Time", "title": "x"}
    ) == ("Daft Punk", "One More Time")


def test_artist_title_falls_back_to_dash_split():
    assert _artist_title({"title": "Rick Astley - Never Gonna Give You Up"}) == \
        ("Rick Astley", "Never Gonna Give You Up")


def test_artist_title_gives_up_without_artist():
    artist, title = _artist_title({"title": "just a clip no dash"})
    assert artist is None
    assert title == "just a clip no dash"
