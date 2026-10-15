"""
Thin async client for the Last.fm scrobbling API.

Only the handful of methods the bot needs: the desktop auth handshake
(getToken → user authorises in a browser → getSession) and the two write
calls (updateNowPlaying, scrobble). Session keys don't expire, so once a user
links their account we store the key and never re-auth.

Everything here is best-effort and never raises into the caller — network blips
or API errors return None / False and are logged. The whole feature is gated on
`settings.lastfm_enabled` (both API key and secret configured); when it's off,
these functions short-circuit.

Docs: https://www.last.fm/api
"""
from __future__ import annotations

import hashlib
import time as _time
from typing import Any

import aiohttp

from .config import settings
from .logging_setup import get_logger

log = get_logger(__name__)

_API_ROOT = "https://ws.audioscrobbler.com/2.0/"
_AUTH_URL = "https://www.last.fm/api/auth/"
_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _sign(params: dict[str, str]) -> str:
    """Last.fm api_sig: md5 of every key+value sorted by key, then the shared
    secret. `format` and `callback` are excluded by spec (we add format after)."""
    raw = "".join(f"{k}{params[k]}" for k in sorted(params)
                  if k not in ("format", "callback"))
    return hashlib.md5((raw + settings.lastfm_secret).encode("utf-8")).hexdigest()


def _signed(params: dict[str, Any]) -> dict[str, str]:
    """Stringify, attach api_key + api_sig + json format."""
    p = {k: str(v) for k, v in params.items() if v is not None}
    p["api_key"] = settings.lastfm_key
    p["api_sig"] = _sign(p)
    p["format"] = "json"
    return p


async def _call(method: str, params: dict[str, Any], *, post: bool) -> dict | None:
    if not settings.lastfm_enabled:
        return None
    body = _signed({**params, "method": method})
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            if post:
                resp = await session.post(_API_ROOT, data=body)
            else:
                resp = await session.get(_API_ROOT, params=body)
            async with resp:
                data = await resp.json(content_type=None)
    except Exception:
        log.exception("Last.fm %s call failed", method)
        return None
    if isinstance(data, dict) and data.get("error"):
        log.warning("Last.fm %s error %s: %s", method,
                    data.get("error"), data.get("message"))
        return None
    return data


# ── Auth handshake ───────────────────────────────────────────────────────────

async def get_token() -> str | None:
    """Request an unauthorised token to start the link flow."""
    data = await _call("auth.getToken", {}, post=False)
    return data.get("token") if data else None


def auth_url(token: str) -> str:
    """The page a user opens to authorise the token for their account."""
    return f"{_AUTH_URL}?api_key={settings.lastfm_key}&token={token}"


async def get_session(token: str) -> dict[str, str] | None:
    """Exchange an authorised token for a permanent session.
    Returns {"key": session_key, "name": username} or None if not yet
    authorised / on error."""
    data = await _call("auth.getSession", {"token": token}, post=False)
    if not data:
        return None
    sess = data.get("session") or {}
    if sess.get("key"):
        return {"key": sess["key"], "name": sess.get("name", "")}
    return None


# ── Write calls ──────────────────────────────────────────────────────────────

async def update_now_playing(
    session_key: str, *, artist: str, track: str,
    album: str | None = None, duration: int | None = None,
) -> bool:
    data = await _call("track.updateNowPlaying", {
        "sk": session_key, "artist": artist, "track": track,
        "album": album, "duration": duration,
    }, post=True)
    return data is not None


async def scrobble(
    session_key: str, *, artist: str, track: str, timestamp: int | None = None,
    album: str | None = None, duration: int | None = None,
) -> bool:
    data = await _call("track.scrobble", {
        "sk": session_key, "artist": artist, "track": track,
        "timestamp": timestamp if timestamp is not None else int(_time.time()),
        "album": album, "duration": duration,
    }, post=True)
    return data is not None
