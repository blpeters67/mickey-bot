"""
Audio + track-fetching layer.

Key design decisions, all of which fix bugs from the original bot.py:

1. **Lazy stream-URL resolution.** yt-dlp stream URLs expire (~6 hours). The
   old code resolved them at queue-time, so a track sitting in a long queue
   would silently fail. We resolve right before `vc.play(...)` instead.

2. **Lazy Spotify resolution.** Spotify gives us only metadata; the actual
   audio comes from a yt-dlp YouTube search. The stub holds the search query;
   the playback engine does the search at play-time.

3. **Trust extract_flat for playlists.** The original did extract_flat *and*
   a per-entry full extract, which made a 100-track YouTube playlist take
   minutes to queue. We trust the flat metadata and resolve lazily.
"""
from __future__ import annotations

import asyncio
import calendar
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit

import aiohttp
import yt_dlp

from . import audiocom
from .config import settings
from .formatters import redact_secrets
from .logging_setup import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# yt-dlp options
# ──────────────────────────────────────────────────────────────────────────────

def ytdlp_cookie_opts() -> dict[str, Any]:
    """Authentication options for yt-dlp, if the operator configured cookies.

    Spread (`**ytdlp_cookie_opts()`) into any YoutubeDL opts dict so login-gated
    sources (Instagram posts, private/age-restricted videos) work. Two mutually
    exclusive sources — a cookies.txt wins over browser extraction:

        settings.ytdlp_cookiefile           → {"cookiefile": path}
        settings.ytdlp_cookies_from_browser → {"cookiesfrombrowser": (browser,)}

    Returns {} when neither is set (default: public content only).
    """
    if settings.ytdlp_cookiefile:
        return {"cookiefile": settings.ytdlp_cookiefile}
    if settings.ytdlp_cookies_from_browser:
        # yt-dlp wants a tuple; accept "firefox" or "firefox:profile".
        browser, _, profile = settings.ytdlp_cookies_from_browser.partition(":")
        spec = (browser.strip(), profile.strip() or None, None, None)
        return {"cookiesfrombrowser": spec}
    return {}


# Resolve once at import — cookie config doesn't change at runtime.
_COOKIE_OPTS = ytdlp_cookie_opts()
if _COOKIE_OPTS:
    log.info("yt-dlp authentication enabled (%s)", next(iter(_COOKIE_OPTS)))


class _YtDlpLog:
    """yt-dlp's own warnings, into the bot's log.

    They used to be silenced (quiet + no_warnings) — and with them the
    likeliest explanation for on-and-off YouTube 403s. "n challenge solving
    failed: Some formats may be missing" means yt-dlp couldn't run YouTube's
    JavaScript challenge (no JS runtime, or no solver scripts), so it worked
    with fewer formats than it should. Each kind of warning is logged at most
    once per half hour; errors are raised to (and reported by) the caller."""

    _EVERY = 1800.0
    _VIDEO_PREFIX = re.compile(r"^\[[^\]]+\]\s*[\w-]{6,}:\s*")   # "[youtube] <id>: "

    def __init__(self):
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def debug(self, msg) -> None:
        pass

    info = debug

    def warning(self, msg) -> None:
        text = redact_secrets(str(msg))
        key = self._VIDEO_PREFIX.sub("", text)[:160]
        now = time.monotonic()
        with self._lock:
            if now - self._last.get(key, float("-inf")) < self._EVERY:
                return
            if len(self._last) > 300:
                self._last.clear()
            self._last[key] = now
        log.warning("yt-dlp: %s", text)

    def error(self, msg) -> None:
        log.debug("yt-dlp error: %s", redact_secrets(str(msg)))


_YTDLP_LOG = _YtDlpLog()


def _runtime_candidates(name: str) -> list[str]:
    """Where an installer puts `name` when it may not be on our PATH.

    Deno's installer (and its release zip) live in ~/.deno/bin, and it only
    edits the *next* shell's PATH. A bot already running in screen/tmux never
    sees that, and `restart` (execv) keeps the old environment — so without
    this, installing Deno would also need a fresh terminal."""
    if name != "deno":
        return []
    exe = "deno.exe" if os.name == "nt" else "deno"
    out = []
    if os.environ.get("DENO_INSTALL"):
        out.append(os.path.join(os.environ["DENO_INSTALL"], "bin", exe))
    out.append(os.path.join(os.path.expanduser("~"), ".deno", "bin", exe))
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):      # winget's shims
        out.append(os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "WinGet",
                                "Links", exe))
    return out


def _js_runtimes() -> dict[str, dict[str, str]]:
    """yt-dlp's `js_runtimes`: Deno, then Node 22+. A runtime on PATH is left
    for yt-dlp to find; one that isn't is pointed at where its installer put
    it (see `_runtime_candidates`)."""
    out: dict[str, dict[str, str]] = {}
    for name in ("deno", "node"):
        cfg: dict[str, str] = {}
        if shutil.which(name) is None:
            for cand in _runtime_candidates(name):
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    cfg = {"path": cand}
                    break
        out[name] = cfg
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Shared yt-dlp options
#
# EVERY yt-dlp call in this project must start from `ydl_opts()`. They used to
# be hand-rolled per call site, and the copies drifted: `source_address` lived
# only here, so playback forced IPv4 while the download cog quietly used IPv6.
# Google blocks datacenter IPv6 ranges far harder than IPv4, so `play` worked
# and `download` returned 403 Forbidden on the same video, on the same host,
# at the same moment. Diagnosing that took a while precisely because the two
# paths looked identical from the outside.
#
# tests/test_ydl_opts.py fails if a new call site skips this.
# ──────────────────────────────────────────────────────────────────────────────

#: Networking + authentication. Everything here is about *reaching* the site,
#: so it applies equally to metadata probes and real downloads.
YDL_NET: dict[str, Any] = {
    "quiet":            True,
    "no_warnings":      True,
    "default_search":   "ytsearch",
    # Binding the socket to the IPv4 wildcard forces IPv4. Load-bearing: on a
    # dual-stack VPS, YouTube 403s a lot of datacenter IPv6 space while the
    # same request over IPv4 succeeds. Set YTDLP_SOURCE_ADDRESS="" to disable
    # if your host's IPv4 is the blocked one instead.
    **({"source_address": settings.ytdlp_source_address}
       if settings.ytdlp_source_address else {}),
    "socket_timeout":     60,
    "retries":             3,
    "fragment_retries":    3,
    # YouTube hides its audio links behind a JavaScript challenge. yt-dlp
    # 2025.11+ solves it with an external runtime: Deno (yt-dlp's default) or
    # Node 22+ — whichever the host has (Deno wins if both). The solver
    # scripts ship in the `yt-dlp-ejs` package, which `yt-dlp[default]` in
    # requirements.txt installs. Without both, yt-dlp works with fewer formats
    # (a likely cause of on-and-off 403s). `ytdiag` / `--preflight` check it.
    "js_runtimes":      _js_runtimes(),
    # yt-dlp's warnings into our log (deduplicated) instead of nowhere.
    "logger":           _YTDLP_LOG,
    **_COOKIE_OPTS,
}


def ydl_opts(**overrides: Any) -> dict[str, Any]:
    """Options for a yt-dlp call, with the shared networking/auth baked in.

    Always build option dicts through this rather than from a literal — see
    the note above for what a drifted copy costs.
    """
    return {**YDL_NET, **overrides}


YDL_BASE: dict[str, Any] = {
    **YDL_NET,
    "format":           "bestaudio/best",
    "noplaylist":       True,
    "skip_download":    True,
}

YDL_FLAT: dict[str, Any] = {
    **YDL_BASE,
    "noplaylist":   False,
    "extract_flat": "in_playlist",
}


# ──────────────────────────────────────────────────────────────────────────────
# URL classification
# ──────────────────────────────────────────────────────────────────────────────

def is_discord_cdn(url: str) -> bool:
    return "cdn.discordapp.com" in url or "media.discordapp.net" in url

def is_spotify_url(url: str) -> bool:
    return "open.spotify.com" in url

def is_spotify_playlist(url: str) -> bool:
    return is_spotify_url(url) and ("/playlist/" in url or "/album/" in url)

def is_audiocom_url(url: str) -> bool:
    """Any audio.com link — a track, a collection, or a page that's neither
    (which is refused with a message rather than handed to yt-dlp, whose only
    answer is the site's browser check)."""
    return audiocom.parse(url) is not None

def is_any_playlist(url: str) -> bool:
    if is_spotify_playlist(url):
        return True
    if "list=" in url and ("youtube.com" in url or "youtu.be" in url):
        return True
    if "soundcloud.com" in url and "/sets/" in url:
        return True
    if "bandcamp.com" in url and ("/album/" in url or "/music" in url):
        return True
    link = audiocom.parse(url)
    if link is not None and link.kind == "collection":
        return True
    return False


def playlist_source_label(url: str) -> str:
    if is_audiocom_url(url):      return "audio.com"
    if is_spotify_url(url):       return "Spotify"
    if "soundcloud.com" in url:   return "SoundCloud"
    if "bandcamp.com" in url:     return "Bandcamp"
    if "youtube.com" in url or "youtu.be" in url: return "YouTube"
    return "playlist"


_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/)|youtu\.be/)([\w-]{11})"
)

def youtube_video_id(url: str | None) -> str | None:
    """Extract the 11-char video id from a YouTube URL, or None."""
    if not url:
        return None
    m = _YT_ID_RE.search(url)
    return m.group(1) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Spotify client (lazy)
# ──────────────────────────────────────────────────────────────────────────────

_spotify_client: Any | None = None
_spotify_ready: bool         = False

def init_spotify() -> bool:
    """Returns True if Spotify is usable. Logs the reason if not."""
    global _spotify_client, _spotify_ready
    if not settings.spotify_enabled:
        log.info("Spotify credentials not set — Spotify support disabled")
        return False
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        _spotify_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=settings.spotify_id,
                client_secret=settings.spotify_secret,
            )
        )
        _spotify_client.search("test", limit=1)   # auth round-trip
        _spotify_ready = True
        log.info("Spotify client ready")
        return True
    except Exception as e:
        log.warning("Spotify init failed: %s", e)
        _spotify_client = None
        _spotify_ready  = False
        return False


def spotify_ready() -> bool:
    return _spotify_ready


async def fetch_spotify_tracks(url: str) -> list[dict[str, Any]]:
    """Resolve a Spotify track / album / playlist URL to track stubs."""
    if not _spotify_ready:
        raise RuntimeError("Spotify client not initialised")

    def _fetch():
        from spotipy.exceptions import SpotifyException
        sp     = _spotify_client
        tracks = []

        try:
            if "/track/" in url:
                tid = url.split("/track/")[1].split("?")[0]
                tracks.append(_spotify_track_to_stub(sp.track(tid), url))

            elif "/album/" in url:
                aid   = url.split("/album/")[1].split("?")[0]
                album = sp.album(aid)
                thumb = album["images"][0]["url"] if album["images"] else None
                results = sp.album_tracks(aid)
                items   = list(results["items"])
                while results["next"]:
                    results = sp.next(results)
                    items.extend(results["items"])
                for t in items:
                    tracks.append(_spotify_track_to_stub(t, url, thumb=thumb))

            elif "/playlist/" in url:
                pid = url.split("/playlist/")[1].split("?")[0]
                results = sp.playlist_items(pid)
                items   = list(results["items"])
                while results["next"]:
                    results = sp.next(results)
                    items.extend(results["items"])
                for entry in items:
                    t = entry.get("track")
                    if not t:
                        continue
                    tracks.append(_spotify_track_to_stub(t, url))
        except SpotifyException as e:
            if e.http_status in (403, 404):
                raise ValueError(
                    "That Spotify playlist is private or unavailable. "
                    "Private and Blend playlists require login, which the "
                    "bot doesn't support."
                ) from e
            raise ValueError(f"Spotify error: {e.msg}") from e
        return tracks

    return await asyncio.to_thread(_fetch)


def _spotify_track_to_stub(t: dict, source_url: str, *, thumb=None) -> dict:
    artists = ", ".join(a["name"] for a in t["artists"])
    title   = f"{artists} - {t['name']}"
    if thumb is None and t.get("album", {}).get("images"):
        thumb = t["album"]["images"][0]["url"]
    return {
        "title":          title,
        "stream_url":     None,                 # resolved at play-time
        "webpage_url":    t.get("external_urls", {}).get("spotify", source_url),
        "duration":       t["duration_ms"] // 1000,
        "thumbnail":      thumb,
        "requester":      None,
        "_needs_resolve": True,
        "_search_query":  title,
    }


# ──────────────────────────────────────────────────────────────────────────────
# audio.com — through its own API (core/audiocom.py), never yt-dlp: there's no
# extractor for it, and its pages answer anything but a browser with a 403.
# ──────────────────────────────────────────────────────────────────────────────

async def _audiocom_resolve(link: audiocom.Link) -> audiocom.Track:
    """The track a link plays. A collection link given as one song plays its
    first track, as yt-dlp's first entry would for any other playlist."""
    if link.kind == "collection":
        entries = await audiocom.fetch_collection(link, pages=1)
        first = audiocom.parse(entries[0].page_url) if entries else None
        if first is None:
            raise audiocom.AudioComError("That audio.com collection is empty.")
        link = first
    return await audiocom.fetch_track(link)


def _apply_audiocom(track: dict[str, Any], t: audiocom.Track, *, retitle: bool = False) -> None:
    """A fetched audio.com track onto a queue entry: its stream (and when the
    signed link expires), and what it is — the uploader, and for music the
    artist and song, which play stats, the quiz and Last.fm read."""
    track["stream_url"]       = t.stream_url
    track["_http_headers"]    = {}
    track["_stream_duration"] = t.duration
    track["_expires_at"]      = _url_expiry(t.stream_url)
    track["_resolved_at"]     = time.time()
    track["webpage_url"]      = t.page_url
    if retitle or not track.get("title"):
        track["title"] = t.display_title
    if not track.get("duration"):
        track["duration"] = t.duration
    if not track.get("thumbnail"):
        track["thumbnail"] = t.image
    if t.author:
        track["uploader"] = t.author
    if t.is_music and t.author:
        track["artist"] = t.author
        if t.title:
            track["song_title"] = t.title
    track["is_music_meta"]  = t.is_music
    track["_needs_resolve"] = False


def _audiocom_stub(t: audiocom.Track) -> dict[str, Any]:
    """A collection entry for the queue, fetched (through the API) when it plays."""
    return {
        "title":          t.display_title,
        "stream_url":     None,
        "webpage_url":    t.page_url,
        "duration":       t.duration,
        "thumbnail":      t.image,
        "requester":      None,
        "_needs_resolve": True,
        "_search_query":  t.page_url,
    }


async def fetch_audiocom_track(link: audiocom.Link) -> dict[str, Any]:
    """An audio.com link as a ready-to-play track (raises ValueError with a
    message for the user when it can't be played)."""
    t = await _audiocom_resolve(link)
    track: dict[str, Any] = {
        "title":          t.display_title,
        "stream_url":     None,
        "webpage_url":    t.page_url,
        "duration":       t.duration,
        "thumbnail":      t.image,
        "requester":      None,
        "_needs_resolve": False,
        "_search_query":  None,
    }
    _apply_audiocom(track, t)
    return track


# ──────────────────────────────────────────────────────────────────────────────
# yt-dlp track / playlist fetching
# ──────────────────────────────────────────────────────────────────────────────

async def get_ffprobe_duration(url: str) -> int:
    def _probe():
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", url],
                capture_output=True, text=True, timeout=10,
            )
            return int(float(json.loads(r.stdout)["format"].get("duration", 0)))
        except Exception:
            return 0
    return await asyncio.to_thread(_probe)


async def fetch_discord_track(url: str) -> dict[str, Any]:
    """A direct Discord-CDN audio link. Stream URL is the URL itself."""
    filename = url.split("?")[0].split("/")[-1]
    title    = os.path.splitext(filename)[0] or "Discord Upload"
    duration = await get_ffprobe_duration(url)
    return {
        "title":          title,
        "stream_url":     url,
        "webpage_url":    url,
        "duration":       duration,
        "thumbnail":      None,
        "requester":      None,
        "_needs_resolve": False,
        "_search_query":  None,
        "_resolved_at":   time.time(),
    }


def _is_transient_network_error(exc: BaseException) -> bool:
    """Best-effort detection of one-off network blips worth retrying.

    yt-dlp wraps underlying errors as DownloadError, but the str() of the
    exception preserves the original message. We sniff for the usual
    suspects: connection-aborted, RemoteDisconnected, connection reset,
    timed out, network unreachable. False positives just cause an extra
    retry, which is cheap.
    """
    s = str(exc).lower()
    needles = (
        "remotedisconnected", "connection aborted", "connection reset",
        "timed out", "timeout", "network is unreachable",
        "temporary failure", "name or service not known",
        "incompleteread", "ssl", "broken pipe",
        # 403 on a *media* fetch is usually a signed URL that expired or was
        # invalidated, not a permissions problem — YouTube hands these out
        # freely and rejects them minutes later. Retrying re-extracts, which
        # mints a fresh URL, so it genuinely recovers. A truly forbidden
        # video fails again on the retry and costs one extra request.
        "http error 403", "403: forbidden",
    )
    return any(n in s for n in needles)


def auth_required_hint(exc: BaseException) -> str | None:
    """If a fetch failed because the source needs a login, return a short,
    actionable message; otherwise None.

    Sites like Instagram increasingly gate posts behind an account and yt-dlp
    surfaces that as a long wall of text ("empty media response … use --cookies
    …"). We collapse it to one line and, when no cookies are configured, point
    the operator at the fix.
    """
    s = str(exc).lower()
    needles = (
        "--cookies",
        "cookies-from-browser",
        "empty media response",
        "login required",
        "requested content is not available",
        "sign in to confirm",
        "private video",
        "this video is private",
        "members-only",
        "you need to be logged in",
        "rate-limit reached",
    )
    if not any(n in s for n in needles):
        return None
    msg = ("That post looks private or login-gated, so I can't fetch it "
           "without signing in.")
    if not settings.ytdlp_cookiefile and not settings.ytdlp_cookies_from_browser:
        msg += (" An admin can enable these by adding a cookies.txt "
                "(set YTDLP_COOKIES).")
    return msg


def friendly_error_msg(exc: BaseException) -> str:
    """Translate a fetch exception into something a user can read.

    Raw exception strings like `('Connection aborted.', RemoteDisconnected(...))`
    leak Python internals and don't tell the user what to do. This collapses
    known transient errors into one short, actionable message and clips
    anything else to a reasonable length.
    """
    if _is_transient_network_error(exc):
        return (
            "Couldn't reach the source right now — connection dropped. "
            "Try again in a moment."
        )
    hint = auth_required_hint(exc)
    if hint:
        return hint
    # yt-dlp errors often contain "ERROR:" prefixes and ANSI codes; strip them.
    # Redact before clipping: these go to chat, and an extractor error can
    # quote a signed stream URL or a credential header.
    msg = redact_secrets(str(exc)).strip()
    if msg.startswith("ERROR:"):
        msg = msg[len("ERROR:"):].strip()
    if len(msg) > 200:
        msg = msg[:197] + "…"
    return msg or "Unknown error"


async def _with_retry(fn, *, attempts: int = 5, base_delay: float = 1.0,
                       max_delay: float = 8.0):
    """Run a sync function in a thread, retrying transient failures with
    exponential backoff. Default schedule: 1s, 2s, 4s, 8s, 8s (capped).
    Re-raises the last exception on permanent failure or non-transient errors.

    Bumped from 3→5 attempts after observing that YouTube backend hiccups
    sometimes last 10-20 seconds — 3 retries (max ~7s of waiting) wasn't
    quite enough for the worst stretches.
    """
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            last_exc = e
            if not _is_transient_network_error(e) or attempt == attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            log.warning(
                "Transient network error on attempt %d/%d (%s) — retrying in %.1fs",
                attempt + 1, attempts, type(e).__name__, delay,
            )
            await asyncio.sleep(delay)
    # Unreachable, but keep the type-checker happy.
    if last_exc:
        raise last_exc


async def fetch_track(query: str) -> dict[str, Any] | None:
    """One track: search query, YouTube/SoundCloud URL, Discord CDN, Spotify,
    or audio.com."""
    link = audiocom.parse(query)
    if link is not None:            # its API has its own guard for tests
        return await fetch_audiocom_track(link)
    if BLOCK_REAL_SEARCH:
        raise SearchBlocked(f"real lookup of {query!r} in a test")
    if is_discord_cdn(query):
        return await fetch_discord_track(query)

    if is_spotify_url(query):
        if not _spotify_ready:
            return None
        tracks = await fetch_spotify_tracks(query)
        return tracks[0] if tracks else None

    def _extract():
        with yt_dlp.YoutubeDL({**YDL_BASE, "skip_download": True}) as ydl:
            info = ydl.extract_info(query, download=False)
            if not info:
                return None
            # A search/playlist result is a container with an `entries` list.
            # When the search matched nothing, yt-dlp still returns the
            # container (title = the query, webpage_url = "ytsearch:<query>",
            # no stream url) — building a "track" from *that* is what queued the
            # unplayable ytsearch: stub that then played forever. Treat an empty
            # result as "not found".
            if "entries" in info:
                entries = info.get("entries") or []
                if not entries:
                    return None
                info = entries[0]
            if not info:
                return None
            track = _ytdl_info_to_track(info)
        # Final guard: a real hit has a stream URL, or at least an http(s) page
        # to resolve from. No stream + only a "ytsearch:"/pseudo URL ⇒ not found.
        if not track.get("stream_url") and "://" not in (track.get("webpage_url") or ""):
            return None
        return track
    return await _with_retry(_extract)


async def fetch_info(url: str) -> dict[str, Any] | None:
    """Raw yt-dlp metadata for one URL, without downloading anything.

    `fetch_track` reshapes info into the player's track dict, which drops
    everything the player doesn't use — including `chapters`. This returns the
    original dict for callers that need those extra fields (`-chapters`).
    """
    link = audiocom.parse(url)
    if link is not None:
        # audio.com has no chapters; this is what -chapters needs to say so.
        t = await _audiocom_resolve(link)
        return {"title": t.display_title, "duration": t.duration, "webpage_url": t.page_url,
                "url": t.stream_url, "thumbnail": t.image, "chapters": None}

    def _extract():
        with yt_dlp.YoutubeDL({**YDL_BASE, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            if info and "entries" in info:
                entries = info.get("entries") or []
                info = entries[0] if entries else None
            return info
    return await _with_retry(_extract)


async def fetch_playlist(url: str) -> list[dict[str, Any]]:
    """A whole YouTube/SoundCloud/Bandcamp/Spotify playlist or album, or an
    audio.com collection."""
    if is_spotify_url(url):
        return await fetch_spotify_tracks(url)
    link = audiocom.parse(url)
    if link is not None:
        if link.kind == "track":
            return [await fetch_audiocom_track(link)]
        return [_audiocom_stub(t) for t in await audiocom.fetch_collection(link)]

    def _extract():
        with yt_dlp.YoutubeDL(YDL_FLAT) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = info.get("entries") or [info]
        out: list[dict[str, Any]] = []
        for e in entries:
            if not e:
                continue
            page = e.get("webpage_url") or e.get("url") or ""
            out.append({
                "title":          e.get("title") or "Unknown",
                "stream_url":     None,
                "webpage_url":    page,
                "duration":       e.get("duration") or 0,
                "thumbnail":      e.get("thumbnail"),
                "requester":      None,
                "_needs_resolve": True,
                "_search_query":  page or e.get("title", ""),
            })
        return out
    return await _with_retry(_extract)


def _apply_music_meta(track: dict[str, Any], info: dict) -> None:
    """Capture best-effort music metadata onto a track dict — used by play
    stats, the music quiz, and Last.fm scrobbling.

    yt-dlp populates `artist`/`track` for content it recognises as music
    (YouTube Music, "- Topic" channels, official song uploads); random clips
    and memes almost never have them. The Music *category* is a weaker
    secondary signal. `is_music_meta` is the combined "this is a real song"
    flag the rest of the bot reads."""
    artist = info.get("artist") or info.get("creator")
    if artist:
        track["artist"] = artist
    uploader = info.get("uploader") or info.get("channel")
    if uploader:
        track["uploader"] = uploader
    if info.get("track"):
        track["song_title"] = info.get("track")
    cats = info.get("categories") or []
    track["is_music_meta"] = bool(
        info.get("artist") or info.get("track")
        or any(str(c).lower() == "music" for c in cats)
    )


# Headers FFmpeg must not be handed: connection management and encodings it
# negotiates itself. Everything else the extractor specifies (User-Agent,
# Referer, Origin, Accept-*, Sec-Fetch-*, …) is what the stream URL was issued
# for, and some sites refuse the request without it.
_HOP_BY_HOP = frozenset({
    "host", "content-length", "connection", "range", "accept-encoding",
    "transfer-encoding", "te", "upgrade", "keep-alive", "proxy-connection",
})
_HEADER_NAME = re.compile(r"[A-Za-z0-9-]{1,64}")


def stream_headers(info: dict) -> dict[str, str]:
    """The HTTP headers yt-dlp says the selected stream needs, cleaned for
    FFmpeg: no hop-by-hop headers, nothing with CR/LF/NUL (header injection),
    nothing oversized. With a single selected format, yt-dlp merges that
    format's `http_headers` into the top-level info dict."""
    raw = info.get("http_headers") or {}
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if not _HEADER_NAME.fullmatch(k) or k.lower() in _HOP_BY_HOP:
            continue
        if any(c in v for c in "\r\n\0") or len(v) > 4096:
            continue
        out[k] = v
    return out


def _url_expiry(url: str | None) -> float | None:
    """Unix expiry carried in a signed stream URL, or None if the URL doesn't
    say: `expire=` (YouTube's googlevideo links), or an S3-style signature —
    signed at `X-Amz-Date` (UTC), valid for `X-Amz-Expires` seconds — as on
    audio.com's links."""
    if not url:
        return None
    try:
        query = {k.lower(): v for k, v in parse_qs(urlsplit(url).query).items()}
        if query.get("expire"):
            return float(query["expire"][0])
        signed, lasts = query.get("x-amz-date"), query.get("x-amz-expires")
        if signed and lasts:
            return calendar.timegm(time.strptime(signed[0], "%Y%m%dT%H%M%SZ")) + float(lasts[0])
        return None
    except (ValueError, TypeError, OverflowError):
        return None


def _apply_stream_info(track: dict[str, Any], info: dict) -> None:
    """Everything about the *selected stream* that playback needs, refreshed
    together whenever the stream URL is: headers, its own duration (a Spotify
    stub's duration is Spotify's, not the YouTube upload's), and its expiry."""
    track["stream_url"]      = info.get("url")
    track["_http_headers"]   = stream_headers(info)
    track["_stream_duration"] = info.get("duration") or 0
    track["_expires_at"]     = _url_expiry(info.get("url"))
    track["_resolved_at"]    = time.time()


def _ytdl_info_to_track(info: dict) -> dict[str, Any]:
    track = {
        "title":          info.get("title", "Unknown"),
        "stream_url":     info.get("url"),
        "webpage_url":    info.get("webpage_url") or info.get("url", ""),
        "duration":       info.get("duration", 0) or 0,
        "thumbnail":      info.get("thumbnail"),
        "requester":      None,
        "_needs_resolve": False,
        "_search_query":  None,
        "_resolved_at":   time.time(),
    }
    _apply_stream_info(track, info)
    _apply_music_meta(track, info)
    return track


# How long a resolved stream URL is trusted before play-time re-resolution.
# yt-dlp URLs expire (YouTube ~6h, tighter on other extractors, and they're
# IP-bound) — the old check trusted any existing stream_url FOREVER, so a
# track queued via `-play` that sat deep in a long queue "played" straight
# into a 403 and got skipped. 45 min is conservative: well inside every
# extractor's window, and the refresh costs one yt-dlp call per track start.
STREAM_URL_TTL = 45 * 60


async def resolve_stream_url(track: dict[str, Any], *,
                             force: bool = False) -> dict[str, Any]:
    """Make sure `track["stream_url"]` is a fresh, playable HTTP URL.

    "Fresh" now means resolved within STREAM_URL_TTL — not merely present.
    Tracks fetched at queue time (fetch_track fully resolves) get a free
    pass while young and a re-resolve once stale, which restores the
    resolve-right-before-play design promise for every code path."""
    # `force` bypasses the TTL entirely: a 403 mid-playback means the signed
    # URL was invalidated *early*, so "resolved 3 minutes ago" is exactly the
    # answer we must not trust.
    fresh = (time.time() - track.get("_resolved_at", 0.0)) < STREAM_URL_TTL
    expires = track.get("_expires_at")
    if expires is not None and time.time() > expires - 120:
        fresh = False          # the URL itself says it's (nearly) dead
    if not force and not track.get("_needs_resolve") and track.get("stream_url") and fresh:
        return track

    target = track.get("_search_query") or track.get("webpage_url")
    if not target:
        raise ValueError(f"Track {track.get('title')!r} has no resolvable source")

    link = audiocom.parse(target)
    if link is not None:
        # A bare link queued from a saved playlist is its own title until now.
        placeholder = track.get("title") in (target, track.get("webpage_url"))
        _apply_audiocom(track, await _audiocom_resolve(link), retitle=placeholder)
        return track

    def _resolve():
        with yt_dlp.YoutubeDL({**YDL_BASE, "skip_download": True}) as ydl:
            info = ydl.extract_info(target, download=False)
            if info and "entries" in info:
                entries = info.get("entries") or []
                info = entries[0] if entries else None
            return info

    info = await _with_retry(_resolve)
    # No usable stream ⇒ raise so _play_next skips this track instead of handing
    # ffmpeg an empty input, which (with -reconnect) hangs and "plays" forever.
    if not info or not info.get("url"):
        raise ValueError(f"No playable stream found for {track.get('title')!r}")
    _apply_stream_info(track, info)     # URL + its headers, duration, expiry
    track["webpage_url"]    = info.get("webpage_url") or track["webpage_url"]
    if not track.get("duration"):
        track["duration"]   = info.get("duration") or 0
    if not track.get("thumbnail"):
        track["thumbnail"]  = info.get("thumbnail")
    _apply_music_meta(track, info)      # artist/categories for stats + quiz
    track["_needs_resolve"] = False
    return track


# ──────────────────────────────────────────────────────────────────────────────
# YouTube search (for the -search picker)
# ──────────────────────────────────────────────────────────────────────────────

#: Tests set this (tests/conftest.py): a test that didn't stub the search
#: must never reach the real YouTube.
BLOCK_REAL_SEARCH = False


class SearchBlocked(RuntimeError):
    """A real YouTube lookup in a test that didn't stub it."""


async def search_tracks(query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """Top YouTube results for a query as resolve-on-demand track stubs.

    Uses extract_flat so this is a single cheap request — the chosen result
    gets fully resolved later, the other four cost nothing.
    """
    if BLOCK_REAL_SEARCH:
        raise SearchBlocked(f"real YouTube search for {query!r} in a test")
    def _extract():
        opts = {**YDL_BASE, "noplaylist": False, "extract_flat": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        out: list[dict[str, Any]] = []
        for e in info.get("entries") or []:
            if not e:
                continue
            vid  = e.get("id") or ""
            page = e.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
            if not page:
                continue
            out.append({
                "title":          e.get("title") or "Unknown",
                "stream_url":     None,
                "webpage_url":    page,
                "duration":       e.get("duration") or 0,
                "thumbnail":      None,
                "uploader":       e.get("channel") or e.get("uploader") or "",
                "requester":      None,
                "_needs_resolve": True,
                "_search_query":  page,
            })
        return out
    return await _with_retry(_extract)


# Uploads that are a different take on a song — only wanted when asked for.
_OTHER_TAKE = re.compile(
    r"\b(cover|covered|karaoke|instrumental|reaction|reacts?|reacting|slowed|sped[ -]?up|"
    r"nightcore|8d|bass[ -]?boosted|tutorial|lesson|how to play|remix|live (at|from|in|on|"
    r"session|performance)|acoustic|mashup|parody|teaser|trailer|snippet|interview|"
    r"\d+ hours?|1 hour)\b", re.IGNORECASE)
_OFFICIAL = re.compile(r"official (audio|music video|video|lyric video|visualizer)|"
                       r"provided to youtube", re.IGNORECASE)


def upload_score(entry: dict, title: str, artist: str = "", duration: int = 0) -> float | None:
    """How well a YouTube search result fits a known song (from the music
    catalogue), or None if it's clearly not that song: most of the song's
    title words must be in the upload's title. The artist in the title or
    channel ("Artist - Topic", "ArtistVEVO"), a matching length and an
    official upload count for it; a cover, remix, live take, slowed/sped-up
    edit, karaoke… counts against it unless that's what the title asks for."""
    from . import music_catalog as mc
    name = entry.get("title") or ""
    channel = re.sub(r"\s*-\s*topic$", "", entry.get("uploader") or "", flags=re.IGNORECASE)
    channel = re.sub(r"vevo$", "", channel, flags=re.IGNORECASE)
    have, want = mc.norm(name, keep_brackets=True).split(), mc.norm(title).split()
    if not want or not have:
        return None
    found = sum(1 for w in want if w in have or any(
        abs(len(w) - len(h)) <= 2 and mc._ratio(w, h) >= 0.85 for h in have))
    cover = found / len(want)
    if cover < 0.6:
        return None
    score = cover
    got = entry.get("duration") or 0
    off = abs(got - duration) if (duration and got) else None
    same_length = off is not None and off <= max(12, 0.1 * duration)
    if artist:
        a = mc.norm(artist)
        if a and (a in mc.norm(name) or a in mc.norm(channel)
                  or mc.name_score(artist, channel) >= 0.8):
            score += 0.4
        elif same_length and cover >= 1.0 and len(want) >= 3:
            # Every word of a longer title and the right length, but not
            # their name anywhere (some label channels): weak evidence — it
            # loses to any upload that does name them. A matching length
            # alone never makes a stranger's video the song.
            score -= 0.3
        else:
            return None               # same words, someone else's: "Martha Stewart cooking show"
    takes = {m.group(0).lower() for m in _OTHER_TAKE.finditer(name)}
    if takes - {m.group(0).lower() for m in _OTHER_TAKE.finditer(title)}:
        score -= 0.5
    if same_length:
        score += 0.2
    elif off is not None and off > max(60, 0.5 * duration):
        score -= 0.3
    if _OFFICIAL.search(name) or (entry.get("uploader") or "").lower().endswith("- topic"):
        score += 0.1
    return score


#: How good a search result must be to be taken as the song when the music
#: catalogue couldn't say which song it is (unknown to it, or unreachable):
#: with an artist named, nearly every title word plus their name in the title
#: or channel; without one, every title word.
UPLOAD_ACCEPT_UNKNOWN = 1.25
UPLOAD_ACCEPT_UNKNOWN_NO_ARTIST = 0.95


def rank_uploads(entries: list[dict], title: str, artist: str = "",
                 duration: int = 0) -> list[tuple[float, dict]]:
    """Search results that could be the song, best first, with their scores
    (YouTube's own order breaks near-ties)."""
    out = []
    for rank, e in enumerate(entries):
        s = upload_score(e, title, artist, duration)
        if s is not None:
            out.append((s - 0.03 * rank, rank, e))
    out.sort(key=lambda x: (-x[0], x[1]))
    return [(s, e) for s, _r, e in out]


def pick_upload(entries: list[dict], title: str, artist: str = "",
                duration: int = 0, *, min_score: float = 0.0) -> dict | None:
    """The search result that is the song, or None if none of them is (rather
    than whatever YouTube ranked first: "Yung Gravy - I Write Him's Not
    Travis D's" once played Mr. Clean, and a hymn after that)."""
    ranked = rank_uploads(entries, title, artist, duration)
    if not ranked or ranked[0][0] < min_score:
        return None
    return ranked[0][1]


async def find_upload(query: str, *, title: str, artist: str = "", duration: int = 0,
                      limit: int = 8, min_score: float = 0.0) -> dict | None:
    """A known song (title/artist/length from the catalogue) on YouTube: a
    resolve-on-demand stub of the best matching search result, or None."""
    return pick_upload(await search_tracks(query, limit=limit), title, artist, duration,
                       min_score=min_score)


async def find_upload_checked(query: str, *, title: str, artist: str = "", duration: int = 0,
                              limit: int = 8, min_score: float = 0.0
                              ) -> tuple[dict | None, str]:
    """find_upload, plus a sanitized note of what was considered and why the
    pick (or no pick) — for the voice DJ's decision log."""
    entries = await search_tracks(query, limit=limit)
    ranked = rank_uploads(entries, title, artist, duration)
    if not ranked:
        return None, f"{len(entries)} results, none shows {title!r}" + (
            f" by {artist!r}" if artist else "")
    score, best = ranked[0]
    note = (f"{len(entries)} results, {len(ranked)} plausible; best {best.get('title')!r} "
            f"[{best.get('uploader') or '?'}] {score:.2f}")
    if score < min_score:
        return None, note + f" < {min_score:.2f} needed"
    return best, note


# ──────────────────────────────────────────────────────────────────────────────
# SponsorBlock — crowd-sourced skip segments for YouTube videos
# ──────────────────────────────────────────────────────────────────────────────

# Sponsor reads, self-promo, and like/subscribe begging. Deliberately NOT
# music_offtopic/intro/outro — those are mistagged often enough to cut real
# content on a music bot.
SPONSORBLOCK_CATEGORIES = ["sponsor", "selfpromo", "interaction"]
_SB_API = "https://sponsor.ajay.app/api/skipSegments"

_sb_cache: dict[str, list[tuple[float, float]]] = {}
_SB_CACHE_MAX = 512


async def fetch_sponsor_segments(webpage_url: str | None) -> list[tuple[float, float]]:
    """Skip segments [(start, end), ...] for a YouTube URL, sorted and merged.

    Returns [] for non-YouTube URLs, videos with no submissions, and any
    API failure — playback must never break because SponsorBlock is down.
    """
    vid = youtube_video_id(webpage_url)
    if not vid:
        return []
    if vid in _sb_cache:
        return _sb_cache[vid]

    segments: list[tuple[float, float]] = []
    try:
        params = {"videoID": vid, "categories": json.dumps(SPONSORBLOCK_CATEGORIES)}
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(_SB_API, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw = sorted(
                        (float(d["segment"][0]), float(d["segment"][1]))
                        for d in data if d.get("segment")
                    )
                    # Merge overlapping/adjacent segments.
                    for s, e in raw:
                        if segments and s <= segments[-1][1] + 0.5:
                            segments[-1] = (segments[-1][0], max(segments[-1][1], e))
                        else:
                            segments.append((s, e))
                # 404 = no segments submitted; anything else we just ignore.
    except Exception as e:
        log.debug("SponsorBlock lookup failed for %s: %s", vid, e)
        return []   # don't cache failures — the API may come back

    if len(_sb_cache) >= _SB_CACHE_MAX:
        _sb_cache.pop(next(iter(_sb_cache)))
    _sb_cache[vid] = segments
    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Autoplay — find a related track when the queue runs dry
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_autoplay_track(
    seed: dict[str, Any], *, exclude_urls: set[str],
) -> dict[str, Any] | None:
    """Pick a track related to `seed`, avoiding anything in `exclude_urls`.

    YouTube seeds use the video's RD (radio/mix) playlist — that's what
    powers YouTube's own autoplay. Anything else falls back to a title
    search. Returns a resolve-on-demand stub, or None if nothing suitable.
    """
    seed_url = seed.get("webpage_url") or ""
    vid = youtube_video_id(seed_url)

    def _extract():
        if vid:
            target = f"https://www.youtube.com/watch?v={vid}&list=RD{vid}"
            opts: dict[str, Any] = {**YDL_FLAT, "playlist_items": "1:15"}
        else:
            target = f"ytsearch10:{seed.get('title', '')}"
            opts = {**YDL_BASE, "noplaylist": False, "extract_flat": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=False)
        return info.get("entries") or []

    try:
        entries = await _with_retry(_extract, attempts=2)
    except Exception as e:
        log.warning("Autoplay lookup failed for %r: %s", seed.get("title"), e)
        return None

    candidates: list[dict[str, Any]] = []
    seed_title = (seed.get("title") or "").lower()
    for e in entries:
        if not e:
            continue
        eid  = e.get("id") or ""
        page = e.get("url") or (f"https://www.youtube.com/watch?v={eid}" if eid else "")
        if not page or page == seed_url or (vid and eid == vid):
            continue
        if page in exclude_urls:
            continue
        if (e.get("title") or "").lower() == seed_title:
            continue
        candidates.append({
            "title":          e.get("title") or "Unknown",
            "stream_url":     None,
            "webpage_url":    page,
            "duration":       e.get("duration") or 0,
            "thumbnail":      None,
            "requester":      "Autoplay",
            "_needs_resolve": True,
            "_search_query":  page,
        })

    if not candidates:
        return None
    # Random among the top few — deterministic "always track #2 of the mix"
    # turns autoplay into a loop after a while.
    return random.choice(candidates[:8])


# ──────────────────────────────────────────────────────────────────────────────
# Audio filters — named ffmpeg -af chains for the -filter command.
# No spaces allowed in these strings (discord.py shlex-splits the options).
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_FILTERS: dict[str, str] = {
    "bassboost": "bass=g=12:f=110:w=0.6",
    "nightcore": "asetrate=48000*1.25,aresample=48000",
    "slowed":    "asetrate=48000*0.85,aresample=48000",
    "8d":        "apulsator=hz=0.125",
    "tremolo":   "tremolo=f=4:d=0.8",
    "vibrato":   "vibrato=f=6:d=0.5",
    "karaoke":   "pan=stereo|c0=0.5*c0+-0.5*c1|c1=0.5*c1+-0.5*c0",
}


def speed_filter(multiplier: float) -> str:
    """Build an atempo chain for an arbitrary speed multiplier.

    ffmpeg's atempo only accepts 0.5–2.0 per instance, so out-of-range
    speeds chain several. Caller should clamp to something sane (0.25–4).
    """
    parts: list[str] = []
    x = multiplier
    while x > 2.0:
        parts.append("atempo=2.0")
        x /= 2.0
    while x < 0.5:
        parts.append("atempo=0.5")
        x /= 0.5
    parts.append(f"atempo={x:.4f}")
    return ",".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Audio source factory — builds the FFmpeg pipeline Discord plays from
# ──────────────────────────────────────────────────────────────────────────────

def create_audio_source(stream_url: str, *, volume: float = 1.0,
                        seek: float = 0.0, filter_af: str | None = None,
                        http_headers: dict[str, str] | None = None):
    """Wrap an FFmpeg stream. Imported lazily so tests don't pull discord.py.

    Returns a DuckingSource around a MusicFFmpegAudio: the outer wrapper counts
    real music frames and forwards FFmpeg's error; the inner one keeps a
    bounded tail of FFmpeg's stderr and its exit status, so a failed stream is
    reported as a failure instead of looking like the end of the song."""
    # Never spawn ffmpeg on an empty input: with the -reconnect flags below it
    # would hang retrying forever (the "song plays for infinity" symptom) or
    # raise a cryptic TypeError. Callers resolve the URL first; fail loud here.
    if not stream_url:
        raise ValueError("create_audio_source: no stream URL to play")
    # -rw_timeout: abort if the source stalls (no bytes for 15s). Without it,
    # a connected-but-silent CDN (throttled/expiring URL) blocks the player
    # thread forever — vc.is_playing() stays True with zero audio, which is
    # exactly the "says it's playing but nothing comes out" symptom. With it,
    # ffmpeg errors out, the after-callback fires, and the queue advances.
    before = (
        "-hide_banner "
        "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 "
        "-rw_timeout 15000000 "
        "-analyzeduration 0 -probesize 32"
    )
    if http_headers and stream_url.startswith(("http://", "https://")):
        # The headers the extractor says this URL needs (Referer, User-Agent…).
        # One shlex-quoted argument: discord.py shlex-splits before_options
        # and passes a list to Popen (no shell), so values can't escape it;
        # stream_headers() already dropped anything containing CR/LF.
        blob = "".join(f"{k}: {v}\r\n" for k, v in http_headers.items())
        before += f" -headers {shlex.quote(blob)}"
    if seek > 0:
        # Sub-second precision matters: the playback clock is anchored to this
        # exact value, so truncating to whole seconds put the clock and the
        # audio up to a second apart on every seek — enough to visibly desync
        # synced lyrics.
        before += f" -ss {seek:.3f}"
    options = "-vn -ar 48000 -ac 2"
    if filter_af:
        options += f" -af {filter_af}"
    from .ffmpeg_source import MusicFFmpegAudio
    src = MusicFFmpegAudio(stream_url, before_options=before, options=options)
    # A PCMVolumeTransformer that `-tts` can talk over (music ducks under the
    # speech). With nothing being said it's a plain passthrough.
    from .voice_mix import DuckingSource
    try:
        return DuckingSource(src, volume=volume)
    except Exception:
        src.cleanup()           # don't leave an orphaned ffmpeg running
        raise
