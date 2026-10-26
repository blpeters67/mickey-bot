"""
audio.com tracks and collections, through audio.com's own API.

yt-dlp has no audio.com extractor, and the site's pages sit behind a
Cloudflare browser check that a server can't pass (every page answers 403).
The API behind audio.com's player and embeds answers plain requests:

    /oembed?url=<track page>&format=json       the track's id (in `iframe_url`)
    /audio/<id>                                title, uploader, length, and
                                               signed links to its MP3, M4A
                                               and original upload
    /collections/<id>                          a collection's owner and slug
    /collection/items?username=&slug=&page=    a collection, 20 tracks a page

The signed links last six days from when audio.com signed them. It caches
them, so a fresh answer can carry a link signed days earlier; core/audio.py
reads the expiry off the link (`_url_expiry`) and re-fetches a track from here
like any other stale stream.

Only public tracks and collections can be reached: there's no login, and
nothing here needs a key.
"""
from __future__ import annotations

import asyncio
import re
import string
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .logging_setup import get_logger

log = get_logger(__name__)

API = "https://api.audio.com"
SITE = "https://audio.com"
#: Tests set this: the real API is then refused, while a local stand-in
#: server (tests/test_audiocom.py) still answers.
BLOCK_REAL_HOST = False
#: Seconds per request (it answers in well under one).
TIMEOUT = 10
#: Before the one retry of a request that hit an outage or a dropped connection.
RETRY_DELAY = 0.5
#: A collection is read 20 tracks a page; this bounds a runaway one.
MAX_PAGES = 50
_HEADERS = {"User-Agent": "DiscordMusicBot/2.0 (+audio.com)", "Accept": "application/json"}

MISSING_TRACK = ("I couldn't find that audio.com track — check the link "
                 "(private tracks can't be played).")
MISSING_COLLECTION = ("I couldn't find that audio.com collection — check the link "
                      "(private collections can't be played).")
NOT_PLAYABLE = ("That audio.com track has no playable audio yet — it may still be "
                "processing. Try again later.")
WRONG_PAGE = ("That audio.com link isn't a track or a collection. Paste a track's link "
              "(audio.com/<user>/audio/<name>) or a collection's.")
CHANGED = ("audio.com answered in a way I don't understand — it may have changed "
           "how its site works.")


class AudioComError(ValueError):
    """Something to tell the user: a missing track, nothing playable, an outage."""


class HostBlocked(RuntimeError):
    """A test reached for the real api.audio.com."""


# ──────────────────────────────────────────────────────────────────────────────
# Links
# ──────────────────────────────────────────────────────────────────────────────

_HOSTS = frozenset({"audio.com", "www.audio.com"})
_SCHEMELESS = re.compile(r"^(?:www\.)?audio\.com(?:/|$)", re.IGNORECASE)
_NUMERIC_ID = re.compile(r"\d{1,30}")
_EMBED_TRACK = re.compile(r"/embed/audio/(\d{1,30})(?:[/?#]|$)")
_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


@dataclass(frozen=True)
class Link:
    """An audio.com link: a track, a collection, or another page of the site."""
    kind: str        # "track" | "collection" | "other"
    url: str         # https://audio.com/<path>, without query, fragment or trailing slash
    user: str = ""
    slug: str = ""
    id: str = ""     # an embed link names the id instead of the user and slug


def parse(text: str | None) -> Link | None:
    """The audio.com link `text` is, or None if it isn't one (another site, a
    search, …). The path is lower-cased: audio.com's own links always are,
    and its API refuses anything else — as it does a query string, a
    trailing slash or a www., all of which turn up in pasted links."""
    t = (text or "").strip()
    if not t or any(c.isspace() for c in t):
        return None
    if _SCHEMELESS.match(t):
        t = "https://" + t
    try:
        parts = urlsplit(t)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or host not in _HOSTS:
        return None
    segs = [s.translate(_LOWER) for s in parts.path.split("/") if s]
    url = f"{SITE}/{'/'.join(segs)}"
    if len(segs) == 3:
        a, b, c = segs
        if a == "embed" and _NUMERIC_ID.fullmatch(c):
            if b == "audio":
                return Link("track", url, id=c)
            if b == "collection":
                return Link("collection", url, id=c)
        if b == "audio":
            return Link("track", url, user=a, slug=c)
        if b == "collections":
            return Link("collection", url, user=a, slug=c)
    return Link("other", url)


# ──────────────────────────────────────────────────────────────────────────────
# What the API says about a track
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Track:
    id: str
    title: str          # as uploaded: "Dive"
    author: str         # the uploader's display name: "Firedies"
    page_url: str       # https://audio.com/<user>/audio/<slug>
    stream_url: str     # signed link to play; a collection entry's may be ""
    duration: int       # seconds
    image: str | None
    is_music: bool

    @property
    def display_title(self) -> str:
        """The title the queue shows: "Uploader - Title", like a Spotify entry,
        unless the title already names them ("Top (prod.by Yellxw Mxtey)")."""
        if not self.title:
            return self.author or "audio.com track"
        if self.author and self.author.casefold() not in self.title.casefold():
            return f"{self.author} - {self.title}"
        return self.title


def _text(v: Any, limit: int = 200) -> str:
    return v.strip()[:limit] if isinstance(v, str) else ""


def _http(v: Any) -> bool:
    return isinstance(v, str) and v.startswith(("https://", "http://"))


def _int(v: Any) -> int:
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def _dicts(v: Any) -> list[dict]:
    return [d for d in v if isinstance(d, dict)] if isinstance(v, list) else []


def _seconds(ms: int) -> int:
    return int(round(ms / 1000))


def _is_music(doc: dict) -> bool:
    """Uploaded as music: the Music category (4), or music:* tags."""
    if str(doc.get("category_name") or "").casefold() == "music":
        return True
    if str(doc.get("category_id", doc.get("category"))) == "4":
        return True
    for tag in doc.get("cross_link_tags") or []:
        tid = tag.get("id") if isinstance(tag, dict) else tag
        if isinstance(tid, str) and tid.startswith("music:"):
            return True
    return False


def _rank(transcoding: dict) -> int:
    fmt = str(transcoding.get("format") or "").lower()
    kind = str(transcoding.get("type") or "").lower()
    if fmt == "mp3" or kind == "audio/mpeg":
        return 0
    if fmt == "m4a" or str(transcoding.get("codec") or "").lower() == "aac":
        return 1
    return 2


def _best_stream(doc: dict) -> tuple[str, int]:
    """(link, length in ms) of what to play: the MP3, which starts fastest —
    audio.com's M4A files keep their index at the end, so FFmpeg has to fetch
    the tail before the first note (≈2 s against ≈0.85 s, measured) — then the
    M4A, then anything else transcoded, then the original upload."""
    transcoded = sorted((t for t in _dicts(doc.get("transcodings")) if _http(t.get("url"))),
                        key=_rank)
    originals = [s for s in _dicts(doc.get("sources")) if _http(s.get("url"))]
    candidates = transcoded + originals
    if not candidates:
        return "", 0
    best = candidates[0]
    length = _int(best.get("duration")) or max(_int(c.get("duration")) for c in candidates)
    return best["url"], length


_SLUG = re.compile(r"[a-z0-9_-]{1,200}")       # what audio.com's page links use


def _page(user: Any, slug: Any, aid: str, fallback: str = "") -> str:
    """The track's page link, rebuilt from its uploader and slug. It's what a
    replay resolves from, so anything that wouldn't parse back to this track
    gets the embed link instead, which names it by id."""
    if (isinstance(user, str) and _SLUG.fullmatch(user)
            and isinstance(slug, str) and _SLUG.fullmatch(slug)):
        return f"{SITE}/{user}/audio/{slug}"
    return fallback or f"{SITE}/embed/audio/{aid}"


def _track_from_doc(doc: Any, *, fallback_page: str = "") -> Track:
    if not isinstance(doc, dict) or not doc.get("id"):
        raise AudioComError(CHANGED)
    aid = str(doc["id"])
    stream, length = _best_stream(doc)
    if not stream:
        raise AudioComError(NOT_PLAYABLE)
    image = doc.get("image")
    return Track(id=aid, title=_text(doc.get("title")), author=_text(doc.get("author_name")),
                 page_url=_page(doc.get("username"), doc.get("slug"), aid, fallback_page),
                 stream_url=stream, duration=_seconds(length),
                 image=image if _http(image) else None, is_music=_is_music(doc))


def _entry_from_item(item: Any) -> Track | None:
    """A collection entry: enough to queue it (it's re-fetched when it plays)."""
    if not isinstance(item, dict) or not item.get("id"):
        return None
    aid = str(item["id"])
    play = item.get("play") if isinstance(item.get("play"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    image = item.get("image")
    return Track(id=aid, title=_text(item.get("title")), author=_text(item.get("author")),
                 page_url=_page(item.get("username"), item.get("slug"), aid),
                 stream_url=play.get("url") if _http(play.get("url")) else "",
                 duration=_seconds(_int(play.get("duration")) or _int(source.get("duration"))),
                 image=image if _http(image) else None, is_music=_is_music(item))


# ──────────────────────────────────────────────────────────────────────────────
# Talking to the API
# ──────────────────────────────────────────────────────────────────────────────

def _session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(headers=_HEADERS,
                                 timeout=aiohttp.ClientTimeout(total=TIMEOUT))


async def _get(session: aiohttp.ClientSession, path: str, params: dict | None = None, *,
               missing: str) -> Any:
    """GET <API><path> → its JSON. Not found or private raises
    AudioComError(missing); an outage (5xx, 429, a dropped connection, no
    answer) is retried once, then raises AudioComError saying to try again."""
    base = API
    if BLOCK_REAL_HOST and base.startswith("https://api.audio.com"):
        raise HostBlocked(f"real audio.com lookup of {path} in a test")
    trouble = ""
    for attempt in range(2):
        if attempt:
            await asyncio.sleep(RETRY_DELAY)
        try:
            async with session.get(base + path, params=params) as r:
                status = r.status
                try:
                    body = await r.json(content_type=None)
                except ValueError:
                    body = None
        except asyncio.TimeoutError:
            trouble = "didn't answer in time"
            continue
        except aiohttp.ClientError as e:
            log.info("audio.com %s: %s", path, type(e).__name__)
            trouble = "couldn't be reached"
            continue
        # Missing, or private (a refusal that's a real API answer, not a
        # browser check's HTML page).
        if status == 404 or (status in (401, 403) and body is not None):
            raise AudioComError(missing)
        if status >= 500 or status == 429:
            trouble = f"is having trouble (HTTP {status})"
            continue
        if status != 200 or body is None:
            log.warning("audio.com %s answered HTTP %s", path, status)
            raise AudioComError(CHANGED)
        return body
    log.warning("audio.com %s %s", path, trouble)
    raise AudioComError(f"audio.com {trouble} — try again in a moment.")


async def fetch_track(link: Link) -> Track:
    """A track link's details and the link to play. Raises AudioComError."""
    if link.kind != "track":
        raise AudioComError(WRONG_PAGE)
    async with _session() as s:
        aid = link.id
        if not aid:
            data = await _get(s, "/oembed", {"url": link.url, "format": "json"},
                              missing=MISSING_TRACK)
            m = _EMBED_TRACK.search(str(data.get("iframe_url") or "")) if isinstance(data, dict) else None
            if not m:
                raise AudioComError(CHANGED)
            aid = m.group(1)
        doc = await _get(s, f"/audio/{aid}", missing=MISSING_TRACK)
    return _track_from_doc(doc, fallback_page=link.url if link.user else "")


async def fetch_collection(link: Link, *, pages: int | None = None) -> list[Track]:
    """A collection's tracks in its order (at most `pages` pages of 20, default
    MAX_PAGES). Raises AudioComError."""
    if link.kind != "collection":
        raise AudioComError(WRONG_PAGE)
    limit = max(1, pages or MAX_PAGES)
    out: list[Track] = []
    async with _session() as s:
        user, slug = link.user, link.slug
        if not (user and slug):
            meta = await _get(s, f"/collections/{link.id}", missing=MISSING_COLLECTION)
            if not isinstance(meta, dict):
                raise AudioComError(CHANGED)
            user, slug = _text(meta.get("username")), _text(meta.get("slug"))
            if not (user and slug):
                raise AudioComError(CHANGED)
        page, page_count = 1, 1
        while page <= min(page_count, limit):
            data = await _get(s, "/collection/items",
                              {"username": user, "slug": slug, "page": page},
                              missing=MISSING_COLLECTION)
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                raise AudioComError(CHANGED)
            if not data["items"]:
                break
            out.extend(e for e in map(_entry_from_item, data["items"]) if e is not None)
            pagination = data.get("pagination") if isinstance(data.get("pagination"), dict) else {}
            page_count = _int(pagination.get("pages")) or page
            page += 1
    if page_count > limit:
        log.info("audio.com collection %s/%s: read %d of its %d pages", user, slug, limit, page_count)
    return out
