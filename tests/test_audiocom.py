"""audio.com links: play, queue, re-resolve and download them.

yt-dlp has no audio.com extractor and the site's pages sit behind a browser
check, so the bot goes through audio.com's own API instead (oEmbed → track id
→ track details with signed MP3/M4A links; collection items 20 a page).

Everything here runs against a local stand-in for api.audio.com whose answers
copy the shape of the real ones (fields, nesting, error bodies, pagination
headers) — no test reaches the real site, and yt-dlp is poisoned so a link
that slipped through to it would fail loudly.
"""
from __future__ import annotations

import os
import re
import time
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlsplit

import pytest

from core import audio, audiocom
from core.state import requeue_copy

CDN = "https://s3.example.test/audio.com.audio"
SOURCE_ID = "1879481920426188"


# ── what api.audio.com answers ────────────────────────────────────────────────

def signed(path: str, *, when: float | None = None, sig: str = "f00d") -> str:
    """A signed storage link like the API hands out (valid 6 days from X-Amz-Date)."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if when is None else when))
    return (f"{CDN}/{path}?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD"
            "&X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=EXAMPLEKEY%2F20261026%2Fauto%2Fs3%2Faws4_request"
            f"&X-Amz-Date={stamp}&X-Amz-SignedHeaders=host&X-Amz-Expires=518400"
            f"&X-Amz-Signature={sig}")


_FORMATS = {   # format: (codec, type, bit_rate, duration ms, transcoding id)
    "mp3": ("mp3", "audio/mpeg", 192000, 54096, "1879481924159131"),
    "m4a": ("aac", "audio/m4a", 322521, 54049, "1879481924179705"),
}


def audio_doc(aid: str = "1879481920371808", *, username: str = "firedies",
              author: str = "Firedies", slug: str = "dive", title: str = "Dive",
              category_id: int = 4, tags=("music:electronic", "music:trance"),
              formats=("mp3", "m4a"), source: bool = True, sig: str = "f00d") -> dict:
    """GET /audio/<id>, as the real API answers it."""
    def transcoding(fmt: str) -> dict:
        codec, typ, rate, dur, tid = _FORMATS[fmt]
        return {
            "id": tid, "source_id": SOURCE_ID, "format": fmt, "type": typ,
            "url": signed(f"transcoding/08/18/{aid}-{SOURCE_ID}-{tid}.{fmt}", sig=sig),
            "size": 1298880, "codec": codec, "sample_rate": 48000, "channels": 2,
            "duration": dur, "bit_rate": rate, "date_created": 1792413639,
            "date_updated": 1792413642, "date_uploaded": 1792413642,
            "date_waveform_generated": 1792413642, "md5": "937f879f4cfa6461b9802d95a9da68a1",
            "waveform": [0.2668, 0.5918, 0.5462],
        }
    sources = [{
        "id": SOURCE_ID, "format": "wav", "type": "audio/wav",
        "url": signed(f"source/08/18/{aid}-{SOURCE_ID}.wav", sig=sig),
        "from": "audiocom",
        "download": signed(f"source/08/18/{aid}-{SOURCE_ID}.wav", sig=sig)
                    + "&response-content-disposition=attachment",
        "size": 20874408, "name": f"{author} - {title}.wav", "codec": "pcm_s24le",
        "sample_rate": 48000, "channels": 2, "duration": 54049, "bit_rate": 2304000,
        "date_created": 1792413635, "date_updated": 1792413646, "date_uploaded": 1792413639,
        "md5": "2ea63578e6e3aa08d6887a77fdf73ebf",
    }] if source else []
    return {
        "id": aid, "category_id": category_id, "username": username, "author_name": author,
        "slug": slug, "license": "none", "title": title, "caption": "",
        "description": "wires humming in the dark.", "tags": [t.split(":")[-1] for t in tags],
        "cross_link_tags": [{"id": t, "sort": ""} for t in tags],
        "avatar": "https://audio.com/s3w/audio.com.static/user/avatar/42/00/1879485111150042.jpeg",
        "image": f"https://audio.com/s3w/audio.com.static/audio/image/08/18/{aid}-1.jpeg",
        "is_public": True, "is_downloadable": True, "transcribed": False,
        "date_created": 1792413635, "date_updated": 1792773083,
        "sources": sources,
        "transcodings": [transcoding(f) for f in formats],
        "waveform": signed(f"waveform/08/18/{aid}-{SOURCE_ID}.json", sig=sig),
        "stats": {"followers_count": 1968, "plays_count": 200, "collections_count": 0,
                  "downloads_count": 13, "comments_count": 0, "shares_count": 1,
                  "likes_count": 5},
        "options": {"transcribe": False}, "price": 0,
    }


def item_doc(n: int, *, username: str = "beatmaker", author: str = "Beat Maker") -> dict:
    """One entry of GET /collection/items, as the real API answers it."""
    aid = f"17879165491293{n:02d}"
    play = signed(f"transcoding/68/93/{aid}-1787916551072092.mp3")
    return {
        "id": aid, "user": "1787910145796443", "username": username,
        "scraped_from": "none", "author": author, "category": 4, "category_name": "Music",
        "slug": f"beat-{n}", "license": "none", "title": f"Beat {n}", "description": "",
        "avatar": "https://audio.com/s3w/audio.com.static/user/avatar/32/05/1.jpeg",
        "image": f"https://audio.com/s3w/audio.com.static/audio/image/68/93/{aid}-1.jpeg",
        "listed": True, "downloadable": True, "browsable": True, "transcribed": False,
        "created": 1705090092, "options": {"transcribe": False}, "updated": 1705090253,
        "play": {"duration": 100_000 + n * 1000, "url": play,
                 "waveform": signed(f"waveform/68/93/{aid}.json")},
        "playhq": {"duration": 100_000 + n * 1000, "url": "",
                   "waveform": signed(f"waveform/68/93/{aid}.json")},
        "source": {"id": "1787916551072092", "format": "mp3", "type": "audio/mpeg",
                   "url": signed(f"source/68/93/{aid}.mp3"), "from": "audiocom",
                   "download": signed(f"source/68/93/{aid}.mp3"), "size": 5503727,
                   "name": f"Beat {n}.mp3", "codec": "mp3", "sample_rate": 44100,
                   "channels": 2, "duration": 100_000 + n * 1000, "bit_rate": 320000,
                   "date_created": 1705090094, "date_updated": 1705180394,
                   "date_uploaded": 1705090157, "md5": "313f3f50cf2d94f03c397e127f56dc49"},
        "metadata": None,
        "stats": {"followers_count": 103, "plays_count": 14913, "collections_count": 0,
                  "downloads_count": 485, "comments_count": 7, "shares_count": 101,
                  "likes_count": 53},
        "tags": ["music", "lofi"],
        "cross_link_tags": [{"id": "music:lofi", "sort": ""}], "price": 0,
    }


def not_found(message: str) -> dict:
    return {"name": "Not Found", "message": message, "code": 0, "status": 404}


_OEMBED_PAGE = re.compile(r"^https?://audio\.com/(?:[a-z0-9_-]+/(?:audio|collections)/[a-z0-9_-]+"
                          r"|embed/(?:audio|collection)/\d+)$")


class FakeAudioCom:
    """A local api.audio.com: /oembed, /audio/<id>, /collections/<id> and
    /collection/items, answering (and failing) the way the real one does."""

    PER_PAGE = 20

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.audios: dict[str, dict] = {}
        self.pages: dict[str, str] = {}                        # track page → id
        self.collections: dict[tuple[str, str], list[dict]] = {}
        self.collection_ids: dict[str, tuple[str, str]] = {}   # id → (user, slug)
        self.claimed_pages: int | None = None                  # lie about the page count
        self.trouble: list[int] = []                           # statuses to answer first
        self.private: set[str] = set()                         # ids answering 403
        self.hits: list[tuple[str, dict]] = []

    def add_track(self, doc: dict) -> str:
        page = f"https://audio.com/{doc['username']}/audio/{doc['slug']}"
        self.audios[doc["id"]] = doc
        self.pages[page] = doc["id"]
        return page

    def add_collection(self, user: str, slug: str, items: list[dict], cid: str = "1758071724280935"):
        self.collections[(user, slug)] = items
        self.collection_ids[cid] = (user, slug)
        return f"https://audio.com/{user}/collections/{slug}"

    def oembed_urls(self) -> list[str]:
        return [q.get("url") for p, q in self.hits if p == "/oembed"]

    async def handler(self, request):
        from aiohttp import web
        self.hits.append((request.path, dict(request.query)))
        if self.trouble:
            status = self.trouble.pop(0)
            return web.json_response({"name": "Service Unavailable", "message": "Down for a bit",
                                      "code": 0, "status": status}, status=status)
        path, q = request.path, request.query
        if path == "/oembed":
            url = q.get("url", "")
            if not _OEMBED_PAGE.match(url):
                return web.json_response(not_found("Unknown url type"), status=404)
            m = re.search(r"/embed/audio/(\d+)$", url)
            aid = m.group(1) if m else self.pages.get(url.replace("http://", "https://", 1))
            doc = self.audios.get(aid or "")
            if doc is None:
                return web.json_response(not_found("Resource not found!"), status=404)
            embed = f"https://audio.com/embed/audio/{doc['id']}?theme=image"
            return web.json_response({
                "type": "rich", "version": 1, "title": doc["title"],
                "author_name": doc["author_name"], "author_url": f"https://audio.com/{doc['username']}",
                "provider_name": "audio.com", "provider_url": "https://audio.com/",
                "cache_age": 432000,
                "html": f'<iframe loading="lazy" src="{embed}" width="600" height="204"></iframe>',
                "width": 600, "height": 204, "thumbnail_url": doc["image"],
                "thumbnail_width": 300, "thumbnail_height": 300, "iframe_url": embed})
        m = re.fullmatch(r"/audio/(\d+)", path)
        if m:
            if m.group(1) in self.private:
                return web.json_response({"name": "Forbidden", "message": "Access denied.",
                                          "code": 0, "status": 403}, status=403)
            doc = self.audios.get(m.group(1))
            if doc is None:
                return web.json_response(not_found("Audio not found!"), status=404)
            return web.json_response(doc)
        m = re.fullmatch(r"/collections/(\d+)", path)
        if m:
            owner = self.collection_ids.get(m.group(1))
            if owner is None:
                return web.json_response(not_found("Collection not found!"), status=404)
            return web.json_response({
                "id": m.group(1), "release_id": "", "type": "playlist", "user_id": "1757896418571832",
                "username": owner[0], "author_name": owner[0], "avatar": "https://audio.com/a.jpeg",
                "slug": owner[1], "title": "Lo-fi Rap", "seo_title": "", "description": "",
                "image": "https://audio.com/c.jpeg", "is_public": True, "is_browsable": True,
                "is_downloadable": False, "is_immutable": False, "is_private": False,
                "date_created": 1676627849, "date_updated": 1718883894,
                "tags": ["Rap"], "cross_link_tags": ["beats"]})
        if path == "/collection/items":
            items = self.collections.get((q.get("username", ""), q.get("slug", "")))
            if items is None:
                return web.json_response(not_found("Collection not found!"), status=404)
            page = int(q.get("page", "1"))
            pages = self.claimed_pages or max(1, -(-len(items) // self.PER_PAGE))
            chunk = items[(page - 1) * self.PER_PAGE: page * self.PER_PAGE]
            if self.claimed_pages:
                chunk = [item_doc(page * 100 + i) for i in range(self.PER_PAGE)]
            return web.json_response(
                {"items": chunk, "pagination": {"total": len(items), "pages": pages,
                                                "page": page, "size": self.PER_PAGE}},
                headers={"X-Pagination-Total-Count": str(len(items)),
                         "X-Pagination-Page-Count": str(pages),
                         "X-Pagination-Current-Page": str(page),
                         "X-Pagination-Per-Page": str(self.PER_PAGE)})
        return web.json_response({"name": "Not Found", "message": "Page not found.",
                                  "code": 0, "status": 404}, status=404)

    async def __aenter__(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_get("/{tail:.*}", self.handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.monkeypatch.setattr(audiocom, "API", f"http://127.0.0.1:{port}")
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()


@pytest.fixture(autouse=True)
def no_ytdlp(monkeypatch):
    """audio.com never goes through yt-dlp for metadata or stream links."""
    class Poisoned:
        def __init__(self, *a, **k):
            raise AssertionError("yt-dlp was asked about an audio.com link")
    monkeypatch.setattr(audio.yt_dlp, "YoutubeDL", Poisoned)


def ext_of(url: str) -> str:
    return os.path.splitext(urlsplit(url).path)[1]


# ── which links are audio.com links ──────────────────────────────────────────

@pytest.mark.parametrize("text, kind, user, slug, aid", [
    ("https://audio.com/firedies/audio/dive", "track", "firedies", "dive", ""),
    ("https://audio.com/embed/audio/1879481920371808?theme=image", "track", "", "", "1879481920371808"),
    ("https://audio.com/audio-com/collections/lo-fi-rap", "collection", "audio-com", "lo-fi-rap", ""),
    ("https://audio.com/embed/collection/1758071724280935", "collection", "", "", "1758071724280935"),
    ("https://audio.com/firedies", "other", "", "", ""),
    ("https://audio.com/", "other", "", "", ""),
])
def test_track_collection_and_other_audiocom_links_are_told_apart(text, kind, user, slug, aid):
    link = audiocom.parse(text)
    assert link is not None
    assert (link.kind, link.user, link.slug, link.id) == (kind, user, slug, aid)


@pytest.mark.parametrize("text", [
    "https://audio.com/firedies/audio/dive?utm_source=share&utm_medium=x#comments",
    "https://audio.com/firedies/audio/dive/",
    "https://www.audio.com/firedies/audio/dive",
    "http://audio.com/firedies/audio/dive",
    "https://AUDIO.com/FireDies/audio/Dive",
    "audio.com/firedies/audio/dive",
    "  https://audio.com/firedies/audio/dive  ",
])
def test_pasted_track_links_are_normalised_to_the_page_link(text):
    link = audiocom.parse(text)
    assert link is not None and link.kind == "track"
    assert link.url == "https://audio.com/firedies/audio/dive"


@pytest.mark.parametrize("text", [
    "https://api.audio.com/audio/1879481920371808",
    "https://notaudio.com/firedies/audio/dive",
    "https://audio.community/firedies/audio/dive",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "dive by firedies",
    "audio.com dive",
    "https://example.com/?next=https://audio.com/firedies/audio/dive",
    "ftp://audio.com/firedies/audio/dive",
    "",
    None,
])
def test_other_sites_and_plain_searches_are_not_audiocom(text):
    assert audiocom.parse(text) is None
    assert not audio.is_audiocom_url(text or "")


@pytest.mark.asyncio
async def test_tests_can_never_reach_the_real_audiocom():
    assert audiocom.API == "https://api.audio.com"
    with pytest.raises(audiocom.HostBlocked):
        await audio.fetch_track("https://audio.com/firedies/audio/dive")


def test_only_collections_count_as_playlists():
    assert audio.is_any_playlist("https://audio.com/audio-com/collections/lo-fi-rap")
    assert not audio.is_any_playlist("https://audio.com/firedies/audio/dive")
    assert audio.playlist_source_label("https://audio.com/audio-com/collections/lo-fi-rap") == "audio.com"


# ── one track ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_track_link_queues_its_mp3_with_title_length_and_artist(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        track = await audio.fetch_track(page)

    assert track["title"] == "Firedies - Dive"
    assert track["webpage_url"] == "https://audio.com/firedies/audio/dive"
    assert ext_of(track["stream_url"]) == ".mp3"
    assert track["duration"] == 54
    assert track["thumbnail"] == ("https://audio.com/s3w/audio.com.static/audio/image/08/18/"
                                  "1879481920371808-1.jpeg")
    assert (track["artist"], track["song_title"], track["uploader"]) == ("Firedies", "Dive", "Firedies")
    assert track["is_music_meta"] is True
    assert track["_needs_resolve"] is False
    # The link says when it stops working (signed just now, for six days).
    assert track["_expires_at"] == pytest.approx(time.time() + 518400, abs=5)


@pytest.mark.asyncio
async def test_the_page_lookup_sends_the_bare_page_link(monkeypatch):
    """audio.com's oEmbed answers 404 to a link with a query string, a
    trailing slash, www. or capitals — all of which turn up in pasted links."""
    async with FakeAudioCom(monkeypatch) as srv:
        srv.add_track(audio_doc())
        track = await audio.fetch_track("https://www.audio.com/FireDies/audio/Dive/?utm_source=share")
        assert srv.oembed_urls() == ["https://audio.com/firedies/audio/dive"]
    assert track["title"] == "Firedies - Dive"


@pytest.mark.parametrize("formats, source, want", [
    (("m4a", "mp3"), True, ".mp3"),     # MP3 starts fastest: the M4A's index is at its end
    (("m4a",), True, ".m4a"),
    ((), True, ".wav"),                 # nothing transcoded yet: the original upload
])
@pytest.mark.asyncio
async def test_mp3_is_preferred_then_m4a_then_the_original_upload(monkeypatch, formats, source, want):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc(formats=formats, source=source))
        track = await audio.fetch_track(page)
    assert ext_of(track["stream_url"]) == want


@pytest.mark.asyncio
async def test_a_track_with_nothing_playable_is_refused(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc(formats=(), source=False))
        with pytest.raises(ValueError) as err:
            await audio.fetch_track(page)
    assert "audio.com" in audio.friendly_error_msg(err.value)
    assert "no playable audio" in str(err.value)


@pytest.mark.asyncio
async def test_a_missing_track_is_reported_as_missing(monkeypatch):
    async with FakeAudioCom(monkeypatch):
        with pytest.raises(ValueError) as err:
            await audio.fetch_track("https://audio.com/firedies/audio/not-a-song")
    assert "couldn't find that audio.com track" in audio.friendly_error_msg(err.value)


@pytest.mark.asyncio
async def test_a_private_track_is_reported_like_a_missing_one(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        srv.private.add("1879481920371808")
        with pytest.raises(ValueError) as err:
            await audio.fetch_track(page)
    assert "couldn't find that audio.com track" in audio.friendly_error_msg(err.value)


@pytest.mark.asyncio
async def test_a_track_whose_page_cant_be_rebuilt_keeps_a_link_that_resolves(monkeypatch):
    """The page link is rebuilt from the uploader and slug; one that wouldn't
    survive being pasted back (it's re-read on every replay) uses the embed
    link instead, which names the track by id."""
    async with FakeAudioCom(monkeypatch) as srv:
        srv.add_track(audio_doc(slug="odd/slug?x"))
        track = await audio.fetch_track("https://audio.com/embed/audio/1879481920371808")
        assert track["webpage_url"] == "https://audio.com/embed/audio/1879481920371808"
        again = requeue_copy(track)
        await audio.resolve_stream_url(again)
    assert ext_of(again["stream_url"]) == ".mp3"


@pytest.mark.asyncio
async def test_an_embed_link_goes_straight_to_the_track(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        srv.add_track(audio_doc())
        track = await audio.fetch_track("https://audio.com/embed/audio/1879481920371808?theme=image")
        assert srv.oembed_urls() == []
    assert track["webpage_url"] == "https://audio.com/firedies/audio/dive"


@pytest.mark.asyncio
async def test_a_profile_link_is_refused_without_asking_audiocom(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        with pytest.raises(ValueError) as err:
            await audio.fetch_track("https://audio.com/firedies")
        assert srv.hits == []
    assert "track or a collection" in str(err.value)


@pytest.mark.asyncio
async def test_one_api_hiccup_is_retried(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        srv.trouble = [503]
        track = await audio.fetch_track(page)
        assert [p for p, _ in srv.hits] == ["/oembed", "/oembed", "/audio/1879481920371808"]
    assert track["title"] == "Firedies - Dive"


@pytest.mark.asyncio
async def test_an_api_outage_is_reported_as_audiocom_trouble(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        srv.trouble = [503] * 10
        with pytest.raises(ValueError) as err:
            await audio.fetch_track(page)
        assert len(srv.hits) == 2            # one retry, then give up
    msg = audio.friendly_error_msg(err.value)
    assert "audio.com" in msg and "try again" in msg


@pytest.mark.asyncio
async def test_an_uploader_already_in_the_title_is_not_repeated(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc(title="Top (prod.by Yellxw Mxtey)", author="Yellxw Mxtey",
                                       username="yellxw-mxtey", slug="top-prodby-yellxw-mxtey"))
        track = await audio.fetch_track(page)
    assert track["title"] == "Top (prod.by Yellxw Mxtey)"


@pytest.mark.asyncio
async def test_a_non_music_upload_gets_an_uploader_but_no_artist(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc(title="Rain on a tin roof", author="Field Notes",
                                       username="field-notes", slug="rain-on-a-tin-roof",
                                       category_id=7, tags=("sound:rain", "sound:ambience")))
        track = await audio.fetch_track(page)
    assert track["uploader"] == "Field Notes"
    assert "artist" not in track
    assert track["is_music_meta"] is False


# ── collections ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_collection_queues_every_track_across_its_pages(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        url = srv.add_collection("audio-com", "lo-fi-rap", [item_doc(n) for n in range(1, 24)])
        tracks = await audio.fetch_playlist(url)
        pages = [q["page"] for p, q in srv.hits if p == "/collection/items"]
    assert pages == ["1", "2"]
    assert [t["title"] for t in tracks] == [f"Beat Maker - Beat {n}" for n in range(1, 24)]
    first = tracks[0]
    assert first["webpage_url"] == "https://audio.com/beatmaker/audio/beat-1"
    assert first["duration"] == 101
    assert first["_needs_resolve"] is True and first["stream_url"] is None
    assert first["_search_query"] == "https://audio.com/beatmaker/audio/beat-1"


@pytest.mark.asyncio
async def test_a_collection_embed_link_finds_the_collection_it_shows(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        srv.add_collection("audio-com", "lo-fi-rap", [item_doc(1), item_doc(2)], cid="1758071724280935")
        tracks = await audio.fetch_playlist("https://audio.com/embed/collection/1758071724280935")
    assert [t["webpage_url"] for t in tracks] == ["https://audio.com/beatmaker/audio/beat-1",
                                                   "https://audio.com/beatmaker/audio/beat-2"]


@pytest.mark.asyncio
async def test_a_missing_collection_is_reported_as_missing(monkeypatch):
    async with FakeAudioCom(monkeypatch):
        with pytest.raises(ValueError) as err:
            await audio.fetch_playlist("https://audio.com/audio-com/collections/nothing-here")
    assert "couldn't find that audio.com collection" in audio.friendly_error_msg(err.value)


@pytest.mark.asyncio
async def test_a_runaway_collection_stops_at_the_page_cap(monkeypatch):
    monkeypatch.setattr(audiocom, "MAX_PAGES", 3)
    async with FakeAudioCom(monkeypatch) as srv:
        url = srv.add_collection("audio-com", "endless", [item_doc(1)])
        srv.claimed_pages = 10_000
        tracks = await audio.fetch_playlist(url)
        requests = [p for p, _ in srv.hits if p == "/collection/items"]
    assert len(requests) == 3
    assert len(tracks) == 3 * FakeAudioCom.PER_PAGE


@pytest.mark.asyncio
async def test_a_single_track_lookup_of_a_collection_gives_its_first_track(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        for n in (1, 2):
            srv.add_track(audio_doc(aid=f"17879165491293{n:02d}", username="beatmaker",
                                    author="Beat Maker", slug=f"beat-{n}", title=f"Beat {n}"))
        url = srv.add_collection("audio-com", "lo-fi-rap", [item_doc(1), item_doc(2)])
        track = await audio.fetch_track(url)
    assert track["title"] == "Beat Maker - Beat 1"
    assert ext_of(track["stream_url"]) == ".mp3"


# ── playing a queued entry ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_queued_audiocom_link_resolves_through_the_api(monkeypatch):
    """A saved playlist queues bare links: the title is the link until it plays."""
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        stub = {"title": page, "stream_url": None, "webpage_url": page, "duration": 0,
                "thumbnail": None, "requester": "alex", "_needs_resolve": True,
                "_search_query": page}
        await audio.resolve_stream_url(stub)
    assert ext_of(stub["stream_url"]) == ".mp3"
    assert stub["_needs_resolve"] is False
    assert stub["title"] == "Firedies - Dive"
    assert stub["duration"] == 54
    assert stub["artist"] == "Firedies"


@pytest.mark.parametrize("how", ["stale", "forced", "requeued"])
@pytest.mark.asyncio
async def test_an_old_or_rejected_link_is_fetched_again(monkeypatch, how):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc(sig="0ld"))
        track = await audio.fetch_track(page)
        srv.add_track(audio_doc(sig="fresh"))
        if how == "stale":
            track["_resolved_at"] -= audio.STREAM_URL_TTL + 1
        elif how == "requeued":
            track = requeue_copy(track)
        await audio.resolve_stream_url(track, force=(how == "forced"))
    assert "X-Amz-Signature=fresh" in track["stream_url"]


@pytest.mark.asyncio
async def test_a_fresh_link_is_not_fetched_again(monkeypatch):
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        track = await audio.fetch_track(page)
        before = len(srv.hits)
        await audio.resolve_stream_url(track)
        assert len(srv.hits) == before


# ── signed links say when they expire ────────────────────────────────────────

@pytest.mark.parametrize("url, want", [
    # S3-style signature: X-Amz-Date (UTC) + X-Amz-Expires seconds.
    ("https://s3.example.test/a.mp3?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20261026T215400Z"
     "&X-Amz-SignedHeaders=host&X-Amz-Expires=518400&X-Amz-Signature=abc", 1793570040),
    ("https://s3.example.test/a.mp3?x-amz-date=20261027T080000Z&x-amz-expires=518400", 1793606400),
    # YouTube's googlevideo links, as before.
    ("https://rr1.googlevideo.com/videoplayback?expire=1793051640&ei=x", 1793051640),
    ("https://s3.example.test/a.mp3?X-Amz-Date=garbage&X-Amz-Expires=518400", None),
    ("https://cdn.example.test/a.mp3", None),
])
def test_signed_links_carry_their_expiry(url, want):
    assert audio._url_expiry(url) == want


# ── other commands that take a link ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_chapters_lookup_on_audiocom_finds_none_without_yt_dlp(monkeypatch):
    from cogs.downloads import extract_chapters
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        info = await audio.fetch_info(page)
    assert info["title"] == "Firedies - Dive"
    assert info["duration"] == 54
    assert extract_chapters(info) == []


class FakeDownloader:
    """yt_dlp.YoutubeDL for a direct file link: the generic extractor names
    the file after the link's last path segment."""
    calls: list[tuple[str, bool]] = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _info(self, url: str) -> dict:
        name = os.path.splitext(os.path.basename(urlsplit(url).path))[0]
        return {"title": name, "ext": "mp3", "vcodec": "none", "acodec": "mp3",
                "url": url, "duration": None, "webpage_url": url}

    def extract_info(self, url, download=False):
        FakeDownloader.calls.append((url, download))
        info = self._info(url)
        if download:
            path = (self.opts["outtmpl"].replace("%(title)s", info["title"])
                    .replace("%(ext)s", "mp3"))
            with open(path, "wb") as fh:
                fh.write(b"ID3 fake mp3")
            info["requested_downloads"] = [{"filepath": path}]
        return info

    def prepare_filename(self, info):
        return self.opts["outtmpl"].replace("%(title)s", info["title"]).replace("%(ext)s", "mp3")


@pytest.fixture
def downloads_cog():
    from cogs.downloads import DownloadsCog
    return DownloadsCog(MagicMock())


@pytest.fixture
def status():
    s = MagicMock()
    s.edit = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_dl_of_an_audiocom_link_downloads_its_mp3_named_after_the_song(
        monkeypatch, downloads_cog, status, tmp_path):
    from cogs import downloads
    from cogs.downloads import JobCanceller
    FakeDownloader.calls = []
    monkeypatch.setattr(downloads.yt_dlp, "YoutubeDL", FakeDownloader)
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        got = await downloads_cog._fetch_via_ytdlp(
            page, forced=None, workdir=str(tmp_path), status=status, canceller=JobCanceller())
    assert got is not None
    path, title, duration, has_video, _height = got
    assert (title, duration, has_video) == ("Firedies - Dive", 54, False)
    assert os.path.basename(path) == "Firedies - Dive.mp3" and os.path.exists(path)
    # yt-dlp only ever saw the signed MP3 link, never the audio.com page.
    assert FakeDownloader.calls
    assert all(urlsplit(u).netloc == "s3.example.test" and ext_of(u) == ".mp3"
               for u, _ in FakeDownloader.calls)


@pytest.mark.asyncio
async def test_dl_mp4_of_an_audiocom_link_is_refused(monkeypatch, downloads_cog, status, tmp_path):
    from cogs import downloads
    from cogs.downloads import JobCanceller
    FakeDownloader.calls = []
    monkeypatch.setattr(downloads.yt_dlp, "YoutubeDL", FakeDownloader)
    async with FakeAudioCom(monkeypatch) as srv:
        page = srv.add_track(audio_doc())
        got = await downloads_cog._fetch_via_ytdlp(
            page, forced="mp4", workdir=str(tmp_path), status=status, canceller=JobCanceller())
    assert got is None
    assert "audio-only" in status.edit.await_args.kwargs["content"]
    assert FakeDownloader.calls == []


@pytest.mark.asyncio
async def test_dl_of_a_missing_audiocom_track_explains_itself(monkeypatch, downloads_cog, status, tmp_path):
    from cogs.downloads import JobCanceller
    async with FakeAudioCom(monkeypatch):
        got = await downloads_cog._fetch_via_ytdlp(
            "https://audio.com/firedies/audio/gone", forced=None, workdir=str(tmp_path),
            status=status, canceller=JobCanceller())
    assert got is None
    msg = status.edit.await_args.kwargs["content"]
    assert msg.startswith("❌") and "couldn't find that audio.com track" in msg


@pytest.mark.asyncio
async def test_convert_of_an_audiocom_link_goes_through_the_api_not_a_page_fetch(
        downloads_cog, status, tmp_path):
    """The page is behind a browser check; fetching it directly gets a 403."""
    from cogs.downloads import JobCanceller
    song = tmp_path / "Firedies - Dive.mp3"
    song.write_bytes(b"ID3")
    downloads_cog._download_direct = AsyncMock(side_effect=AssertionError("fetched the page"))
    downloads_cog._fetch_via_ytdlp = AsyncMock(return_value=(str(song), "Firedies - Dive", 54, False, 0))
    got = await downloads_cog._materialise_source(
        attachment=None, url="https://audio.com/firedies/audio/dive", workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="convert")
    assert got == (str(song), "Firedies - Dive.mp3")
    assert downloads_cog._fetch_via_ytdlp.await_args.kwargs["raw_media"] is True
