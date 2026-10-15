"""core/music_catalog.py: picking the right album/artist out of search
results, choosing tracks, and the Deezer → iTunes fallback. No real network:
a local aiohttp server stands in for both APIs (real lookups were checked by
hand from the bot host; see the module docstring)."""
from __future__ import annotations

import asyncio
import random

import pytest

from core import music_catalog as mc

T = [mc.CatalogTrack(f"Song {n}", "Band", 200, n) for n in range(1, 11)]


# ── which tracks ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sel,want", [
    ("all", list(range(1, 11))), ("", list(range(1, 11))), ("whole", list(range(1, 11))),
    ("1-3", [1, 2, 3]), ("3-1", [1, 2, 3]), ("tracks 1 to 3", [1, 2, 3]), ("1 – 3", [1, 2, 3]),
    ("5", [5]), ("track 5", [5]), ("1,4,6", [1, 4, 6]), ("2, 2", [2, 2]),
    ("2, 2, 99", []),            # no track 99: said, not silently dropped (select() says why)
    ("first 3", [1, 2, 3]), ("first", [1]), ("top 2", [1, 2]),
    ("last 2", [9, 10]), ("last", [10]), ("final 1", [10]),
    ("9-15", [9, 10]), ("0", []), ("the good ones", []),
])
def test_select_tracks(sel, want):
    assert [t.position for t in mc.select_tracks(T, sel)] == want


@pytest.mark.parametrize("sel,want", [
    # How the LLM (or a person) may put it, not just the forms it was asked for —
    # "the seventh and ninth song off Pray for the Wicked" must be tracks 7 and 9.
    ("seventh and ninth", [7, 9]), ("7 and 9", [7, 9]), ("the 7th & 9th songs", [7, 9]),
    ("tracks seven and nine", [7, 9]), ("7th, 9th", [7, 9]), ("#7 and #9", [7, 9]),
    ("the seventh one and the ninth one", [7, 9]), ("7 9", [7, 9]),
    ("1, 2, and 3", [1, 2, 3]), ("2nd, 4th, and 6th", [2, 4, 6]), ("one, three and five", [1, 3, 5]),
    ("first and third", [1, 3]), ("first and last", [1, 10]), ("first two and the last", [1, 2, 10]),
    ("the first three", [1, 2, 3]), ("the top three", [1, 2, 3]), ("first to fourth", [1, 2, 3, 4]),
    ("three to five", [3, 4, 5]), ("tracks 1 to 3 and 5", [1, 2, 3, 5]), ("the 1st through 3rd", [1, 2, 3]),
    ("the last two", [9, 10]), ("the last one", [10]), ("the last couple", [9, 10]),
    ("the whole thing", list(range(1, 11))), ("the entire album", list(range(1, 11))),
    ("all of the first three", [1, 2, 3]),
    ("all but the first", list(range(2, 11))), ("second to last", [9]),
    # Unclear: nothing, so the bot asks — never the opposite of what was meant.
    ("the good ones", []), ("the middle one", []),
])
def test_select_tracks_said_in_words(sel, want):
    assert [t.position for t in mc.select_tracks(T, sel)] == want


# ── the brief's selectors (2026-10-13): exclusions, halves, combinations ─────

def _album(n):
    return [mc.CatalogTrack(f"Song {i}", "Band", 200, i) for i in range(1, n + 1)]


def _picked(n, sel):
    return [t.position for t in mc.select(_album(n), sel).tracks]


@pytest.mark.parametrize("sel,want", [
    ("all except last", list(range(1, 11))), ("all except the last", list(range(1, 11))),
    ("the entire album except for the last song", list(range(1, 11))),
    ("everything but tracks 2 and 5", [1, 3, 4, 6, 7, 8, 9, 10, 11]),
    ("all except 2, 5", [1, 3, 4, 6, 7, 8, 9, 10, 11]),
    ("first half", [1, 2, 3, 4, 5, 6]), ("the first half", [1, 2, 3, 4, 5, 6]),
    ("second half", [7, 8, 9, 10, 11]), ("latter half", [7, 8, 9, 10, 11]),
    ("the back half", [7, 8, 9, 10, 11]), ("last half", [7, 8, 9, 10, 11]),
    ("first half except its last", [1, 2, 3, 4, 5]),
    ("first half except the last one", [1, 2, 3, 4, 5]),
    ("first 7 except 6", [1, 2, 3, 4, 5, 7]),
    ("the first seven but leave out the sixth", [1, 2, 3, 4, 5, 7]),
    ("all but the first one", list(range(2, 12))),
    ("both halves", list(range(1, 12))),
    ("9 and 7", [9, 7]),                         # in the order asked
    ("1-6 and 5-8", [1, 2, 3, 4, 5, 6, 7, 8]),   # overlapping ranges: once each
    ("3rd to last", [9]), ("3 to the last", list(range(3, 12))),
])
def test_selectors_on_an_11_track_album(sel, want):
    assert _picked(11, sel) == want


def test_halves_split_odd_albums_ceil_first_and_never_overlap():
    for n in range(1, 16):
        first, second = _picked(n, "first half"), _picked(n, "second half")
        assert first == list(range(1, -(-n // 2) + 1)), n
        assert first + second == list(range(1, n + 1)), n      # nothing left out, no overlap
    assert _picked(10, "first half") == [1, 2, 3, 4, 5] and _picked(10, "latter half") == [6, 7, 8, 9, 10]
    assert _picked(1, "first half") == [1] and _picked(1, "second half") == []
    assert _picked(2, "second half") == [2]


@pytest.mark.parametrize("sel", ["the good ones", "the middle one", "all except", "which half",
                                 "half", "the half", "track 12", "songs 3 to 15 except 99"])
def test_unsupported_or_impossible_selectors_are_errors_not_guesses(sel):
    with pytest.raises(mc.SelectionError):
        mc.select(_album(11), sel)


def test_explicit_repeats_survive_selection_and_ranges_are_clipped_with_a_note():
    assert _picked(11, "2, 2") == [2, 2]
    got = mc.select(_album(11), "9-15")
    assert [t.position for t in got.tracks] == [9, 10, 11] and "11" in got.note


def test_how_a_track_number_may_be_said():
    assert mc.said_as(7) == ["7", "7th", "seventh", "seven"]
    assert mc.said_as(2)[:3] == ["2", "2nd", "second"] and mc.said_as(12)[1] == "12th"
    assert "couple" not in mc.said_as(2) and mc.said_as(23) == ["23", "23rd"]


@pytest.mark.parametrize("sel", ["3 random", "three random ones", "a couple of random songs",
                                 "any two", "random two"])
def test_random_picks_said_in_words(sel):
    got = mc.select_tracks(T, sel, rng=random.Random(2))
    assert len(got) == (3 if "3" in sel or "three" in sel else 2) and len(set(got)) == len(got)


def test_random_tracks_are_distinct_and_from_the_album():
    got = mc.select_tracks(T, "random 3", rng=random.Random(4))
    assert len(got) == 3 and len(set(got)) == 3 and set(got) <= set(T)
    assert len(mc.select_tracks(T, "random 50")) == 10


def test_a_huge_album_is_capped():
    big = [mc.CatalogTrack(f"S{n}", "B", 1, n) for n in range(1, 80)]
    assert len(mc.select_tracks(big, "all")) == mc.MAX_TRACKS


@pytest.mark.parametrize("value,want", [("3", 3), ("three", 3), ("a couple", 2), ("", 5),
                                        ("50", 10), ("0", 1), ("some", 5)])
def test_parse_count(value, want):
    assert mc.parse_count(value, 5, 10) == want


# ── which album / artist ──────────────────────────────────────────────────────

def test_the_plain_album_beats_a_deluxe_edition_and_ties_keep_catalogue_order():
    cands = [{"id": 1, "title": "Vices & Virtues (Deluxe Edition)", "artist": "Panic! At the Disco"},
             {"id": 2, "title": "Vices & Virtues", "artist": "Panic! At the Disco"},
             {"id": 3, "title": "Vices & Virtues", "artist": "Panic! At the Disco"},
             {"id": 4, "title": "Vices & Virtues - Single", "artist": "Reinaeiry",
              "record_type": "single"}]
    assert mc.pick_album(cands, "Vices and Virtues", "Panic! At The Disco")["id"] == 2
    assert mc.pick_album(cands, "Vices and Virtues deluxe", "Panic! At The Disco")["id"] == 1


def test_editions_rank_plain_then_remaster_then_deluxe_then_live():
    cands = [{"id": 1, "title": "Abbey Road (Super Deluxe Edition)", "artist": "The Beatles"},
             {"id": 2, "title": "Abbey Road (Remastered 2009)", "artist": "The Beatles"},
             {"id": 3, "title": "Abbey Road (2019 Mix)", "artist": "The Beatles"},
             {"id": 4, "title": "Abbey Road", "artist": "The Beatles Complete On Ukulele"}]
    assert mc.pick_album(cands, "Abbey Road", "The Beatles")["id"] == 2      # first of the variants
    live = [{"id": 5, "title": "Live In Denver", "artist": "Panic! At The Disco"},
            {"id": 6, "title": "A Fever You Can't Sweat Out (20th Anniversary Deluxe)",
             "artist": "Panic! At The Disco"}]
    assert mc.edition_penalty("Live In Denver", "Fever") == 0.5
    assert mc.edition_penalty("Live Through This", "Live Through This") <= 0.0
    assert mc.pick_album(live, "A Fever You Can't Sweat Out", "Panic! At The Disco")["id"] == 6


def test_an_unasked_for_deluxe_edition_is_trimmed_to_the_album():
    tracks = [mc.CatalogTrack("Intro", "P", position=1, version="(2025 Remaster)"),
              mc.CatalogTrack("Song", "P", position=2, version="(2025 Remaster)"),
              mc.CatalogTrack("Song", "P", position=3, version="(Demo)"),
              mc.CatalogTrack("Song", "P", position=4, version="(Live In Denver)"),
              mc.CatalogTrack("Bonus", "P", position=5, version="(Bonus Track)")]
    assert [(t.title, t.position) for t in mc.core_tracks(tracks)] == [
        ("Intro", 1), ("Song", 2), ("Bonus", 5)]
    only_live = [mc.CatalogTrack("A", "P", version="(Live)")]
    assert mc.core_tracks(only_live) == only_live


@pytest.mark.parametrize("title,base,version", [
    ("Cave In (Album Version)", "Cave In", "(Album Version)"),
    ("Hurricane", "Hurricane", ""),
    ("Nearly Witches (Ever Since We Met...)", "Nearly Witches", "(Ever Since We Met...)"),
    ("Song [Live]", "Song", "[Live]"),
])
def test_split_version(title, base, version):
    assert mc.split_version(title) == (base, version)


def test_live_takes_search_as_live_and_remasters_as_the_song():
    assert mc.CatalogTrack("Song", "P", version="(Live In Denver)").query == "P - Song (Live In Denver)"
    assert mc.CatalogTrack("Song", "P", version="(2009 Remaster)").query == "P - Song"


def test_another_artists_album_is_never_taken():
    cands = [{"id": 9, "title": "Greatest Hits", "artist": "Queen"}]
    assert mc.pick_album(cands, "Greatest Hits", "ABBA") is None
    assert mc.pick_album(cands, "Totally Different", "Queen") is None


def test_bracketed_words_said_in_full_still_match():
    cands = [{"id": 5, "title": "(What's the Story) Morning Glory?", "artist": "Oasis"}]
    assert mc.pick_album(cands, "What's the Story Morning Glory", "Oasis")["id"] == 5
    assert mc.pick_album(cands, "Morning Glory", "Oasis")["id"] == 5


def test_the_real_artist_beats_a_fanless_duplicate():
    cands = [{"id": 412845981, "name": "Panic! At The Disco", "fans": 0},
             {"id": 402, "name": "Panic! At the Disco", "fans": 1706521},
             {"id": 7, "name": "Panic Room", "fans": 99999999}]
    assert mc.pick_artist(cands, "panic at the disco")["id"] == 402
    assert mc.pick_artist(cands, "Nobody Like This") is None


def test_newest_means_original_releases_first():
    albums = [{"title": "Fever (20th Anniversary Deluxe)", "released": "2026-01-23"},
              {"title": "Song (2025 Remaster)", "released": "2025-11-01"},
              {"title": "Live In Denver", "released": "2025-10-20"},
              {"title": "Viva Las Vengeance", "released": "2022-08-19"},
              {"title": "Pray for the Wicked", "released": "2018-06-22"}]
    assert [a["title"] for a in mc._released_order(albums)] == [
        "Viva Las Vengeance", "Pray for the Wicked"]
    only_live = [{"title": "Live 1", "released": "2020"}, {"title": "Live 2", "released": "2021"}]
    assert [a["title"] for a in mc._released_order(only_live)] == ["Live 2", "Live 1"]


def test_tracks_search_youtube_as_artist_dash_title():
    assert mc.CatalogTrack("Hurricane", "Panic! At the Disco").query == "Panic! At the Disco - Hurricane"


# ── against a stand-in for Deezer and iTunes ──────────────────────────────────

ALBUMS = {"data": [
    {"id": 928878, "title": "Vices & Virtues", "artist": {"name": "Panic! At the Disco"},
     "record_type": "album", "cover_big": "https://img/vv.jpg", "link": "https://deezer/album/928878"},
    {"id": 1, "title": "Vices & Virtues", "artist": {"name": "Somebody Else"}}]}
TRACKS = {"data": [
    {"title": t, "duration": 200 + n, "track_position": n, "disk_number": 1,
     "artist": {"name": "Panic! At the Disco"}}
    for n, t in [(2, "Let's Kill Tonight"), (1, "The Ballad of Mona Lisa"), (3, "Hurricane"),
                 (4, "Memories (Album Version)")]], "total": 4}


class FakeCatalog:
    """/deezer/... and /itunes/... on one local server."""

    def __init__(self):
        self.hits: list[str] = []
        self.deezer_broken = False
        self.deezer_quota = False

    async def handler(self, request):
        from aiohttp import web
        path = request.path
        self.hits.append(path)
        if path.startswith("/deezer"):
            if self.deezer_broken:
                return web.Response(status=503, text="down")
            if self.deezer_quota:
                return web.json_response({"error": {"type": "Exception",
                                                    "message": "Quota limit exceeded", "code": 4}})
            if path == "/deezer/search/album":
                return web.json_response(ALBUMS)
            if path == "/deezer/album/928878/tracks":
                return web.json_response(TRACKS)
            if path == "/deezer/search/artist":
                return web.json_response({"data": [{"id": 402, "name": "Owl City", "nb_fan": 5}]})
            if path == "/deezer/artist/402/top":
                n = int(request.query.get("limit", 5))
                return web.json_response({"data": [
                    {"title": f"Hit {i}", "duration": 180, "artist": {"name": "Owl City"}}
                    for i in range(1, n + 1)]})
        if path == "/itunes/search":
            return web.json_response({"results": [
                {"collectionId": 77, "collectionName": "Vices & Virtues", "artistName": "Panic! At the Disco",
                 "artworkUrl100": "https://img/100x100.jpg"}]})
        if path == "/itunes/lookup":
            return web.json_response({"results": [
                {"wrapperType": "collection", "collectionId": 77},
                {"wrapperType": "track", "trackName": "Hurricane", "artistName": "Panic! At the Disco",
                 "trackNumber": 3, "discNumber": 1, "trackTimeMillis": 265000},
                {"wrapperType": "track", "trackName": "The Ballad of Mona Lisa",
                 "artistName": "Panic! At the Disco", "trackNumber": 1, "discNumber": 1,
                 "trackTimeMillis": 226000}]})
        return web.json_response({"data": []})

    async def __aenter__(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_get("/{tail:.*}", self.handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.deezer, self.itunes = f"http://127.0.0.1:{port}/deezer", f"http://127.0.0.1:{port}/itunes"
        mc.clear_cache()
        return self

    async def __aexit__(self, *exc):
        mc.clear_cache()
        await self.runner.cleanup()


def run(coro):
    return asyncio.run(coro)


def test_an_album_comes_back_in_track_order_from_deezer():
    async def go():
        async with FakeCatalog() as srv:
            album = await mc.find_album("Vices and Virtues", "Panic! At The Disco",
                                        deezer=srv.deezer, itunes=srv.itunes)
            assert album.source == "deezer" and album.title == "Vices & Virtues"
            assert [t.title for t in album.tracks] == [
                "The Ballad of Mona Lisa", "Let's Kill Tonight", "Hurricane", "Memories"]
            assert album.tracks[3].version == "(Album Version)"
            assert album.cover == "https://img/vv.jpg"
            assert not any(h.startswith("/itunes") for h in srv.hits)
    run(go())


@pytest.mark.parametrize("trouble", ["deezer_broken", "deezer_quota"])
def test_deezer_trouble_falls_back_to_itunes(trouble):
    async def go():
        async with FakeCatalog() as srv:
            setattr(srv, trouble, True)
            album = await mc.find_album("Vices & Virtues", "Panic! at the disco",
                                        deezer=srv.deezer, itunes=srv.itunes)
            assert album.source == "itunes"
            assert [t.title for t in album.tracks] == ["The Ballad of Mona Lisa", "Hurricane"]
            assert album.tracks[1].duration == 265
            assert album.cover == "https://img/600x600.jpg"
    run(go())


def test_nothing_reachable_is_an_error_not_a_miss():
    async def go():
        with pytest.raises(mc.CatalogError):
            await mc.find_album("X", "Y", deezer="http://127.0.0.1:9/deezer",
                                itunes="http://127.0.0.1:9/itunes")
    run(go())


def test_an_unknown_album_is_none():
    async def go():
        async with FakeCatalog() as srv:
            assert await mc.find_album("Nothing Like It", "Nobody",
                                       deezer=srv.deezer, itunes=srv.itunes) is None
    run(go())


def test_lookups_are_cached():
    async def go():
        async with FakeCatalog() as srv:
            for _ in range(2):
                await mc.find_album("Vices and Virtues", "Panic! At The Disco",
                                    deezer=srv.deezer, itunes=srv.itunes)
            assert srv.hits.count("/deezer/search/album") == 1
    run(go())


def test_top_tracks():
    async def go():
        async with FakeCatalog() as srv:
            got = await mc.top_tracks("owl city", 3, deezer=srv.deezer, itunes=srv.itunes)
            assert [t.query for t in got] == ["Owl City - Hit 1", "Owl City - Hit 2",
                                              "Owl City - Hit 3"]
    run(go())


# ── which song: misheard or newer-than-the-AI titles ─────────────────────────
# Reported 2026-10-12: "Play I Write Him's Not Travis D's by Young Gravy" played
# Mr. Clean, then a church hymn; "I played cordial by sighted off" queued the
# Set It Off song that was already playing. The real song is in the artist's
# catalogue — matched by spelling AND sound.

def test_misheard_titles_sound_the_same():
    assert mc.sound_key("I Write Him's Not Travis D's") == mc.sound_key("I Write Hymns Not Travesties")
    assert mc.sound_key("I write hymns, not Travis Dees") == mc.sound_key("I Write Hymns Not Travesties")
    assert mc.sound_key("sighted off") == mc.sound_key("Set It Off") == mc.sound_key("the set off")
    assert mc.sound_key("Young Gravy") == mc.sound_key("Yung Gravy")


@pytest.mark.parametrize("heard,title,low,high", [
    ("I Write Him's Not Travis D's", "I Write Hymns Not Travesties", 0.9, 1.0),
    ("cordial", "Cordial", 1.0, 1.0),
    ("Mr Brightside", "Mr. Brightside", 1.0, 1.0),
    ("betty", "Betty (Get Money)", 1.0, 1.0),                 # the bracketed part can go unsaid
    ("No Disrespect", "Cordial", 0.0, 0.4),
    ("Stop", "Step", 0.0, mc.SURE_TITLE - 0.01),             # short: sound alone can't decide
])
def test_title_score(heard, title, low, high):
    assert low <= mc.title_score(heard, title) <= high


def test_artist_names_by_sound_and_credits():
    assert mc.name_score("young gravy", "Yung Gravy") >= 0.9
    assert mc.name_score("the set off", "Set It Off") >= 0.75
    assert mc.name_score("Yung Gravy", "Yung Gravy & bbno$") >= 0.9
    assert mc.name_score("Yung Gravy", "The Killers") < 0.5


YG = [mc.CatalogTrack(t, "Yung Gravy", 150) for t in (
    "Mr. Clean", "Betty (Get Money)", "Martha Stewart", "I Write Hymns Not Travesties",
    "Rotisserie (Resauced)", "oops!")]
SIO = [mc.CatalogTrack(t, "Set It Off", 200) for t in (
    "No Disrespect", "Cordial", "Wolf in Sheep's Clothing", "Skeleton")]


def test_a_misheard_title_finds_the_artists_real_song():
    m = mc.pick_song(YG, ["I Write Him's Not Travis D's"], "Young Gravy")
    assert m.track.title == "I Write Hymns Not Travesties" and m.sure


def test_the_words_said_beat_the_ais_swap():
    # The model "fixed" a title it didn't know into the song that was playing.
    m = mc.pick_song(SIO, ["No Disrespect", "cordial"], "Set It Off", said="cordial")
    assert m.track.title == "Cordial" and m.sure
    assert [t.title for t in m.others][:1] == ["No Disrespect"]


def test_another_artists_song_is_never_taken_when_one_was_named():
    other = [mc.CatalogTrack("Cordial", "Somebody Else", 200)]
    assert mc.pick_song(other, ["Cordial"], "Set It Off") is None
    assert mc.pick_song(other + SIO, ["Cordial"], "Set It Off").track.artist == "Set It Off"


def test_close_rivals_are_not_sure_and_come_with_the_runners_up():
    cands = [mc.CatalogTrack("Hymns", "Band", 1), mc.CatalogTrack("Hims", "Band", 1)]
    m = mc.pick_song(cands, ["hymms"], "Band")
    assert not m.sure and {m.track.title, *(t.title for t in m.others)} == {"Hymns", "Hims"}


def test_a_slowed_or_sped_up_edit_loses_to_the_song_unless_asked_for():
    cands = [mc.CatalogTrack("Skeleton (Slowed & Reverb)", "Set It Off", 1),
             mc.CatalogTrack("Skeleton", "Set It Off", 1)]
    assert mc.pick_song(cands, ["Skeleton"], "Set It Off").track.title == "Skeleton"
    assert mc.pick_song(cands, ["Skeleton slowed and reverb"], "Set It Off").track.title == \
        "Skeleton (Slowed & Reverb)"


def test_no_title_no_answer():
    assert mc.pick_song(YG, ["", "  "], "Yung Gravy") is None


class FakeSongCatalog(FakeCatalog):
    """Deezer's track search knows only exact words; the artist's top list
    and releases hold the real titles (like the real API, 2026-10-12)."""

    def __init__(self):
        super().__init__()
        self.itunes_songs = []

    async def handler(self, request):
        from aiohttp import web
        path, q = request.path, request.query
        if path.startswith("/deezer") and not (self.deezer_broken or self.deezer_quota):
            self.hits.append(path)
            if path == "/deezer/search":
                words = q.get("q", "").lower()
                data = [{"title": t.title, "duration": 150, "artist": {"name": t.artist}}
                        for t in YG + SIO if t.title.lower() in words]
                return web.json_response({"data": data})
            if path == "/deezer/search/artist":
                names = {"yung gravy": (11203568, "Yung Gravy", 46552),
                         "set it off": (472399, "Set It Off", 85323)}
                hit = [v for k, v in names.items() if mc.name_score(q.get("q", ""), k) >= 0.75]
                return web.json_response({"data": [{"id": i, "name": n, "nb_fan": f}
                                                   for i, n, f in hit]})
            if path in ("/deezer/artist/11203568/top", "/deezer/artist/472399/top"):
                songs = YG if "11203568" in path else SIO
                return web.json_response({"data": [
                    {"title": t.title, "duration": 150, "artist": {"name": t.artist}}
                    for t in songs if t.title != "I Write Hymns Not Travesties"]})
            if path == "/deezer/artist/11203568/albums":
                return web.json_response({"data": [
                    {"id": 9, "title": "I Write Hymns Not Travesties", "record_type": "single",
                     "release_date": "2025-06-01"}]})
            if path.endswith("/albums"):
                return web.json_response({"data": []})
            return web.json_response({"data": []})
        if path == "/itunes/search" and q.get("entity") == "song":
            self.hits.append(path)
            return web.json_response({"results": self.itunes_songs})
        return await super().handler(request)


def test_find_song_finds_a_single_the_top_list_doesnt_have():
    async def go():
        async with FakeSongCatalog() as srv:
            m = await mc.find_song("I Write Him's Not Travis D's", "Young Gravy",
                                   deezer=srv.deezer, itunes=srv.itunes)
            assert m.sure and m.track.title == "I Write Hymns Not Travesties"
            assert m.track.artist == "Yung Gravy"
            assert m.track.query == "Yung Gravy - I Write Hymns Not Travesties"
            # cached: asking again doesn't search again
            n = len(srv.hits)
            await mc.find_song("I Write Him's Not Travis D's", "Young Gravy",
                               deezer=srv.deezer, itunes=srv.itunes)
            assert len(srv.hits) == n
    run(go())


def test_find_song_prefers_what_was_heard_over_the_ais_swap():
    async def go():
        async with FakeSongCatalog() as srv:
            m = await mc.find_song("No Disrespect", "Set It Off", heard="cordial",
                                   deezer=srv.deezer, itunes=srv.itunes)
            assert m.sure and m.track.title == "Cordial"
    run(go())


def test_find_song_falls_back_to_itunes_and_errors_only_when_nothing_answers():
    async def go():
        async with FakeSongCatalog() as srv:
            srv.deezer_broken = True
            srv.itunes_songs = [{"trackName": "Cordial", "artistName": "Set It Off",
                                 "trackTimeMillis": 201000}]
            m = await mc.find_song("cordial", "Set It Off", deezer=srv.deezer, itunes=srv.itunes)
            assert m.sure and m.track.duration == 201
        with pytest.raises(mc.CatalogError):
            await mc.find_song("x", "y", deezer="http://127.0.0.1:9/deezer",
                               itunes="http://127.0.0.1:9/itunes")
    run(go())


def test_the_real_hosts_are_blocked_in_tests():
    async def go():
        with pytest.raises(mc.CatalogError):
            await mc.find_song("Cordial", "Set It Off")
    run(go())


def test_artist_songs_lists_each_song_once_without_other_takes():
    async def go():
        async with FakeSongCatalog() as srv:
            songs = await mc.artist_songs("young gravy", deezer=srv.deezer, itunes=srv.itunes)
            titles = [t.title for t in songs]
            assert titles[0] == "Mr. Clean" and "I Write Hymns Not Travesties" in titles
            assert len(titles) == len(set(titles))
    run(go())
