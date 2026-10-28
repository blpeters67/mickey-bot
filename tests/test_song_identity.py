"""Song identity from request to upload (round 8): the artist named is the
artist played — or the bot asks. Never "Prince" for "Priest" because Prince is
better known, never an unrelated video because nothing better turned up.

Catalogue cases run against a local stand-in for Deezer/iTunes (no real
network); upload cases score made-up search results. They prove the rules, not
what the real services answer for any given song.
"""
from __future__ import annotations

import asyncio

import pytest

from core import audio
from core import music_catalog as mc
from tests.test_music_catalog import FakeCatalog

PRINCE = mc.CatalogTrack("Call My Name", "Prince", 300)
PRIEST = mc.CatalogTrack("Call My Name", "Priest", 240)


def run(coro):
    return asyncio.run(coro)


# ── picking among catalogue candidates ───────────────────────────────────────

def test_name_scores_say_priest_is_not_prince():
    assert mc.name_score("Priest", "Prince") < mc.ARTIST_SURE
    assert not mc.plausible_correction("Priest", "Prince")
    # real mishearings stay fixable
    assert mc.plausible_correction("saying chaos", "Saint Chaos")
    assert mc.plausible_correction("the set off", "Set It Off")
    assert mc.plausible_correction("sighted off", "Set It Off")          # same consonants
    assert not mc.plausible_correction("drake", "Blake")


def test_only_a_similar_named_artists_song_is_a_suggestion_never_the_answer():
    m = mc.pick_song([PRINCE], ["Call My Name"], "Priest")
    assert m is not None and not m.matched and not m.sure
    assert m.near == (PRINCE,) and "Prince" in m.why


def test_the_named_artist_wins_when_both_are_there():
    m = mc.pick_song([PRINCE, PRIEST], ["Call My Name"], "Priest")
    assert m.matched and m.sure and m.track.artist == "Priest"


def test_same_title_by_different_artists_without_an_artist_is_only_sure_at_the_top():
    m = mc.pick_song([PRINCE, PRIEST], ["Call My Name"])
    assert m.track.artist == "Prince" and m.sure           # the catalogue's own first, exact
    m = mc.pick_song([PRINCE, PRIEST], ["Call My Nam"])
    assert not m.sure and "other artist" in m.why          # not exact: which one? ask


def test_a_credit_with_the_named_artist_counts():
    duo = mc.CatalogTrack("Betty (Get Money)", "Yung Gravy & bbno$", 180)
    m = mc.pick_song([duo], ["Betty Get Money"], "Yung Gravy")
    assert m.matched and m.track is duo


# ── the catalogue itself, as the brief's reproductions put it ─────────────────

class PriestCatalog(FakeCatalog):
    """Deezer knows Prince's "Call My Name" (popular) and — depending on the
    test — the band Priest and their song, or neither."""

    def __init__(self, *, priest_known=True, priest_song=True):
        super().__init__()
        self.priest_known, self.priest_song = priest_known, priest_song
        self.queries = []

    async def handler(self, request):
        from aiohttp import web
        path, q = request.path, request.query
        if path.startswith("/deezer"):
            self.hits.append(path)
            if path == "/deezer/search":
                self.queries.append(q.get("q", ""))
                data = [{"title": "Call My Name", "duration": 300, "artist": {"name": "Prince"}}]
                if self.priest_song and "priest" in q.get("q", "").lower():
                    data.append({"title": "Call My Name", "duration": 240,
                                 "artist": {"name": "Priest"}})
                return web.json_response({"data": data})
            if path == "/deezer/search/artist":
                name = q.get("q", "").lower()
                out = []
                if "priest" in name and self.priest_known:
                    out.append({"id": 7, "name": "Priest", "nb_fan": 900})
                if "prince" in name:
                    out.append({"id": 8, "name": "Prince", "nb_fan": 3_000_000})
                return web.json_response({"data": out})
            if path == "/deezer/artist/7/top":
                return web.json_response({"data": [
                    {"title": "Call My Name", "duration": 240, "artist": {"name": "Priest"}}]
                    if self.priest_song else []})
            if path == "/deezer/artist/8/top":
                return web.json_response({"data": [
                    {"title": "Call My Name", "duration": 300, "artist": {"name": "Prince"}},
                    {"title": "Purple Rain", "duration": 500, "artist": {"name": "Prince"}}]})
            return web.json_response({"data": []})
        if path == "/itunes/search":
            return web.json_response({"results": []})
        return web.json_response({"data": []})


def test_priest_as_said_is_found_even_when_the_model_rewrote_it_to_prince():
    async def go():
        async with PriestCatalog() as srv:
            m = await mc.find_song("Call My Name", "Prince", heard_artist="Priest",
                                   deezer=srv.deezer, itunes=srv.itunes)
            assert m.matched and m.sure and m.track.artist == "Priest"
            assert m.basis == "said"
    run(go())


def test_priest_unknown_to_the_catalogue_is_never_answered_with_prince():
    async def go():
        async with PriestCatalog(priest_known=False, priest_song=False) as srv:
            m = await mc.find_song("Call My Name", "Prince", heard_artist="Priest",
                                   deezer=srv.deezer, itunes=srv.itunes)
            # Only Prince's song exists here: a suggestion at most, never a match.
            assert m is None or not m.matched
            assert not any("prince" in q.lower() for q in srv.queries)   # not even searched
    run(go())


def test_the_exact_artist_missing_from_the_search_results_is_not_replaced():
    async def go():
        async with PriestCatalog(priest_known=True, priest_song=False) as srv:
            m = await mc.find_song("Call My Name", "Priest", heard_artist="Priest",
                                   deezer=srv.deezer, itunes=srv.itunes)
            assert m is None or (not m.matched and m.track.artist == "Prince")
    run(go())


def test_a_catalogue_outage_is_an_error_not_a_guess():
    async def go():
        async with PriestCatalog():
            with pytest.raises(mc.CatalogError):
                await mc.find_song("Call My Name", "Priest", deezer="http://127.0.0.1:9/deezer",
                                   itunes="http://127.0.0.1:9/itunes")
    run(go())


# ── the upload: the song, or nothing ──────────────────────────────────────────

def _up(title, channel="", duration=0):
    return {"title": title, "uploader": channel, "duration": duration}


def test_an_upload_needs_the_artist_not_just_the_right_length():
    # same words, right length, someone else's channel and no artist named:
    # weak — it loses to the real one, and alone it isn't proof
    stranger = _up("Call My Name (Official Audio)", "Random Uploads", 240)
    real = _up("Priest - Call My Name", "Priest - Topic", 240)
    assert audio.pick_upload([stranger, real], "Call My Name", "Priest", 240) is real
    assert audio.pick_upload([_up("Call Me Maybe", "Random", 240)], "Call My Name", "Priest",
                             240) is None
    assert audio.upload_score(_up("Prince - Call My Name", "Prince", 300), "Call My Name",
                              "Priest", 240) is None


def test_a_search_for_a_song_the_catalogue_doesnt_know_must_show_it():
    junk = [_up("MA. Nacik Doll BH kaimara narasimharajapura. contact number 8105170528",
                "someone"),
            _up("WAX - ALTE WUNDEN", "WAX")]
    need = audio.UPLOAD_ACCEPT_UNKNOWN
    assert audio.pick_upload(junk, "Filium", "Wax and Million", min_score=need) is None
    # "Kaimara by" (no artist said): the junk video has the word, but the
    # planner asks who it's by instead of searching (tests/test_voice_exec.py)
    good = [_up("Waxx & Milli - Filium (Official Video)", "Waxx & Milli")]
    assert audio.pick_upload(good, "Filium", "Waxx & Milli", min_score=need) is good[0]


def test_other_takes_lose_unless_asked_for():
    live = _up("Priest - Call My Name (Live at Sweden Rock)", "Priest", 260)
    studio = _up("Priest - Call My Name", "Priest - Topic", 240)
    assert audio.pick_upload([live, studio], "Call My Name", "Priest", 240) is studio


# ── featured artists in the request ("Low by Flo Rida ft. T-Pain") ──────────
#
# The planner names a song's whole credit, featured artists included; the
# catalogue files it under the lead artist with the feature in the title
# ("Low (feat. T-Pain)" by Flo Rida) or lists every credit ("Lil Nas X, Billy
# Ray Cyrus & Diplo"). Live on 2026-10-27, four of a 24-song request were
# refused this way.

@pytest.mark.parametrize("named, credited", [
    ("Flo Rida ft. T-Pain", "Flo Rida"),
    ("Mark Ronson ft. Bruno Mars", "Mark Ronson"),
    ("Gotye feat. Kimbra", "Gotye"),
    ("Gotye featuring Kimbra", "Gotye"),
    ("Mark Ronson (feat. Bruno Mars)", "Mark Ronson"),
    ("Lil Nas X ft. Billy Ray Cyrus", "Lil Nas X, Billy Ray Cyrus & Diplo"),
    ("Lil Nas X", "Lil Nas X, Billy Ray Cyrus"),
])
def test_a_named_feature_doesnt_hide_the_lead_artist(named, credited):
    assert mc.name_score(named, credited) >= mc.ARTIST_SURE


@pytest.mark.parametrize("named, credited", [
    ("Prince ft. Somebody", "Priest"),          # the lead still has to be them
    ("Kimbra", "Gotye"),
    ("Billy Ray Cyrus", "Lil Nas X, Billy Ray Cyrus & Diplo"),   # a feature isn't the lead
])
def test_featuring_doesnt_loosen_who_the_artist_is(named, credited):
    assert mc.name_score(named, credited) < mc.ARTIST_SURE


def test_a_song_named_with_its_feature_is_found():
    low = mc.CatalogTrack("Low (feat. T-Pain)", "Flo Rida", 231)
    other = mc.CatalogTrack("Low", "Ciara", 200)
    m = mc.pick_song([other, low], ["Low"], "Flo Rida ft. T-Pain")
    assert m.matched and m.sure and m.track is low


def test_an_upload_counts_the_lead_artist_of_a_featured_credit():
    up = _up("Flo Rida - Low feat. T-Pain [Official Video]", "Flo Rida", 231)
    assert audio.upload_score(up, "Low", "Flo Rida ft. T-Pain", 231) is not None
    assert audio.upload_score(_up("Low (Official Video)", "Ciara", 200), "Low",
                              "Flo Rida ft. T-Pain", 231) is None


class GotyeCatalog(FakeCatalog):
    """Like Deezer: an artist search for "Gotye ft. Kimbra" finds nobody, and a
    track search with the feature in it turns up a popular cover first."""

    def __init__(self):
        super().__init__()
        self.artist_queries = []

    async def handler(self, request):
        from aiohttp import web
        path, q = request.path, request.query
        song = {"title": "Somebody That I Used To Know", "duration": 244,
                "artist": {"name": "Gotye"}}
        cover = {"title": "Somebody That I Used To Know", "duration": 250,
                 "artist": {"name": "Kurt Hugo Schneider"}}
        if path == "/deezer/search":
            text = q.get("q", "").lower()
            return web.json_response({"data": [cover] + ([song] if "kimbra" not in text else [])})
        if path == "/deezer/search/artist":
            self.artist_queries.append(q.get("q", ""))
            name = q.get("q", "").lower()
            return web.json_response({"data": [{"id": 9, "name": "Gotye", "nb_fan": 500_000}]
                                      if name == "gotye" else []})
        if path == "/deezer/artist/9/top":
            return web.json_response({"data": [song]})
        if path == "/itunes/search":
            return web.json_response({"results": []})
        return web.json_response({"data": []})


def test_a_request_naming_the_feature_finds_the_lead_artists_song():
    async def go():
        async with GotyeCatalog() as srv:
            m = await mc.find_song("Somebody That I Used to Know", "Gotye ft. Kimbra",
                                   deezer=srv.deezer, itunes=srv.itunes)
            assert m is not None and m.matched and m.sure
            assert m.track.artist == "Gotye"
            assert "Gotye" in srv.artist_queries
    run(go())
