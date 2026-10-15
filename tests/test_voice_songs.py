"""Songs by voice: the right song, or a plain "couldn't find it" — never a
random upload.

Reported 2026-10-12, all through the AI tier:
  * "Play I Write Him's Not Travis D's by Young Gravy and skip the song" →
    played Yung Gravy's "Mr. Clean"; said again more clearly → a Catholic
    hymn. (The song: Yung Gravy's "I Write Hymns Not Travesties", newer than
    the model.)
  * "Can you play that one Yung Gravy song that samples Panic! At The Disco
    and skip this song?" → The Killers' "Mr. Brightside".
  * "I played cordial by sighted off" → the Set It Off song already playing.

The real VoiceDJ cog through the plan rig (tests/test_voice_plans.py), with a
scripted LLM, the real catalogue matching (core.music_catalog.pick_song) over
a fixed song list, and scripted YouTube search results."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

import cogs.voicedj as vd                                       # noqa: E402
from core import music_catalog as mc                            # noqa: E402
from core.state import get_state                                # noqa: E402
from core.voice_intent import Intent, interpret                 # noqa: E402
from core.voice_llm import LLMReply                             # noqa: E402
from tests.test_voice_plans import _playing, _titles, actions, planned, replies  # noqa: E402
from tests.test_voice_smart import FakeLLM                      # noqa: E402

HYMNS = "I Write Hymns Not Travesties"
CATALOG = [mc.CatalogTrack(t, "Yung Gravy", 150) for t in (
    "Mr. Clean", "Betty (Get Money)", "Martha Stewart", HYMNS, "oops!")] + [
    mc.CatalogTrack(t, "Set It Off", 200) for t in ("No Disrespect", "Cordial", "Skeleton")] + [
    mc.CatalogTrack("Danger Zone", "Kenny Loggins", 216)]

UPLOADS = {   # what YouTube answers for a search, first result first
    "Yung Gravy - I Write Hymns Not Travesties": [
        ("Yung Gravy - Mr. Clean (Official Video)", "Yung Gravy", 160),
        ("Yung Gravy - i write hymns not travesties (Official Audio)", "Yung Gravy", 151),
        ("Be Not Afraid | Catholic Hymn", "Sunday Choir", 240)],
    "Set It Off - Cordial": [
        ("Set It Off - No Disrespect", "Set It Off", 200),
        ("Set It Off - Cordial (Official Audio)", "Set It Off - Topic", 199)],
    "Kenny Loggins - Danger Zone": [
        ("Kenny Loggins - Danger Zone (Official Video - Top Gun)", "KennyLogginsVEVO", 218)],
}


class SongLLM(FakeLLM):
    web, web_per_hour = True, 30

    def __init__(self, *replies, choices=(), lookups=()):
        super().__init__(*replies)
        self.choices, self.lookups = list(choices), list(lookups)
        self.chosen, self.looked_up = [], []

    async def choose_song(self, said, wanted, songs):
        self.chosen.append((said, wanted, [s.title for s in songs]))
        pick = self.choices.pop(0) if self.choices else None
        return None if pick is None else [s.title for s in songs].index(pick)

    async def lookup_song(self, description, artist=""):
        self.looked_up.append((description, artist))
        return self.lookups.pop(0) if self.lookups else None


def songs_rig(fn):
    """The plan rig + a catalogue that matches for real over CATALOG, and
    scripted YouTube results."""
    @planned
    async def wrapper(rig):
        rig.searched, rig.fetched, rig.song_calls = [], [], []
        rig.catalog_down = False
        rig.force_unsure = False

        async def find_song(title, artist="", *, heard="", heard_artist=""):
            rig.song_calls.append((title, artist, heard))
            if rig.catalog_down:
                raise mc.CatalogError("deezer: down; itunes: down")
            m = mc.pick_song(CATALOG, [title, heard], artist, said=heard)
            if m is not None and rig.force_unsure:
                m = mc.SongMatch(m.track, m.score, False, m.others)
            return m

        async def artist_songs(artist):
            return [t for t in CATALOG if mc.name_score(artist, t.artist) >= 0.75]

        async def search_tracks(query, *, limit=5):
            rig.searched.append(query)
            return [{"title": t, "uploader": ch, "duration": d, "webpage_url": f"https://yt/{n}",
                     "stream_url": None, "_needs_resolve": True, "_search_query": f"https://yt/{n}"}
                    for n, (t, ch, d) in enumerate(UPLOADS.get(query, []))]

        async def fetch(query):
            rig.fetched.append(query)
            return {"title": f"(top result for) {query}", "stream_url": "s://x",
                    "webpage_url": "https://e/x", "duration": 100, "_needs_resolve": False}
        rig.monkeypatch.setattr(vd.music_catalog, "find_song", find_song)
        rig.monkeypatch.setattr(vd.music_catalog, "artist_songs", artist_songs)
        rig.monkeypatch.setattr(vd.audio, "search_tracks", search_tracks)
        rig.monkeypatch.setattr(vd.audio, "fetch_track", fetch)
        await fn(rig)
    wrapper.__name__ = fn.__name__
    return wrapper


SKIP_AND = (Intent("skip"),)


@songs_rig
async def test_a_misheard_title_plays_the_artists_real_song(rig):
    st = _playing(rig, "Queued Before")
    # As the model answered it: the song "now" (the skip folds into that).
    rig.cog.llm = SongLLM(LLMReply("commands", (
        Intent("play", "I Write Him's Not Travis D's", artist="Young Gravy", when="now"),
        Intent("skip"))))
    rig.say("hey dj play I write him's not travis d's by young gravy and skip the song")
    await rig.settle(0.5)
    assert rig.searched == ["Yung Gravy - I Write Hymns Not Travesties"]
    assert rig.fetched == []                        # never searched as said
    assert st.current["title"] == "Yung Gravy - i write hymns not travesties (Official Audio)"
    assert _titles(st) == ["Queued Before"]
    assert replies(rig) == ["⏭️ Skipped — playing **Yung Gravy - i write hymns not travesties "
                            "(Official Audio)** now."]


@songs_rig
async def test_the_words_said_overrule_the_ais_swap(rig):
    st = _playing(rig)
    rig.cog.llm = SongLLM(LLMReply("commands", (
        Intent("play", "No Disrespect", artist="Set It Off", heard="cordial"),)))
    rig.say("hey dj I played cordial by sighted off")
    await rig.settle(0.5)
    assert rig.song_calls == [("No Disrespect", "Set It Off", "cordial")]
    assert _titles(st) == ["Set It Off - Cordial (Official Audio)"]


@songs_rig
async def test_youtube_without_the_song_says_so_and_skips_nothing(rig):
    st = _playing(rig, "Queued Before")
    UPLOADS["Yung Gravy - Martha Stewart"] = [("Martha Stewart cooking show", "Food TV", 900)]
    rig.cog.llm = SongLLM(LLMReply("commands", (
        Intent("skip"), Intent("play", "Martha Stewart", artist="Yung Gravy"))))
    try:
        rig.say("hey dj skip this and play martha stewart by yung gravy")
        await rig.settle(0.5)
    finally:
        UPLOADS.pop("Yung Gravy - Martha Stewart")
    assert rig.searched == ["Yung Gravy - Martha Stewart", "Yung Gravy Martha Stewart audio"]
    assert rig.vc.stopped == 0 and st.current["title"] == "Something Else"
    assert _titles(st) == ["Queued Before"]
    assert replies(rig) == ["❌ Couldn't find **Martha Stewart** by Yung Gravy on YouTube.\n"
                            "↪️ So I didn't skip this song."]


@songs_rig
async def test_an_unsure_match_is_settled_by_the_ai(rig):
    st = _playing(rig)
    rig.force_unsure = True
    rig.cog.llm = SongLLM(LLMReply("commands", (
        Intent("play", "hymns not travis", artist="Yung Gravy"),)), choices=[HYMNS])
    rig.say("hey dj play hymns not travis by yung gravy")
    await rig.settle(0.5)
    (said, wanted, offered), = rig.cog.llm.chosen
    assert said.startswith("hey dj play hymns") and wanted == "hymns not travis by Yung Gravy"
    assert offered[0] == HYMNS
    assert _titles(st) == ["Yung Gravy - i write hymns not travesties (Official Audio)"]
    assert rig.cog.metrics.guild(vd_gid(rig)).counts["song_asked_ai"] == 1


@songs_rig
async def test_when_the_ai_says_none_of_them_it_asks_rather_than_guessing(rig):
    # Round 8 (the brief): an uncertain match never becomes "the first YouTube
    # result" — it's a checked search (title and artist must show), else a
    # question. Nothing here shows "hymns not travis" by Yung Gravy.
    st = _playing(rig)
    rig.force_unsure = True
    rig.cog.llm = SongLLM(LLMReply("commands", (
        Intent("play", "hymns not travis", artist="Yung Gravy"),)), choices=[None])
    rig.say("hey dj play hymns not travis by yung gravy")
    await rig.settle(0.5)
    assert rig.fetched == []                         # never the unchecked top result
    assert rig.searched == ["Yung Gravy - hymns not travis"]
    assert _titles(st) == []
    (reply,) = replies(rig)
    assert reply.startswith("❓ Which one did you mean — ")
    assert (GID, UID) in rig.cog._armed and rig.cog._armed[(GID, UID)].question


@songs_rig
async def test_a_described_song_is_picked_from_the_artists_songs(rig):
    st = _playing(rig, "Queued Before")
    rig.cog.llm = SongLLM(LLMReply("commands", (
        Intent("find", "Yung Gravy song that samples Panic! At The Disco", artist="Yung Gravy",
               when="now"), Intent("skip"))), choices=[HYMNS])
    rig.say("hey dj can you play that one young gravy song that samples panic at the disco "
            "and skip this song")
    await rig.settle(0.5)
    (_said, wanted, offered), = rig.cog.llm.chosen
    assert wanted == "a song they described: Yung Gravy song that samples Panic! At The Disco"
    assert HYMNS in offered and "Cordial" not in offered        # only that artist's songs
    assert rig.cog.llm.looked_up == []                           # no web search needed
    assert st.current["title"] == "Yung Gravy - i write hymns not travesties (Official Audio)"
    assert replies(rig) == ["🔎 That's **I Write Hymns Not Travesties** by Yung Gravy. "
                            "⏭️ Skipped — playing **Yung Gravy - i write hymns not travesties "
                            "(Official Audio)** now."]


@songs_rig
async def test_a_described_song_is_looked_up_online_when_the_list_doesnt_have_it(rig):
    st = _playing(rig)
    rig.cog.llm = SongLLM(LLMReply("commands", (Intent("find", "the song from Top Gun"),)),
                          lookups=[("Danger Zone", "Kenny Loggins")])
    rig.say("hey dj play the song from top gun")
    await rig.settle(0.5)
    assert rig.cog.llm.looked_up == [("the song from Top Gun", "")]
    assert _titles(st) == ["Kenny Loggins - Danger Zone (Official Video - Top Gun)"]
    assert replies(rig)[0].startswith("🔎 That's **Danger Zone** by Kenny Loggins. ✅ Queued")


@songs_rig
async def test_a_description_nobody_can_place_is_said_plainly(rig):
    st = _playing(rig, "Queued Before")
    rig.cog.llm = SongLLM(LLMReply("commands", (
        Intent("skip"), Intent("find", "that song that goes da da da"))), lookups=[None])
    rig.say("hey dj skip this and play that song that goes da da da")
    await rig.settle(0.5)
    assert rig.vc.stopped == 0 and _titles(st) == ["Queued Before"]
    assert replies(rig) == ["❌ I couldn't work out which song “that song that goes da da da” "
                            "is — try saying its name.\n↪️ So I didn't skip this song."]


@songs_rig
async def test_catalogue_down_falls_back_to_a_checked_search(rig):
    # With the catalogue unreachable the song is still found — but only an
    # upload showing both the title and the artist is taken: here YouTube's
    # first answer is another Set It Off song, and it loses to Cordial.
    st = _playing(rig)
    rig.catalog_down = True
    rig.cog.llm = SongLLM(LLMReply("commands", (Intent("play", "Cordial", artist="Set It Off"),)))
    rig.say("hey dj play cordial by set it off")
    await rig.settle(0.5)
    assert rig.fetched == [] and rig.searched == ["Set It Off - Cordial"]
    assert _titles(st) == ["Set It Off - Cordial (Official Audio)"]
    assert rig.cog.metrics.guild(vd_gid(rig)).counts["catalog_failed"] == 1


@songs_rig
async def test_catalogue_down_and_no_upload_shows_it_asks(rig):
    st = _playing(rig)
    rig.catalog_down = True
    rig.cog.llm = SongLLM(LLMReply("commands", (Intent("play", "Filium", artist="Wax and Million"),)))
    rig.say("hey dj play filium by wax and million")
    await rig.settle(0.5)
    assert rig.fetched == [] and _titles(st) == []
    (reply,) = replies(rig)
    assert "couldn't find **Filium** by Wax and Million" in reply and "catalogue" in reply


from tests.test_voicedj_flow import GID, UID  # noqa: E402


def vd_gid(rig):
    from tests.test_voicedj_flow import GID
    return GID


def test_tip_this_is_skip_this():
    # "Tip this." (reported 2026-10-12) went to the AI, which answered a
    # question about tips. Misheard command words map in the grammar.
    for said in ("hey dj tip this", "hey dj, tip this.", "hey dj kip it"):
        assert interpret(said).intent.action == "skip"
    assert interpret("tip this", in_window=True).intent.action == "skip"


def test_find_counts_as_a_song_request_in_plans():
    assert "find" in vd.SONG_ACTIONS
    assert vd.VoiceDJ._describe(Intent("find", "the song from Top Gun")) == \
        "Find and play the song from Top Gun"
    from core import voice_plan as vp
    merged = vp.fold_skips(vp.from_intents((Intent("skip"), Intent("find", "x")))).steps
    assert [(st.op, st.value, st.when, st.took_skip) for st in merged] == [
        ("find_song", "x", "now", True)]


# ── which YouTube upload is the song ─────────────────────────────────────────

from core import audio  # noqa: E402


def _up(title, channel="", duration=0):
    return {"title": title, "uploader": channel, "duration": duration}


def test_the_upload_that_is_the_song_beats_youtubes_first_hit():
    got = audio.pick_upload([_up("Yung Gravy - Mr. Clean (Official Video)", "Yung Gravy", 160),
                             _up("Be Not Afraid | Catholic Hymn", "Choir", 240),
                             _up("Yung Gravy - i write hymns not travesties", "Yung Gravy", 151)],
                            HYMNS, "Yung Gravy", 150)
    assert got["title"] == "Yung Gravy - i write hymns not travesties"


def test_no_upload_of_the_song_is_none_rather_than_something_else():
    assert audio.pick_upload([_up("Yung Gravy - Mr. Clean", "Yung Gravy"),
                              _up("Be Not Afraid | Catholic Hymn", "Choir")],
                             HYMNS, "Yung Gravy") is None
    # the same words by someone else, and nothing else to go on
    assert audio.pick_upload([_up("Martha Stewart cooking show", "Food TV", 900)],
                             "Martha Stewart", "Yung Gravy", 150) is None


def test_the_artists_channel_and_topic_channels_count():
    got = audio.pick_upload([_up("Cordial", "Some Lyrics Channel", 400),
                             _up("Cordial", "Set It Off - Topic", 199)], "Cordial", "Set It Off", 200)
    assert got["uploader"] == "Set It Off - Topic"


@pytest.mark.parametrize("other", ["Skeleton (Slowed + Reverb)", "Skeleton - Set It Off (cover)",
                                   "Set It Off - Skeleton (Live at the Warfield)",
                                   "Skeleton - Set It Off | Nightcore",
                                   "Set It Off - Skeleton (Karaoke Version)"])
def test_other_takes_lose_to_the_song_unless_asked_for(other):
    ups = [_up(other, "Set It Off", 200), _up("Set It Off - Skeleton", "Set It Off", 200)]
    assert audio.pick_upload(ups, "Skeleton", "Set It Off", 200)["title"] == "Set It Off - Skeleton"
    assert audio.pick_upload(ups[:1] + [_up("Set It Off - Skeleton", "Set It Off", 200)],
                             "Skeleton (Slowed & Reverb)" if "Slowed" in other else "Skeleton",
                             "Set It Off", 200)


def test_a_matching_length_settles_near_ties():
    ups = [_up("Set It Off - Cordial", "Set It Off", 330), _up("Set It Off - Cordial", "Set It Off", 199)]
    assert audio.pick_upload(ups, "Cordial", "Set It Off", 200)["duration"] == 199
