"""Several things in one spoken request, done in the order said — and songs
named by what they are rather than searched as said.

"Play the first three songs off Panic! At The Disco's Vices and Virtues
album, turn the volume down to 50 percent, shuffle the queue, skip the first
song, then set the other two songs to loop" must: look the album up (tracks
1-3, in album order), queue them, set the volume, shuffle, skip, then loop
the two that are left — each step after the one before.

The real VoiceDJ cog through the flow rig (tests/test_voicedj_flow.py), with
a scripted LLM, a stand-in music catalogue and a stand-in music player whose
queue advances like the real one. These prove the decisions and their order,
not what a real model or catalogue answers (tests/test_voice_llm.py,
tests/test_music_catalog.py and the measurements in core/voice_llm.py do)."""
from __future__ import annotations

import asyncio
import random
import time

import pytest

np = pytest.importorskip("numpy")

import cogs.voicedj as vd                                       # noqa: E402
from core import music_catalog as mc                            # noqa: E402
from core.state import clear_queue, get_state, requeue_copy     # noqa: E402
from core.voice_intent import Intent                            # noqa: E402
from core.voice_llm import LLMReply                             # noqa: E402
from tests.test_voice_smart import FakeChannel, FakeLLM, FakeTTS  # noqa: E402
from tests.test_voicedj_flow import GID, UID, Rig               # noqa: E402

VV = mc.CatalogAlbum("Vices & Virtues", "Panic! At the Disco", tuple(
    mc.CatalogTrack(t, "Panic! At the Disco", 200, n) for n, t in enumerate(
        ["The Ballad of Mona Lisa", "Let's Kill Tonight", "Hurricane", "Memories",
         "Trade Mistakes", "Ready to Go", "Always", "The Calendar", "Sarah Smiles",
         "Nearly Witches"], 1)), cover="https://img/vv.jpg")
FIRST_THREE = ["Panic! At the Disco - The Ballad of Mona Lisa",
               "Panic! At the Disco - Let's Kill Tonight", "Panic! At the Disco - Hurricane"]


class FakeMusic:
    """The parts of the music cog the voice cog drives: a kickstart takes the
    next queued song as "starting", like _advance_locked does."""

    def __init__(self):
        self.kicks = 0

    def _busy(self, guild):
        return get_state(guild.id).current is not None

    def _kickstart(self, guild, vc):
        self.kicks += 1
        st = get_state(guild.id)

        def advance():
            if st.current is None and st.queue:
                st.current = st.queue.popleft()
                st.current["_status"] = "starting"
        asyncio.get_running_loop().call_soon(advance)    # scheduled, like the real one

    @staticmethod
    def _abandon_starting(state):
        cur = state.current
        if cur is not None and cur.get("_status") in ("starting", "retrying"):
            cur["_status"] = "skipped"
            if state.loop_queue:
                state.queue.append(requeue_copy(cur))
            state.current = None
            return True
        return False


def planned(fn):
    """Flow rig + reply channel/TTS/LLM stand-ins + a music player + a catalogue."""
    def wrapper(monkeypatch):
        async def go():
            rig = Rig(monkeypatch)
            rig.monkeypatch = monkeypatch
            rig.music = FakeMusic()
            real_get_cog = rig.bot.get_cog
            rig.bot.get_cog = lambda name: rig.music if name == "Music" else real_get_cog(name)
            rig.albums = {"vices and virtues": VV}
            rig.catalog_calls = []
            rig.catalog_gate = None
            rig.catalog_error = None

            async def find_album(title, artist=""):
                rig.catalog_calls.append(("album", title, artist))
                if rig.catalog_gate is not None:
                    await rig.catalog_gate.wait()
                if rig.catalog_error:
                    raise rig.catalog_error
                return rig.albums.get(mc.norm(title))

            async def top_tracks(artist, n=5):
                rig.catalog_calls.append(("top", artist, n))
                return [mc.CatalogTrack(f"Hit {i}", artist, 180) for i in range(1, n + 1)]

            async def latest_tracks(artist, n=1):
                rig.catalog_calls.append(("latest", artist, n))
                return [mc.CatalogTrack(f"New {i}", artist, 180) for i in range(1, n + 1)]
            monkeypatch.setattr(vd.music_catalog, "find_album", find_album)
            monkeypatch.setattr(vd.music_catalog, "top_tracks", top_tracks)
            monkeypatch.setattr(vd.music_catalog, "latest_tracks", latest_tracks)

            async def fetch(query):
                return {"title": query, "stream_url": "s://x", "webpage_url": "https://e/x",
                        "duration": 100, "_needs_resolve": False, "_resolved_at": time.time()}
            monkeypatch.setattr(vd.audio, "fetch_track", fetch)
            rig.channel_txt = FakeChannel()

            async def echo(guild):
                return rig.channel_txt
            rig.cog._echo_channel = echo
            rig.tts = FakeTTS()
            rig.cog._tts = lambda: rig.tts
            rig.cog._cfg[GID].update(ai=True, chat=True)
            st = get_state(GID)
            st.current, st.loop, st.loop_queue = None, False, False
            rig.vc = rig.guilds[GID].voice_client
            rig.vc.playing = False                       # an idle player
            try:
                await fn(rig)
            finally:
                for key in list(rig.cog._typing):
                    rig.cog._typing_stop(key)
                for t in list(rig.cog._play_workers.values()):
                    t.cancel()
                st.current, st.loop, st.loop_queue = None, False, False
                st.queue.clear()
                await rig.close()
        asyncio.run(go())
    wrapper.__name__ = fn.__name__
    return wrapper


def actions(rig):
    return [a for _g, _u, a, _x in rig.executed]


def replies(rig):
    return [r for _g, _u, r, _f in rig.feedback if r]


ISSUE = ("Hey DJ, play the first three songs off Panic! At The Disco's Vices and Virtues "
         "album, turn the volume down to 50 percent, shuffle the queue, skip the first song, "
         "then set the other two songs to loop")
ISSUE_PLAN = (Intent("album", "Vices and Virtues", artist="Panic! At The Disco", count="1-3"),
              Intent("volume", "50"), Intent("shuffle"), Intent("skip"), Intent("loop", "queue"))


@planned
async def test_the_request_from_the_issue_runs_in_order(rig):
    random.seed(3)
    rig.cog.llm = FakeLLM(LLMReply("commands", ISSUE_PLAN))
    rig.say(ISSUE)
    await rig.settle(1.0)
    st = get_state(GID)
    assert rig.cog.llm.asked[0][0].startswith("play the first three songs off Panic!")
    assert rig.catalog_calls == [("album", "Vices and Virtues", "Panic! At The Disco")]
    assert actions(rig) == ["album", "volume", "shuffle", "skip", "loop"]
    # The first song started, was skipped; the other two are what loops.
    left = [st.current["title"]] + [t["title"] for t in st.queue]
    assert sorted(left) == sorted(FIRST_THREE[1:])
    assert st.volume == 0.5 and st.loop_queue and not st.loop
    # One reply for the whole request, step by step.
    (reply,) = replies(rig)
    assert reply.splitlines()[0].startswith("✅ Queued 3 songs from **Vices & Virtues**")
    for bit in ("Volume 50%", "Shuffled 2 songs", "Skipped", "Looping the queue (2 songs)"):
        assert bit in reply


@planned
async def test_queued_album_songs_are_found_on_youtube_when_their_turn_comes(rig):
    rig.vc.playing = True                               # something else is on
    get_state(GID).current = {"title": "Already Playing", "_status": "playing"}
    rig.cog.llm = FakeLLM(LLMReply("commands", ISSUE_PLAN[:1]))
    rig.say("hey dj queue up the first three off vices and virtues")
    await rig.settle(0.5)
    queued = list(get_state(GID).queue)
    assert [t["title"] for t in queued] == FIRST_THREE
    first = queued[0]
    assert first["_needs_resolve"] and first["stream_url"] is None
    assert first["_search_query"] == FIRST_THREE[0]
    assert first["webpage_url"].startswith("https://www.youtube.com/results?search_query=")
    assert first["requester_id"] == UID and first["thumbnail"] == "https://img/vv.jpg"
    assert rig.music.kicks == 0                         # it was busy: nothing restarted
    assert "Starting" not in replies(rig)[0]


@planned
async def test_a_missing_album_leaves_out_what_depended_on_it(rig):
    rig.albums.clear()
    rig.cog.llm = FakeLLM(LLMReply("commands", ISSUE_PLAN))
    rig.say(ISSUE)
    await rig.settle(0.5)
    assert actions(rig) == ["album", "volume"]          # the volume change still happened
    (reply,) = replies(rig)
    assert "Couldn't find the album “Vices and Virtues” by Panic! At The Disco" in reply
    assert ("So I didn't shuffle the queue, skip, loop the queue — play 1-3 of Vices and "
            "Virtues by Panic! At The Disco didn't work, so it would have acted on other "
            "songs.") in reply
    assert not get_state(GID).loop_queue


@planned
async def test_catalogue_down_is_said_plainly(rig):
    rig.catalog_error = mc.CatalogError("deezer: HTTP 503; itunes: timeout")
    rig.cog.llm = FakeLLM(LLMReply("commands", ISSUE_PLAN[:1]))
    rig.say("hey dj play the first three songs off vices and virtues")
    await rig.settle(0.5)
    assert replies(rig) == ["❌ Couldn't reach the music catalogue just now — try again in a bit."]


@planned
async def test_unclear_track_choice_is_asked_back(rig):
    rig.cog.llm = FakeLLM(LLMReply("commands", (
        Intent("album", "Vices and Virtues", artist="Panic!", count="the good ones"),)))
    rig.say("hey dj play the good ones off vices and virtues")
    await rig.settle(0.5)
    assert "has 10 songs — which ones?" in replies(rig)[0]
    assert not get_state(GID).queue


@planned
async def test_a_typed_clear_mid_request_calls_off_the_rest(rig):
    rig.catalog_gate = asyncio.Event()
    rig.cog.llm = FakeLLM(LLMReply("commands", ISSUE_PLAN))
    rig.say(ISSUE)
    for _ in range(100):                                 # until the album lookup is waiting
        await asyncio.sleep(0.02)
        if rig.catalog_calls:
            break
    clear_queue(GID)                                     # what typed `-clear` does …
    rig.cog._cancel_plays(GID, reason="clear")           # … and tells the voice cog
    rig.catalog_gate.set()
    await rig.settle(0.5)
    assert actions(rig) == ["album"]                     # started, then called off
    assert not get_state(GID).queue and not get_state(GID).loop_queue
    assert replies(rig) == ["✖ Didn't finish your request — the queue was cleared after you asked."]


@planned
async def test_most_popular_and_newest(rig):
    rig.vc.playing = True
    get_state(GID).current = {"title": "x", "_status": "playing"}
    rig.cog.llm = FakeLLM(
        LLMReply("commands", (Intent("top", None, artist="Owl City", count="3"),)),
        LLMReply("commands", (Intent("latest", None, artist="Owl City", count="1"),)))
    rig.say("hey dj play owl city's three biggest songs")
    await rig.settle(0.5)
    rig.say("hey dj play the newest owl city song", start=time.monotonic())
    await rig.settle(0.5)
    assert rig.catalog_calls == [("top", "Owl City", 3), ("latest", "Owl City", 1)]
    assert [t["title"] for t in get_state(GID).queue] == [
        "Owl City - Hit 1", "Owl City - Hit 2", "Owl City - Hit 3", "Owl City - New 1"]
    assert "most popular" in replies(rig)[0] and "newest" in replies(rig)[1]


@planned
async def test_controls_only_still_run_one_by_one(rig):
    rig.vc.playing = True
    get_state(GID).current = {"title": "x", "_status": "playing"}
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("volume", "down"), Intent("skip"))))
    rig.say("hey dj turn it down a bit and skip this one")
    await rig.settle(0.5)
    assert actions(rig) == ["volume", "skip"]
    # Controls only: the control lane, at once (not behind song lookups), in
    # order — reported together, as one request.
    assert replies(rig) == ["🔊 Volume 75%.\n⏭️ Skipped."]
    assert not rig.cog._plays.get(GID)


@planned
async def test_skip_two(rig):
    rig.vc.playing = True
    st = get_state(GID)
    st.current = {"title": "now", "_status": "playing"}
    st.queue.extend({"title": t} for t in ("next", "after", "later"))
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("skip", count="2"),)))
    rig.say("hey dj skip the next two songs")
    await rig.settle(0.5)
    assert [t["title"] for t in st.queue] == ["after", "later"]
    assert st.current is None and rig.vc.stopped == 1
    assert replies(rig) == ["⏭️ Skipped 2 songs."]


@planned
async def test_skip_keeps_the_song_in_rotation_when_the_queue_loops(rig):
    rig.vc.playing = True
    st = get_state(GID)
    st.loop_queue = True
    st.current = {"title": "now", "webpage_url": "https://yt/now", "_status": "playing",
                  "stream_url": "s://old"}
    st.queue.extend({"title": t} for t in ("next",))
    rig.say("hey dj skip")
    await rig.settle(0.3)
    again = st.queue[-1]
    assert [t["title"] for t in st.queue] == ["next", "now"]
    assert again["_needs_resolve"] and again["stream_url"] is None
    assert again["_search_query"] == "https://yt/now"           # the same upload comes back
    assert "_status" not in again


@planned
async def test_a_stop_while_the_ai_was_thinking_is_reported(rig):
    llm = FakeLLM(LLMReply("commands", ISSUE_PLAN[:1]))
    real = llm.ask

    async def ask(said, *, now_playing=None):
        clear_queue(GID)                                    # someone typed -clear meanwhile
        return await real(said, now_playing=now_playing)
    llm.ask = ask
    rig.cog.llm = llm
    rig.say("hey dj play the first three off vices and virtues")
    await rig.settle(0.5)
    assert actions(rig) == [] and rig.catalog_calls == []
    assert replies(rig) == ["✖ Didn't do that — the queue was cleared while I was working it "
                            "out. Say it again if you still want it."]


@planned
async def test_a_question_is_not_dropped_by_a_queue_change(rig):
    llm = FakeLLM(LLMReply("question", reply="Eight."))
    real = llm.ask

    async def ask(said, *, now_playing=None):
        clear_queue(GID)
        return await real(said, now_playing=now_playing)
    llm.ask = ask
    rig.cog.llm = llm
    rig.say("hey dj how many legs does a spider have")
    await rig.settle(0.5)
    assert rig.tts.said == ["Eight."]


@planned
async def test_shuffle_and_loop_queue_by_plain_command(rig):
    random.seed(1)
    st = get_state(GID)
    st.queue.extend({"title": str(n)} for n in range(6))
    rig.say("hey dj shuffle the queue")
    await rig.settle(0.3)
    assert sorted(t["title"] for t in st.queue) == [str(n) for n in range(6)]
    assert [t["title"] for t in st.queue] != [str(n) for n in range(6)]
    rig.say("hey dj loop the queue", start=time.monotonic())
    await rig.settle(0.3)
    assert st.loop_queue and not st.loop
    rig.say("hey dj loop off", start=time.monotonic())
    await rig.settle(0.3)
    assert not st.loop_queue and not st.loop
    assert replies(rig)[0] == "🔀 Shuffled 6 songs."
    assert replies(rig)[1].startswith("🔁 Looping the queue (6 songs).")


@planned
async def test_a_dont_with_an_album_named_freely_still_plays_it(rig):
    rig.cog.llm = FakeLLM(LLMReply("commands", (
        Intent("album", "Vices and Virtues", artist="Panic! At The Disco", count="all"),)))
    rig.say("hey dj don't play that, play the whole vices and virtues album instead")
    await rig.settle(0.5)
    assert actions(rig) == ["album"]
    assert len(get_state(GID).queue) + (1 if get_state(GID).current else 0) == 10


# ── "skip this and play …": songs at the front, and a receive hiccup ──────────

PFTW = mc.CatalogAlbum("Pray for the Wicked", "Panic! At the Disco", tuple(
    mc.CatalogTrack(t, "Panic! At the Disco", 200, n) for n, t in enumerate(
        ["(Fuck A) Silver Lining", "Say Amen (Saturday Night)", "Hey Look Ma, I Made It",
         "High Hopes", "Roaring 20s", "Dancing's Not a Crime", "One of the Drunks",
         "The Overpass", "King of the Clouds", "Old Fashioned", "Dying in LA"], 1)))
REPORTED = ("Hey DJ, can you skip this song and instead can you play the seventh and ninth "
            "song off of Panic at the Discos Pray for the Wicked album?")


def _pftw(when="now", count="seventh and ninth"):
    return (Intent("skip"), Intent("album", "Pray for the Wicked", artist="Panic! At The Disco",
                                   count=count, when=when))


def _playing(rig, *queued):
    rig.albums["pray for the wicked"] = PFTW
    rig.vc.playing = True
    st = get_state(GID)
    st.current = {"title": "Something Else", "_status": "playing"}
    st.queue.extend({"title": t} for t in queued)
    return st


def _titles(st):
    return [t["title"] for t in st.queue]


@planned
async def test_skip_this_and_play_two_songs_off_an_album_instead(rig):
    # The request that came back as only a skip (2026-10-12): the two songs
    # must play now, ahead of what was already queued, with exactly one skip.
    st = _playing(rig, "Queued Before")
    rig.cog.llm = FakeLLM(LLMReply("commands", _pftw()))
    rig.say(REPORTED)
    await rig.settle(0.5)
    assert rig.cog.llm.asked[0][0].startswith("can you skip this song and instead")
    assert actions(rig) == ["album"]                     # the skip is the songs' "now"
    assert rig.vc.stopped == 1
    assert st.current["title"] == "Panic! At the Disco - One of the Drunks"
    assert _titles(st) == ["Panic! At the Disco - King of the Clouds", "Queued Before"]
    assert replies(rig) == ["⏭️ Skipped — playing 2 songs from **Pray for the Wicked** by "
                            "Panic! At the Disco now: One of the Drunks, King of the Clouds."]


@planned
async def test_a_receive_hiccup_mid_lookup_keeps_the_request(rig):
    # What most likely happened that time: voice receive dropped and came back
    # (DAVE) while the album was being looked up — which used to start a new
    # "session" and silently call the lookup off after the skip had run.
    st = _playing(rig, "Queued Before")
    rig.catalog_gate = asyncio.Event()
    rig.cog.llm = FakeLLM(LLMReply("commands", _pftw(count="7,9")))
    rig.say(REPORTED)
    for _ in range(100):
        await asyncio.sleep(0.02)
        if rig.catalog_calls:
            break
    session = rig.cog._session.get(GID, 0)
    rig.cog._on_receiver_session(GID, "stopped")
    rig.cog._on_receiver_session(GID, "recovered")
    rig.catalog_gate.set()
    await rig.settle(0.5)
    assert rig.cog._session.get(GID, 0) == session
    assert rig.cog.metrics.guild(GID).counts["receiver_stopped"] == 1
    assert st.current["title"] == "Panic! At the Disco - One of the Drunks"
    assert _titles(st) == ["Panic! At the Disco - King of the Clouds", "Queued Before"]
    assert replies(rig)[0].startswith("⏭️ Skipped — playing 2 songs from **Pray for the Wicked**")


@planned
async def test_leaving_voice_mid_lookup_says_why_nothing_happened(rig):
    st = _playing(rig, "Queued Before")
    rig.catalog_gate = asyncio.Event()
    rig.cog.llm = FakeLLM(LLMReply("commands", _pftw()))
    rig.say(REPORTED)
    for _ in range(100):
        await asyncio.sleep(0.02)
        if rig.catalog_calls:
            break
    rig.vc.is_connected = lambda: False                  # really gone this time
    rig.cog._on_receiver_session(GID, "stopped")
    rig.catalog_gate.set()
    await rig.settle(0.5)
    assert rig.vc.stopped == 0 and st.current["title"] == "Something Else"
    assert _titles(st) == ["Queued Before"]
    assert replies(rig) == ["✖ Didn't finish your request — I moved, left or reconnected "
                            "after you asked."]


@planned
async def test_no_songs_found_means_no_skip_either(rig):
    st = _playing(rig, "Queued Before")
    rig.albums.clear()
    rig.cog.llm = FakeLLM(LLMReply("commands", _pftw()))
    rig.say(REPORTED)
    await rig.settle(0.5)
    assert rig.vc.stopped == 0 and st.current["title"] == "Something Else"
    (reply,) = replies(rig)
    assert reply == ("❌ Couldn't find the album “Pray for the Wicked” by Panic! At The Disco.\n"
                     "↪️ So I didn't skip this song.")


@planned
async def test_play_next_goes_in_front_without_skipping(rig):
    st = _playing(rig, "A", "B")
    rig.cog.llm = FakeLLM(LLMReply("commands", (
        Intent("play", "Mr. Brightside", artist="The Killers", when="next"),)))
    rig.say("hey dj play mr brightside next")
    await rig.settle(0.5)
    assert rig.vc.stopped == 0 and st.current["title"] == "Something Else"
    assert _titles(st) == ["The Killers - Mr. Brightside", "A", "B"]
    assert replies(rig) == ["✅ **The Killers - Mr. Brightside** is up next."]


@planned
async def test_a_now_song_then_a_next_song_keep_their_order(rig):
    st = _playing(rig, "A", "B")
    rig.cog.llm = FakeLLM(LLMReply("commands", (
        Intent("play", "X", when="now"), Intent("album", "Pray for the Wicked", count="7,9",
                                                when="next"))))
    rig.say("hey dj play X now and then the seventh and ninth off pray for the wicked")
    await rig.settle(0.5)
    assert rig.vc.stopped == 1                           # one skip for the whole request
    assert st.current["title"] == "X"
    assert _titles(st) == ["Panic! At the Disco - One of the Drunks",
                           "Panic! At the Disco - King of the Clouds", "A", "B"]
    first, second = replies(rig)[0].splitlines()
    assert first == "⏭️ Skipped — playing **X** now."
    assert second.startswith("✅ Up next, 2 songs from **Pray for the Wicked**")


@planned
async def test_songs_wanted_now_into_an_idle_player_just_start(rig):
    rig.albums["pray for the wicked"] = PFTW
    st = get_state(GID)
    st.queue.extend({"title": t} for t in ("A",))
    rig.cog.llm = FakeLLM(LLMReply("commands", _pftw(count="7")))
    rig.say("hey dj skip this and play the seventh song off pray for the wicked")
    await rig.settle(0.5)
    assert rig.vc.stopped == 0
    assert st.current["title"] == "Panic! At the Disco - One of the Drunks"
    assert _titles(st) == ["A"]
    assert replies(rig)[0].startswith("✅ Queued 1 song from **Pray for the Wicked**")


@planned
async def test_without_next_or_now_songs_still_go_last(rig):
    st = _playing(rig, "A")
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("play", "X"),)))
    rig.say("hey dj add X to the queue")
    await rig.settle(0.5)
    assert _titles(st) == ["A", "X"] and rig.vc.stopped == 0


def test_a_skip_next_to_songs_wanted_soon_becomes_their_now():
    from core import voice_plan as vp

    def fold(*intents):
        return [(st.op, st.when, st.took_skip) for st in vp.fold_skips(vp.from_intents(intents)).steps]
    album = Intent("album", "PFTW", count="7,9", when="next")
    now = Intent("play", "X", when="now")
    plain = Intent("play", "X")
    assert fold(Intent("skip"), album) == [("add_album", "now", True)]
    assert fold(now, Intent("skip")) == [("add_song", "now", True)]
    # "skip this and play X" is X now, whatever "when" the model gave.
    assert fold(Intent("skip"), plain, Intent("volume", "up")) == [
        ("add_song", "now", True), ("volume", "", False)]
    # Left as said: a skip with a count, "play X (for later) and skip this",
    # or something else in between.
    assert fold(Intent("skip", count="2"), album) == [("skip", "", False), ("add_album", "next", False)]
    assert fold(plain, Intent("skip")) == [("add_song", "", False), ("skip", "", False)]
    assert fold(Intent("skip"), Intent("shuffle"), album) == [
        ("skip", "", False), ("shuffle", "", False), ("add_album", "next", False)]


@planned
async def test_the_reported_sentence_as_the_model_actually_answered_it(rig):
    # What gemini-2.5-flash-lite gave for the reported sentence (both rounds,
    # 2026-10-12): a skip, then the album for the END of the queue.
    st = _playing(rig, "Queued Before")
    rig.cog.llm = FakeLLM(LLMReply("commands", _pftw(when=None, count="7,9")))
    rig.say(REPORTED)
    await rig.settle(0.5)
    assert rig.vc.stopped == 1
    assert st.current["title"] == "Panic! At the Disco - One of the Drunks"
    assert _titles(st) == ["Panic! At the Disco - King of the Clouds", "Queued Before"]


@planned
async def test_a_spoken_line_is_not_a_song_to_skip(rig):
    # Nothing but the bot's own voice playing (a TTS reply, solo): "now"
    # songs go in front and start when it finishes — the line isn't cut off.
    st = get_state(GID)
    st.queue.extend({"title": t} for t in ("A",))
    rig.vc.playing = True
    rig.vc.source = vd.voice_mix.SpeechSource(vd.voice_mix.SpeechClip(b""))
    rig.music._busy = lambda guild: True                 # like the real one: the player is busy
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("play", "X", when="now"),)))
    rig.say("hey dj play X now")
    await rig.settle(0.5)
    assert rig.vc.stopped == 0 and rig.music.kicks == 0
    assert _titles(st) == ["X", "A"]
    assert replies(rig) == ["✅ **X** is up next."]
