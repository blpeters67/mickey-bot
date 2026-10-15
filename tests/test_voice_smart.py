"""The smart voice DJ: several songs in one request, the LLM tier, spoken
answers to questions, and "typing…" from the moment the wake word is heard.

Grammar tests are pure. The flow tests drive the real VoiceDJ cog through the
rig in tests/test_voicedj_flow.py (scripted "whisper", fake Discord) with a
scripted LLM in place of OpenRouter — they prove the decisions, not what a
real model would answer (tests/test_voice_llm.py covers the request/answer
contract; real-model accuracy was measured separately, see core/voice_llm.py).
"""
from __future__ import annotations

import asyncio
import re
import time

import pytest

np = pytest.importorskip("numpy")

import cogs.voicedj as vd                                       # noqa: E402
from core.state import get_state                                # noqa: E402
from core.voice_frontend import (                               # noqa: E402
    VAD_WINDOW, VAD_WINDOW_S, Probe, Segmenter, SegmenterConfig,
)
from core.voice_intent import (                                 # noqa: E402
    MAX_VOICE_PLAYS, Intent, interpret, split_songs, text_after_wake,
)
from core.voice_llm import LLMReply                             # noqa: E402
from tests.test_voicedj_flow import GID, OTHER, UID, Rig        # noqa: E402

W = ["hey dj"]


def songs(text):
    r = interpret(text, W)
    return [(i.action, i.arg) for i in r.intents], r.list_like


# ── grammar: several songs, lead-ins ─────────────────────────────────────────

def test_the_two_song_request_from_the_issue():
    got, unsure = songs("Hey DJ, I want to hear two songs; can you play Fireflies by Owl City "
                        "and The Pretender by the Foo Fighters")
    assert got == [("play", "Fireflies by Owl City"),
                   ("play", "The Pretender by the Foo Fighters")]
    assert not unsure


@pytest.mark.parametrize("text,want", [
    ("hey dj queue up mr brightside then hotel california",
     ["mr brightside", "hotel california"]),
    ("hey dj play mr brightside and then hotel california",
     ["mr brightside", "hotel california"]),
    ("hey dj play rich girl by hall and oates and africa by toto",
     ["rich girl by hall and oates", "africa by toto"]),
])
def test_lists_it_can_split(text, want):
    got, _ = songs(text)
    assert [q for _a, q in got] == want


@pytest.mark.parametrize("text,query", [
    ("hey dj play the sound of silence by simon and garfunkel",
     "the sound of silence by simon and garfunkel"),
    ("hey dj play me and bobby mcgee", "me and bobby mcgee"),
    ("hey dj play now and then by the beatles", "now and then by the beatles"),
    ("hey dj play fireflies by owl city and the pretender",
     "fireflies by owl city and the pretender"),
])
def test_and_inside_a_name_stays_one_song_but_is_flagged(text, query):
    got, unsure = songs(text)
    assert got == [("play", query)]
    assert unsure            # the LLM tier decides, when it's on


def test_a_count_that_does_not_match_is_unsure():
    _got, unsure = songs("hey dj I want to hear three songs, can you play X by A and Y by B")
    assert unsure


def test_plain_requests_are_not_flagged():
    for text in ("hey dj play mr brightside", "hey dj can you play hotel california",
                 "hey dj put on fireflies by owl city"):
        got, unsure = songs(text)
        assert len(got) == 1 and not unsure, text


@pytest.mark.parametrize("text,query", [
    ("hey dj I want to hear fireflies by owl city", "fireflies by owl city"),
    ("hey dj listen to hotel california", "hotel california"),
    ("hey dj could you please queue up mr brightside", "mr brightside"),
    ("hey dj queue up mr brightside", "mr brightside"),
])
def test_lead_ins(text, query):
    assert songs(text)[0] == [("play", query)]


@pytest.mark.parametrize("text", ["hey dj play two songs", "hey dj I want to hear some songs",
                                  "hey dj play me a song"])
def test_a_request_for_nothing_in_particular_is_not_a_song(text):
    r = interpret(text, W)
    assert r.intent is None and r.reason == "no_command"     # → the LLM tier, if on


def test_existing_phrases_are_unchanged():
    assert songs("hey dj play the next song")[0] == [("skip", None)]
    assert songs("hey dj play")[0] == [("resume", None)]
    assert songs("hey dj play it again")[0] == [("restart", None)]


def test_song_requests_are_capped():
    many = " then ".join(f"song number {n}" for n in range(9))
    out, _ = split_songs(many)
    assert len(out) == MAX_VOICE_PLAYS


def test_text_after_wake_keeps_what_was_said():
    assert text_after_wake("Hey DJ, I'd love some Fireflies!", W) == "I'd love some Fireflies"
    assert text_after_wake("hey dj", W) == ""
    assert text_after_wake("I said hey dj skip", W) is None


# ── the front end hands over the start of a long utterance once ─────────────

def _speech(seg, n, prob=0.9, t0=0.0):
    w = np.zeros(VAD_WINDOW, dtype=np.float32)
    for i in range(n):
        seg.push(w, prob, t0 + (i + 1) * VAD_WINDOW_S)
    return t0 + n * VAD_WINDOW_S


def test_probe_fires_once_per_utterance_after_probe_ms():
    probes = []
    seg = Segmenter(SegmenterConfig(probe_ms=300, end_silence_ms=200),
                    lambda *a: None, lambda audio, start, onset: probes.append(len(audio)))
    t = _speech(seg, 5)                        # 160 ms: too early
    assert probes == []
    t = _speech(seg, 30, t0=t)                 # well past 300 ms, still one probe
    assert len(probes) == 1 and probes[0] >= int(0.3 / VAD_WINDOW_S) * VAD_WINDOW
    t = _speech(seg, 10, prob=0.0, t0=t)       # silence ends it
    _speech(seg, 20, t0=t)                     # the next utterance gets its own
    assert len(probes) == 2


def test_no_probe_when_off_or_short():
    probes = []
    seg = Segmenter(SegmenterConfig(probe_ms=0), lambda *a: None, lambda *a: probes.append(1))
    _speech(seg, 60)
    seg2 = Segmenter(SegmenterConfig(probe_ms=1000, end_silence_ms=200),
                     lambda *a: None, lambda *a: probes.append(1))
    t = _speech(seg2, 10)
    _speech(seg2, 10, prob=0.0, t0=t)
    assert probes == []


# ── flows through the cog ────────────────────────────────────────────────────

class FakeLLM:
    model, fallbacks = "fake/model", ()
    calls = failures = 0
    cost, last_error, kinds = 0.0, "", {}

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked = []
        self.contexts = []

    async def ask(self, said, *, now_playing=None):
        self.asked.append((said, now_playing))
        return self.replies.pop(0) if self.replies else None

    async def plan(self, said, context, *, lookup=None, **_kw):
        """The planner's answer: a scripted Plan, or an old-style LLMReply
        turned into one (core/voice_plan.from_reply, plus the cog's folding
        of "skip this and play X"). Records (said, what's playing)."""
        from core import voice_plan as vp
        line = next((ln for ln in context.splitlines() if ln.startswith("Now playing:")), "")
        playing = re.sub(r"^Now playing:\s*(\[[^\]]*\]\s*)?", "", line)
        self.contexts.append(context)
        # Through `ask`, so tests that wrap it (to act mid-"thinking") still do.
        reply = await self.ask(said, now_playing=None if playing in ("", "nothing") else playing)
        if reply is None or isinstance(reply, vp.Plan):
            return reply
        return vp.fold_skips(vp.from_reply(reply))

    def median_seconds(self):
        return None

    async def close(self):
        pass


class FakeChannel:
    def __init__(self):
        self.id = 4242
        self.typing_calls = 0
        self.sent = []

    async def typing(self):
        self.typing_calls += 1

    async def send(self, content=None, **_kw):
        self.sent.append(content)


class FakeTTS:
    def __init__(self):
        self.said = []

    async def say(self, guild, text, **_kw):
        self.said.append(text)
        return True


def smart(fn):
    """A Rig with: instant song lookups, a reply channel that records typing
    and posts, a TTS that records speech, and no LLM until the test sets one."""
    def wrapper(monkeypatch):
        async def go():
            rig = Rig(monkeypatch)

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
            get_state(GID).current = None
            try:
                await fn(rig)
            finally:
                for key in list(rig.cog._typing):
                    rig.cog._typing_stop(key)
                for t in list(rig.cog._play_workers.values()):
                    t.cancel()
                await rig.close()
        asyncio.run(go())
    wrapper.__name__ = fn.__name__
    return wrapper


def plays(rig):
    return [arg for _g, _u, action, arg in rig.executed if action == "play"]


def posted(rig):
    """Reply-channel posts that are not the wake-word cue.

    Saying "hey DJ …" always posts `:ear:*Now listening...*` (see
    tests/test_wake_cue.py). These tests are about what the *AI tier* answers,
    so they filter that one message out instead of pretending nothing is
    posted — an unexpected extra post still shows up here."""
    return [m for m in rig.channel_txt.sent if m != vd._CUE_MESSAGE]


@smart
async def test_free_phrasing_goes_to_the_llm_and_runs_every_command_in_order(rig):
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("volume", "down"), Intent("skip"),
                                                Intent("play", "Fireflies Owl City"))))
    rig.say("Hey DJ, could you turn it down a bit, skip this, and put Fireflies on after?")
    await rig.settle(0.5)
    assert rig.cog.llm.asked[0][0] == ("could you turn it down a bit, skip this, "
                                       "and put Fireflies on after")
    # "skip this, and put Fireflies on after" is Fireflies instead of this
    # song: one plan, the skip folded into the song (tests/test_voice_plans.py).
    assert [a for _g, _u, a, _x in rig.executed] == ["volume", "play"]
    assert plays(rig) == ["Fireflies Owl City"]
    assert get_state(GID).queue[0]["title"] == "Fireflies Owl City"


@smart
async def test_two_songs_through_the_llm(rig):
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("play", "Fireflies Owl City"),
                                                Intent("play", "The Pretender Foo Fighters"))))
    rig.say("hey dj I'd love some fireflies and also that pretender song")
    await rig.settle(0.5)
    assert plays(rig) == ["Fireflies Owl City", "The Pretender Foo Fighters"]
    assert [t["title"] for t in get_state(GID).queue] == plays(rig)


@smart
async def test_every_song_request_is_worked_out_by_the_llm(rig):
    # Even one the grammar could split: the LLM names the actual songs, and
    # they're searched as "Artist - Title", not as the sentence.
    rig.cog.llm = FakeLLM(LLMReply("commands", (
        Intent("play", "Fireflies", artist="Owl City"),
        Intent("play", "The Pretender", artist="Foo Fighters"))))
    rig.say("hey dj play fireflies by owl city and the pretender by foo fighters")
    await rig.settle(0.5)
    assert rig.cog.llm.asked == [
        ("play fireflies by owl city and the pretender by foo fighters", None)]
    assert plays(rig) == ["Fireflies", "The Pretender"]
    assert [t["title"] for t in get_state(GID).queue] == [
        "Owl City - Fireflies", "Foo Fighters - The Pretender"]


@smart
async def test_if_the_llm_fails_the_grammars_split_still_plays(rig):
    rig.cog.llm = FakeLLM(None)
    rig.say("hey dj play fireflies by owl city and the pretender by foo fighters")
    await rig.settle(0.5)
    assert plays(rig) == ["fireflies by owl city", "the pretender by foo fighters"]


@smart
async def test_plain_controls_stay_instant_and_never_ask_the_llm(rig):
    rig.cog.llm = FakeLLM()
    get_state(GID).queue.extend({"title": t} for t in "ABC")
    rig.say("hey dj skip")
    await rig.settle(0.3)
    rig.say("hey dj shuffle the queue")
    await rig.settle(0.3)
    rig.say("hey dj loop the queue")
    await rig.settle(0.3)
    assert rig.cog.llm.asked == []
    assert [a for _g, _u, a, _x in rig.executed] == ["skip", "shuffle", "loop"]
    assert get_state(GID).loop_queue and not get_state(GID).loop


@smart
async def test_an_unsure_list_asks_the_llm(rig):
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("play", "Fireflies"),
                                                Intent("play", "The Pretender"))))
    rig.say("hey dj play fireflies and the pretender")
    await rig.settle(0.5)
    assert len(rig.cog.llm.asked) == 1
    assert plays(rig) == ["Fireflies", "The Pretender"]


@smart
async def test_an_unsure_list_falls_back_to_the_grammar_if_the_llm_fails(rig):
    rig.cog.llm = FakeLLM(None)
    rig.say("hey dj play the sound of silence by simon and garfunkel")
    await rig.settle(0.5)
    assert plays(rig) == ["the sound of silence by simon and garfunkel"]


@smart
async def test_a_play_request_is_never_turned_into_a_chat_answer(rig):
    rig.cog.llm = FakeLLM(LLMReply("question", reply="Nice choice!"))
    rig.say("hey dj play me and bobby mcgee")
    await rig.settle(0.5)
    assert rig.tts.said == []
    assert plays(rig) == ["me and bobby mcgee"]


@smart
async def test_a_question_is_answered_out_loud_and_in_text(rig):
    get_state(GID).current = {"title": "Fireflies", "artist": "Owl City"}
    rig.cog.llm = FakeLLM(LLMReply("question", reply="That's Owl City."))
    rig.say("hey dj who sings this song")
    await rig.settle(0.5)
    assert rig.cog.llm.asked == [("who sings this song", "Fireflies by Owl City")]
    assert rig.tts.said == ["That's Owl City."]
    assert any("💬 That's Owl City." in m for m in rig.channel_txt.sent)
    assert rig.executed == []


@smart
async def test_no_answers_when_chat_is_off(rig):
    rig.cog._cfg[GID]["chat"] = False
    rig.cog.llm = FakeLLM(LLMReply("question", reply="Canberra."))
    rig.say("hey dj what's the capital of australia")
    await rig.settle(0.5)
    assert rig.tts.said == [] and posted(rig) == [] and rig.executed == []


@smart
async def test_ai_off_for_the_server_means_grammar_only(rig):
    rig.cog._cfg[GID]["ai"] = False
    rig.cog.llm = FakeLLM(LLMReply("question", reply="Canberra."))
    rig.say("hey dj what's the capital of australia")
    await rig.settle(0.5)
    assert rig.cog.llm.asked == [] and rig.tts.said == []


@smart
async def test_without_a_wake_word_nothing_reaches_the_llm(rig):
    rig.cog.llm = FakeLLM(LLMReply("question", reply="Hi"))
    rig.say("so what's the capital of australia anyway")
    rig.say("I said hey dj what's up")                  # quoted, not addressed
    await rig.settle(0.5)
    assert rig.cog.llm.asked == []


@smart
async def test_not_for_me_stays_quiet(rig):
    rig.cog.llm = FakeLLM(LLMReply("other"))
    rig.say("hey dj no wait I was talking to jake")
    await rig.settle(0.5)
    assert rig.executed == [] and rig.feedback == [] and rig.tts.said == []
    assert rig.cog._typing == {}


@smart
async def test_llm_failure_says_so(rig):
    rig.cog.llm = FakeLLM(None)
    rig.say("hey dj what's the capital of australia")
    await rig.settle(0.5)
    assert len(rig.feedback) == 1 and "didn't get that" in rig.feedback[0][2]


@smart
async def test_a_follow_up_in_the_wake_window_can_be_a_question(rig):
    rig.cog.llm = FakeLLM(LLMReply("question", reply="Paris."))
    rig.say("hey dj")
    await rig.settle(0.3)
    rig.say("what's the capital of france", start=time.monotonic() - 0.2)
    await rig.settle(0.5)
    assert rig.cog.llm.asked[0][0] == "what's the capital of france"
    assert rig.tts.said == ["Paris."]


@smart
async def test_stop_or_clear_inside_a_request_asks_once_then_runs_it_all_in_order(rig):
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("clear"), Intent("play", "Fireflies"))))
    rig.say("hey dj get rid of everything queued and put on fireflies instead")
    await rig.settle(0.5)
    assert rig.executed == []
    question = rig.feedback[-1][2]
    assert question.startswith("⚠️ Clear the whole queue, then play Fireflies?")
    rig.say("yes, clear it", start=time.monotonic() + 0.05)
    await rig.settle(0.5)
    assert [a for _g, _u, a, _x in rig.executed] == ["clear", "play"]


@smart
async def test_what_the_grammar_reads_as_a_negation_goes_to_the_llm(rig):
    # "stop" is a negation word to the grammar ("stop skipping"), so on its
    # own it would do nothing here.
    rig.cog.llm = FakeLLM(LLMReply("commands", (Intent("stop"), Intent("clear"))))
    rig.say("hey dj stop the music and clear the queue")
    await rig.settle(0.5)
    assert rig.cog.llm.asked[0][0] == "stop the music and clear the queue"
    assert rig.executed == [] and rig.feedback[-1][2].startswith("⚠️ ")   # still asks first
    rig.say("yes", start=time.monotonic() + 0.05)
    await rig.settle(0.5)
    assert [a for _g, _u, a, _x in rig.executed] == ["stop", "clear"]


@smart
async def test_dont_stays_a_dont(rig):
    rig.cog.llm = FakeLLM(LLMReply("other"))
    rig.say("hey dj don't skip this one")
    await rig.settle(0.5)
    assert len(rig.cog.llm.asked) == 1 and rig.executed == [] and rig.feedback == []


@smart
async def test_typing_runs_while_the_llm_works_and_ends_with_the_answer(rig):
    seen = {}
    llm = FakeLLM(LLMReply("question", reply="Eight."))
    real_ask = llm.ask

    async def ask(said, *, now_playing=None):
        seen["typing_during_call"] = rig.channel_txt.typing_calls
        return await real_ask(said, now_playing=now_playing)
    llm.ask = ask
    rig.cog.llm = llm
    rig.say("hey dj how many legs does a spider have")
    await rig.settle(0.5)
    assert seen["typing_during_call"] >= 1
    assert rig.cog._typing == {}                     # stopped when the answer was posted
    assert rig.channel_txt.sent and "💬 Eight." in rig.channel_txt.sent[-1]


@smart
async def test_saying_only_the_wake_word_shows_typing_for_the_window_only(rig):
    rig.say("hey dj")
    await rig.settle(0.3)
    assert rig.channel_txt.typing_calls >= 1
    left = rig.cog._typing[(GID, UID)].until - time.monotonic()
    assert 0 < left <= vd._ARM_SECONDS + 2.0           # not the full _TYPING_MAX


@smart
async def test_typing_ends_when_the_speech_is_dropped_unheard(rig):
    rig.cog._on_probe(_probe(rig, "hey dj play something"))
    await rig.settle(0.3)
    assert (GID, UID) in rig.cog._typing
    rig.say("hey dj play something", start=time.monotonic() - 20)   # too old to transcribe
    await rig.settle(0.3)
    assert rig.drops().get("stale_before_asr") == 1
    assert rig.cog._typing == {}


@smart
async def test_no_answer_for_someone_who_opted_out_while_the_ai_was_thinking(rig):
    llm = FakeLLM(LLMReply("question", reply="Eight."))
    real_ask = llm.ask

    async def ask(said, *, now_playing=None):
        rig.cog._auth[GID] = frozenset({OTHER})
        rig.cog._revoke(GID, UID, "revoked")
        return await real_ask(said, now_playing=now_playing)
    llm.ask = ask
    rig.cog.llm = llm
    rig.say("hey dj how many legs does a spider have")
    await rig.settle(0.5)
    assert rig.tts.said == [] and posted(rig) == [] and rig.cog._typing == {}


@smart
async def test_no_typing_for_people_on_silent(rig, ):
    async def silent(gid, uid):
        return "silent"
    rig.cog._feedback_for = silent
    rig.say("hey dj")
    await rig.settle(0.3)
    assert rig.channel_txt.typing_calls == 0


def _probe(rig, text, *, uid=UID):
    rig._marker += 1
    rig.transcripts[rig._marker] = text
    audio = np.full(int(2.5 * 16000), rig._marker, dtype=np.float32)
    now = time.monotonic()
    return Probe(GID, uid, audio, now - 1.5, now - 1.2, tag=None)


@smart
async def test_the_early_look_starts_typing_when_it_hears_the_wake_word(rig):
    rig.cog._on_probe(_probe(rig, "Hey DJ, I want to hear"))
    await rig.settle(0.3)
    assert rig.channel_txt.typing_calls >= 1
    assert rig.cog.metrics.guild(GID).counts.get("probe_wake") == 1


@smart
async def test_the_early_look_ignores_other_talk(rig):
    rig.cog._on_probe(_probe(rig, "so anyway I was saying"))
    await rig.settle(0.3)
    assert rig.channel_txt.typing_calls == 0 and rig.cog._typing == {}


@smart
async def test_the_early_look_never_gets_in_the_way_of_recognition(rig):
    rig.slow_asr = 0.3
    rig.say("hey dj skip")                           # real recognition under way
    await asyncio.sleep(0.05)
    rig.cog._on_probe(_probe(rig, "hey dj play something"))
    await rig.settle(0.8)
    assert rig.cog.metrics.guild(GID).counts.get("probe_skipped_busy") == 1
    assert [a for _g, _u, a, _x in rig.executed] == ["skip"]


@smart
async def test_a_command_waits_for_an_early_look_instead_of_being_dropped(rig):
    rig.slow_asr = 0.3
    rig.cog._asr.busy_wait = 0.05            # the runner refuses a second call quickly
    rig.cog._on_probe(_probe(rig, "hey dj play something"))
    await asyncio.sleep(0.05)                # the early look is running
    rig.say("hey dj skip")
    await rig.settle(1.0)
    assert rig.drops().get("asr_busy") is None
    assert [a for _g, _u, a, _x in rig.executed] == ["skip"]


@smart
async def test_no_early_look_where_recognition_is_slow(rig):
    rig.cog.metrics.guild(GID).asr_seconds.extend([2.0, 2.5, 3.0])
    rig.cog._on_probe(_probe(rig, "hey dj play something"))
    await rig.settle(0.3)
    assert rig.cog.metrics.guild(GID).counts.get("probe_skipped_slow") == 1
    assert rig.channel_txt.typing_calls == 0


@smart
async def test_typing_stops_by_itself(rig):
    guild = rig.guilds[GID]
    await rig.cog._typing_start(guild, guild.get_member(UID), reason="heard", limit=0.2)
    assert (GID, UID) in rig.cog._typing
    await asyncio.sleep(0.4)
    assert rig.cog._typing == {} and rig.channel_txt.typing_calls == 1


@smart
async def test_more_work_extends_the_typing_instead_of_starting_another(rig):
    guild = rig.guilds[GID]
    member = guild.get_member(UID)
    await rig.cog._typing_start(guild, member, reason="listening", limit=0.2)
    first = rig.cog._typing[(GID, UID)]
    await rig.cog._typing_start(guild, member, reason="ai", limit=5.0)
    assert rig.cog._typing[(GID, UID)] is first and first.until - time.monotonic() > 4
    await asyncio.sleep(0.4)
    assert (GID, UID) in rig.cog._typing                # still going past the first limit
