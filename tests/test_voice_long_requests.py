"""Long requests (round 8): an utterance longer than the front end's 10 s
cap arrives as overlapping parts. The parts addressed to us are held, the
whole audio is recognised once more (no seam where a name was cut), and the
planner gets one request — all of it, however long.

The real cog through the flow rig (tests/test_voicedj_flow.py): scripted
"whisper" (each part's audio carries a marker; the whole carries them all),
fake Discord, a scripted planner. This proves the cog's handling; what real
recordings sound like after joining is measured by scripts/voice_replay.py."""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

import cogs.voicedj as vd                                         # noqa: E402
from core import voice_plan as vp                                 # noqa: E402
from core.state import get_state                                  # noqa: E402
from core.voice_tools import OPS                                  # noqa: E402
from tests.test_voice_smart import FakeLLM, posted, smart         # noqa: E402
from tests.test_voicedj_flow import GID, UID                      # noqa: E402

KEY = (GID, UID)


def a_plan(*steps):
    full = []
    for n, (op, kw) in enumerate(steps, 1):
        s = {"id": n, "op": op, "title": "", "artist": "", "heard_title": "", "heard_artist": "",
             "album": "", "tracks": "", "target": "", "to": "", "value": "", "count": 1,
             "when": "", "after": []}
        s.update(kw)
        full.append(s)
    return vp.parse_plan(json.dumps({"kind": "plan", "steps": full, "reply": "", "ask": "",
                                     "lookups": []}), ops=OPS)


PHOENIX_AND_SHUFFLE = (("add_song", {"title": "The Phoenix", "artist": "Fall Out Boy"}),
                       ("shuffle", {}))


def counts(rig):
    return dict(rig.cog.metrics.guild(GID).counts)


@smart
async def test_a_long_request_is_planned_once_from_its_whole_audio(rig):
    rig.cog.llm = FakeLLM(a_plan(*PHOENIX_AND_SHUFFLE))
    m = rig.say_long("Hey DJ, play The Phoenix by fall off?",
                     "Phoenix by Fallout Boy and then shuffle the queue.")
    rig.wholes[m] = "Hey DJ, play The Phoenix by Fallout Boy and then shuffle the queue."
    await rig.settle(1.0)
    # each part heard once, then the whole once more — and one planner call,
    # with the whole's words (no "fall off? Phoenix" seam)
    assert rig.heard_audio == [(m[0],), (m[1],), m]
    assert rig.cog.llm.asked == [("play The Phoenix by Fallout Boy and then shuffle the queue", None)]
    assert [a for _g, _u, a, _x in rig.executed] == ["play", "shuffle"]
    c = counts(rig)
    assert c["long_request_parts"] == 1 and c["long_request_rejoined"] == 1
    assert c["long_requests"] == 1


@smart
async def test_the_first_part_alone_is_never_acted_on(rig):
    # "Hey DJ, skip this song" is a whole command by itself — but it's only
    # the start: "…, but only after you've played X" changes what it means.
    rig.cog.llm = FakeLLM(a_plan(("add_song", {"title": "X", "artist": "Y"}),
                                 ("skip", {"after": [1]})))
    m = rig.say_long("Hey DJ, skip this song,", "song, but only after you've queued X by Y.")
    rig.wholes[m] = "Hey DJ, skip this song, but only after you've queued X by Y."
    await rig.settle(1.0)
    assert len(rig.cog.llm.asked) == 1
    assert [a for _g, _u, a, _x in rig.executed] == ["play", "skip"]


@pytest.mark.parametrize("failure", [RuntimeError("decoder blew up"), vd.AsrTimeout()])
def test_if_the_whole_cant_be_heard_again_its_parts_words_are_used(monkeypatch, failure):
    @smart
    async def body(rig):
        rig.cog.llm = FakeLLM(a_plan(*PHOENIX_AND_SHUFFLE))
        m = rig.say_long("Hey DJ, play The Phoenix by Fall Out Boy and",
                         "boy and then shuffle the queue.")
        rig.wholes[m] = failure
        await rig.settle(1.0)
        assert rig.heard_audio == [(m[0],), (m[1],), m]
        # the overlap's repeated words taken once (core/voice_intent.stitch)
        assert rig.cog.llm.asked == [
            ("play The Phoenix by Fall Out Boy and then shuffle the queue", None)]
        assert [a for _g, _u, a, _x in rig.executed] == ["play", "shuffle"]
    body(monkeypatch)


@smart
async def test_a_whole_heard_without_its_wake_word_goes_by_the_parts(rig):
    # Real recording 1: the 5-second check heard "and DJ" for "hey DJ".
    rig.cog.llm = FakeLLM(a_plan(*PHOENIX_AND_SHUFFLE))
    m = rig.say_long("Hey DJ, play The Phoenix by Fall Out Boy and",
                     "boy and then shuffle the queue.")
    rig.wholes[m] = "And DJ, play The Phoenix by Fall Out Boy and then shuffle the queue."
    await rig.settle(1.0)
    assert rig.cog.llm.asked == [
        ("play The Phoenix by Fall Out Boy and then shuffle the queue", None)]
    assert counts(rig)["long_request_whole_unused"] == 1


@smart
async def test_a_long_utterance_not_for_us_isnt_transcribed_past_its_first_part(rig):
    rig.cog.llm = FakeLLM()
    # as they really come: each part ~10 s after the last (here: once the
    # one before has been dealt with)
    first = rig.say("so anyway I was telling him about the game last night and",
                    reason="max_length")
    await rig.settle(0.3)
    rig.say("and he said the referee was blind, which, honestly,", overlap_ms=800,
            reason="max_length")
    rig.say("honestly, fair.", overlap_ms=800)
    await rig.settle(1.0)
    assert rig.heard_audio == [(first,)]                  # the rest never recognised
    assert rig.cog.llm.asked == [] and rig.executed == []
    assert rig.drops()["not_addressed"] == 2


@smart
async def test_parts_arriving_together_are_still_understood_in_order(rig):
    # A busy recogniser can have several parts of one utterance waiting at
    # once: the start is heard first, and never pushed out by its own rest.
    rig.cog.llm = FakeLLM(a_plan(("shuffle", {})))
    m = rig.say_long("Hey DJ, could you, um,", "um, maybe, if it's not", "not too much, shuffle.")
    rig.wholes[m] = "Hey DJ, could you, um, maybe, if it's not too much, shuffle."
    await rig.settle(1.0)
    assert rig.heard_audio == [(m[0],), (m[1],), (m[2],), m]
    assert "superseded" not in rig.drops()
    assert [a for _g, _u, a, _x in rig.executed] == ["shuffle"]


@smart
async def test_a_long_request_whose_end_never_comes_is_taken_as_far_as_it_got(rig):
    rig.cog.llm = FakeLLM(a_plan(("add_song", {"title": "The Phoenix", "artist": "Fall Out Boy"})))
    rig.say_long("Hey DJ, play The Phoenix by Fall Out Boy and",
                 "boy and then", last_reason="max_length")      # …and nothing more
    await rig.settle(0.5)
    assert rig.cog.llm.asked == [] and KEY in rig.cog._parts       # still waiting
    rig.cog._parts[KEY].since -= vd._PARTS_WAIT + 1
    rig.cog._flush_stale_parts()
    await rig.settle(1.0)
    assert rig.cog.llm.asked == [("play The Phoenix by Fall Out Boy and then", None)]
    assert [a for _g, _u, a, _x in rig.executed] == ["play"]
    assert any("I may have missed the end of what you said" in str(f[2]) for f in rig.feedback)
    assert counts(rig)["long_requests_incomplete"] == 1


@smart
async def test_a_request_of_a_minute_and_over_500_characters_reaches_the_planner_whole(rig):
    rig.cog.llm = FakeLLM(a_plan(("shuffle", {})))
    words = [f"then do thing number {n} for me" for n in range(1, 21)]
    said = "Hey DJ, " + ", ".join(words) + ", and finally shuffle the queue."
    assert len(said) > 500
    # six parts of ~10 s each, as the front end cuts them
    parts = [", ".join(c) for c in np.array_split(np.array(words, dtype=object), 6)]
    parts[0] = "Hey DJ, " + parts[0]
    m = rig.say_long(*parts)
    rig.wholes[m] = said
    await rig.settle(1.5)
    assert len(m) == 6 and rig.heard_audio[-1] == m
    asked = rig.cog.llm.asked[0][0]
    assert asked == said[len("Hey DJ, "):].rstrip(".")          # every word, nothing cut
    assert len(asked) > 500
    assert [a for _g, _u, a, _x in rig.executed] == ["shuffle"]


@smart
async def test_two_peoples_long_requests_stay_apart(rig):
    from tests.test_voicedj_flow import OTHER
    # Answers by what was asked, not in call order: which of the two reaches
    # the planner first is down to scheduling, and isn't what this tests.
    class ByText(FakeLLM):
        async def ask(self, said, *, now_playing=None):
            self.asked.append((said, now_playing))
            title, artist = ("A", "B") if "play A" in said else ("C", "D")
            return a_plan(("add_song", {"title": title, "artist": artist}))
    rig.cog.llm = ByText()
    mine = rig.say_long("Hey DJ, play A by", "by B please.", uid=UID)
    theirs = rig.say_long("Hey DJ, play C by", "by D please.", uid=OTHER)
    rig.wholes[mine] = "Hey DJ, play A by B please."
    rig.wholes[theirs] = "Hey DJ, play C by D please."
    await rig.settle(1.5)
    assert sorted(a[0] for a in rig.cog.llm.asked) == ["play A by B please", "play C by D please"]
    assert sorted((u, x) for _g, u, a, x in rig.executed) == [(UID, "A"), (OTHER, "C")]
    assert not posted(rig) or all("missed the end" not in p for p in posted(rig))
    assert get_state(GID).queue


def test_the_planner_model_settings_reach_the_planner(monkeypatch):
    import dataclasses

    from core.config import settings
    monkeypatch.setattr(vd, "settings", dataclasses.replace(
        settings, voice_plan_model="x/planner", voice_plan_fallbacks=("y/backup",),
        voice_plan_timeout=12.0))

    class Recording(FakeLLM):
        def __init__(self, *a):
            super().__init__(*a)
            self.kw = []

        async def plan(self, said, context, **kw):
            self.kw.append(kw)
            return await super().plan(said, context, **kw)

    @smart
    async def body(rig):
        rig.cog.llm = Recording(a_plan(("shuffle", {})))
        rig.say("Hey DJ, could you mix the queue up a bit?")
        await rig.settle(0.5)
        kw = rig.cog.llm.kw[0]
        assert (kw["model"], kw["fallbacks"], kw["timeout"]) == ("x/planner", ("y/backup",), 12.0)
        assert callable(kw["lookup"])
    body(monkeypatch)
