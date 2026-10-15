"""The planner request (core/voice_llm.py, round 8): what the model is told
and sent, and what the router does with the answer — long requests, answers
that run out of room, lookup rounds, garbled answers, and the "don't" veto.

No real network and no real model: a local aiohttp server stands in for
OpenRouter and returns the JSON each test scripts. These prove our side of
the exchange, not how well any model plans (scripts/voice_plan_bench.py runs
the live model)."""
from __future__ import annotations

import asyncio
import json

import pytest

from core import voice_llm as vl
from core import voice_plan as vp
from core.voice_tools import NOT_BY_VOICE, OPS

KEY = "sk-or-v1-" + "fedcba9876543210" * 4


def step(id, op, **kw):
    base = {"id": id, "op": op, "title": "", "artist": "", "heard_title": "", "heard_artist": "",
            "album": "", "tracks": "", "target": "", "to": "", "value": "", "count": 1,
            "when": "", "after": []}
    base.update(kw)
    return base


def plan_json(*steps, kind="plan", reply="", ask="", lookups=()):
    return json.dumps({"kind": kind, "steps": list(steps), "reply": reply, "ask": ask,
                       "lookups": list(lookups)})


# ── what the model is told ───────────────────────────────────────────────────

def test_the_schema_is_strict_and_lists_every_voice_operation():
    s = vl._plan_schema()
    assert set(s["required"]) == set(s["properties"])
    item = s["properties"]["steps"]["items"]
    assert set(item["required"]) == set(item["properties"])          # strict mode needs all
    assert item["properties"]["op"]["enum"] == sorted(OPS)
    assert item["properties"]["when"]["enum"] == ["", "next", "now"]
    assert item["properties"]["after"]["items"] == {"type": "integer"}
    look = s["properties"]["lookups"]["items"]
    assert look["properties"]["tool"]["enum"] == ["album", "songs"]


def test_the_prompt_describes_every_operation_and_names_the_typed_only_commands():
    p = vl.plan_prompt()
    for name, spec in OPS.items():
        assert f"  {name}: " in p, name
        if spec.confirm:
            assert f"  {name}: {spec.doc} (they must say yes first)" in p
    for name in NOT_BY_VOICE:
        assert name in p, name
    # the rules the 13-effect request depends on
    assert "but only after B" in p and "count 5" in p
    assert "pos:1 is the next one" in p and '"current"' in p
    assert "step:K" in p and "entry:qN" in p
    assert "never swap the artist they said for a more famous one" in p
    assert "heard_title / heard_artist exactly as the speech-to-text wrote them" in p


def test_the_request_keeps_the_instructions_fixed_and_puts_the_situation_last():
    a = vl.build_plan_request("shuffle", "Now playing: nothing\nQueue: empty", model="m1",
                              fallbacks=("m2", "m1", ""))
    b = vl.build_plan_request("skip", "Now playing: [q4] X by Y", model="m1",
                              facts="Album X: 1. A 2. B")
    assert a["messages"][0] == b["messages"][0]                       # cacheable
    assert a["models"] == ["m1", "m2"] and "models" not in b
    assert a["max_tokens"] == vl.PLAN_MAX_TOKENS and a["temperature"] == 0
    rf = a["response_format"]["json_schema"]
    assert rf["strict"] is True and rf["name"] == "voice_plan"
    assert a["provider"]["require_parameters"] is True
    user = b["messages"][1]["content"]
    assert user.index("Now playing") < user.index("Facts you asked for") < user.index("They said: skip")


def test_the_context_shows_ids_positions_and_says_how_many_are_not_shown():
    up = [{"title": f"Song {n}", "uploader": "Band - Topic", "requester": "ben", "_qid": f"q{n}"}
          for n in range(1, 36)]
    ctx = vl.plan_context(current={"title": "Now", "artist": "Them", "_qid": "q99"}, upcoming=up,
                          player="playing, volume 100%", recent=["play x → added X by Y"],
                          question="Which one did you mean?", show=30)
    lines = ctx.splitlines()
    assert lines[0] == "Now playing: [q99] Now by Them"
    assert "Queue (35 upcoming; pos:1 plays next):" in lines
    assert "1. [q1] Song 1 by Band (added by ben)" in lines                # "- Topic" dropped
    assert "30. [q30] Song 30 by Band (added by ben)" in lines
    assert not any("[q31]" in x for x in lines)
    assert "… and 5 more (ask by name with match:TEXT)" in lines           # never silently cut
    assert "- play x → added X by Y" in lines
    assert lines[-1].startswith("You asked them: Which one did you mean?")
    assert vl.plan_context(current=None, upcoming=[]).splitlines() == ["Now playing: nothing",
                                                                        "Queue: empty"]


# ── the router, against a stand-in for OpenRouter ────────────────────────────

class FakePlanner:
    """Answers /chat/completions with scripted (content, finish_reason) pairs."""

    def __init__(self):
        self.requests: list[dict] = []
        self.queued: list[tuple[str, str]] = []
        self.default = (plan_json(), "stop")

    async def handler(self, request):
        from aiohttp import web
        self.requests.append(await request.json())
        content, finish = self.queued.pop(0) if self.queued else self.default
        return web.json_response({
            "model": self.requests[-1]["model"],
            "choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": finish}],
            "usage": {"cost": 0.0001}})

    async def __aenter__(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_post("/api/v1/chat/completions", self.handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/api/v1"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()


def with_router(test, **kw):
    async def go():
        async with FakePlanner() as srv:
            router = vl.LLMRouter(KEY, model=kw.pop("model", "google/a"), base_url=srv.url, **kw)
            try:
                await test(srv, router)
            finally:
                await router.close()
    asyncio.run(go())


def test_a_plan_comes_back_parsed_ordered_and_costed():
    async def t(srv, router):
        srv.queued = [(plan_json(step(1, "add_album", album="Pray for the Wicked",
                                      artist="Panic! At The Disco", tracks="latter half",
                                      after=[2]),
                                 step(2, "add_song", title="I Write Hymns Not Travesties",
                                      artist="Yung Gravy", count=5)), "stop")]
        p = await router.plan("play the latter half of pray for the wicked but only after "
                              "i write hymns five times", "Now playing: nothing\nQueue: empty")
        assert [s.id for s in p.steps] == [2, 1] and p.steps[0].count == 5
        assert p.cost == pytest.approx(0.0001) and p.model == "google/a"
        assert router.kinds == {"plan_plan": 1}
        user = srv.requests[0]["messages"][1]["content"]
        assert "They said: play the latter half" in user
    with_router(t)


def test_a_request_too_long_for_the_planner_is_refused_whole_not_cut():
    async def t(srv, router):
        said = "play x and then " * 200                      # ~3200 characters
        p = await router.plan(said, "ctx")
        assert p.kind == "plan" and not p.steps
        assert "characters long" in p.problems[0] and str(vl.MAX_SAID_CHARS) in p.problems[0]
        assert srv.requests == [] and router.calls == 0         # nothing sent, nothing done
    with_router(t)


def test_a_long_request_under_the_limit_is_sent_whole():
    async def t(srv, router):
        said = " ".join(f"word{n}" for n in range(250))          # ~1900 characters, 250 words
        assert 1500 < len(said) <= vl.MAX_SAID_CHARS
        await router.plan(said, "ctx")
        assert srv.requests[0]["messages"][1]["content"].endswith(said)
    with_router(t)


def test_an_answer_that_ran_out_of_room_is_retried_then_reported():
    async def t(srv, router):
        cut = (plan_json(step(1, "shuffle"))[:-30], "length")
        srv.queued = [cut, cut]
        p = await router.plan("a very long request", "ctx")
        assert [r["model"] for r in srv.requests] == ["google/a", "openai/b"]
        assert p.kind == "plan" and not p.steps and "cut off" in p.problems[0]
        assert "ran out of room" in router.last_error
    with_router(t, fallbacks=("openai/b",))


def test_a_complete_answer_marked_length_is_not_trusted():
    async def t(srv, router):
        # valid JSON, but the model stopped at the limit: steps may be missing
        srv.queued = [(plan_json(step(1, "shuffle")), "length"),
                      (plan_json(step(1, "shuffle"), step(2, "skip")), "stop")]
        p = await router.plan("shuffle and skip", "ctx")
        assert [s.op for s in p.steps] == ["shuffle", "skip"]
    with_router(t, fallbacks=("openai/b",))


def test_a_garbled_answer_gets_one_more_try_on_the_fallback():
    async def t(srv, router):
        srv.queued = [('{"kind": "plan", "steps": [{"id": 1, "op": "', "stop"),
                      (plan_json(step(1, "skip")), "stop")]
        p = await router.plan("skip", "ctx")
        assert [s.op for s in p.steps] == ["skip"]
        assert "models" not in srv.requests[1]                 # nothing left to fall back to
        assert router.kinds["retried"] == 1
    with_router(t, fallbacks=("openai/b",))
    async def t2(srv, router):
        srv.default = ("{oops", "stop")
        assert await router.plan("skip", "ctx") is None
        assert len(srv.requests) == 1                          # no fallback: given up
    with_router(t2)


def test_lookups_are_answered_and_the_facts_sent_back():
    asked = []

    async def lookup(tool, title, artist):
        asked.append((tool, title, artist))
        return f"{title} by {artist}: 1. Dancing's Not a Crime 2. Hey Look Ma"

    async def t(srv, router):
        srv.queued = [(plan_json(kind="lookup", lookups=[{"tool": "album", "title": "Pray for the Wicked",
                                                          "artist": "Panic! At The Disco"}]), "stop"),
                      (plan_json(step(1, "add_album", album="Pray for the Wicked",
                                      artist="Panic! At The Disco", tracks="first half")), "stop")]
        p = await router.plan("play the first half of pray for the wicked", "ctx", lookup=lookup)
        assert asked == [("album", "Pray for the Wicked", "Panic! At The Disco")]
        assert "Facts you asked for:" in srv.requests[1]["messages"][1]["content"]
        assert "Dancing's Not a Crime" in srv.requests[1]["messages"][1]["content"]
        assert [s.op for s in p.steps] == ["add_album"] and router.kinds["lookups"] == 1
    with_router(t)


def test_lookup_rounds_are_capped_and_a_model_that_keeps_asking_is_given_up_on():
    calls = []

    async def lookup(tool, title, artist):
        calls.append(title)
        if title == "boom":
            raise RuntimeError("catalogue down")
        return "facts"

    look = plan_json(kind="lookup", lookups=[{"tool": "album", "title": "boom", "artist": ""}])

    async def t(srv, router):
        srv.default = (look, "stop")
        assert await router.plan("play an album", "ctx", lookup=lookup) is None
        assert len(calls) == vl.MAX_LOOKUP_ROUNDS                 # then "no more lookups"
        last = srv.requests[-1]["messages"][1]["content"]
        assert "(lookup failed: RuntimeError)" in last and "no more lookups" in last
        assert len(srv.requests) == vl.MAX_LOOKUP_ROUNDS + 2       # bounded, not until timeout
        assert "kept asking for lookups" in router.last_error
    with_router(t)


def test_without_a_lookup_tool_the_model_is_told_to_plan_with_what_it_has():
    async def t(srv, router):
        srv.queued = [(plan_json(kind="lookup", lookups=[{"tool": "songs", "title": "",
                                                          "artist": "X"}]), "stop"),
                      (plan_json(step(1, "add_top", artist="X")), "stop")]
        p = await router.plan("play x's top songs", "ctx")
        assert "no more lookups" in srv.requests[1]["messages"][1]["content"]
        assert [s.op for s in p.steps] == ["add_top"]
    with_router(t)


def test_a_plan_that_does_what_they_said_not_to_is_vetoed():
    async def t(srv, router):
        srv.queued = [(plan_json(step(1, "skip")), "stop")]
        p = await router.plan("don't skip this one, I love it", "ctx")
        assert p.kind == "other" and p.steps == () and router.kinds["vetoed"] == 1
        srv.queued = [(plan_json(step(1, "shuffle")), "stop")]
        p = await router.plan("don't skip it, shuffle the queue", "ctx")
        assert [s.op for s in p.steps] == ["shuffle"]
    with_router(t)


def test_steps_the_plan_cant_use_are_reported_never_dropped_quietly():
    async def t(srv, router):
        srv.queued = [(plan_json(step(1, "add_song", title="X", count=99),
                                 step(2, "teleport"),
                                 step(3, "move", target="step:1", to="front"),
                                 step(4, "shuffle")), "stop")]
        p = await router.plan("x 99 times, teleport, move it to the front, shuffle", "ctx")
        assert [s.op for s in p.steps] == ["shuffle"]
        text = " | ".join(p.problems)
        assert "99 copies" in text and "teleport" in text and "step 1 adds" in text
    with_router(t)


def test_too_many_steps_is_refused_whole():
    p = vp.parse_plan(plan_json(*[step(n, "shuffle") for n in range(1, vp.MAX_STEPS + 2)]), ops=OPS)
    assert p.steps == () and f"at most {vp.MAX_STEPS}" in p.problems[0]


def test_the_key_never_appears_in_what_is_recorded():
    async def t(srv, router):
        srv.default = (f"not json {KEY}", "stop")
        assert await router.plan("skip", "ctx") is None
        assert KEY not in router.last_error
    with_router(t)


def test_a_whitespace_runaway_after_a_finished_answer_is_read_not_reported_as_cut_off():
    """Seen live (Gemini, strict schema, 2026-10-14): '{"kind": "other"' and
    then spaces until the token limit. "Nothing to do" is what it said."""
    async def t(srv, router):
        srv.queued = [('{\n  "kind": "other"' + " " * 6000, "length")]
        p = await router.plan("don't skip this one, I love it", "ctx")
        assert p.kind == "other" and not p.problems and router.kinds["runaway_salvaged"] == 1
        srv.queued = [('{"kind": "answer", "reply": "It\'s Centuries by Fall Out Boy."'
                       + "\n" * 3000, "length")]
        p = await router.plan("what song is this", "ctx")
        assert p.kind == "answer" and p.reply.startswith("It's Centuries")
        # a plan cut short is never guessed at, even with trailing spaces
        srv.queued = [('{"kind": "plan", "steps": [{"id": 1, "op": "shuffle"}' + " " * 3000,
                       "length")]
        p = await router.plan("shuffle and more", "ctx")
        assert not p.steps and "cut off" in p.problems[0]
    with_router(t)
