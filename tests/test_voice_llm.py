"""Tier 2 of the voice DJ (core/voice_llm.py): the request we send, how the
model's answer is checked before anything acts on it, and the router's
behaviour on success and failure. No real network: the router talks to a
local aiohttp server standing in for OpenRouter."""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from core import voice_llm as vl
from core.formatters import redact_secrets

KEY = "sk-or-v1-" + "0123456789abcdef" * 4


# ── the request ──────────────────────────────────────────────────────────────

def test_request_is_a_strict_schema_with_fallbacks():
    body = vl.build_request("play x and y", now_playing="Fireflies — Owl City",
                            model="m1", fallbacks=("m2", "m1", ""))
    assert body["model"] == "m1" and body["models"] == ["m1", "m2"]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] is vl.SCHEMA
    assert body["provider"]["require_parameters"] is True       # only schema-honouring endpoints
    assert body["temperature"] == 0
    # The system prompt never varies (cacheable); what changes is in the user turn.
    assert body["messages"][0] == {"role": "system", "content": vl.SYSTEM_PROMPT}
    user = body["messages"][1]["content"]
    assert "Fireflies — Owl City" in user and "play x and y" in user


def test_no_fallbacks_no_models_list_and_nothing_playing():
    body = vl.build_request("hi", now_playing=None, model="m1")
    assert "models" not in body
    assert "Now playing: nothing" in body["messages"][1]["content"]


def test_schema_requires_every_field_for_strict_mode():
    item = vl.SCHEMA["properties"]["actions"]["items"]
    assert set(item["required"]) == set(item["properties"]) == {"action", "query", "artist",
                                                                 "heard", "value", "when"}
    assert set(vl.SCHEMA["required"]) == set(vl.SCHEMA["properties"])
    assert item["properties"]["action"]["enum"] == list(vl.ACTIONS)
    assert item["properties"]["when"]["enum"] == ["", "next", "now"]


def test_the_prompt_asks_for_the_actual_song_not_the_sentence():
    p = vl.SYSTEM_PROMPT
    assert "instead of repeating their words" in p
    for action in ("album", "top", "latest", "shuffle", "loop"):
        assert action in p
    assert '"1-3"' in p and '"queue"' in p


# ── the answer, checked before anything acts on it ───────────────────────────

def _cmds(*actions):
    """(action, query, value) or (action, query, artist, value)."""
    items = []
    for t in actions:
        a, q, artist, v = t if len(t) == 4 else (t[0], t[1], "", t[2])
        items.append({"action": a, "query": q, "artist": artist, "value": v})
    return json.dumps({"kind": "commands", "reply": "", "actions": items})


def test_commands_become_intents_in_the_order_asked():
    r = vl.parse_reply(_cmds(("play", "Fireflies Owl City", ""), ("volume", "", "down"),
                             ("skip", "", ""), ("loop", "", "off")))
    assert r.kind == "commands"
    assert [(i.action, i.arg) for i in r.intents] == [
        ("play", "Fireflies Owl City"), ("volume", "down"), ("skip", None), ("loop", "off")]


@pytest.mark.parametrize("value,expected", [
    ("50", "50"), ("50%", "50"), ("fifty", "50"), ("up", "up"), ("louder", "up"),
    ("down", "down"), ("quieter", "down"), ("banana", None), ("-5", None)])
def test_volume_values(value, expected):
    r = vl.parse_reply(_cmds(("volume", "", value)))
    got = [(i.action, i.arg) for i in r.intents]
    assert got == ([("volume", expected)] if expected else [])


def test_a_loud_volume_from_the_llm_still_needs_a_yes():
    (intent,) = vl.parse_reply(_cmds(("volume", "", "300"))).intents
    assert intent.needs_confirmation          # same gate as the grammar's


def test_stop_and_clear_from_the_llm_still_need_a_yes():
    intents = vl.parse_reply(_cmds(("clear", "", ""), ("stop", "", ""))).intents
    assert all(i.needs_confirmation for i in intents)


def test_songs_are_capped_and_unknown_actions_dropped():
    acts = [("play", f"song {n}", "") for n in range(9)] + [("dance", "", "")]
    r = vl.parse_reply(_cmds(*acts))
    assert [i.arg for i in r.intents] == [f"song {n}" for n in range(vl.MAX_VOICE_PLAYS)]


def test_long_play_queries_are_cut():
    (intent,) = vl.parse_reply(_cmds(("play", "x" * 999, ""))).intents
    assert len(intent.arg) == vl.MAX_QUERY_CHARS


def test_a_song_put_in_the_wrong_field_still_plays():
    (intent,) = vl.parse_reply(_cmds(("play", "", "The Pretender by Foo Fighters"))).intents
    assert intent.arg == "The Pretender by Foo Fighters"


def test_json_spilling_into_a_title_is_not_a_song():
    # A small model under a strict schema was seen writing the next object
    # into the string instead of closing it.
    r = vl.parse_reply(_cmds(("play", "Fireflies by Owl City},{", ""), ("skip", "", "")))
    assert [(i.action, i.arg) for i in r.intents] == [("skip", None)]


def test_a_song_comes_back_as_title_and_artist_and_searches_as_both():
    (i,) = vl.parse_reply(_cmds(("play", "Danger Zone", "Kenny Loggins", ""))).intents
    assert (i.action, i.arg, i.artist) == ("play", "Danger Zone", "Kenny Loggins")
    assert i.search == "Kenny Loggins - Danger Zone"
    (j,) = vl.parse_reply(_cmds(("play", "lofi hip hop mix", "", ""))).intents
    assert j.artist is None and j.search == "lofi hip hop mix"


def test_when_puts_songs_next_or_now_and_nothing_else():
    def item(action, query="", artist="", value="", when=""):
        return {"action": action, "query": query, "artist": artist, "value": value, "when": when}
    r = vl.parse_reply(json.dumps({"kind": "commands", "reply": "", "actions": [
        item("skip"),
        item("album", "Pray for the Wicked", "Panic! At The Disco", "7,9", "now"),
        item("play", "Mr. Brightside", "The Killers", "", "NEXT"),
        item("top", "", "Owl City", "3", ""),
        item("play", "Fireflies", "Owl City", "", "tomorrow"),
        item("volume", "", "", "50", "now")]}))
    assert [(i.action, i.when) for i in r.intents] == [
        ("skip", None), ("album", "now"), ("play", "next"), ("top", None), ("play", None),
        ("volume", None)]
    assert r.intents[1].count == "7,9"
    # Answers without the field (older prompt, other model) still parse.
    (i,) = vl.parse_reply(_cmds(("play", "Fireflies", "Owl City", ""))).intents
    assert i.when is None


def test_the_prompt_says_when_songs_go_in_front():
    p = vl.SYSTEM_PROMPT
    assert "Songs go to the end of the queue unless they want them sooner" in p
    assert 'when = "next"' in p and '"skip this and play X instead"' in p
    assert 'a "now" song needs no separate skip' in p and 'Otherwise when = "".' in p


@pytest.mark.parametrize("said,count,want", [
    # "the seventh and ninth" came back once as "7-9" (2026-10-12): 7, 8 and 9.
    ("play the seventh and ninth songs off pray for the wicked", "7-9", "7,9"),
    ("can you skip this song and instead can you play the seventh and ninth song off of "
     "Panic at the Discos Pray for the Wicked album?", "7-9", "7,9"),
    ("play the 7th & 9th songs off it", "7-9", "7,9"),
    ("play tracks 2 and 5 off it", "2-5", "2,5"),
    ("play the 2nd one, and the 5th one", "2-5", "2,5"),
    # Real ranges stay ranges.
    ("play songs seven through nine off it", "7-9", "7-9"),
    ("play the seventh to ninth songs and turn it up", "7-9", "7-9"),
    ("play the first three songs off it and turn it up", "1-3", "1-3"),
    ("play tracks 1-3 and 5 off it", "1-3", "1-3"),
    ("play songs 3 to 5 and 7 off it", "3-5", "3-5"),
])
def test_tracks_said_with_and_are_a_list_not_a_range(said, count, want):
    (i,) = vl.settle_tracks(said, (vl.Intent("album", "Pray for the Wicked", count=count),))
    assert i.count == want and i.arg == "Pray for the Wicked"


@pytest.mark.parametrize("said,model,kept", [
    # "next"/"now" only when the words ask for it — "now" skips the song on.
    ("play mr brightside", "next", None), ("play fireflies and turn it up", "now", None),
    ("play mr brightside next", "next", "next"), ("after this one play the pretender", "next", "next"),
    ("play mr brightside now", "now", "now"), ("replace this song with hotel california", "now", "now"),
    ("skip this and play bohemian rhapsody instead", "now", "now"),
    ("can you skip this song and instead can you play the seventh and ninth song off of "
     "Panic at the Discos Pray for the Wicked album?", "now", "now"),
    ("play the pretender next", "now", "next"),               # said next, not now
    ("skip this and play the pretender", "next", "next"),     # the cog makes it "now"
])
def test_next_and_now_need_the_words(said, model, kept):
    (i,) = vl.settle_when(said, (vl.Intent("play", "X", artist="Y", when=model),))
    assert i.when == kept and (i.arg, i.artist) == ("X", "Y")


def test_router_puts_songs_last_unless_asked():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.content = json.dumps({"kind": "commands", "reply": "", "actions": [
                {"action": "play", "query": "Mr. Brightside", "artist": "The Killers",
                 "value": "", "when": "next"}]})
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url)
            try:
                plain = await router.ask("play mr brightside")
                asked = await router.ask("play mr brightside next")
            finally:
                await router.close()
            assert [i.when for i in plain.intents] == [None]
            assert [i.when for i in asked.intents] == ["next"]
            assert router.kinds["when_ignored"] == 1
    run(go())


def test_router_reads_a_and_b_as_two_tracks():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.content = _cmds(("album", "Pray for the Wicked", "Panic! At The Disco", "7-9"))
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url)
            try:
                r = await router.ask("play the seventh and ninth songs off pray for the wicked")
            finally:
                await router.close()
            assert [(i.action, i.count) for i in r.intents] == [("album", "7,9")]
            assert router.kinds["tracks_listed"] == 1
    run(go())


def test_the_whole_request_from_the_issue_parses_in_order():
    r = vl.parse_reply(_cmds(
        ("album", "Vices & Virtues", "Panic! At The Disco", "1-3"),
        ("volume", "", "", "50"), ("shuffle", "", "", ""), ("skip", "", "", ""),
        ("loop", "", "", "queue")))
    assert [(i.action, i.arg, i.artist, i.count) for i in r.intents] == [
        ("album", "Vices & Virtues", "Panic! At The Disco", "1-3"),
        ("volume", "50", None, None), ("shuffle", None, None, None),
        ("skip", None, None, None), ("loop", "queue", None, None)]


@pytest.mark.parametrize("item,expected", [
    (("album", "Ocean Eyes", "Owl City", ""), ("album", "Ocean Eyes", "Owl City", "all")),
    (("album", "", "Owl City", "1-3"), None),                       # which album?
    (("top", "", "Queen", "3"), ("top", None, "Queen", "3")),
    (("top", "Queen", "", ""), ("top", None, "Queen", "5")),         # artist in the wrong field
    (("latest", "", "Taylor Swift", ""), ("latest", None, "Taylor Swift", "1")),
    (("skip", "", "", "2"), ("skip", None, None, "2")),
    (("skip", "", "", "1"), ("skip", None, None, None)),
    (("skip", "", "", "99"), ("skip", None, None, "10")),
    (("loop", "", "", "queue"), ("loop", "queue", None, None)),
    (("loop", "", "", "song"), ("loop", "song", None, None)),
    (("loop", "", "", "on"), ("loop", "song", None, None)),
    (("loop", "", "", "all"), ("loop", "queue", None, None)),
    (("loop", "", "", "off"), ("loop", "off", None, None)),
    (("shuffle", "", "", ""), ("shuffle", None, None, None)),
])
def test_new_actions(item, expected):
    r = vl.parse_reply(_cmds(item))
    got = [(i.action, i.arg, i.artist, i.count) for i in r.intents]
    assert got == ([expected] if expected else [])


def test_an_album_counts_once_against_the_song_cap():
    acts = [("album", f"album {n}", "x", "all") for n in range(3)] + \
           [("play", f"song {n}", "y", "") for n in range(9)]
    r = vl.parse_reply(_cmds(*acts))
    assert len(r.intents) == vl.MAX_SONG_ACTIONS
    assert [i.action for i in r.intents].count("album") == 3


def test_no_usable_command_is_other():
    assert vl.parse_reply(_cmds(("play", "  ", ""), ("volume", "", "loud-ish"))).kind == "other"
    assert vl.parse_reply(_cmds()).kind == "other"


def test_question_reply_is_tidied_and_capped():
    long = "word " * 200
    r = vl.parse_reply(json.dumps({"kind": "question", "actions": [], "reply": long}))
    assert r.kind == "question" and len(r.reply) <= vl.MAX_REPLY_CHARS + 1
    assert r.reply.endswith("…")
    r = vl.parse_reply(json.dumps({"kind": "question", "actions": [],
                                   "reply": "  Canberra   is the\ncapital. "}))
    assert r.reply == "Canberra is the capital."


def test_question_without_an_answer_is_other():
    assert vl.parse_reply(json.dumps({"kind": "question", "actions": [], "reply": " "})).kind == "other"


def test_json_in_a_code_fence_is_accepted():
    r = vl.parse_reply('```json\n{"kind": "commands", "actions": [{"action": "skip", '
                       '"query": "", "value": ""}], "reply": ""}\n```')
    assert [i.action for i in r.intents] == ["skip"]


@pytest.mark.parametrize("bad", ["", "nope", "[1, 2]", "{not json}"])
def test_garbage_is_rejected(bad):
    assert vl.parse_reply(bad) is None


def test_unknown_kind_is_other():
    assert vl.parse_reply('{"kind": "weird", "actions": [], "reply": ""}').kind == "other"


@pytest.mark.parametrize("said,actions,vetoed", [
    # the refused action itself
    ("don't skip this one, I love it", ["skip"], True),
    ("do not clear the queue", ["clear"], True),
    ("never stop the music", ["stop"], True),
    ("wait, DON’T pause", ["pause"], True),
    ("don't turn it down, turn it up", ["volume"], False),     # said again, freely: model's call
    # with a "don't" in the sentence, only actions they named
    ("no wait, don't stop", ["skip"], True),
    ("please don't play that one again", ["skip"], True),
    ("don't loop it, just turn it up", ["volume"], False),
    ("don't skip it, shuffle the queue", ["shuffle"], False),
    ("don't play that, play the album instead", ["album"], False),
    ("don't stop the music, just skip this song", ["skip"], False),
    ("don't restart it, skip it", ["skip"], False),
    ("I don't care, just skip it", ["skip"], False),
    ("why don't you play some queen", ["play"], False),
    ("never mind, skip it", ["skip"], False),
    ("don't you dare skip this song", ["skip"], True),   # seen from the model, 2026-10-12
    ("don't u dare stop it", ["stop"], True),
    ("don't even think about skipping this", ["skip"], True),
    ("never ever stop the music", ["stop"], True),
    ("don't go skipping it", ["skip"], True),
    ("do not ever clear the queue", ["clear"], True),
    ("why don't you go play some queen", ["play"], False),
    ("don't think, just skip it", ["skip"], False),
    # no refusal: left to the model
    ("stop the music and clear the queue", ["stop", "clear"], False),
    ("hold on a sec", ["pause"], False),
])
def test_the_safety_net_for_dont(said, actions, vetoed):
    assert vl.contradicts(said, [vl.Intent(a) for a in actions]) is vetoed


# ── the router, against a stand-in for OpenRouter ────────────────────────────

class FakeOpenRouter:
    """A local aiohttp server answering /chat/completions."""

    def __init__(self):
        self.requests = []
        self.status = 200
        self.content = _cmds(("skip", "", ""))
        self.queued: list[str] = []          # answers for the next requests, in order
        self.delay = 0.0

    async def handler(self, request):
        from aiohttp import web
        self.requests.append((dict(request.headers), await request.json()))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.status != 200:
            return web.Response(status=self.status, text='{"error": {"message": "Insufficient credits"}}')
        content = self.queued.pop(0) if self.queued else self.content
        return web.json_response({
            "model": "google/gemini-2.5-flash-lite",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 20, "cost": 0.00005}})

    async def __aenter__(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_post("/api/v1/chat/completions", self.handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/api/v1"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()


def run(coro):
    return asyncio.run(coro)


def test_router_success_sends_the_key_and_returns_a_checked_answer():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.content = _cmds(("play", "Fireflies Owl City", ""), ("play", "The Pretender", ""))
            router = vl.LLMRouter(KEY, model="google/gemini-2.5-flash-lite",
                                  fallbacks=("openai/gpt-4.1-nano",), base_url=srv.url)
            try:
                r = await router.ask("I want to hear two songs", now_playing="X — Y")
            finally:
                await router.close()
            headers, body = srv.requests[0]
            assert headers["Authorization"] == f"Bearer {KEY}"
            assert body["models"] == ["google/gemini-2.5-flash-lite", "openai/gpt-4.1-nano"]
            assert "I want to hear two songs" in body["messages"][1]["content"]
            assert [i.arg for i in r.intents] == ["Fireflies Owl City", "The Pretender"]
            assert r.cost == pytest.approx(0.00005) and router.cost == pytest.approx(0.00005)
            assert router.calls == 1 and router.failures == 0 and router.kinds == {"commands": 1}
            assert router.median_seconds() is not None
    run(go())


def test_router_http_error_is_none_and_never_logs_the_key(caplog):
    async def go():
        async with FakeOpenRouter() as srv:
            srv.status = 402
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url)
            try:
                with caplog.at_level(logging.WARNING):
                    assert await router.ask("hello") is None
            finally:
                await router.close()
            assert router.failures == 1 and "HTTP 402" in router.last_error
    run(go())
    assert KEY not in caplog.text and "voice LLM: HTTP 402" in caplog.text


def test_router_timeout_is_none():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.delay = 1.0
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url, timeout=0.2)
            try:
                assert await router.ask("hello") is None
            finally:
                await router.close()
            assert "timeout" in router.last_error
    run(go())


def test_router_rejects_an_unusable_answer():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.content = "Sure! I'd love to help with that."
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url)
            try:
                assert await router.ask("hello") is None
            finally:
                await router.close()
            assert "unusable" in router.last_error
    run(go())


def test_router_does_nothing_when_the_model_does_what_was_refused():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.content = _cmds(("skip", "", ""))
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url)
            try:
                r = await router.ask("don't skip this one, I love it")
            finally:
                await router.close()
            assert r.kind == "other" and r.intents == ()
            assert router.kinds == {"vetoed": 1, "other": 1}
    run(go())


def test_a_garbled_answer_is_retried_once_on_the_fallback_model():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.queued = ['{"kind": "commands", "actions": [{"action": "', _cmds(("skip", "", ""))]
            router = vl.LLMRouter(KEY, model="google/x", fallbacks=("openai/y",), base_url=srv.url)
            try:
                r = await router.ask("skip it")
            finally:
                await router.close()
            assert [i.action for i in r.intents] == ["skip"]
            models = [body["model"] for _h, body in srv.requests]
            assert models == ["google/x", "openai/y"]
            assert "models" not in srv.requests[1][1]          # nothing left to fall back to
            assert router.calls == 2 and router.kinds == {"retried": 1, "commands": 1}
            assert router.cost == pytest.approx(0.0001)          # both answers cost
    run(go())


def test_without_a_fallback_a_garbled_answer_is_given_up_on():
    async def go():
        async with FakeOpenRouter() as srv:
            srv.content = "{oops"
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url)
            try:
                assert await router.ask("hello") is None
            finally:
                await router.close()
            assert len(srv.requests) == 1
    run(go())


def test_router_rate_limit_skips_without_calling():
    async def go():
        async with FakeOpenRouter() as srv:
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url, per_minute=2)
            try:
                assert await router.ask("one") is not None
                assert await router.ask("two") is not None
                assert await router.ask("three") is None
            finally:
                await router.close()
            assert len(srv.requests) == 2 and "rate limited" in router.last_error
    run(go())


def test_router_with_nothing_to_say_does_not_call():
    async def go():
        router = vl.LLMRouter(KEY, model="m", base_url="http://127.0.0.1:9/api/v1")
        assert await router.ask("   ") is None
        assert router.calls == 0
    run(go())


def test_router_unreachable_is_none():
    async def go():
        router = vl.LLMRouter(KEY, model="m", base_url="http://127.0.0.1:9/api/v1", timeout=2)
        try:
            assert await router.ask("hello") is None
        finally:
            await router.close()
        assert router.failures == 1 and KEY not in router.last_error
    run(go())


def test_api_keys_are_redacted_from_any_text():
    text = f"failed with key {KEY} and header Bearer {KEY}"
    out = redact_secrets(text)
    assert KEY not in out and "sk-or-v1-<redacted>" in out


# ── songs the model may not know, and songs described rather than named ──────

def test_the_prompt_keeps_unknown_titles_and_looks_up_described_songs():
    p = vl.SYSTEM_PROMPT
    assert "heard = the title as the speech-to-text wrote it" in p
    assert "never put a different song in its place" in p      # the "Cordial" swap
    assert "find:" in p and "samples Panic! At The Disco" in p
    assert '"tip this" is "skip this"' in p                       # mishearings are commands


def test_play_keeps_what_was_heard_when_it_differs():
    item = {"action": "play", "query": "No Disrespect", "artist": "Set It Off",
            "heard": "cordial", "value": "", "when": ""}
    i = vl._intent_from(item)
    assert (i.arg, i.artist, i.heard) == ("No Disrespect", "Set It Off", "cordial")
    same = vl._intent_from({**item, "query": "cordial"})
    assert same.heard is None                                    # nothing extra to check
    only_heard = vl._intent_from({**item, "query": ""})
    assert only_heard.arg == "cordial"


def test_find_is_a_described_song():
    i = vl._intent_from({"action": "find", "query": "Yung Gravy song that samples Panic! At The Disco",
                         "artist": "Yung Gravy", "heard": "", "value": "", "when": "now"})
    assert (i.action, i.artist, i.when) == ("find", "Yung Gravy", "now")
    assert i.action in vl.SONG_ACTIONS
    assert vl._intent_from({"action": "find", "query": "", "artist": "", "heard": "",
                            "value": "", "when": ""}) is None


def test_choice_request_lists_real_songs_and_wants_a_number():
    from core.music_catalog import CatalogTrack
    songs = [CatalogTrack("Mr. Clean", "Yung Gravy"), CatalogTrack("I Write Hymns Not Travesties",
                                                                    "Yung Gravy")]
    body = vl.build_choice_request("play I write him's not travis d's by young gravy",
                                   "I write him's not travis d's by Yung Gravy", songs, model="m")
    user = body["messages"][1]["content"]
    assert "1. Mr. Clean - Yung Gravy" in user and "2. I Write Hymns Not Travesties - Yung Gravy" in user
    assert body["response_format"]["json_schema"]["schema"] == vl.CHOICE_SCHEMA


@pytest.mark.parametrize("content,n,want", [
    ('{"choice": 2}', 3, 2), ('{"choice": 0}', 3, 0), ('{"choice": 4}', 3, None),
    ('{"choice": -1}', 3, None), ('{"choice": "2"}', 3, 2), ('{"choice": true}', 3, None),
    ("no idea", 3, None), ('{"pick": 1}', 3, None),
])
def test_parse_choice(content, n, want):
    assert vl.parse_choice(content, n) == want


def test_lookup_request_searches_the_description_cheaply():
    body = vl.build_lookup_request("the song that samples Panic! At The Disco", "Yung Gravy",
                                   model="m")
    assert body["plugins"] == [{"id": "web", "engine": "exa", "max_results": 5}]
    assert body["messages"][1]["content"] == ("song: the song that samples Panic! At The Disco "
                                              "Yung Gravy")
    named = vl.build_lookup_request("Yung Gravy song that samples Panic", "Yung Gravy", model="m")
    assert named["messages"][1]["content"] == "song: Yung Gravy song that samples Panic"


@pytest.mark.parametrize("content,want", [
    ('{"found": true, "title": "I Write Hymns Not Travesties", "artist": "Yung Gravy"}',
     ("I Write Hymns Not Travesties", "Yung Gravy")),
    ('{"found": false, "title": "Mr. Brightside", "artist": "The Killers"}', None),
    ('{"found": true, "title": "", "artist": "X"}', None),
    ('{"found": true, "title": "A},{B", "artist": "X"}', None),
    ("Mr. Brightside", None),
])
def test_parse_lookup(content, want):
    assert vl.parse_lookup(content) == want


def test_router_choose_song_and_lookup_song():
    from core.music_catalog import CatalogTrack

    async def go():
        async with FakeOpenRouter() as srv:
            router = vl.LLMRouter(KEY, model="m", base_url=srv.url, web_per_hour=1)
            try:
                srv.queued = ['{"choice": 2}', '{"choice": 0}', "garbage"]
                songs = [CatalogTrack("A", "X"), CatalogTrack("B", "X")]
                assert await router.choose_song("play b", "b by x", songs) == 1
                assert await router.choose_song("play c", "c by x", songs) is None
                assert await router.choose_song("play c", "c by x", songs) is None
                assert router.failures == 1                       # the garbage, recorded
                srv.queued = ['{"found": true, "title": "I Write Hymns Not Travesties", '
                              '"artist": "Yung Gravy"}']
                assert await router.lookup_song("song that samples P!ATD", "Yung Gravy") == (
                    "I Write Hymns Not Travesties", "Yung Gravy")
                _h, body = srv.requests[-1]
                assert body["plugins"][0]["id"] == "web"
                # one an hour here: the second is refused without a call
                n = len(srv.requests)
                assert await router.lookup_song("another", "") is None and len(srv.requests) == n
                router.web = False
                router.web_per_hour = 99
                assert await router.lookup_song("another", "") is None and len(srv.requests) == n
                assert router.kinds["song_picked"] == 1 and router.kinds["web_found"] == 1
            finally:
                await router.close()
    run(go())
