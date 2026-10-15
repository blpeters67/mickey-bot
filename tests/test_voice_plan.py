"""core/voice_plan.py — what the planner's JSON means, before anything runs:
parsing, checks, run order, and queue references; plus joining the parts of
a long utterance (core/voice_intent.stitch). Pure: no cog, no network."""
from __future__ import annotations

import json

import pytest

from core import voice_plan as vp
from core.state import GuildState, qid
from core.voice_intent import Intent, stitch
from core.voice_tools import OPS


def step(id, op, **kw):
    base = {"id": id, "op": op, "title": "", "artist": "", "heard_title": "", "heard_artist": "",
            "album": "", "tracks": "", "target": "", "to": "", "value": "", "count": 1,
            "when": "", "after": []}
    base.update(kw)
    return base


def parse(*steps, **kw):
    return vp.parse_plan(json.dumps({"kind": "plan", "steps": list(steps), "reply": "",
                                     "ask": kw.get("ask", ""), "lookups": []}), ops=OPS)


def queue(*titles):
    st = GuildState()
    st.queue.extend({"title": t} for t in titles)
    st.queue.append({"title": "playing"})
    st.current = st.queue.pop()
    return st


# ── parsing ───────────────────────────────────────────────────────────────────

def test_kinds_and_garbage():
    assert vp.parse_plan("not json", ops=OPS) is None
    assert vp.parse_plan("", ops=OPS) is None
    p = vp.parse_plan('```json\n{"kind":"answer","reply":"Paris.","steps":[],"ask":"","lookups":[]}\n```',
                      ops=OPS)
    assert p.kind == "answer" and p.reply == "Paris."
    assert vp.parse_plan('{"kind":"other"}', ops=OPS).kind == "other"
    p = vp.parse_plan('{"kind":"lookup","lookups":[{"tool":"album","title":"X","artist":"Y"}]}',
                      ops=OPS)
    assert p.kind == "lookup" and p.lookups[0]["tool"] == "album"


def test_skip_of_a_queued_song_is_a_removal_not_a_skip():
    p = parse(step(1, "skip", target="pos:2"))
    assert [(s.op, s.target) for s in p.steps] == [("remove", "pos:2")]
    p = parse(step(1, "skip", target="current"))
    assert p.steps[0].op == "skip"


def test_position_ranges_and_duplicate_ids_are_normalised():
    p = parse(step(1, "remove", target="pos:1-3"), step(1, "shuffle"))
    assert p.steps[0].target == "range:1-3" and [s.id for s in p.steps] == [1, 2]


def test_garbled_fields_and_missing_arguments_are_refused_with_reasons():
    p = parse(step(1, "add_song", title='X"},{"op'), step(2, "move", target="pos:1"),
              step(3, "volume"))
    assert not p.steps
    assert len(p.problems) == 3 and any("garbled" in x for x in p.problems)
    assert any("missing its to" in x for x in p.problems)


# ── run order ─────────────────────────────────────────────────────────────────

def test_only_after_moves_a_step_behind_what_it_waits_for():
    p = parse(step(1, "add_album", album="A", tracks="latter half", after=[2]),
              step(2, "add_song", title="H", count=5), step(3, "shuffle"))
    assert [s.id for s in p.steps] == [2, 1, 3]


def test_references_to_a_step_order_it_too():
    p = parse(step(1, "move", target="step:2", to="front"), step(2, "add_song", title="X"))
    assert [s.id for s in p.steps] == [2, 1]
    assert vp.needs(p.steps[1]) == {2} and vp.deps(p.steps[1]) == {2}


# ── references ───────────────────────────────────────────────────────────────

def test_positions_count_the_upcoming_queue_only():
    st = queue("A", "B", "C", "D")
    ids = st.queue.ids()
    up = list(st.queue)
    assert vp.resolve_target("pos:1", up, current=st.current) == [ids[0]]
    assert vp.resolve_target("2", up, current=st.current) == [ids[1]]
    assert vp.resolve_target("last", up) == [ids[3]]
    assert vp.resolve_target("last:2", up) == ids[2:]
    assert vp.resolve_target("range:2-3", up) == ids[1:3]
    assert vp.resolve_target("current", up, current=st.current) == ["current"]
    with pytest.raises(vp.RefError):
        vp.resolve_target("pos:9", up)
    with pytest.raises(vp.RefError):
        vp.resolve_target("pos:1", [])


def test_step_references_bind_this_requests_new_entries():
    st = queue("Old Phoenix", "B")
    st.queue.extend([{"title": "Phoenix"}, {"title": "Phoenix"}])
    a, b = st.queue.ids()[-2:]
    made = {4: [a, b]}
    up = list(st.queue)
    assert vp.resolve_target("step:4", up, step_entries=made) == [a, b]
    assert vp.resolve_target("step:4#2", up, step_entries=made) == [b]
    with pytest.raises(vp.RefError):
        vp.resolve_target("step:4#3", up, step_entries=made)


def test_names_match_by_sound_and_copies_resolve_to_the_newest():
    st = queue("WAX - ALTE WUNDEN", "Fall Out Boy - The Phoenix", "Saint Chaos - Ghosts & Monsters",
               "Fall Out Boy - The Phoenix")
    ids = st.queue.ids()
    up = list(st.queue)
    assert vp.resolve_target("match:Wax by Alte One Den", up) == [ids[0]]
    assert vp.resolve_target("match:the phoenix", up) == [ids[3]]            # newest copy
    assert vp.resolve_target("match:the phoenix", up, mine=[ids[1]]) == [ids[1]]   # theirs
    with pytest.raises(vp.RefError):
        vp.resolve_target("match:hotel california", up)


def test_two_different_songs_fitting_a_name_equally_is_a_question():
    st = queue("Adele - Hello", "Lionel Richie - Hello")
    with pytest.raises(vp.RefError, match="could be"):
        vp.resolve_target("match:hello", list(st.queue))


def test_destinations():
    st = queue("A", "B", "C")
    up = list(st.queue)
    assert vp.resolve_destination("front", up) == 0
    assert vp.resolve_destination("end", up) == 3
    assert vp.resolve_destination("pos:2", up) == 1
    assert vp.resolve_destination("after:pos:1", up, view=up) == 1
    assert vp.resolve_destination("before:last", up, view=up) == 2


# ── the grammar's commands as a plan ─────────────────────────────────────────

def test_from_intents_and_folding_skip_this_and_play_x():
    p = vp.fold_skips(vp.from_intents([Intent("skip"), Intent("play", "X", artist="Y"),
                                       Intent("volume", "50")]))
    assert [(s.op, s.when, s.took_skip) for s in p.steps] == [("add_song", "now", True),
                                                              ("volume", "", False)]
    assert p.steps[0].title == "X" and p.steps[0].artist == "Y"


def test_describe_for_logs_has_no_user_data_beyond_the_plan():
    p = parse(step(1, "add_song", title="X", artist="Y", count=5), step(2, "remove", target="last"))
    d = vp.describe(p)
    assert "1:add_song(" in d and "count=5" in d and "2:remove(target='last')" in d


# ── one sentence, several recognised parts ────────────────────────────────────

@pytest.mark.parametrize("a,b,want", [
    ("play the first half of pray for the wicked, shuffle", "shuffle, skip the second song",
     "play the first half of pray for the wicked, shuffle skip the second song"),
    ("then set the bot to loop.", "to loop.", "then set the bot to loop."),
    ("play the phoenix by fall", "fallout boy then shuffle", "play the phoenix by fallout boy then shuffle"),
    ("hey dj play x", "", "hey dj play x"),
    ("", "and then y", "and then y"),
])
def test_stitch_takes_the_overlap_once(a, b, want):
    assert stitch(a, b) == want


def test_an_entry_id_written_as_the_context_shows_it_is_that_entry():
    # seen from gemini-2.5-flash (2026-10-14): target "q23" instead of "entry:q23"
    p = parse(step(1, "remove", target="q23"), step(2, "move", target="[q7]", to="after:q9"))
    assert [(s.target, s.to) for s in p.steps] == [("entry:q23", ""), ("entry:q7", "after:entry:q9")]
    assert not p.problems
