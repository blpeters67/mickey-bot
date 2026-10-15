"""The voice DJ's executor (core/voice_exec.py) through the real cog.

The central case is the request from the round-8 brief, run as the planner
would answer it (model-shaped JSON → core/voice_plan.parse_plan → the cog's
plan lane → PlanRun), with a stand-in catalogue and YouTube, a stand-in
player whose queue advances like the real one, and seeded shuffles. Asserted:
every intermediate queue state, which entries (by occurrence id) each step
touched, what is playing, and the final order — with something already
playing, with an idle player, and with older copies of the same songs queued.

MOCKED: these prove the executor's decisions, ids and order — not what a real
model answers (scripts/voice_plan_bench.py measures that) nor real lookups.
"""
from __future__ import annotations

import asyncio
import json
import random

import pytest

np = pytest.importorskip("numpy")

import cogs.voicedj as vd                                       # noqa: E402
from core import music_catalog as mc                            # noqa: E402
from core import voice_plan as vp                               # noqa: E402
from core.state import get_state, qid                           # noqa: E402
from core.voice_tools import OPS                                # noqa: E402
from tests.test_voice_plans import planned, replies             # noqa: E402
from tests.test_voice_smart import FakeLLM                      # noqa: E402
from tests.test_voicedj_flow import GID, OTHER, UID             # noqa: E402

PANIC = "Panic! At the Disco"
PFTW = mc.CatalogAlbum("Pray for the Wicked", PANIC, tuple(
    mc.CatalogTrack(t, PANIC, 200, n) for n, t in enumerate(
        ["(Fuck A) Silver Lining", "Say Amen (Saturday Night)", "Hey Look Ma, I Made It",
         "High Hopes", "Roaring 20s", "Dancing's Not a Crime", "One of the Drunks",
         "The Overpass", "King of the Clouds", "Old Fashioned", "Dying in LA"], 1)),
    source="deezer")
PFTW_Q = [f"{PANIC} - {t.title}" for t in PFTW.tracks]
PHOENIX = mc.CatalogTrack("The Phoenix", "Fall Out Boy", 244)
HYMNS = mc.CatalogTrack("I Write Hymns Not Travesties", "Yung Gravy", 151)
SONGS = [PHOENIX, HYMNS]

CENTRAL = ("Hey DJ, play the first half of the Pray for the Wicked album by Panic! At The "
           "Disco, shuffle, skip the second song in the queue, play The Phoenix by Fallout Boy, "
           "move the Phoenix to the front of the queue, remove the last song from the queue, "
           "then shuffle the queue again and play the latter half of Pray for the Wicked, but "
           "only after you have queued I Write Hymns, not Travesties by Yung Gravy five "
           "separate times.")


def step(id, op, **kw):
    base = {"id": id, "op": op, "title": "", "artist": "", "heard_title": "", "heard_artist": "",
            "album": "", "tracks": "", "target": "", "to": "", "value": "", "count": 1,
            "when": "", "after": []}
    base.update(kw)
    return base


def plan_json(steps, **kw) -> str:
    return json.dumps({"kind": "plan", "steps": steps, "reply": "", "ask": kw.get("ask", ""),
                       "lookups": []})


#: How the planner should answer CENTRAL: in the order said, except that the
#: five hymns come before the latter half, which waits for them.
CENTRAL_STEPS = [
    step(1, "add_album", album="Pray for the Wicked", artist="Panic! At The Disco",
         tracks="first half"),
    step(2, "shuffle"),
    step(3, "remove", target="pos:2"),
    step(4, "add_song", title="The Phoenix", artist="Fall Out Boy", heard_title="The Phoenix",
         heard_artist="Fallout Boy"),
    step(5, "move", target="step:4", to="front"),
    step(6, "remove", target="last"),
    step(7, "shuffle"),
    step(8, "add_song", title="I Write Hymns Not Travesties", artist="Yung Gravy",
         heard_title="I Write Hymns, not Travesties", heard_artist="Yung Gravy", count=5),
    step(9, "add_album", album="Pray for the Wicked", artist="Panic! At The Disco",
         tracks="latter half", after=[8]),
]
#: The same plan with the latter half listed where it was said (before the
#: hymns) and told to wait for them: the executor reorders it.
CENTRAL_STEPS_AS_SAID = CENTRAL_STEPS[:7] + [
    step(8, "add_album", album="Pray for the Wicked", artist="Panic! At The Disco",
         tracks="latter half", after=[9]),
    step(9, "add_song", title="I Write Hymns Not Travesties", artist="Yung Gravy",
         heard_title="I Write Hymns, not Travesties", heard_artist="Yung Gravy", count=5),
]


def with_catalogue(rig, *, missing=()):
    rig.albums["pray for the wicked"] = PFTW
    rig.song_lookups = []

    async def find_song(title, artist="", *, heard="", heard_artist=""):
        rig.song_lookups.append((title, artist, heard, heard_artist))
        if title in missing:
            return None
        return mc.pick_song(SONGS, [title, heard], heard_artist or artist, said=heard)
    rig.monkeypatch.setattr(vd.music_catalog, "find_song", find_song)

    async def search(query, *, limit=5):
        rig.searches.append(query)
        if any(m in query for m in missing):
            return []
        return [{"title": f"{query} (Official Audio)", "uploader": query.split(" - ")[0],
                 "duration": 0, "stream_url": None,
                 "webpage_url": f"https://youtu.be/{abs(hash(query)) % 10 ** 8}",
                 "_needs_resolve": True, "_search_query": query}]
    rig.monkeypatch.setattr(vd.audio, "search_tracks", search)


def titles(entries):
    return [e["title"] for e in entries]


def playing(rig, title="Already Playing", *, queued=()):
    st = get_state(GID)
    rig.vc.playing = True
    st.queue.clear()
    st.queue.append({"title": title, "_status": "playing"})   # it came from the queue,
    st.current = st.queue.popleft()                           # so it has an id
    st.queue.extend({"title": t} for t in queued)
    return st


async def run_central(rig, steps=CENTRAL_STEPS, seed=7):
    rig.cog._rng = random.Random(seed)
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json(steps), ops=OPS))
    rig.say(CENTRAL)
    await rig.settle(2.0)
    return rig.cog._last_run[GID]


def check_central(rig, run, *, before_current, before_ids):
    """Every effect of the brief's list, from the run's own trace (the queue
    as each step left it) and the step results (which ids each touched)."""
    st = get_state(GID)
    order = [s.id for s in run.plan.steps]
    res = {s.op + str(s.id): run.results[s.id] for s in run.plan.steps}
    assert all(r.status == "done" for r in run.results.values()), [
        (r.step.label, r.status, r.message) for r in run.results.values()]
    trace = {sid: (cur, ids) for sid, cur, ids in run.trace}
    by_id = {}
    for e in list(st.queue) + ([st.current] if st.current else []):
        by_id[qid(e)] = e

    album1 = run.results[1].added
    # 1. the first half of the 11-track album: tracks 1-6, as six new entries
    assert len(album1) == 6 and len(set(album1)) == 6
    cur1, q1 = trace[1]
    if before_current is None:
        # an idle player took the first of them: it is playing, the rest queued
        assert cur1 == album1[0] and q1 == list(album1[1:])
    else:
        assert cur1 == before_current and q1 == before_ids + list(album1)
    # 2. shuffled: the same entries, current untouched
    cur2, q2 = trace[2]
    assert cur2 == cur1 and sorted(q2) == sorted(q1)
    # 3. the second *upcoming* entry removed — not a skip
    cur3, q3 = trace[3]
    assert cur3 == cur1 and run.results[3].removed == (q2[1],) and q3 == q2[:1] + q2[2:]
    assert rig.vc.stopped == 0
    # 4. The Phoenix by Fall Out Boy added (its own new id), last
    (phoenix,) = run.results[4].added
    cur4, q4 = trace[4]
    assert q4 == q3 + [phoenix] and "The Phoenix" in by_id.get(phoenix, {"title": "The Phoenix"})["title"]
    # 5. that very occurrence moved to the front — older copies aren't it
    cur5, q5 = trace[5]
    assert run.results[5].moved == (phoenix,) and q5 == [phoenix] + q3
    # 6. the last upcoming entry removed
    cur6, q6 = trace[6]
    assert run.results[6].removed == (q5[-1],) and q6 == q5[:-1]
    # 7. shuffled again
    cur7, q7 = trace[7]
    assert sorted(q7) == sorted(q6) and cur7 == cur1
    # 8-12. five separate hymns entries, before the latter half
    hymn_step = next(s.id for s in run.plan.steps if s.op == "add_song" and s.count == 5)
    half_step = next(s.id for s in run.plan.steps if s.op == "add_album" and "latter" in s.tracks)
    hymns = run.results[hymn_step].added
    assert len(hymns) == 5 and len(set(hymns)) == 5
    assert order.index(hymn_step) < order.index(half_step)
    _c, qh = trace[hymn_step]
    assert qh == q7 + list(hymns)
    # 13. the latter half: tracks 7-11, in album order, at the end
    half = run.results[half_step].added
    final_cur, final = trace[half_step]
    assert final == q7 + list(hymns) + list(half) and final_cur == cur1
    assert [by_id[i]["title"] for i in half] == PFTW_Q[6:]
    assert [e["title"] for e in st.queue] == [by_id[i]["title"] for i in final]
    assert qid(st.current) == final_cur
    return res


@planned
async def test_central_request_with_something_already_playing(rig):
    with_catalogue(rig)
    st = playing(rig, queued=("Queued A", "Queued B"))
    before_current, before_ids = qid(st.current), st.queue.ids()
    run = await run_central(rig)
    check_central(rig, run, before_current=before_current, before_ids=before_ids)
    assert st.current["title"] == "Already Playing"                 # never skipped
    (reply,) = replies(rig)
    assert reply.splitlines()[0].startswith("✅ Queued 6 songs from **Pray for the Wicked**")
    assert "Moved **Fall Out Boy - The Phoenix (Official Audio)** to the front" in reply
    assert "×5" in reply
    # the catalogue was asked for the artist as said, not only the model's fix
    assert ("The Phoenix", "Fall Out Boy", "The Phoenix", "Fallout Boy") in rig.song_lookups


@planned
async def test_central_request_into_an_idle_player(rig):
    with_catalogue(rig)
    st = get_state(GID)
    st.queue.clear()
    run = await run_central(rig)
    check_central(rig, run, before_current=None, before_ids=[])
    assert st.current["title"] == PFTW_Q[0]                       # the first track started


@planned
async def test_central_request_as_said_order_is_reordered_by_its_dependency(rig):
    with_catalogue(rig)
    st = playing(rig, queued=("Queued A",))
    run = await run_central(rig, CENTRAL_STEPS_AS_SAID, seed=11)
    ops = [s.op for s in run.plan.steps]
    assert ops[-2:] == ["add_song", "add_album"]                  # hymns first, then the half
    check_central(rig, run, before_current=qid(st.current), before_ids=_ids_before(run))


def _ids_before(run):
    """The ids queued before the request (the first step's entries removed
    from the queue as it left it)."""
    sid, _cur, ids = run.trace[0]
    added = set(run.results[sid].added)
    return [i for i in ids if i not in added]


@planned
async def test_central_request_with_older_copies_of_the_same_songs_queued(rig):
    with_catalogue(rig)
    st = playing(rig, queued=("Fall Out Boy - The Phoenix (Official Audio)",
                              "Yung Gravy - I Write Hymns Not Travesties (Official Audio)",
                              "Fall Out Boy - The Phoenix (Official Audio)"))
    old = st.queue.ids()
    run = await run_central(rig, seed=3)
    check_central(rig, run, before_current=qid(st.current), before_ids=old)
    (phoenix,) = run.results[4].added
    assert phoenix not in old
    # the shuffles and the removals may have taken some of the old copies,
    # but the move took the new one, and every copy kept its own id
    ids = st.queue.ids()
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("seed", [1, 2, 5, 9])
def test_central_request_holds_for_any_shuffle(monkeypatch, seed):
    @planned
    async def body(rig):
        with_catalogue(rig)
        st = playing(rig, queued=("Queued A", "Queued B", "Queued C"))
        run = await run_central(rig, seed=seed)
        check_central(rig, run, before_current=qid(st.current), before_ids=_ids_before(run))
    body(monkeypatch)


# ── references against a queue that changes under the request ─────────────────

@planned
async def test_playback_advancing_between_steps_doesnt_shift_what_a_position_means(rig):
    with_catalogue(rig)
    st = playing(rig, queued=("A", "B", "C", "D"))
    a, b, c, d = st.queue.ids()
    gate = asyncio.Event()
    real = vd.music_catalog.find_song

    async def slow_song(*args, **kw):
        await gate.wait()
        return await real(*args, **kw)
    rig.monkeypatch.setattr(vd.music_catalog, "find_song", slow_song)
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "add_song", title="The Phoenix", artist="Fall Out Boy"),
        step(2, "remove", target="pos:2")]), ops=OPS))
    rig.say("hey dj play the phoenix by fall out boy and then remove the second song in the queue")
    await asyncio.sleep(0.3)
    # Meanwhile the song ends: A starts playing (the queue moves up one).
    st.current = st.queue.popleft()
    gate.set()
    await rig.settle(1.0)
    run = rig.cog._last_run[GID]
    # "The second song in the queue" was B when the request was made — and
    # B is what goes, not C (which is second now).
    assert run.results[2].removed == (b,)
    assert st.queue.ids()[:2] == [c, d]


@planned
async def test_an_entry_someone_else_removed_meanwhile_is_a_conflict_not_another_song(rig):
    with_catalogue(rig)
    st = playing(rig, queued=("A", "B", "C"))
    a, b, c = st.queue.ids()
    gate = asyncio.Event()
    real = vd.music_catalog.find_song

    async def slow_song(*args, **kw):
        await gate.wait()
        return await real(*args, **kw)
    rig.monkeypatch.setattr(vd.music_catalog, "find_song", slow_song)
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "add_song", title="The Phoenix", artist="Fall Out Boy"),
        step(2, "remove", target="pos:2")]), ops=OPS))
    rig.say("hey dj play the phoenix by fall out boy then remove the second song")
    await asyncio.sleep(0.3)
    from core import queue_ops
    queue_ops.remove_ids(st, [b])                    # a typed `remove 2` by someone else
    gate.set()
    await rig.settle(1.0)
    run = rig.cog._last_run[GID]
    assert run.results[2].status == "conflict" and not run.results[2].removed
    assert a in st.queue.ids() and c in st.queue.ids()          # nothing else removed instead
    (reply,) = replies(rig)
    assert "isn't in the queue any more" in reply


@planned
async def test_skip_this_after_the_song_already_ended_skips_nothing(rig):
    with_catalogue(rig)
    st = playing(rig, "Song One", queued=("Song Two",))
    gate = asyncio.Event()
    real = vd.music_catalog.find_song

    async def slow_song(*args, **kw):
        await gate.wait()
        return await real(*args, **kw)
    rig.monkeypatch.setattr(vd.music_catalog, "find_song", slow_song)
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "add_song", title="The Phoenix", artist="Fall Out Boy"),
        step(2, "skip")]), ops=OPS))
    rig.say("hey dj queue the phoenix by fall out boy and then skip this one")
    await asyncio.sleep(0.3)
    st.current = st.queue.popleft()                  # Song One ended on its own
    gate.set()
    await rig.settle(1.0)
    run = rig.cog._last_run[GID]
    assert run.results[2].status == "conflict" and rig.vc.stopped == 0
    assert st.current["title"] == "Song Two"


@planned
async def test_moving_a_new_song_that_started_playing_meanwhile_says_so(rig):
    with_catalogue(rig)
    st = get_state(GID)
    st.queue.clear()                                  # idle: the new song starts at once
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "add_song", title="The Phoenix", artist="Fall Out Boy"),
        step(2, "move", target="step:1", to="front")]), ops=OPS))
    rig.say("hey dj play the phoenix by fall out boy and move it to the front")
    await rig.settle(1.0)
    run = rig.cog._last_run[GID]
    assert qid(st.current) == run.results[1].added[0]
    assert run.results[2].status == "noop" and "playing now" in run.results[2].message


# ── what a failed step means for the rest ────────────────────────────────────

@planned
async def test_a_failed_insertion_never_moves_or_removes_some_other_song(rig):
    with_catalogue(rig, missing=("The Phoenix",))
    st = playing(rig, queued=("A", "B", "C"))
    before = st.queue.ids()
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "add_song", title="The Phoenix", artist="Fall Out Boy"),
        step(2, "move", target="step:1", to="front"),
        step(3, "remove", target="last"),
        step(4, "volume", value="-20")]), ops=OPS))
    rig.say("hey dj play the phoenix by fall out boy, move it to the front, remove the last "
            "song and turn it down by 20")
    await rig.settle(1.0)
    run = rig.cog._last_run[GID]
    assert run.results[1].status == "failed"
    assert run.results[2].status == "skipped" and run.results[3].status == "skipped"
    assert run.results[4].status == "done" and st.volume == pytest.approx(0.8)
    assert st.queue.ids() == before                               # untouched
    (reply,) = replies(rig)
    assert "couldn't find" in reply.lower() and "So I didn't move" in reply


@planned
async def test_a_plan_cut_off_by_a_typed_clear_reports_what_wasnt_done(rig):
    with_catalogue(rig)
    st = playing(rig, queued=("A",))
    gate = asyncio.Event()
    real = vd.music_catalog.find_song
    calls = []

    async def slow_song(title, *args, **kw):
        calls.append(title)
        if len(calls) == 2:
            await gate.wait()
        return await real(title, *args, **kw)
    rig.monkeypatch.setattr(vd.music_catalog, "find_song", slow_song)
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "add_song", title="The Phoenix", artist="Fall Out Boy"),
        step(2, "add_song", title="I Write Hymns Not Travesties", artist="Yung Gravy"),
        step(3, "shuffle")]), ops=OPS))
    rig.say("hey dj play the phoenix and then i write hymns not travesties then shuffle")
    await asyncio.sleep(0.4)
    from core.state import clear_queue
    clear_queue(GID)                                   # a typed `clear`…
    rig.cog._cancel_plays(GID, reason="clear")         # …calls voice lookups off
    gate.set()
    await rig.settle(1.0)
    assert not st.queue                                # the hymns never went in after the clear
    assert any("Didn't finish your request — the queue was cleared" in r for r in replies(rig))


@planned
async def test_a_clear_landing_while_a_step_runs_stops_the_rest(rig):
    """A control-only plan isn't in the song lane, so nothing calls it off:
    the queue epoch does — even when the clear lands *during* a step (here
    the shuffle), not only between steps."""
    from core.state import clear_queue
    with_catalogue(rig)
    st = playing(rig, queued=("A", "B", "C"))
    real = rig.cog._execute_intent

    async def shuffle_then_someone_clears(guild, member, vc, intent, **kw):
        out = await real(guild, member, vc, intent, **kw)
        if intent.action == "shuffle":
            clear_queue(GID)                               # a typed clear, mid-step
            st.queue.extend({"title": t} for t in ("X", "Y"))   # …and new songs after it
        return out
    rig.cog._execute_intent = shuffle_then_someone_clears
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "shuffle"), step(2, "remove", target="pos:1")]), ops=OPS))
    rig.say("hey dj shuffle and then take the next song off")
    await rig.settle(1.0)
    assert [e["title"] for e in st.queue] == ["X", "Y"]          # nothing of theirs removed
    run = rig.cog._last_run[GID]
    assert run.results[2].status == "not_done" and "cleared" in run.stopped


@planned
async def test_the_same_request_twice_is_carried_out_once(rig):
    with_catalogue(rig)
    st = playing(rig)
    plan = vp.parse_plan(plan_json([step(1, "add_song", title="The Phoenix",
                                         artist="Fall Out Boy", count=2)]), ops=OPS)
    import time
    now = time.monotonic()
    job = vd.Job(GID, UID, None, now - 1.2, now - 0.2, 900, session=0, epoch=0,
                 speech_start=now - 1.0)
    rig.cog._enqueue_plan(job, plan, "hey dj play the phoenix twice")
    rig.cog._enqueue_plan(job, plan, "hey dj play the phoenix twice")      # a retry
    await rig.settle(1.0)
    assert len(st.queue) == 2                         # its two copies — not four
    assert rig.cog.metrics.guild(GID).drops["duplicate_request"] == 1


@planned
async def test_queue_questions_are_answered_from_the_live_queue(rig):
    with_catalogue(rig)
    playing(rig, "Fall Out Boy - Centuries", queued=("Panic! At The Disco - High Hopes",
                                                     "The Killers - Mr. Brightside"))
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([step(1, "now_playing"),
                                                   step(2, "queue_info")]), ops=OPS))
    rig.say("hey dj what's this song and what's coming up")
    await rig.settle(0.5)
    (said,) = rig.tts.said
    assert said.startswith("This is Fall Out Boy - Centuries")
    assert "2 songs up next: Panic! At The Disco - High Hopes, The Killers - Mr. Brightside" in said


@planned
async def test_mixed_question_and_action(rig):
    with_catalogue(rig)
    st = playing(rig, "Song One", queued=("Song Two",))
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([step(1, "now_playing"), step(2, "skip")]),
                                        ops=OPS))
    rig.say("hey dj what song is this, and skip it after")
    await rig.settle(0.5)
    assert rig.tts.said == ["This is Song One."] and rig.vc.stopped == 1
    assert st.current is None or st.current["title"] != "Song One"


@planned
async def test_remove_by_name_with_an_imperfect_transcript(rig):
    with_catalogue(rig)
    st = playing(rig, "Demi The Daredevil - AMERICAN ZOMBIE",
                 queued=("Point North - Below The Belt", "Skrillex - The Devil's Den",
                         "WAX - ALTE WUNDEN", "Saint Chaos - Ghosts & Monsters"))
    wax = st.queue.ids()[2]
    rig.cog.llm = FakeLLM(vp.parse_plan(plan_json([
        step(1, "remove", target="match:Wax by Alte One Den")]), ops=OPS))
    rig.say("Hey DJ, can you remove that song called Wax by Alte One Den from the queue?")
    await rig.settle(0.5)
    assert wax not in st.queue.ids() and len(st.queue) == 3 and rig.vc.stopped == 0
    assert st.current["title"] == "Demi The Daredevil - AMERICAN ZOMBIE"


@planned
async def test_move_the_one_i_just_added_binds_this_speakers_newest_copy(rig):
    with_catalogue(rig)
    st = playing(rig, queued=("X",))
    rig.cog.llm = FakeLLM(
        vp.parse_plan(plan_json([step(1, "add_song", title="The Phoenix", artist="Fall Out Boy",
                                      count=2)]), ops=OPS),
        vp.parse_plan(plan_json([step(1, "move", target="mine:last", to="front")]), ops=OPS))
    rig.say("hey dj play the phoenix by fall out boy twice")
    await rig.settle(0.5)
    first, second = rig.cog._last_run[GID].results[1].added
    rig.say("hey dj move the one I just added to the front")
    await rig.settle(0.5)
    assert st.queue.ids()[0] == second
    # …and the planner was told what they added, with ids
    assert "added [" in rig.cog.llm.contexts[1] and second in rig.cog.llm.contexts[1]


@planned
async def test_other_speakers_additions_are_not_mine(rig):
    with_catalogue(rig)
    st = playing(rig)
    rig.cog.llm = FakeLLM(
        vp.parse_plan(plan_json([step(1, "add_song", title="The Phoenix", artist="Fall Out Boy")]),
                      ops=OPS),
        vp.parse_plan(plan_json([step(1, "add_song", title="I Write Hymns Not Travesties",
                                      artist="Yung Gravy")]), ops=OPS),
        vp.parse_plan(plan_json([step(1, "remove", target="mine:last")]), ops=OPS))
    rig.say("hey dj play the phoenix by fall out boy")
    await rig.settle(0.5)
    mine = st.queue.ids()[-1]
    rig.say("hey dj play i write hymns not travesties by yung gravy", uid=OTHER)
    await rig.settle(0.5)
    rig.say("hey dj remove the one I added")
    await rig.settle(0.5)
    assert mine not in st.queue.ids() and len(st.queue) == 1


# ── the plan itself ───────────────────────────────────────────────────────────

def test_unknown_operations_and_broken_references_are_refused_out_loud():
    p = vp.parse_plan(plan_json([
        step(1, "add_song", title="X"),
        step(2, "teleport"),
        step(3, "move", target="step:9", to="front"),
        step(4, "remove", target="step:2"),
        step(5, "move", target="step:1", to="front")]), ops=OPS)
    assert [s.id for s in p.steps] == [1, 5]
    assert any("teleport" in x for x in p.problems)
    assert any("step 9 that isn't there" in x for x in p.problems)
    assert any("step 2" in x for x in p.problems)


def test_limits_are_reported_never_silently_cut():
    many = [step(i, "add_song", title=f"S{i}") for i in range(1, 46)]
    p = vp.parse_plan(plan_json(many), ops=OPS)
    assert not p.steps and "at most" in p.problems[0]
    p = vp.parse_plan(plan_json([step(1, "add_song", title="X", count=99)]), ops=OPS)
    assert not p.steps and "at most" in p.problems[0]


def test_dependency_cycles_are_refused():
    p = vp.parse_plan(plan_json([step(1, "shuffle", after=[2]), step(2, "skip", after=[1])]),
                      ops=OPS)
    assert not p.steps and "circle" in p.problems[-1]


# ── the real boundary: HTTP answer → LLMRouter.plan → the cog → the queue ──

@planned
async def test_central_request_through_the_real_router_and_an_http_answer(rig):
    """The same request with the real planner client: the model's answer comes
    over HTTP from a stand-in for OpenRouter, is parsed and checked by
    LLMRouter.plan, and runs through the cog. Also checks what the model was
    shown: the queue with its entry ids."""
    from core import voice_llm as vl
    from tests.test_voice_planner import FakePlanner
    with_catalogue(rig)
    st = playing(rig, queued=("Queued A", "Queued B"))
    before_current, before_ids = qid(st.current), st.queue.ids()
    async with FakePlanner() as srv:
        srv.queued = [(plan_json(CENTRAL_STEPS), "stop")]
        router = vl.LLMRouter("sk-or-v1-" + "ab" * 32, model="google/x", base_url=srv.url)
        rig.cog.llm = router
        rig.cog._rng = random.Random(7)
        try:
            rig.say(CENTRAL)
            await rig.settle(2.5)
        finally:
            await router.close()
    run = rig.cog._last_run[GID]
    check_central(rig, run, before_current=before_current, before_ids=before_ids)
    (body,) = srv.requests
    user = body["messages"][1]["content"]
    assert f"Now playing: [{before_current}] Already Playing" in user
    assert f"1. [{before_ids[0]}] Queued A" in user and f"2. [{before_ids[1]}] Queued B" in user
    assert user.endswith("They said: " + CENTRAL.split(",", 1)[1].strip().rstrip("."))
