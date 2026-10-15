"""Queue entries with identities (core/state.TrackQueue) and the queue
operations the typed commands and voice control share (core/queue_ops.py)."""
from __future__ import annotations

import copy
import random
from collections import deque

import pytest

from core import queue_ops as qo
from core.state import GuildState, TrackQueue, qid, requeue_copy


def _q(*titles):
    st = GuildState()
    st.queue.extend({"title": t} for t in titles)
    return st


# ── identities ────────────────────────────────────────────────────────────────

def test_every_entry_gets_its_own_id_and_copies_of_one_song_are_separate():
    st = _q("A", "A", "A", "A", "A")
    ids = st.queue.ids()
    assert len(set(ids)) == 5 and all(i and i.startswith("q") for i in ids)


def test_the_same_dict_queued_twice_gets_a_copy_with_its_own_id():
    st = GuildState()
    song = {"title": "X"}
    st.queue.append(song)
    st.queue.append(song)
    a, b = st.queue
    assert a is song and b is not song and qid(a) != qid(b)


def test_an_entry_that_comes_back_keeps_its_id_a_loop_copy_is_new():
    st = _q("A")
    entry = st.queue.popleft()
    st.queue.appendleft(entry)                    # loop mode / put back after a failed start
    assert qid(st.queue[0]) == qid(entry)
    st.queue.append(requeue_copy(entry))          # queue loop: a new occurrence
    assert qid(st.queue[1]) not in (None, qid(entry))


def test_rebuilding_the_queue_keeps_it_a_trackqueue_and_moves_the_revision_on():
    st = _q("A", "B")
    rev = st.queue.rev
    ids = st.queue.ids()
    st.queue = deque(reversed(list(st.queue)))
    assert isinstance(st.queue, TrackQueue) and st.queue.rev > rev
    assert st.queue.ids() == ids[::-1]
    st.queue.append({"title": "C"})               # still stamping after the rebuild
    assert qid(st.queue[-1])


def test_copying_and_pickling_keep_the_ids():
    import pickle
    st = _q("A", "B")
    assert copy.copy(st.queue).ids() == st.queue.ids()
    assert pickle.loads(pickle.dumps(st.queue)).ids() == st.queue.ids()


# ── operations ────────────────────────────────────────────────────────────────

def test_positions_and_removal_by_id():
    st = _q("A", "B", "C", "D")
    b, d = qo.ids_at(st, [2, 4])
    gone = qo.remove_ids(st, [d, b])
    assert [e["title"] for e in gone] == ["B", "D"] and [e["title"] for e in st.queue] == ["A", "C"]
    with pytest.raises(qo.QueueOpError):
        qo.ids_at(st, [3])


def test_move_counts_the_destination_without_the_entry_and_clamps():
    st = _q("A", "B", "C", "D")
    d = qo.ids_at(st, [4])[0]
    entry, frm, to = qo.move_id(st, d, 0)
    assert (entry["title"], frm, to) == ("D", 3, 0)
    assert [e["title"] for e in st.queue] == ["D", "A", "B", "C"]
    qo.move_id(st, d, 99)
    assert [e["title"] for e in st.queue][-1] == "D"
    with pytest.raises(qo.QueueOpError):
        qo.move_id(st, "q-nope", 0)


def test_swap_skipto_reverse_balance_dedupe():
    st = _q("A", "B", "C")
    a, c = qo.ids_at(st, [1, 3])
    qo.swap_ids(st, a, c)
    assert [e["title"] for e in st.queue] == ["C", "B", "A"]
    dropped = qo.drop_before(st, a)
    assert [e["title"] for e in dropped] == ["C", "B"] and [e["title"] for e in st.queue] == ["A"]
    st = _q("A", "B")
    qo.reverse(st)
    assert [e["title"] for e in st.queue] == ["B", "A"]
    st = GuildState()
    st.queue.extend({"title": t, "requester": r} for t, r in
                    (("1", "x"), ("2", "x"), ("3", "x"), ("4", "y")))
    qo.balance(st)
    assert [e["requester"] for e in st.queue] == ["x", "y", "x", "x"]
    st = _q("A", "B", "A")
    st.current = {"title": "B"}
    assert [e["title"] for e in qo.dedupe(st)] == ["B", "A"] and [e["title"] for e in st.queue] == ["A"]


def test_shuffle_is_seedable_and_keeps_the_entries():
    st = _q(*"ABCDEFG")
    ids = st.queue.ids()
    qo.shuffle(st, random.Random(4))
    assert sorted(st.queue.ids()) == sorted(ids) and st.queue.ids() != ids
    other = _q(*"ABCDEFG")
    qo.shuffle(other, random.Random(4))
    assert [e["title"] for e in other.queue] == [e["title"] for e in st.queue]


def test_insert_returns_the_new_ids_where_they_went():
    st = _q("A", "B")
    ids = qo.insert_entries(st, [{"title": "X"}, {"title": "Y"}], at=1)
    assert [e["title"] for e in st.queue] == ["A", "X", "Y", "B"] and ids == st.queue.ids()[1:3]
    end = qo.insert_entries(st, [{"title": "Z"}])
    assert end == st.queue.ids()[-1:]


def test_loop_and_volume_helpers():
    st = GuildState()
    assert qo.set_loop(st, "queue") == ("off", "queue") and st.loop_queue and not st.loop
    assert qo.set_loop(st, "song") == ("queue", "song") and st.loop
    with pytest.raises(qo.QueueOpError):
        qo.set_loop(st, "sideways")
    vc = type("VC", (), {"source": type("S", (), {"volume": 1.0})()})()
    assert qo.set_volume(st, vc, 1500) == 1000 and vc.source.volume == 10.0


def test_skip_current_drops_it_from_a_song_loop_and_keeps_it_in_a_queue_loop():
    class VC:
        def __init__(self):
            self.stopped = 0

        def is_playing(self):
            return True

        def is_paused(self):
            return False

        def stop(self):
            self.stopped += 1
    st = _q("Next")
    st.current = {"title": "Now", "_qid": "q-now"}
    st.loop_queue = True
    vc = VC()
    assert qo.skip_current(st, vc, None, None) and vc.stopped == 1
    assert st.current is None and [e["title"] for e in st.queue] == ["Next", "Now"]
