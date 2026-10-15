"""Tests for core/voice_pipeline.py — the job queue, the one-at-a-time ASR
runner and the shared model loader. Driven with asyncio.run, no Discord."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.voice_pipeline import (PRIORITY_ARMED, PRIORITY_CONFIRM, PRIORITY_NORMAL,
                                 AsrBusy, AsrRunner, AsrTimeout, Job, JobQueue,
                                 ModelManager, VoiceMetrics)


def run(coro):
    return asyncio.run(coro)


def job(uid=1, end=0.0, priority=PRIORITY_NORMAL, gid=1):
    return Job(gid, uid, None, end - 1, end, 900, session=0, epoch=0, priority=priority)


# ── JobQueue ──────────────────────────────────────────────────────────────────

def test_queue_order():
    async def go():
        q = JobQueue(per_speaker=5, total=10)
        a, b = job(1, 5.0), job(2, 3.0)
        c, d = job(3, 9.0, PRIORITY_CONFIRM), job(4, 8.0, PRIORITY_ARMED)
        for j in (a, b, c, d):
            q.put(j)
        got = [await q.get() for _ in range(4)]
        return got, (c, d, b, a)
    got, expected = run(go())
    assert got == list(expected)


def test_per_speaker_cap_supersedes_the_oldest():
    drops = []
    q = JobQueue(per_speaker=2, total=10, on_drop=lambda j, r: drops.append((j.end, r)))
    for end in (1.0, 2.0, 3.0):
        q.put(job(7, end))
    assert len(q) == 2 and drops == [(1.0, "superseded")]


def test_a_confirmation_is_never_evicted_by_chatter():
    drops = []
    q = JobQueue(per_speaker=1, total=10, on_drop=lambda j, r: drops.append(r))
    q.put(job(7, 1.0, PRIORITY_CONFIRM))
    q.put(job(7, 2.0))                     # lower priority: refused, not swapped in
    assert drops == ["speaker_backlog"] and len(q) == 1


def test_global_cap_drops_lowest_priority_oldest_and_counts_it():
    drops = []
    q = JobQueue(per_speaker=5, total=3, on_drop=lambda j, r: drops.append((j.user_id, r)))
    q.put(job(1, 1.0))
    q.put(job(2, 2.0))
    q.put(job(3, 3.0, PRIORITY_CONFIRM))
    q.put(job(4, 4.0))
    assert (1, "queue_full") in drops and len(q) == 3


def _part(utt, part, end, priority=None, uid=7):
    j = job(uid, end, PRIORITY_NORMAL if priority is None else priority)
    j.utt, j.part = utt, part
    return j


def test_the_parts_of_one_long_utterance_come_out_in_order_and_dont_push_each_other_out():
    async def go():
        drops = []
        q = JobQueue(per_speaker=2, total=10, on_drop=lambda j, r: drops.append((j.part, r)))
        first = _part(3, 0, 10.0)                              # judged as ordinary speech…
        q.put(first)
        rest = [_part(3, n, 10.0 + 10 * n, PRIORITY_ARMED) for n in (1, 2, 3)]
        for j in rest:
            q.put(j)
        assert drops == [] and first.priority == PRIORITY_ARMED   # …until its rest arrives
        other = job(8, 5.0)                                     # someone else, said earlier
        q.put(other)
        got = [await q.get() for _ in range(5)]
        return got, [first, *rest, other]
    got, want = run(go())
    assert got == want


def test_a_new_utterance_still_supersedes_the_same_persons_old_ones():
    drops = []
    q = JobQueue(per_speaker=2, total=10, on_drop=lambda j, r: drops.append((j.utt, r)))
    q.put(_part(1, 0, 1.0))
    q.put(_part(2, 0, 2.0))
    q.put(_part(3, 0, 3.0))
    assert drops == [(1, "superseded")]


def test_purge_by_speaker():
    drops = []
    q = JobQueue(on_drop=lambda j, r: drops.append(r))
    q.put(job(1, 1.0))
    q.put(job(2, 2.0))
    assert q.purge(lambda j: j.user_id == 1, "revoked") == 1
    assert drops == ["revoked"] and len(q) == 1


# ── AsrRunner ─────────────────────────────────────────────────────────────────

def test_asr_runs_on_its_own_thread():
    async def go():
        r = AsrRunner(timeout=2)
        name = await r.run(lambda: threading.current_thread().name)
        r.shutdown()
        return name
    assert run(go()).startswith("voice-asr")


def test_timed_out_inference_is_not_stacked_on():
    """The review's finding: wait_for(to_thread(...)) abandoned a running call
    and started the next one beside it. Now the next one is refused until the
    first finishes."""
    release = threading.Event()
    concurrent = []
    lock = threading.Lock()
    active = [0]

    def slow():
        with lock:
            active[0] += 1
            concurrent.append(active[0])
        release.wait(5)
        with lock:
            active[0] -= 1
        return "slow"

    async def go():
        r = AsrRunner(timeout=0.1, busy_wait=0.05)
        with pytest.raises(AsrTimeout):
            await r.run(slow)
        with pytest.raises(AsrBusy):
            await r.run(lambda: "next")
        assert r.stuck_for >= 0
        release.set()
        await asyncio.sleep(0.2)
        out = await r.run(lambda: "after")
        r.shutdown()
        return out, r.timeouts
    out, timeouts = run(go())
    assert out == "after" and timeouts == 1
    assert max(concurrent) == 1                      # never two at once


# ── ModelManager ──────────────────────────────────────────────────────────────

def test_one_shared_load_and_the_result_is_kept():
    """The review's finding: a load timeout cancelled the waiter before the model
    was stored, so the next utterance started another load. Here two callers
    time out, the load finishes once, and the model is kept."""
    loads = []

    def loader(name):
        loads.append(name)
        time.sleep(0.3)
        return f"model:{name}"

    async def go():
        m = ModelManager(loader, "base.en")
        first = await asyncio.gather(m.get(0.05), m.get(0.05))
        await asyncio.sleep(0.5)
        later = await m.get(0.05)
        m.shutdown()
        return first, later, m.state
    first, later, state = run(go())
    assert first == [None, None]
    assert later == "model:base.en" and state == "ready"
    assert loads == ["base.en"]


def test_failed_load_backs_off_instead_of_retrying_every_utterance():
    calls = []

    def loader(name):
        calls.append(name)
        raise RuntimeError("no network")

    async def go():
        m = ModelManager(loader, "base.en", retry_after=60)
        for _ in range(4):
            assert await m.get(1.0) is None
        m.shutdown()
        return m.state, m.error
    state, error = run(go())
    assert state == "failed" and "no network" in error
    assert calls == ["base.en"]


def test_switch_is_atomic_and_serialized():
    def loader(name):
        if name == "bad":
            raise RuntimeError("nope")
        time.sleep(0.05)
        return f"model:{name}"

    async def go():
        m = ModelManager(loader, "base.en")
        await m.get(2)
        with pytest.raises(RuntimeError):
            await m.switch("bad")
        unchanged = (m.name, m.model)
        await asyncio.gather(m.switch("small.en"), m.switch("tiny.en"))
        m.shutdown()
        return unchanged, (m.name, m.model)
    unchanged, final = run(go())
    assert unchanged == ("base.en", "model:base.en")
    assert final[0] in ("small.en", "tiny.en") and final[1] == f"model:{final[0]}"


def test_warmup_runs_once_after_load():
    warmed = []

    async def go():
        m = ModelManager(lambda n: object(), "base.en", warmup=warmed.append)
        model = await m.get(2)
        m.shutdown()
        return model
    model = run(go())
    assert warmed == [model]


# ── metrics ───────────────────────────────────────────────────────────────────

def test_metrics_summary():
    m = VoiceMetrics()
    m.count(1, "segments", 3)
    m.drop(1, "too_short")
    m.guild(1).asr_seconds.extend([0.2, 0.4, 0.9])
    s = m.summary(1)
    assert s["counts"] == {"segments": 3} and s["drops"] == {"too_short": 1}
    assert s["asr_median"] == 0.4 and s["asr_p95"] == 0.9
