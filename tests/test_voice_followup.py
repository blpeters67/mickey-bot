"""Regressions for the follow-up voice review (cogs/voicedj.py, core/voice_*).

The real VoiceDJ cog, job queue, ASR runner, grammar and front end; fake
Discord objects and a scripted "whisper" (see tests/test_voicedj_flow.py).
These are MOCKED tests: they prove the decision logic and its timing, not
recognition accuracy — a stand-in transcriber says exactly what it's told.
"""
from __future__ import annotations

import asyncio
import time
import types

import pytest

np = pytest.importorskip("numpy")

import cogs.voicedj as vd                                   # noqa: E402
from core.state import get_state, reset_state               # noqa: E402
from core.voice_frontend import (                           # noqa: E402
    VAD_WINDOW_S, Segment, Segmenter, SegmenterConfig, VoiceFrontEnd,
)
from core.voice_intent import Intent                        # noqa: E402
from core.voice_pipeline import Job                         # noqa: E402
from tests.test_voicedj_flow import GID, OTHER, UID, Rig, scenario   # noqa: E402


def segment(rig, text, *, start, speech_start=None, speech_ms=900, tag=None,
            uid=UID, gid=GID, length=0.9):
    """A Segment whose scripted transcript is `text`."""
    rig._marker += 1
    rig.transcripts[rig._marker] = text
    audio = np.full(16000, rig._marker, dtype=np.float32)
    return Segment(gid, uid, audio, start, start + length, speech_ms, "silence", 0,
                   speech_start=speech_start, tag=tag)


def gated_lookup(monkeypatch):
    """Patch the song search with one that waits for `release`. (The music
    catalogue is unreachable in tests, so a voice "play" is a checked YouTube
    search: this one answers with an upload titled like the query.)"""
    state = types.SimpleNamespace(started=asyncio.Event(), release=asyncio.Event(), calls=[])

    async def search(query, *, limit=5):
        state.calls.append(query)
        state.started.set()
        await state.release.wait()
        return [{"title": query.title(), "uploader": "", "stream_url": "s://x",
                 "webpage_url": "https://e/x", "duration": 100, "_needs_resolve": False,
                 "_resolved_at": time.time()}]
    monkeypatch.setattr(vd.audio, "search_tracks", search)
    return state


def stages(rig, action):
    return dict(rig.cog.metrics.guild(GID).actions.get(action, {}))


# ── identity is fixed at admission (finding 4) ───────────────────────────────

@scenario
async def test_segment_emitted_just_before_a_session_change_is_dropped(rig):
    session = rig.cog._session.get(GID, 0)
    seg = segment(rig, "hey dj skip", start=time.monotonic() - 1,
                  tag=(session, rig.cog._epoch.get((GID, UID), 0)))
    rig.cog._segment_from_thread(seg)                # front-end thread → loop…
    rig.cog._new_session(GID, "moved before delivery")   # …but this runs first
    await rig.settle()
    assert rig.executed == [] and len(rig.cog._jobs) == 0
    assert rig.drops()["session_changed"] == 1


@scenario
async def test_segment_from_before_an_opt_out_is_dropped(rig):
    seg = segment(rig, "hey dj skip", start=time.monotonic() - 1,
                  tag=(rig.cog._session.get(GID, 0), rig.cog._epoch.get((GID, UID), 0)))
    rig.cog._revoke(GID, UID, "revoked")             # opted out and straight back in
    rig.cog._on_segment(seg)
    await rig.settle()
    assert rig.executed == [] and rig.drops()["revoked"] == 1


def test_frame_admission_stamps_session_and_consent():
    fed = []
    cog = vd.VoiceDJ(types.SimpleNamespace(loop=None))
    cog._front = types.SimpleNamespace(feed=lambda *a, **kw: fed.append(kw["tag"]))
    cog._auth[GID] = frozenset({UID})
    cog._session[GID], cog._epoch[(GID, UID)] = 3, 5
    cog._on_frame(GID, UID, b"x", 1, 960, 10, 0.0)
    cog._asr.shutdown()
    cog._models.shutdown()
    assert fed == [(3, 5)]


class _Dec:
    def decode(self, opus, fec=False):
        return b"\x00\x10" * 1920              # loud-ish 20 ms of PCM


class _Vad:
    def window(self, w):
        return 0.9                              # everything is speech


def test_first_queued_packet_is_invalidated_by_drop_guild():
    """A packet for a speaker the worker hasn't seen yet sits only in the
    queue; drop_guild must still void it (guild generation)."""
    front = VoiceFrontEnd(lambda s: None, decoder_factory=_Dec, vad_factory=_Vad)
    front.feed(GID, UID, b"x", 1, 960, 10, 1.0, tag=(0, 0))
    front.drop_guild(GID)
    key, epoch, ggen, *_ = front._q.get_nowait()
    assert not front._valid(key, epoch, ggen)


def test_drop_guild_voids_queued_first_packets_end_to_end():
    got = []
    front = VoiceFrontEnd(got.append, decoder_factory=_Dec, vad_factory=_Vad)
    for i in range(40):                         # 0.8 s of "speech", queued first…
        front.feed(GID, UID, b"x", i, i * 960, 10, 1.0 + i * 0.02, tag=(0, 0))
    front.drop_guild(GID)                       # …then the guild is reset
    front.start()
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not front._q.empty():
            time.sleep(0.02)
        time.sleep(0.8)                         # let any timeout-ended segment out
    finally:
        front.stop()
    assert got == [] and not front.stats(GID)


def test_segments_carry_the_tag_of_their_packets():
    got = []
    front = VoiceFrontEnd(got.append, decoder_factory=_Dec, vad_factory=_Vad)
    front.start()
    try:
        t0 = time.monotonic()
        for i in range(30):
            front.feed(GID, UID, b"x", i, i * 960, 10, t0 + i * 0.02, tag=(7, 2))
        deadline = time.monotonic() + 3
        while not got and time.monotonic() < deadline:   # ended by the DTX timeout
            time.sleep(0.02)
    finally:
        front.stop()
    assert got and got[0].tag == (7, 2)
    assert got[0].speech_start is not None and got[0].speech_start >= got[0].start


def test_a_new_tag_never_splices_into_an_old_stream():
    front = VoiceFrontEnd(lambda s: None, decoder_factory=_Dec, vad_factory=_Vad)
    old = front._stream_for((GID, UID), (0, 0))
    assert front._stream_for((GID, UID), (0, 0)) is old
    assert front._stream_for((GID, UID), (1, 0)) is not old


# ── onset vs padded start; reply timing (findings 5, 6) ──────────────────────

def test_segmenter_reports_onset_separately_from_preroll():
    out = []
    seg = Segmenter(SegmenterConfig(), lambda *a: out.append(a))
    w = np.zeros(512, dtype=np.float32)
    t = 100.0
    for _ in range(10):
        t += VAD_WINDOW_S
        seg.push(w, 0, t)
    prompt_at = t
    for _ in range(3):
        t += VAD_WINDOW_S
        seg.push(w, 0, t)
    for _ in range(6):
        t += VAD_WINDOW_S
        seg.push(w, 1, t)
    for _ in range(25):
        t += VAD_WINDOW_S
        seg.push(w, 0, t)
    _audio, start, _end, _ms, _reason, _ov, onset = out[0]
    assert start < prompt_at < onset              # pre-roll reaches back; speech didn't
    job = Job(GID, UID, None, start, _end, 192, session=0, epoch=0, speech_start=onset)
    pending = vd._Pending(Intent("stop"), asked_at=prompt_at, deadline=prompt_at + 12, session=0)
    assert pending.accepts(job)


@scenario
async def test_quick_yes_with_preroll_before_the_prompt_confirms(rig):
    rig.say("hey dj stop")
    await rig.settle()
    asked = rig.cog._pending[(GID, UID)].asked_at
    # Speech began 0.1 s after the question; the clip's pre-roll starts 0.2 s before it.
    rig.cog._on_segment(segment(rig, "yes", start=asked - 0.2, speech_start=asked + 0.1,
                                speech_ms=300, length=0.6))
    await rig.settle()
    assert rig.executed == [(GID, UID, "stop", None)]


@scenario
async def test_reply_window_starts_after_the_question_is_delivered(rig):
    async def slow_feedback(guild, member, heard, result, *, force=False, chime=None):
        await asyncio.sleep(0.3)                 # a slow send
        rig.feedback.append((guild.id, member.id, result, force))
    rig.cog._deliver_feedback = slow_feedback
    rig.say("hey dj clear the queue")
    await rig.settle(0.8)
    p = rig.cog._pending[(GID, UID)]
    assert p.deadline - p.asked_at >= vd._CONFIRM_SECONDS + 0.25


@scenario
async def test_timely_yes_survives_the_capture_deadline_while_asr_is_slow(rig):
    rig.say("hey dj stop")
    await rig.settle()
    p = rig.cog._pending[(GID, UID)]
    rig.slow_asr = 0.6                           # a slow recogniser
    now = time.monotonic()
    rig.cog._on_segment(segment(rig, "yes", start=now - 0.1, speech_start=now, speech_ms=300,
                                length=0.4))
    p.deadline = now + 0.05                      # its capture deadline passes…
    await asyncio.sleep(0.2)
    rig.cog._last_refresh = time.monotonic()
    await rig.cog._maintenance.coro(rig.cog)     # …and maintenance runs mid-recognition
    assert rig.cog._pending.get((GID, UID)) is p
    await rig.settle(1.5)
    assert rig.executed == [(GID, UID, "stop", None)]
    assert not any("No answer" in (f[2] or "") for f in rig.feedback)


@scenario
async def test_speech_that_starts_after_the_deadline_is_not_an_answer(rig):
    rig.say("hey dj stop")
    await rig.settle()
    p = rig.cog._pending[(GID, UID)]
    p.deadline = time.monotonic() - 0.5
    rig.say("yes", start=time.monotonic(), speech_ms=300)
    await rig.settle()
    assert rig.executed == []


@scenario
async def test_unanswered_question_still_expires_with_a_notice(rig):
    rig.say("hey dj stop")
    await rig.settle()
    rig.cog._pending[(GID, UID)].deadline = time.monotonic() - 2
    rig.cog._last_refresh = time.monotonic()
    await rig.cog._maintenance.coro(rig.cog)
    assert (GID, UID) not in rig.cog._pending
    assert rig.feedback[-1][2] == "⌛ No answer — cancelled."


@scenario
async def test_armed_window_follow_up_survives_slow_recognition(rig):
    rig.slow_asr = 0.4
    t = time.monotonic()
    rig.say("hey dj", start=t - 1.2, speech_ms=500)               # wake only
    # "skip" — short, said 0.6 s later, while "hey dj" is still being recognised.
    rig.cog._on_segment(segment(rig, "skip", start=t - 0.4, speech_start=t - 0.3,
                                speech_ms=250, length=0.3))
    await rig.settle(2.0)
    assert rig.executed == [(GID, UID, "skip", None)]
    assert "too_short" not in rig.drops()


@scenario
async def test_armed_window_is_kept_for_a_timely_command_in_flight(rig):
    rig.say("hey dj", speech_ms=500)
    await rig.settle()
    armed = rig.cog._armed[(GID, UID)]
    rig.slow_asr = 0.6
    now = time.monotonic()
    rig.cog._on_segment(segment(rig, "pause", start=now - 0.1, speech_start=now,
                                speech_ms=400, length=0.4))
    armed.until = now + 0.05
    await asyncio.sleep(0.2)
    rig.cog._last_refresh = time.monotonic()
    await rig.cog._maintenance.coro(rig.cog)
    assert rig.cog._armed.get((GID, UID)) is armed
    await rig.settle(1.5)
    assert rig.executed == [(GID, UID, "pause", None)]


# ── lookups vs. stop / opt-out / session (findings 3, 8) ─────────────────────

def _lookup_scenario(fn):
    def wrapper(monkeypatch):
        async def go():
            rig = Rig(monkeypatch)
            look = gated_lookup(monkeypatch)
            try:
                await fn(rig, look)
            finally:
                look.release.set()
                await rig.close()
        asyncio.run(go())
    wrapper.__name__ = fn.__name__
    return wrapper


@_lookup_scenario
async def test_opting_out_during_a_voice_lookup_queues_nothing(rig, look):
    rig.say("hey dj play late song")
    await asyncio.wait_for(look.started.wait(), 3)
    rig.cog._auth[GID] = frozenset({OTHER})
    rig.cog._revoke(GID, UID, "revoked")
    look.release.set()
    await rig.settle()
    assert not get_state(GID).queue
    assert not any("Queued" in (f[2] or "") for f in rig.feedback)     # silent: they opted out


@_lookup_scenario
async def test_typed_stop_during_a_voice_lookup_queues_nothing(rig, look):
    rig.say("hey dj play late song")
    await asyncio.wait_for(look.started.wait(), 3)
    reset_state(GID)                                   # someone typed `stop`
    look.release.set()
    await rig.settle()
    assert not get_state(GID).queue
    assert "stopped while I was searching" in rig.feedback[-1][2]


@_lookup_scenario
async def test_bot_moving_during_a_voice_lookup_queues_nothing(rig, look):
    rig.say("hey dj play late song")
    await asyncio.wait_for(look.started.wait(), 3)
    rig.cog._new_session(GID, "rejoin")
    look.release.set()
    await rig.settle()
    assert not get_state(GID).queue
    assert stages(rig, "play").get("cancelled") == 1


@_lookup_scenario
async def test_voice_stop_cancels_the_lookup_heard_before_it(rig, look):
    rig.say("hey dj play late song")
    await asyncio.wait_for(look.started.wait(), 3)
    rig.say("hey dj stop", start=time.monotonic() - 0.5)
    await rig.settle()
    rig.say("yes", start=time.monotonic(), speech_ms=300)
    await rig.settle()
    assert (GID, UID, "stop", None) in rig.executed
    look.release.set()
    await rig.settle()
    assert not get_state(GID).queue
    assert any("Didn't queue" in (f[2] or "") and "stopped" in f[2] for f in rig.feedback)


@_lookup_scenario
async def test_a_play_requested_after_a_stop_still_queues(rig, look):
    look.release.set()
    rig.say("hey dj stop")
    await rig.settle()
    rig.say("yes", start=time.monotonic(), speech_ms=300)
    await rig.settle()
    rig.say("hey dj play next one", start=time.monotonic())
    await rig.settle()
    assert [t["title"] for t in get_state(GID).queue] == ["Next One"]


@_lookup_scenario
async def test_controls_are_not_stuck_behind_a_slow_lookup(rig, look):
    rig.say("hey dj play something obscure")
    await asyncio.wait_for(look.started.wait(), 3)
    t0 = time.monotonic()
    rig.say("hey dj skip", start=t0 - 0.9)
    while not any(e[2] == "skip" for e in rig.executed):
        await asyncio.sleep(0.02)
        assert time.monotonic() - t0 < 1.0, "skip waited for the song search"
    assert not look.release.is_set()             # the search really was still running
    look.release.set()
    await rig.settle()
    assert [t["title"] for t in get_state(GID).queue] == ["Something Obscure"]


@scenario
async def test_stale_command_is_reported_not_silently_dropped(rig):
    job = Job(GID, UID, None, time.monotonic() - 30, time.monotonic() - 29, 500,
              session=rig.cog._session.get(GID, 0), epoch=rig.cog._epoch.get((GID, UID), 0))
    rig.cog._enqueue_action(job, Intent("skip"), "hey dj skip")
    await rig.settle()
    assert rig.executed == []
    assert "too late" in rig.feedback[-1][2]
    assert stages(rig, "skip") == {"stale": 1}


# ── quoted / negated speech (finding 7) ──────────────────────────────────────

@pytest.mark.parametrize("text", ["do not say hey dj skip", "I said hey dj skip",
                                  "so I told him, hey DJ stop"])
def test_quoted_or_negated_commands_never_execute(monkeypatch, text):
    async def body(rig):
        rig.say(text)
        await rig.settle()
        assert rig.executed == [] and (GID, UID) not in rig.cog._pending
        assert rig.cog.metrics.guild(GID).parse["wake_not_at_start"] == 1
    scenario(body)(monkeypatch)


# ── outcome metrics (finding 12) ─────────────────────────────────────────────

def test_a_failed_play_is_not_counted_as_done(monkeypatch):
    async def body(rig):
        async def nothing(query, *, limit=5):
            return []
        monkeypatch.setattr(vd.audio, "search_tracks", nothing)
        rig.say("hey dj play zzzz nothing")
        await rig.settle()
        st = stages(rig, "play")
        assert st.get("recognized") == 1 and st.get("admitted") == 1
        assert st.get("failed") == 1 and "queued" not in st and "completed" not in st
        assert "couldn't find" in rig.feedback[-1][2].lower()
        assert rig.cog.metrics.guild(GID).counts.get("did_play") is None
    scenario(body)(monkeypatch)


@scenario
async def test_asr_quality_is_kept_with_the_outcome(rig):
    rig.say("hey dj skip")
    rig.say("we should get food", start=time.monotonic() - 0.5)
    await rig.settle()
    q = list(rig.cog.metrics.guild(GID).asr_quality)
    assert (-0.2, 0.01) == q[0][:2] and {r[3] for r in q} == {"→ skip", "no_wake"}
    s = rig.cog.metrics.summary(GID)
    assert s["quality_commands"][2] == 1 and s["quality_other"][2] == 1


@scenario
async def test_voice_requested_song_reports_when_it_actually_starts(rig):
    track = {"title": "T", "_voice_job_end": time.monotonic() - 2.0}
    guild = rig.guilds[GID]
    await rig.cog.on_track_audible(guild, track)
    await rig.cog.on_track_audible(guild, track)          # a loop lap: counted once
    other = {"title": "U", "_voice_job_end": time.monotonic()}
    await rig.cog.on_track_failed(guild, other, "forbidden")
    st = stages(rig, "play")
    assert st == {"started": 1, "playback_failed": 1}
    assert 1.9 < rig.cog.metrics.guild(GID).audible_seconds[0] < 3


# ── diagnostics stay under Discord's limit (finding 10) ─────────────────────

@scenario
async def test_long_populated_diag_fits_discord(rig):
    cog, guild = rig.cog, rig.guilds[GID]
    many = {(GID, 1000 + i): {"packets": 99999, "concealed": 999, "decode_errors": 9,
                              "speech_windows": 9999, "vad_windows": 99999, "segments": 999,
                              "dropped_short": 99} for i in range(25)}
    cog._front = types.SimpleNamespace(stats=lambda gid: many)
    for i in range(12):
        cog.metrics.event(GID, UID, "skip")
        cog.metrics.drop(GID, f"reason_number_{i}_with_a_long_name")
        cog.metrics.guild(GID).parse[f"parse_reason_{i}"] += 1
        cog.metrics.stage(GID, f"action{i}", "recognized")
    cog._debug_until[GID] = time.monotonic() + 600
    cog._debug_log[GID] = [(f"user{i}", "a very long transcript " * 12, "no_wake")
                           for i in range(20)]
    sent = []

    async def send(content=None, **kw):
        sent.append((content, kw.get("file")))
    ctx = types.SimpleNamespace(guild=guild, send=send, prefix="-")
    guild.voice_client.is_connected = lambda: True
    await vd.VoiceDJ.voice_diag.callback(cog, ctx)
    content, file = sent[0]
    full = cog._diag_text(guild)
    assert len(full) > 2000                         # the worst case really is long
    assert len(content) <= 2000 and file is not None
    attached = file.fp.read().decode("utf-8")
    assert attached == full


@scenario
async def test_short_diag_is_plain_content(rig):
    sent = []

    async def send(content=None, **kw):
        sent.append((content, kw.get("file")))
    ctx = types.SimpleNamespace(guild=rig.guilds[GID], send=send, prefix="-")
    await vd.VoiceDJ.voice_diag.callback(rig.cog, ctx)
    assert sent and len(sent[0][0]) <= 2000 and sent[0][1] is None


def test_a_song_request_waiting_behind_another_lookup_is_not_stale(monkeypatch):
    async def body(rig):
        monkeypatch.setattr(vd, "_DISPATCH_MAX_AGE", 0.6)

        async def search(query, *, limit=5):
            if query == "slow one":
                await asyncio.sleep(0.9)            # longer than the age limit
            return [{"title": query.title(), "uploader": "", "stream_url": "s://x",
                     "webpage_url": "https://e/x", "duration": 100, "_needs_resolve": False,
                     "_resolved_at": time.time()}]
        monkeypatch.setattr(vd.audio, "search_tracks", search)
        now = time.monotonic()
        rig.say("hey dj play slow one", start=now - 0.95)
        rig.say("hey dj play second one", start=now - 0.9)
        await rig.settle(2.5)
        assert [t["title"] for t in get_state(GID).queue] == ["Slow One", "Second One"]
        assert stages(rig, "play").get("stale") is None
    scenario(body)(monkeypatch)
