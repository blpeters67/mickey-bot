"""The instant wake word (core/wake_spotter.py): "hey DJ" heard within a
fraction of a second, so the cue and "typing…" come at once — and only where
the grammar would accept it, once per utterance, taken back when speech-to-
text disagrees.

Stream mechanics use a stand-in model (scores scripted from the audio); the
bundled model itself is checked on recorded clips in
test_the_bundled_model_hears_hey_dj_and_not_hey_jay (skipped without it)."""
from __future__ import annotations

import asyncio
import os
import types

import pytest

np = pytest.importorskip("numpy")

from core import voice_frontend as vf                           # noqa: E402
from core import wake_spotter as ws                             # noqa: E402


class FakeWakeModel:
    """Scores a chunk high when it's loud: "the wake word" is a loud bit."""
    threshold = 0.5

    def __init__(self):
        self.chunks = 0

    def initial_state(self):
        return np.ones((76, 32), np.float32), np.zeros((16, 96), np.float32)

    def melspec(self, samples):
        return np.full((8, 32), float(np.sqrt(np.mean(np.square(samples[-1280:])))), np.float32)

    def embed(self, mel):
        return np.full(96, mel[-1, 0], np.float32)

    def score(self, window):
        self.chunks += 1
        return 0.95 if window[-1, 0] > 3000 else 0.02


def test_one_score_per_80_ms_however_the_audio_arrives():
    s = ws.WakeStream(FakeWakeModel())
    got = []
    for n in (100, 1180, 1, 2559, 2560, 0, 1280):
        got += s.push(np.zeros(n, np.float32))
    assert len(got) == (100 + 1180 + 1 + 2559 + 2560 + 1280) // 1280
    # offsets are where each chunk ends inside the input that completed it
    s2 = ws.WakeStream(FakeWakeModel())
    assert [o for o, _ in s2.push(np.zeros(1000, np.float32))] == []
    assert [o for o, _ in s2.push(np.zeros(3000, np.float32))] == [280, 1560, 2840]


# ── the front end: where and how often it fires ───────────────────────────────

class LoudVAD:
    """Speech wherever the audio isn't silent."""

    def reset(self):
        pass

    def window(self, w):
        return 0.9 if np.max(np.abs(w)) > 0.01 else 0.0


class PCMDecoder:
    """The 'Opus' packets are raw s16le stereo 48 kHz PCM in these tests."""

    def decode(self, data, fec=False):
        return data if data else b"\0" * 3840


def _frames(seconds, level):
    n = int(seconds * 50)
    pcm = (np.full(960 * 2, level * 32767)).astype("<i2").tobytes()
    return [pcm] * n


def _stream(events, model=None):
    return vf.SpeakerStream(1, 2, vf.SegmenterConfig(), lambda seg: None,
                            decoder_factory=PCMDecoder, vad_factory=LoudVAD, tag=("s", 0),
                            wake=events.append, wake_model=model or FakeWakeModel(),
                            wake_threshold=0.5)


def _feed(stream, frames, t0, seq0=0):
    t = t0
    for k, f in enumerate(frames):
        stream.push(f, seq0 + k, (seq0 + k) * 960, 1, t)
        t += 0.02
    return t, seq0 + len(frames)


def test_the_wake_word_at_the_start_of_an_utterance_fires_with_the_audio_so_far():
    events = []
    s = _stream(events)
    t, q = _feed(s, _frames(1.0, 0.0), 100.0)              # quiet
    t, q = _feed(s, _frames(0.3, 0.3), t, q)               # "hey DJ" (loud)
    assert len(events) == 1
    ev = events[0]
    assert ev.guild_id == 1 and ev.user_id == 2 and ev.tag == ("s", 0) and ev.score >= 0.5
    assert 100.95 <= ev.onset <= 101.1                     # the utterance's start
    assert ev.onset < ev.at <= t
    assert ev.audio is not None and 0.1 * 16000 <= len(ev.audio) <= 0.8 * 16000   # pre-roll + speech
    _feed(s, _frames(1.0, 0.3), t, q)
    # It may fire again (the verifier may have turned an early one down), but
    # not more often than the refractory allows — and always for this utterance.
    gaps = [b.at - a.at for a, b in zip(events, events[1:])]
    assert all(g >= vf.WAKE_REFRACTORY_S - 1e-6 for g in gaps)
    assert {e.onset for e in events} == {ev.onset}


def test_not_deep_into_an_utterance():
    # The grammar only takes the wake word at the start: a cue in the middle of
    # someone's long sentence would always be a false alarm.
    events = []
    s = _stream(events)
    t, q = _feed(s, _frames(1.0, 0.0), 100.0)
    model = s._spotter.model
    model.score = lambda w: 0.02                           # chatter, no wake word …
    t, q = _feed(s, _frames(4.5, 0.3), t, q)
    model.score = lambda w: 0.95                           # … then "hey DJ" mid-sentence
    _feed(s, _frames(0.5, 0.3), t, q)
    assert events == []


def test_wake_events_are_dropped_with_the_speakers_consent():
    async def go():
        events = []
        front = vf.VoiceFrontEnd(lambda seg: None, on_wake=events.append,
                                 wake_model=FakeWakeModel(), decoder_factory=PCMDecoder,
                                 vad_factory=LoudVAD)
        assert front.wake_threshold == 0.5
        with front._lock:
            st = front._stream_for((1, 2), ("s", 0))
        front.drop_speaker(1, 2)                           # opted out: nothing more from them
        _feed(st, _frames(1.0, 0.0) + _frames(1.0, 0.3), 100.0)
        assert events == []
    asyncio.run(go())


def test_no_model_no_spotting():
    s = vf.SpeakerStream(1, 2, vf.SegmenterConfig(), lambda seg: None,
                         decoder_factory=PCMDecoder, vad_factory=LoudVAD,
                         wake=lambda ev: None, wake_model=None)
    assert s._spotter is None


# ── the cog: cue and typing at once, taken back on a false alarm ─────────────

import cogs.voicedj as vd                                       # noqa: E402
from tests.test_voicedj_flow import GID, OTHER, UID, Rig        # noqa: E402


class Msg:
    def __init__(self):
        self.deleted = False

    async def delete(self):
        self.deleted = True


class FakeVerifier:
    """The recogniser's second opinion, scripted."""

    def __init__(self, *heard):
        self.heard_texts = list(heard)
        self.busy = False
        self.asked = []

    async def heard(self, audio):
        self.asked.append(len(audio))
        return (self.heard_texts.pop(0) if self.heard_texts else "Hey DJ"), 0.3

    def median_seconds(self):
        return 0.3


def _wake_rig(monkeypatch):
    rig = Rig(monkeypatch)
    rig.cog.wake_model = FakeWakeModel()
    rig.cog.wake_verifier = FakeVerifier()
    rig.cues, rig.typing = [], []

    async def cue(gid, uid, onset, **_kw):
        rig.cues.append((gid, uid, onset))
        rig.cog._cue_msgs[(gid, uid)] = (onset, rig.msg)
    rig.msg = Msg()

    async def typing_start(guild, member, *, reason, limit=25.0):
        rig.typing.append((guild.id, member.id, reason))
    rig.cog._wake_cue = cue
    rig.cog._typing_start = typing_start
    return rig


def _event(onset=50.0, uid=UID, tag=None, rig=None):
    if tag is None and rig is not None:
        tag = (rig.cog._session.get(GID, 0), rig.cog._epoch.get((GID, uid), 0))
    return ws.WakeEvent(GID, uid, onset + 0.7, 0.93, onset, tag,
                        audio=np.zeros(16000, np.float32))


def _job(onset, text_rig=None, overlap_ms=0):
    return vd.Job(GID, UID, np.zeros(16000, np.float32), onset - 0.3, onset + 1.0, 900,
                  session=0, epoch=0, overlap_ms=overlap_ms, speech_start=onset)


def test_the_cue_and_typing_come_as_soon_as_the_recogniser_agrees(monkeypatch):
    async def go():
        rig = _wake_rig(monkeypatch)
        rig.cog._on_fast_wake(_event(rig=rig))
        await asyncio.sleep(0.05)
        assert rig.cog.wake_verifier.asked == [16000]      # it heard just those words
        assert rig.cues == [(GID, UID, 50.0)]
        assert rig.typing == [(GID, UID, "heard")]
        c = rig.cog.metrics.guild(GID).counts
        assert c["fast_wake_fired"] == 1 and c["fast_wake"] == 1
        await rig.close()
    asyncio.run(go())


def test_a_fire_the_recogniser_doesnt_back_is_never_shown(monkeypatch):
    # "hey BJ", "hey, did you…": the tiny model fires, the recogniser says no.
    async def go():
        rig = _wake_rig(monkeypatch)
        rig.cog.wake_verifier = FakeVerifier("Hey Jay, did you see that?")
        rig.cog._on_fast_wake(_event(rig=rig))
        await asyncio.sleep(0.05)
        assert rig.cues == [] and rig.typing == []
        assert rig.cog.metrics.guild(GID).counts["fast_wake_rejected"] == 1
        await rig.close()
    asyncio.run(go())


def test_one_cue_per_utterance_and_an_early_no_doesnt_block_a_later_yes(monkeypatch):
    async def go():
        rig = _wake_rig(monkeypatch)
        # fired on "hey D…" (too early: the recogniser hears "Hey, D"), then again
        rig.cog.wake_verifier = FakeVerifier("Hey, D", "Hey DJ")
        rig.cog._on_fast_wake(_event(onset=50.0, rig=rig))
        await asyncio.sleep(0.05)
        rig.cog._on_fast_wake(_event(onset=50.0, rig=rig))
        await asyncio.sleep(0.05)
        rig.cog._on_fast_wake(_event(onset=50.0, rig=rig))       # same utterance: done already
        await asyncio.sleep(0.05)
        assert rig.cues == [(GID, UID, 50.0)]
        assert len(rig.cog.wake_verifier.asked) == 2
        await rig.close()
    asyncio.run(go())


def test_without_its_checker_it_waits_rather_than_guess(monkeypatch):
    async def go():
        rig = _wake_rig(monkeypatch)
        rig.cog.wake_verifier = None                              # still loading
        rig.cog._on_fast_wake(_event(rig=rig))
        await asyncio.sleep(0.05)
        assert rig.cues == []
        assert rig.cog.metrics.guild(GID).counts["fast_wake_unverified"] == 1
        import dataclasses                                         # host chose: no checker
        monkeypatch.setattr(vd, "settings", dataclasses.replace(vd.settings,
                                                                voice_fast_wake_verify=False))
        rig.cog._on_fast_wake(_event(onset=60.0, rig=rig))
        await asyncio.sleep(0.05)
        assert rig.cues == [(GID, UID, 60.0)]
        await rig.close()
    asyncio.run(go())


def test_not_for_people_who_havent_opted_in_or_old_audio(monkeypatch):
    async def go():
        rig = _wake_rig(monkeypatch)
        rig.cog._auth[GID] = frozenset({OTHER})
        rig.cog._on_fast_wake(_event(rig=rig))                     # not opted in
        rig.cog._auth[GID] = frozenset({UID})
        rig.cog._on_fast_wake(_event(tag=(99, 0)))                 # captured in another session
        await asyncio.sleep(0.05)
        assert rig.cues == [] and rig.typing == []
        await rig.close()
    asyncio.run(go())


def test_not_where_the_wake_word_was_changed(monkeypatch):
    async def go():
        rig = _wake_rig(monkeypatch)
        rig.cog._cfg[GID]["wake"] = ["computer"]                    # the model knows "hey DJ" only
        rig.cog._on_fast_wake(_event(rig=rig))
        await asyncio.sleep(0.05)
        assert rig.cues == []
        rig.cog._cfg[GID]["wake"] = ["computer", "hey dj"]
        rig.cog._on_fast_wake(_event(rig=rig))
        await asyncio.sleep(0.05)
        assert len(rig.cues) == 1
        await rig.close()
    asyncio.run(go())


def test_speech_to_text_confirms_or_takes_it_back(monkeypatch):
    async def go():
        rig = _wake_rig(monkeypatch)
        wakes = ["hey dj"]
        rig.cog._on_fast_wake(_event(onset=50.0, rig=rig))
        await asyncio.sleep(0.05)
        rig.cog._settle_fast_wake(_job(50.0), "hey dj skip this", wakes)
        assert rig.cog.metrics.guild(GID).counts["fast_wake_confirmed"] == 1
        assert not rig.msg.deleted
        # a false alarm: "hey Jay" — the ":ear: Now listening..." goes away
        rig.msg = Msg()
        rig.cog._on_fast_wake(_event(onset=80.0, rig=rig))
        await asyncio.sleep(0.05)
        rig.cog._settle_fast_wake(_job(80.0), "hey jay did you see that", wakes)
        await asyncio.sleep(0.05)
        assert rig.cog.metrics.guild(GID).counts["fast_wake_false"] == 1 and rig.msg.deleted
        # heard by speech-to-text only: counted as a miss
        rig.cog._settle_fast_wake(_job(120.0), "hey dj pause", wakes)
        assert rig.cog.metrics.guild(GID).counts["fast_wake_missed"] == 1
        await rig.close()
    asyncio.run(go())


# ── the bundled model, on recorded speech ─────────────────────────────────────

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "wake")


@pytest.mark.skipif(not ws.WakeModel.available() or not os.path.isdir(FIXTURES),
                    reason="wake-word model or test clips not bundled")
def test_the_bundled_model_hears_hey_dj_and_not_hey_jay():
    import wave
    model = ws.WakeModel()

    def best(path):
        with wave.open(path) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        s = ws.WakeStream(model)
        return max(score for _o, score in s.push(x))
    names = sorted(os.listdir(FIXTURES))
    yes = [n for n in names if n.startswith("wake_")]
    no = [n for n in names if n.startswith("not_")]
    assert yes and no
    for n in yes:
        assert best(os.path.join(FIXTURES, n)) >= model.threshold, n
    for n in no:
        assert best(os.path.join(FIXTURES, n)) < model.threshold, n
