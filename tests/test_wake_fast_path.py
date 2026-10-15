"""The wake-word cue's fast path (round 8): the sound goes straight onto the
mixer's effect layer instead of the TTS queue, the recogniser's check runs on
a short window, fires wait their turn fairly, and every stage is timed.

Mixer and TTS-cog tests use fake voice clients that pull frames like
discord.py's audio player; the cog tests use the flow rig (scripted checker).
MOCKED: real latency to a listener is measured on recordings separately
(scripts/wake_latency_bench.py) and live in `voice diag`.
"""
from __future__ import annotations

import asyncio
import time
import types

import discord
import numpy as np
import pytest

import cogs.tts as tts_mod
import cogs.voicedj as vd
from core import voice_mix as vm
from core import wake_spotter as ws
from core.config import settings
from core.voice_mix import FRAME_BYTES, CueTiming, DuckingSource, EffectClip, SpeechSource
from core.whisper import short_window_seconds
from tests.test_voicedj_flow import GID, OTHER, UID, Rig

TONE = (b"\x00\x10" * (FRAME_BYTES // 2)) * 5           # 5 frames, audible
MUSIC = b"\x00\x01" * (FRAME_BYTES // 2)                 # one quiet music frame


class _Music(discord.AudioSource):
    """A music source: endless quiet frames."""

    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return MUSIC

    def is_opus(self):
        return False

    def cleanup(self):
        pass


def test_an_effect_rides_on_the_music_without_ducking_it():
    src = DuckingSource(_Music(), volume=1.0)
    assert src.read() == MUSIC                            # passthrough
    fx = EffectClip(TONE)
    src.attach_effect(fx)
    mixed = src.read()
    assert mixed != MUSIC and len(mixed) == FRAME_BYTES
    assert fx.first_frame_at is not None and src.music_frames == 2
    for _ in range(4):
        src.read()
    assert fx.done and src.read() == MUSIC                # gone: passthrough again


def test_an_effect_over_held_music_never_pulls_the_track():
    music = _Music()
    src = DuckingSource(music, volume=1.0)
    src.hold = True                                       # paused, speech/effect only
    src.attach_effect(EffectClip(TONE))
    frames = [src.read() for _ in range(6)]
    assert music.reads == 0 and src.music_frames == 0     # the track stayed where it was
    assert frames[-1] == vm.SILENCE                       # held: silence keeps it alive


def test_an_effect_on_its_own_ends_when_it_has_played():
    src = SpeechSource(effect=EffectClip(TONE))
    assert all(src.read() for _ in range(5))
    assert src.read() == b""


def test_speech_can_join_a_source_that_was_only_playing_an_effect():
    src = SpeechSource(effect=EffectClip(TONE))
    src.read()
    speech = vm.SpeechClip(TONE, lead_in=0.0)
    src.attach(speech)
    assert speech.hosted and src.read()
    while src.read():
        pass
    assert speech.done


def test_the_first_audible_frame_is_stamped_not_the_silence_before_it():
    timing = CueTiming(100.0)
    fx = EffectClip(bytes(FRAME_BYTES) + TONE, timing=timing)
    src = SpeechSource(effect=fx)
    src.read()                                            # silence
    assert timing.first_frame is None
    src.read()
    assert timing.first_frame is not None and timing.attached is not None
    assert timing.ms("attached", "first_frame") >= 0


# ── the TTS cog places it at once, whatever is playing ───────────────────────

class _VC:
    """discord.VoiceClient, as far as the TTS cog uses it."""

    def __init__(self, source=None, *, paused=False):
        self.source = source
        self._playing = source is not None and not paused
        self._paused = paused
        self.after = None

    def is_connected(self):
        return True

    def is_playing(self):
        return self._playing

    def is_paused(self):
        return self._paused

    def play(self, source, *, after=None):
        if self._playing or self._paused:
            raise RuntimeError("Already playing audio.")
        self.source, self._playing, self.after = source, True, after

    def pause(self):
        self._playing, self._paused = False, True

    def resume(self):
        self._playing, self._paused = True, False

    def tick(self, n=1):
        """What discord.py's audio thread does every 20 ms."""
        for _ in range(n):
            if self._playing and self.source is not None:
                if not self.source.read():
                    self._playing = False
                    if self.after:
                        self.after(None)


def _tts(vc):
    guild = types.SimpleNamespace(id=GID, voice_client=vc)
    bot = types.SimpleNamespace(get_guild=lambda gid: guild, loop=None, get_cog=lambda n: None)
    return tts_mod.TTSCog(bot), guild


def _run(coro):
    return asyncio.run(coro)


def test_the_cue_doesnt_wait_behind_a_line_being_rendered_or_a_full_queue():
    async def go():
        vc = _VC(DuckingSource(_Music()))
        cog, guild = _tts(vc)
        gs = tts_mod._GuildSpeech()
        cog._guilds[GID] = gs
        never = asyncio.get_running_loop().create_future()      # a line still rendering…
        gs.rendering = tts_mod._Line(0, never, None, None)
        for _ in range(tts_mod._MAX_PENDING):                  # …and a full queue behind it
            gs.pending.append(tts_mod._Line(0, never, None, None))
        timing = CueTiming(time.monotonic())
        assert cog.cue(guild, settings.voice_cue_file, timing=timing)
        assert vc.source.effect is not None and timing.attached is not None
        vc.tick()
        assert timing.first_frame is not None                  # out with the next frame
        assert timing.ms("scheduled", "first_frame") < 50
        never.cancel()
    _run(go())


def test_the_cue_over_a_line_being_spoken():
    async def go():
        line = vm.SpeechClip(TONE * 20, lead_in=0.0)
        vc = _VC(SpeechSource(line))
        cog, guild = _tts(vc)
        assert cog.cue(guild, settings.voice_cue_file)
        assert vc.source.effect is not None and vc.source.clip is line
    _run(go())


def test_the_cue_over_paused_music_leaves_it_paused_where_it_was():
    async def go():
        music = _Music()
        src = DuckingSource(music)
        vc = _VC(src, paused=True)
        cog, guild = _tts(vc)
        assert cog.cue(guild, settings.voice_cue_file)
        assert src.hold and vc.is_playing()                    # resumed in hold mode for it
        while src.effect is not None:
            vc.tick()
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)                              # the watcher re-pauses
        assert vc.is_paused() and music.reads == 0             # the track never moved
    _run(go())


def test_the_cue_with_nothing_playing_plays_on_its_own():
    async def go():
        vc = _VC()
        cog, guild = _tts(vc)
        assert cog.cue(guild, settings.voice_cue_file)
        assert isinstance(vc.source, SpeechSource) and vc.source.effect is not None
    _run(go())


def test_a_song_starting_under_the_cue_takes_the_rest_of_it():
    async def go():
        vc = _VC()
        cog, guild = _tts(vc)
        assert cog.cue(guild, settings.voice_cue_file)
        fx = vc.source.effect
        vc.tick(3)
        # the music cog starts a track: stop the solo source, play the music
        old = vc.source
        vc._playing = False
        old.cleanup()
        vc.play(DuckingSource(_Music()))
        await asyncio.sleep(0.06)                              # the watcher moves it over
        assert vc.source.effect is fx and not fx.done
    _run(go())


def test_the_cue_never_counts_as_the_bot_speaking():
    async def go():
        vc = _VC(DuckingSource(_Music()))
        cog, guild = _tts(vc)
        cog.cue(guild, settings.voice_cue_file)
        assert cog.speaking_windows(GID) == []
    _run(go())


# ── the check: short window, fair queue, stale and rechecks ──────────────────

def test_the_checks_window_is_the_clip_plus_silence():
    assert short_window_seconds(3.0) == 5 and short_window_seconds(4.2) == 7
    assert short_window_seconds(0.5) == 5 and short_window_seconds(40) == 30


class _SlowVerifier:
    def __init__(self, delay=0.1, heard="Hey DJ"):
        self.delay, self.text, self.asked = delay, heard, []
        self.busy = False

    async def heard(self, audio):
        self.asked.append(time.monotonic())
        await asyncio.sleep(self.delay)
        return self.text, self.delay

    def median_seconds(self):
        return self.delay


def _fast_rig(monkeypatch, verifier):
    from tests.test_wake_spotter import FakeWakeModel
    rig = Rig(monkeypatch)
    rig.cog.wake_model = FakeWakeModel()
    rig.cog.wake_verifier = verifier
    rig.cues = []

    async def cue(gid, uid, onset, *, timing=None):
        rig.cues.append((uid, onset, timing.tier if timing else None))

    async def typing_start(*a, **kw):
        return None
    rig.cog._wake_cue = cue
    rig.cog._typing_start = typing_start
    return rig


def _fire(rig, uid=UID, onset=50.0, score=0.9, kind="fire"):
    tag = (rig.cog._session.get(GID, 0), rig.cog._epoch.get((GID, uid), 0))
    return ws.WakeEvent(GID, uid, onset + 0.7, score, onset, tag,
                        audio=np.zeros(16000, np.float32), kind=kind)


def test_two_speakers_at_once_are_both_checked_in_turn(monkeypatch):
    async def go():
        rig = _fast_rig(monkeypatch, _SlowVerifier(0.1))
        rig.cog._on_fast_wake(_fire(rig, UID))
        rig.cog._on_fast_wake(_fire(rig, OTHER))
        await asyncio.sleep(0.35)
        assert [u for u, _o, _t in rig.cues] == [UID, OTHER]       # nobody dropped
        assert rig.cog.metrics.guild(GID).counts.get("fast_wake_busy") is None
        await rig.close()
    asyncio.run(go())


def test_a_fire_left_waiting_too_long_is_not_cued_late(monkeypatch):
    async def go():
        monkeypatch.setattr(vd, "_WAKE_CHECK_STALE", 0.15)
        rig = _fast_rig(monkeypatch, _SlowVerifier(0.3))
        rig.cog._on_fast_wake(_fire(rig, UID))
        rig.cog._on_fast_wake(_fire(rig, OTHER))
        await asyncio.sleep(0.5)
        assert [u for u, _o, _t in rig.cues] == [UID]
        assert rig.cog.metrics.guild(GID).counts["fast_wake_stale"] == 1
        await rig.close()
    asyncio.run(go())


def test_instant_tier_cues_before_the_check_and_a_no_takes_the_message_back(monkeypatch):
    async def go():
        import dataclasses
        monkeypatch.setattr(vd, "settings", dataclasses.replace(
            vd.settings, voice_fast_wake_instant=0.85))
        rig = _fast_rig(monkeypatch, _SlowVerifier(0.1, heard="Hey Jay, did you see that?"))
        retracted = []

        async def retract(gid, uid, onset):
            retracted.append(onset)
        rig.cog._retract_cue = retract
        rig.cog._on_fast_wake(_fire(rig, UID, score=0.95))
        await asyncio.sleep(0.02)
        assert rig.cues == [(UID, 50.0, "instant")]                 # before the check came back
        await asyncio.sleep(0.2)
        c = rig.cog.metrics.guild(GID).counts
        assert c["fast_wake_false_cue"] == 1 and retracted == [50.0]
        # below the instant level: checked first, as before
        rig.cog._on_fast_wake(_fire(rig, UID, onset=70.0, score=0.5))
        await asyncio.sleep(0.2)
        assert len(rig.cues) == 1
        await rig.close()
    asyncio.run(go())


def test_a_check_that_heard_only_hey_asks_for_another_look(monkeypatch):
    async def go():
        rig = _fast_rig(monkeypatch, _SlowVerifier(0.01, heard="Hey, D"))
        asked = []
        rig.cog._front = types.SimpleNamespace(
            request_wake_recheck=lambda g, u, at: asked.append((g, u, round(at, 2))))
        rig.cog._on_fast_wake(_fire(rig, UID))
        await asyncio.sleep(0.05)
        assert asked == [(GID, UID, round(50.7 + vd._WAKE_RECHECK_AFTER, 2))]
        rig.cog.wake_verifier.text = "Hey DJ"
        rig.cog._on_fast_wake(_fire(rig, UID, kind="recheck"))
        await asyncio.sleep(0.05)
        assert [u for u, _o, _t in rig.cues] == [UID]
        assert rig.cog.metrics.guild(GID).counts["fast_wake_recheck"] == 1
        rig.cog._front = None
        await rig.close()
    asyncio.run(go())


def test_old_session_fires_are_ignored_and_one_utterance_is_one_cue(monkeypatch):
    async def go():
        rig = _fast_rig(monkeypatch, _SlowVerifier(0.01))
        stale = ws.WakeEvent(GID, UID, 50.7, 0.9, 50.0, (99, 0), audio=np.zeros(16000, np.float32))
        rig.cog._on_fast_wake(stale)
        for _ in range(3):
            rig.cog._on_fast_wake(_fire(rig, UID))
            await asyncio.sleep(0.03)
        assert rig.cues == [(UID, 50.0, "checked")]
        await rig.close()
    asyncio.run(go())


def test_the_command_said_right_after_the_cue_still_reaches_recognition(monkeypatch):
    async def go():
        rig = Rig(monkeypatch)
        rig.say("hey dj")                                 # wake only: the window opens
        await rig.settle()
        rig.say("skip", start=time.monotonic() - 0.2, speech_ms=400)
        await rig.settle()
        assert rig.executed == [(GID, UID, "skip", None)]
        assert "bot_speaking" not in rig.drops()
        await rig.close()
    asyncio.run(go())


def test_the_front_end_hands_the_utterance_over_again_when_asked(monkeypatch):
    from tests.test_wake_spotter import FakeWakeModel
    events = []
    stream = __import__("core.voice_frontend", fromlist=["SpeakerStream"]).SpeakerStream(
        GID, UID, __import__("core.voice_frontend", fromlist=["SegmenterConfig"]).SegmenterConfig(),
        lambda seg: None, decoder_factory=lambda: None, vad_factory=lambda: None,
        wake=events.append, wake_model=FakeWakeModel(), wake_threshold=0.5)
    stream.segmenter._in_speech = True
    stream.segmenter._onset_t = 10.0
    stream.segmenter._seg = [np.zeros(512, np.float32)]
    stream.recheck_after = 10.4
    stream._spot(np.zeros(1280, np.float32), 10.3)
    assert events == []                                   # not yet
    stream._spot(np.zeros(1280, np.float32), 10.5)
    assert [e.kind for e in events] == ["recheck"] and events[0].onset == 10.0


def test_the_check_looks_cheapest_first_and_ends_with_the_full_pass(monkeypatch):
    """5-second window → 10-second window → the old 30-second pass, stopping
    at the first that hears the wake word (assets/wakeword/README.md has why
    the last one stays)."""
    import types as _t

    from core import whisper
    from core.wake_spotter import check_words
    looks = []

    def short(model, clip, *, language=None, minimum=5.0, max_tokens=16):
        looks.append(f"s{minimum:g}")
        return model[f"s{minimum:g}"]

    def full(model, clip, opts):
        looks.append("full")
        return _t.SimpleNamespace(text=model["full"])
    monkeypatch.setattr(whisper, "transcribe_short", short)
    monkeypatch.setattr(whisper, "transcribe_command", full)
    ok = lambda t: t.lower().startswith("hey dj")                   # noqa: E731
    audio = np.zeros(16000, np.float32)
    heard = {"s5": "Hey DJ", "s10": "", "full": ""}
    assert check_words(heard, audio, accept=ok) == "Hey DJ" and looks == ["s5"]
    looks.clear()
    heard = {"s5": "and DJ", "s10": "Hey DJ", "full": ""}
    assert check_words(heard, audio, accept=ok) == "Hey DJ" and looks == ["s5", "s10"]
    looks.clear()
    heard = {"s5": "Thank you", "s10": "Thank you, Jay", "full": "Hey DJ, play"}
    assert check_words(heard, audio, accept=ok) == "Hey DJ, play"
    assert looks == ["s5", "s10", "full"]
    looks.clear()
    assert check_words(heard, audio, accept=ok, full=False) == "Thank you"   # off: no third look
    assert looks == ["s5", "s10"]
    looks.clear()
    heard = {"s5": "Hey BJ", "s10": "Hey BJ", "full": "Hey BJ"}
    assert check_words(heard, audio, accept=ok) == "Hey BJ"         # a no is still a no
