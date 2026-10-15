"""The wake-word cue — "I heard you", in audio.

The ask: when the wake word is heard, play a sound effect in the voice
channel (a smart speaker's light, in audio) and post `:ear:*Now listening...*`
in the server's spam channel at the same time. The sound must be switchable off.

Two paths report the wake word: the early probe (the first ~1.2 s of audio, so
the cue lands while you are still talking) and the finished transcript (the
backstop for a busy/slow box and filler-heavy openings). Both carry the same
VAD onset, and the speaker must be greeted once per utterance, not twice.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import os
import types

import aiosqlite
import numpy as np
import pytest

import cogs.tts as tts_mod
import cogs.voicedj as vd
from core import db
from core.config import settings
from tests.test_voicedj_flow import GID, OTHER, UID, Rig

CUE_BYTES = os.path.getsize(settings.voice_cue_file) if os.path.exists(
    settings.voice_cue_file) else 0


def run(coro):
    return asyncio.run(coro)


def asynctest(fn):
    """House style (see test_db_features.py): no pytest-asyncio dependency."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


# ── a cog with a channel, a TTS stub and a recorder for the sound ────────────

class _Channel:
    def __init__(self):
        self.sent = []

    def permissions_for(self, me):
        return types.SimpleNamespace(send_messages=True)

    async def send(self, text):
        self.sent.append(text)


class _CueRig:
    def __init__(self, *, cue=True, in_voice=True):
        self.channel = _Channel()
        self.sounds = []
        self.member = types.SimpleNamespace(id=UID, display_name="Alex", bot=False,
                                            voice=types.SimpleNamespace(channel=None))
        vc = types.SimpleNamespace(is_connected=lambda: in_voice, channel=None)
        self.guild = types.SimpleNamespace(id=GID, name="Test Server",
                                           voice_client=vc if in_voice else None,
                                           get_member=lambda uid: self.member)
        tts_cog = types.SimpleNamespace(cue=self._cue, speaking_windows=lambda gid: [])
        self.bot = types.SimpleNamespace(
            user=types.SimpleNamespace(id=1),
            get_guild=lambda gid: self.guild,
            get_cog=lambda name: tts_cog if name == "TTS" else None,
            loop=None)
        self.cog = vd.VoiceDJ(self.bot)
        self.cog._whisper_ok = True
        self.cog._cfg[GID] = {"enabled": True, "wake": ["hey dj"], "channel_id": None,
                              "ai": True, "chat": True, "cue": cue}
        self.cog._auth[GID] = frozenset({UID})
        self.cog._echo_channel = self._echo

    async def _echo(self, guild):
        return self.channel

    def _cue(self, guild, path, *, volume=1.0, timing=None):
        self.sounds.append((path, volume))
        return True


def _tts_stub(sounds):
    """A TTS cog that only records the cue it was asked to play."""
    return types.SimpleNamespace(cue=lambda g, p, **kw: sounds.append(p) or True,
                                 speaking_windows=lambda gid: [])


async def _cued(rig) -> None:
    """Let the fire-and-forget cue coroutines finish."""
    await asyncio.gather(*list(rig.cog._bg))


# ── what the user asked for, exactly ─────────────────────────────────────────

def test_the_message_is_the_one_asked_for():
    assert vd._CUE_MESSAGE == ":ear:*Now listening...*"


def test_the_bundled_sound_effect_is_shipped():
    assert CUE_BYTES > 0, "assets/wake_cue.wav is missing"
    assert settings.voice_cue_file.endswith("assets/wake_cue.wav")


# ── the cue itself ───────────────────────────────────────────────────────────

@asynctest
async def test_cue_plays_the_sound_and_posts_the_message_together():
    rig = _CueRig()
    await rig.cog._wake_cue(GID, UID, 100.0)
    assert rig.sounds == [(settings.voice_cue_file, settings.voice_cue_volume)]
    assert rig.channel.sent == [":ear:*Now listening...*"]
    assert rig.cog.metrics.summary(GID)["counts"].get("wake_cue") == 1


@asynctest
async def test_one_cue_per_utterance_even_though_two_paths_report_it():
    rig = _CueRig()
    await rig.cog._wake_cue(GID, UID, 100.0)      # the early probe
    await rig.cog._wake_cue(GID, UID, 100.0)      # the finished transcript
    assert len(rig.sounds) == 1
    assert len(rig.channel.sent) == 1
    assert rig.cog.metrics.summary(GID)["counts"].get("wake_cue") == 1


@asynctest
async def test_a_later_utterance_is_greeted_again():
    rig = _CueRig()
    await rig.cog._wake_cue(GID, UID, 100.0)
    await rig.cog._wake_cue(GID, UID, 100.0 + vd._CUE_SAME_UTTERANCE + 0.1)
    assert len(rig.channel.sent) == 2


@asynctest
async def test_two_speakers_do_not_silence_each_other():
    rig = _CueRig()
    rig.member.id = UID
    await rig.cog._wake_cue(GID, UID, 100.0)
    await rig.cog._wake_cue(GID, OTHER, 100.0)
    assert len(rig.channel.sent) == 2


# ── the switches ─────────────────────────────────────────────────────────────

@asynctest
async def test_voice_cue_off_keeps_the_message_and_drops_the_sound():
    rig = _CueRig(cue=False)
    await rig.cog._wake_cue(GID, UID, 100.0)
    assert rig.sounds == []
    assert rig.channel.sent == [":ear:*Now listening...*"]


@asynctest
async def test_host_sound_switch_keeps_the_message(monkeypatch):
    monkeypatch.setattr(vd, "settings", dataclasses.replace(settings,
                                                            voice_cue_sound=False))
    rig = _CueRig()
    await rig.cog._wake_cue(GID, UID, 100.0)
    assert rig.sounds == []
    assert rig.channel.sent == [":ear:*Now listening...*"]


@asynctest
async def test_host_switch_turns_the_whole_cue_off(monkeypatch):
    monkeypatch.setattr(vd, "settings", dataclasses.replace(settings, voice_cue=False))
    rig = _CueRig()
    await rig.cog._wake_cue(GID, UID, 100.0)
    assert rig.sounds == [] and rig.channel.sent == []


@asynctest
async def test_cue_setting_survives_a_refresh():
    gid = 990002                       # a guild id no other test uses
    await db.init_db()
    async with aiosqlite.connect(db.settings.db_path) as conn:
        await conn.execute("DELETE FROM voice_settings WHERE guild_id = ?", (gid,))
        await conn.execute("DELETE FROM app_settings WHERE key = ?", (f"voice_cue:{gid}",))
        await conn.commit()
    await db.set_voice_enabled(gid, True)
    await db.set_app_setting(f"voice_cue:{gid}", "off")

    cog = vd.VoiceDJ(types.SimpleNamespace(get_guild=lambda i: None,
                                           get_cog=lambda n: None))
    await cog._refresh_all()
    assert cog._cfg[gid]["cue"] is False
    assert cog._cue_sound_on(gid) is False     # the sound is off…
    assert cog._cue_on(gid) is True            # …but "Now listening" still posts
    assert "sound is off for this server" in cog._cue_line(gid)


# ── both paths that report the wake word ─────────────────────────────────────

@asynctest
async def test_the_wake_probe_cues_the_moment_it_hears_the_wake_word(monkeypatch):
    rig = _CueRig()
    cog = rig.cog

    async def fake_run(fn, model, audio, opts):
        return types.SimpleNamespace(text="Hey DJ, play something", avg_logprob=-0.2,
                                     no_speech_prob=0.01, audio_seconds=1.0)

    async def fake_get(*a, **kw):
        return object()
    monkeypatch.setattr(cog._asr, "run", fake_run)
    monkeypatch.setattr(cog._models, "get", fake_get)

    await cog._run_probe(types.SimpleNamespace(
        guild_id=GID, user_id=UID, audio=np.zeros(32000, dtype=np.float32),
        start=99.5, onset=100.0, tag=(0, 0)))
    await _cued(rig)
    assert len(rig.sounds) == 1
    assert rig.channel.sent == [":ear:*Now listening...*"]


@asynctest
async def test_a_finished_wake_utterance_cues_through_the_real_decision_path(monkeypatch):
    rig = Rig(monkeypatch)
    channel, sounds = _Channel(), []
    rig.bot.get_cog = lambda name: _tts_stub(sounds) if name == "TTS" else None

    async def echo(guild):
        return channel
    rig.cog._echo_channel = echo
    try:
        rig.say("hey dj")                       # wake only → the armed window opens
        await rig.settle()
        assert channel.sent == [":ear:*Now listening...*"]
        assert sounds == [settings.voice_cue_file]
    finally:
        await rig.close()


@asynctest
async def test_an_utterance_without_the_wake_word_is_not_greeted(monkeypatch):
    rig = Rig(monkeypatch)
    channel, sounds = _Channel(), []
    rig.bot.get_cog = lambda name: _tts_stub(sounds) if name == "TTS" else None

    async def echo(guild):
        return channel
    rig.cog._echo_channel = echo
    try:
        rig.cog._armed[(GID, UID)] = vd._Armed(since=0.0, until=1e9, session=0)
        rig.say("pause")                        # a bare command, no wake word
        await rig.settle()
        assert channel.sent == [] and sounds == []
    finally:
        await rig.close()


# ── the sound itself (TTS cog) ───────────────────────────────────────────────

def test_the_sound_file_loads_in_the_mixers_format():
    pcm = tts_mod._load_cue_pcm(settings.voice_cue_file)
    assert pcm, "the bundled cue wouldn't load"
    assert len(pcm) % 4 == 0                    # whole stereo 16-bit frames
    seconds = len(pcm) / (48_000 * 4)
    assert 0.2 < seconds < 1.5, f"unexpected cue length: {seconds:.2f}s"


def test_a_missing_or_broken_file_is_not_fatal(tmp_path):
    assert tts_mod._load_cue_pcm(str(tmp_path / "nope.wav")) == b""
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav at all")
    assert tts_mod._load_cue_pcm(str(junk)) == b""


@asynctest
async def test_a_cue_plays_but_never_counts_as_the_bot_speaking(monkeypatch):
    """A 0.6 s blip must not make voice control ignore the command said next:
    the echo guard only looks at lines that are *speech*."""
    vc = types.SimpleNamespace(is_connected=lambda: True)
    guild = types.SimpleNamespace(id=GID, voice_client=vc)
    cog = tts_mod.TTSCog(types.SimpleNamespace(get_guild=lambda gid: guild))
    gs = tts_mod._GuildSpeech()
    cog._guilds[GID] = gs

    async def fake_deliver(guild, clip):
        return
    monkeypatch.setattr(cog, "_deliver", fake_deliver)

    fut = asyncio.get_running_loop().create_future()
    fut.set_result(tts_mod._load_cue_pcm(settings.voice_cue_file))
    gs.pending.append(tts_mod._Line(0, fut, None, None, speech=False))
    cog._ensure_worker(GID, gs)
    await gs.worker

    assert list(gs.spoke) == []
    assert cog.speaking_windows(GID) == []


@asynctest
async def test_a_spoken_line_still_registers_as_speaking(monkeypatch):
    vc = types.SimpleNamespace(is_connected=lambda: True)
    guild = types.SimpleNamespace(id=GID, voice_client=vc)
    cog = tts_mod.TTSCog(types.SimpleNamespace(get_guild=lambda gid: guild))
    gs = tts_mod._GuildSpeech()
    cog._guilds[GID] = gs

    async def fake_deliver(guild, clip):
        return
    monkeypatch.setattr(cog, "_deliver", fake_deliver)

    fut = asyncio.get_running_loop().create_future()
    fut.set_result(tts_mod._chime_pcm("ok"))
    gs.pending.append(tts_mod._Line(0, fut, None, None))       # speech=True
    cog._ensure_worker(GID, gs)
    await gs.worker

    windows = cog.speaking_windows(GID)
    assert len(windows) == 1 and windows[0][1] is not None


@asynctest
async def test_cue_reports_false_when_the_bot_is_not_in_voice():
    guild = types.SimpleNamespace(id=GID, voice_client=None)
    cog = tts_mod.TTSCog(types.SimpleNamespace(get_guild=lambda gid: guild))
    assert cog.cue(guild, settings.voice_cue_file) is False
