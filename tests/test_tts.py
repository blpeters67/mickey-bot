"""Tests for -tts: text cleanup, voice resolution, the speech-over-music mixer,
and the routing that keeps a sentence going through skips and pauses.

The routing tests drive discord.py's real AudioPlayer thread against a fake
voice socket, with silent music and speech frames numbered 1..N — so the
captured output shows exactly which speech frames were sent, in what order and
how often. Synthesis itself needs Microsoft's service, so edge-tts is stubbed;
decoding runs through real ffmpeg when it's installed.
"""
from __future__ import annotations

import array
import asyncio
import shutil
import subprocess
import sys
import types

import discord
import pytest

from core import tts, voice_mix
from core.voice_mix import FRAME_BYTES, DuckingSource, SpeechClip


def run(coro):
    return asyncio.run(coro)


def frame(value: int) -> bytes:
    return array.array("h", [value] * (FRAME_BYTES // 2)).tobytes()


def sample(frame_bytes: bytes) -> int:
    a = array.array("h")
    a.frombytes(frame_bytes[:4])
    return a[0]


# ── text cleanup ──────────────────────────────────────────────────────────────

NAMES = {111111111111111111: "Ty", 222222222222222222: "xX_Sniper_Xx 🎯"}
ROLES = {333333333333333333: "Night-Owls"}
CHANNELS = {444444444444444444: "general-chat"}


def clean(text, **kw):
    return tts.clean_for_speech(text, member_name=NAMES.get, role_name=ROLES.get,
                                channel_name=CHANNELS.get, **kw)


@pytest.mark.parametrize("text, spoken", [
    ("idk, brb", "I don't know, be right back"),
    ("hey <@111111111111111111> ty", "hey Ty thank you"),          # a name is never "expanded"
    ("<@!222222222222222222> hi", "xX Sniper Xx hi"),
    ("<@999999999999999999> left", "someone left"),
    ("see <#444444444444444444>, <@&333333333333333333>", "see general chat, Night Owls"),
    ("watch https://www.youtube.com/watch?v=dQw4w9WgXcQ", "watch a YouTube link"),
    ("(see https://x.com/a/status/1).", "(see a Twitter link)."),
    ("https://tenor.com/view/cat-123", "a GIF"),
    ("[docs](https://example.com/x) or https://sub.example.org/y",
     "docs or a link to sub.example.org"),
    ("it was ||the butler||", "it was spoiler"),
    ("**bold** *it* __under__ ~~s~~ `x`", "bold it under s x"),
    ("```py\nprint(1)\n```", "code"),
    ("nooooooooo 1000000!!!!!", "nooo 1000000!"),
    ("lol 😂😂 ok 👍🏽", "lol ok"),
    (":) hello :D xD <3", "hello"),
    ("<:pepe_laugh:123456789012345678> nice", "pepe laugh nice"),
    ("@everyone game night <t:1700000000:R>", "everyone game night"),
    ("@silent going afk", "going afk"),
    ("first\nsecond\nthird!", "first. second. third!"),
    ("> quoted\n- item", "quoted. item"),
    ("</ttsvoice:123456789012345678> to change", "slash ttsvoice to change"),
])
def test_clean_for_speech(text, spoken):
    assert clean(text) == spoken


@pytest.mark.parametrize("text", [
    "I live in the U.S.",          # not "you.S."
    "500 BC",                      # capitalised: an era, not "because"
    "an RN on shift",              # a nurse, not "right now"
    "Ty is here",                  # a name
])
def test_shorthand_leaves_real_words_alone(text):
    assert clean(text) == text


def test_shorthand_can_be_disabled_for_names():
    assert tts.clean_for_speech("u", expand=False) == "u"
    assert tts.clean_for_speech("u") == "you"


@pytest.mark.parametrize("text", ["", "😂😂", "❤️", "   ", ":) :("])
def test_nothing_pronounceable_is_empty(text):
    assert clean(text) == ""


def test_trim_cuts_at_a_word():
    text, trimmed = tts.trim_for_speech("word " * 100, 50)
    assert trimmed and len(text) <= 50 and text.endswith("word")
    assert tts.trim_for_speech("short", 50) == ("short", False)


# ── voices ────────────────────────────────────────────────────────────────────

CATALOG = {
    "en-us-avamultilingualneural": {"ShortName": "en-US-AvaMultilingualNeural",
                                    "Gender": "Female", "Locale": "en-US"},
    "fr-fr-deniseneural": {"ShortName": "fr-FR-DeniseNeural",
                           "Gender": "Female", "Locale": "fr-FR"},
    "fr-fr-henrineural": {"ShortName": "fr-FR-HenriNeural",
                          "Gender": "Male", "Locale": "fr-FR"},
    "fr-ca-henrineural": {"ShortName": "fr-CA-HenriNeural",
                          "Gender": "Male", "Locale": "fr-CA"},
}


def test_resolve_voice_offline():
    assert tts.resolve_voice("Ava") == "en-US-AvaMultilingualNeural"
    assert tts.resolve_voice("en-gb-sonianeural") == "en-GB-SoniaNeural"   # curated id, any case
    assert tts.resolve_voice("fr-FR-DeniseNeural") == "fr-FR-DeniseNeural"  # well-formed, trusted
    assert tts.resolve_voice("fr-fr-deniseneural") is None                  # can't fix its case offline
    assert tts.resolve_voice("bogus") is None
    assert tts.resolve_voice("") is None


def test_resolve_voice_with_live_catalog():
    assert tts.resolve_voice("ava", CATALOG) == "en-US-AvaMultilingualNeural"
    assert tts.resolve_voice("andrew", CATALOG) is None          # retired → refused
    assert tts.resolve_voice("FR-fr-DeniseNeural", CATALOG) == "fr-FR-DeniseNeural"
    assert tts.resolve_voice("fr-FR-Denise", CATALOG) == "fr-FR-DeniseNeural"
    assert tts.resolve_voice("denise", CATALOG) == "fr-FR-DeniseNeural"
    assert tts.resolve_voice("henri", CATALOG) is None           # ambiguous first name


def test_curated_list_hides_retired_voices():
    names = [n for n, _, _ in tts.curated_voices(CATALOG)]
    assert names == ["ava"]
    assert len(tts.curated_voices()) == len(tts.VOICES)


def test_voice_label():
    assert tts.voice_label("en-US-AvaMultilingualNeural").startswith("Ava (")
    assert tts.voice_label("fr-FR-DeniseNeural", CATALOG) == "fr-FR-DeniseNeural (female, fr-FR)"
    assert tts.voice_label("xx-XX-WhoNeural") == "xx-XX-WhoNeural"


def test_default_voice_falls_back(monkeypatch):
    monkeypatch.setattr(tts, "settings", types.SimpleNamespace(tts_voice="nonsense"))
    assert tts.default_voice() == tts.FALLBACK_VOICE
    monkeypatch.setattr(tts, "settings", types.SimpleNamespace(tts_voice="sonia"))
    assert tts.default_voice() == "en-GB-SoniaNeural"


def test_every_curated_voice_id_is_well_formed():
    for nick, (vid, blurb) in tts.VOICES.items():
        assert tts._VOICE_ID.fullmatch(vid), vid
        assert nick == nick.lower() and blurb


# ── synthesis (edge-tts stubbed) ──────────────────────────────────────────────

class _EdgeErrors:
    class EdgeTTSException(Exception): ...
    class NoAudioReceived(EdgeTTSException): ...
    class WebSocketError(EdgeTTSException): ...


def _fake_edge(monkeypatch, behaviour):
    """behaviour(voice) → bytes to stream, or an exception to raise."""
    calls = []

    class Communicate:
        def __init__(self, text, voice, **kw):
            calls.append(voice)
            self.voice = voice

        async def stream(self):
            out = behaviour(self.voice)
            if isinstance(out, Exception):
                raise out
            yield {"type": "WordBoundary"}
            yield {"type": "audio", "data": out}

    mod = types.ModuleType("edge_tts")
    mod.Communicate = Communicate
    mod.exceptions = _EdgeErrors
    monkeypatch.setitem(sys.modules, "edge_tts", mod)
    monkeypatch.setitem(sys.modules, "edge_tts.exceptions", _EdgeErrors)
    return calls


def test_synthesize_collects_audio(monkeypatch):
    _fake_edge(monkeypatch, lambda v: b"mp3bytes")
    assert run(tts.synthesize("hi", "en-US-AriaNeural")) == b"mp3bytes"


def test_synthesize_maps_errors(monkeypatch):
    _fake_edge(monkeypatch, lambda v: _EdgeErrors.NoAudioReceived("nope"))
    with pytest.raises(tts.NoSpeech):
        run(tts.synthesize("hi", "en-US-AriaNeural"))
    _fake_edge(monkeypatch, lambda v: _EdgeErrors.WebSocketError("down"))
    with pytest.raises(tts.TTSError, match="Microsoft"):
        run(tts.synthesize("hi", "en-US-AriaNeural"))


def test_render_falls_back_to_default_voice(monkeypatch):
    calls = _fake_edge(monkeypatch, lambda v: (_EdgeErrors.NoAudioReceived("x")
                                               if v == "en-US-GoneNeural" else b"ok"))

    async def fake_pcm(data):
        return data

    monkeypatch.setattr(tts, "to_pcm", fake_pcm)
    assert run(tts.render("hi", "en-US-GoneNeural")) == b"ok"
    assert calls == ["en-US-GoneNeural", tts.default_voice()]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_to_pcm_decodes_to_discord_format():
    mp3 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=300:duration=1", "-ar", "24000", "-ac", "1",
         "-b:a", "48k", "-f", "mp3", "pipe:1"],
        capture_output=True, check=True).stdout
    pcm = run(tts.to_pcm(mp3))
    assert abs(len(pcm) / 192_000 - 1.0) < 0.1      # 48 kHz × stereo × 16-bit
    assert any(pcm)
    with pytest.raises(tts.TTSError):
        run(tts.to_pcm(b"definitely not audio"))


# ── SpeechClip ────────────────────────────────────────────────────────────────

def test_clip_pads_and_leads_in():
    clip = SpeechClip(frame(5) + b"\x01\x00", lead_in=0.04)     # 2 silent frames first
    frames = []
    while (f := clip.next_frame()) is not None:
        frames.append(f)
    assert [len(f) for f in frames] == [FRAME_BYTES] * 4
    assert sample(frames[0]) == 0 and sample(frames[2]) == 5
    assert clip.done and abs(clip.seconds - 0.08) < 1e-9


def test_clip_cancel():
    clip = SpeechClip(frame(1) * 10, lead_in=0)
    clip.next_frame()
    clip.cancel()
    assert clip.done and clip.next_frame() is None


# ── DuckingSource ─────────────────────────────────────────────────────────────

class FakeMusic(discord.AudioSource):
    def __init__(self, n, value=8000):
        self.n, self.value, self.reads, self.cleaned = n, value, 0, False

    def read(self):
        if self.reads >= self.n:
            return b""
        self.reads += 1
        return frame(self.value)

    def is_opus(self):
        return False

    def cleanup(self):
        self.cleaned = True


def test_ducking_source_is_a_plain_passthrough_when_quiet():
    plain = discord.PCMVolumeTransformer(FakeMusic(5), volume=0.5)
    ducking = DuckingSource(FakeMusic(5), volume=0.5, duck=0.25)
    for _ in range(6):
        assert ducking.read() == plain.read()
    assert isinstance(ducking, discord.PCMVolumeTransformer)


def test_ducks_under_speech_then_recovers():
    src = DuckingSource(FakeMusic(200), duck=0.25)
    src.read()
    src.attach(SpeechClip(frame(1000) * 10, lead_in=0))
    out = [sample(src.read()) for _ in range(40)]
    speech_part, after = out[:10], out[10:]
    # Ducks over a few frames (no click), then music×0.25 + speech.
    assert speech_part[0] > speech_part[4]
    assert speech_part[-1] == 8000 * 0.25 + 1000
    # Speech over: the music swells back up to full, never jumping.
    assert after[0] < after[-1] == 8000
    assert all(b >= a for a, b in zip(after, after[1:]))
    assert not src.speaking


def test_hold_speaks_without_touching_the_music():
    music = FakeMusic(100)
    src = DuckingSource(music, duck=0.25)
    src.read()
    src.hold = True
    clip = SpeechClip(frame(700) * 3, lead_in=0)
    src.attach(clip)
    out = [src.read() for _ in range(5)]
    assert music.reads == 1                         # track didn't move
    assert [sample(f) for f in out[:3]] == [700, 700, 700]
    assert out[3] == voice_mix.SILENCE              # held: keep the player alive
    src.hold = False
    assert sample(src.read()) > 0 and music.reads == 2


def test_speech_outlives_the_track():
    src = DuckingSource(FakeMusic(2), duck=0.25)
    src.attach(SpeechClip(frame(500) * 5, lead_in=0))
    out = [src.read() for _ in range(6)]
    assert all(out[:5]) and out[5] == b""          # finishes the sentence, then ends


def test_cleanup_releases_the_clip_for_rehoming():
    music = FakeMusic(100)
    src = DuckingSource(music)
    clip = SpeechClip(frame(1) * 10, lead_in=0)
    src.attach(clip)
    src.read()
    assert clip.hosted
    src.cleanup()
    assert not clip.hosted and not clip.done and music.cleaned


def test_mixing_saturates_instead_of_wrapping():
    src = DuckingSource(FakeMusic(10, value=30000), duck=1.0)
    src.attach(SpeechClip(frame(30000) * 2, lead_in=0))
    assert sample(src.read()) == 32767


# ── pause/resume helpers ──────────────────────────────────────────────────────

class _VC:
    def __init__(self, source, playing, paused):
        self.source, self._playing, self._paused, self.resumed = source, playing, paused, False

    def is_playing(self):
        return self._playing

    def is_paused(self):
        return self._paused

    def resume(self):
        self.resumed = True


def test_pause_resume_helpers_see_through_a_hold():
    src = DuckingSource(FakeMusic(10))
    src.hold = True
    vc = _VC(src, playing=True, paused=False)      # talking over paused music
    assert voice_mix.music_held(vc)
    assert not voice_mix.music_is_playing(vc)
    assert voice_mix.music_is_paused(vc)
    voice_mix.resume_music(vc)
    assert vc.resumed and not src.hold
    plain = _VC(DuckingSource(FakeMusic(10)), playing=True, paused=False)
    assert voice_mix.music_is_playing(plain) and not voice_mix.music_is_paused(plain)


# ── routing against a real AudioPlayer ────────────────────────────────────────

class FakeVoiceClient:
    """Just enough of VoiceClient for discord.py's AudioPlayer thread, with
    play/stop/pause/resume/source copied from the real thing."""

    def __init__(self, loop):
        self.client = types.SimpleNamespace(loop=loop)
        self.ws = types.SimpleNamespace(speak=self._speak)
        self._player = None
        self.encoder = None
        self.timeout = 1
        self.sent: list[bytes] = []
        self.channel = types.SimpleNamespace(name="vc")

    async def _speak(self, state):
        pass

    def is_connected(self):
        return True

    def wait_until_connected(self, timeout):
        return True

    def send_audio_packet(self, data, *, encode=True):
        if encode:
            self.sent.append(data)

    def play(self, source, *, after=None):
        if self.is_playing():
            raise discord.ClientException("Already playing audio.")
        self._player = discord.player.AudioPlayer(source, self, after=after)
        self._player.start()

    def is_playing(self):
        return self._player is not None and self._player.is_playing()

    def is_paused(self):
        return self._player is not None and self._player.is_paused()

    def stop(self):
        if self._player:
            self._player.stop()
            self._player = None

    def pause(self):
        if self._player:
            self._player.pause()

    def resume(self):
        if self._player:
            self._player.resume()

    @property
    def source(self):
        return self._player.source if self._player else None


def numbered_speech(n, base):
    return b"".join(frame(base + k) for k in range(n))


def speech_markers(sent):
    return [v for v in (sample(f) for f in sent) if v]


def _routing_rig(loop):
    import cogs.tts as tts_cog
    vc = FakeVoiceClient(loop)
    guild = types.SimpleNamespace(id=4242, voice_client=vc)
    kicks = []
    music = types.SimpleNamespace(_kickstart=lambda g, v: kicks.append(g.id))
    bot = types.SimpleNamespace(loop=loop, get_guild=lambda gid: guild,
                                get_cog=lambda name: music if name == "Music" else None)
    return tts_cog.TTSCog(bot), vc, guild, kicks


def test_routing_rehomes_a_sentence_across_a_skip():
    """Mid-sentence skip: the rest plays solo, then rides the next track."""
    async def scenario():
        loop = asyncio.get_running_loop()
        cog, vc, guild, _ = _routing_rig(loop)
        vc.play(DuckingSource(FakeMusic(1000, value=0)))
        clip = SpeechClip(numbered_speech(30, 1), lead_in=0)
        task = asyncio.create_task(cog._deliver(guild, clip))
        await asyncio.sleep(0.2)
        vc.stop()                                       # skip
        await asyncio.sleep(0.15)
        vc.stop()                                       # next track takes over
        nxt = FakeMusic(1000, value=0)
        vc.play(DuckingSource(nxt))
        await asyncio.wait_for(task, 5)
        vc.stop()
        return speech_markers(vc.sent), nxt.reads

    markers, next_reads = run(scenario())
    assert markers == list(range(1, 31))               # every frame once, in order
    assert next_reads > 0


def test_routing_talks_over_a_pause_without_moving_the_track():
    async def scenario():
        loop = asyncio.get_running_loop()
        cog, vc, guild, _ = _routing_rig(loop)
        music = FakeMusic(1000, value=0)
        vc.play(DuckingSource(music))
        await asyncio.sleep(0.1)
        vc.pause()
        await asyncio.sleep(0.05)
        reads = music.reads
        await asyncio.wait_for(cog._deliver(guild, SpeechClip(numbered_speech(15, 1), lead_in=0)), 5)
        await asyncio.sleep(0.1)
        state = (vc.is_paused(), voice_mix.music_is_paused(vc), music.reads - reads)
        vc.stop()
        return speech_markers(vc.sent), state

    markers, (paused, music_paused, drift) = run(scenario())
    assert markers == list(range(1, 16))
    assert paused and music_paused and drift == 0


def test_solo_line_starts_music_queued_meanwhile():
    from core.state import get_state

    async def scenario():
        loop = asyncio.get_running_loop()
        cog, vc, guild, kicks = _routing_rig(loop)
        get_state(guild.id).queue.append({"title": "queued while talking"})
        await asyncio.wait_for(cog._deliver(guild, SpeechClip(numbered_speech(10, 1), lead_in=0)), 5)
        await asyncio.sleep(0.15)
        return speech_markers(vc.sent), kicks

    try:
        markers, kicks = run(scenario())
    finally:
        get_state(4242).queue.clear()
    assert markers == list(range(1, 11))
    assert kicks == [4242]
