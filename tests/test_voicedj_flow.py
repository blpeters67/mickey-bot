"""End-to-end tests of the voice-command decision flow in cogs/voicedj.py.

The real cog, job queue, ASR runner, model manager and grammar; fake Discord
objects and a scripted "whisper" (audio carries a marker → canned transcript).
Segments are injected where the audio front end would deliver them.

Each test pins one of the review's P1 findings: consent revoked mid-flight,
stale or old-session speech, confirmations that must not accept refusals or
earlier speech, the wake-then-pause window, and one slow song search no longer
blocking every other command."""
from __future__ import annotations

import asyncio
import time
import types

import pytest

np = pytest.importorskip("numpy")

import cogs.voicedj as vd                          # noqa: E402
from core.voice_frontend import Segment            # noqa: E402
from core.state import get_state                   # noqa: E402

GID, UID, OTHER = 501, 7, 8


class FakeVC:
    def __init__(self, channel):
        self.channel = channel
        self.playing = True
        self.paused = False
        self.source = None
        self.stopped = 0

    def is_connected(self):
        return True

    def is_playing(self):
        return self.playing and not self.paused

    def is_paused(self):
        return self.paused

    def stop(self):
        self.stopped += 1
        self.playing = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class Rig:
    """A bot, a guild, a channel, two members and a scripted transcriber."""

    def __init__(self, monkeypatch, *, guilds=(GID,), slow_asr=0.0):
        self.loop = asyncio.get_running_loop()
        self.channel = types.SimpleNamespace(name="vc", id=99)
        self.guilds = {}
        for gid in guilds:
            vc = FakeVC(self.channel)
            members = {uid: types.SimpleNamespace(
                id=uid, display_name=f"user{uid}",
                voice=types.SimpleNamespace(channel=self.channel)) for uid in (UID, OTHER)}
            g = types.SimpleNamespace(id=gid, voice_client=vc, get_member=members.get,
                                      members=members)
            self.guilds[gid] = g
        self.bot = types.SimpleNamespace(
            loop=self.loop, user=types.SimpleNamespace(id=1),
            get_guild=self.guilds.get, get_cog=self._get_cog)
        self.tts_windows = []
        self.cog = vd.VoiceDJ(self.bot)
        self.cog._whisper_ok = True
        self.feedback = []
        self.executed = []
        self.transcripts = {}
        self._marker = 0
        self.slow_asr = slow_asr

        async def fake_feedback(guild, member, heard, result, *, force=False, chime=None):
            self.feedback.append((guild.id, member.id, result, force))
        self.cog._deliver_feedback = fake_feedback

        real_execute = self.cog._execute_intent

        async def record_execute(guild, member, vc, intent, **kw):
            self.executed.append((guild.id, member.id, intent.action, intent.arg))
            return await real_execute(guild, member, vc, intent, **kw)
        self.cog._execute_intent = record_execute

        #: what the whole audio of a long request (several markers) is heard
        #: as; by default its parts' transcripts, joined
        self.wholes: dict[tuple, str] = {}
        self.heard_audio: list[tuple] = []          # markers of each recognition, in order

        def fake_transcribe(model, audio, opts):
            if self.slow_asr:
                time.sleep(self.slow_asr)
            first = np.unique(audio, return_index=True)[1]
            markers = tuple(int(audio[i]) for i in sorted(first))
            self.heard_audio.append(markers)
            if len(markers) > 1:
                text = self.wholes.get(markers, " ".join(self.transcripts[m] for m in markers))
                if isinstance(text, Exception):
                    raise text
            else:
                text = self.transcripts[markers[0]]
            return types.SimpleNamespace(text=text, avg_logprob=-0.2, no_speech_prob=0.01,
                                         audio_seconds=len(audio) / 16000)
        monkeypatch.setattr(vd, "transcribe_command", fake_transcribe)
        # YouTube, as a stand-in: one upload titled exactly what was searched
        # (so a checked search accepts it). Tests that need more replace it.
        self.searches = []

        async def fake_search(query, *, limit=5):
            self.searches.append(query)
            return [{"title": query, "uploader": "", "duration": 0, "stream_url": None,
                     "webpage_url": f"https://youtu.be/{abs(hash(query)) % 10 ** 8}",
                     "_needs_resolve": True, "_search_query": query}]
        monkeypatch.setattr(vd.audio, "search_tracks", fake_search)
        self.cog._models = vd.ModelManager(lambda name: object(), "base.en")
        for gid in guilds:
            self.cog._auth[gid] = frozenset({UID, OTHER})
            self.cog._cfg[gid] = {"enabled": True, "wake": ["hey dj"], "channel_id": None}
            get_state(gid).loop = False
            get_state(gid).volume = 1.0
            get_state(gid).queue.clear()
        self.cog._worker = asyncio.create_task(self.cog._asr_worker())

    def _get_cog(self, name):
        if name == "TTS":
            return types.SimpleNamespace(speaking_windows=lambda gid: list(self.tts_windows))
        return None

    def say(self, text, *, uid=UID, gid=GID, start=None, speech_ms=900, overlap_ms=0,
            reason="silence"):
        """Inject a finished segment whose transcript will be `text`."""
        self._marker += 1
        self.transcripts[self._marker] = text
        now = time.monotonic()
        start = now - 1.0 if start is None else start
        audio = np.full(16000, self._marker, dtype=np.float32)
        self.cog._on_segment(Segment(gid, uid, audio, start, start + 0.9, speech_ms,
                                     reason, overlap_ms))
        return self._marker

    def say_long(self, *parts, uid=UID, gid=GID, last_reason="silence"):
        """One long utterance as the front end delivers it: cut at the length
        cap into parts, each overlapping the one before by 800 ms. Returns
        the markers (the whole's audio carries them all, in order)."""
        now = time.monotonic()
        n = len(parts)
        out = []
        for i, text in enumerate(parts):
            out.append(self.say(text, uid=uid, gid=gid, start=now - (n - i) * 0.85,
                                overlap_ms=800 if i else 0,
                                reason=last_reason if i == n - 1 else "max_length"))
        return tuple(out)

    def _busy(self) -> bool:
        inflight = self.cog._asr._inflight
        return bool(len(self.cog._jobs)
                    or (inflight is not None and not inflight.done())
                    or self.cog._current_job is not None
                    or any(not t.done() for t in self.cog._decision_workers.values())
                    or any(not t.done() for t in self.cog._dispatchers.values())
                    or any(not t.done() for t in self.cog._play_workers.values()))

    async def settle(self, seconds=0.3):
        """Wait until queued speech, in-flight recognition and actions are done
        (at least 0.15 s, at most `seconds` + 2 s)."""
        start = time.monotonic()
        while time.monotonic() - start < seconds + 2:
            await asyncio.sleep(0.02)
            if time.monotonic() - start > 0.15 and not self._busy():
                break
        await asyncio.sleep(0.05)

    def drops(self, gid=GID):
        return dict(self.cog.metrics.guild(gid).drops)

    async def close(self):
        self.cog._worker.cancel()
        for t in self.cog._dispatchers.values():
            t.cancel()
        for t in [*self.cog._decision_workers.values(), *self.cog._play_workers.values(), *self.cog._bg]:
            t.cancel()
        self.cog._asr.shutdown()
        self.cog._models.shutdown()


def scenario(fn):
    """Run an async test body with a fresh Rig."""
    def wrapper(monkeypatch):
        async def go():
            rig = Rig(monkeypatch)
            try:
                await fn(rig)
            finally:
                await rig.close()
        asyncio.run(go())
    wrapper.__name__ = fn.__name__
    return wrapper


@scenario
async def test_wake_and_command_executes(rig):
    rig.say("Hey DJ, skip this song.")
    await rig.settle()
    assert rig.executed == [(GID, UID, "skip", None)]


@scenario
async def test_people_who_arent_opted_in_are_ignored(rig):
    rig.cog._auth[GID] = frozenset({OTHER})
    rig.say("hey dj skip")
    await rig.settle()
    assert rig.executed == [] and rig.drops()["not_authorized"] == 1


@scenario
async def test_opt_out_purges_queued_speech(rig):
    rig.cog._models = vd.ModelManager(lambda n: time.sleep(0.4) or object(), "base.en")
    rig.say("hey dj skip")
    rig.cog._auth[GID] = frozenset({OTHER})          # what _refresh_guild does…
    rig.cog._revoke(GID, UID, "revoked")             # …for someone who opted out
    await rig.settle(0.8)
    assert rig.executed == [] and rig.drops().get("revoked", 0) >= 1


@scenario
async def test_opt_out_during_transcription_still_stops_the_action(rig):
    rig.slow_asr = 0.3
    rig.say("hey dj skip")
    await asyncio.sleep(0.1)                         # transcription under way
    rig.cog._auth[GID] = frozenset({OTHER})
    rig.cog._revoke(GID, UID, "revoked")
    await rig.settle(0.8)
    assert rig.executed == [] and rig.drops().get("revoked", 0) == 1


@scenario
async def test_bot_moving_invalidates_older_speech(rig):
    rig.slow_asr = 0.2
    rig.say("hey dj skip")
    await asyncio.sleep(0.05)
    rig.cog._new_session(GID, "bot moved")
    await rig.settle(0.6)
    assert rig.executed == [] and rig.drops().get("session_changed", 0) == 1


@scenario
async def test_speaker_who_left_cannot_trigger(rig):
    rig.guilds[GID].members[UID].voice = None
    rig.say("hey dj skip")
    await rig.settle()
    assert rig.executed == [] and rig.drops()["speaker_left"] == 1


@scenario
async def test_stale_speech_is_not_transcribed(rig):
    rig.say("hey dj skip", start=time.monotonic() - 20)
    await rig.settle()
    assert rig.executed == [] and rig.drops()["stale_before_asr"] == 1


@scenario
async def test_short_noise_without_a_window_skips_recognition(rig):
    rig.say("thank you.", speech_ms=200)            # classic whisper hallucination on a blip
    await rig.settle()
    assert rig.drops()["too_short"] == 1


# ── confirmations ─────────────────────────────────────────────────────────────

@scenario
async def test_confirmation_flow(rig):
    rig.say("hey dj stop")
    await rig.settle()
    assert rig.executed == [] and rig.feedback[-1][3] is True      # question forced to text
    asked_at = time.monotonic()
    rig.say("so what are we playing next")                         # unrelated: keeps waiting
    await rig.settle()
    assert (GID, UID) in rig.cog._pending
    rig.say("yes", start=asked_at + 0.2, speech_ms=300)            # short "yes" is fine
    await rig.settle()
    assert rig.executed == [(GID, UID, "stop", None)]


@scenario
async def test_refusals_cancel(rig):
    rig.say("hey dj clear the queue")
    await rig.settle()
    asked = time.monotonic()
    rig.say("yes but do not clear it", start=asked + 0.2)
    await rig.settle()
    assert rig.executed == [] and (GID, UID) not in rig.cog._pending
    assert rig.feedback[-1][2] == "✖ Cancelled."


@scenario
async def test_answer_captured_before_the_question_does_not_count(rig):
    rig.slow_asr = 0.1
    early = time.monotonic()
    rig.say("hey dj stop", start=early - 1)
    rig.say("yes", start=early - 0.5, speech_ms=300)               # said before being asked
    await rig.settle(0.8)
    assert rig.executed == [] and (GID, UID) in rig.cog._pending


@scenario
async def test_another_persons_yes_does_not_confirm(rig):
    rig.say("hey dj stop")
    await rig.settle()
    rig.say("yes", uid=OTHER, start=time.monotonic() + 0.1, speech_ms=300)
    await rig.settle()
    assert rig.executed == []


@scenario
async def test_echo_of_the_bot_cannot_answer(rig):
    rig.say("hey dj stop")
    await rig.settle()
    now = time.monotonic()
    rig.tts_windows.append((now, now + 2.0))                       # bot talking
    rig.say("yes", start=now + 0.5, speech_ms=300)
    await rig.settle()
    assert rig.executed == [] and rig.drops()["bot_speaking"] == 1


# ── the wake window ───────────────────────────────────────────────────────────

@scenario
async def test_wake_then_pause_then_command(rig):
    rig.say("Hey DJ.", speech_ms=500)
    await rig.settle()
    assert (GID, UID) in rig.cog._armed
    rig.say("play mr brightside", start=time.monotonic() + 0.5)
    rig.cog._do_play = _fake_play(rig)
    await rig.settle()
    assert rig.executed == [(GID, UID, "play", "mr brightside")]
    assert (GID, UID) not in rig.cog._armed


@scenario
async def test_wake_window_expires(rig):
    rig.say("hey dj", speech_ms=500)
    await rig.settle()
    rig.say("skip", start=time.monotonic() + vd._ARM_SECONDS + 1)
    await rig.settle()
    assert rig.executed == []


# ── dispatch isolation and dedupe ─────────────────────────────────────────────

def _fake_play(rig, delay=0.0):
    async def play(guild, vc, member, query, music, **kw):
        await asyncio.sleep(delay)
        return vd.ActionResult("done", f"✅ Queued **{query}**.")
    return play


def test_slow_song_search_in_one_server_does_not_block_another(monkeypatch):
    async def go():
        rig = Rig(monkeypatch, guilds=(GID, GID + 1))
        try:
            rig.cog._do_play = _fake_play(rig, delay=1.0)
            rig.say("hey dj play something obscure", gid=GID)
            await asyncio.sleep(0.15)
            t0 = time.monotonic()
            rig.say("hey dj skip", gid=GID + 1)
            while not any(e[0] == GID + 1 for e in rig.executed):
                await asyncio.sleep(0.02)
                assert time.monotonic() - t0 < 0.8, "blocked behind the other guild's search"
        finally:
            await rig.close()
    asyncio.run(go())


@scenario
async def test_command_heard_in_both_halves_of_a_split_runs_once(rig):
    t = time.monotonic()
    rig.say("hey dj skip", start=t - 2)
    await rig.settle()
    rig.say("hey dj skip", start=t - 0.5, overlap_ms=800)
    await rig.settle()
    assert [e[2] for e in rig.executed] == ["skip"]
    assert rig.drops()["duplicate_from_split"] == 1


@scenario
async def test_loop_on_is_idempotent(rig):
    rig.say("hey dj loop on")
    await rig.settle()
    rig.say("hey dj loop on")
    await rig.settle()
    assert get_state(GID).loop is True


@scenario
async def test_louder_never_lowers_a_typed_boost(rig):
    get_state(GID).volume = 3.0
    rig.say("hey dj louder")
    await rig.settle()
    assert get_state(GID).volume == 3.0


# ── the whole path, from real Opus packets ────────────────────────────────────

def _audio_stack_ready():
    import shutil
    import subprocess
    try:
        import discord
        if not discord.opus.is_loaded():
            import ctypes.util
            discord.opus.load_opus(ctypes.util.find_library("opus") or "libopus.so.0")
        from core.voice_frontend import SileroVAD
        if not (discord.opus.is_loaded() and SileroVAD.available() and shutil.which("ffmpeg")):
            return False
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
        return " flite " in out.stdout
    except Exception:
        return False


@pytest.mark.skipif(not _audio_stack_ready(),
                    reason="needs libopus, ffmpeg with flite, and faster-whisper's Silero model")
def test_packets_through_the_tap_become_one_action(monkeypatch):
    """Chatter, a 700 ms pause, then "hey DJ skip" — as real 20 ms Opus packets
    through the recorder tap, the front-end thread, Silero VAD, the job queue
    and dispatch. Two segments come out; only the second acts."""
    import subprocess

    from discord.opus import Encoder

    from core.voice_frontend import VoiceFrontEnd

    def speech(text):
        raw = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", f"flite=text='{text}':voice=slt", "-ar", "48000", "-ac", "2",
             "-f", "s16le", "pipe:1"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype="<i2").reshape(-1, 2)

    async def go():
        rig = Rig(monkeypatch)
        heard = iter(["so anyway the game was good", "hey dj skip"])
        segments = []

        def scripted(model, audio, opts):
            segments.append(len(audio) / 16000)
            return types.SimpleNamespace(text=next(heard), avg_logprob=-0.2,
                                         no_speech_prob=0.01, audio_seconds=len(audio) / 16000)
        monkeypatch.setattr(vd, "transcribe_command", scripted)
        rig.cog._front = VoiceFrontEnd(rig.cog._segment_from_thread)
        rig.cog._front.start()
        try:
            gap = np.zeros((int(48000 * 0.7), 2), dtype="<i2")
            tail = np.zeros((int(48000 * 1.0), 2), dtype="<i2")
            timeline = np.concatenate([speech("so anyway I was saying the game was good"),
                                       gap, speech("hey D J skip"), tail])
            enc, seq, ts = Encoder(), 0, 0
            data = timeline.reshape(-1).tobytes()
            t0 = time.monotonic()
            step = 960 * 4
            for i in range(0, len(data) - step + 1, step):
                chunk = data[i:i + step]
                if np.any(np.frombuffer(chunk, dtype="<i2")):          # DTX when silent
                    rig.cog._on_frame(GID, UID, enc.encode(chunk, 960), seq, ts, 1234,
                                      t0 + (i // step) * 0.02)
                    seq += 1
                ts += 960
            deadline = time.monotonic() + 8
            while not rig.executed and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        finally:
            rig.cog._front.stop()
            await rig.close()
        return rig.executed, segments

    executed, segments = asyncio.run(go())
    assert executed == [(GID, UID, "skip", None)]
    assert len(segments) == 2
