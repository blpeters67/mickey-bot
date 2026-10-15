"""Regression tests for "Queued … then nothing played" (cogs/music.py engine).

Two kinds of test, kept apart on purpose:

* **Mocked** — the real MusicCog methods and discord.py's real AudioPlayer
  thread, with scripted stand-ins for FFmpeg (ScriptedSource). They pin the
  decision logic: failure vs. completion, retry, notices, queue handling.
* **real_audio** — real FFmpeg processes (core.audio.create_audio_source →
  MusicFFmpegAudio) reading from a local HTTP server that returns 403s or
  demands a header. They prove FFmpeg's actual exit status, stderr capture and
  header forwarding work end to end. No Discord connection is made in either.
"""
from __future__ import annotations

import asyncio
import http.server
import shutil
import subprocess
import threading
import types

import discord
import pytest

import cogs.music as music
from core.ffmpeg_source import FFmpegExit, StderrTail, classify_failure
from core.formatters import redact_secrets
from core.state import get_state, reset_state
from core.voice_mix import SpeechClip
from tests._playback_rig import (
    Bot, Channel, Guild, PlayerVC, ScriptedSource, ducking, make_track, wait_for,
)

FORBIDDEN = FFmpegExit(8, "[http @ 0x1] HTTP error 403 Forbidden\n"
                          "Error opening input file https://rr1---sn-x.googlevideo.com/"
                          "videoplayback?expire=1&sig=SECRETSIG&ip=10.0.0.1\n"
                          "Error opening input files: Server returned 403 Forbidden (access denied)")


class Engine:
    """A MusicCog wired to a PlayerVC, with FFmpeg and yt-dlp scripted."""

    def __init__(self, monkeypatch, gid, *, real_sources=False):
        self.loop = asyncio.get_running_loop()
        self.req = Channel(9000 + gid % 1000, "requests")
        self.bot = Bot(self.loop, channels=[self.req])
        self.vchan = types.SimpleNamespace(id=77, name="vc", members=[])
        self.vc = PlayerVC(self.loop, self.vchan)
        self.guild = Guild(gid, self.vc)
        self.vc.guild = self.guild
        self.bot.guilds_by_id[gid] = self.guild
        self.cog = music.MusicCog(self.bot)
        self.finished: list[str] = []

        async def finished(guild, track):
            self.finished.append(track["title"])
        self.cog._on_track_finished = finished

        reset_state(gid)
        self.state = get_state(gid)
        self.state.queue.clear()
        self.state.failures.clear()
        self.state.loop = False
        self.state.filter_af = None
        self.state.text_channel_id = None
        self.sources: list = []
        self.created: list[tuple[str, dict]] = []
        self.script: dict = {}
        self.resolves: list[tuple[str, bool]] = []
        self.resolve_gate: asyncio.Event | None = None
        self.on_source = None         # e.g. lay a TTS clip over the next source
        if not real_sources:
            monkeypatch.setattr(music.audio, "create_audio_source", self._create)
        else:
            real = music.audio.create_audio_source

            def spy(url, **kw):
                src = real(url, **kw)
                self.created.append((url, kw))
                self.sources.append(src)
                return src
            monkeypatch.setattr(music.audio, "create_audio_source", spy)
        monkeypatch.setattr(music.audio, "resolve_stream_url", self._resolve)
        monkeypatch.setattr(music.db, "get_guild_settings", self._settings)
        # Short fake tracks: tighten the "ended early" and resume margins.
        monkeypatch.setattr(music, "_EARLY_END_MIN_SLACK", 0.2)
        monkeypatch.setattr(music, "_RESUME_BACKOFF", 0.1)
        monkeypatch.setattr(music, "_RETRY_HEALTHY_SECONDS", 1.0)

    @staticmethod
    async def _settings(_gid):
        return {"idle_seconds": None, "leave_when_alone": None,
                "autoplay": None, "sponsorblock": 0}

    def _create(self, url, **kw):
        self.created.append((url, kw))
        plan = self.script[url]
        step = plan.pop(0) if isinstance(plan, list) else plan
        if isinstance(step, BaseException):
            raise step
        src = ducking(step())
        if self.on_source is not None:
            hook, self.on_source = self.on_source, None
            hook(src)                 # before the player's first read
        self.sources.append(src)
        return src

    async def _resolve(self, track, *, force=False):
        self.resolves.append((track["title"], force))
        if force:
            if self.resolve_gate is not None:
                await self.resolve_gate.wait()
            if isinstance(track.get("_refresh_to"), BaseException):
                raise track["_refresh_to"]
            track["stream_url"] = track.get("_refresh_to", track["stream_url"])
        return track

    def queue(self, *tracks):
        self.state.queue.extend(tracks)
        self.cog._kickstart(self.guild, self.vc)

    def events(self, name):
        return [e for e in self.bot.events if e[0] == name]

    async def idle(self, timeout=5.0):
        """Wait until nothing is playing, starting or being retried."""
        await wait_for(lambda: not self.vc.is_playing() and not self.cog._busy(self.guild)
                       and not self.cog._tasks, timeout)
        await asyncio.sleep(0.05)


def run(coro_fn, monkeypatch, gid, **kw):
    async def go():
        e = Engine(monkeypatch, gid, **kw)
        try:
            await coro_fn(e)
        finally:
            # Let background saves (e.g. the remembered music channel) finish
            # before the loop closes.
            if music._bg_tasks:
                await asyncio.gather(*music._bg_tasks, return_exceptions=True)
    asyncio.run(go())


def ok(frames):
    return lambda: ScriptedSource(frames)


def fails(frames=0, error=FORBIDDEN, **kw):
    return lambda: ScriptedSource(frames, error=error,
                                  stderr=getattr(error, "stderr", ""), **kw)


# ── 1. empty first read caused by an FFmpeg error ────────────────────────────

def test_empty_first_read_with_an_error_is_reported_not_finished(monkeypatch):
    async def body(e):
        t = make_track("The Pretender", url="s://1", duration=2, channel_id=e.req.id)
        t["_refresh_to"] = "s://2"
        e.script = {"s://1": [fails()], "s://2": [fails()]}
        e.queue(t)
        await wait_for(lambda: e.req.sent)
        await e.idle()
        # Retried once with a forced refresh, then reported — never "finished".
        assert e.resolves == [("The Pretender", False), ("The Pretender", True)]
        assert [c[0] for c in e.created] == ["s://1", "s://2"]
        assert e.finished == []
        assert all(s.music_frames == 0 for s in e.sources)
        msg = e.req.sent[-1]
        assert "Couldn't play **The Pretender**" in msg and "HTTP 403" in msg
        assert "retried once" in msg
        assert "SECRETSIG" not in msg and "googlevideo" not in msg and "://" not in msg
        assert e.state.current is None and not e.state.queue
        assert e.events("track_failed") and not e.events("track_audible")
        assert t not in e.state.history
    run(body, monkeypatch, 7101)


def test_failed_start_then_refreshed_url_plays(monkeypatch):
    async def body(e):
        t = make_track("Retry Me", url="s://1", duration=1.0, channel_id=e.req.id)
        t["_refresh_to"] = "s://2"
        e.script = {"s://1": [fails()], "s://2": [ok(50)]}
        e.queue(t)
        await wait_for(lambda: e.finished)
        await e.idle()
        assert e.finished == ["Retry Me"]
        assert e.sources[0].music_frames == 0 and e.sources[1].music_frames == 50
        assert e.req.sent == []                             # recovered: nothing to report
        assert len(e.events("track_audible")) == 1          # "playing" only once audio went out
        assert e.state.history and e.state.history[-1]["title"] == "Retry Me"
    run(body, monkeypatch, 7102)


def test_first_read_raising_is_a_startup_failure(monkeypatch):
    async def body(e):
        t = make_track("Raiser", url="s://1", duration=1, channel_id=e.req.id)
        t["_refresh_to"] = "s://2"
        e.script = {"s://1": [lambda: ScriptedSource(0, raise_on_read=OSError("pipe broke"))],
                    "s://2": [ok(50)]}
        e.queue(t)
        await wait_for(lambda: e.finished)
        await e.idle()
        assert e.finished == ["Raiser"] and e.req.sent == []
    run(body, monkeypatch, 7103)


# ── 2. partial playback, then failure ────────────────────────────────────────

def test_partial_playback_then_failure_resumes_where_it_stopped(monkeypatch):
    async def body(e):
        t = make_track("Long Song", url="s://1", duration=2.0, channel_id=e.req.id)
        t["_refresh_to"] = "s://2"
        reset = FFmpegExit(1, "[https @ 0x2] Connection reset by peer\nError in the pull function")
        e.script = {"s://1": [fails(40, error=reset)], "s://2": [ok(60)]}
        e.queue(t)
        await wait_for(lambda: e.finished)
        await e.idle()
        seek = e.created[1][1]["seek"]
        assert 0.6 < seek < 0.8, seek          # 40 frames = 0.8 s, minus the backoff
        assert e.finished == ["Long Song"] and e.req.sent == []
        assert len(e.events("track_audible")) == 1           # a resume isn't a new play
    run(body, monkeypatch, 7104)


def test_early_end_without_an_error_is_premature_not_complete(monkeypatch):
    async def body(e):
        t = make_track("Cut Short", url="s://1", duration=4.0, channel_id=e.req.id)
        t["_refresh_to"] = "s://2"
        e.script = {"s://1": [ok(30)], "s://2": [ok(10)]}     # 0.6 s, then 0.2 s more
        e.queue(t)
        await wait_for(lambda: e.req.sent)
        await e.idle()
        assert e.finished == []
        msg = e.req.sent[-1]
        assert "stopped at" in msg and "ended early" in msg
    run(body, monkeypatch, 7105)


def test_stream_duration_beats_metadata_duration(monkeypatch):
    # A Spotify stub's duration is Spotify's; the YouTube upload can be shorter.
    async def body(e):
        t = make_track("Radio Edit", url="s://1", duration=60, channel_id=e.req.id)
        t["_stream_duration"] = 1.0
        e.script = {"s://1": [ok(50)]}
        e.queue(t)
        await wait_for(lambda: e.finished)
        assert e.req.sent == []
    run(body, monkeypatch, 7106)


# ── 3. source construction and vc.play failures ──────────────────────────────

def test_source_construction_failure_reports_and_plays_the_next(monkeypatch):
    async def body(e):
        bad = make_track("No Stream", url="s://bad", duration=1, channel_id=e.req.id)
        good = make_track("Good", url="s://good", duration=1, channel_id=e.req.id)
        e.script = {"s://bad": ValueError("create_audio_source: no stream URL to play"),
                    "s://good": [ok(50)]}
        e.queue(bad, good)
        await wait_for(lambda: e.finished)
        await e.idle()
        assert e.finished == ["Good"]
        assert any("Couldn't play **No Stream**" in m for m in e.req.sent)
        assert e.state.failures[-1][1] == "No Stream"
    run(body, monkeypatch, 7107)


def test_missing_ffmpeg_keeps_the_queue_and_says_why(monkeypatch):
    async def body(e):
        a = make_track("First", url="s://a", duration=1, channel_id=e.req.id)
        b = make_track("Second", url="s://b", duration=1, channel_id=e.req.id)
        e.script = {"s://a": discord.ClientException("ffmpeg was not found."),
                    "s://b": [ok(50)]}
        e.queue(a, b)
        await wait_for(lambda: e.req.sent)
        await e.idle()
        # Nothing could play — so nothing was consumed.
        assert [t["title"] for t in e.state.queue] == ["First", "Second"]
        assert e.state.current is None
        assert "FFmpeg" in e.req.sent[-1] and "still queued" in e.req.sent[-1]
        assert e.finished == []
    run(body, monkeypatch, 7108)


def test_vc_play_not_connected_keeps_the_track_and_cleans_up(monkeypatch):
    async def body(e):
        a = make_track("A", url="s://a", duration=1, channel_id=e.req.id)
        b = make_track("B", url="s://b", duration=1, channel_id=e.req.id)
        e.script = {"s://a": [ok(50)], "s://b": [ok(50)]}
        e.vc.play_error = discord.ClientException("Not connected to voice.")
        e.queue(a, b)
        await wait_for(lambda: e.req.sent)
        await e.idle()
        assert e.sources[0].original.cleaned          # the spawned source was cleaned up
        assert [t["title"] for t in e.state.queue] == ["A", "B"]
        assert "rejoin" in e.req.sent[-1]
    run(body, monkeypatch, 7109)


def test_vc_play_rejecting_one_track_moves_on(monkeypatch):
    async def body(e):
        a = make_track("Rejected", url="s://a", duration=1, channel_id=e.req.id)
        b = make_track("Next", url="s://b", duration=1, channel_id=e.req.id)
        e.script = {"s://a": [ok(50)], "s://b": [ok(50)]}
        e.vc.play_error = TypeError("bad source")
        e.queue(a, b)
        await wait_for(lambda: e.finished)
        await e.idle()
        assert e.finished == ["Next"]
        assert any("Couldn't play **Rejected**" in m for m in e.req.sent)
    run(body, monkeypatch, 7110)


# ── 4. a failed track followed by a valid queued track ───────────────────────

def test_failed_track_then_valid_track(monkeypatch):
    async def body(e):
        a = make_track("Broken", url="s://a1", duration=1, channel_id=e.req.id)
        a["_refresh_to"] = "s://a2"
        b = make_track("Fine", url="s://b", duration=1, channel_id=e.req.id)
        e.script = {"s://a1": [fails()], "s://a2": [fails()], "s://b": [ok(50)]}
        e.queue(a, b)
        await wait_for(lambda: e.finished)
        await e.idle()
        assert e.finished == ["Fine"]
        failures = [m for m in e.req.sent if "Couldn't play" in m]
        assert len(failures) == 1 and "Broken" in failures[0]
        assert "Moving on to the next track" in failures[0]
        assert [ev[2]["title"] for ev in e.events("track_audible")] == ["Fine"]
    run(body, monkeypatch, 7111)


def test_queue_command_explains_an_empty_queue_after_a_failure(monkeypatch):
    async def body(e):
        a = make_track("The Pretender", url="s://1", duration=2, channel_id=e.req.id)
        a["_refresh_to"] = "s://2"
        e.script = {"s://1": [fails()], "s://2": [fails()]}
        e.queue(a)
        await wait_for(lambda: e.req.sent)
        await e.idle()
        out = []

        async def send(content=None, **kw):
            out.append(content)
        ctx = types.SimpleNamespace(guild=e.guild, send=send)
        await music.MusicCog.queue.callback(e.cog, ctx)
        assert out[0].startswith("Queue is empty.")
        assert "The Pretender" in out[0] and "failed to play" in out[0]
    run(body, monkeypatch, 7112)


# ── 5. intentional stops are never failures, and are never undone ───────────

def test_stop_during_the_retry_refresh_does_not_resurrect(monkeypatch):
    async def body(e):
        t = make_track("Stop Me", url="s://1", duration=1, channel_id=e.req.id)
        t["_refresh_to"] = "s://2"
        e.script = {"s://1": [fails()], "s://2": [ok(50)]}
        e.resolve_gate = asyncio.Event()
        e.queue(t)
        await wait_for(lambda: ("Stop Me", True) in e.resolves)
        assert music._now_label(e.state.current).startswith("⏳")    # visibly not playing
        reset_state(e.guild.id)                                      # `stop`
        e.resolve_gate.set()
        await e.idle()
        assert len(e.created) == 1 and e.state.current is None
        assert e.req.sent == [] and e.finished == []
    run(body, monkeypatch, 7113)


def test_skip_while_a_track_is_being_retried_moves_on(monkeypatch):
    async def body(e):
        a = make_track("Stuck", url="s://a1", duration=1, channel_id=e.req.id)
        a["_refresh_to"] = "s://a2"
        b = make_track("After", url="s://b", duration=1, channel_id=e.req.id)
        e.script = {"s://a1": [fails()], "s://a2": [ok(50)], "s://b": [ok(50)]}
        e.resolve_gate = asyncio.Event()
        e.queue(a, b)
        await wait_for(lambda: ("Stuck", True) in e.resolves)
        out = []

        async def send(content=None, **kw):
            out.append(content)
        ctx = types.SimpleNamespace(guild=e.guild, voice_client=e.vc, send=send,
                                    author=types.SimpleNamespace(voice=types.SimpleNamespace(channel=e.vchan)))
        await music.MusicCog.skip.callback(e.cog, ctx)
        e.resolve_gate.set()
        await wait_for(lambda: e.finished)
        await e.idle()
        assert out == ["⏭️ Skipped."] and e.finished == ["After"]
        assert [c[0] for c in e.created] == ["s://a1", "s://b"]
    run(body, monkeypatch, 7114)


def test_kickstart_while_a_track_ends_cannot_steal_it(monkeypatch):
    async def body(e):
        t = make_track("Ending", url="s://1", duration=1)
        nxt = make_track("Waiting", url="s://2", duration=1)
        e.state.current = t
        t["_status"] = "playing"                  # its end hasn't been handled yet
        e.state.queue.append(nxt)
        await e.cog._play_next(e.guild, e.vc, e.state.playback_gen)
        assert e.state.current is t and list(e.state.queue) == [nxt] and not e.created
    run(body, monkeypatch, 7115)


def test_lost_voice_connection_requeues_with_a_resume_point(monkeypatch):
    async def body(e):
        t = make_track("Interrupted", url="s://1", duration=4, channel_id=e.req.id)
        gate = threading.Event()
        e.script = {"s://1": [lambda: ScriptedSource(1000, gate=gate)]}
        e.queue(t)
        await wait_for(lambda: e.sources)
        gate.set()
        await wait_for(lambda: e.sources[0].music_frames >= 30)
        e.vc._connected = False                  # the voice websocket died
        await wait_for(lambda: e.req.sent)
        await asyncio.sleep(0.05)
        assert e.state.queue and e.state.queue[0] is t and e.state.current is None
        assert t["_resume_at"] > 0.3 and "rejoin" in e.req.sent[-1]
        assert e.finished == []
    run(body, monkeypatch, 7116)


# ── 6. concurrency: simultaneous enqueues, skip then enqueue ────────────────

def test_concurrent_enqueues_both_play_in_order(monkeypatch):
    async def body(e):
        a = make_track("A", url="s://a", duration=1)
        b = make_track("B", url="s://b", duration=1)
        e.script = {"s://a": [ok(50)], "s://b": [ok(50)]}
        e.state.queue.append(a)
        e.cog._kickstart(e.guild, e.vc)
        e.state.queue.append(b)
        e.cog._kickstart(e.guild, e.vc)           # two users, same instant
        await wait_for(lambda: len(e.finished) == 2)
        await e.idle()
        assert e.finished == ["A", "B"] and [c[0] for c in e.created] == ["s://a", "s://b"]
    run(body, monkeypatch, 7117)


def test_skip_immediately_followed_by_enqueue(monkeypatch):
    async def body(e):
        a = make_track("Playing", url="s://a", duration=60)
        b = make_track("Requested", url="s://b", duration=1)
        e.script = {"s://a": [ok(100000)], "s://b": [ok(50)]}
        e.queue(a)
        await wait_for(lambda: e.sources and e.sources[0].music_frames > 5)
        e.state.current = None                    # `skip`
        e.vc.stop()
        e.state.queue.append(b)                   # …and a `play` in the same breath
        e.cog._kickstart(e.guild, e.vc)
        await wait_for(lambda: e.finished)
        await e.idle()
        assert e.finished == ["Requested"]
        assert [c[0] for c in e.created] == ["s://a", "s://b"]   # started exactly once
        assert not e.req.sent
    run(body, monkeypatch, 7118)



def test_queue_loop_plays_the_songs_round_again(monkeypatch):
    async def body(e):
        async def resolve(track, *, force=False):
            e.resolves.append((track["title"], force))
            if not track.get("stream_url"):                  # a lap: found again from its page
                track["stream_url"] = track["_search_query"].replace("https://example.test/", "s://")
            return track
        monkeypatch.setattr(music.audio, "resolve_stream_url", resolve)
        a = make_track("A", url="s://A", duration=1)
        b = make_track("B", url="s://B", duration=1)
        e.script = {"s://A": ok(50), "s://B": ok(50)}
        e.state.loop_queue = True
        e.queue(a, b)
        await wait_for(lambda: len(e.finished) >= 5)
        e.state.loop_queue = False
        await e.idle()
        assert e.finished[:5] == ["A", "B", "A", "B", "A"]
        # Each lap is a fresh entry, not the old dict with its old run on it.
        assert a not in e.state.queue and e.state.current is not a
    run(body, monkeypatch, 7119)


def test_a_failed_song_drops_out_of_the_queue_loop(monkeypatch):
    async def body(e):
        real = e._resolve

        async def resolve(track, *, force=False):
            if not track.get("stream_url"):                  # a lap: found again from its page
                track["stream_url"] = track["_search_query"].replace("https://example.test/", "s://")
            return await real(track, force=force)
        monkeypatch.setattr(music.audio, "resolve_stream_url", resolve)
        a = make_track("Broken", url="s://bad", duration=1)
        b = make_track("Good", url="s://good", duration=1)
        e.script = {"s://bad": [fails(), fails()], "s://Good": ok(50), "s://good": ok(50)}
        a["_refresh_to"] = "s://bad"
        e.state.loop_queue = True
        e.queue(a, b)
        await wait_for(lambda: len(e.finished) >= 2)
        e.state.loop_queue = False
        await e.idle()
        assert set(e.finished) == {"Good"}
    run(body, monkeypatch, 7120)

# ── 7. TTS/chimes overlapping music startup ──────────────────────────────────

def test_speech_over_a_stream_that_never_started_is_not_music(monkeypatch):
    async def body(e):
        t = make_track("Silent", url="s://1", duration=2, channel_id=e.req.id)
        t["_refresh_to"] = "s://2"
        e.script = {"s://1": [fails()], "s://2": [fails()]}
        clip = SpeechClip(b"\x20\x00" * (3840 * 20), lead_in=0)   # 20 frames of "speech"
        e.on_source = lambda src: src.attach(clip)        # TTS lands as the music starts
        e.queue(t)
        await wait_for(lambda: e.req.sent)
        await e.idle()
        assert e.vc.frames_sent >= 20               # the speech really went out…
        assert e.sources[0].music_frames == 0       # …but it wasn't music
        assert e.finished == [] and "Couldn't play **Silent**" in e.req.sent[-1]
    run(body, monkeypatch, 7119)


def test_chime_over_a_good_start_counts_only_music_frames(monkeypatch):
    async def body(e):
        t = make_track("Chimed", url="s://1", duration=1)
        e.script = {"s://1": [ok(50)]}
        e.on_source = lambda src: src.attach(SpeechClip(b"\x20\x00" * (3840 * 10), lead_in=0))
        e.queue(t)
        await wait_for(lambda: e.finished)
        assert e.sources[0].music_frames == 50 and e.finished == ["Chimed"]
        assert len(e.events("track_audible")) == 1
    run(body, monkeypatch, 7120)


# ── 8. typed requests revalidate after the lookup ────────────────────────────

class _Ctx:
    def __init__(self, e, author_channel):
        self.e = e
        self.bot = e.bot
        self.guild = e.guild
        self.channel = e.req
        self.prefix = "-"
        self.sent: list[str] = []
        self.message = types.SimpleNamespace(attachments=[])
        self.author = types.SimpleNamespace(
            id=5, display_name="typer", voice=types.SimpleNamespace(channel=author_channel))

    @property
    def voice_client(self):
        return self.guild.voice_client

    async def send(self, content=None, **kw):
        self.sent.append(content)

    def typing(self):
        class _T:
            async def __aenter__(self_inner):
                return None

            async def __aexit__(self_inner, *a):
                return False
        return _T()


def _slow_lookup(e, monkeypatch):
    gate = asyncio.Event()
    started = asyncio.Event()

    async def fetch(query):
        started.set()
        await gate.wait()
        return make_track(query.title(), url="s://found", duration=1)
    monkeypatch.setattr(music.audio, "fetch_track", fetch)
    monkeypatch.setattr(music.audio, "is_any_playlist", lambda q: False)
    e.script = {"s://found": [ok(50)]}
    return gate, started


def test_typed_play_stopped_during_lookup_is_not_queued(monkeypatch):
    async def body(e):
        gate, started = _slow_lookup(e, monkeypatch)
        ctx = _Ctx(e, e.vchan)
        task = asyncio.create_task(music.MusicCog.play.callback(e.cog, ctx, None, query="late song"))
        await asyncio.wait_for(started.wait(), 5)
        reset_state(e.guild.id)                         # someone ran `stop`
        gate.set()
        await asyncio.wait_for(task, 5)
        await asyncio.sleep(0.05)
        assert not e.state.queue and not e.created
        assert "stopped while I was looking" in ctx.sent[-1]
    run(body, monkeypatch, 7121)


def test_typed_play_after_disconnect_during_lookup_is_not_queued(monkeypatch):
    async def body(e):
        gate, started = _slow_lookup(e, monkeypatch)
        ctx = _Ctx(e, e.vchan)
        task = asyncio.create_task(music.MusicCog.play.callback(e.cog, ctx, None, query="x"))
        await asyncio.wait_for(started.wait(), 5)
        e.guild.voice_client = None                     # kicked from voice
        gate.set()
        await asyncio.wait_for(task, 5)
        assert not e.state.queue and "left the voice channel" in ctx.sent[-1]
    run(body, monkeypatch, 7122)


def test_typed_play_survives_a_rejoin_during_lookup(monkeypatch):
    async def body(e):
        gate, started = _slow_lookup(e, monkeypatch)
        ctx = _Ctx(e, e.vchan)
        task = asyncio.create_task(music.MusicCog.play.callback(e.cog, ctx, None, query="keep"))
        await asyncio.wait_for(started.wait(), 5)
        fresh = PlayerVC(e.loop, e.vchan, e.guild)      # `rejoin` built a new connection
        e.guild.voice_client = fresh
        gate.set()
        await asyncio.wait_for(task, 5)
        await wait_for(lambda: e.finished)
        assert e.finished == ["Keep"] and fresh.frames_sent >= 50 and e.vc.play_calls == 0
        assert "✅ Added **Keep**" in ctx.sent[-1]
    run(body, monkeypatch, 7123)


# ── 9. pieces: stderr capture, classification, redaction, headers ────────────

def test_stderr_tail_is_bounded_and_keeps_the_end():
    tail = StderrTail(limit=64)
    for i in range(100):
        tail.write(f"line {i}\n".encode())
    text = tail.text()
    assert len(text.encode()) <= 64 and text.endswith("line 99\n")
    assert not tail.closed.is_set()
    tail.write(b"")
    assert tail.closed.is_set()


@pytest.mark.parametrize("stderr,category,retry", [
    ("Server returned 403 Forbidden (access denied)", "forbidden", True),
    ("HTTP error 429 Too Many Requests", "rate_limited", True),
    ("Server returned 404 Not Found", "gone", True),
    ("Connection reset by peer", "network", True),
    ("Invalid data found when processing input", "invalid_data", True),
    ("Output file #0 does not contain any stream", "no_audio", False),
    ("Option headers not found.", "bad_options", False),
    ("No such filter: 'bogus'", "bad_options", False),
    ("", "no_audio_produced", True),
])
def test_classify_failure(stderr, category, retry):
    info = classify_failure(None, stderr)
    assert info.category == category and info.retryable is retry
    assert "://" not in info.reason and "://" not in info.advice


def test_a_signed_url_cannot_masquerade_as_a_status_code():
    stderr = "Opening https://h.example/v?id=403&sig=1403 for reading\nInvalid data found when processing input"
    assert classify_failure(None, stderr).category == "invalid_data"


def test_redact_secrets():
    text = ("Error opening input file https://rr1.googlevideo.com/videoplayback?expire=1&sig=ABC\n"
            "Cookie: SID=secret\nAuthorization: Bearer xyz\nlsig=QQQ")
    out = redact_secrets(text)
    for secret in ("ABC", "secret", "xyz", "QQQ", "videoplayback", "expire"):
        assert secret not in out
    assert "https://rr1.googlevideo.com/…" in out


def test_stream_headers_are_cleaned():
    info = {"http_headers": {
        "User-Agent": "Mozilla/5.0", "Referer": "https://site.example/",
        "Accept-Encoding": "gzip", "Host": "evil", "X-Bad": "a\r\nInjected: 1",
        "Bad Name": "x", "Sec-Fetch-Mode": "navigate"}}
    assert music.audio.stream_headers(info) == {
        "User-Agent": "Mozilla/5.0", "Referer": "https://site.example/",
        "Sec-Fetch-Mode": "navigate"}


def test_resolved_tracks_keep_stream_headers_duration_and_expiry():
    info = {"title": "T", "url": "https://h.example/v?expire=2000000000&sig=x",
            "webpage_url": "https://w.example/t", "duration": 200,
            "http_headers": {"Referer": "https://w.example/"}}
    t = music.audio._ytdl_info_to_track(info)
    assert t["_http_headers"] == {"Referer": "https://w.example/"}
    assert t["_stream_duration"] == 200 and t["_expires_at"] == 2000000000.0


# ── 10. real FFmpeg (no mocks below the music cog) ───────────────────────────

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@pytest.fixture(scope="module")
def media_server(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    wav = tmp_path_factory.mktemp("media") / "tone.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=1", "-ar", "48000", "-ac", "2",
                    str(wav)], check=True)
    data = wav.read_bytes()
    seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append({"path": self.path, "headers": dict(self.headers)})
            refuse = self.path.startswith("/forbidden") or (
                self.path.startswith("/needs-referer")
                and self.headers.get("Referer") != "https://site.example/")
            if refuse:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield types.SimpleNamespace(base=f"http://127.0.0.1:{srv.server_address[1]}", seen=seen)
    srv.shutdown()


@needs_ffmpeg
def test_real_ffmpeg_403_is_detected_and_a_refreshed_url_plays(monkeypatch, media_server):
    async def body(e):
        t = make_track("Real Retry", url=f"{media_server.base}/forbidden?sig=SECRETSIG",
                       duration=1.0, channel_id=e.req.id)
        t["_refresh_to"] = f"{media_server.base}/tone.wav"
        e.queue(t)
        await wait_for(lambda: e.finished, timeout=15)
        first, second = e.sources
        assert first.music_frames == 0 and first.original.returncode not in (0, None)
        assert "403" in first.original.stderr_text()
        assert isinstance(first._current_error, FFmpegExit)       # forwarded by the wrapper
        assert 45 <= second.music_frames <= 52                    # 1 s of real audio
        assert e.req.sent == [] and e.finished == ["Real Retry"]
    run(body, monkeypatch, 7130, real_sources=True)


@needs_ffmpeg
def test_real_ffmpeg_403_twice_is_reported_without_the_url(monkeypatch, media_server):
    async def body(e):
        t = make_track("Real Fail", url=f"{media_server.base}/forbidden?sig=SECRETSIG",
                       duration=1.0, channel_id=e.req.id)
        t["_refresh_to"] = f"{media_server.base}/forbidden?sig=OTHERSECRET"
        e.queue(t)
        await wait_for(lambda: e.req.sent, timeout=15)
        await e.idle()
        msg = e.req.sent[-1]
        assert "HTTP 403" in msg and "retried once" in msg
        for leak in ("SECRET", "127.0.0.1", "forbidden?"):
            assert leak not in msg
        assert e.finished == [] and not e.state.queue
    run(body, monkeypatch, 7131, real_sources=True)


@needs_ffmpeg
def test_real_ffmpeg_gets_the_extractor_headers(monkeypatch, media_server):
    async def body(e):
        with_hdr = make_track("Needs Referer", url=f"{media_server.base}/needs-referer",
                              duration=1.0, channel_id=e.req.id)
        with_hdr["_http_headers"] = {"Referer": "https://site.example/",
                                     "User-Agent": "Mozilla/5.0 (Test; rv:1) Gecko/20100101"}
        e.queue(with_hdr)
        await wait_for(lambda: e.finished, timeout=15)
        hit = [s for s in media_server.seen if s["path"] == "/needs-referer"][-1]
        assert hit["headers"].get("Referer") == "https://site.example/"
        assert hit["headers"].get("User-Agent", "").startswith("Mozilla/5.0 (Test")
        assert e.req.sent == []
    run(body, monkeypatch, 7132, real_sources=True)


@needs_ffmpeg
def test_real_ffmpeg_without_the_header_fails_visibly(monkeypatch, media_server):
    async def body(e):
        bare = make_track("No Referer", url=f"{media_server.base}/needs-referer",
                          duration=1.0, channel_id=e.req.id)
        e.queue(bare)
        await wait_for(lambda: e.req.sent, timeout=15)
        assert "Couldn't play **No Referer**" in e.req.sent[-1] and e.finished == []
    run(body, monkeypatch, 7133, real_sources=True)


def test_a_crash_in_end_handling_still_moves_on(monkeypatch):
    async def body(e):
        a = make_track("A", url="s://a", duration=1, channel_id=e.req.id)
        b = make_track("B", url="s://b", duration=1, channel_id=e.req.id)
        e.script = {"s://a": [ok(50)], "s://b": [ok(50)]}
        real = music.judge_playback
        calls = []

        def boom(track, outcome):
            calls.append(track["title"])
            if track["title"] == "A":
                raise RuntimeError("bug")
            return real(track, outcome)
        monkeypatch.setattr(music, "judge_playback", boom)
        e.queue(a, b)
        await wait_for(lambda: e.finished)
        await e.idle()
        assert e.finished == ["B"] and any("went wrong" in m for m in e.req.sent)
    run(body, monkeypatch, 7140)
