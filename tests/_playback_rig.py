"""Test rig for the playback engine (cogs/music.py).

The real MusicCog, the real DuckingSource, and discord.py's real AudioPlayer
thread — only the voice *transport* is fake (frames are counted instead of
being Opus-encoded and sent over UDP). Inner audio sources are either scripted
fakes (mocked tests) or real FFmpeg processes (the `real_audio` tests).
"""
from __future__ import annotations

import asyncio
import threading
import time
import types

import discord
from discord.player import AudioPlayer

from core.voice_mix import FRAME_BYTES, DuckingSource

TONE = b"\x10\x00" * (FRAME_BYTES // 2)        # one non-silent 20 ms frame


class FastPlayer(AudioPlayer):
    """discord.py's player, paced at 1 ms per frame instead of 20 ms."""
    DELAY = 0.001


class PlayerVC:
    """A voice client whose play()/stop()/pause() are discord.py's own
    (VoiceClient.play minus the Opus encoder), over a fake transport."""

    def __init__(self, loop, channel, guild=None, *, player_cls=FastPlayer):
        self.loop = loop
        self.channel = channel
        self.guild = guild
        self._player = None
        self._connected = True
        self.timeout = 0.2
        self.client = types.SimpleNamespace(loop=loop)
        self.ws = types.SimpleNamespace(speak=self._speak)
        self.player_cls = player_cls
        self.frames_sent = 0              # PCM frames handed to the "socket"
        self.play_calls = 0
        self.play_error: BaseException | None = None   # make the next play() raise

    async def _speak(self, state):
        return None

    # transport
    def send_audio_packet(self, data, *, encode=True):
        if encode:
            self.frames_sent += 1

    def is_connected(self):
        return self._connected

    def wait_until_connected(self, timeout):
        return self._connected

    # VoiceClient API (discord/voice_client.py, 2.7.1)
    def play(self, source, *, after=None, **_kw):
        self.play_calls += 1
        if self.play_error is not None:
            err, self.play_error = self.play_error, None
            raise err
        if not self.is_connected():
            raise discord.ClientException("Not connected to voice.")
        if self.is_playing():
            raise discord.ClientException("Already playing audio.")
        if not isinstance(source, discord.AudioSource):
            raise TypeError("source must be an AudioSource")
        self._player = self.player_cls(source, self, after=after)
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

    async def disconnect(self, *, force=False):
        self.stop()
        self._connected = False


class ScriptedSource(discord.AudioSource):
    """Stands in for FFmpeg: `frames` frames of tone, then end-of-stream —
    optionally with an error recorded the way MusicFFmpegAudio records one,
    or raised from read()."""

    def __init__(self, frames: int, *, error: BaseException | None = None,
                 raise_on_read: BaseException | None = None, stderr: str = "",
                 gate: threading.Event | None = None):
        self.frames = frames
        self.error = error
        self.raise_on_read = raise_on_read
        self._stderr = stderr
        self._current_error = None
        self.returncode = None
        self.read_count = 0
        self.cleaned = False
        self.gate = gate

    def read(self):
        if self.gate is not None:
            self.gate.wait(5)
        if self.raise_on_read is not None:
            raise self.raise_on_read
        if self.read_count < self.frames:
            self.read_count += 1
            return TONE
        if self.error is not None and self._current_error is None:
            self._current_error = self.error
            self.returncode = 1
        elif self.returncode is None:
            self.returncode = 0
        return b""

    def stderr_text(self):
        return self._stderr

    def is_opus(self):
        return False

    def cleanup(self):
        self.cleaned = True


class Channel:
    def __init__(self, cid: int, name: str = "music"):
        self.id = cid
        self.name = name
        self.sent: list[str] = []

    async def send(self, content=None, **_kw):
        self.sent.append(content)

    def permissions_for(self, _member):
        return types.SimpleNamespace(send_messages=True)


class Guild:
    def __init__(self, gid: int, vc=None, *, system_channel=None):
        self.id = gid
        self.name = f"guild{gid}"
        self.voice_client = vc
        self.system_channel = system_channel
        self.me = types.SimpleNamespace(id=1)
        self.members = {}

    def get_member(self, uid):
        return self.members.get(uid)


class Bot:
    def __init__(self, loop, channels=(), cogs=None):
        self.loop = loop
        self.user = types.SimpleNamespace(id=1)
        self._channels = {c.id: c for c in channels}
        self._cogs = cogs or {}
        self.events: list[tuple] = []
        self.guilds_by_id = {}

    def get_channel(self, cid):
        return self._channels.get(cid)

    def get_cog(self, name):
        return self._cogs.get(name)

    def get_guild(self, gid):
        return self.guilds_by_id.get(gid)

    def dispatch(self, event, *args):
        self.events.append((event, *args))


async def wait_for(pred, timeout: float = 5.0, step: float = 0.005):
    """Await until pred() is truthy (or fail the test)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(step)
    raise AssertionError("timed out waiting for condition")


def make_track(title: str, *, url: str = "stream://ok", duration: float = 0,
               channel_id: int | None = None) -> dict:
    t = {
        "title": title, "stream_url": url, "webpage_url": f"https://example.test/{title}",
        "duration": duration, "thumbnail": None, "requester": "tester",
        "_needs_resolve": False, "_search_query": None, "_resolved_at": time.time(),
    }
    if channel_id is not None:
        t["request_channel_id"] = channel_id
    return t


def ducking(inner, **kw) -> DuckingSource:
    return DuckingSource(inner, volume=1.0, duck=0.3, **kw)
