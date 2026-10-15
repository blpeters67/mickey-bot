"""
Speech over music — how `-tts` talks while a track is playing.

discord.py plays exactly one AudioSource per voice connection, so "speak while
the music plays" can't be a second player. Instead every music source is a
`DuckingSource` (core.audio.create_audio_source builds them): a drop-in
PCMVolumeTransformer that normally just passes frames through untouched, but
can have a `SpeechClip` laid over it. While a clip plays, each 20 ms frame is
the music, ducked to TTS_DUCK_PERCENT, plus the speech; when the clip ends the
music eases back up. Ramped both ways — a hard volume step clicks.

Sound effects (the wake-word cue) have a layer of their own: an
`EffectClip` is laid over whatever is playing — music, a spoken line, a held
pause, or nothing — the moment it's asked for, rather than waiting its turn
behind speech in the TTS queue, and it never ducks the music or counts as the
bot talking. It records when its first audible frame left the mixer.

One clip object, three situations:

  * **Music playing** — the clip rides on the track's DuckingSource.
  * **Music paused** — the source goes into `hold`: it stops pulling music
    (so the track, and the frame-counting playback clock, stay exactly where
    they were) and emits only speech. The player is resumed for the clip and
    paused again afterwards. `music_is_paused()` / `resume_music()` make the
    pause/resume commands see through this, so `resume` mid-sentence brings
    the track back in under the voice.
  * **Nothing playing** — the clip plays on its own as a `SpeechSource`.

A clip remembers how far it got, so if its host goes away mid-sentence (a
skip, a seek, the track ending) the TTS cog re-attaches the rest wherever
playback is now, and the sentence carries on instead of dying.
"""
from __future__ import annotations

import logging
import threading
import time
import warnings
from typing import Callable

import discord

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)   # stdlib audioop on 3.12
    import audioop                                        # audioop-lts on 3.13+

log = logging.getLogger(__name__)

#: One 20 ms frame of 48 kHz stereo s16le — what discord.py sends per tick.
FRAME_BYTES = discord.opus.Encoder.FRAME_SIZE            # 3840
FRAME_SECONDS = discord.opus.Encoder.FRAME_LENGTH / 1000  # 0.02
SILENCE = bytes(FRAME_BYTES)

# Gain change per frame. Duck in ~100 ms (it's done before the clip's lead-in
# silence ends, so the first word is never fighting full-volume music), and
# recover over ~400 ms so the music swells back rather than jumping.
_ATTACK = 0.15
_RELEASE = 0.0375


def _default_duck() -> float:
    from .config import settings
    return max(0.0, min(1.0, settings.tts_duck_percent / 100))


class SpeechClip:
    """Decoded speech (48 kHz stereo s16le), handed out one frame at a time.

    Frames are pulled on discord.py's audio thread while the TTS cog attaches,
    cancels and inspects the clip from the event loop, hence the lock.
    """

    def __init__(self, pcm: bytes, *, lead_in: float = 0.12):
        pad = (-len(pcm)) % FRAME_BYTES
        lead = bytes(round(lead_in / FRAME_SECONDS) * FRAME_BYTES)
        self._pcm = memoryview(lead + pcm + bytes(pad))
        self._pos = 0
        self._cancelled = False
        self._host: object | None = None
        self._lock = threading.Lock()

    @property
    def seconds(self) -> float:
        return len(self._pcm) / FRAME_BYTES * FRAME_SECONDS

    @property
    def done(self) -> bool:
        """Played to the end, or cancelled."""
        return self._cancelled or self._pos >= len(self._pcm)

    @property
    def hosted(self) -> bool:
        """Some source is currently playing this clip."""
        return self._host is not None

    def next_frame(self) -> bytes | None:
        with self._lock:
            if self._cancelled or self._pos >= len(self._pcm):
                return None
            frame = bytes(self._pcm[self._pos:self._pos + FRAME_BYTES])
            self._pos += FRAME_BYTES
            return frame

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    def _bind(self, host: object) -> None:
        with self._lock:
            self._host = host

    def _release(self, host: object) -> None:
        with self._lock:
            if self._host is host:
                self._host = None


def _audible(frame: bytes, floor: int = 100) -> bool:
    """Any sample above `floor` (16-bit): not the silence a sound may open with."""
    try:
        return audioop.max(frame, 2) > floor
    except Exception:
        return True


class CueTiming:
    """When each stage of one wake-word cue happened (time.monotonic()):
    the detector's frame, the event reaching the bot, the check, the sound
    being asked for, laid onto the mixer, and its first audible frame handed
    to Discord's audio sender. What a listener hears is later still — Discord's
    transport and the listener's jitter buffer, which the bot can't see."""

    __slots__ = ("detected", "received", "verify_start", "verify_end", "scheduled",
                 "attached", "first_frame", "score", "tier")

    def __init__(self, detected: float | None = None, *, score: float = 0.0, tier: str = ""):
        self.detected = detected
        self.received = self.verify_start = self.verify_end = None
        self.scheduled = self.attached = self.first_frame = None
        self.score = score
        self.tier = tier

    def ms(self, a: str, b: str) -> float | None:
        x, y = getattr(self, a), getattr(self, b)
        return None if x is None or y is None else (y - x) * 1000.0


class EffectClip(SpeechClip):
    """A short sound effect on the mixer's effect layer (see the module doc).
    No lead-in silence: the first frame is the sound."""

    def __init__(self, pcm: bytes, *, timing: CueTiming | None = None):
        super().__init__(pcm, lead_in=0.0)
        self.timing = timing
        self.attached_at: float | None = None
        self.first_frame_at: float | None = None

    def next_frame(self) -> bytes | None:
        frame = super().next_frame()
        if frame is not None and self.first_frame_at is None and _audible(frame):
            self.first_frame_at = time.monotonic()
            if self.timing is not None:
                self.timing.first_frame = self.first_frame_at
        return frame


def _overlay(base: bytes, fx: bytes | None) -> bytes:
    if fx is None:
        return base
    if not base:
        return fx
    return audioop.add(base, fx, 2)


class _EffectHost:
    """The effect layer, for any source that can carry it."""

    _fx: EffectClip | None = None

    def attach_effect(self, fx: EffectClip) -> None:
        old, self._fx = self._fx, fx
        if old is not None and old is not fx:
            old._release(self)
        fx._bind(self)
        if fx.attached_at is None:
            fx.attached_at = time.monotonic()
            if fx.timing is not None and fx.timing.attached is None:
                fx.timing.attached = fx.attached_at

    @property
    def effect(self) -> EffectClip | None:
        return self._fx

    def _effect_frame(self) -> bytes | None:
        fx = self._fx
        if fx is None:
            return None
        frame = fx.next_frame()
        if frame is None:
            if self._fx is fx:
                self._fx = None
            fx._release(self)
        return frame


class DuckingSource(_EffectHost, discord.PCMVolumeTransformer):
    """The music source. Identical to PCMVolumeTransformer (volume control,
    isinstance checks, cleanup) until a SpeechClip is attached.

    It also keeps the evidence the music cog needs to tell a track that
    *played* from one that merely *ran*:

    * ``music_frames`` counts frames of the track itself handed to the player.
      Speech and chimes laid over the music (or played while the music is held
      or already dead) are not counted — a TTS line over a stream that never
      started used to look exactly like a song that played.
    * ``_current_error`` forwards the wrapped FFmpeg source's error. discord.py
      checks it on the *outer* source after an empty read; without the
      forwarding an FFmpeg failure looked like a normal end of track.
    """

    def __init__(self, original: discord.AudioSource, volume: float = 1.0,
                 *, duck: float | None = None,
                 on_first_music: Callable[[], None] | None = None):
        super().__init__(original, volume=volume)
        self.duck = _default_duck() if duck is None else duck
        #: Paused-music mode: emit only speech, never pull music frames.
        self.hold = False
        self._clip: SpeechClip | None = None
        self._gain = 1.0
        self._music_done = False
        self._lock = threading.Lock()
        #: Frames of actual music delivered (see class docstring).
        self.music_frames = 0
        #: time.monotonic() of the first music frame / of the music's end.
        self.first_music_at: float | None = None
        self.music_ended_at: float | None = None
        #: Called once, on the audio thread, when the first music frame goes
        #: out — the moment a track is really "playing". Must not block.
        self.on_first_music = on_first_music

    @property
    def _current_error(self) -> Exception | None:
        return getattr(self.original, "_current_error", None)

    @property
    def speaking(self) -> bool:
        return self._clip is not None

    def _pull_music(self) -> bytes:
        """One frame from the track (volume applied), with the accounting."""
        data = super().read()
        if data:
            self.music_frames += 1
            if self.music_frames == 1:
                self.first_music_at = time.monotonic()
                cb = self.on_first_music
                if cb is not None:
                    try:
                        cb()
                    except Exception:
                        log.exception("first-audio callback failed")
        elif self.music_ended_at is None:
            self.music_ended_at = time.monotonic()
        return data

    def attach(self, clip: SpeechClip) -> None:
        """Lay `clip` over this source (replacing any previous one)."""
        with self._lock:
            old, self._clip = self._clip, clip
        if old is not None and old is not clip:
            old._release(self)
        clip._bind(self)

    def _drop(self, clip: SpeechClip) -> None:
        with self._lock:
            if self._clip is clip:
                self._clip = None
        clip._release(self)

    def read(self) -> bytes:
        clip = self._clip
        if clip is None and self._fx is None and not self.hold and self._gain >= 1.0:
            return self._pull_music()   # the normal case: an untouched passthrough
        return _overlay(self._read_mixed(clip), self._effect_frame())

    def _read_mixed(self, clip: SpeechClip | None) -> bytes:
        music = b""
        if not self.hold and not self._music_done:
            music = self._pull_music()
            if not music:
                self._music_done = True     # track over; keep going while speech lasts

        speech = clip.next_frame() if clip is not None else None
        if clip is not None and speech is None:
            self._drop(clip)

        target = self.duck if speech is not None else 1.0
        if self._gain > target:
            self._gain = max(target, self._gain - _ATTACK)
        elif self._gain < target:
            self._gain = min(target, self._gain + _RELEASE)

        if music:
            if self._gain < 1.0:
                music = audioop.mul(music, 2, self._gain)
            return audioop.add(music, speech, 2) if speech is not None else music
        if speech is not None:
            return speech
        # Nothing left to say. Held (paused) music must keep the player alive
        # with silence until the TTS cog re-pauses it — an empty read would end
        # the track. A finished track can end now.
        return SILENCE if self.hold else b""

    def cleanup(self) -> None:
        with self._lock:
            clip, self._clip = self._clip, None
        if clip is not None:
            clip._release(self)          # the TTS cog re-homes the rest of it
        fx, self._fx = self._fx, None
        if fx is not None:
            fx._release(self)            # …and the cue's watcher the rest of the cue
        super().cleanup()


class SpeechSource(_EffectHost, discord.AudioSource):
    """Plays a spoken clip, and/or a sound effect, when nothing else is
    playing. Ends when both are done."""

    def __init__(self, clip: SpeechClip | None = None, *, effect: EffectClip | None = None):
        self.clip = None
        self._fx = None
        if clip is not None:
            self.attach(clip)
        if effect is not None:
            self.attach_effect(effect)

    def attach(self, clip: SpeechClip) -> None:
        """A spoken line joining a source that was only playing an effect."""
        old, self.clip = self.clip, clip
        if old is not None and old is not clip:
            old._release(self)
        clip._bind(self)

    def read(self) -> bytes:
        speech = self.clip.next_frame() if self.clip is not None else None
        fx = self._effect_frame()
        if speech is None and fx is None:
            return b""
        if speech is None:
            return fx
        return _overlay(speech, fx)

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        if self.clip is not None:
            self.clip._release(self)
        if self._fx is not None:
            self._fx._release(self)


# ──────────────────────────────────────────────────────────────────────────────
# Pause/resume that see through a speech hold. The music cog and voice control
# use these instead of vc.is_playing()/is_paused()/resume() directly.
# ──────────────────────────────────────────────────────────────────────────────

def music_held(vc) -> bool:
    """Music paused while speech plays over the pause."""
    src = getattr(vc, "source", None)
    return isinstance(src, DuckingSource) and src.hold


def music_is_playing(vc) -> bool:
    """vc.is_playing(), except paused music with speech over it doesn't count."""
    return vc.is_playing() and not music_held(vc)


def music_is_paused(vc) -> bool:
    """vc.is_paused(), including paused music with speech over it."""
    return vc.is_paused() or (vc.is_playing() and music_held(vc))


def resume_music(vc) -> None:
    """vc.resume() that also lifts a speech hold, so the track comes back in
    under whatever is still being said. Like vc.resume() it zeroes the frame
    counter — call the playback clock's resume() afterwards, as before."""
    src = getattr(vc, "source", None)
    if isinstance(src, DuckingSource):
        src.hold = False
    vc.resume()
