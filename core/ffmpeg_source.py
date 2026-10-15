"""
The FFmpeg process behind every music track — with evidence when it fails.

discord.py's FFmpegPCMAudio reports the end of a stream as an empty read. A
song that played to the end and an FFmpeg that died on a 403 before producing
a single frame looked identical from the outside, and the music cog treated
both as "finished": the track was consumed and nobody was told. That is how
"Queued … The Pretender" could be followed straight away by "Queue is empty."

This module keeps what's needed to tell them apart:

  * :class:`StderrTail` — FFmpeg's stderr, captured into a bounded buffer
    (the end of the log is where the fatal line is) instead of being dropped
    or flooding the console.
  * :class:`MusicFFmpegAudio` — on end-of-stream it waits briefly for the
    process to exit, records the exit status, and turns a non-zero exit into
    ``_current_error`` (an :class:`FFmpegExit`), which discord.py hands to the
    ``after`` callback.
  * :func:`classify_failure` — FFmpeg/exception text → a category, whether a
    fresh stream URL could help, and a user-facing explanation that never
    contains the (signed) stream URL or any header value.

Tests: tests/test_playback_engine.py (including real-FFmpeg cases).
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass

import discord

from .formatters import redact_secrets

FRAME_BYTES = discord.opus.Encoder.FRAME_SIZE


class StderrTail:
    """A write-only, file-like sink that keeps the last `limit` bytes.

    discord.py runs a reader thread that calls :meth:`write` with each chunk
    of FFmpeg's stderr (it needs an object *without* ``fileno`` to do that).
    """

    def __init__(self, limit: int = 8192):
        self._buf = bytearray()
        self._limit = limit
        self._lock = threading.Lock()
        self.total = 0
        #: Set once the stream hit EOF (FFmpeg exited and its stderr closed).
        self.closed = threading.Event()

    def write(self, data: bytes) -> int:
        if not data:
            # EOF. discord.py's reader keeps calling write(b"") until the
            # source is cleaned up — note it, and don't let it spin a core.
            self.closed.set()
            time.sleep(0.05)
            return 0
        with self._lock:
            self._buf += data
            self.total += len(data)
            if len(self._buf) > self._limit:
                del self._buf[:len(self._buf) - self._limit]
        return len(data)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        with self._lock:
            return bytes(self._buf).decode("utf-8", "replace")


class FFmpegExit(Exception):
    """FFmpeg exited with a non-zero status. ``stderr`` is the raw tail —
    log it through :func:`redact_secrets`, never show it to users."""

    def __init__(self, returncode: int | None, stderr: str):
        self.returncode = returncode
        self.stderr = stderr
        last = next((ln for ln in reversed(stderr.strip().splitlines()) if ln.strip()), "")
        super().__init__(f"FFmpeg exited with code {returncode}"
                         + (f": {redact_secrets(last)[:200]}" if last else ""))


class MusicFFmpegAudio(discord.FFmpegPCMAudio):
    """FFmpegPCMAudio that records how the stream ended."""

    #: On EOF, how long to wait for FFmpeg's exit status / the rest of stderr.
    #: Runs on the audio thread at the very end of a track; normally instant.
    EXIT_WAIT = 1.0
    STDERR_WAIT = 0.3

    def __init__(self, source: str, *, before_options: str | None = None,
                 options: str | None = None, executable: str = "ffmpeg",
                 tail_bytes: int = 8192):
        self.stderr_tail = StderrTail(tail_bytes)
        self.returncode: int | None = None
        self.eof_at: float | None = None
        self.created_at = time.monotonic()
        # Defined here as well as by discord.py ≥2.6 so the error survives on
        # any 2.x version: the music cog reads it after the player stops.
        self._current_error: Exception | None = None
        super().__init__(source, executable=executable, before_options=before_options,
                         options=options, stderr=self.stderr_tail)

    def read(self) -> bytes:
        ret = self._stdout.read(FRAME_BYTES)
        if len(ret) == FRAME_BYTES:
            return ret
        self._on_eof()
        return b""

    def _on_eof(self) -> None:
        if self.eof_at is not None:
            return
        self.eof_at = time.monotonic()
        proc = getattr(self, "_process", None)
        if not proc or getattr(self, "_stopped", False):   # MISSING is falsy
            return
        try:
            rc = proc.wait(timeout=self.EXIT_WAIT)
        except subprocess.TimeoutExpired:
            rc = proc.poll()
        except Exception:
            rc = None
        self.returncode = rc
        self.stderr_tail.closed.wait(self.STDERR_WAIT)
        if rc not in (None, 0) and self._current_error is None:
            self._current_error = FFmpegExit(rc, self.stderr_tail.text())

    def stderr_text(self) -> str:
        return self.stderr_tail.text()


# ──────────────────────────────────────────────────────────────────────────────
# What went wrong, in words a user can act on
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FailureInfo:
    category: str
    retryable: bool        # could a freshly resolved stream URL help?
    reason: str            # "what happened", user-facing, no URLs
    advice: str            # "what to do", user-facing


_RULES: list[tuple[str, re.Pattern, bool, str, str]] = [
    ("forbidden", re.compile(r"\b403\b|forbidden|access denied"), True,
     "the site refused the audio stream (HTTP 403)",
     "That's usually the site blocking or throttling my host for a while — try "
     "again in a few minutes or pick a different upload. If it keeps happening, "
     "the bot owner can run `ytdiag <link>` to see why."),
    ("rate_limited", re.compile(r"\b429\b|too many requests"), True,
     "the site is rate-limiting my host (HTTP 429)",
     "Wait a few minutes and try again."),
    ("bad_options", re.compile(
        r"option .* not found|unrecognized option|error splitting the argument list|"
        r"no such filter|error initializing filter|error reinitializing filters"), False,
     "FFmpeg rejected my audio settings (a filter or option)",
     "Turn filters off with `filter off` and try again; the bot owner should "
     "check the log."),
    ("gone", re.compile(r"\b(404|410)\b"), True,
     "the audio link had expired or was removed",
     "Try the request again, or a different upload."),
    ("server_error", re.compile(r"server returned 5\d\d|\b5\d\d (internal|bad gateway|service)"), True,
     "the site's server returned an error",
     "Try again shortly."),
    ("network", re.compile(
        r"connection (reset|refused|timed out|aborted)|timed out|network is unreachable|"
        r"error in the pull function|i/o error|\bio error\b|stream ends prematurely|"
        r"will reconnect|broken pipe|failed to resolve hostname|name or service not known|"
        r"temporary failure in name resolution|\btls\b|\bssl\b"), True,
     "the connection to the audio source dropped",
     "Check that my host's internet connection is OK, then try again."),
    ("no_audio", re.compile(
        r"does not contain any stream|matches no streams|output file is empty|"
        r"no audio stream"), False,
     "the source has no audio track I can play",
     "Try a different link."),
    ("invalid_data", re.compile(
        r"invalid data found|could not find codec parameters|moov atom not found|"
        r"failed to read frame|error while decoding|invalid frame"), True,
     "the stream didn't contain playable audio",
     "Try a different upload or link."),
    ("missing_file", re.compile(r"no such file or directory"), False,
     "the file to play is missing",
     "Upload or link it again."),
]


def classify_failure(error: BaseException | None, stderr: str = "") -> FailureInfo:
    """Map an FFmpeg failure to a FailureInfo. Unknown failures are treated as
    retryable once: one refreshed URL is cheap, and an empty first read with no
    detail is exactly the case that used to vanish."""
    # Classify the redacted text: a signed URL's query string must not be
    # able to look like a status code.
    hay = redact_secrets(f"{stderr}\n{error or ''}").lower()
    for category, pattern, retryable, reason, advice in _RULES:
        if pattern.search(hay):
            return FailureInfo(category, retryable, reason, advice)
    if error is not None:
        return FailureInfo("ffmpeg_error", True,
                           "FFmpeg stopped with an error",
                           "Try again; if it keeps happening, the bot owner can "
                           "look for “Playback failed” in my log.")
    return FailureInfo("no_audio_produced", True,
                       "the stream ended before any sound came through",
                       "Try again; if it keeps happening, the bot owner can look "
                       "for “Playback failed” in my log.")
