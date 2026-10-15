"""
Logging configuration. Call `setup_logging()` once at startup.

Logs go to stderr with a clean format. Level is read from settings.LOG_LEVEL.
discord.py and yt-dlp loggers are pinned to WARNING by default to keep
the console quiet — bump them via the env var if you need to debug.
"""
from __future__ import annotations

import logging
import sys

from .config import settings


def setup_logging() -> None:
    """Configure root + library loggers. Idempotent."""
    root = logging.getLogger()
    if root.handlers:                       # already configured
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    root.addHandler(handler)

    # Pin chatty libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("spotipy").setLevel(logging.WARNING)
    # voice-recv emits a torrent of "packets were lost being flushed" WARNINGs
    # whenever anyone talks. It used to be pinned to ERROR, which also hid real
    # packet-loss problems. Now: at most one line per message type per minute,
    # carrying a count of what was suppressed. The filter sits on the logger,
    # so suppressed records never reach a handler (or its lock — the old
    # suspect for the transcribe-while-talking hang).
    vr = logging.getLogger("discord.ext.voice_recv")
    vr.setLevel(logging.WARNING)
    vr.addFilter(RateLimitFilter(60.0))


class RateLimitFilter(logging.Filter):
    """Pass each distinct message template at most once per `interval`
    seconds; the next one that passes says how many were suppressed."""

    def __init__(self, interval: float = 60.0):
        super().__init__()
        self.interval = interval
        self._last: dict[tuple[str, str], float] = {}
        self._suppressed: dict[tuple[str, str], int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        import time
        key = (record.name, str(record.msg)[:120])
        now = time.monotonic()
        last = self._last.get(key)
        if last is not None and now - last < self.interval:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False
        self._last[key] = now
        n = self._suppressed.pop(key, 0)
        if n:
            record.msg = f"{record.msg} (+{n} similar in the last {int(self.interval)}s)"
        return True


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor. Modules use `log = get_logger(__name__)`."""
    return logging.getLogger(name)
