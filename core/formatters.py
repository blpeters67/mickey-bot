"""
Pure utility functions. No Discord, no I/O, no global state — everything here
should be unit-testable in isolation. The matching test file is tests/test_formatters.py.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict, deque


# ──────────────────────────────────────────────────────────────────────────────
# Duration / time formatting
# ──────────────────────────────────────────────────────────────────────────────

def fmt_duration(seconds: int | float | None) -> str:
    """0 / None → '??:??', else 'm:ss' or 'h:mm:ss'."""
    if not seconds:
        return "??:??"
    s     = int(seconds)
    h, r  = divmod(s, 3600)
    m, ss = divmod(r, 60)
    return f"{h}:{m:02d}:{ss:02d}" if h else f"{m}:{ss:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# File size — parse / format. Used by the -resize command and upload routing.
# ──────────────────────────────────────────────────────────────────────────────

# Matches: "5", "5g", "5gb", "5GB", "5 GB", "0.5g", "500m", "500mb", "1.5tb".
# A bare number defaults to MB since that's the natural unit for video sizes.
_SIZE_RE = re.compile(r"""
    ^\s*
    (\d+(?:\.\d+)?)        # number (group 1)
    \s*
    ([kmgt])?              # optional magnitude (group 2): k/m/g/t
    [bB]?                  # optional trailing 'b' — purely decorative
    \s*$
""", re.VERBOSE | re.IGNORECASE)

_SIZE_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def parse_size(text: str) -> int | None:
    """Parse '5g', '5gb', '500mb', '500m', '500', '0.5g' → bytes, or None on error.

    A bare number is treated as megabytes — `-resize url 500` means 500 MB.
    Multipliers are binary (1 KB = 1024 B), to match how disks and Discord report sizes.
    """
    m = _SIZE_RE.match(text or "")
    if not m:
        return None
    num  = float(m.group(1))
    unit = (m.group(2) or "m").lower()
    return int(num * _SIZE_UNITS[unit])


def fmt_size(n: int | float) -> str:
    """1500 → '1.46 KB', 1.5GB → '1.50 GB'. Uses binary units (1 KB = 1024 B)."""
    n = float(n)
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024
        if n < 1024:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"


# ──────────────────────────────────────────────────────────────────────────────
# Seek parsing — supports h:m:s, m:s, plain seconds, +/- relative
# ──────────────────────────────────────────────────────────────────────────────

def parse_seek(text: str, current_pos: float) -> float | None:
    """
    Returns the absolute target position in seconds, or None on parse error.

    Accepts: '1:30', '1:23:45', '90', '+30', '-15', '0:05'.
    Original code only handled m:s — long tracks couldn't be seeked past 60:00.
    """
    text = text.strip()
    if not text:
        return None

    sign = 0
    if text[0] in "+-":
        sign = 1 if text[0] == "+" else -1
        text = text[1:].strip()

    try:
        if ":" in text:
            parts = [float(p) for p in text.split(":")]
            if len(parts) == 2:           # m:s
                value = parts[0] * 60 + parts[1]
            elif len(parts) == 3:         # h:m:s
                value = parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                return None
        else:
            value = float(text)
    except ValueError:
        return None

    if sign:
        return max(0.0, current_pos + sign * value)
    return max(0.0, value)


# ──────────────────────────────────────────────────────────────────────────────
# Index range parsing — '3', '1,3,5', '2-5', '1,3,5-8'
# ──────────────────────────────────────────────────────────────────────────────

def parse_indices(text: str) -> set[int] | None:
    """1-based indices. Returns None on parse error."""
    out: set[int] = set()
    try:
        for chunk in text.replace(" ", "").split(","):
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-", 1)
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                out.add(int(chunk))
    except ValueError:
        return None
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Queue manipulation
# ──────────────────────────────────────────────────────────────────────────────

def interleave_queue(
    tracks, *, shuffled: bool = False,
) -> deque:
    """
    Round-robin tracks across requesters: A A A B B → A B A B A.
    If `shuffled`, each requester's bucket is shuffled first so balance and
    shuffle compose instead of fighting.
    """
    buckets: dict = defaultdict(list)
    for t in tracks:
        buckets[t.get("requester")].append(t)
    if shuffled:
        for r in buckets:
            random.shuffle(buckets[r])

    requesters = list(buckets.keys())
    out: list = []
    i = 0
    while any(buckets[r] for r in requesters):
        r = requesters[i % len(requesters)]
        if buckets[r]:
            out.append(buckets[r].pop(0))
        i += 1
    return deque(out)


# ──────────────────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────────────────

def unwrap_url(text: str | None) -> str:
    """Strip one layer of `<...>` wrapping from a URL-ish string.

    Discord users wrap links in angle brackets to suppress the embed preview;
    commands should treat `<https://x>` exactly like `https://x`. Only strips
    when the string is wrapped end-to-end, so comparison text like `a < b`
    is left alone.
    """
    t = (text or "").strip()
    if len(t) >= 2 and t[0] == "<" and t[-1] == ">":
        inner = t[1:-1].strip()
        if inner and "<" not in inner and ">" not in inner:
            return inner
    return t

def make_progress_bar(elapsed: float, duration, *, width: int = 20) -> str:
    """`██████░░░░░░░░░░░░░░` — graceful when duration is 0/missing."""
    if not duration or duration <= 0:
        return "░" * width
    pct    = max(0.0, min(1.0, elapsed / duration))
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def parse_duration(s: str) -> int | None:
    """Parse a human-friendly duration into seconds.

    Accepted forms:
        - "0", "off", "never", "none", "disabled"      → 0 (sentinel for "disabled")
        - "30", "30s", "30sec", "30secs"               → 30
        - "5m", "5min", "5mins"                        → 300
        - "1h", "2hr", "3hours"                        → 3600 / 7200 / 10800
        - "2d", "1day", "3days"                        → 172800 / 86400 / 259200
        - "1h30m", "2d12h", "2h15m30s"                 → compound; sum of parts

    Returns None if the input doesn't parse — the caller can show an
    actionable error.
    """
    if s is None:
        return None
    s = s.strip().lower().replace(" ", "")
    if not s:
        return None
    if s in ("0", "off", "never", "none", "disabled", "no"):
        return 0

    # Bare integer → seconds. Common case for power users.
    if s.isdigit():
        return int(s)

    # Compound like "1h30m". Walk: digits + unit, repeat. Reject anything else.
    # Alternatives are ordered longest-first so 'sec' beats 's' on "30sec".
    import re
    total = 0
    matched_any = False
    pattern = re.compile(
        r"(\d+)\s*(seconds|secs|sec|minutes|mins|min|hours|hour|hrs|hr|"
        r"days|day|s|m|h|d)"
    )
    pos = 0
    while pos < len(s):
        m = pattern.match(s, pos)
        if not m:
            return None
        n    = int(m.group(1))
        unit = m.group(2)
        if   unit.startswith("s"): total += n
        elif unit.startswith("m"): total += n * 60
        elif unit.startswith("h"): total += n * 3600
        elif unit.startswith("d"): total += n * 86400
        pos = m.end()
        matched_any = True
    return total if matched_any else None


def fmt_duration_long(seconds: int) -> str:
    """Format a duration as `2d 3h` / `1h 30m 5s` / `45m` / `30s` / `off`.

    Companion to parse_duration above — used to display the current config
    in a human-readable way. parse(fmt(x)) round-trips for any non-negative x.
    """
    if seconds <= 0:
        return "off"
    parts = []
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s: parts.append(f"{s}s")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics hygiene
# ──────────────────────────────────────────────────────────────────────────────

_URL_IN_TEXT = re.compile(r"(?i)\b(?:https?|wss?|rtmps?|rtsp|mms)://[^\s'\"<>()\[\]]+")
_SECRET_HEADER = re.compile(
    r"(?im)^(\s*(?:cookie|set-cookie|authorization|proxy-authorization|x-goog-[\w-]+)"
    r"\s*:\s*).*$")
_SECRET_PARAM = re.compile(
    r"(?i)\b((?:l?sig|signature|token|access_token|key|api_key|password|passwd|auth|pot)=)"
    r"[^\s&'\"]+")
# API keys that turn up in error text (OpenRouter/OpenAI style) and bearer tokens.
_SECRET_TOKEN = re.compile(r"\b(sk-(?:or-v1-|proj-)?)[A-Za-z0-9_-]{16,}")
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")


def redact_secrets(text: str | None) -> str:
    """Make diagnostic text safe to log or show: every URL is cut down to its
    scheme and host (signed stream URLs carry IP-bound signatures, tokens and
    expiry in the path/query), credential headers lose their values, and
    stray `sig=…`/`token=…` parameters are blanked."""
    if not text:
        return ""

    def _url(m: re.Match) -> str:
        url = m.group(0)
        scheme, _, rest = url.partition("://")
        host = re.split(r"[/?#]", rest, maxsplit=1)[0].rsplit("@", 1)[-1]
        return f"{scheme.lower()}://{host}/…"

    text = _URL_IN_TEXT.sub(_url, text)
    text = _SECRET_HEADER.sub(lambda m: m.group(1) + "<redacted>", text)
    text = _SECRET_TOKEN.sub(lambda m: m.group(1) + "<redacted>", text)
    text = _BEARER.sub(lambda m: m.group(1) + "<redacted>", text)
    return _SECRET_PARAM.sub(lambda m: m.group(1) + "<redacted>", text)


def chunk_lines(text: str, limit: int = 1900) -> list[str]:
    """Split text into pieces of at most `limit` characters, breaking between
    lines where possible (an over-long line is hard-split). Discord rejects
    message content over 2000 characters, so diagnostics go out in pieces."""
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{cur}\n{line}" if cur else line
        if len(candidate) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks
