"""
Lyrics matching and time-alignment.

The old `lyrics` command asked lrclib for one result and trusted it. That's
where the "lyrics are out of sync" complaints came from, and the causes turned
out to be four different bugs wearing one coat:

  1. **Wrong version matched.** A YouTube upload that's a live take, a remix,
     an extended mix, or has a 20-second intro card gets the *studio* LRC.
     Every timestamp is then wrong by a growing amount and no offset fixes it.
     The old `_clean_title` actively caused this: it stripped "extended
     version" and "full version" from the title before searching, which is the
     one piece of text that says "do not match me to the album cut".
  2. **Front padding.** Rips routinely carry 0.5–4 s of digital silence, a
     channel sting, or a countdown before the music. The LRC assumes t=0 is
     the first sample of the master.
  3. **Tempo-shifted uploads.** Re-uploads are often nudged 1–6% faster to
     dodge Content ID. The error is proportional to elapsed time, so the first
     verse looks fine and the last chorus is seconds out — the classic
     "it starts right and drifts" report.
  4. **A lying playback clock.** Fixed separately, in core/playback_clock.py.

This module handles 1–3. The strategy is: gather several candidates, score
them against the *actual* stream duration, refuse to claim sync when nothing
plausibly matches, then solve for an offset and a tempo scale before handing
timestamps to the renderer.

Nothing here imports discord, so it's directly unit-testable.
"""
from __future__ import annotations

import asyncio
import difflib
import re
import shutil
import time
import urllib.parse
from dataclasses import dataclass, field

import aiohttp

from core.logging_setup import get_logger

log = get_logger(__name__)

__all__ = [
    "LyricLine", "SyncedLyrics", "Alignment", "TrackQuery",
    "parse_lrc", "build_query", "fetch_lyrics", "align",
    "detect_leading_silence", "clear_cache",
]

LRCLIB_BASE = "https://lrclib.net/api"
USER_AGENT = "DiscordMusicBot/2.0 (+lyrics-sync)"
HTTP_TIMEOUT = 10.0

# ── matching thresholds ───────────────────────────────────────────────────────

#: Duration difference at which a candidate is a near-certain match.
DUR_EXACT = 2.0
#: Beyond this, the candidate is almost certainly a different recording.
DUR_REJECT = 18.0
#: Ratio band outside which we refuse to claim synced playback at all.
RATIO_MIN, RATIO_MAX = 0.88, 1.14
#: Below this relative difference, assume identical tempo.
TEMPO_MIN_DEV = 0.015
#: Above this, it isn't a tempo shift, it's a different arrangement.
TEMPO_MAX_DEV = 0.10
#: Leading silence above this means the length difference is front padding,
#: not tempo — measured, so it wins over the ratio heuristic.
LEAD_SIGNIFICANT = 2.0
#: Minimum score for "we're confident enough to run the karaoke display".
CONFIDENCE_FLOOR = 0.55

# ── title cleaning ────────────────────────────────────────────────────────────

# Bracketed junk that is safe to strip: it describes the *upload*, not the
# recording. Deliberately excludes live / remix / extended / acoustic / sped
# up / cover — those change the timing and must reach the matcher so the
# duration check can do its job.
_NOISE_BRACKET = re.compile(
    r"\s*[\(\[]\s*(?:"
    r"official\s*(?:music\s*)?(?:video|audio|lyric\s*video|lyrics|visuali[sz]er|hd|4k)?|"
    r"lyrics?(?:\s*video)?|"
    # "Remastered", "2011 Remaster", "Remastered in 4K", "Remaster 2009" — the
    # trailing wildcard matters: without it "[Remastered in 4K]" survived into
    # the search string and stopped the track matching at all.
    r"(?:\d{4}\s*)?remaster(?:ed)?[^\)\]]*|"
    r"hd|hq|4k|8k|full\s*hd|"
    r"explicit|clean|dirty|"
    r"audio|visuali[sz]er|mv|m/v|"
    r"free\s*download|download|out\s*now|"
    r"colou?r\s*coded(?:\s*lyrics)?(?:\s*\w+)?"
    r")\s*[\)\]]",
    re.IGNORECASE,
)
_TRAILING_JUNK = re.compile(
    r"\s*[\|\-–—]\s*(?:official\s*(?:music\s*)?video|official\s*audio|"
    r"lyrics?(?:\s*video)?|visuali[sz]er|hd|4k)\s*$",
    re.IGNORECASE,
)
# Word boundaries are load-bearing here. Without them "ft" matched inside
# "Daft Punk" and the artist came through as "Da", which of course matched
# nothing. "with" is deliberately only honoured inside brackets — plenty of
# song titles legitimately contain the word ("Dancing With Myself").
_FEAT = re.compile(
    r"\s*[\(\[]\s*(?:feat|ft|featuring|with)\b\.?\s+[^\)\]]*[\)\]]"
    r"|"
    r"\s+(?<![\w])(?:feat|ft|featuring)\b\.?\s+.*$",
    re.IGNORECASE,
)
_TOPIC = re.compile(r"\s*-\s*topic\s*$", re.IGNORECASE)
_DASHES = re.compile(r"\s+[-–—]\s+")

#: Markers that mean "this is not the album cut". If the title has one and the
#: candidate's title does not (or vice versa), the match is penalised hard.
_VARIANT_MARKERS = (
    "live", "remix", "acoustic", "instrumental", "extended", "radio edit",
    "sped up", "slowed", "nightcore", "reverb", "cover", "demo", "unplugged",
    "karaoke", "mashup", "bootleg", "vip mix", "orchestral", "piano",
    "8d", "reprise", "session", "version",
)


def _norm(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    s = (s or "").lower()
    s = re.sub(r"['’`]", "", s)
    s = re.sub(r"[^\w\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_title(text: str) -> str:
    """Strip upload-describing noise while preserving version markers."""
    out = _NOISE_BRACKET.sub(" ", text or "")
    out = _TRAILING_JUNK.sub("", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip(" |-–—_·")


def _variants(text: str) -> set[str]:
    n = _norm(text)
    return {m for m in _VARIANT_MARKERS if m in n}


# ── data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LyricLine:
    time: float
    text: str


@dataclass
class TrackQuery:
    """What we're looking for, extracted from the playing track."""
    title: str
    artist: str = ""
    album: str = ""
    duration: float = 0.0
    #: Stable identity for caching learned offsets — the watch URL.
    source_id: str = ""
    #: Direct media URL, used for silence probing when available.
    stream_url: str = ""


@dataclass
class SyncedLyrics:
    """A matched lyrics result, before alignment."""
    lines: list[LyricLine]
    plain: str
    track_name: str
    artist_name: str
    #: Duration of the recording the LRC was authored against.
    lrc_duration: float
    #: 0..1 — how sure we are this is the same recording.
    confidence: float
    #: Human-readable reason when confidence is low.
    note: str = ""

    @property
    def has_sync(self) -> bool:
        return bool(self.lines)


@dataclass
class Alignment:
    """The transform from LRC time to playback time.

    playback_time = lrc_time * scale + offset
    """
    offset: float = 0.0
    scale: float = 1.0
    #: How the numbers were arrived at, surfaced in the UI so a wrong guess is
    #: diagnosable instead of mysterious.
    reasons: list[str] = field(default_factory=list)
    #: True when a human-supplied offset was used and must not be overridden.
    learned: bool = False

    def apply(self, lines: list[LyricLine]) -> list[LyricLine]:
        return [LyricLine(ln.time * self.scale + self.offset, ln.text)
                for ln in lines]

    def describe(self) -> str:
        bits = []
        if abs(self.offset) >= 0.05:
            bits.append(f"{self.offset:+.2f}s")
        if abs(self.scale - 1.0) >= 0.001:
            bits.append(f"×{self.scale:.4f}")
        return " ".join(bits) or "no correction"


# ── LRC parsing ───────────────────────────────────────────────────────────────

_TS = re.compile(r"\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]")
_META = re.compile(r"^\[([a-z_]+):(.*)\]$", re.IGNORECASE)


def parse_lrc(text: str) -> tuple[list[LyricLine], float]:
    """Parse LRC into sorted lines plus the file's own offset tag (seconds).

    Handles the parts of the format the old five-line regex didn't:
    multiple timestamps on one line (`[00:12.00][01:30.00]chorus`), `[mm:ss]`
    with no decimals, `[mm:ss:cc]` colon-separated centiseconds, metadata tags,
    and the `[offset:]` tag — which by LRC convention is milliseconds where a
    *positive* value makes lyrics appear *earlier*.
    """
    lines: list[LyricLine] = []
    file_offset = 0.0
    for raw in (text or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        meta = _META.match(raw)
        if meta and meta.group(1).lower() == "offset":
            try:
                file_offset = float(meta.group(2).strip()) / 1000.0
            except (TypeError, ValueError):
                pass
            continue
        stamps = list(_TS.finditer(raw))
        if not stamps:
            continue
        body = raw[stamps[-1].end():].strip()
        for m in stamps:
            mins = int(m.group(1))
            sec_part = m.group(2)
            if ":" in sec_part:            # [mm:ss:cc]
                whole, frac = sec_part.split(":", 1)
                secs = int(whole) + int(frac.ljust(2, "0")[:2]) / 100.0
            else:
                secs = float(sec_part)
            lines.append(LyricLine(mins * 60 + secs, body))
    lines.sort(key=lambda ln: ln.time)
    return lines, file_offset


# ── query building ────────────────────────────────────────────────────────────

def build_query(track: dict) -> TrackQuery:
    """Pull the best available artist/title out of a resolved track dict.

    yt-dlp gives structured `artist`/`track` fields for anything that came from
    a music service or a "- Topic" channel; those are far more reliable than
    splitting a title on a hyphen. We use them when present and only fall back
    to string surgery otherwise.
    """
    raw_title = (track.get("title") or "").strip()
    artist = (track.get("artist") or track.get("creator") or "").strip()
    title = (track.get("track") or "").strip()
    album = (track.get("album") or "").strip()

    if not artist:
        uploader = (track.get("uploader") or track.get("channel") or "").strip()
        if _TOPIC.search(uploader):
            artist = _TOPIC.sub("", uploader).strip()

    if not title:
        cleaned = clean_title(raw_title)
        parts = _DASHES.split(cleaned, maxsplit=1)
        if len(parts) == 2 and all(p.strip() for p in parts):
            left, right = parts[0].strip(), parts[1].strip()
            # "Artist - Title" is overwhelmingly the convention; only flip when
            # the uploader name matches the right-hand side.
            if artist and _norm(artist) == _norm(right):
                title, artist = left, right
            else:
                artist = artist or left
                title = right
        else:
            title = cleaned

    # Featured artists confuse lrclib's matcher more often than they help.
    title = _FEAT.sub("", title).strip(" -–—")
    artist = _FEAT.sub("", artist).strip(" -–—")

    return TrackQuery(
        title=title or raw_title,
        artist=artist,
        album=album,
        duration=float(track.get("duration") or 0.0),
        source_id=(track.get("webpage_url") or "").strip(),
        stream_url=(track.get("stream_url") or "").strip(),
    )


# ── fetching ──────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, SyncedLyrics | None]] = {}
_CACHE_TTL = 3600.0


def clear_cache() -> None:
    _cache.clear()


def _score(cand: dict, q: TrackQuery) -> tuple[float, str]:
    """Score a candidate 0..1 against the query. Returns (score, note)."""
    c_title = cand.get("trackName") or ""
    c_artist = cand.get("artistName") or ""
    c_dur = float(cand.get("duration") or 0.0)

    title_sim = difflib.SequenceMatcher(
        None, _norm(q.title), _norm(c_title)).ratio()
    artist_sim = difflib.SequenceMatcher(
        None, _norm(q.artist), _norm(c_artist)).ratio() if q.artist else 0.5

    notes = []
    # Duration is the strongest signal we have: two recordings of the same song
    # that differ by more than a few seconds are different performances.
    if q.duration and c_dur:
        delta = abs(c_dur - q.duration)
        if delta <= DUR_EXACT:
            dur_score = 1.0
        elif delta >= DUR_REJECT:
            dur_score = 0.0
            notes.append(f"length differs by {delta:.0f}s")
        else:
            dur_score = 1.0 - (delta - DUR_EXACT) / (DUR_REJECT - DUR_EXACT)
    else:
        dur_score = 0.4          # unknown: neither credit nor blame
        notes.append("track length unknown")

    score = 0.50 * dur_score + 0.30 * title_sim + 0.20 * artist_sim

    # A version mismatch ("live" on one side only) is disqualifying even when
    # the words and the length happen to line up.
    mine, theirs = _variants(f"{q.title} {q.album}"), _variants(c_title)
    if mine ^ theirs:
        score *= 0.55
        diff = ", ".join(sorted(mine ^ theirs))
        notes.append(f"version mismatch ({diff})")

    if cand.get("instrumental"):
        score *= 0.3
        notes.append("marked instrumental")
    if not (cand.get("syncedLyrics") or "").strip():
        score *= 0.75            # usable, but can't drive karaoke
    return min(1.0, score), "; ".join(notes)


async def _get_json(sess: aiohttp.ClientSession, path: str, params: dict):
    url = f"{LRCLIB_BASE}/{path}?" + urllib.parse.urlencode(
        {k: v for k, v in params.items() if v not in ("", None, 0)})
    try:
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
                            headers={"User-Agent": USER_AGENT}) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None
    except Exception:
        log.exception("lrclib request failed: %s", url)
        return None


async def fetch_lyrics(q: TrackQuery, *, session: aiohttp.ClientSession | None = None
                       ) -> SyncedLyrics | None:
    """Find the best lyrics for `q`, or None.

    Tries the precise endpoint first (lrclib's `/get` requires the duration to
    be within a couple of seconds, so a hit there is a strong match), then
    falls back to progressively looser searches. Every candidate from every
    strategy is pooled and scored together rather than taking the first hit —
    the first hit is frequently a karaoke or cover upload.
    """
    key = f"{_norm(q.artist)}|{_norm(q.title)}|{int(q.duration)}"
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]

    own_session = session is None
    sess = session or aiohttp.ClientSession()
    candidates: list[dict] = []
    try:
        exact = await _get_json(sess, "get", {
            "track_name": q.title, "artist_name": q.artist,
            "album_name": q.album, "duration": int(q.duration) or None,
        })
        if isinstance(exact, dict):
            candidates.append(exact)

        searches = [
            {"track_name": q.title, "artist_name": q.artist},
            {"q": f"{q.artist} {q.title}".strip()},
        ]
        # Last resort: the bare title. Only worth it when we have a duration to
        # disambiguate with, otherwise it returns noise.
        if q.duration:
            searches.append({"q": q.title})

        for params in searches:
            if any(c.get("syncedLyrics") for c in candidates) and len(candidates) > 3:
                break
            found = await _get_json(sess, "search", params)
            if isinstance(found, list):
                candidates.extend(c for c in found if isinstance(c, dict))
    finally:
        if own_session:
            await sess.close()

    if not candidates:
        _cache[key] = (time.time(), None)
        return None

    # Deduplicate by lrclib id, keeping first occurrence.
    seen, unique = set(), []
    for c in candidates:
        cid = c.get("id") or (c.get("trackName"), c.get("artistName"), c.get("duration"))
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(c)

    scored = sorted(((_score(c, q), c) for c in unique),
                    key=lambda pair: pair[0][0], reverse=True)
    # Prefer the best *synced* candidate; fall back to the best of any kind.
    best = next(((s, c) for s, c in scored if (c.get("syncedLyrics") or "").strip()),
                scored[0])
    (score, note), cand = best

    lines, file_offset = parse_lrc(cand.get("syncedLyrics") or "")
    if file_offset:
        lines = [LyricLine(ln.time - file_offset, ln.text) for ln in lines]

    result = SyncedLyrics(
        lines=lines,
        plain=(cand.get("plainLyrics") or "").strip(),
        track_name=cand.get("trackName") or q.title,
        artist_name=cand.get("artistName") or q.artist,
        lrc_duration=float(cand.get("duration") or 0.0),
        confidence=score,
        note=note,
    )
    _cache[key] = (time.time(), result)
    log.info("lyrics match %r — %r by %r (score %.2f%s)",
             q.title, result.track_name, result.artist_name, score,
             f"; {note}" if note else "")
    return result


# ── silence probing ───────────────────────────────────────────────────────────

_SIL_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SIL_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")

_silence_cache: dict[str, float] = {}


async def detect_leading_silence(stream_url: str, *, probe_seconds: int = 40,
                                 cache_key: str = "") -> float | None:
    """Seconds of near-silence at the head of the stream, or None if unknown.

    This is the measurement that fixes rips with an intro card, a channel
    sting, or plain digital silence before the first sample — cases where no
    amount of metadata reasoning can tell you the audio starts late.

    Runs ffmpeg over only the first `probe_seconds`, discarding output, so it
    costs a fraction of a second of CPU and no disk.
    """
    if not stream_url:
        return None
    key = cache_key or stream_url
    if key in _silence_cache:
        return _silence_cache[key]
    if not shutil.which("ffmpeg"):
        return None

    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "3",
        "-t", str(probe_seconds), "-i", stream_url,
        "-af", "silencedetect=noise=-42dB:d=0.3",
        "-f", "null", "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=25)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None
    except (OSError, ValueError):
        log.debug("silencedetect could not start", exc_info=True)
        return None

    text = (stderr or b"").decode("utf-8", "replace")
    starts = [float(m) for m in _SIL_START.findall(text)]
    ends = [float(m) for m in _SIL_END.findall(text)]
    lead = 0.0
    # Only a silence that begins at the very start of the file is leading
    # silence; a gap in the middle of the track is just a gap.
    if starts and ends and starts[0] <= 0.15:
        lead = max(0.0, ends[0])
    if lead > probe_seconds * 0.75:
        return None                       # probably a silent/failed probe
    _silence_cache[key] = lead
    return lead


# ── alignment ─────────────────────────────────────────────────────────────────

def align(lyr: SyncedLyrics, q: TrackQuery, *,
          leading_silence: float | None = None,
          learned_offset: float | None = None) -> Alignment:
    """Solve for the offset and tempo scale that map LRC time to playback time.

    Priority:
      1. A **learned offset** — someone already corrected this exact upload, so
         trust them over any heuristic.
      2. **Measured leading silence** — an observation beats an inference.
      3. **Duration reasoning** — decide whether the length difference between
         the LRC's recording and this upload is padding (a constant offset) or
         tempo (a multiplicative scale).

    The padding-vs-tempo call is genuinely ambiguous from metadata alone, so it
    is made explicitly and reported in `reasons` rather than buried. A wrong
    call costs one `lyrics offset` nudge, which then becomes case 1 forever.
    """
    a = Alignment()
    d_act, d_lrc = q.duration or 0.0, lyr.lrc_duration or 0.0

    if learned_offset is not None:
        a.offset = float(learned_offset)
        a.learned = True
        a.reasons.append(f"using saved correction {learned_offset:+.2f}s")
        return a

    if leading_silence and leading_silence > 0.2:
        a.offset += leading_silence
        a.reasons.append(f"{leading_silence:.2f}s of silence before the audio starts")

    if d_act > 0 and d_lrc > 0:
        ratio = d_act / d_lrc
        dev = abs(ratio - 1.0)
        excess = d_act - d_lrc
        measured_lead = leading_silence or 0.0

        if TEMPO_MIN_DEV <= dev <= TEMPO_MAX_DEV and measured_lead < LEAD_SIGNIFICANT:
            # Proportional difference with no front padding to explain it —
            # this upload runs at a different speed than the reference.
            a.scale = ratio
            pct = (ratio - 1.0) * 100.0
            a.reasons.append(
                f"upload runs {abs(pct):.1f}% {'slower' if pct > 0 else 'faster'} "
                f"than the reference recording"
            )
        elif dev > TEMPO_MAX_DEV:
            a.reasons.append(
                f"length differs by {excess:+.0f}s — likely a different "
                f"arrangement, timing may not line up"
            )
        elif abs(excess) > DUR_EXACT and measured_lead < LEAD_SIGNIFICANT:
            a.reasons.append(
                f"{abs(excess):.0f}s of extra {'audio' if excess > 0 else 'trim'} "
                f"assumed to be at the end"
            )

    if not a.reasons:
        a.reasons.append("timings used as published")
    return a


def confidence_ok(lyr: SyncedLyrics, q: TrackQuery) -> bool:
    """Whether we should run the karaoke display rather than a static dump."""
    if not lyr.has_sync:
        return False
    if lyr.confidence < CONFIDENCE_FLOOR:
        return False
    if q.duration and lyr.lrc_duration:
        ratio = q.duration / lyr.lrc_duration
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            return False
    return True
