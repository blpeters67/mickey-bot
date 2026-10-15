"""
Text-to-speech engine for `-tts` — realistic neural voices via edge-tts.

edge-tts talks to the same service as Microsoft Edge's "Read aloud", so it gets
Microsoft's neural voices (Andrew, Ava, Emma, Sonia, …): about as natural as
TTS gets short of a paid API, free, and no key to manage. The catch is that
it's online — a non-issue for a bot that already streams from YouTube.

Pipeline:

    text ─ clean_for_speech() ─ synthesize() ─ to_pcm() ─ core.voice_mix
           mentions → names,     mp3 from      48 kHz stereo s16le,
           links → "a YouTube    Microsoft     loudness-normalised so it
           link", idk → "I                     sits level with the music
           don't know", …

Nothing here imports discord — it's all unit-testable.
"""
from __future__ import annotations

import asyncio
import importlib.util
import re
import time
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

from .config import settings
from .logging_setup import get_logger

log = get_logger(__name__)


class TTSError(Exception):
    """A failure worth showing the user; str(exc) is the explanation."""


class NoSpeech(TTSError):
    """The service answered but sent no audio — a retired voice, or text with
    nothing pronounceable in it."""


def available() -> bool:
    """True if edge-tts is installed (it's in requirements.txt)."""
    return importlib.util.find_spec("edge_tts") is not None


# ──────────────────────────────────────────────────────────────────────────────
# Voices
# ──────────────────────────────────────────────────────────────────────────────

#: The short list `-ttsvoice` shows: name → (voice id, blurb). Any other
#: Microsoft neural voice id works too (`fr-FR-DeniseNeural`, …) — these are
#: just the natural-sounding English ones worth knowing about. The four
#: "Multilingual" voices are the most lifelike of the lot, and they cope with a
#: stray foreign word or name instead of mangling it.
VOICES: dict[str, tuple[str, str]] = {
    "andrew":      ("en-US-AndrewMultilingualNeural", "US · warm, conversational"),
    "ava":         ("en-US-AvaMultilingualNeural",    "US · bright, expressive"),
    "brian":       ("en-US-BrianMultilingualNeural",  "US · laid-back"),
    "emma":        ("en-US-EmmaMultilingualNeural",   "US · cheerful, clear"),
    "aria":        ("en-US-AriaNeural",               "US · confident"),
    "jenny":       ("en-US-JennyNeural",              "US · friendly"),
    "guy":         ("en-US-GuyNeural",                "US · announcer"),
    "christopher": ("en-US-ChristopherNeural",        "US · deep, authoritative"),
    "eric":        ("en-US-EricNeural",               "US · calm"),
    "michelle":    ("en-US-MichelleNeural",           "US · pleasant"),
    "roger":       ("en-US-RogerNeural",              "US · lively"),
    "steffan":     ("en-US-SteffanNeural",            "US · measured"),
    "sonia":       ("en-GB-SoniaNeural",              "British · friendly"),
    "ryan":        ("en-GB-RyanNeural",               "British · easygoing"),
    "libby":       ("en-GB-LibbyNeural",              "British · light"),
    "thomas":      ("en-GB-ThomasNeural",             "British · crisp"),
    "connor":      ("en-IE-ConnorNeural",             "Irish"),
    "emily":       ("en-IE-EmilyNeural",              "Irish"),
    "natasha":     ("en-AU-NatashaNeural",            "Australian"),
    "clara":       ("en-CA-ClaraNeural",              "Canadian"),
    "liam":        ("en-CA-LiamNeural",               "Canadian"),
}

#: Used when TTS_VOICE is unset or unusable.
FALLBACK_VOICE = "en-US-AndrewMultilingualNeural"

_BY_ID = {vid.lower(): (nick, blurb) for nick, (vid, blurb) in VOICES.items()}

# edge-tts accepts `xx-YY-NameNeural` (case-sensitive — it validates with a
# regex before ever contacting the service).
_VOICE_ID = re.compile(r"[a-z]{2,3}-[A-Z][A-Za-z]{1,3}(?:-[A-Za-z]+)?-\w+Neural")

_catalog: dict[str, dict] | None = None
_catalog_at = 0.0
_CATALOG_TTL = 6 * 3600


async def fetch_voice_catalog() -> dict[str, dict] | None:
    """Microsoft's live voice list keyed by lower-cased id, or None if it can't
    be reached (a stale copy beats nothing). Cached for a few hours — it's a
    ~400-entry list that changes a few times a year."""
    global _catalog, _catalog_at
    if _catalog is not None and time.monotonic() - _catalog_at < _CATALOG_TTL:
        return _catalog
    try:
        import edge_tts
        voices = await asyncio.wait_for(edge_tts.list_voices(), timeout=10)
    except Exception as e:
        log.info("Couldn't fetch the TTS voice list: %s", e)
        return _catalog
    _catalog = {v["ShortName"].lower(): v for v in voices if v.get("ShortName")}
    _catalog_at = time.monotonic()
    return _catalog


def resolve_voice(name: str | None,
                  catalog: Mapping[str, Mapping] | None = None) -> str | None:
    """A user's voice choice → a voice id edge-tts accepts, or None.

    Takes a name from VOICES (`ava`), a full id in any case
    (`en-gb-sonianeural`), or — when the live catalog is available — a bare
    first name that's unique across every language (`denise` →
    fr-FR-DeniseNeural). With a catalog, anything Microsoft no longer serves is
    rejected; offline, curated names and correctly-cased ids are trusted.
    """
    raw = (name or "").strip().strip("`'\"")
    key = raw.lower()
    if not key:
        return None
    if key in VOICES:
        vid = VOICES[key][0]
        return vid if catalog is None or vid.lower() in catalog else None
    if catalog is not None:
        hit = catalog.get(key) or catalog.get(key + "neural")
        if hit:
            return hit["ShortName"]
        # First names: "denise" → fr-FR-DeniseNeural, if it's unambiguous.
        matches = [v["ShortName"] for v in catalog.values()
                   if v["ShortName"].rsplit("-", 1)[-1].lower() in (key, key + "neural")]
        return matches[0] if len(matches) == 1 else None
    if key in _BY_ID:
        return next(vid for vid, _ in VOICES.values() if vid.lower() == key)
    return raw if _VOICE_ID.fullmatch(raw) else None


def default_voice() -> str:
    """The voice for anyone who hasn't picked one (TTS_VOICE in .env)."""
    return resolve_voice(settings.tts_voice) or FALLBACK_VOICE


def voice_label(voice_id: str, catalog: Mapping[str, Mapping] | None = None) -> str:
    """Human name for a voice: 'Ava (US · bright, expressive)'."""
    nick = _BY_ID.get(voice_id.lower())
    if nick:
        return f"{nick[0].capitalize()} ({nick[1]})"
    info = (catalog or {}).get(voice_id.lower())
    if info:
        return f"{voice_id} ({info.get('Gender', '?').lower()}, {info.get('Locale', '?')})"
    return voice_id


def curated_voices(catalog: Mapping[str, Mapping] | None = None) -> list[tuple[str, str, str]]:
    """(name, id, blurb) for the short list, minus any voice Microsoft has
    since retired (when the live catalog says so)."""
    return [(nick, vid, blurb) for nick, (vid, blurb) in VOICES.items()
            if catalog is None or vid.lower() in catalog]


# ──────────────────────────────────────────────────────────────────────────────
# Text cleanup — what a person would actually say out loud
# ──────────────────────────────────────────────────────────────────────────────

_SNOWFLAKE = r"\d{15,25}"
_CODE_BLOCK   = re.compile(r"```.*?```", re.S)
_SPOILER      = re.compile(r"\|\|.+?\|\|", re.S)
_MASKED_LINK  = re.compile(r"\[([^\]\n]+)\]\(\s*<?https?://[^)\s>]+>?\s*\)")
_TIMESTAMP    = re.compile(r"<t:-?\d{1,13}(?::[tTdDfFR])?>")
_USER         = re.compile(rf"<@!?({_SNOWFLAKE})>")
_ROLE         = re.compile(rf"<@&({_SNOWFLAKE})>")
_CHANNEL      = re.compile(rf"<#({_SNOWFLAKE})>")
_SLASH        = re.compile(rf"</([^:>]+):{_SNOWFLAKE}>")
_CUSTOM_EMOJI = re.compile(rf"<a?:(\w+):{_SNOWFLAKE}>")
_EVERYONE     = re.compile(r"@(everyone|here)\b")
_SILENT       = re.compile(r"^\s*@silent\s+", re.I)
# Trailing punctuation stays in the sentence: "(see https://x.com)." → "(see a Twitter link)."
_URL          = re.compile(r"<?\bhttps?://[^\s<>]*[^\s<>.,!?;:)\]}'\"]>?")
_INLINE_CODE  = re.compile(r"`([^`\n]+)`")
_LINE_PREFIX  = re.compile(r"^[ \t]{0,3}(?:#{1,3}[ \t]+|-#[ \t]+|>{1,3}[ \t]?|[-*•][ \t]+)", re.M)
_EMPHASIS     = re.compile(r"(\*\*|__|~~)(.+?)\1", re.S)
_SINGLE_EMPH  = re.compile(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])")
_EMOTICON     = re.compile(
    r"(?<!\S)(?:[:;=][-']?[)(\]\[DPpOo/\\|3*$@]+|[xX][dD]+|<3+|[dD]:|"
    r"\^_*\^|-_+-|[Tt]_+[Tt]|[oO]_+[oO])(?!\S)"
)
_LETTER_RUN   = re.compile(r"([^\W\d_])\1{3,}")    # "noooooo" → "nooo" (never digits)
_PUNCT_RUN    = re.compile(r"([!?])[!?]+")
_DOTS         = re.compile(r"\.{4,}")
# Placeholder brackets (private-use code points) for text that later passes
# must not touch — see protect() in clean_for_speech.
_TOK_OPEN, _TOK_CLOSE = "\ue000", "\ue001"
_PROTECTED    = re.compile(_TOK_OPEN + r"(\d+)" + _TOK_CLOSE)
# Emoji and pictographs. Read aloud they come out as "face with tears of joy"
# three times over, which is never what the sender meant in voice chat.
_EMOJI        = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2300-\u23FF"
    "\uFE00-\uFE0F\u200D\u20E3\U000E0020-\U000E007F]")

# Chat shorthand a person would say in full. Only unambiguous ones: the
# lower-case-only set skips things that are also real words or acronyms when
# capitalised (RN, BC, FR, a name like Ty).
_ABBREVIATIONS = {
    "idk": "I don't know", "idc": "I don't care", "brb": "be right back",
    "tbh": "to be honest", "imo": "in my opinion", "imho": "in my honest opinion",
    "ngl": "not gonna lie", "nvm": "never mind", "btw": "by the way",
    "tysm": "thank you so much", "thx": "thanks", "omg": "oh my god",
    "irl": "in real life", "smh": "shaking my head", "wdym": "what do you mean",
    "hbu": "how about you", "wbu": "what about you", "wyd": "what are you doing",
    "afaik": "as far as I know", "iirc": "if I remember correctly",
    "pls": "please", "plz": "please", "ofc": "of course",
}
_ABBREVIATIONS_LOWER = {
    "u": "you", "ur": "your", "ty": "thank you", "rn": "right now",
    "bc": "because", "fr": "for real", "np": "no problem", "ik": "I know",
    "gn": "good night",
}
_WORD_EDGE_L = r"(?<![\w'@#/.\-])"
_WORD_EDGE_R = r"(?![\w'/@#\-]|\.\w)"     # not "U.S." / "ur-" / "idk's"
_ABBREV_RE = re.compile(
    _WORD_EDGE_L + "(" + "|".join(sorted(_ABBREVIATIONS, key=len, reverse=True)) + ")"
    + _WORD_EDGE_R, re.I)
_ABBREV_LOWER_RE = re.compile(
    _WORD_EDGE_L + "(" + "|".join(sorted(_ABBREVIATIONS_LOWER, key=len, reverse=True)) + ")"
    + _WORD_EDGE_R)

# Sites people link constantly → what you'd call the link out loud.
_SITE_NAMES = {
    "youtube.com": "a YouTube link", "youtu.be": "a YouTube link",
    "music.youtube.com": "a YouTube link", "open.spotify.com": "a Spotify link",
    "spotify.com": "a Spotify link", "soundcloud.com": "a SoundCloud link",
    "twitter.com": "a Twitter link", "x.com": "a Twitter link",
    "reddit.com": "a Reddit link", "twitch.tv": "a Twitch link",
    "tiktok.com": "a TikTok link", "instagram.com": "an Instagram link",
    "github.com": "a GitHub link", "steamcommunity.com": "a Steam link",
    "store.steampowered.com": "a Steam link",
    "tenor.com": "a GIF", "giphy.com": "a GIF",
    "cdn.discordapp.com": "an attachment", "media.discordapp.net": "an attachment",
}

NameLookup = Optional[Callable[[int], Optional[str]]]


def _link_phrase(url: str) -> str:
    try:
        parsed = urlparse(url.strip("<>"))
        host, path = (parsed.hostname or "").lower(), parsed.path.lower()
    except ValueError:
        return "a link"
    for prefix in ("www.", "m.", "mobile.", "vm.", "vt."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    if path.endswith(".gif"):
        return "a GIF"
    if host in _SITE_NAMES:
        return _SITE_NAMES[host]
    if host.endswith(".discordapp.net") or host.endswith(".discordapp.com"):
        return "an attachment"
    return f"a link to {host}" if host else "a link"


def clean_for_speech(
    text: str,
    *,
    member_name: NameLookup = None,
    role_name: NameLookup = None,
    channel_name: NameLookup = None,
    expand: bool = True,
) -> str:
    """Turn a Discord message into what a person would say out loud.

    Mentions become names, links become "a YouTube link", spoilers stay
    spoilers, markdown and emoticons disappear, "noooooooo" is capped, and
    (with `expand`) chat shorthand is said in full. Returns "" when nothing
    pronounceable is left. The *_name callables map an id to a display name.
    """
    if not text:
        return ""
    keep: list[str] = []

    def protect(spoken: str) -> str:
        # Park inserted text (names, link phrases) behind a token so the
        # shorthand expander can't turn someone called "Ty" into "thank you".
        keep.append(spoken)
        return f" {_TOK_OPEN}{len(keep) - 1}{_TOK_CLOSE} "

    def name_of(lookup: NameLookup, raw_id: str, fallback: str) -> str:
        name = None
        if lookup is not None:
            try:
                name = lookup(int(raw_id))
            except Exception:
                name = None
        if name and lookup is not member_name:
            name = name.replace("-", " ")   # "general-chat" → "general chat"
        cleaned = clean_for_speech(name, expand=False) if name else ""
        return protect(cleaned or fallback)

    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = _SILENT.sub("", s)
    s = _CODE_BLOCK.sub(" code ", s)
    s = _SPOILER.sub(" spoiler ", s)
    s = _MASKED_LINK.sub(r"\1", s)
    s = _TIMESTAMP.sub(" ", s)
    s = _USER.sub(lambda m: name_of(member_name, m.group(1), "someone"), s)
    s = _ROLE.sub(lambda m: name_of(role_name, m.group(1), "a role"), s)
    s = _CHANNEL.sub(lambda m: name_of(channel_name, m.group(1), "a channel"), s)
    s = _SLASH.sub(lambda m: protect("slash " + m.group(1).strip()), s)
    s = _CUSTOM_EMOJI.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", s)
    s = _EMOJI.sub(" ", s)
    s = _EVERYONE.sub(r"\1", s)
    s = _URL.sub(lambda m: protect(_link_phrase(m.group(0))), s)
    s = _INLINE_CODE.sub(r"\1", s)
    s = _LINE_PREFIX.sub("", s)
    for _ in range(3):                      # nested **_like this_**
        s = _EMPHASIS.sub(r"\2", s)
        s = _SINGLE_EMPH.sub(r"\2", s)
    s = s.replace("`", "").replace("*", "").replace("~~", "").replace("_", " ")
    s = _EMOTICON.sub(" ", s)
    s = _LETTER_RUN.sub(r"\1\1\1", s)
    s = _PUNCT_RUN.sub(r"\1", s)
    s = _DOTS.sub("...", s)
    if expand:
        s = _ABBREV_RE.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], s)
        s = _ABBREV_LOWER_RE.sub(lambda m: _ABBREVIATIONS_LOWER[m.group(1)], s)
    s = _PROTECTED.sub(lambda m: keep[int(m.group(1))], s)

    # Lines → sentences, so a multi-line message gets natural pauses instead
    # of running "first line second line" together.
    lines = [" ".join(ln.split()) for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    out = ""
    for ln in lines:
        if out:
            out += " " if out[-1] in ".!?,;:…" else ". "
        out += ln
    out = re.sub(r"\s+([.,!?;:)\]}])", r"\1", out)
    out = re.sub(r"([(\[{])\s+", r"\1", out).strip(" ,;:")
    return out if re.search(r"\w", out) else ""


def trim_for_speech(text: str, limit: int) -> tuple[str, bool]:
    """Cap `text` at `limit` characters, cutting at a word boundary.
    Returns (text, was_trimmed)."""
    if limit <= 0 or len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit - 40:
        cut = cut[:space]
    return cut.rstrip(" ,;:-"), True


# ──────────────────────────────────────────────────────────────────────────────
# Synthesis + decoding
# ──────────────────────────────────────────────────────────────────────────────

_SYNTH_TIMEOUT  = 30.0
_DECODE_TIMEOUT = 30.0
# Speech normalised to -16 LUFS lands level with typical music, so a TTS line
# with nothing playing is neither a whisper nor a shout, and over (ducked)
# music it sits clearly on top.
_LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"


async def synthesize(text: str, voice: str) -> bytes:
    """Text → MP3 bytes from Microsoft's neural TTS."""
    try:
        import aiohttp
        import edge_tts
        from edge_tts import exceptions as edge_errors
    except ImportError as e:
        raise TTSError("TTS isn't installed on the host — the owner needs to "
                       "run `./update.sh` (or `pip install edge-tts`).") from e
    try:
        comm = edge_tts.Communicate(text, voice, connect_timeout=10, receive_timeout=20)
    except ValueError as e:                 # malformed voice id
        raise TTSError(f"`{voice}` isn't a valid voice name.") from e

    async def _collect() -> bytes:
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                buf += chunk["data"]
        return bytes(buf)

    try:
        audio = await asyncio.wait_for(_collect(), timeout=_SYNTH_TIMEOUT)
    except edge_errors.NoAudioReceived as e:
        raise NoSpeech("the speech service sent back no audio.") from e
    except (asyncio.TimeoutError, aiohttp.ClientError, edge_errors.EdgeTTSException,
            OSError) as e:
        log.warning("TTS synthesis failed (%s): %s", type(e).__name__, e)
        raise TTSError("couldn't reach Microsoft's speech service — "
                       "try again in a moment.") from e
    if not audio:
        raise NoSpeech("the speech service sent back no audio.")
    return audio


async def to_pcm(encoded: bytes) -> bytes:
    """Any audio ffmpeg can read → 48 kHz stereo s16le, loudness-normalised —
    exactly the frame format discord.py sends."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-af", _LOUDNESS,
        "-ar", "48000", "-ac", "2", "-f", "s16le", "pipe:1",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise TTSError("ffmpeg isn't installed on the host.") from e
    try:
        out, err = await asyncio.wait_for(proc.communicate(encoded), _DECODE_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TTSError("decoding the speech took too long.")
    if proc.returncode != 0 or not out:
        log.warning("TTS decode failed (rc=%s): %s", proc.returncode,
                    err.decode(errors="replace")[-300:])
        raise TTSError("couldn't decode the speech audio.")
    return out


async def render(text: str, voice: str) -> bytes:
    """Text → playable PCM in `voice`. If that voice yields nothing (Microsoft
    retires voices now and then), says it in the default voice instead of
    failing."""
    try:
        encoded = await synthesize(text, voice)
    except NoSpeech:
        fallback = default_voice()
        if voice == fallback:
            raise
        log.warning("TTS voice %s produced no audio — falling back to %s",
                    voice, fallback)
        encoded = await synthesize(text, fallback)
    return await to_pcm(encoded)
