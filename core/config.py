"""
Centralised configuration. Loads from (in order of precedence):

  1. Real environment variables
  2. `.env` file in the project root (via python-dotenv)
  3. `token.txt` in the project root, for the Discord token only

Importing this module reads everything once and exposes the validated
`settings` singleton. Required values fail loud.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present. Never overrides real env vars.
load_dotenv()


def _get(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        sys.exit(f"❌ Missing required environment variable: {name}")
    return val or ""


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"❌ Environment variable {name} must be an integer (got {raw!r})")


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        sys.exit(f"❌ Environment variable {name} must be a number (got {raw!r})")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_token_txt() -> str:
    """Read the Discord token from token.txt next to bot.py, if present.

    Returns an empty string if the file is missing, empty, or still contains
    the placeholder text. The first non-empty, non-comment line wins.
    """
    here = Path(__file__).resolve().parent.parent      # project root
    path = here / "token.txt"
    if not path.is_file():
        return ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Reject the placeholder so we don't try to log in with literal text.
            if "paste" in line.lower() and "token" in line.lower():
                continue
            return line
    except OSError as e:
        print(f"⚠️  Couldn't read token.txt: {e}", file=sys.stderr)
    return ""


def _resolve_discord_token() -> str:
    env_token = os.getenv("DISCORD_TOKEN", "").strip()
    if env_token:
        return env_token
    file_token = _read_token_txt()
    if file_token:
        return file_token
    sys.exit(
        "❌ No Discord token found. Either:\n"
        "   • set DISCORD_TOKEN in your .env file, or\n"
        "   • paste your token into token.txt (one line, no quotes)"
    )


def _read_spotify_txt() -> tuple[str, str]:
    """Read Spotify credentials from spotify.txt next to bot.py, if present.

    The file uses `key=value` format with `client_id` and `client_secret`.
    Lines starting with `#` are ignored. Returns ("", "") if the file is
    missing, malformed, or still contains placeholder text. Whitespace
    around the `=` is tolerated.
    """
    here = Path(__file__).resolve().parent.parent
    path = here / "spotify.txt"
    if not path.is_file():
        return "", ""
    creds: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip().lower()
            value = value.strip()
            # Skip placeholders like PASTE_YOUR_CLIENT_ID_HERE.
            if "paste" in value.lower() and "here" in value.lower():
                continue
            if key in ("client_id", "client_secret") and value:
                creds[key] = value
    except OSError as e:
        print(f"⚠️  Couldn't read spotify.txt: {e}", file=sys.stderr)
        return "", ""
    return creds.get("client_id", ""), creds.get("client_secret", "")


def _resolve_spotify_credentials() -> tuple[str, str]:
    """Env vars beat spotify.txt. Both halves must come from the same source —
    we don't mix env-ID with file-secret, since that's almost always a config
    mistake worth surfacing.
    """
    env_id     = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    env_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if env_id and env_secret:
        return env_id, env_secret
    if env_id or env_secret:
        # Half-configured env. Warn but try to fill in from the file.
        print(
            "⚠️  Only one of SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET is set "
            "in the environment. Spotify needs both.",
            file=sys.stderr,
        )
    file_id, file_secret = _read_spotify_txt()
    # If env has one and file has the other, prefer file's matching pair.
    if file_id and file_secret:
        return file_id, file_secret
    # Otherwise return whatever non-empty pair we managed to assemble, which
    # might be empty-empty. The bot just disables Spotify in that case.
    return env_id or file_id, env_secret or file_secret


def _resolve_lastfm_credentials() -> tuple[str, str]:
    """Last.fm API key + shared secret for scrobbling. Env only (LASTFM_API_KEY,
    LASTFM_API_SECRET) — both must be set or the feature stays disabled. Register
    an app at https://www.last.fm/api/account/create to get them."""
    key    = os.getenv("LASTFM_API_KEY", "").strip()
    secret = os.getenv("LASTFM_API_SECRET", "").strip()
    if bool(key) ^ bool(secret):
        print(
            "⚠️  Only one of LASTFM_API_KEY / LASTFM_API_SECRET is set. "
            "Scrobbling needs both — it will stay disabled.",
            file=sys.stderr,
        )
        return "", ""
    return key, secret


def _resolve_cookiefile() -> str:
    """Path to a Netscape-format cookies.txt yt-dlp uses to fetch login-gated
    content (Instagram posts, age-restricted YouTube, private/members-only, …).

    Precedence: YTDLP_COOKIES env → a `cookies.txt` sitting in the project root.
    Returns "" when neither exists, in which case the bot fetches public content
    only (the historical behaviour). Export the file from a logged-in browser
    with an extension like "Get cookies.txt LOCALLY".
    """
    explicit = os.getenv("YTDLP_COOKIES", "").strip()
    if explicit:
        if not Path(explicit).is_file():
            print(f"⚠️  YTDLP_COOKIES points to a missing file: {explicit}",
                  file=sys.stderr)
        return explicit
    root_cookie = Path(__file__).resolve().parent.parent / "cookies.txt"
    return str(root_cookie) if root_cookie.is_file() else ""


def _resolve_cue_file() -> str:
    """The wake-word cue sound, anchored to the project root (same reasoning as
    the DB path: a relative path would depend on the launch directory).
    VOICE_CUE_FILE overrides; a missing file just means no sound."""
    explicit = os.getenv("VOICE_CUE_FILE", "").strip()
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return str(Path(__file__).resolve().parent.parent / "assets" / "wake_cue.wav")


def _resolve_db_path() -> str:
    """Absolute path to the SQLite database.

    Anchored to the project root, NOT the process's working directory. A bare
    relative `./bot.db` silently becomes a *different* file when the bot is
    launched from a different cwd (systemd, a wrapper script, cron) than the one
    you run `sqlite3 bot.db` / `dbcheck.py` from — which shows up as tables that
    exist in one file but are "missing" in the other. Anchoring kills that whole
    class of bug: everything resolves to the same absolute path every time.

    DB_PATH still overrides (expanded + made absolute so it's launch-cwd-proof
    too). Set it if your data lives somewhere other than the repo root.
    """
    raw = os.getenv("DB_PATH", "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return str(Path(__file__).resolve().parent.parent / "bot.db")


@dataclass(frozen=True)
class Settings:
    discord_token:   str
    spotify_id:      str
    spotify_secret:  str
    lastfm_key:      str
    lastfm_secret:   str
    default_prefix:  str
    db_path:         str
    download_tmp:    str
    stem_cache_dir:  str
    stem_cache_mb:   int
    stem_bitrate:    str
    separation_model: str
    separation_threads: int
    idle_seconds:    int
    log_level:       str
    whisper_model:   str
    voice_wake_word: str
    voice_whisper_model: str
    voice_language:  str
    voice_beam_size: int
    voice_device:    str
    voice_compute_type: str
    voice_cpu_threads: int
    voice_asr_hints: bool
    voice_vad_threshold: float
    voice_end_silence_ms: int
    ytdlp_cookiefile: str
    ytdlp_cookies_from_browser: str
    ytdlp_source_address: str
    tts_voice:       str
    tts_max_chars:   int
    tts_duck_percent: int
    openrouter_api_key: str = ""
    voice_llm:       bool = True
    voice_llm_model: str = "google/gemini-2.5-flash-lite"
    voice_llm_fallbacks: tuple = ("openai/gpt-4.1-nano",)
    voice_llm_url:   str = "https://openrouter.ai/api/v1"
    voice_llm_timeout: float = 8.0
    voice_web_lookup: bool = True
    voice_web_lookups_per_hour: int = 30
    voice_plan_model: str = "google/gemini-2.5-flash"
    voice_plan_fallbacks: tuple = ("openai/gpt-4.1-mini",)
    voice_plan_timeout: float = 20.0
    voice_typing:    bool = True
    voice_fast_wake: bool = True
    voice_fast_wake_threshold: float = 0.0
    voice_fast_wake_verify: bool = True
    voice_fast_wake_verify_model: str = ""
    voice_fast_wake_verify_window: float = 5.0
    voice_fast_wake_verify_full: bool = True
    voice_fast_wake_instant: float = 0.0
    voice_probe_ms:  int = 1200
    voice_cue:       bool = True
    voice_cue_sound: bool = True
    voice_cue_file:  str = ""
    voice_cue_volume: float = 1.0

    @property
    def spotify_enabled(self) -> bool:
        return bool(self.spotify_id and self.spotify_secret)

    @property
    def lastfm_enabled(self) -> bool:
        return bool(self.lastfm_key and self.lastfm_secret)


_spotify_id, _spotify_secret = _resolve_spotify_credentials()
_lastfm_key, _lastfm_secret  = _resolve_lastfm_credentials()

settings = Settings(
    discord_token  = _resolve_discord_token(),
    spotify_id     = _spotify_id,
    spotify_secret = _spotify_secret,
    lastfm_key     = _lastfm_key,
    lastfm_secret  = _lastfm_secret,
    default_prefix = _get("DEFAULT_PREFIX", "-"),
    db_path        = _resolve_db_path(),
    # tempfile.gettempdir() instead of a hardcoded /tmp so this also works
    # on Windows.
    download_tmp   = _get("DOWNLOAD_TMP_DIR",
                          os.path.join(tempfile.gettempdir(), "musicbot-dl")),
    # Separated instrumentals live NEXT TO THE DB, not in /tmp. Separation is
    # the most expensive thing this bot does — minutes of CPU per track — and
    # /tmp is cleared on reboot, which would throw all of it away. Anchoring to
    # the repo makes the cache survive restarts, which is the whole point.
    stem_cache_dir = _get("STEM_CACHE_DIR",
                          str(Path(__file__).resolve().parent.parent / "stem_cache")),
    stem_cache_mb  = _get_int("STEM_CACHE_MB", 2048),
    # Instrumentals are delivered as a Discord attachment, and Discord's
    # unboosted limit is 10 MB. At 320k a 4-minute song lands at ~9.6 MB —
    # right on the line, so half of them would spill over to a catbox link
    # instead of a clean attachment. 192k halves that (~5.8 MB) and is
    # transparent enough for a backing track.
    stem_bitrate   = _get("STEM_BITRATE", "192k").strip(),
    # Quality ladder: fast | good | best. On a CPU-only host `fast` is the only
    # sane default — see core/separation.py for the measured trade-off.
    separation_model = _get("SEPARATION_MODEL", "fast").strip().lower(),
    # 0 = auto (leave one core free for the audio player thread).
    separation_threads = _get_int("SEPARATION_THREADS", 0),
    idle_seconds   = _get_int("IDLE_DISCONNECT_SECONDS", 300),
    log_level      = _get("LOG_LEVEL", "INFO").upper(),
    # Model for -transcribe and voice control (needs `pip install faster-whisper`).
    # tiny/base/small/medium/large-v3 — bigger = more accurate but slower.
    whisper_model  = _get("WHISPER_MODEL", "base"),
    # Default wake phrase for hands-free voice control; per-guild overridable.
    voice_wake_word = _get("VOICE_WAKE_WORD", "hey dj").lower(),
    # Whisper model for live voice commands. English-only (`.en`) variants are
    # more accurate than the multilingual ones at the same size, and voice
    # control forces language="en". base.en balances accuracy vs latency; go
    # tiny.en if the box struggles, or small.en for better proper-noun accuracy.
    voice_whisper_model = _get("VOICE_WHISPER_MODEL", "base.en"),
    # Voice-command recognition knobs — the baseline the voice review asks to
    # measure from (scripts/voice_eval.py compares settings on real clips):
    # language forced (empty = auto-detect), beam 1, CPU/int8, CTranslate2's
    # own thread count (0). VOICE_DEVICE=cuda only after checking the GPU with
    # `voice doctor`. VOICE_ASR_HINTS puts the wake words in whisper's prompt
    # — off by default: prompt text can be echoed back on noise.
    voice_language = _get("VOICE_LANGUAGE", "en").strip().lower(),
    voice_beam_size = max(1, _get_int("VOICE_BEAM_SIZE", 1)),
    voice_device = _get("VOICE_DEVICE", "cpu").strip().lower(),
    voice_compute_type = _get("VOICE_COMPUTE_TYPE", "int8").strip().lower(),
    voice_cpu_threads = max(0, _get_int("VOICE_CPU_THREADS", 0)),
    voice_asr_hints = _get_bool("VOICE_ASR_HINTS", False),
    # Speech detection: Silero probability that starts speech, and how much
    # quiet ends an utterance. Starting points — tune from `voice diag`.
    voice_vad_threshold = min(0.95, max(0.05, _get_float("VOICE_VAD_THRESHOLD", 0.5))),
    voice_end_silence_ms = max(200, _get_int("VOICE_END_SILENCE_MS", 600)),
    # yt-dlp authentication for login-gated sources (Instagram, private/
    # age-restricted content). cookiefile wins; browser extraction is a
    # convenience for desktop hosts (e.g. "firefox" or "firefox:profile").
    ytdlp_cookiefile = _resolve_cookiefile(),
    ytdlp_cookies_from_browser = _get("YTDLP_COOKIES_FROM_BROWSER", "").strip(),
    # Binds yt-dlp's outgoing socket, which in practice selects the IP family.
    # "0.0.0.0" forces IPv4 — the default because YouTube 403s a lot of
    # datacenter IPv6 space while the same request over IPv4 goes through.
    # Set to "" to let the OS choose, or "::" to force IPv6.
    ytdlp_source_address = _get("YTDLP_SOURCE_ADDRESS", "0.0.0.0").strip(),
    # -tts: the voice for anyone who hasn't picked one with `-ttsvoice` (a name
    # from that list, or any Microsoft neural voice id), the longest message
    # it will read out (~25 s of speech at 400), and how loud the music stays
    # under speech, as a percentage (25 ≈ -12 dB: clearly behind the voice,
    # still there).
    tts_voice      = _get("TTS_VOICE", "andrew").strip(),
    tts_max_chars  = _get_int("TTS_MAX_CHARS", 400),
    tts_duck_percent = _get_int("TTS_DUCK_PERCENT", 25),
    # Voice DJ tier 2 — an LLM (via OpenRouter) for what the voice grammar
    # can't parse: several songs in one request, mixed commands, and spoken
    # answers to questions ("hey DJ, who sings this?"). Off without a key.
    # Any OpenAI-compatible endpoint works for VOICE_LLM_URL (e.g. a local
    # llama-server). The model choice is explained in core/voice_llm.py.
    openrouter_api_key = _get("OPENROUTER_API_KEY", "").strip(),
    voice_llm = _get_bool("VOICE_LLM", True),
    voice_llm_model = _get("VOICE_LLM_MODEL", "google/gemini-2.5-flash-lite").strip(),
    voice_llm_fallbacks = tuple(m.strip() for m in
                                _get("VOICE_LLM_FALLBACKS", "openai/gpt-4.1-nano").split(",")
                                if m.strip()),
    voice_llm_url = _get("VOICE_LLM_URL", "https://openrouter.ai/api/v1").strip(),
    voice_llm_timeout = max(1.0, _get_float("VOICE_LLM_TIMEOUT", 8.0)),
    # A song described rather than named ("that Yung Gravy song that samples
    # Panic! At The Disco") is looked up online through the same OpenRouter
    # key (web search plugin, about a cent per lookup), at most this many an hour.
    voice_web_lookup = _get_bool("VOICE_WEB_LOOKUP", True),
    voice_web_lookups_per_hour = max(0, _get_int("VOICE_WEB_LOOKUPS_PER_HOUR", 30)),
    # The planner (round 8): the model that turns a request into steps. Chosen
    # on the live bench (scripts/voice_plan_bench.py: the brief's requests and
    # 11 real microphone recordings, as said and as the bot heard them, 2026-10-14):
    # gemini-2.5-flash planned 72 of 72 right (median 2.4 s, p90 4.9 s, about
    # $0.0015 a request); flash-lite 32 of 36 (1.3 s) — it drops steps from
    # long requests and misreads queue references; gpt-4.1-mini 32 of 36
    # (1.95 s), so it's the fallback. "" = the same as VOICE_LLM_MODEL.
    voice_plan_model = _get("VOICE_PLAN_MODEL", "google/gemini-2.5-flash").strip(),
    voice_plan_fallbacks = tuple(m.strip() for m in _get(
        "VOICE_PLAN_FALLBACKS", "openai/gpt-4.1-mini").split(",") if m.strip()),
    voice_plan_timeout = max(2.0, _get_float("VOICE_PLAN_TIMEOUT", 20.0)),
    # "<bot> is typing…" in the reply channel from the moment the wake word
    # is heard until the answer is posted: a quick early look at the first
    # VOICE_PROBE_MS of each utterance (0 = only once the whole thing is heard).
    voice_typing = _get_bool("VOICE_TYPING", True),
    # The instant wake word (core/wake_spotter.py): a tiny model that hears
    # "hey DJ" within a fraction of a second, so the cue and "typing…" come
    # at once. Speech-to-text still decides what was said. Threshold 0 = the
    # model's own (stored with it); higher = fewer false alarms, more misses.
    voice_fast_wake = _get_bool("VOICE_FAST_WAKE", True),
    voice_fast_wake_threshold = min(0.999, max(0.0, _get_float("VOICE_FAST_WAKE_THRESHOLD", 0.0))),
    # …and before the cue, the speech recogniser's second opinion on just
    # those words (its own instance of the listener's model, unless
    # VOICE_FAST_WAKE_VERIFY_MODEL names another, e.g. tiny.en: faster, misses more).
    voice_fast_wake_verify = _get_bool("VOICE_FAST_WAKE_VERIFY", True),
    voice_fast_wake_verify_model = _get("VOICE_FAST_WAKE_VERIFY_MODEL", "").strip(),
    # The check runs Whisper's encoder on a window this long (seconds; the
    # clip plus 2 s of silence if that's longer) instead of the usual 30 s:
    # ~0.1 s instead of ~0.9 s on the bot's host. 0 = the full 30-second pass.
    voice_fast_wake_verify_window = max(0.0, min(30.0, _get_float(
        "VOICE_FAST_WAKE_VERIFY_WINDOW", 5.0))),
    # …and when the short looks don't hear the wake word, one last look with
    # the full 30-second pass (as before this round): hard ones — noise, an
    # unusual voice — still get a fast cue, at the old speed.
    voice_fast_wake_verify_full = _get_bool("VOICE_FAST_WAKE_VERIFY_FULL", True),
    # A detector score at or above this plays the cue *before* the check comes
    # back (the check still runs; a "no" takes the text cue back — the sound
    # can't be). 0 = always check first. assets/wakeword/README.md has what
    # each level costs in false cues.
    voice_fast_wake_instant = min(1.0, max(0.0, _get_float("VOICE_FAST_WAKE_INSTANT", 0.0))),
    voice_probe_ms = max(0, _get_int("VOICE_PROBE_MS", 1200)),
    # The "I heard the wake word" cue: a short sound effect in the voice
    # channel (Alexa's light, in audio) plus ":ear: Now listening…" in the
    # voice-reply channel. VOICE_CUE=false silences the sound on the host;
    # `voice cue on|off` is the per-server switch; the message always posts.
    voice_cue = _get_bool("VOICE_CUE", True),
    voice_cue_sound = _get_bool("VOICE_CUE_SOUND", True),
    voice_cue_file = _resolve_cue_file(),
    voice_cue_volume = max(0.0, _get_float("VOICE_CUE_VOLUME", 1.0)),
)
