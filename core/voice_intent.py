"""
Pure voice-command parsing — wake-word detection + intent matching.

No Discord, no audio, no I/O: this turns a (possibly mis-transcribed) line of
speech into a structured :class:`Intent`, so the whole thing is unit-testable
in isolation. The voice cog feeds it whisper output and dispatches the result
to the existing music commands. Tests: tests/test_voice_intent.py.

Design — correctness first, tolerance second:

  * **The wake word gates everything — at the start.** Nothing parses unless
    the utterance OPENS with a wake phrase (a little filler like "um, so" is
    allowed first). A wake phrase later in a sentence is quoted or reported
    speech far more often than a command — "I said hey DJ skip", "do not say
    hey DJ skip" used to skip — so it never acts. Saying the wake word on its
    own and the command next ("hey DJ" … "skip") is the explicit alternative.
    Wake matching is deliberately stricter than command matching:
    exact tokens, a few known spellings of "hey" and "DJ" (whisper writes
    "hey D.J.", "hey dee jay", "heydj"), and light fuzz only for long custom
    wake words ("computor" → computer).
  * **Exact grammar before anything fuzzy.** After the wake word (and polite
    filler like "can you … please"), the *whole* remainder must be a known
    phrase — "hold up" is pause, "loop off" is loop-off, "start over" is
    restart. An earlier version scored every verb fuzzily and phonetically
    against the leading words, which turned "loop off" into STOP (Soundex says
    loop ≈ leave), "hold up" into volume-up, and "next week I'm away" into
    skip. Trailing chatter is no longer accepted.
  * **Then a short alias list** of real mis-hearings ("skib", "paws"), then a
    single-word, edit-distance-1 fallback that must be unambiguous. No
    Soundex: it matched far too much.
  * **Negation wins.** "don't stop" is not a stop command (it used to be
    volume-DOWN, via "don" ≈ "down").
  * **Song queries come from the original transcript.** Accents and non-Latin
    titles survive ("play Beyoncé — Déjà Vu", "play 夜に駆ける").
  * **Confirmations have their own grammar** (:func:`parse_confirmation`):
    any refusal word wins ("yeah no", "yes but don't" → no), and unrelated
    speech is neither yes nor no.

Confidence values are a rough match-quality label (exact 1.0, alias 0.9,
fuzzy 0.75) — not calibrated probabilities.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

# The closed action vocabulary. The voice cog maps these onto real commands.
# album / top / latest come only from the smart tier (an LLM works out which
# album or artist was meant; core/music_catalog.py finds the tracks).
ACTIONS = (
    "skip", "pause", "resume", "restart", "stop", "clear", "loop", "shuffle",
    "volume", "play", "album", "top", "latest", "find", "confirm", "cancel",
)

#: Actions that add songs to the queue. ("find": a song described rather
#: than named — "that Yung Gravy song that samples Panic! At The Disco" —
#: looked up before it's queued.)
SONG_ACTIONS = frozenset({"play", "album", "top", "latest", "find"})

DESTRUCTIVE = frozenset({"stop", "clear"})

#: Spoken volume is limited to this (percent). The typed command allows more,
#: but a mis-heard "volume five hundred" shouldn't blast anyone, so a spoken
#: level above it needs a confirmation.
VOICE_VOLUME_MAX = 200


@dataclass(frozen=True)
class Intent:
    """A recognised command. ``arg`` carries the play query (or, from the
    smart tier, the song title — with ``artist`` alongside); for album the
    album title; for volume ``"up"``/``"down"``/a number string like ``"50"``;
    for loop ``"on"``/``"song"``/``"queue"``/``"off"``/``"toggle"``.
    ``count`` says which/how many: album tracks (``"1-3"``, ``"all"``,
    ``"last 2"``), top/latest songs, songs to skip. ``when`` puts songs at
    the front of the queue: ``"next"`` (after the current song) or ``"now"``
    (instead of it). ``heard`` (play, from the smart tier) is the title as the
    speech recogniser wrote it, misheard or not — ``arg`` is what the LLM made
    of it — so the catalogue can check both against the artist's real songs."""
    action: str
    arg: str | None = None
    confidence: float = 1.0
    artist: str | None = None
    count: str | None = None
    when: str | None = None          # songs: None = end of the queue, "next", "now"
    heard: str | None = None

    @property
    def search(self) -> str | None:
        """What a play request searches for: "Artist - Title" when the artist
        is known (the smart tier worked it out), else the words as said."""
        if self.action != "play" or not self.arg:
            return self.arg
        return f"{self.artist} - {self.arg}" if self.artist else self.arg

    @property
    def destructive(self) -> bool:
        return self.action in DESTRUCTIVE

    @property
    def needs_confirmation(self) -> bool:
        """Destructive, or a spoken volume above VOICE_VOLUME_MAX."""
        if self.destructive:
            return True
        return (self.action == "volume" and self.arg is not None
                and self.arg.isdigit() and int(self.arg) > VOICE_VOLUME_MAX)


@dataclass(frozen=True)
class ParseResult:
    """:func:`parse_detailed`'s answer, with a reason code for diagnostics.

    reason ∈ matched, empty, no_wake, wake_not_at_start, wake_only,
    no_command, negated, bad_volume, ambiguous.

    ``more`` holds further songs from one request ("play X by A and Y by B").
    ``list_like`` flags a song request that reads like a list the grammar
    couldn't split with confidence ("play X and Y" — two songs, or one called
    "X and Y"?): the voice cog asks the LLM tier when that's enabled."""
    intent: Intent | None
    reason: str
    wake: str | None = None
    wake_at_start: bool = False
    command: str = ""
    more: tuple[Intent, ...] = ()
    list_like: bool = False

    @property
    def intents(self) -> tuple[Intent, ...]:
        return ((self.intent,) + self.more) if self.intent is not None else ()


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Token:
    text: str     # folded: casefolded, accents and apostrophes removed
    start: int    # span in the original transcript, for lifting song queries
    end: int


# Numbers (optionally signed / with %), or runs of letters/digits in any script
# with internal apostrophes ("don't", "what's").
_TOKEN_RE = re.compile(r"[-−–]?\d+(?:[.,]\d+)*%?|[^\W_]+(?:['’][^\W_]+)*")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.replace("'", "").replace("’", "")


# Spellings whisper uses for "DJ", merged back into one token.
_DJ_PAIRS = {("d", "j"), ("dee", "jay"), ("d", "jay"), ("dee", "j")}
_DJ_SINGLE = {"deejay", "dejay", "deejays", "djs", "djay"}


def tokenize(text: str) -> list[Token]:
    """Folded tokens with spans into ``text``; "D.J." / "dee jay" → "dj"."""
    raw = [Token(_fold(m.group()), m.start(), m.end())
           for m in _TOKEN_RE.finditer(text or "")]
    out: list[Token] = []
    i = 0
    while i < len(raw):
        t = raw[i]
        if i + 1 < len(raw) and (t.text, raw[i + 1].text) in _DJ_PAIRS:
            out.append(Token("dj", t.start, raw[i + 1].end))
            i += 2
            continue
        out.append(Token("dj", t.start, t.end) if t.text in _DJ_SINGLE else t)
        i += 1
    return out


def _normalize(text: str) -> str:
    """Folded command text: lowercase, no accents/punctuation, DJ merged."""
    return " ".join(t.text for t in tokenize(text))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ratio(a: str, b: str) -> float:
    """Similarity in [0, 1]: 1 - edit_distance / longest. Empty vs empty = 1."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    return 1.0 - _levenshtein(a, b) / longest if longest else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Wake words
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_WAKE = "hey dj"

# How whisper writes "hey" at the start of "hey DJ".
_HEY_ALIASES = frozenset({"hey", "hay", "hei", "heh", "hi", "ay", "eh", "a", "yo"})
# Harmless words allowed before a wake phrase at the start of a line.
_LEAD_FILLER = frozenset({"um", "uh", "er", "erm", "so", "ok", "okay", "oh",
                          "and", "but", "well", "like", "yeah",
                          "hey", "hi", "hello", "yo"})     # "hey, hey DJ …"

#: After a wake-only utterance ("hey DJ" … pause), the next thing the same
#: person starts saying within this many seconds is taken as the command.
WAKE_WINDOW_SECONDS = 4.0


def normalize_wake(phrase: str) -> str | None:
    """Canonical form of a wake phrase as the matcher sees it, or None if it
    can't work: empty after normalisation, non-Latin letters (the speech model
    transcribes English, so they'd never be heard), digits, or too long.
    Validate configured wake words with this — an earlier validator accepted
    phrases the matcher then normalised to nothing, which opened the gate."""
    tokens = [t.text for t in tokenize(phrase)]
    if not tokens or len(tokens) > 4:
        return None
    joined = " ".join(tokens)
    if len(joined) > 32 or not all(c == " " or ("a" <= c <= "z") for c in joined):
        return None
    return joined


def _wake_token_ok(tok: str, want: str, loose: bool) -> bool:
    if tok == want:
        return True
    if not loose:
        return False
    if want == "hey":
        return tok in _HEY_ALIASES
    # Light fuzz for long, distinctive custom words only ("computor"). Short
    # words stay exact — fuzzing them is how conversation starts waking the bot.
    return len(want) >= 5 and len(tok) >= 4 and _ratio(tok, want) >= 0.8


def _wake_at(tokens: list[str], i: int, wake: list[str], loose: bool) -> int | None:
    """Wake phrase starting at tokens[i]? Returns the index after it.
    ``loose`` (start of the line only) allows the "hey" spellings and light
    fuzz; mid-sentence the phrase must be exact — "he's a DJ, play his stuff"
    must not wake the bot."""
    if i >= len(tokens):
        return None
    # Observed from base.en on the bot host: "Hi DJ" fused into "HIDJ".
    # Keep this a narrow, utterance-leading alias, not broad fuzzy matching.
    if loose and wake == ["hey", "dj"] and tokens[i] in {"hidj", "haydj"}:
        return i + 1
    glued = "".join(wake)
    if len(wake) > 1 and (tokens[i] == glued or (
            loose and len(glued) >= 5 and _ratio(tokens[i], glued) >= 0.85)):
        return i + 1                                   # "heydj"
    if i + len(wake) <= len(tokens) and all(
            _wake_token_ok(tokens[i + k], wake[k], loose) for k in range(len(wake))):
        return i + len(wake)
    return None


def _wake_list(wake: str | list[str] | tuple[str, ...] | None) -> list[tuple[str, list[str]]]:
    """Usable wake phrases as (canonical, tokens), longest first. Phrases that
    normalise to nothing are dropped — never treated as "no gate"."""
    phrases = [wake] if isinstance(wake, str) else list(wake or [])
    out = []
    for p in phrases:
        canon = normalize_wake(p or "")
        if canon and canon not in (c for c, _ in out):
            out.append((canon, canon.split()))
    out.sort(key=lambda cw: (len(cw[1]), len(cw[0])), reverse=True)
    return out


def _find_wakes(tokens: list[str], wakes) -> list[tuple[int, int, str, bool]]:
    """Every wake occurrence as (start, end, wake, at_start). "At start" allows
    up to two filler words first ("um, hey DJ …")."""
    found: list[tuple[int, int, str, bool]] = []
    taken: set[int] = set()
    for canon, wt in wakes:
        for i in range(len(tokens)):
            if i in taken:
                continue
            at_start = i <= 2 and all(t in _LEAD_FILLER for t in tokens[:i])
            end = _wake_at(tokens, i, wt, loose=at_start)
            if end is None:
                continue
            found.append((i, end, canon, at_start))
            taken.update(range(i, end))
    found.sort(key=lambda f: (not f[3], f[0]))
    return found


def strip_wake(text: str, wake: str | list[str] = DEFAULT_WAKE) -> str | None:
    """If ``text`` begins with a wake phrase, return the normalised remainder
    (possibly ""); else None. An empty or unusable wake list closes the gate."""
    tokens = [t.text for t in tokenize(text)]
    for start, end, _canon, at_start in _find_wakes(tokens, _wake_list(wake)):
        if at_start:
            return " ".join(tokens[end:])
    return None


def text_after_wake(text: str, wake: str | list[str] = DEFAULT_WAKE) -> str | None:
    """Like :func:`strip_wake`, but the remainder as it was said (original
    case and punctuation) — what an LLM should read. None if the line doesn't
    open with a wake phrase."""
    tokens = tokenize(text)
    words = [t.text for t in tokens]
    for _start, end, _canon, at_start in _find_wakes(words, _wake_list(wake)):
        if at_start:
            return text[tokens[end].start:].strip(" ,.!?;:") if end < len(tokens) else ""
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Command grammar
# ──────────────────────────────────────────────────────────────────────────────

_OBJECTS = ("", " it", " this", " that", " the song", " this song", " that song",
            " the track", " this track", " the music", " music", " song", " track",
            " playback", " the playback")


def _with_objects(*verbs: str) -> list[str]:
    return [v + o for v in verbs for o in _OBJECTS]


# phrase → (action, arg). The WHOLE command must equal one of these.
_PHRASE_LISTS: dict[tuple[str, str | None], list[str]] = {
    ("skip", None): _with_objects("skip") + [
        "skip this one", "skip that one", "next", "next song", "next track",
        "next one", "the next song", "the next one", "play the next song",
        "play next song", "play the next one", "go to the next song",
        "go to the next one"],
    ("pause", None): _with_objects("pause") + [
        "hold on", "hold up", "wait", "freeze", "pause for a second"],
    ("resume", None): _with_objects("resume", "unpause", "continue") + [
        "keep playing", "carry on", "start playing", "resume playing", "keep going",
        "play", "play it", "play this", "play the music", "play music",
        "play the song", "start the music", "unpause please"],
    ("restart", None): _with_objects("restart", "replay") + [
        "start over", "start it over", "start again", "start from the beginning",
        "start the song over", "from the top", "from the beginning",
        "play it again", "play this again", "play that again", "play the song again",
        "go back to the start", "back to the start", "back to the beginning",
        "take it from the top"],
    ("stop", None): _with_objects("stop") + [
        "stop playing", "leave", "disconnect", "get out", "leave the channel",
        "leave the call", "leave voice"],
    ("clear", None): [
        "clear", "clear it", "clear the queue", "clear queue", "empty the queue",
        "empty queue", "wipe the queue", "clear the playlist", "delete the queue",
        "clear everything"],
    ("loop", "on"): [
        "loop", "repeat", "loop on", "repeat on", "loop it", "loop this", "repeat it",
        "repeat this", "loop the song", "loop this song", "repeat the song",
        "repeat this song", "loop song", "repeat song", "loop the track",
        "loop this track", "turn on loop", "turn loop on", "turn on repeat",
        "turn repeat on", "enable loop", "enable repeat", "start looping",
        "keep looping", "put it on repeat", "put this on repeat", "on repeat",
        "loop mode on", "repeat mode on"],
    ("loop", "off"): [
        "loop off", "repeat off", "turn off loop", "turn loop off", "turn off repeat",
        "turn repeat off", "disable loop", "disable repeat", "stop looping",
        "stop repeating", "stop the loop", "stop the repeat", "unloop",
        "no more looping", "loop mode off", "repeat mode off", "end the loop",
        "cancel the loop", "turn off looping"],
    ("loop", "toggle"): ["toggle loop", "toggle repeat", "loop toggle"],
    ("loop", "queue"): [
        "loop the queue", "loop queue", "repeat the queue", "repeat queue",
        "loop the playlist", "repeat the playlist", "loop everything",
        "repeat everything", "loop all", "repeat all", "loop all songs",
        "loop all the songs", "loop the whole queue", "repeat the whole queue",
        "loop these songs", "loop the songs", "queue loop", "loop mode queue"],
    ("shuffle", None): [
        "shuffle", "shuffle it", "shuffle the queue", "shuffle queue",
        "shuffle the playlist", "shuffle the songs", "shuffle everything",
        "shuffle the music", "mix it up", "mix up the queue", "randomize the queue",
        "randomise the queue", "shuffle up the queue", "shuffle them"],
    ("volume", "up"): [
        "louder", "volume up", "turn it up", "turn up", "turn the volume up",
        "turn up the volume", "turn the music up", "raise the volume", "raise volume",
        "up the volume", "crank it", "crank it up", "pump it up", "a bit louder",
        "a little louder", "bit louder", "little louder", "more volume",
        "boost the volume", "louder please"],
    ("volume", "down"): [
        "quieter", "softer", "lower", "lower it", "volume down", "turn it down",
        "turn down", "turn the volume down", "turn down the volume",
        "turn the music down", "lower the volume", "lower volume", "a bit quieter",
        "a little quieter", "bit quieter", "little quieter", "less volume",
        "a bit softer", "a little softer", "quieter please"],
    ("confirm", None): [
        "confirm", "confirmed", "yes", "yeah", "yep", "yup", "do it", "go ahead",
        "affirmative", "yes do it", "sure"],
    ("cancel", None): [
        "cancel", "cancel that", "nevermind", "never mind", "no", "nope", "nah",
        "abort", "forget it"],
}

_PHRASES: dict[str, tuple[str, str | None]] = {}
for _key, _phrases in _PHRASE_LISTS.items():
    for _p in _phrases:
        if _p in _PHRASES and _PHRASES[_p] != _key:     # pragma: no cover
            raise RuntimeError(f"voice grammar conflict: {_p!r}")
        _PHRASES[_p] = _key

# Observed/likely whisper mis-hearings of command words (token level). Grow this
# from `voice debug` logs rather than loosening the matcher.
_ALIASES = {
    "skipp": "skip", "skib": "skip", "skit": "skip", "skips": "skip", "scip": "skip",
    "tip": "skip", "kip": "skip",           # "Tip this." (reported 2026-10-12)
    "necks": "next", "nex": "next",
    "paws": "pause", "pours": "pause", "pores": "pause", "paus": "pause",
    "pawz": "pause", "porz": "pause", "unpaws": "unpause",
    "resoom": "resume", "rezoom": "resume", "resum": "resume",
    "stahp": "stop", "stopp": "stop",
    "loupe": "loop", "lupe": "loop", "loops": "loop",
    "cue": "queue", "que": "queue", "kew": "queue",
    "volumn": "volume", "volum": "volume",
    "clare": "clear", "clair": "clear",
    "lauder": "louder", "quiter": "quieter",
    "replays": "replay",
    "shuffel": "shuffle", "shuffled": "shuffle", "shuffles": "shuffle",
}

# Single command words the edit-distance-1 fallback may land on.
_FUZZY_WORDS = {
    "skip": ("skip", None), "next": ("skip", None), "pause": ("pause", None),
    "resume": ("resume", None), "unpause": ("resume", None),
    "continue": ("resume", None), "restart": ("restart", None),
    "stop": ("stop", None), "leave": ("stop", None), "clear": ("clear", None),
    "loop": ("loop", "on"), "repeat": ("loop", "on"),
    "louder": ("volume", "up"), "quieter": ("volume", "down"),
    "shuffle": ("shuffle", None),
}

# Politeness and filler around a command. Multi-word entries are token tuples.
_LEAD_POLITE = [("please",), ("can", "you"), ("could", "you"), ("would", "you"),
                ("will", "you"), ("can", "u"), ("could", "u"), ("just",),
                ("go", "ahead", "and"), ("lets",), ("um",), ("uh",), ("er",),
                ("so",), ("and",), ("ok",), ("okay",), ("now",), ("i", "said"),
                ("i", "want", "to"), ("id", "like", "to")]
_TRAIL_POLITE = [("please",), ("now",), ("thanks",), ("thank", "you"),
                 ("for", "me"), ("right", "now"), ("already",), ("dj",)]
_NEGATIONS = frozenset({"dont", "not", "never", "no", "nope", "nah", "doesnt",
                        "cant", "cannot", "wont", "shouldnt", "stop"})

_PLAY_VERBS = frozenset({"play", "queue", "add"})

#: Words that may come before the play verb in a longer request — "I want to
#: hear two songs, can you play …", "could you please queue up …".
_LEAD_IN = frozenset({
    "hear", "listen", "to", "two", "three", "four", "five", "a", "few", "couple",
    "of", "some", "songs", "song", "tracks", "track", "tunes", "can", "could",
    "would", "will", "you", "u", "please", "me", "us", "for", "i", "id", "im",
    "wanna", "want", "like", "love", "lets", "let", "so", "um", "uh", "ok",
    "okay", "now", "and", "then", "also", "just", "go", "ahead", "the", "next",
    "dj", "hey", "yo", "need", "gonna", "going", "should", "maybe", "how",
    "about", "up", "real", "quick"})

#: At most this many songs from one spoken request.
MAX_VOICE_PLAYS = 5

_COUNTS = {"two": 2, "three": 3, "four": 4, "five": 5, "2": 2, "3": 3, "4": 4, "5": 5}


def _play_query_tokens(tokens: list[Token]) -> list[Token] | None:
    """The song part of a play request, or None if this isn't one.

    "play X" / "queue X" / "add X" / "put on X", optionally after lead-in
    words only ("I want to hear two songs, can you play X"); then "hear X" /
    "listen to X" as verbs of their own. A request for nothing in particular
    ("play two songs") isn't a play request."""
    words = [t.text for t in tokens]
    for i, w in enumerate(words):
        verb_len = 0
        if _ALIASES.get(w, w) in _PLAY_VERBS:
            verb_len = 1
        elif w == "put" and i + 1 < len(words) and words[i + 1] == "on":
            verb_len = 2
        if verb_len:
            if all(x in _LEAD_IN for x in words[:i]):
                rest = tokens[i + verb_len:]
                if rest and rest[0].text == "up" and words[i] == "queue":
                    rest = rest[1:]                       # "queue up X"
                return rest
            break
        if w not in _LEAD_IN:
            break
    for i, w in enumerate(words):
        if w not in _LEAD_IN:
            break
        if w == "hear" or (w == "listen" and i + 1 < len(words) and words[i + 1] == "to"):
            rest = tokens[i + (1 if w == "hear" else 2):]
            if rest and not all(t.text in _LEAD_IN for t in rest):
                return rest
            return None
    return None


_STRONG_SEP = re.compile(r"\s*,?\s*\b(?:and then|then|followed by|and after that|after that)\b\s*,?\s*",
                         re.IGNORECASE)
_LIST_SEP = re.compile(r"\s*,\s*(?:and\s+)?|\s+(?:and|&|plus)\s+", re.IGNORECASE)
_BY_SPLIT = re.compile(r"^(?P<title>.*\S)\s+by\s+(?P<artist>\S.*)$", re.IGNORECASE)
_LISTY = re.compile(r",|&|\b(?:and|plus|then|followed by)\b", re.IGNORECASE)


def _song_like(part: str) -> bool:
    """A piece that can stand on its own as a song request."""
    words = part.split()
    return bool(words) and words[0].lower() != "by" and (len(words) >= 2 or len(words[0]) > 3)


def split_songs(query: str, expected: int | None = None) -> tuple[list[str], bool]:
    """One spoken song request → (song queries, unsure).

    Splits only where it can't be a single title: "X then Y" (each side a
    song on its own), and "X by A and Y by B" (every piece names an artist —
    a piece without one belongs to the artist before it, as in "by Simon and
    Garfunkel"). `unsure` means it reads like a list that this couldn't split
    with confidence — "play X and Y" might be two songs or one called "X and
    Y" — so a smarter tier should decide when there is one."""
    query = query.strip(" ,.;")
    unsure = False
    parts = [p.strip(" ,.;") for p in _STRONG_SEP.split(query) if p.strip(" ,.;")]
    if len(parts) > 1 and not all(_song_like(p) for p in parts):
        parts, unsure = [query], True
    out: list[str] = []
    for part in parts:
        pieces = [x.strip() for x in _LIST_SEP.split(part) if x.strip()]
        if len(pieces) == 1:
            out.append(part)
            continue
        merged: list[str] = []
        joins = 0
        for piece in pieces:
            if _BY_SPLIT.match(piece) or not merged:
                merged.append(piece)
            else:
                merged[-1] += " and " + piece
                joins += 1
        if len(merged) > 1 and all(_BY_SPLIT.match(m) for m in merged):
            out.extend(merged)
            unsure = unsure or joins > 0
        else:
            out.append(part)
            unsure = True
    if expected is not None and expected != len(out):
        unsure = True
    return out[:MAX_VOICE_PLAYS], unsure


def _strip_affixes(tokens: list[Token], *, trailing: bool = True) -> list[Token]:
    changed = True
    while changed and tokens:
        changed = False
        for pre in _LEAD_POLITE:
            if len(tokens) > len(pre) and tuple(t.text for t in tokens[:len(pre)]) == pre:
                tokens = tokens[len(pre):]
                changed = True
                break
    changed = trailing
    while changed and tokens:
        changed = False
        for suf in _TRAIL_POLITE:
            if len(tokens) > len(suf) and tuple(t.text for t in tokens[-len(suf):]) == suf:
                tokens = tokens[:-len(suf)]
                changed = True
                break
    return tokens


# ── numbers ───────────────────────────────────────────────────────────────────

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}


def parse_number(words: list[str]) -> tuple[int | None, bool]:
    """Spoken or written number → (value, negative). The *entire* word list
    must be the number (plus an optional "percent"); else (None, False).
    "fifty", "a hundred and twenty", "seventy five", "50", "50%", "minus ten"."""
    w = list(words)
    if w and w[-1] in ("percent", "per", "%"):
        w = w[:-1]
    if w and w[-1] == "cent" and len(w) >= 2 and w[-2] == "per":   # "per cent"
        w = w[:-2]
    negative = False
    if w and w[0] in ("minus", "negative"):
        negative, w = True, w[1:]
    if len(w) == 1 and re.fullmatch(r"[-−–]?\d{1,4}(?:[.,]\d+)?%?", w[0]):
        raw = w[0].rstrip("%")
        if raw[0] in "-−–":
            negative, raw = True, raw[1:]
        try:
            return int(float(raw.replace(",", ""))), negative
        except ValueError:
            return None, False
    if not w:
        return None, False
    total, i = 0, 0
    if w[0] == "hundred":
        total, i = 100, 1
    elif w[i] in ("a", "one") and i + 1 < len(w) and w[i + 1] == "hundred":
        total, i = 100, 2
    elif w[i] in _UNITS and i + 1 < len(w) and w[i + 1] == "hundred":
        total, i = 100 * _UNITS[w[i]], 2
    if i + 1 < len(w) and w[i] == "and" and total:      # "a hundred AND twenty"
        i += 1
    rest = 0
    if i < len(w) and w[i] in _TENS:
        rest, i = _TENS[w[i]], i + 1
        if i < len(w) and w[i] in _UNITS and 0 < _UNITS[w[i]] < 10:
            rest, i = rest + _UNITS[w[i]], i + 1
    elif i < len(w) and w[i] in _UNITS:
        rest, i = _UNITS[w[i]], i + 1
    if i != len(w) or (total == 0 and rest == 0 and w[0] not in ("zero",)):
        return None, False
    return total + rest, negative


_VOLUME_PREFIXES = [
    ("volume",), ("volume", "to"), ("volume", "at"), ("set", "volume"),
    ("set", "volume", "to"), ("set", "the", "volume", "to"), ("set", "it", "to"),
    ("turn", "it", "to"), ("turn", "volume", "to"), ("turn", "the", "volume", "to"),
    ("change", "volume", "to"), ("change", "the", "volume", "to"),
    ("put", "the", "volume", "at"), ("volume", "level"),
]


def _try_volume_level(words: list[str]) -> tuple[Intent | None, str | None]:
    for pre in sorted(_VOLUME_PREFIXES, key=len, reverse=True):
        if len(words) > len(pre) and tuple(words[:len(pre)]) == pre:
            value, negative = parse_number(words[len(pre):])
            if value is None:
                continue
            if negative:
                return None, "bad_volume"
            return Intent("volume", str(value)), None
    return None, None


# ── the matcher ───────────────────────────────────────────────────────────────

def _lookup(words: list[str]) -> tuple[str, str | None] | None:
    return _PHRASES.get(" ".join(words))


def _match_tokens(tokens: list[Token], original: str, *,
                  strict: bool) -> tuple[Intent | None, str]:
    """Match a de-waked command. ``strict`` (wake heard mid-sentence) allows
    only exact grammar — no fuzzy fallback."""
    tokens = _strip_affixes(tokens, trailing=False)
    if not tokens:
        return None, "no_command"

    # play / queue / add / put on <query> — before trailing-filler stripping, so
    # a title ending in "please" or "now" is left alone. Exact-phrase uses of
    # "play" ("play it again", "play the next song", bare "play") win first.
    head = tokens[0].text
    words_all = [t.text for t in _strip_affixes(tokens)]
    exact = _lookup(words_all)
    if exact is None and words_all:
        aliased = [_ALIASES.get(w, w) for w in words_all]
        if aliased != words_all:
            hit = _lookup(aliased)
            if hit is not None:
                return Intent(hit[0], hit[1], confidence=0.9), "matched"
    if exact is not None:
        return Intent(exact[0], exact[1]), "matched"

    query_tokens = None
    if _ALIASES.get(head, head) in _PLAY_VERBS:
        query_tokens = tokens[1:]
        if head == "queue" and query_tokens and query_tokens[0].text == "up":
            query_tokens = query_tokens[1:]               # "queue up X"
    elif head == "put" and len(tokens) > 1 and tokens[1].text == "on":
        query_tokens = tokens[2:]
    elif not strict:
        query_tokens = _play_query_tokens(tokens)
    if query_tokens is not None:
        if not query_tokens:
            return Intent("resume"), "matched"
        if all(t.text in _LEAD_IN for t in query_tokens):
            return None, "no_command"                     # "play two songs" — which?
        query = original[query_tokens[0].start:query_tokens[-1].end].strip(" \t,.!?;:")
        if not query:
            return None, "no_command"
        return Intent("play", query), "matched"

    words = words_all
    level, problem = _try_volume_level(words)
    if level is not None:
        return level, "matched"
    if problem:
        return None, problem

    if any(w in _NEGATIONS for w in words):
        return None, "negated"
    if strict:
        return None, "no_command"

    # Last resort: one command word, edit distance 1, and only one candidate.
    if len(words) == 1 and len(words[0]) >= 4:
        hits = {v for k, v in _FUZZY_WORDS.items()
                if len(k) >= 4 and _levenshtein(words[0], k) == 1}
        if len(hits) == 1:
            action, arg = hits.pop()
            return Intent(action, arg, confidence=0.75), "matched"
        if len(hits) > 1:
            return None, "ambiguous"
    return None, "no_command"


def match_intent(command: str) -> Intent | None:
    """Match an already-de-waked command string to an :class:`Intent`, or None
    if nothing is confidently recognised (the bot then stays silent)."""
    intent, _reason = _match_tokens(tokenize(command), command or "", strict=False)
    return intent


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry points
# ──────────────────────────────────────────────────────────────────────────────

def parse_detailed(text: str, wake: str | list[str] = DEFAULT_WAKE, *,
                   require_wake: bool = True) -> ParseResult:
    """Full pipeline with a reason code. ``require_wake=False`` is for a line
    that follows a wake-only utterance (the "hey DJ … skip" window)."""
    tokens = tokenize(text)
    if not tokens:
        return ParseResult(None, "empty")
    if not require_wake:
        intent, reason = _match_tokens(tokens, text, strict=False)
        return _with_songs(ParseResult(intent, reason, command=" ".join(t.text for t in tokens)))

    words = [t.text for t in tokens]
    occurrences = _find_wakes(words, _wake_list(wake))
    if not occurrences:
        return ParseResult(None, "no_wake")
    opening = [o for o in occurrences if o[3]]
    if not opening:
        # Heard, but not addressed: the wake phrase sits inside a sentence.
        _start, end, canon, _ = occurrences[0]
        return ParseResult(None, "wake_not_at_start", canon, False,
                           " ".join(t.text for t in tokens[end:]))
    _start, end, canon, _ = opening[0]
    # "hey DJ, hey DJ, skip" — someone repeating the wake word isn't chatter.
    starts_at = {o[0]: o[1] for o in occurrences}
    while end in starts_at:
        end = starts_at[end]
    rest = tokens[end:]
    command = " ".join(t.text for t in rest)
    if not _strip_affixes(rest):
        return ParseResult(None, "wake_only", canon, True, "")
    intent, reason = _match_tokens(rest, text, strict=False)
    if intent is not None:
        return _with_songs(ParseResult(intent, "matched", canon, True, command))
    return ParseResult(None, reason, canon, True, command)


def _with_songs(result: ParseResult) -> ParseResult:
    """Split one play request into its songs (see :func:`split_songs`)."""
    intent = result.intent
    if intent is None or intent.action != "play" or not intent.arg:
        return result
    words = result.command.split()
    expected = next((_COUNTS[w] for i, w in enumerate(words[:-1])
                     if w in _COUNTS and words[i + 1] in ("songs", "tracks", "tunes")), None)
    songs, unsure = split_songs(intent.arg, expected)
    unsure = unsure or (len(songs) == 1 and bool(_LISTY.search(songs[0])))
    first, *rest = songs
    return ParseResult(Intent("play", first, intent.confidence), result.reason, result.wake,
                       result.wake_at_start, result.command,
                       tuple(Intent("play", q, intent.confidence) for q in rest), unsure)


def interpret(text: str, wake: str | list[str] = DEFAULT_WAKE, *,
              in_window: bool = False) -> ParseResult:
    """What the bot does with one utterance: :func:`parse_detailed`, plus the
    wake-then-command window — inside it (the same person said only the wake
    phrase moments ago) a command needs no wake word. The voice cog and
    scripts/voice_eval.py both decide through this, so the benchmark measures
    the bot's real behaviour."""
    result = parse_detailed(text, wake)
    if result.intent is None and in_window and result.reason != "wake_only":
        followup = parse_detailed(text, wake, require_wake=False)
        if followup.intent is not None:
            return ParseResult(followup.intent, "matched", None, False, followup.command,
                               followup.more, followup.list_like)
    return result


def parse(text: str, wake: str | list[str] = DEFAULT_WAKE, *,
          require_wake: bool = True) -> Intent | None:
    """Wake-gate then intent-match. None = not addressed to the bot, or not a
    recognised command — callers stay silent."""
    return parse_detailed(text, wake, require_wake=require_wake).intent


# ── confirmations ─────────────────────────────────────────────────────────────

_YES = frozenset({"yes", "yeah", "yep", "yup", "yea", "ya", "sure", "confirm",
                  "confirmed", "affirmative", "ok", "okay", "absolutely",
                  "definitely", "correct"})
_YES_PHRASES = (("do", "it"), ("go", "ahead"), ("go", "for", "it"))
_NO = frozenset({"no", "nope", "nah", "cancel", "nevermind", "abort", "dont",
                 "not", "never", "negative", "wait"})
_NO_PHRASES = (("never", "mind"), ("forget", "it"), ("hold", "on"))
_ANSWER_FILLER = frozenset({"um", "uh", "er", "oh", "well", "please", "it", "the",
                            "that", "this", "do", "go", "ahead", "for", "dj", "hey",
                            "queue", "music", "song", "everything", "i", "said",
                            "thanks", "thank", "you", "so"})


def _has_seq(words: list[str], seqs) -> bool:
    return any(tuple(words[i:i + len(q)]) == q
               for q in seqs for i in range(len(words) - len(q) + 1))


def parse_confirmation(text: str, wake: str | list[str] | None = None,
                       action: str | None = None) -> str | None:
    """A reply to "are you sure?" → "yes", "no", or None (not an answer).

    Any refusal word wins ("yeah no", "yes but don't stop" → no). "Yes" needs
    the whole reply to be yes-words, filler, the wake word or the action being
    confirmed ("yes stop it") — so unrelated chatter isn't an answer and
    doesn't use up the pending confirmation."""
    words = [t.text for t in tokenize(text)]
    if wake:
        for _start, end, _canon, at_start in _find_wakes(words, _wake_list(wake)):
            if at_start:
                words = words[end:]
                break
    if not words:
        return None
    if any(w in _NO for w in words) or _has_seq(words, _NO_PHRASES):
        return "no"
    has_yes = any(w in _YES for w in words) or _has_seq(words, _YES_PHRASES)
    allowed = _YES | _ANSWER_FILLER | ({action} if action else set()) | {"but"}
    if has_yes and all(w in allowed for w in words):
        return "yes"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Long requests: one sentence, several recognised parts
# ──────────────────────────────────────────────────────────────────────────────

def _close_word(a: str, b: str) -> bool:
    return a == b or (len(a) > 3 and len(b) > 3 and _ratio(a, b) >= 0.8)


def stitch(first: str, second: str, *, max_overlap: int = 10) -> str:
    """Join the transcripts of two consecutive parts of one long utterance.
    The front end splits speech at its length cap with ~0.8 s of overlap, so
    the end of `first` and the start of `second` usually repeat a few words
    ("… Panic at the Disco, shuffle" + "shuffle, skip the second …"). The
    longest such repeat (fuzzily: "Disco"/"disco", "Wicked"/"wicked,") is
    taken once; a word cut in half at the boundary ("Wick" + "Wicked") too.
    With no repeat found, the parts are simply joined."""
    a, b = (first or "").strip(), (second or "").strip()
    if not a or not b:
        return a or b
    ta, tb = tokenize(a), tokenize(b)
    wa, wb = [t.text for t in ta], [t.text for t in tb]
    best = 0
    for k in range(min(max_overlap, len(wa), len(wb)), 0, -1):
        if all(_close_word(x, y) for x, y in zip(wa[-k:], wb[:k])):
            best = k
            break
    if best:
        cut = tb[best - 1].end
        rest = b[cut:].lstrip(" ,.;:!?")
        return f"{a} {rest}".strip() if rest else a
    # A word cut at the boundary: the last word of the first part is the
    # start of the second part's first word.
    if wa and wb and len(wa[-1]) >= 2 and wb[0].startswith(wa[-1]) and wb[0] != wa[-1]:
        return f"{a[:ta[-1].start].rstrip()} {b}".strip()
    return f"{a} {b}"
