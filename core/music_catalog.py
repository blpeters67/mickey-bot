"""
Music catalogue lookups for the voice DJ — the facts a language model can't
be trusted to recite: an album's track list in order (and which edition),
an artist's most popular songs, their newest releases — and which real song
a (possibly misheard) title means: "I Write Him's Not Travis D's" by "Young
Gravy" is Yung Gravy's "I Write Hymns Not Travesties" (find_song, matched
by spelling *and* sound against the artist's own songs).

Keyless public APIs: Deezer first, the iTunes Search API as a fallback.
Measured from the bot host on 2026-10-12: about 0.3 s per call for each.
Results are cached in memory for a day.

Everything here returns plain data (CatalogTrack / CatalogAlbum); the voice
cog turns tracks into resolve-at-play-time queue entries searched on YouTube
as "Artist - Title". Pure parts (track selection, picking the right album or
artist out of search results) are module functions, tested without a network
in tests/test_music_catalog.py.
"""
from __future__ import annotations

import asyncio
import difflib
import random
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from typing import Any

from .logging_setup import get_logger

log = get_logger(__name__)

DEEZER = "https://api.deezer.com"
ITUNES = "https://itunes.apple.com"
TIMEOUT = 8.0                 # whole lookup, all calls
MAX_TRACKS = 30               # most songs one spoken request may queue from an album
MAX_TOP = 10
CACHE_SECONDS = 24 * 3600
_CACHE_MAX = 256
_HEADERS = {"User-Agent": "discord-music-bot/1.0 (voice DJ catalogue lookup)",
            "Accept": "application/json"}
#: Tests set this: the real catalogue hosts are then refused (a CatalogError,
#: like being offline), while a local stand-in server still answers.
BLOCK_REAL_HOSTS = False


class CatalogError(Exception):
    """The catalogue couldn't be reached or answered with an error."""


@dataclass(frozen=True)
class CatalogTrack:
    title: str                # the song's own title, without "(Album Version)" etc.
    artist: str
    duration: int = 0         # seconds
    position: int = 0         # track number on its disc
    disc: int = 1
    version: str = ""         # "(Live In Denver)", "(2009 Remaster)", "(Demo)" …

    @property
    def query(self) -> str:
        """What YouTube is searched for (same shape as the Spotify stubs). A
        different recording — live, demo, acoustic, remix — keeps its label so
        that's what is found; a remaster or "album version" doesn't need it."""
        title = f"{self.title} {self.version}" if _RECORDING.search(self.version) else self.title
        return f"{self.artist} - {title}" if self.artist else title


@dataclass(frozen=True)
class CatalogAlbum:
    title: str
    artist: str
    tracks: tuple[CatalogTrack, ...]
    cover: str | None = None
    url: str | None = None
    released: str = ""        # YYYY-MM-DD when known
    source: str = ""          # "deezer" | "itunes"


# ── matching ──────────────────────────────────────────────────────────────────

#: Releases that are different recordings of the songs, editions with extra
#: material, and minor variants. A request that doesn't ask for one prefers
#: the plain album: different recordings least of all.
_RECORDING = re.compile(
    r"\b(live|demos?|acoustic|instrumentals?|karaoke|remix(es)?|commentary|sessions?|"
    r"tour|rehearsals?|voice memos?|sped|slowed|unplugged)\b", re.IGNORECASE)
_EXPANDED = re.compile(r"\b(deluxe|expanded|anniversary|bonus|super|special|collector'?s|"
                       r"complete)\b", re.IGNORECASE)
_VARIANT = re.compile(r"[\(\[]|\b(remaster(ed)?|edition|version|mix|re-?issue|mono|stereo)\b",
                      re.IGNORECASE)
_EDITION = re.compile(f"{_RECORDING.pattern}|{_EXPANDED.pattern}|{_VARIANT.pattern}",
                      re.IGNORECASE)
_BRACKETED_TAIL = re.compile(r"^(?P<base>.*?\S)\s*(?P<version>[\(\[][^\(\[]*[\)\]])\s*$")


def norm(text: str, *, keep_brackets: bool = False) -> str:
    """Casefolded, accents and punctuation gone, "&" → "and", no leading
    "the", and (unless `keep_brackets`) parenthesised extras dropped:
    "Vices & Virtues (Deluxe)" and "vices and virtues" compare equal."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = s.replace("&", " and ").replace("+", " and ")
    if not keep_brackets:
        s = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"^the\s+", "", s)
    return " ".join(s.split())


def similarity(a: str, b: str) -> float:
    """0..1, the better of comparing with and without bracketed parts — so
    "(What's the Story) Morning Glory?" still matches when it's said in full."""
    best = 0.0
    for keep in (False, True):
        x, y = norm(a, keep_brackets=keep), norm(b, keep_brackets=keep)
        if not x or not y:
            continue
        if x == y:
            return 1.0
        best = max(best, difflib.SequenceMatcher(None, x, y).ratio())
    return best


def pick_album(candidates: list[dict], title: str, artist: str) -> dict | None:
    """The search result that is the album asked for, or None.

    `candidates` are dicts with title, artist and optional record_type, in the
    catalogue's own relevance order (the tie-breaker). A different artist or a
    clearly different title is never accepted; a deluxe/live/remaster edition
    loses to the plain album unless the request names the edition."""
    best, best_score = None, None
    for rank, c in enumerate(candidates):
        name = c.get("title", "")
        t = similarity(name, title)
        a = similarity(c.get("artist", ""), artist) if artist else 0.8
        if t < 0.6 or (artist and a < 0.6):
            continue
        score = 2 * t + a - 0.02 * rank
        score -= edition_penalty(name, title or "")
        if (c.get("record_type") or "album") in ("single",):
            score -= 0.2
        if best_score is None or score > best_score:
            best, best_score = c, score
    return best


def edition_penalty(name: str, asked: str) -> float:
    """How far a release is from what was asked for. Unasked-for: a
    different recording (live, demos…) 0.5, extra material (deluxe,
    anniversary…) 0.3, a minor variant (a remaster, a "(2019 Mix)") 0.1.
    Asked for ("the deluxe edition", "the live album"): that kind of release
    gets a bonus and the rest a 0.3 penalty."""
    for pattern, cost in ((_RECORDING, 0.5), (_EXPANDED, 0.3)):
        has, wants = bool(pattern.search(name)), bool(pattern.search(asked))
        if has and wants:
            return -0.2
        if has:
            return cost
        if wants:
            return 0.3
    if _VARIANT.search(name) and not _VARIANT.search(asked):
        return 0.1
    return 0.0


def split_version(title: str) -> tuple[str, str]:
    """ "Cave In (Album Version)" → ("Cave In", "(Album Version)")."""
    m = _BRACKETED_TAIL.match(title or "")
    return (m.group("base"), m.group("version")) if m else ((title or "").strip(), "")


def core_tracks(tracks) -> list[CatalogTrack]:
    """The album itself out of an edition nobody asked for: no live takes,
    demos or remixes tacked on, each song once (first occurrence). Falls back
    to everything if that would leave nothing."""
    seen: set[str] = set()
    out: list[CatalogTrack] = []
    for t in tracks:
        if _RECORDING.search(t.version):
            continue
        key = norm(t.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out or list(tracks)


def pick_artist(candidates: list[dict], name: str) -> dict | None:
    """The artist asked for: a close name match — by spelling, or by sound
    ("Young Gravy" is Yung Gravy) — the most-followed of those (catalogues
    carry junk duplicates with no fans)."""
    good = [c for c in candidates if name_score(name, c.get("name", "")) >= 0.75]
    if not good:
        return None
    return max(good, key=lambda c: (name_score(name, c.get("name", "")) >= 0.95,
                                    c.get("fans") or 0))


# ── which song: by spelling and by sound ──────────────────────────────────────

#: Spellings that sound like something simpler, applied before reducing a
#: phrase to its consonants: "sighted" ~ "sited", "phone" ~ "fone".
_SOUND_SPELLINGS = (("ight", "it"), ("igh", "i"), ("ough", "o"), ("tion", "son"),
                    ("sch", "sk"), ("ph", "f"), ("ck", "k"), ("sh", "s"), ("ch", "s"),
                    ("th", "t"), ("qu", "k"), ("wh", "w"), ("gh", "g"), ("kn", "n"),
                    ("wr", "r"), ("dg", "j"), ("mb", "m"), ("ce", "se"), ("ci", "si"),
                    ("cy", "sy"), ("ge", "je"), ("gi", "ji"))
#: Voiced and voiceless pairs sound alike to a speech recogniser; vowels,
#: "h", "w" and "y" carry little — so they're dropped and the pairs merged.
_SOUND_CLASSES = str.maketrans({"b": "p", "d": "t", "g": "k", "c": "k", "q": "k", "z": "s",
                                "v": "f", "n": "m", "x": "ks", "a": None, "e": None,
                                "i": None, "o": None, "u": None, "y": None, "h": None,
                                "w": None, " ": None})


def sound_key(text: str) -> str:
    """A rough key for how a phrase sounds: its consonants, similar ones
    merged, repeats collapsed. "I Write Him's Not Travis D's" and "I Write
    Hymns Not Travesties" both become "rtmsmtrfsts"; "sighted off", "the
    set off" and "Set It Off" all become "stf"."""
    s = norm(text)
    s = re.sub(r"\d+", lambda m: f" {m.group(0)} ", s)
    for a, b in _SOUND_SPELLINGS:
        s = s.replace(a, b)
    s = s.translate(_SOUND_CLASSES)
    return re.sub(r"(.)\1+", r"\1", s)


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def title_score(heard: str, title: str) -> float:
    """0..1: how likely `heard` (a title as the speech recogniser wrote it, or
    as the LLM understood it) means the real `title` — by spelling, or by
    sound when the spelling is off ("Him's Not Travis D's" ~ "Hymns Not
    Travesties"). A short title can't score high on sound alone ("Stop" ~
    "Step"): the less there is to compare, the less sound counts."""
    best = 0.0
    for keep in (False, True):
        h, t = norm(heard, keep_brackets=keep), norm(title, keep_brackets=keep)
        if not h or not t:
            continue
        if h == t:
            return 1.0
        spelled = _ratio(h, t)
        kh, kt = sound_key(h), sound_key(t)
        sounded = _ratio(kh, kt)
        weight = min(1.0, min(len(kh), len(kt)) / 7)
        best = max(best, spelled, sounded * (0.55 + 0.4 * weight))
    return best


def name_score(heard: str, name: str) -> float:
    """title_score for artist names, plus: a duo/feature credit that starts
    with the asked-for name ("Yung Gravy & bbno$" for "Yung Gravy") counts
    nearly fully, and so does the same name written without its spaces
    ("FallOutBoy" for "Fall Out Boy")."""
    if not heard or not name:
        return 0.0
    h, n = norm(heard), norm(name)
    if h and n and (n.startswith(h + " and ") or n.startswith(h + " feat")
                    or n.startswith(h + " x ") or n.startswith(h + " with ")):
        return 0.95
    if h and n and h.replace(" ", "") == n.replace(" ", ""):
        return 1.0
    return title_score(heard, name)


#: An artist named in a request is only matched by a candidate this close
#: (spelling or sound, or a credit that includes them). Lower agreement —
#: "Prince" for "Priest" scores 0.67 — is never accepted as that artist: it
#: is only offered back as a "did you mean…?".
ARTIST_SURE = 0.85
ARTIST_NEAR = 0.6


@dataclass(frozen=True)
class SongMatch:
    """The catalogue's answer to "which song did they mean?"."""
    track: CatalogTrack
    score: float                    # 0..1, title (and artist) agreement
    sure: bool                      # good and clearly better than the rest
    others: tuple[CatalogTrack, ...] = ()   # the next best, for asking the LLM
    #: Same (or nearly the same) title by an artist that only resembles the
    #: one named — "Prince - Call My Name" when Priest was asked for. Never
    #: the answer; a suggestion if nothing by the named artist turns up.
    near: tuple[CatalogTrack, ...] = ()
    #: Which reading of the artist found it: "said" (as the speech recogniser
    #: wrote it), "named" (as the LLM put it), "matched" (the catalogue's own
    #: artist for a misheard name: "the set off" → Set It Off), "" (no artist).
    basis: str = ""
    why: str = ""                   # sanitized: the scores that decided it
    #: False: nothing by the named artist — `track` is then only the best of
    #: `near`, a suggestion, never something to play.
    matched: bool = True


#: Titles of a different recording ("Skeleton (Slowed & Reverb)", "Why Do I
#: (Sped Up)") only win when that's what was asked for.
def _recording_penalty(title: str, version: str, asked: str) -> float:
    label = f"{title} {version}"
    return 0.12 if _RECORDING.search(label) and not _RECORDING.search(asked or "") else 0.0


SURE_TITLE = 0.86        # a title this close (spelling or sound) is the song…
SURE_MARGIN = 0.08       # …if nothing else is nearly as close


def pick_song(candidates, titles, artist: str = "", *, said: str = "",
              artist_aliases=()) -> SongMatch | None:
    """The candidate (a CatalogTrack) that best fits any of `titles` — what
    the LLM understood and what the recogniser heard — by `artist` if given.

    An artist that was named must be matched (ARTIST_SURE, by spelling, sound
    or a credit that includes them — or one of `artist_aliases`, the
    catalogue's own name for that artist): another artist's song is never
    the answer, however popular or similarly spelled ("Prince" for "Priest").
    Such near-misses come back in `near`, for a "did you mean" only.

    When the words actually `said` clearly name one song and the LLM's
    version names another, the words win: the LLM once swapped "Cordial"
    (which it didn't know) for the Set It Off song that was playing.

    With no artist, the same title by several artists is ambiguous: the
    catalogue's first (most relevant / popular) one is only taken when its
    title is exact. Ties keep the catalogue's own order."""
    titles = [t for t in dict.fromkeys(" ".join((t or "").split()) for t in titles) if t]
    if not titles:
        return None
    said = " ".join((said or "").split())
    names = [n for n in (artist, *artist_aliases) if n]
    scored = []
    near = []
    seen: set[tuple[str, str]] = set()
    for rank, c in enumerate(candidates):
        key = (norm(f"{c.title} {c.version}", keep_brackets=True), norm(c.artist))
        if not key[0] or key in seen:
            continue                      # (a different take of it is kept: it may be the one)
        seen.add(key)
        t = max(title_score(h, c.title) for h in titles)
        if names:
            a = max(name_score(n, c.artist) for n in names)
            if a < ARTIST_SURE:
                if a >= ARTIST_NEAR and t >= SURE_TITLE:
                    near.append((t, a, -rank, c))
                continue
        else:
            a = 1.0
        s = t * (0.85 + 0.15 * a) - _recording_penalty(c.title, c.version, " ".join(titles))
        if said and title_score(said, c.title) >= SURE_TITLE:
            s += 0.1
        scored.append((s, t, -rank, c, a))
    near.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    near_tracks = tuple(x[3] for x in near[:3])
    if not scored:
        if near_tracks:
            return SongMatch(near_tracks[0], 0.0, False, (), near_tracks,
                             why=f"no song by {artist!r} matched; similar artists only: "
                                 + ", ".join(f"{x[3].artist} ({x[1]:.2f})" for x in near[:3]),
                             matched=False)
        return None
    scored.sort(key=lambda x: (x[0], x[2]), reverse=True)
    best = scored[0]
    rival = next((x for x in scored[1:] if norm(x[3].title) != norm(best[3].title)), None)
    sure = best[1] >= SURE_TITLE and (rival is None or best[0] - rival[0] >= SURE_MARGIN
                                      or best[1] == 1.0 and rival[1] < 1.0)
    why = (f"best {best[3].artist} - {best[3].title} (title {best[1]:.2f}, artist {best[4]:.2f}, "
           f"score {best[0]:.2f})")
    if rival is not None:
        why += f"; next other title {rival[3].title!r} ({rival[0]:.2f})"
    if not names:
        # The same title by other artists: without an artist to go by, only
        # an exact title at the top of the catalogue's own ranking will do.
        same = [x for x in scored[1:] if norm(x[3].title) == norm(best[3].title)
                and norm(x[3].artist) != norm(best[3].artist)]
        if same:
            top = best[2] >= max(x[2] for x in same)
            why += f"; same title by {len(same)} other artist(s)"
            if not (best[1] >= 0.97 and top):
                sure = False
    return SongMatch(best[3], round(best[0], 3), sure, tuple(x[3] for x in scored[1:8]),
                     near_tracks, why=why)


_ALL = {"all", "whole", "everything", "full", "entire", "every", "complete", "*"}
_ORDINALS = {w: n for n, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
    "twentieth".split(), 1)}
_NUMBERS = {w: n for n, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split(), 1)} | {
    "couple": 2, "pair": 2, "few": 3}


def said_as(n: int) -> list[str]:
    """How track number `n` may be said: "7", "7th", "seventh", "seven"."""
    words = [str(n), f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"]
    words += [w for w, v in _ORDINALS.items() if v == n] + [w for w, v in _NUMBERS.items()
                                                             if v == n and w not in ("couple", "pair", "few")]
    return words


# ── which tracks of an album: a small selector language ───────────────────────
#
# "all", "1-3", "7, 9", "first 3", "last 2", "random 3", "first half",
# "second half", "all except last", "everything but 2 and 5", "first half
# except its last", "3rd to last" — written by the LLM in that canonical form,
# or the same said in words ("the seventh and ninth songs", "the latter half",
# "all but the first one", "the first seven but leave out the sixth").
#
# Rules (documented, deterministic):
#   * Track numbers are the album's own positions, 1-based, in the edition
#     that was found. Numbers past its end are an error; a range is clipped
#     to it (and that's reported).
#   * "first half" is ceil(N/2) tracks, "second/latter/back half" the rest:
#     11 tracks → 1-6 and 7-11, no overlap, nothing left out.
#   * After except/but/without/…, "first N", "last N" and halves are counted
#     within what is being trimmed: "first half except its last" (11 tracks)
#     is 1-5; "all except the last" is 1-(N-1). Numbers stay album positions.
#   * Order: as asked ("9 and 7" → 9, 7); ranges and halves in album order.
#     A track named twice on purpose ("2, 2") is queued twice; overlapping
#     ranges are not ("1-6 and 5-8" → 1-8). A repeat count is the request's
#     business (the planner's `count`), not the selector's.
#   * Words it doesn't understand ("the good ones", "the middle one") make
#     the whole selection unsupported, never silently ignored — the bot asks.

_LEADS = {"first": "first", "top": "first", "opening": "first", "last": "last",
          "final": "last", "closing": "last", "random": "random", "any": "random"}
_RANGE_WORDS = {"-", "to", "through", "thru", "till", "until"}
_SEPARATORS = {",", "and", "plus", "also", "then", "&"}
_EXCEPT = {"except", "but", "without", "minus", "excluding", "besides", "not", "skip",
           "skipping", "save", "sans", "remove", "removing", "exclude", "omit", "omitting",
           "drop", "dropping"}
_EXCEPT_PAIRS = {("other", "than"), ("apart", "from"), ("leave", "out"), ("leaving", "out"),
                 ("except", "for"), ("aside", "from"), ("bar", "the")}
#: Words of an exclusion said twice over ("but leave out the sixth", "except
#: for the last"): fine inside the part being left out.
_EXCEPT_WORDS = {w for pair in _EXCEPT_PAIRS for w in pair} - {"the"}
_HALF_SECOND = {"second", "latter", "back", "later", "other", "last", "final", "2nd"}
_HALF_FIRST = {"first", "front", "former", "1st", "opening"}
#: Words that say nothing about which tracks.
_SEL_FILLER = {"the", "a", "an", "song", "songs", "track", "tracks", "number", "numbers", "no",
               "num", "of", "off", "from", "on", "in", "album", "albums", "record", "ep",
               "its", "it", "it's", "ones", "them", "please", "just", "only", "both", "those",
               "these", "that", "this", "is", "are", "be", "up", "for", "all's", "over",
               "way", "end", "via", "each", "tune", "tunes", "cut", "cuts", "piece", "pieces",
               "thing", "whole's", "do", "dont", "don't", "want", "i", "me", "us", "we",
               "you", "can", "could", "would", "will", "should", "and", "or", "too", "as",
               "well", "go", "going", "play", "queue", "cue", "add", "put", "after", "half's"}
_SEL_TOKEN = re.compile(r"\d+(?:st|nd|rd|th)?|[a-z]+(?:'[a-z]+)?|[-,*&]")


class SelectionError(ValueError):
    """A track selection that can't be followed (unknown words, a track the
    album doesn't have). The message says why, in plain words."""


@dataclass(frozen=True)
class SelTerm:
    """One piece of a selection. kind: all | index | range | first | last |
    random | half | from_end."""
    kind: str
    a: int = 0                # index, range start, count, half (1/2), from-the-end position
    b: int = 0                # range end

    def text(self) -> str:
        return {"all": "all", "index": str(self.a), "range": f"{self.a}-{self.b}",
                "first": f"first {self.a}", "last": f"last {self.a}",
                "random": f"random {self.a}",
                "half": "first half" if self.a == 1 else "second half",
                "from_end": f"{self.a} from the end"}[self.kind]


@dataclass(frozen=True)
class Selection:
    include: tuple[SelTerm, ...]
    exclude: tuple[SelTerm, ...] = ()

    def text(self) -> str:
        inc = ", ".join(t.text() for t in self.include)
        return inc + (" except " + ", ".join(t.text() for t in self.exclude) if self.exclude else "")


@dataclass(frozen=True)
class Selected:
    """What a selection picked from one album: the tracks in order (repeats
    included), and a note when something was clipped or capped."""
    tracks: tuple[CatalogTrack, ...]
    note: str = ""


def _sel_tokens(text: str) -> list:
    s = (text or "").casefold()
    for a, b in (("–", "-"), ("—", "-"), ("+", " and "), ("’", "'"), ("#", " ")):
        s = s.replace(a, b)
    out: list = []
    for w in _SEL_TOKEN.findall(s):
        prev = out[-1] if out else None
        if w in ("one", "ones") and prev is not None and (
                (prev[0] == "num" and prev[2]) or (prev[0] == "word" and prev[1] in _LEADS)):
            continue                     # "the seventh one", "the last one": that "one" isn't a 1
        if w[0].isdigit():
            n = int(re.match(r"\d+", w).group(0))
            out.append(("num", n, not w.isdigit()))            # (kind, value, said as an ordinal)
        elif w in _LEADS:
            out.append(("word", w, False))                     # "first 3" is a count, not track 1
        elif w in _ORDINALS:
            out.append(("num", _ORDINALS[w], True))
        elif w in _NUMBERS:
            out.append(("num", _NUMBERS[w], False))
        else:
            out.append(("word", w, False))
    return out


def _split_except(toks: list) -> tuple[list, list, bool]:
    """(what to include, what to leave out, whether an exclusion was said)."""
    for i, t in enumerate(toks):
        if t[0] != "word":
            continue
        w = t[1]
        nxt = toks[i + 1][1] if i + 1 < len(toks) and toks[i + 1][0] == "word" else None
        if (w, nxt) in _EXCEPT_PAIRS:
            return toks[:i], toks[i + 2:], True
        if w in _EXCEPT:
            return toks[:i], toks[i + 1:], True
    return toks, [], False


def _terms(toks: list, *, excluding: bool = False) -> list[SelTerm]:
    """The terms in one part (include or exclude) of a selection."""
    # First pass: words into grammar symbols, dropping filler.
    sym: list = []
    i = 0
    while i < len(toks):
        kind, v, ordinal = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if nxt is not None and nxt[:2] == ("word", "half"):
            if kind == "num" and v in (1, 2) and (ordinal or v == 1):
                sym.append(("half", v))
                i += 2
                continue
            if kind == "word" and (v in _HALF_SECOND or v in _HALF_FIRST):
                sym.append(("half", 1 if v in _HALF_FIRST else 2))
                i += 2
                continue
        if kind == "num":
            sym.append(("num", v, ordinal))
        elif v == "half" or v == "halves":
            if i and toks[i - 1][:2] in (("word", "both"), ("word", "two")):
                sym.append(("all",))                           # "both halves"
            else:
                raise SelectionError("which half")
        elif v in _LEADS:
            sym.append(("lead", _LEADS[v]))
        elif v in _RANGE_WORDS:
            sym.append(("-",))
        elif v in _SEPARATORS or v == ",":
            sym.append((",",))
        elif v in _ALL:
            sym.append(("all",))
        elif v == "*":
            sym.append(("all",))
        elif v == "one" or v in _SEL_FILLER or (excluding and (
                v in _EXCEPT or v in _EXCEPT_WORDS)):
            pass
        elif v == "both" and nxt is not None and nxt[:2] == ("word", "halves"):
            pass
        else:
            raise SelectionError(f"“{v}”")
        i += 1
    # Second pass: symbols into terms.
    out: list[SelTerm] = []
    i = 0
    while i < len(sym):
        s = sym[i]
        nxt = sym[i + 1] if i + 1 < len(sym) else None
        nxt2 = sym[i + 2] if i + 2 < len(sym) else None
        if s[0] == ",":
            i += 1
            continue
        if s[0] == "all":
            out.append(SelTerm("all"))
            i += 1
            continue
        if s[0] == "half":
            out.append(SelTerm("half", s[1]))
            i += 1
            continue
        if s[0] == "num":
            # "3rd to last": counted from the end (not "3 to the last one").
            if nxt == ("-",) and nxt2 == ("lead", "last") and s[2] and (
                    i + 3 >= len(sym) or sym[i + 3][0] != "num"):
                out.append(SelTerm("from_end", s[1]))
                i += 3
                continue
            if nxt == ("-",) and nxt2 is not None and nxt2[0] == "num":
                a, b = sorted((s[1], nxt2[1]))
                out.append(SelTerm("range", a, b))
                i += 3
                continue
            if nxt == ("-",) and nxt2 == ("lead", "last"):
                out.append(SelTerm("range", s[1], 10 ** 6))     # "3 to the last": through the end
                i += 3
                continue
            if nxt is not None and nxt[0] == "lead" and not s[2]:
                out.append(SelTerm(nxt[1], s[1]))                # "three random", "2 last"
                i += 2
                continue
            out.append(SelTerm("index", s[1]))
            i += 1
            continue
        if s[0] == "lead":
            lead = s[1]
            if lead == "first" and nxt == ("-",) and nxt2 is not None and nxt2[0] == "num":
                out.append(SelTerm("range", 1, nxt2[1]))         # "first to fourth"
                i += 3
                continue
            if nxt is not None and nxt[0] == "num" and nxt[2] and lead != "random":
                # "first second and third" (no commas from the recogniser):
                # the first one, then the ordinal on its own.
                out.append(SelTerm(lead, 1))
                i += 1
                continue
            if nxt is not None and nxt[0] == "num" and not (nxt2 == ("-",)):
                out.append(SelTerm(lead, nxt[1]))
                i += 2
                continue
            out.append(SelTerm(lead, 1))                         # "the first", "the last one"
            i += 1
            continue
        if s[0] == "-":
            raise SelectionError("a range with no start")
        i += 1
    return out


def parse_selection(selection: str | None) -> Selection:
    """A selection said in words or written canonically → Selection. Raises
    SelectionError for anything it can't follow."""
    toks = _sel_tokens(selection or "")
    inc_toks, exc_toks, excepted = _split_except(toks)
    include = _terms(inc_toks)
    exclude = _terms(exc_toks, excluding=True) if excepted else []
    if excepted and not exclude:
        raise SelectionError("except what")
    if not include:
        include = [SelTerm("all")]
    if any(t.kind == "all" for t in include) and len(include) > 1:
        include = [t for t in include if t.kind != "all"]     # "all of the first three"
    return Selection(tuple(include), tuple(exclude))


def _term_positions(t: SelTerm, pool: list[int], n: int, rng) -> list[int]:
    """0-based album indices a term picks. `pool` is what first/last/half
    count within (the album, or for an exclusion what is being trimmed)."""
    k = len(pool)
    if t.kind == "all":
        return list(pool)
    if t.kind == "index":
        if not 1 <= t.a <= n:
            raise SelectionError(f"there's no track {t.a} (it has {n})")
        return [t.a - 1]
    if t.kind == "range":
        if t.a > n:
            raise SelectionError(f"there's no track {t.a} (it has {n})")
        return list(range(max(t.a, 1) - 1, min(t.b, n)))
    if t.kind == "first":
        return pool[:t.a]
    if t.kind == "last":
        return pool[max(k - t.a, 0):]
    if t.kind == "half":
        cut = -(-k // 2)                                       # ceil(k / 2)
        return pool[:cut] if t.a == 1 else pool[cut:]
    if t.kind == "from_end":
        if not 1 <= t.a <= k:
            raise SelectionError(f"there's no track {t.a} from the end (it has {k})")
        return [pool[k - t.a]]
    if t.kind == "random":
        return []                                              # drawn last, see apply_selection
    raise SelectionError(t.kind)


def apply_selection(tracks, sel: Selection, *, rng: random.Random | None = None) -> Selected:
    """The tracks a Selection picks, in order (see the rules above)."""
    tracks = list(tracks)
    n = len(tracks)
    if n == 0:
        return Selected(())
    album = list(range(n))
    picked: list[int] = []
    explicit: list[int] = []
    for t in sel.include:
        got = _term_positions(t, album, n, rng)
        if t.kind == "index":
            explicit.append(got[0])
            picked.append(got[0])                              # "2, 2" on purpose: twice
        else:
            picked.extend(i for i in got if i not in picked)
    # Exclusions count first/last/halves within what was picked (album order)
    # — or within the album when only random picks were asked for.
    ordered = sorted(set(picked)) or album
    drop: set[int] = set()
    for t in sel.exclude:
        drop.update(_term_positions(t, ordered, n, rng))
    out = [i for i in picked if i not in drop]
    randoms = [t for t in sel.include if t.kind == "random"]
    for t in randoms:
        pool = [i for i in album if i not in out and i not in drop]
        out.extend((rng or random).sample(pool, min(t.a, len(pool))))
    note = ""
    clipped = [t for t in sel.include if t.kind == "range" and t.b > n and t.b < 10 ** 6]
    if clipped:
        note = f"(it has {n} songs, so I stopped at {n})"
    if len(out) > MAX_TRACKS:
        note = f"(that's {len(out)} songs; I queued the first {MAX_TRACKS})"
        out = out[:MAX_TRACKS]
    return Selected(tuple(tracks[i] for i in out), note)


def select(tracks, selection: str | None, *, rng: random.Random | None = None) -> Selected:
    """parse_selection + apply_selection. Raises SelectionError."""
    return apply_selection(tracks, parse_selection(selection), rng=rng)


def select_tracks(tracks, selection: str | None, *, rng: random.Random | None = None
                  ) -> list[CatalogTrack]:
    """The tracks of an album a selection picks, or [] if it can't be
    followed (see select for why)."""
    try:
        return list(select(tracks, selection, rng=rng).tracks)
    except SelectionError:
        return []


def parse_count(value: str | None, default: int, cap: int) -> int:
    """ "3", "three", "a couple" → a count within 1..cap."""
    words = {"one": 1, "two": 2, "couple": 2, "three": 3, "few": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "some": 5, "several": 5,
             "a": 1, "an": 1}            # "a couple" is two: articles count last
    s = (value or "").casefold()
    m = re.search(r"\d+", s)
    if m:
        n = int(m.group(0))
    else:
        n = next((v for w, v in words.items() if re.search(rf"\b{w}\b", s)), default)
    return max(1, min(cap, n))


_REISSUE = re.compile(r"\b(remaster(ed)?|re-?issue|re-?recorded|version)\b", re.IGNORECASE)


def _released_order(albums: list[dict]) -> list[dict]:
    """Newest original releases first — reissues, live albums, demos and
    deluxe re-releases only if there's nothing else. (A "(feat. …)" single
    is new music; a "(2025 Remaster)" isn't.)"""
    plain = [a for a in albums
             if not (_RECORDING.search(a.get("title", "")) or _EXPANDED.search(a.get("title", ""))
                     or _REISSUE.search(a.get("title", "")))]
    pool = plain or albums
    return sorted(pool, key=lambda a: a.get("released") or "", reverse=True)


# ── the cache ─────────────────────────────────────────────────────────────────

_cache: dict[tuple, tuple[float, Any]] = {}


def _cached(key: tuple):
    hit = _cache.get(key)
    if hit is not None and hit[0] > time.monotonic():
        return hit[1]
    return None


def _remember(key: tuple, value) -> None:
    if len(_cache) >= _CACHE_MAX:
        for k in sorted(_cache, key=lambda k: _cache[k][0])[: _CACHE_MAX // 4]:
            _cache.pop(k, None)
    _cache[key] = (time.monotonic() + CACHE_SECONDS, value)


def clear_cache() -> None:
    _cache.clear()


# ── HTTP ──────────────────────────────────────────────────────────────────────

class _Http:
    """One aiohttp session for one lookup (a few calls)."""

    def __init__(self, deezer: str, itunes: str):
        self.deezer, self.itunes = deezer.rstrip("/"), itunes.rstrip("/")
        self._session = None

    async def __aenter__(self):
        import aiohttp
        self._session = aiohttp.ClientSession(headers=_HEADERS)
        return self

    async def __aexit__(self, *exc):
        await self._session.close()

    async def get(self, url: str, params: dict | None = None) -> Any:
        import aiohttp
        if BLOCK_REAL_HOSTS and url.startswith((DEEZER, ITUNES)):
            raise CatalogError(f"{url.split('/')[2]} is blocked (tests)")
        try:
            async with self._session.get(url, params=params,
                                         timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
                if r.status != 200:
                    raise CatalogError(f"HTTP {r.status} from {url.split('/')[2]}")
                data = await r.json(content_type=None)
        except asyncio.TimeoutError as e:
            raise CatalogError(f"timeout from {url.split('/')[2]}") from e
        except aiohttp.ClientError as e:
            raise CatalogError(f"{type(e).__name__} from {url.split('/')[2]}") from e
        except ValueError as e:
            raise CatalogError(f"bad JSON from {url.split('/')[2]}") from e
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise CatalogError(f"{url.split('/')[2]}: {msg}")
        return data


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ── Deezer ────────────────────────────────────────────────────────────────────

async def _deezer_album(http: _Http, title: str, artist: str) -> CatalogAlbum | None:
    data = await http.get(f"{http.deezer}/search/album",
                          {"q": f"{artist} {title}".strip(), "limit": 15})
    cands = [{"id": a.get("id"), "title": a.get("title", ""),
              "artist": (a.get("artist") or {}).get("name", ""),
              "record_type": a.get("record_type"), "cover": a.get("cover_big"),
              "url": a.get("link")} for a in (data.get("data") or []) if isinstance(a, dict)]
    best = pick_album(cands, title, artist)
    if best is None:
        return None
    return await _deezer_album_tracks(http, best)


async def _deezer_album_tracks(http: _Http, album: dict) -> CatalogAlbum | None:
    items: list[dict] = []
    url: str | None = f"{http.deezer}/album/{album['id']}/tracks"
    params: dict | None = {"limit": 100}
    while url and len(items) < 200:
        data = await http.get(url, params)
        items.extend(t for t in (data.get("data") or []) if isinstance(t, dict))
        url, params = data.get("next"), None
    def track(i: int, t: dict) -> CatalogTrack:
        full = (t.get("title") or "").strip()
        short = (t.get("title_short") or "").strip()
        version = (t.get("title_version") or "").strip()
        if not short:
            short, version = split_version(full)
        return CatalogTrack(title=short, version=version,
                            artist=((t.get("artist") or {}).get("name") or album["artist"]).strip(),
                            duration=_int(t.get("duration")),
                            position=_int(t.get("track_position"), i + 1),
                            disc=_int(t.get("disk_number"), 1))
    tracks = sorted((track(i, t) for i, t in enumerate(items) if t.get("title")),
                    key=lambda t: (t.disc, t.position))
    if not tracks:
        return None
    return CatalogAlbum(album["title"], album["artist"], tuple(tracks), album.get("cover"),
                        album.get("url"), album.get("released", ""), "deezer")


async def _deezer_artist(http: _Http, name: str) -> dict | None:
    key = ("deezer_artist", http.deezer, norm(name))
    hit = _cached(key)
    if hit is not None:
        return hit
    data = await http.get(f"{http.deezer}/search/artist", {"q": name, "limit": 10})
    cands = [{"id": a.get("id"), "name": a.get("name", ""), "fans": a.get("nb_fan")}
             for a in (data.get("data") or []) if isinstance(a, dict)]
    found = pick_artist(cands, name)
    if found is not None:
        _remember(key, found)
    return found


async def _deezer_top(http: _Http, artist: str, n: int) -> list[CatalogTrack]:
    found = await _deezer_artist(http, artist)
    if found is None:
        return []
    data = await http.get(f"{http.deezer}/artist/{found['id']}/top", {"limit": n})
    out = []
    for t in data.get("data") or []:
        if isinstance(t, dict) and t.get("title"):
            base, version = split_version(t["title"].strip())
            out.append(CatalogTrack((t.get("title_short") or base).strip(),
                                    ((t.get("artist") or {}).get("name") or found["name"]).strip(),
                                    _int(t.get("duration")),
                                    version=(t.get("title_version") or version).strip()))
    return out[:n]


def _deezer_track(t: dict, artist: str = "") -> CatalogTrack | None:
    title = (t.get("title_short") or "").strip()
    version = (t.get("title_version") or "").strip()
    if not title:
        title, version = split_version((t.get("title") or "").strip())
    name = ((t.get("artist") or {}).get("name") or artist).strip()
    return CatalogTrack(title, name, _int(t.get("duration")), version=version) if title else None


#: Deep look: this many of an artist's releases have their track lists read
#: when their 100 most popular songs and their singles don't have the title.
_DEEP_RELEASES = 12


async def _deezer_artist_songs(http: _Http, artist: str, *, deep: bool = False
                               ) -> list[CatalogTrack]:
    """An artist's songs, most popular first: their top 100 and their
    singles (a single's title is the song's), and with `deep` the tracks of
    their newest original albums/EPs too. Cached a day."""
    found = await _deezer_artist(http, artist)
    if found is None:
        return []
    key = ("artist_songs", found["id"], deep)
    hit = _cached(key)
    if hit is not None:
        return hit
    top, rel = await asyncio.gather(
        http.get(f"{http.deezer}/artist/{found['id']}/top", {"limit": 100}),
        http.get(f"{http.deezer}/artist/{found['id']}/albums", {"limit": 100}))
    out = [t for t in (_deezer_track(x, found["name"]) for x in (top.get("data") or [])
                       if isinstance(x, dict)) if t]
    releases = [{"id": a.get("id"), "title": a.get("title", ""), "artist": found["name"],
                 "record_type": a.get("record_type") or "album",
                 "released": a.get("release_date") or ""}
                for a in (rel.get("data") or []) if isinstance(a, dict) and a.get("title")]
    for r in releases:
        if r["record_type"] == "single":
            base, version = split_version(r["title"].strip())
            out.append(CatalogTrack(base, found["name"], 0, version=version))
    if deep:
        albums = [r for r in _released_order(releases) if r["record_type"] != "single"]
        lists = await asyncio.gather(*(_deezer_album_tracks(http, a)
                                       for a in albums[:_DEEP_RELEASES]),
                                     return_exceptions=True)
        for full in lists:
            if isinstance(full, CatalogAlbum):
                out.extend(full.tracks)
    _remember(key, out)
    return out


async def _deezer_find_song(http: _Http, titles: list[str], artist: str, said: str = ""
                            ) -> tuple[SongMatch | None, str | None]:
    """(the match, the catalogue's own name for `artist` if it knows that
    artist — None if not)."""
    queries = list(dict.fromkeys(q for t in titles for q in
                                 ((f"{artist} {t}".strip(), t) if artist else (t,))))[:4]
    jobs = [http.get(f"{http.deezer}/search", {"q": q, "limit": 25}) for q in queries]
    if artist:
        jobs.append(_deezer_artist(http, artist))
        jobs.append(_deezer_artist_songs(http, artist))
    res = await asyncio.gather(*jobs, return_exceptions=True)
    errors = [r for r in res if isinstance(r, BaseException)]
    if errors and len(errors) == len(res):
        raise errors[0] if isinstance(errors[0], CatalogError) else CatalogError(
            f"{type(errors[0]).__name__} from deezer")
    cands: list[CatalogTrack] = []
    entity = None
    if artist:
        found, own = res[-2], res[-1]
        if isinstance(found, dict):
            entity = found.get("name") or None
        if isinstance(own, list):
            cands += own                          # the artist's own, most popular first
    for r in res[:len(queries)]:
        if isinstance(r, dict):
            cands += [t for t in (_deezer_track(x) for x in (r.get("data") or [])
                                  if isinstance(x, dict)) if t]
    # A misheard name the catalogue resolves to a real artist ("the set off"
    # → Set It Off: by sound) counts as that artist — only theirs, though.
    aliases = (entity,) if entity and name_score(artist, entity) < ARTIST_SURE else ()
    match = pick_song(cands, titles, artist, said=said, artist_aliases=aliases)
    if artist and entity and (match is None or not match.sure):
        try:
            deep = await _deezer_artist_songs(http, artist, deep=True)
        except CatalogError:
            deep = []
        if deep:
            again = pick_song(deep + cands, titles, artist, said=said, artist_aliases=aliases)
            if again is not None and (match is None or not match.matched
                                      or (again.matched and again.score > match.score)):
                match = again
    if match is not None and match.matched and aliases:
        match = replace(match, basis="matched")
    return match, entity


async def _deezer_latest(http: _Http, artist: str, n: int) -> list[CatalogTrack]:
    found = await _deezer_artist(http, artist)
    if found is None:
        return []
    data = await http.get(f"{http.deezer}/artist/{found['id']}/albums", {"limit": 100})
    albums = [{"id": a.get("id"), "title": a.get("title", ""), "artist": found["name"],
               "released": a.get("release_date") or "", "cover": a.get("cover_big"),
               "url": a.get("link")} for a in (data.get("data") or []) if isinstance(a, dict)]
    out: list[CatalogTrack] = []
    for album in _released_order(albums)[:5]:
        full = await _deezer_album_tracks(http, album)
        for t in core_tracks(full.tracks) if full else ():
            if all(norm(t.title) != norm(o.title) for o in out):
                out.append(t)
        if len(out) >= n:
            break
    return out[:n]


# ── iTunes (fallback) ─────────────────────────────────────────────────────────

async def _itunes_album(http: _Http, title: str, artist: str) -> CatalogAlbum | None:
    data = await http.get(f"{http.itunes}/search",
                          {"term": f"{artist} {title}".strip(), "entity": "album", "limit": 15})
    cands = [{"id": a.get("collectionId"), "title": a.get("collectionName", ""),
              "artist": a.get("artistName", ""),
              "record_type": "single" if (a.get("collectionName", "").endswith(" - Single")) else "album",
              "cover": (a.get("artworkUrl100") or "").replace("100x100", "600x600") or None,
              "url": a.get("collectionViewUrl"), "released": (a.get("releaseDate") or "")[:10]}
             for a in (data.get("results") or []) if isinstance(a, dict)]
    best = pick_album(cands, title, artist)
    if best is None:
        return None
    return await _itunes_album_tracks(http, best)


async def _itunes_album_tracks(http: _Http, album: dict) -> CatalogAlbum | None:
    data = await http.get(f"{http.itunes}/lookup",
                          {"id": album["id"], "entity": "song", "limit": 200})
    songs = [s for s in (data.get("results") or [])
             if isinstance(s, dict) and s.get("wrapperType") == "track" and s.get("trackName")]
    def track(s: dict) -> CatalogTrack:
        base, version = split_version(s["trackName"].strip())
        return CatalogTrack(base, (s.get("artistName") or album["artist"]).strip(),
                            _int(s.get("trackTimeMillis")) // 1000,
                            _int(s.get("trackNumber")), _int(s.get("discNumber"), 1), version)
    tracks = sorted((track(s) for s in songs), key=lambda t: (t.disc, t.position))
    if not tracks:
        return None
    return CatalogAlbum(album["title"], album["artist"], tuple(tracks), album.get("cover"),
                        album.get("url"), album.get("released", ""), "itunes")


async def _itunes_top(http: _Http, artist: str, n: int) -> list[CatalogTrack]:
    data = await http.get(f"{http.itunes}/search", {"term": artist, "entity": "song",
                                                    "attribute": "artistTerm", "limit": 50})
    out: list[CatalogTrack] = []
    for s in data.get("results") or []:
        if not isinstance(s, dict) or similarity(s.get("artistName", ""), artist) < 0.75:
            continue
        base, version = split_version((s.get("trackName") or "").strip())
        t = CatalogTrack(base, s.get("artistName", "").strip(),
                         _int(s.get("trackTimeMillis")) // 1000, version=version)
        if t.title and all(norm(t.title) != norm(o.title) for o in out):
            out.append(t)
        if len(out) >= n:
            break
    return out


def _itunes_track(s: dict) -> CatalogTrack | None:
    base, version = split_version((s.get("trackName") or "").strip())
    if not base:
        return None
    return CatalogTrack(base, (s.get("artistName") or "").strip(),
                        _int(s.get("trackTimeMillis")) // 1000, version=version)


async def _itunes_find_song(http: _Http, titles: list[str], artist: str, said: str = ""
                            ) -> SongMatch | None:
    terms = list(dict.fromkeys(f"{artist} {t}".strip() for t in titles))[:2]
    jobs = [http.get(f"{http.itunes}/search", {"term": q, "entity": "song", "limit": 25})
            for q in terms]
    if artist:          # their songs in general: misheard titles don't search well
        jobs.append(http.get(f"{http.itunes}/search", {"term": artist, "entity": "song",
                                                       "attribute": "artistTerm", "limit": 200}))
    res = await asyncio.gather(*jobs, return_exceptions=True)
    errors = [r for r in res if isinstance(r, BaseException)]
    if errors and len(errors) == len(res):
        raise errors[0] if isinstance(errors[0], CatalogError) else CatalogError(
            f"{type(errors[0]).__name__} from itunes")
    cands = [t for r in res if isinstance(r, dict)
             for t in (_itunes_track(s) for s in (r.get("results") or []) if isinstance(s, dict))
             if t]
    return pick_song(cands, titles, artist, said=said)


async def _itunes_latest(http: _Http, artist: str, n: int) -> list[CatalogTrack]:
    data = await http.get(f"{http.itunes}/search", {"term": artist, "entity": "musicArtist",
                                                    "limit": 10})
    cands = [{"id": a.get("artistId"), "name": a.get("artistName", ""), "fans": 0}
             for a in (data.get("results") or []) if isinstance(a, dict)]
    found = pick_artist(cands, artist)
    if found is None:
        return []
    data = await http.get(f"{http.itunes}/lookup", {"id": found["id"], "entity": "album",
                                                    "sort": "recent", "limit": 25})
    albums = [{"id": a.get("collectionId"), "title": a.get("collectionName", ""),
               "artist": a.get("artistName", found["name"]),
               "released": (a.get("releaseDate") or "")[:10]}
              for a in (data.get("results") or [])
              if isinstance(a, dict) and a.get("wrapperType") == "collection"]
    out: list[CatalogTrack] = []
    for album in _released_order(albums)[:5]:
        full = await _itunes_album_tracks(http, album)
        for t in core_tracks(full.tracks) if full else ():
            if all(norm(t.title) != norm(o.title) for o in out):
                out.append(t)
        if len(out) >= n:
            break
    return out[:n]


# ── public API ────────────────────────────────────────────────────────────────

async def _first(kind: str, key: tuple, attempts) -> Any:
    """Try each source in turn; a source that errors falls through to the
    next, one that answers "not found" is believed only if every source
    agrees. Raises CatalogError if no source could be asked at all."""
    hit = _cached(key)
    if hit is not None:
        return hit
    errors: list[str] = []
    result = None
    for source, fn in attempts:
        try:
            result = await fn()
        except CatalogError as e:
            errors.append(f"{source}: {e}")
            continue
        except Exception as e:                  # a parsing surprise is not fatal
            log.warning("catalogue %s via %s failed: %s", kind, source, e)
            errors.append(f"{source}: {type(e).__name__}")
            continue
        if result:
            _remember(key, result)
            return result
    if errors and len(errors) == len(attempts):
        raise CatalogError("; ".join(errors))
    return result


async def find_album(title: str, artist: str = "", *, deezer: str = DEEZER,
                     itunes: str = ITUNES) -> CatalogAlbum | None:
    """The album's tracks in order, or None if no catalogue knows it."""
    title, artist = " ".join((title or "").split()), " ".join((artist or "").split())
    if not title:
        return None
    async with _Http(deezer, itunes) as http:
        album = await _first("album", ("album", norm(title), norm(artist)), [
            ("deezer", lambda: _deezer_album(http, title, artist)),
            ("itunes", lambda: _itunes_album(http, title, artist))])
    if album is not None and edition_penalty(album.title, title) >= 0.3:
        # Only a deluxe/anniversary/live edition was found, and that isn't
        # what was asked for: keep the album's own songs, once each.
        album = CatalogAlbum(album.title, album.artist, tuple(core_tracks(album.tracks)),
                             album.cover, album.url, album.released, album.source)
    return album


async def top_tracks(artist: str, n: int = 5, *, deezer: str = DEEZER,
                     itunes: str = ITUNES) -> list[CatalogTrack]:
    """An artist's most popular songs, most popular first."""
    artist, n = " ".join((artist or "").split()), max(1, min(MAX_TOP, n))
    if not artist:
        return []
    async with _Http(deezer, itunes) as http:
        return await _first("top", ("top", norm(artist), n), [
            ("deezer", lambda: _deezer_top(http, artist, n)),
            ("itunes", lambda: _itunes_top(http, artist, n))]) or []


async def latest_tracks(artist: str, n: int = 1, *, deezer: str = DEEZER,
                        itunes: str = ITUNES) -> list[CatalogTrack]:
    """Songs from an artist's newest original releases, newest first."""
    artist, n = " ".join((artist or "").split()), max(1, min(MAX_TOP, n))
    if not artist:
        return []
    async with _Http(deezer, itunes) as http:
        return await _first("latest", ("latest", norm(artist), n), [
            ("deezer", lambda: _deezer_latest(http, artist, n)),
            ("itunes", lambda: _itunes_latest(http, artist, n))]) or []


#: A model's reading of the artist is only tried when it could be what was
#: misheard: this close by spelling-and-sound ("saying chaos" → Saint Chaos
#: 0.87, "the set off" → Set It Off 0.82) — or the very same consonants
#: ("sighted off" → Set It Off) — and never when the name as said is itself
#: a real artist. "Priest" → Prince (0.67, p-r-s-t vs p-r-n-s) is a different
#: band, not a mishearing, and isn't tried.
_CORRECTION_MIN = 0.8


def plausible_correction(heard: str, named: str) -> bool:
    """Could `named` (the model's artist) be what the recogniser misheard as
    `heard`? See _CORRECTION_MIN."""
    if not heard or not named:
        return True
    if name_score(heard, named) >= _CORRECTION_MIN:
        return True
    a, b = sound_key(heard), sound_key(named)
    return bool(a) and a == b


async def _find_song_once(http: _Http, titles: list[str], artist: str, said: str
                          ) -> tuple[SongMatch | None, str | None, list[str]]:
    """Deezer, then iTunes if Deezer had nothing or failed. (match, Deezer's
    name for the artist, errors)."""
    errors: list[str] = []
    match, entity = None, None
    try:
        match, entity = await _deezer_find_song(http, titles, artist, said)
    except CatalogError as e:
        errors.append(f"deezer: {e}")
    if match is None or not match.matched:
        try:
            other = await _itunes_find_song(http, titles, artist, said)
        except CatalogError as e:
            errors.append(f"itunes: {e}")
            other = None
        if other is not None and (match is None or other.matched):
            match = other
    return match, entity, errors


async def find_song(title: str, artist: str = "", *, heard: str = "", heard_artist: str = "",
                    deezer: str = DEEZER, itunes: str = ITUNES) -> SongMatch | None:
    """Which real song is meant: `title` as the LLM understood it, `heard` as
    the recogniser wrote it (either may be misheard), by `artist` (the LLM's
    reading) / `heard_artist` (the words said after "by") if known.

    Evidence first for what was said: the artist as heard is tried first,
    and if the catalogue knows an artist by that name, only their songs can
    be the answer — the LLM's "correction" to another artist (Priest →
    Prince) is not tried at all. Only when the name as heard isn't a known
    artist ("saying chaos") is the LLM's reading tried, and only if it is
    plausibly what was misheard. Matching is by spelling and by sound, in
    the catalogue's search and among the artist's own songs, so newer songs
    the LLM doesn't know are found too.

    None if nothing comes close. A SongMatch that isn't `sure` carries the
    runners-up so the caller can ask which; one with `matched` False has only
    songs by similar-sounding other artists (suggestions). Raises
    CatalogError if no catalogue could be asked."""
    titles = [" ".join((t or "").split()) for t in (title, heard)]
    titles = [t for t in dict.fromkeys(titles) if t]
    artist = " ".join((artist or "").split())
    heard_artist = " ".join((heard_artist or "").split())
    if not titles:
        return None
    key = ("song", tuple(norm(t) for t in titles), norm(artist), norm(heard_artist))
    hit = _cached(key)
    if hit is not None:
        return hit
    hyps: list[tuple[str, str]] = []
    if heard_artist:
        hyps.append((heard_artist, "said"))
        if artist and name_score(heard_artist, artist) < 0.9:
            hyps.append((artist, "named"))
    elif artist:
        hyps.append((artist, "named"))
    else:
        hyps.append(("", ""))
    best: SongMatch | None = None
    errors: list[str] = []
    asked = 0
    real_said = None
    async with _Http(deezer, itunes) as http:
        for name, basis in hyps:
            if basis == "named" and heard_artist:
                if real_said:
                    log.info("catalogue: not trying the AI's artist %r — %r is a real artist",
                             name, real_said)
                    break
                if not plausible_correction(heard_artist, name):
                    log.info("catalogue: not trying the AI's artist %r for %r (a different "
                             "name, not a mishearing)", name, heard_artist)
                    break
            match, entity, errs = await _find_song_once(http, titles, name, heard)
            asked += 1
            errors += errs
            if basis == "said" and entity and name_score(heard_artist, entity) >= ARTIST_SURE:
                real_said = entity
            if match is not None and not match.basis:
                match = replace(match, basis=basis)
            if match is not None and match.matched and match.sure:
                best = match
                break
            if match is not None and (best is None or (match.matched and not best.matched)
                                      or (match.matched == best.matched
                                          and match.score > best.score)):
                best = match
    if best is None and errors and len(errors) >= 2 * asked:
        raise CatalogError("; ".join(errors))
    if best is not None:
        _remember(key, best)
    return best


async def artist_songs(artist: str, *, deezer: str = DEEZER, itunes: str = ITUNES
                       ) -> list[CatalogTrack]:
    """An artist's songs — most popular first, then singles and album tracks
    — for "which of their songs did they mean?" (by description). [] if the
    artist isn't found; raises CatalogError if the catalogue can't be asked."""
    artist = " ".join((artist or "").split())
    if not artist:
        return []
    async with _Http(deezer, itunes) as http:
        songs = await _deezer_artist_songs(http, artist, deep=True)
    out, seen = [], set()
    for t in songs:
        k = norm(t.title)
        if k and k not in seen and not _RECORDING.search(f"{t.title} {t.version}"):
            seen.add(k)
            out.append(t)
    return out
