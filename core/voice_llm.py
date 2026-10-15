"""
Voice DJ, tier 2: an LLM that works out what was actually asked for.

The grammar (core/voice_intent.py) handles the everyday controls instantly
and offline. Everything else said after the wake word goes to an LLM, once,
through OpenRouter's OpenAI-compatible API with a strict JSON schema: free
phrasing and several commands at once ("turn it down and skip this one"),
questions ("who sings this?") — and every song request. A song request is
never searched on YouTube as said: "play the first three songs off Panic! At
The Disco's Vices and Virtues album" becomes an *album* lookup (tracks 1-3,
found in a music catalogue — core/music_catalog.py), "the song from Top Gun"
becomes "Kenny Loggins - Danger Zone". The answer is one of:

  * ``commands`` — actions in the order asked: songs (one title + artist per
    song, an album's tracks, an artist's most popular or newest songs — for
    the end of the queue, or "next"/"now") and controls (skip, pause,
    resume, restart, stop, clear, shuffle, volume, loop song/queue). They
    are only *proposals*: checked against the words first (contradicts,
    settle_when, settle_tracks), then the cog runs every one through the
    same checks as a grammar command (caps, a yes/no for stop/clear/very
    loud, consent, staleness) and runs them in order.
  * ``question`` — a short spoken answer ("who sings this?", "what's the
    capital of Australia?"), which the cog says with TTS and posts.
  * ``other`` — not meant for the bot; it stays quiet.

Model (measured on the bot's own host, through this module — see
MEASURED below): google/gemini-2.5-flash-lite, with openai/gpt-4.1-nano as the
automatic fallback when Google is unavailable (a different provider on
purpose). Configurable (VOICE_LLM_MODEL / VOICE_LLM_FALLBACKS); any
OpenAI-compatible endpoint works (VOICE_LLM_URL), including a local
llama-server — the model must support JSON-schema output and temperature.

Pure parts (request building, answer validation) are module functions so they
can be tested without a network. Tests: tests/test_voice_llm.py.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, replace

from . import music_catalog
from .formatters import redact_secrets
from .logging_setup import get_logger
from .voice_intent import MAX_VOICE_PLAYS, SONG_ACTIONS, Intent, parse_number

log = get_logger(__name__)

ACTIONS = ("play", "find", "album", "top", "latest", "skip", "pause", "resume", "restart",
           "stop", "clear", "shuffle", "volume", "loop")
MAX_ACTIONS = 8
MAX_SONG_ACTIONS = MAX_VOICE_PLAYS    # song entries per request (an album counts once)
MAX_QUERY_CHARS = 200
MAX_REPLY_CHARS = 400

SYSTEM_PROMPT = """You are the voice assistant of a Discord music bot. Someone in the voice channel said the wake word and then the words below (speech-to-text: expect misheard words; "tip this" is "skip this").
Work out what they actually want and answer with JSON:
- kind "commands": they want to control the music. List every action they want, in the order asked:
  play: one specific song per entry; query = the song's real title, artist = who performs it, heard = the title as the speech-to-text wrote it ("" if they named none). Work out the actual song instead of repeating their words: "the theme from Titanic" is My Heart Will Go On by Celine Dion; for "something upbeat" choose real songs that fit. Fix mishearings of names you know. A title you don't know may be newer than you: keep it as said and never put a different song in its place. Titles and band names can contain "and" (Rock and Roll All Nite, Simon and Garfunkel), so only split clearly separate songs.
  find: a song they describe without naming it, when you aren't sure which song it is ("that Yung Gravy song that samples Panic! At The Disco", "the new one by X"): query = what to look up online, artist = the performer if known.
  album: songs from one album; query = album title, artist = artist, value = which tracks: "all", "1-3", "4", "last 2" or "random 3". "The first three songs off X" is "1-3". Name the album yourself for "their first album".
  top: an artist's most popular songs; artist, value = how many.
  latest: an artist's newest songs; artist, value = how many. Use it for "new" or "latest" songs, which you may not know.
  Songs go to the end of the queue unless they want them sooner: when = "next" (right after the current song: "play X next", "after this one") or "now" (start right away instead of the current song: "play X now", "skip this and play X instead"; a "now" song needs no separate skip). Otherwise when = "".
  skip (value = how many, "" for one), pause, resume, restart (this song from the start), shuffle (the queue),
  stop (ends the music and empties the queue; for a short break use pause), clear (empties the queue),
  volume (value = "up", "down" or a number 0-200), loop (value = "song" for just the current song, "queue" for more than one song such as "loop them all" or "loop the rest", or "off").
  Never list an action they tell you not to do: "don't pause it" asks for nothing; "don't restart it, skip it" is only skip.
- kind "question": they are asking you anything else or chatting with you, on any topic, music or not: always answer it. Answer like a friendly, knowledgeable chatbot in reply: one to three short spoken sentences, plain words, no lists, links, markdown or emoji. If it needs live information you don't have (news, weather, scores, the time), say so briefly.
- kind "other": clearly not meant for you, not understandable, or nothing for you to do.
Use empty strings for fields that don't apply."""

# MEASURED on 2026-10-12 from the bot's own host, through this module (the
# prompt above; 46 utterances × 2 rounds: albums and track picks — including
# the "first three songs off Vices and Virtues, volume 50, shuffle, skip the
# first, loop the other two" request — "the song from Top Gun", most popular /
# newest songs, several steps in one sentence, "and" inside titles, "don't …",
# questions, chatter): google/gemini-2.5-flash-lite 90/92, median 0.60 s, p90
# 0.76 s, ≈ $0.09 per 1,000 requests. Both misses were the safe kind (nothing
# done for "don't loop it, just turn it up"). One answer in that run came back
# garbled and was recovered by the retry on the fallback. A separate check of
# loop phrasings: 23/24 (the 24th was such a garbled answer). The fallback,
# openai/gpt-4.1-nano, on its own: 39/46, median 0.93 s — weaker, so it only
# stands in when Google is unavailable or an answer is garbled. Album / most
# popular / newest answers resolved through core/music_catalog.py against the
# real Deezer API in 0.3–0.9 s.
# Round 6b, same day, with "when" (next/now) in the schema and the checks
# below (settle_when, the wider "don't" net, the cog folding a "skip" into
# the songs after it; settle_tracks in the second run): 55 utterances × 3
# rounds, twice — 321/330, median 0.62 s, p90 ≈ 0.9 s, ≈ $0.105 per 1,000.
# A reported "skip this song and instead play the seventh and ninth song off …
# Pray for the Wicked" 6/6 → One of the Drunks and King of the Clouds, now.
# The model itself answered that as a skip plus the album with no "when" in
# all 4 raw answers looked at: the cog's folding makes it "now". The misses:
# "don't loop it, just turn it up" → nothing (every round, safe); once a
# track pick came back as "other" (nothing done); once "play fireflies now and
# then turn it up" came back without "now" (queued last). settle_when changed
# the model's next/now in ~5 % of answers. Prompt variants that did worse: a list
# example in the album line ("the second and fifth" is "2,5") stopped an
# occasional "7-9" for "seventh and ninth" but made it chat back to "no dude I
# was talking to jake" 3/3 — hence settle_tracks in code; a wordier "when"
# line made it refuse "what's the capital of Australia" and overuse next/now.
# Not chosen (2026-10-11, previous prompt):
# gemini-3.1-flash-lite (62/64 but ~0.9 s, two 8 s timeouts, ~3.5× the price);
# claude-haiku-5.5, gpt-5.4-nano, gpt-5.6-luna (no OpenRouter endpoint took
# this request's parameters); mistral-small-2603 (rate-limited upstream);
# qwen3.8-flash (empty answers).

#: A safety net under the prompt for sentences with a "don't" in them. Seen
#: when measuring (2026-10-11): without the prompt's "never list an action
#: they tell you not to do", the primary model turned "don't skip this one"
#: into a skip; with it, "no wait, don't stop" still sometimes came back as a
#: skip. So with a refusal in the sentence, the model's commands are only
#: taken if each one is an action they actually named — and not the refused one.
_REFUSALS = frozenset({"dont", "never", "shouldnt"})
#: Words between a refusal and what it refuses: "don't you dare skip", "never
#: ever stop", "don't even think about skipping", "don't go skipping". A bare
#: "don't you …" is left alone: "why don't you play some Queen" is a request.
_REFUSAL_LEAD = frozenset({"dare", "even", "ever", "think", "about", "go"})
_PLAY_WORDS = {"play", "playing", "queue", "add", "put", "hear", "listen"}
_ACTION_WORDS = {
    "skip": {"skip", "skipping", "next"},
    "stop": {"stop", "stopping", "end", "kill"},
    "clear": {"clear", "clearing", "empty", "wipe", "delete", "remove"},
    "pause": {"pause", "pausing"},
    "resume": {"resume", "unpause", "continue"},
    "restart": {"restart", "replay", "start", "rewind"},
    "play": _PLAY_WORDS, "album": _PLAY_WORDS, "top": _PLAY_WORDS, "latest": _PLAY_WORDS,
    "shuffle": {"shuffle", "shuffling", "mix", "randomize", "randomise"},
    "volume": {"turn", "louder", "quieter", "softer", "raise", "lower", "change", "touch",
               "crank", "volume", "mute"},
    "loop": {"loop", "looping", "repeat", "repeating"},
    # plan operations (round 8)
    "move": {"move", "moving", "put", "bring", "bump", "push", "front", "top", "next"},
    "swap": {"swap", "swapping", "switch", "exchange", "trade"},
    "remove": {"remove", "removing", "delete", "drop", "take", "get", "skip", "cut"},
    "reverse": {"reverse", "reversing", "flip", "backwards"},
    "balance": {"balance", "alternate", "fair", "even", "rebalance"},
    "dedupe": {"dedupe", "duplicates", "duplicate", "repeats", "doubles"},
    "seek": {"seek", "jump", "skip", "go", "forward", "back", "rewind", "fast"},
    "back": {"back", "previous", "last", "again", "replay"},
    "skipto": {"skip", "jump", "go", "play"},
}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["kind", "actions", "reply"],
    "properties": {
        "kind": {"type": "string", "enum": ["commands", "question", "other"]},
        "actions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["action", "query", "artist", "heard", "value", "when"],
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "query": {"type": "string"},
                "artist": {"type": "string"},
                "heard": {"type": "string"},
                "value": {"type": "string"},
                "when": {"type": "string", "enum": ["", "next", "now"]}}}},
        "reply": {"type": "string"}}}


@dataclass(frozen=True)
class LLMReply:
    kind: str                          # commands | question | other
    intents: tuple[Intent, ...] = ()
    reply: str = ""
    seconds: float = 0.0
    cost: float = 0.0
    model: str = ""


def build_request(said: str, *, now_playing: str | None, model: str,
                  fallbacks: tuple[str, ...] = ()) -> dict:
    """The chat-completions body. The system prompt never changes, so
    providers can cache it; what varies goes in the user message."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Now playing: {now_playing or 'nothing'}\n"
                                        f"They said: {said}"}],
        "temperature": 0,
        "max_tokens": 768,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "voice_dj", "strict": True, "schema": SCHEMA}},
        # Only endpoints that honour the schema, fastest first.
        "provider": {"require_parameters": True, "sort": "latency"},
        "usage": {"include": True},
    }
    if fallbacks:
        body["models"] = [model, *[m for m in fallbacks if m and m != model]]
    return body


def _volume_value(value: str) -> str | None:
    v = value.strip().lower().rstrip("%").strip()
    if v in ("up", "louder", "higher"):
        return "up"
    if v in ("down", "quieter", "lower", "softer"):
        return "down"
    if v.startswith(("-", "−", "–")):
        return None                                   # no negative volumes
    if v.isdigit():
        return str(int(v))
    n, negative = parse_number(v.replace("-", " ").split())   # "twenty-five"
    if n is not None and not negative:
        return str(n)
    return None


def _clean(text, limit: int = MAX_QUERY_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit]


def _spilled(*texts: str) -> bool:
    """The model's JSON spilling into a string (seen: 'X},{') — not a name."""
    return any("{" in t or "}" in t for t in texts)


_LOOP_VALUES = {"on": "song", "song": "song", "track": "song", "one": "song", "this": "song",
                "queue": "queue", "all": "queue", "playlist": "queue", "everything": "queue",
                "off": "off", "none": "off", "toggle": "toggle"}


def _intent_from(item: dict) -> Intent | None:
    action = str(item.get("action", "")).strip().lower()
    query = _clean(item.get("query"))
    artist = _clean(item.get("artist"), 100)
    value = _clean(item.get("value"), 40)
    heard = _clean(item.get("heard"))
    when = _clean(item.get("when"), 10).lower()
    when = when if when in ("next", "now") else None
    if _spilled(query, artist, value, heard):
        return None
    if action == "play":
        query = query or heard or value
        if not query:
            return None
        return Intent("play", query, 0.8, artist=artist or None, when=when,
                      heard=heard if heard and heard != query else None)
    if action == "find":
        query = query or heard or value
        if not query:
            return None
        return Intent("find", query, 0.8, artist=artist or None, when=when)
    if action == "album":
        if not query:
            return None
        return Intent("album", query, 0.8, artist=artist or None, count=value or "all", when=when)
    if action in ("top", "latest"):
        who = artist or query
        if not who:
            return None
        return Intent(action, None, 0.8, artist=who, when=when,
                      count=value or ("5" if action == "top" else "1"))
    if action == "volume":
        level = _volume_value(value or query)
        return Intent("volume", level, 0.8) if level else None
    if action == "loop":
        mode = _LOOP_VALUES.get((value or query).strip().lower(), "song")
        return Intent("loop", mode, 0.8)
    if action == "skip":
        n = value if value.isdigit() and int(value) > 1 else None
        return Intent("skip", None, 0.8, count=str(min(int(n), 10)) if n else None)
    if action in ACTIONS:
        return Intent(action, None, 0.8)
    return None


#: "next"/"now" put songs in front — "now" even skips the current one — so,
#: like the "don't" net, they're only taken when the words ask for it. Seen
#: (2026-10-12, a wordier prompt): "play mr brightside" as "next" and "play
#: fireflies and turn it up" as "now".
_NOW_CUES = re.compile(r"\b(now|instead|replace|replacing|skip|skipping|immediately|switch|"
                       r"change|right away|straight away)\b")
_NEXT_CUES = re.compile(r"\b(next|afterwards|after (this|that|it|the current))\b")


def settle_when(said: str, intents) -> tuple:
    """`intents` with each song's "next"/"now" kept only if `said` asks for
    it: "now" needs a word like now / instead / replace / skip (else it falls
    back to "next" if they said next, or to the end of the queue); "next"
    needs next / after this — or a "now" word."""
    text = said.lower()
    now_cue, next_cue = bool(_NOW_CUES.search(text)), bool(_NEXT_CUES.search(text))
    out = []
    for i in intents:
        when = i.when
        if when == "now" and not now_cue:
            when = "next" if next_cue else None
        elif when == "next" and not (next_cue or now_cue):
            when = None
        out.append(i if when == i.when else replace(i, when=when))
    return tuple(out)


def settle_tracks(said: str, intents) -> tuple:
    """`intents` with an album's track range "A-B" read back as "A,B" when
    they said "A and B" (no "to"/"through" between them): the model once
    wrote "the seventh and ninth songs" as "7-9" (2026-10-12)."""
    text = said.lower().replace("&", " and ")
    out = []
    for i in intents:
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", i.count or "") if i.action == "album" else None
        if m and int(m.group(2)) - int(m.group(1)) >= 2:
            a, b = int(m.group(1)), int(m.group(2))
            first, last = ("|".join(map(re.escape, music_catalog.said_as(n))) for n in (a, b))
            listed = re.search(rf"\b(?:{first})\b(?:\s+(?:song|track|one)s?)?\s*(?:,\s*and\b|,|\band\b)"
                               rf"\s*(?:the\s+)?(?:{last})\b", text)
            ranged = re.search(rf"\b(?:{first})\b(?:\s+(?:song|track|one)s?)?\s*"
                               rf"(?:-|\b(?:to|through|thru|till|until)\b)", text)
            if listed and not ranged:
                i = replace(i, count=f"{a},{b}")
        out.append(i)
    return tuple(out)


def contradicts(said: str, intents) -> bool:
    """True if `said` refuses something ("don't …", "do not …", "never …")
    and one of `intents` isn't named anywhere *without* a refusal right
    before it: the refused action itself ("don't skip this one", "don't you
    dare skip" → skip), or
    one they never said ("no wait, don't stop" → skip). Kept: "don't stop,
    just skip it" → skip; "don't play that, play the album" → album. A
    sentence without a refusal is left to the model ("stop the music and
    clear the queue")."""
    text = said.lower().replace("’", "'").replace("'", "")
    words = re.sub(r"[^a-z0-9\s]", " ", text).split()
    refused_at: set[int] = set()
    for i, w in enumerate(words[:-1]):
        if w in _REFUSALS:
            j = i + 1
        elif w == "do" and words[i + 1] == "not" and i + 2 < len(words):
            j = i + 2
        else:
            continue
        if (words[j] in ("you", "u", "ya") and j + 1 < len(words) - 1
                and words[j + 1] in ("dare", "even", "ever")):
            j += 1                                   # "don't you dare …", not "don't you play …"
        while words[j] in _REFUSAL_LEAD and j < len(words) - 1:
            j += 1
        refused_at.add(j)
    if not refused_at:
        return False
    free = {w for i, w in enumerate(words) if i not in refused_at}
    return any(not _ACTION_WORDS.get(intent.action, set()) & free for intent in intents)


#: Plan operations as the "don't" check (contradicts) knows them.
_OP_ACTIONS = {"add_song": "play", "find_song": "play", "add_album": "play", "add_top": "play",
               "add_latest": "play", "search": "play", "remove": "remove", "clear": "clear",
               "skip": "skip", "skipto": "skipto", "stop": "stop", "pause": "pause",
               "resume": "resume", "restart": "restart", "shuffle": "shuffle",
               "volume": "volume", "loop": "loop", "move": "move", "swap": "swap",
               "reverse": "reverse", "balance": "balance", "dedupe": "dedupe",
               "seek": "seek", "back": "back"}


def _plan_intents(steps) -> list[Intent]:
    """Plan steps as the Intents the "don't" net (contradicts) checks: the
    operations it has action words for. (Questions, lyrics, likes… can't be
    refused by name there, so they aren't checked.)"""
    return [Intent(_OP_ACTIONS[s.op]) for s in steps if s.op in _OP_ACTIONS]


def parse_reply(content: str) -> LLMReply | None:
    """The model's JSON → a validated :class:`LLMReply`, or None if unusable.

    Unknown actions are dropped, songs are capped, a "commands" answer with
    no usable action — or a "question" with no answer — becomes "other"."""
    if not isinstance(content, str) or not content:
        return None
    m = re.search(r"\{.*\}", content, re.S)          # tolerate ```json fences
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    kind = str(data.get("kind", "")).strip().lower()
    if kind == "commands":
        intents: list[Intent] = []
        songs = 0
        actions = data.get("actions") or []
        if not isinstance(actions, list):
            return None
        for item in actions:
            if not isinstance(item, dict) or len(intents) >= MAX_ACTIONS:
                continue
            intent = _intent_from(item)
            if intent is None:
                continue
            if intent.action in SONG_ACTIONS:
                if songs >= MAX_SONG_ACTIONS:
                    continue
                songs += 1
            intents.append(intent)
        return LLMReply("commands", tuple(intents)) if intents else LLMReply("other")
    if kind == "question":
        reply = " ".join(str(data.get("reply") or "").split())
        if len(reply) > MAX_REPLY_CHARS:
            cut = reply[:MAX_REPLY_CHARS]
            reply = cut[:cut.rfind(" ")].rstrip(",;:") + "…" if " " in cut else cut
        return LLMReply("question", reply=reply) if reply else LLMReply("other")
    return LLMReply("other")


# ── "which of these songs?" and "look it up" ────────────────────────────────
#
# A title the LLM doesn't know (newer than it, or misheard) is matched against
# the artist's real songs by core/music_catalog.py. When that isn't sure, or
# the request only *described* a song ("that Yung Gravy song that samples
# Panic! At The Disco"), the LLM is asked once more — with the real songs to
# choose from, or with web search results.

CHOICE_PROMPT = """You help a Discord music bot that is controlled by voice. Below are the words someone said (speech-to-text: expect misheard words), what the bot understood they want, and a numbered list of real songs.
Which song did they mean? Words that sound alike count ("Him's Not Travis D's" can be "Hymns Not Travesties"), and so does what they described (a song that samples or parodies another one, a theme, a feature).
Answer with JSON: choice = the song's number, or 0 if none of them fits or you can't tell."""

CHOICE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["choice"],
                 "properties": {"choice": {"type": "integer"}}}

LOOKUP_PROMPT = """You help a Discord music bot that is controlled by voice. Someone asked it for a song they described rather than named. Use the web search results to work out which song it is.
Answer with JSON: title = the song's exact title, artist = its main artist, found = false if you can't tell which song it is."""

LOOKUP_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["found", "title", "artist"],
                 "properties": {"found": {"type": "boolean"}, "title": {"type": "string"},
                                "artist": {"type": "string"}}}

MAX_CHOICES = 200          # songs listed for one "which of these?" question
WEB_ENGINE = "exa"         # $0.007 a search (2026-10): far below Google's own grounding
WEB_RESULTS = 5


def _song_line(n: int, song) -> str:
    version = f" {song.version}" if getattr(song, "version", "") else ""
    return f"{n}. {_clean(song.title, 120)}{_clean(version, 60)} - {_clean(song.artist, 80)}"


def build_choice_request(said: str, wanted: str, songs, *, model: str,
                         fallbacks: tuple[str, ...] = ()) -> dict:
    """ "Which of these real songs did they mean?" — `songs` are catalogue
    tracks (title, artist, version), numbered from 1."""
    listing = "\n".join(_song_line(n, t) for n, t in enumerate(list(songs)[:MAX_CHOICES], 1))
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": CHOICE_PROMPT},
            {"role": "user", "content": f"They said: {_clean(said, 500)}\n"
                                        f"Understood: {_clean(wanted, 300)}\nSongs:\n{listing}"}],
        "temperature": 0,
        "max_tokens": 32,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "song_choice", "strict": True,
                                            "schema": CHOICE_SCHEMA}},
        "provider": {"require_parameters": True, "sort": "latency"},
        "usage": {"include": True},
    }
    if fallbacks:
        body["models"] = [model, *[m for m in fallbacks if m and m != model]]
    return body


def parse_choice(content: str, n: int) -> int | None:
    """1..n for a song, 0 for "none of them", None if the answer is unusable."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        choice = data.get("choice") if isinstance(data, dict) else None
        if isinstance(choice, bool):
            return None
        choice = int(choice)
    except (ValueError, TypeError):
        return None
    return choice if 0 <= choice <= n else None


def build_lookup_request(description: str, artist: str, *, model: str,
                         engine: str = WEB_ENGINE, results: int = WEB_RESULTS) -> dict:
    """A web-search-backed "which song is this?" (OpenRouter's web plugin).
    The user message is what gets searched, so it is the description alone."""
    query = _clean(description, 200)
    if artist and norm_words(artist) not in norm_words(query):
        query = f"{query} {_clean(artist, 80)}"
    return {
        "model": model,
        "messages": [{"role": "system", "content": LOOKUP_PROMPT},
                     {"role": "user", "content": f"song: {query}"}],
        "plugins": [{"id": "web", "engine": engine, "max_results": results}],
        "temperature": 0,
        "max_tokens": 120,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "song_lookup", "strict": True,
                                            "schema": LOOKUP_SCHEMA}},
        "usage": {"include": True},
    }


def norm_words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def parse_lookup(content: str) -> tuple[str, str] | None:
    """(title, artist) or None — also when the model says it didn't find it."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("found") is not True:
        return None
    title, artist = _clean(data.get("title"), 150), _clean(data.get("artist"), 100)
    if not title or _spilled(title, artist):
        return None
    return title, artist


# ── the planner: a whole request as explicit steps ───────────────────────────
#
# Round 8 (2026-10-14). The answer above ("commands") could only name a few
# fixed actions, had no idea what was queued, and was cut at 8 actions / 5
# songs / 500 characters / 768 tokens. A plan instead uses every operation in
# the capability registry (core/voice_tools.py), sees the player and the
# queue (with entry ids it can point at), and the speaker's recent requests,
# and says which steps wait for which (core/voice_plan.py). Nothing is cut
# silently: an over-long request, an answer that ran out of room, or steps
# beyond the limits are reported.

#: Longest request sent to the planner (a 40-second sentence is ~600).
MAX_SAID_CHARS = 2000
PLAN_MAX_TOKENS = 4096
#: Lookup rounds (the model asking for an album's tracks first) per request.
MAX_LOOKUP_ROUNDS = 2


def _plan_schema() -> dict:
    from .voice_tools import OPS
    step = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "op", "title", "artist", "heard_title", "heard_artist", "album",
                     "tracks", "target", "to", "value", "count", "when", "after"],
        "properties": {
            "id": {"type": "integer"},
            "op": {"type": "string", "enum": sorted(OPS)},
            **{k: {"type": "string"} for k in ("title", "artist", "heard_title", "heard_artist",
                                               "album", "tracks", "target", "to", "value")},
            "count": {"type": "integer"},
            "when": {"type": "string", "enum": ["", "next", "now"]},
            "after": {"type": "array", "items": {"type": "integer"}}}}
    lookup = {"type": "object", "additionalProperties": False,
              "required": ["tool", "title", "artist"],
              "properties": {"tool": {"type": "string", "enum": ["album", "songs"]},
                             "title": {"type": "string"}, "artist": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["kind", "steps", "reply", "ask", "lookups"],
            "properties": {"kind": {"type": "string", "enum": ["plan", "answer", "lookup", "other"]},
                           "steps": {"type": "array", "items": step},
                           "reply": {"type": "string"}, "ask": {"type": "string"},
                           "lookups": {"type": "array", "items": lookup}}}


PLAN_PROMPT_HEAD = """You plan what the voice assistant of a Discord music bot does. Someone in the voice channel said the wake word and then the words below. They come from speech-to-text, so expect misheard words ("cue" usually means "queue", "tip this" is "skip this", "move the last song" with nowhere to move it to is "remove the last song"). You also get what is playing, the queue, and what this person asked recently.

Answer with JSON:
- kind "plan": steps = every operation they asked for.
- kind "answer": reply = a short spoken answer, when they only asked a question or chatted (one to three short sentences, plain words, no lists or emoji). If it needs live information you don't have (news, weather, the time), say so briefly.
- kind "lookup": lookups = facts you need before you can plan (rare): {"tool": "album", "title", "artist"} for an album's track list, {"tool": "songs", "artist"} for an artist's songs. You then get the facts and answer again.
- kind "other": not meant for you, not understandable, or nothing to do.
Always write all five fields, in this order, even when they're empty — e.g. nothing to do is {"kind": "other", "steps": [], "reply": "", "ask": "", "lookups": []}.

A step is: id (1, 2, 3…), op, the fields that op uses, and "" / 1 / [] for the rest.
Rules:
- One step per thing they asked for, in the order they said them. Nothing they didn't ask for, and never something they tell you not to do ("don't skip it" asks for nothing).
- Steps run in the order you list them: list them in the order they must happen, which is the order said — except that "do A, but only after B" (even when B is said last) means B happens first: list B's steps before A, and give A after = [B's ids]. Otherwise after is [] (never just the step before).
- Repeats are one step with count: "queue it five separate times" is count 5.
- Queue positions count upcoming songs: pos:1 is the next one, and the song playing now is "current" — it is not in the queue. "Skip (or remove) the second song in the queue" is remove pos:2; "skip this song" is skip.
- Point at songs already queued (or playing) by their entry id (entry:q12) when you can tell which one they mean, even if they say the name imperfectly, and at "the one I just added" with the id from their recent requests. Only use ids shown above — never make one up. Songs this same plan adds have no id yet: point at them with step:K (the step that adds them).
- Songs: give the song's real title and artist (fix mishearings of names you know), and heard_title / heard_artist exactly as the speech-to-text wrote them. A title or artist you don't know may be real — newer than you, or misheard: still include the song, as heard (the bot checks it against a music catalogue). Never put a different song in its place and never swap the artist they said for a more famous one. Only when part of a song is missing — "by" with no artist after it — or it makes no sense at all, leave that song out and ask about it in "ask".
- "Play some songs that …", "five random songs from 2019", "a random song": you choose the songs — real, well-known songs that fit — and list each as its own add_song step with its title and artist (five songs = five steps). find_song is only for one particular song they describe but don't name.
- "What song is this?" is now_playing, "what's next / what's in the queue?" is queue_info.
- If part of a request can't be done or worked out, still plan all the rest, and ask about that part. kind "answer" is only for a request that is nothing but a question or chat.
- reply is "" for a plan (the results are reported for you). ask = one short, focused question about whatever you couldn't work out ("Kaimara by who?"), or "".

Queue references (target, and to for swap): pos:N · first · last · last:N (the last N) · range:A-B · current · entry:qN · step:K (everything step K added) · step:K#J (the J-th copy) · match:TEXT (a queued song by its name, if it has no id) · mine:last (the last one this person added).
Destinations (to, for move): front · end · pos:N · after:REF · before:REF.

Operations:
"""


def _salvage_runaway(content: str):
    """An answer that hit the length limit only because the model wrote
    whitespace forever after a finished value — seen live from Gemini under a
    strict schema: '{"kind": "other"' then 4,000 tokens of spaces (2026-10-14).
    Closed and read, but only for kinds that carry no steps ("other", or an
    "answer" whose reply is complete): a plan cut short could be missing
    steps, so it's never guessed at. None if it isn't that."""
    from . import voice_plan
    from .voice_tools import OPS
    body = (content or "").rstrip()
    if not body or len(content) - len(body) < 200:        # not a runaway: really cut off
        return None
    try:
        data = json.loads(body + "}")
    except ValueError:
        return None
    if not isinstance(data, dict) or str(data.get("kind", "")).lower() not in ("other", "answer"):
        return None
    if data.get("steps"):
        return None
    return voice_plan.parse_plan(json.dumps(data), ops=OPS)


def plan_prompt() -> str:
    from .voice_tools import planner_ops_text
    return PLAN_PROMPT_HEAD + planner_ops_text()


def song_label(e: dict | None) -> str:
    """A queue entry or the current song in plain words: "Fireflies by Owl
    City". (With an em dash here the fallback model was seen derailing into
    broken JSON on 2026-10-11.) YouTube's "- Topic" channel suffix is
    dropped; the uploader is only added when the title doesn't name them."""
    if not e:
        return ""
    title = _clean(e.get("title"), 90).replace("—", "-").replace("–", "-")
    by = e.get("artist") or e.get("uploader") or ""
    by = re.sub(r"\s*-\s*Topic$", "", by.replace("—", "-").replace("–", "-")).strip()
    if by and by.lower() not in title.lower():
        return f"{title} by {_clean(by, 60)}"
    return title


def _entry_line(n, e: dict) -> str:
    who = e.get("requester") or "?"
    return f"{n}. [{e.get('_qid', '?')}] {song_label(e)} (added by {_clean(who, 40)})"


def plan_context(*, current: dict | None, upcoming: list[dict], player: str = "",
                 recent: list[str] = (), question: str = "", show: int = 30) -> str:
    """What the planner is told besides the words: the player, the queue
    (entry ids, positions; the first `show` entries and how many more), the
    speaker's recent requests and results, and an open question of ours."""
    lines = []
    if current:
        lines.append(f"Now playing: [{current.get('_qid', 'current')}] {song_label(current)}")
    else:
        lines.append("Now playing: nothing")
    if player:
        lines.append(f"Player: {player}")
    if upcoming:
        lines.append(f"Queue ({len(upcoming)} upcoming; pos:1 plays next):")
        lines += [_entry_line(n, e) for n, e in enumerate(upcoming[:show], 1)]
        if len(upcoming) > show:
            lines.append(f"… and {len(upcoming) - show} more (ask by name with match:TEXT)")
    else:
        lines.append("Queue: empty")
    if recent:
        lines.append("Their recent requests (oldest first):")
        lines += [f"- {_clean(r, 400)}" for r in recent]
    if question:
        lines.append(f"You asked them: {_clean(question, 300)} — what they say now answers it.")
    return "\n".join(lines)


def build_plan_request(said: str, context: str, *, model: str, fallbacks: tuple[str, ...] = (),
                       facts: str = "") -> dict:
    """The planner's chat-completions body (the instructions never change, so
    providers can cache them; the situation and the words go last)."""
    user = f"{context}\n"
    if facts:
        user += f"Facts you asked for:\n{facts}\n"
    user += f"They said: {said}"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": plan_prompt()},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": PLAN_MAX_TOKENS,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "voice_plan", "strict": True,
                                            "schema": _plan_schema()}},
        "provider": {"require_parameters": True, "sort": "latency"},
        "usage": {"include": True},
    }
    if fallbacks:
        body["models"] = [model, *[m for m in fallbacks if m and m != model]]
    return body


class LLMRouter:
    """Ask the LLM about one utterance. Never raises: on any failure (no
    network, HTTP error, timeout, rate limit, unusable answer) it returns
    None and the cog falls back to what the grammar made of it."""

    def __init__(self, api_key: str, *, model: str, fallbacks: tuple[str, ...] = (),
                 base_url: str = "https://openrouter.ai/api/v1", timeout: float = 8.0,
                 per_minute: int = 30, web: bool = True, web_per_hour: int = 30,
                 web_timeout: float = 12.0):
        self._key = api_key
        self.model = model
        self.fallbacks = tuple(fallbacks)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.per_minute = per_minute
        #: Web lookups (OpenRouter's web plugin) cost about a cent each: off
        #: switch, and a cap per hour.
        self.web = web
        self.web_per_hour = web_per_hour
        self.web_timeout = web_timeout
        self._web_recent: deque = deque()
        self._session = None
        self._recent: deque = deque()
        self._last_logged: dict[str, float] = {}
        # for `voice ai` / `voice diag`
        self.calls = 0
        self.failures = 0
        self.cost = 0.0
        self.last_error = ""
        self.kinds: dict[str, int] = {}
        self.seconds: deque = deque(maxlen=50)
        self._last_cost = 0.0

    # ── plumbing ────────────────────────────────────────────────────────────

    async def _http(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _fail(self, why: str) -> None:
        self.failures += 1
        self.last_error = redact_secrets(why)[:200]
        now = time.monotonic()
        kind = why.split(":", 1)[0]
        if now - self._last_logged.get(kind, -1e9) > 60:
            self._last_logged[kind] = now
            log.warning("voice LLM: %s", self.last_error)

    def _allowed(self) -> bool:
        now = time.monotonic()
        while self._recent and now - self._recent[0] > 60:
            self._recent.popleft()
        if len(self._recent) >= self.per_minute:
            return False
        self._recent.append(now)
        return True

    def median_seconds(self) -> float | None:
        if not self.seconds:
            return None
        s = sorted(self.seconds)
        return s[len(s) // 2]

    def _web_allowed(self) -> bool:
        now = time.monotonic()
        while self._web_recent and now - self._web_recent[0] > 3600:
            self._web_recent.popleft()
        if len(self._web_recent) >= self.web_per_hour:
            return False
        self._web_recent.append(now)
        return True

    def _count(self, kind: str) -> None:
        self.kinds[kind] = self.kinds.get(kind, 0) + 1

    # ── the call ────────────────────────────────────────────────────────────

    async def ask(self, said: str, *, now_playing: str | None = None) -> LLMReply | None:
        said = " ".join((said or "").split())[:500]
        if not said:
            return None
        if not self._allowed():
            self._fail("rate limited: more than "
                       f"{self.per_minute} questions a minute — skipped")
            return None
        started = time.monotonic()
        # A garbled answer (seen about once in a hundred on 2026-10-12: JSON
        # that breaks off) gets one more try, on the fallback model — another
        # provider — within the same time budget.
        attempts = [(self.model, self.fallbacks)]
        if self.fallbacks:
            attempts.append((self.fallbacks[0], self.fallbacks[1:]))
        for n, (model, fallbacks) in enumerate(attempts):
            budget = self.timeout - (time.monotonic() - started)
            if n and budget < 1.0:
                break                              # no time left for a second try
            outcome = await self._once(said, now_playing, model, fallbacks, budget)
            if outcome == "garbled" and n + 1 < len(attempts):
                self.kinds["retried"] = self.kinds.get("retried", 0) + 1
                continue
            if not isinstance(outcome, tuple):
                return None
            reply, data = outcome
            seconds = time.monotonic() - started
            self.seconds.append(seconds)
            if reply.kind == "commands" and contradicts(said, reply.intents):
                self.kinds["vetoed"] = self.kinds.get("vetoed", 0) + 1
                reply = LLMReply("other")          # it did what they said not to: do nothing
            elif reply.kind == "commands":
                placed = settle_when(said, reply.intents)
                if placed != reply.intents:
                    self.kinds["when_ignored"] = self.kinds.get("when_ignored", 0) + 1
                listed = settle_tracks(said, placed)
                if listed != placed:
                    self.kinds["tracks_listed"] = self.kinds.get("tracks_listed", 0) + 1
                if listed != reply.intents:
                    reply = LLMReply("commands", listed, reply.reply)
            self.kinds[reply.kind] = self.kinds.get(reply.kind, 0) + 1
            return LLMReply(reply.kind, reply.intents, reply.reply, seconds,
                            self._last_cost, str(data.get("model") or model))
        return None

    async def plan(self, said: str, context: str, *, lookup=None, model: str | None = None,
                   fallbacks: tuple[str, ...] | None = None, timeout: float | None = None):
        """A Plan (core/voice_plan.py) for one request, or None if no usable
        answer came back. `lookup(tool, title, artist)` (async → text) answers
        the model's "lookup" requests. A request longer than MAX_SAID_CHARS
        isn't cut: it comes back as a plan with that problem. Never raises."""
        from . import voice_plan
        from .voice_tools import OPS
        said = " ".join((said or "").split())
        if not said:
            return None
        if len(said) > MAX_SAID_CHARS:
            return voice_plan.Plan("plan", problems=(
                f"that request is {len(said)} characters long — I can take {MAX_SAID_CHARS} "
                f"at once, so I didn't do any of it; say it in parts",))
        if not self._allowed():
            self._fail(f"rate limited: more than {self.per_minute} requests a minute — skipped")
            return None
        model = model or self.model
        fallbacks = self.fallbacks if fallbacks is None else tuple(fallbacks)
        budget_total = timeout or max(self.timeout, 20.0)
        started = time.monotonic()
        facts = ""
        cost = 0.0
        rounds = 0
        attempts = [(model, fallbacks)]
        if fallbacks:
            attempts.append((fallbacks[0], fallbacks[1:]))
        n = 0
        while n < len(attempts):
            m, fb = attempts[n]
            budget = budget_total - (time.monotonic() - started)
            if budget < 1.0:
                self._fail("timeout: no plan within the time allowed")
                return None
            body = build_plan_request(said, context, model=m, fallbacks=fb, facts=facts)
            data = await self._post(body, budget, "plan")
            cost += self._last_cost
            if not isinstance(data, dict):
                return None
            content = self._content(data)
            self.last_raw = content[:6000]
            finish = ""
            try:
                finish = data["choices"][0].get("finish_reason") or ""
            except (KeyError, IndexError, TypeError, AttributeError):
                pass
            plan = voice_plan.parse_plan(content, ops=OPS) if finish != "length" else None
            if plan is None and finish == "length":
                plan = _salvage_runaway(content)
                if plan is not None:
                    self._count("runaway_salvaged")
            if plan is None:
                why = ("the answer ran out of room" if finish == "length"
                       else f"unusable plan: {content[:200]}")
                self._fail(why)
                self.kinds["retried"] = self.kinds.get("retried", 0) + 1
                n += 1
                if n >= len(attempts) and finish == "length":
                    return voice_plan.Plan("plan", problems=(
                        "the plan for that came back cut off, so I didn't do any of it — "
                        "try it in smaller parts",), cost=cost)
                continue
            if plan.kind == "lookup":
                rounds += 1
                if rounds > MAX_LOOKUP_ROUNDS + 1:
                    # told "no more lookups" and asked again: give up rather
                    # than pay for more rounds until the time runs out
                    self._fail("unusable plan: kept asking for lookups")
                    return None
                if lookup is None or rounds > MAX_LOOKUP_ROUNDS:
                    facts += "\n(no more lookups: plan with what you have)"
                else:
                    got = []
                    for item in plan.lookups:
                        try:
                            got.append(await lookup(str(item.get("tool") or ""),
                                                    _clean(item.get("title")),
                                                    _clean(item.get("artist"))))
                        except Exception as e:                 # never fatal
                            got.append(f"(lookup failed: {type(e).__name__})")
                    facts += "\n".join(got) + "\n"
                    self._count("lookups")
                continue                                   # same model, with the facts
            seconds = time.monotonic() - started
            self.seconds.append(seconds)
            self._count(f"plan_{plan.kind}")
            if plan.kind == "plan" and plan.steps and contradicts(said, _plan_intents(plan.steps)):
                self._count("vetoed")
                return replace(plan, kind="other", steps=(), seconds=seconds, cost=cost,
                               model=str(data.get("model") or m),
                               problems=plan.problems)
            return replace(plan, seconds=seconds, cost=cost, model=str(data.get("model") or m))
        return None

    async def choose_song(self, said: str, wanted: str, songs) -> int | None:
        """Which of `songs` (catalogue tracks) they meant: an index into
        `songs`, or None for none of them / no usable answer. Never raises."""
        songs = list(songs)[:MAX_CHOICES]
        if not songs or not self._allowed():
            return None
        started = time.monotonic()
        body = build_choice_request(said, wanted, songs, model=self.model,
                                    fallbacks=self.fallbacks)
        data = await self._post(body, self.timeout, "choice")
        if not isinstance(data, dict):
            return None
        choice = parse_choice(self._content(data), len(songs))
        self.seconds.append(time.monotonic() - started)
        if choice is None:
            self._fail(f"unusable song choice: {self._content(data)[:200]}")
            return None
        self._count("song_picked" if choice else "song_not_listed")
        return choice - 1 if choice else None

    async def lookup_song(self, description: str, artist: str = "") -> tuple[str, str] | None:
        """(title, artist) of a song described rather than named, from a web
        search. None if it can't tell, web lookups are off or used up for
        this hour, or anything fails. Never raises."""
        if not self.web or not description.strip():
            return None
        if not self._web_allowed():
            self._fail(f"web lookups: more than {self.web_per_hour} this hour — skipped")
            return None
        body = build_lookup_request(description, artist, model=self.model)
        data = await self._post(body, self.web_timeout, "web")
        if not isinstance(data, dict):
            return None
        found = parse_lookup(self._content(data))
        self._count("web_found" if found else "web_not_found")
        return found

    @staticmethod
    def _content(data: dict) -> str:
        try:
            return data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    async def _post(self, body: dict, budget: float, what: str = "ask"):
        """POST one request. The decoded answer (a dict) on success, else
        "failed" with the reason recorded. Counts calls and cost."""
        headers = {"Authorization": f"Bearer {self._key}",
                   "Content-Type": "application/json",
                   "X-Title": "Discord music bot voice DJ"}
        self.calls += 1
        self._last_cost = 0.0
        try:
            import aiohttp
            session = await self._http()
            async with session.post(self.url, json=body, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=budget)) as r:
                text = await r.text()
                if r.status != 200:
                    self._fail(f"HTTP {r.status}: {text[:200]}")
                    return "failed"
            data = json.loads(text)
        except asyncio.TimeoutError:
            self._fail(f"timeout: no {what} answer within {budget:g} s")
            return "failed"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
            return "failed"
        usage = data.get("usage") if isinstance(data, dict) else None
        try:
            cost = float(usage.get("cost") or 0.0) if isinstance(usage, dict) else 0.0
        except (TypeError, ValueError):
            cost = 0.0
        if not math.isfinite(cost) or cost < 0:
            cost = 0.0
        self.cost += cost                          # a garbled answer still costs
        self._last_cost = cost
        if not isinstance(data, dict):
            self._fail(f"bad response: {str(data)[:200]}")
            return "failed"
        return data

    async def _once(self, said: str, now_playing: str | None, model: str,
                    fallbacks: tuple[str, ...], budget: float):
        """One request. (reply, data) on success; otherwise "garbled" (the
        answer couldn't be used) or "failed" (network/HTTP/timeout), with the
        reason recorded."""
        body = build_request(said, now_playing=now_playing, model=model, fallbacks=fallbacks)
        data = await self._post(body, budget)
        if not isinstance(data, dict):
            return "failed"
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            self._fail(f"bad response: {str(data)[:200]}")
            return "garbled"
        reply = parse_reply(content)
        if reply is None:
            self._fail(f"unusable answer: {content[:200]}")
            return "garbled"
        return reply, data
