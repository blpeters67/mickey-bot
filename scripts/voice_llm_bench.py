"""End-to-end check of the voice DJ's smart tier on the real services:
utterance → the bot's own LLM router (core/voice_llm.py) on OpenRouter →
for album/top/latest answers, the bot's own catalogue lookup
(core/music_catalog.py) on Deezer/iTunes. Key from the bot's .env; never printed.

    DISCORD_TOKEN=unused python scripts/voice_llm_bench.py <bot folder with .env> [rounds]

(run from the bot folder; costs a few cents per round: ~70 requests + a few
web lookups)

Round 6b: songs wanted "next"/"now" (when), and a plain "skip" next to them
folded into their "now" the way the cog does it (VoiceDJ._merge_skips).
Round 7: titles the model doesn't know (kept as said + "heard"), songs described
rather than named ("find"), misheard commands ("tip this") — and, for those,
the whole way to a YouTube upload: catalogue match, the AI's pick, web lookup.
"""
import asyncio
import os
import statistics
import sys
import time

ENV = os.path.join(sys.argv[1], ".env")
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
KEY = next(line.split("=", 1)[1].strip() for line in open(ENV, encoding="utf-8")
           if line.startswith("OPENROUTER_API_KEY="))

from core import music_catalog as mc      # noqa: E402
from core import voice_llm as vl          # noqa: E402
from core.config import settings          # noqa: E402
from core.voice_intent import text_after_wake  # noqa: E402

try:
    from cogs.voicedj import VoiceDJ      # noqa: E402

    def merged(intents):
        return [s for s, _ in VoiceDJ._merge_skips(tuple(intents))]
    MERGE = "cogs.voicedj"
except Exception as e:                    # pragma: no cover
    import dataclasses
    MERGE = f"local copy ({type(e).__name__})"

    def merged(intents):
        steps, took, drop = list(intents), set(), set()
        for i, s in enumerate(steps):
            if s.action != "skip" or s.count:
                continue
            for j in (i + 1, i - 1):
                if (0 <= j < len(steps) and j not in took and steps[j].action in
                        ("play", "album", "top", "latest") and steps[j].when in ("next", "now")):
                    steps[j] = dataclasses.replace(steps[j], when="now")
                    took.add(j)
                    drop.add(i)
                    break
        return [s for i, s in enumerate(steps) if i not in drop]

ELEVEN = [mc.CatalogTrack(f"T{n}", "A", 1, n) for n in range(1, 12)]


def picks(count):
    return [t.position for t in mc.select_tracks(ELEVEN, count or "all")]


SONGS = ("play", "album", "top", "latest", "find")


def now_song(action, words, *, tracks=None):
    """After folding a "skip" into it: exactly one song step, wanted now."""
    def check(intents):
        got = merged(intents)
        if len(got) != 1:
            return False
        i = got[0]
        text = f"{i.arg or ''} | {i.artist or ''}".lower()
        return (i.action == action and i.when == "now"
                and all(any(alt in text for alt in w.split("|")) for w in words)
                and (tracks is None or picks(i.count) == tracks))
    check.checks_when = True
    return check

NP = os.environ.get("NP", "Fireflies by Owl City")
NOT_CMD = "other|question"


def has_artist(intents):
    return bool(intents) and all(i.action == "play" and i.artist for i in intents)


def owl_city_songs(intents):
    return bool(intents) and all(
        (i.action == "top" and "owl city" in (i.artist or "").lower()) or
        (i.action == "play" and "owl city" in (i.artist or "").lower()) for i in intents)


ISSUE = ("play the first three songs off Panic! At The Disco's Vices and Virtues album, turn the "
         "volume down to 50 percent, shuffle the queue, skip the first song, then set the other "
         "two songs to loop")
def now_song_then_volume(intents):
    got = merged(intents)
    return ([i.action for i in got] == ["play", "volume"] and got[0].when == "now"
            and "fireflies" in (got[0].arg or "").lower())


now_song_then_volume.checks_when = True
REPORTED = ("Hey DJ, can you skip this song and instead can you play the seventh and ninth "
            "song off of Panic at the Discos Pray for the Wicked album?")
R7 = [  # real requests, 2026-10-12 (as the speech-to-text wrote them)
    "Play I Write Him's Not Travis D's by Young Gravy and Skip the Song.",
    "Play I Write Hymns, Not Travis Dees by Young Gravy and skip this song.",
    "Can you skip this song and instead can you play I Write Him's Not Travis Dees by Young Gravy?",
    "Can you play that one young gravy song that samples panic at the disco and skip this song?",
    "I played cordial by sighted off.",
]


def gravy_title(intents):
    got = merged(intents)
    return (len(got) == 1 and got[0].action == "play" and got[0].when == "now"
            and "gravy" in (got[0].artist or "").lower()
            and "write" in f"{got[0].arg} {got[0].heard or ''}".lower())


gravy_title.checks_when = True


def gravy_described(intents):
    got = merged(intents)
    if len(got) != 1 or got[0].when != "now":
        return False
    i = got[0]
    if i.action == "find":
        return "gravy" in f"{i.arg} {i.artist}".lower()
    return (i.action == "play" and "gravy" in (i.artist or "").lower()
            and "hymns" in (i.arg or "").lower())     # it knew the song: fine


gravy_described.checks_when = True


def cordial(intents):
    return (len(intents) == 1 and intents[0].action == "play"
            and "set it off" in (intents[0].artist or "").lower()
            and "cordial" in f"{intents[0].arg} {intents[0].heard or ''}".lower())


def found_or_played(play_words, find_words):
    def check(intents):
        if len(intents) != 1 or intents[0].when:
            return False
        i = intents[0]
        text = f"{i.arg} | {i.artist}".lower()
        words = play_words if i.action == "play" else find_words if i.action == "find" else None
        return words is not None and all(any(a in text for a in w.split("|")) for w in words)
    return check


R7_CASES = [
    (R7[0], "commands", gravy_title),
    (R7[1], "commands", gravy_title),
    (R7[2], "commands", gravy_title),
    (R7[3], "commands", gravy_described),
    (R7[4], "commands", cordial),
    ("tip this", "commands", [("skip", [])]),
    ("play the song from top gun", "commands",
     found_or_played(["danger zone|take my breath away", "kenny loggins|berlin"], ["top gun"])),
    ("play that song that goes is this the real life", "commands",
     found_or_played(["bohemian rhapsody", "queen"], ["real life"])),
]
CASES = [  # (said, kind(s), expected [(action, [words][, when])] | callable | reply words)
    (text_after_wake(REPORTED, ["hey dj"]), "commands",
     now_song("album", ["pray for the wicked", "panic"], tracks=[7, 9])),
    ("play fireflies now and then turn it up", "commands", now_song_then_volume),
    ("play mr brightside next", "commands", [("play", ["brightside", "killers"], "next")]),
    ("after this one play the pretender", "commands", [("play", ["pretender"], "next")]),
    ("play mr brightside now", "commands", now_song("play", ["brightside"])),
    ("skip this and play bohemian rhapsody instead", "commands", now_song("play", ["bohemian"])),
    ("replace this song with hotel california", "commands", now_song("play", ["hotel california"])),
    ("add fireflies to the queue", "commands", [("play", ["fireflies"], "")]),
    ("play the seventh and ninth songs off pray for the wicked", "commands",
     lambda its: len(its) == 1 and its[0].action == "album" and picks(its[0].count) == [7, 9]
     and not its[0].when),
    (ISSUE, "commands", [("album", ["vices", "panic", "1-3|first 3|1,2,3"], ""), ("volume", ["50"]),
                         ("shuffle", []), ("skip", []), ("loop", ["queue"])]),
    ("play Panic! At The Disco's first three songs off their Vices and Virtues album", "commands",
     [("album", ["vices", "panic", "1-3|first 3|1,2,3"])]),
    ("play something chill", "commands", has_artist),
    ("play mr brightside", "commands", [("play", ["brightside", "killers"])]),
    ("play owl city's most popular songs", "commands", [("top", ["owl city"])]),
    ("play the newest taylor swift song", "commands", [("latest", ["taylor swift"])]),
    ("play the whole album that fireflies is on", "commands",
     [("album", ["ocean eyes", "owl city", "all"])]),
    ("play their first album", "commands", [("album", ["owl city"])]),
    ("play three random songs off a fever you can't sweat out", "commands",
     [("album", ["fever", "random"])]),
    ("play the last song on abbey road", "commands", [("album", ["abbey road", "last|17"])]),
    ("shuffle the queue and skip this song", "commands", [("shuffle", []), ("skip", [])]),
    ("skip the next two songs", "commands", [("skip", ["2"])]),
    ("loop these songs", "commands", [("loop", ["queue"])]),
    ("loop this song", "commands", [("loop", ["song"])]),
    ("I want to hear two songs; can you play Fireflies by Owl City and The Pretender by the Foo Fighters",
     "commands", [("play", ["fireflies", "owl"]), ("play", ["pretender", "foo"])]),
    ("play the sound of silence by simon and garfunkel", "commands",
     [("play", ["silence", "simon"])]),
    ("play me and bobby mcgee", "commands", [("play", ["me and bobby"])]),
    ("play fireflies and the pretender", "commands", [("play", ["fireflies"]), ("play", ["pretender"])]),
    ("put on the pretender by the food fighters", "commands", [("play", ["pretender", "foo fighters"])]),
    ("turn it down a bit and skip this one", "commands", [("volume", ["down"]), ("skip", [])]),
    ("queue up mr brightside then hotel california", "commands",
     [("play", ["brightside"], ""), ("play", ["hotel california"], "")]),
    ("skip this and then play bohemian rhapsody", "commands", now_song("play", ["bohemian"])),
    ("play fireflies and turn it up", "commands", [("play", ["fireflies"]), ("volume", ["up"])]),
    ("set the volume to fifty", "commands", [("volume", ["50"])]),
    ("play some owl city", "commands", owl_city_songs),
    ("start this song over", "commands", [("restart", [])]),
    ("stop the music and clear the queue", "commands", [("stop", []), ("clear", [])]),
    ("hold on, stop it for a sec", "commands", [("pause", [])]),
    ("don't stop the music, just skip this song", "commands", [("skip", [])]),
    ("don't skip this one, I love it", NOT_CMD, []),
    ("no wait, don't stop", NOT_CMD, []),
    ("please don't play that one again", NOT_CMD, []),
    ("wait, don't clear the queue", NOT_CMD, []),
    ("don't you dare skip this song", NOT_CMD, []),
    ("never mind, don't turn it down", NOT_CMD, []),
    ("I don't care, just skip it", "commands", [("skip", [])]),
    ("don't loop it, just turn it up", "commands", [("volume", ["up"])]),
    ("what's the capital of australia", "question", ["canberra"]),
    ("who sings this song", "question", ["owl city"]),
    ("how many legs does a spider have", "question", ["eight|8"]),
    ("tell me a fun fact about octopuses", "question", ["octop"]),
    ("what song is this", "question", ["fireflies"]),
    ("no dude I was talking to jake", "other", []),
    ("louder", "commands", [("volume", ["up"])]),
] + R7_CASES


def good(reply, kinds, want):
    if reply is None or reply.kind not in kinds.split("|"):
        return False
    if reply.kind == "commands":
        if callable(want):
            if getattr(want, "checks_when", False):
                return want(reply.intents)
            return want(reply.intents) and not any(
                i.when for i in reply.intents if i.action in SONGS)
        got = reply.intents
        if len(got) != len(want):
            return False
        for i, (action, words, *when) in zip(got, want):
            text = f"{i.arg or ''} | {i.artist or ''} | {i.count or ''}".lower()
            if i.action != action or not all(any(alt in text for alt in w.split("|")) for w in words):
                return False
            want_when = when[0] if when else ""      # songs go last unless asked
            if action in SONGS and (i.when or "") != want_when:
                return False
        return True
    if reply.kind == "question" and kinds == "question":
        text = reply.reply.lower()
        return all(any(alt in text for alt in w.split("|")) for w in want)
    return True


def show(reply):
    if reply is None:
        return None
    if reply.kind != "commands":
        return (reply.kind, reply.reply[:70])
    return [(i.action, i.arg, i.artist, i.count, i.when) for i in reply.intents]


async def catalogue_for(intents):
    out = []
    for i in intents:
        t = time.monotonic()
        try:
            if i.action == "album":
                album = await mc.find_album(i.arg or "", i.artist or "")
                picked = mc.select_tracks(album.tracks, i.count or "all") if album else []
                out.append(f"album {album.title + ' — ' + album.artist + ' via ' + album.source if album else 'NOT FOUND'}"
                           f" → {[x.query for x in picked]} ({time.monotonic() - t:.2f}s)")
            elif i.action == "top":
                got = await mc.top_tracks(i.artist or "", mc.parse_count(i.count, 5, mc.MAX_TOP))
                out.append(f"top → {[x.query for x in got]} ({time.monotonic() - t:.2f}s)")
            elif i.action == "latest":
                got = await mc.latest_tracks(i.artist or "", mc.parse_count(i.count, 1, 5))
                out.append(f"latest → {[x.query for x in got]} ({time.monotonic() - t:.2f}s)")
        except mc.CatalogError as e:
            out.append(f"{i.action} CATALOG ERROR {e}")
    return out


def np_for(said):
    return "No Disrespect by Set It Off" if "cordial" in said.lower() else NP


class Shim:
    """The voice cog's own song resolution (VoiceDJ._find_song and friends),
    run outside Discord: catalogue match, the AI's pick, web lookup, then the
    YouTube upload check."""

    def __init__(self, router):
        import types
        self.llm, self._cfg = router, {}
        self.counts = {}
        self.metrics = types.SimpleNamespace(
            count=lambda gid, k: self.counts.__setitem__(k, self.counts.get(k, 0) + 1))

    from cogs.voicedj import VoiceDJ as _V
    _find_song = _V._find_song
    _match_song = _V._match_song
    _identify_song = _V._identify_song
    _ai_on = _V._ai_on


async def resolve_all(router, firsts):
    import types
    shim = Shim(router)
    guild = types.SimpleNamespace(id=1)
    print("--- round 7: the whole way to a YouTube upload (round 1 answers)")
    for said, reply in firsts.items():
        if said not in R7 and "top gun" not in said and "real life" not in said:
            continue
        if reply is None or reply.kind != "commands":
            print(f"* {said[:70]!r} → {show(reply)}")
            continue
        for intent in merged(reply.intents):
            if intent.action not in ("play", "find"):
                continue
            t = time.monotonic()
            act = types.SimpleNamespace(text=said)
            entry, known, why = await shim._find_song(guild, intent, act)
            took = time.monotonic() - t
            print(f"* {said[:70]!r}\n    {intent.action} {intent.arg!r} by {intent.artist!r} "
                  f"heard={intent.heard!r} when={intent.when}\n    → song: "
                  f"{(known.artist + ' - ' + known.title) if known else None} · upload: "
                  f"{entry.get('title') if entry else None!r} · {why or ''} ({took:.2f}s)")
    print(f"    resolution counts: {shim.counts} · router kinds: {router.kinds} · "
          f"cost so far ${router.cost:.5f}")


async def main():
    router = vl.LLMRouter(KEY, model=settings.voice_llm_model,
                          fallbacks=settings.voice_llm_fallbacks,
                          base_url=settings.voice_llm_url, timeout=settings.voice_llm_timeout,
                          per_minute=1000)
    try:
        await router.ask("pause", now_playing=NP)
        score, walls, misses, models, firsts = 0, [], [], {}, {}
        for rnd in range(ROUNDS):
            for said, kinds, want in CASES:
                t = time.monotonic()
                reply = await router.ask(said, now_playing=np_for(said))
                walls.append(time.monotonic() - t)
                if reply is not None:
                    models[reply.model] = models.get(reply.model, 0) + 1
                if rnd == 0:
                    firsts[said] = reply
                if good(reply, kinds, want):
                    score += 1
                else:
                    misses.append(f"{said[:50]!r}: {show(reply)}")
        walls.sort()
        n = len(walls)
        print(f"{settings.voice_llm_model} (fallback {', '.join(settings.voice_llm_fallbacks)}): "
              f"{score}/{n}  median {statistics.median(walls):.2f}s  "
              f"p90 {walls[int(n * 0.9) - 1]:.2f}s  worst {walls[-1]:.2f}s")
        print(f"answered by: {models} · skip folding: {MERGE} · kinds: {router.kinds}")
        print(f"cost ${router.cost:.5f} for {router.calls} calls "
              f"(${router.cost / max(1, router.calls) * 1000:.3f} per 1000) · failures {router.failures}"
              + (f" · last: {router.last_error}" if router.failures else ""))
        for m in misses:
            print("  miss:", m)
        print("--- next/now as the model gave it (round 1, before the cog folds a skip in)")
        for said, reply in firsts.items():
            if reply is not None and reply.kind == "commands" and (
                    any(i.when for i in reply.intents) or "instead" in said or " now" in said
                    or " next" in said or "replace" in said or "after this" in said
                    or "skip this and" in said):
                print(f"* {said[:60]!r} → {[(i.action, i.when) for i in reply.intents]}")
        if os.environ.get("RESOLVE", "1") == "1":
            await resolve_all(router, firsts)
        if os.environ.get("CATALOG", "1") == "1":
            print("--- what would be queued (round 1 answers → real catalogue)")
            for said, reply in firsts.items():
                if reply is None or reply.kind != "commands":
                    continue
                if not any(i.action in ("album", "top", "latest") for i in reply.intents):
                    continue
                print(f"* {said[:70]!r}")
                for line in await catalogue_for(reply.intents):
                    print("    " + line)
            print("--- plain play answers (searched on YouTube as)")
            for said, reply in firsts.items():
                if reply is not None and reply.kind == "commands":
                    q = [i.search for i in reply.intents if i.action == "play"]
                    if q:
                        print(f"* {said[:50]!r} → {q}")
    finally:
        await router.close()


asyncio.run(main())
