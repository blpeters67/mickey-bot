"""
Live evaluation of the voice DJ planner (core/voice_llm.py LLMRouter.plan) on
real requests: the ones from the round-8 brief, the reference transcripts of
real microphone recordings, and paraphrases. It only asks the model for
plans and checks them — nothing is played, queued or stored, and no bot is
started. Output: one sanitized line per answer (what the plan does), a
pass/fail per check, latency and cost per model.

    venv/bin/python scripts/voice_plan_bench.py --env .env --rounds 2 \\
        --models google/gemini-2.5-flash-lite,openai/gpt-4.1-mini
    venv/bin/python scripts/voice_plan_bench.py --pid <the bot's pid> …

The OpenRouter key is read from --env (OPENROUTER_API_KEY=…) or the running
bot's environment (--pid) inside this process and never printed. Checks are deliberately about meaning (which
operations, which songs, what waits for what), not exact wording.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DISCORD_TOKEN", "bench")

from core import music_catalog as mc          # noqa: E402
from core import voice_plan as vp             # noqa: E402
from core.voice_llm import LLMRouter, plan_context   # noqa: E402


def _key(path: str | None, pid: int | None = None) -> str:
    """The OpenRouter key from an env file, or from a running process's
    environment (the bot's — where the host keeps it). Never printed."""
    if pid:
        try:
            raw = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        except OSError as e:
            raise SystemExit(f"can't read process {pid}'s environment: {type(e).__name__}")
        for item in raw:
            if item.startswith(b"OPENROUTER_API_KEY="):
                return item.split(b"=", 1)[1].decode().strip()
        raise SystemExit(f"no OPENROUTER_API_KEY in process {pid}'s environment")
    if not path:
        if os.environ.get("OPENROUTER_API_KEY"):
            return os.environ["OPENROUTER_API_KEY"].strip()
        raise SystemExit("give --env FILE or --pid PID (or set OPENROUTER_API_KEY)")
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENROUTER_API_KEY in " + path)


# ── situations ────────────────────────────────────────────────────────────────

def _e(qid, title, who="Alex"):
    return {"_qid": qid, "title": title, "requester": who}


IDLE = dict(current=None, upcoming=[], player="idle, volume 100%, loop off")
PLAYING = dict(current=_e("q1", "Fall Out Boy - Centuries"),
               upcoming=[_e("q2", "Panic! At The Disco - High Hopes"),
                         _e("q3", "Set It Off - Cordial (Official Audio)", "Sam"),
                         _e("q4", "The Killers - Mr. Brightside")],
               player="playing, volume 100%, loop off")
WAX = dict(current=_e("q20", "Demi The Daredevil - AMERICAN ZOMBIE"),
           upcoming=[_e("q21", "Point North - Below The Belt"),
                     _e("q22", "Skrillex - The Devil's Den"),
                     _e("q23", "WAX - ALTE WUNDEN"),
                     _e("q24", "Saint Chaos - Ghosts & Monsters")],
           player="playing, volume 100%, loop off")
RECENT_PHOENIX = dict(current=_e("q1", "Fall Out Boy - Centuries"),
                      upcoming=[_e("q2", "Panic! At The Disco - High Hopes"),
                                _e("q9", "Fall Out Boy - The Phoenix")],
                      player="playing, volume 100%, loop off",
                      recent=["40 s ago: “play the phoenix by fall out boy” → added [q9] "
                              "Fall Out Boy - The Phoenix"])


# ── checks ────────────────────────────────────────────────────────────────────

def ops(p):
    return [s.op for s in p.steps]


def has(p, op, **want):
    """First step with this op whose fields contain the wanted text. For a
    song, title/artist are matched against what was said OR the model's
    reading (the catalogue lookup tries what was said first)."""
    for s in p.steps:
        if s.op != op:
            continue
        def field(k):
            if k == "title":
                return f"{s.heard_title} | {s.title}"
            if k == "artist":
                return f"{s.heard_artist} | {s.artist}"
            return getattr(s, k)
        if all(_fits(field(k), v) for k, v in want.items()):
            return s
    return None


def vol(s, want):
    v = (s.value if s is not None else "").replace("%", "").replace(" ", "").lower()
    return v in want


def _fits(have, want):
    if callable(want):
        return want(have)
    if isinstance(want, int):
        return have == want
    return mc.norm(str(want)) in mc.norm(str(have))


def idx(p, step):
    return [s.id for s in p.steps].index(step.id) if step is not None else -1


def first_half(t):
    return mc.norm(t) in ("first half", "1 6", "1 to 6") or t.replace(" ", "") == "1-6"


def second_half(t):
    return mc.norm(t) in ("second half", "latter half", "7 11", "last half") or \
        t.replace(" ", "") == "7-11"


def album_sel(want):
    """The album selector picks exactly these tracks of an 11-track album."""
    def ok(t):
        try:
            got = [x.position for x in mc.select(
                [mc.CatalogTrack(f"S{n}", "A", 1, n) for n in range(1, 12)], t).tracks]
        except mc.SelectionError:
            return False
        return got == want
    return ok


def check_central(p):
    out = {}
    a1 = has(p, "add_album", album="pray for the wicked", tracks=album_sel(list(range(1, 7))))
    sh = [s for s in p.steps if s.op == "shuffle"]
    rm2 = has(p, "remove", target=lambda t: t.lower() in ("pos:2", "2"))
    ph = has(p, "add_song", title="phoenix")
    mv = has(p, "move", to=lambda t: t.lower() in ("front", "top", "pos:1", "1"))
    rml = has(p, "remove", target=lambda t: t.lower() in ("last", "last:1"))
    hymns = [s for s in p.steps if s.op == "add_song"
             and "hymn" in (s.heard_title + s.title).lower()]
    a2 = has(p, "add_album", album="pray for the wicked", tracks=album_sel(list(range(7, 12))))
    out["first half 1-6"] = a1 is not None
    out["two shuffles"] = len(sh) == 2
    out["remove pos:2 (not a skip)"] = rm2 is not None and "skip" not in ops(p)
    out["phoenix by fall out boy"] = ph is not None and "fall out boy" in mc.norm(ph.artist)
    out["move binds the new phoenix"] = mv is not None and ph is not None and (
        f"step:{ph.id}" in mv.target or "phoenix" in mv.target.lower())
    out["remove last"] = rml is not None
    out["hymns ×5"] = sum(max(1, s.count) for s in hymns) == 5
    out["latter half 7-11"] = a2 is not None
    order = [idx(p, x) for x in (a1, rm2, ph, mv, rml)]
    out["order of the first part"] = all(i >= 0 for i in order) and order == sorted(order) and (
        len(sh) == 2 and idx(p, a1) < idx(p, sh[0]) < idx(p, rm2) and idx(p, rml) < idx(p, sh[1]))
    out["latter half waits for the 5 hymns"] = bool(hymns) and a2 is not None and \
        max(idx(p, h) for h in hymns) < idx(p, a2)
    return out


def check_priest(p):
    pr = has(p, "add_song", title="call my name")
    return {
        "four songs": sum(1 for s in p.steps if s.op in ("add_song", "find_song")) == 4,
        "Priest as said (the AI's reading may differ)": pr is not None
        and "priest" in mc.norm(pr.heard_artist or pr.artist),
        "AI didn't swap in Prince": pr is not None and "prince" not in mc.norm(pr.artist),
        "deal z → dlz": has(p, "add_song", artist="tv on the radio") is not None,
    }


def check_demi(p):
    k = [s for s in p.steps if "kaimara" in mc.norm(s.title + s.heard_title + s.value)
         or "chimera" in mc.norm(s.title + s.value)]
    return {
        "first three queued": all(has(p, "add_song", title=t) for t in
                                  ("american zombie", "below the belt", "devil")),
        "no invented artist for the trailing 'by'": all(not s.artist for s in k),
        "asks about what it couldn't resolve": bool(p.ask) or all(not s.artist for s in k),
    }


def check_wax(p):
    rm = has(p, "remove")
    return {"removes the WAX entry": rm is not None and (
        rm.target in ("entry:q23",) or "wax" in rm.target.lower()),
        "doesn't skip": "skip" not in ops(p)}


def check_except_last(p):
    a = has(p, "add_album", album="pray for the wicked")
    return {"tracks 1-10": a is not None and album_sel(list(range(1, 11)))(a.tracks)}


def check_rec1(p):
    a = has(p, "add_album", album="pray for the wicked")
    ph = has(p, "add_song", title="phoenix")
    sh = has(p, "shuffle")
    rml = has(p, "remove", target=lambda t: t.lower() in ("last", "last:1"))
    return {
        "first seven without the sixth": a is not None and (
            album_sel([1, 2, 3, 4, 5, 7])(a.tracks) or album_sel(list(range(1, 8)))(a.tracks)
            and has(p, "remove", target=lambda t: "6" in t) is not None),
        "phoenix": ph is not None, "shuffle": sh is not None, "remove last": rml is not None,
        "order": -1 < idx(p, a) < idx(p, ph) < idx(p, sh) < idx(p, rml),
    }


def check_rec2(p):
    adds = [s for s in p.steps if s.op == "add_song"]
    return {"seven songs": len(adds) == 7 or sum(s.count for s in adds) == 7,
            "named songs": all(s.title and s.artist for s in adds)}


def check_rec3(p):
    return {"shuffle then skip": ops(p)[:2] == ["shuffle", "skip"]}


def check_rec4(p):
    return {"three songs skipped/removed": has(p, "remove", target=lambda t: t.replace(" ", "")
                                                in ("range:1-3",)) is not None
            or has(p, "skip", count=3) is not None,
            "shuffle": has(p, "shuffle") is not None,
            "volume +10": vol(has(p, "volume"), ("+10", "up10", "110")),
            "loop": has(p, "loop") is not None}


def check_rec5(p):
    a = has(p, "add_album", album="pray for the wicked")
    rm = has(p, "remove", target=lambda t: "step:" in t or "match:" in t)
    return {"loop first": ops(p)[:1] == ["loop"],
            "volume -20": vol(has(p, "volume"), ("-20", "down20", "80")),
            "songs 2, 3, 11": a is not None and album_sel([2, 3, 11])(a.tracks),
            "phoenix": has(p, "add_song", title="phoenix") is not None,
            "removes the 2nd album song by reference": rm is not None and a is not None and (
                f"step:{a.id}#2" in rm.target or "match:" in rm.target)}


def check_rec6(p):
    a1 = has(p, "add_album", tracks=album_sel(list(range(1, 7))))
    ph = has(p, "add_song", title="phoenix")
    a2 = has(p, "add_album", tracks=album_sel(list(range(7, 12))))
    return {"first half": a1 is not None, "phoenix": ph is not None, "latter half": a2 is not None,
            "order": -1 < idx(p, a1) < idx(p, ph) < idx(p, a2)}


def check_rec7(p):
    a = has(p, "add_album", album="pray for the wicked", tracks=album_sel(list(range(1, 12))))
    adds = [s for s in p.steps if s.op == "add_song"]
    return {"whole album": a is not None, "five songs": len(adds) == 5,
            "skip": has(p, "skip") is not None, "shuffle": has(p, "shuffle") is not None,
            "volume -20": vol(has(p, "volume"), ("-20", "down20", "80"))}


def check_rec8(p):
    return {"one song": len([s for s in p.steps if s.op == "add_song"]) == 1}


def check_rec9(p):
    return {"volume -10": vol(has(p, "volume"), ("-10", "down10", "90")),
            "shuffle": has(p, "shuffle") is not None}


def check_rec10(p):
    s = has(p, "add_song", title="silver lining", artist="panic")
    return {"silver lining by panic (as said)": s is not None,
            "shuffle": has(p, "shuffle") is not None,
            "volume +10": vol(has(p, "volume"), ("+10", "up10", "110"))}


def check_rec11(p):
    s = has(p, "add_song", title="funky sundays", artist="wombat")
    return {"funky sundays by adhesive wombat": s is not None,
            "shuffle": has(p, "shuffle") is not None, "skip": has(p, "skip") is not None,
            "volume +10": vol(has(p, "volume"), ("+10", "up10", "110"))}


def check_move_recent(p):
    mv = has(p, "move")
    return {"moves the one they added": mv is not None and mv.target in ("entry:q9", "mine:last")}


def check_queue_question(p):
    return {"answered from the queue": p.kind == "answer" or "queue_info" in ops(p)}


def check_mixed(p):
    return {"question + skip": ("now_playing" in ops(p) or "answer" in ops(p)) and "skip" in ops(p)}


def check_dont(p):
    return {"does nothing": p.kind != "plan" or not p.steps}


def check_remove_last_two(p):
    rm = has(p, "remove")
    return {"remove last 2": rm is not None and rm.target.lower() in ("last:2", "range:2-3")}


def check_second_half(p):
    a = has(p, "add_album", tracks=album_sel(list(range(7, 12))))
    return {"tracks 7-11": a is not None}


def check_skip_next(p):
    return {"next song removed, current kept": has(p, "remove", target=lambda t: t.lower() in (
        "pos:1", "first", "next", "1")) is not None and "skip" not in ops(p)}


def check_five_then_move(p):
    s = has(p, "add_song", title="hymns")
    mv = has(p, "move")
    if s is None:
        s = has(p, "add_song", title="travesties")
    return {"5 copies": s is not None and s.count == 5,
            "moves one of them": mv is not None and s is not None and f"step:{s.id}" in mv.target}


CENTRAL = ("Hey DJ, play the first half of the Pray for the Wicked album by Panic! At The "
           "Disco, shuffle, skip the second song in the queue, play The Phoenix by Fallout Boy, "
           "move the Phoenix to the front of the queue, remove the last song from the queue, "
           "then shuffle the queue again and play the latter half of Pray for the Wicked, but "
           "only after you have queued I Write Hymns, not Travesties by Yung Gravy five "
           "separate times.")

#: The reference transcripts of the real microphone recordings (
#: faster-whisper large-v3, beam 5) — what was said, not what base.en heard.
RECORDINGS = {
    1: "Hey DJ, can you play the first seven songs by Panic at the Disco's Pray for the Wicked "
       "album but remove the sixth song and then after that can you also cue up The Phoenix by "
       "Fallout Boy and then shuffle the whole cue and then remove the last song from the cue?",
    2: "Hey DJ, can you play seven completely random songs that were popular in 2020?",
    3: "Hey DJ, can you shuffle the cue and skip the current song?",
    4: "Hey DJ, skip the next three songs and shuffle the cue and turn the volume up by 10% and "
       "then set the bot to loop.",
    5: "Hey DJ, set the bot to loop and then turn the volume down by 20% and then skip the next "
       "song and then play songs 2, 3, and 11 from Panic at the Disco's Pray for the Wicked "
       "album and then play The Phoenix by Fall Out Boy and then remove the second song that I "
       "asked you to play from the Pray for the Wicked album.",
    6: "Hey DJ, play the first half of Panic at the Disco's Pray for the Wicked album, and then "
       "play The Phoenix by Fall Out Boy, and then play the latter half of Panic at the Disco's "
       "Pray for the Wicked album.",
    7: "Hey DJ, play Panic at the Disco's Pray for the Wicked album and then play five random "
       "very popular songs from 2019 and then skip the current song that's playing and shuffle "
       "the cue and turn the volume down by 20%.",
    8: "Hey DJ, play a random song.",
    9: "Hey DJ, turn the volume down by 10% and shuffle the cue.",
    10: "Hey DJ, play Fuck A Silver Lining by Panic at the Disco and then shuffle the cue and "
        "turn the volume up by 10%.",
    11: "Hey DJ, play Funky Sundays by Adhesive Wombat and then shuffle the cue and then skip "
        "the song that is currently playing and turn the volume up by 10%.",
}
_REC_CHECK = {1: check_rec1, 2: check_rec2, 3: check_rec3, 4: check_rec4, 5: check_rec5,
              6: check_rec6, 7: check_rec7, 8: check_rec8, 9: check_rec9, 10: check_rec10,
              11: check_rec11}

CASES = [
    ("central (idle)", CENTRAL, IDLE, check_central),
    ("central (playing)", CENTRAL, PLAYING, check_central),
    ("priest", "Hey DJ, can you play deal Z by TV on the radio and can you also play call my name "
               "by priest and can you play slowly and by infected mushroom and can you play ghosts "
               "and monsters by saying chaos?", PLAYING, check_priest),
    ("demi … kaimara by", "Hey DJ, can you play American Zombie by Demi the Daredevil and Below "
                          "the Belt by Point North and the Devil's Den by Skrillex and Filium by "
                          "Wax and Million and Kaimara by", IDLE, check_demi),
    ("remove wax", "Hey DJ, can you remove that song called Wax by Alte One Den from the queue?",
     WAX, check_wax),
    ("all except last", "Can you queue up the entire pray for the Wicked album by Panic at the "
                        "Disco except for the last song?", PLAYING, check_except_last),
    *[(f"recording {n}", t, PLAYING, _REC_CHECK[n]) for n, t in RECORDINGS.items()],
    ("move the one I added", "hey dj move the one I just added to the front", RECENT_PHOENIX,
     check_move_recent),
    ("what's next", "hey dj what's coming up next?", PLAYING, check_queue_question),
    ("mixed question + skip", "hey dj what song is this, and skip it after", PLAYING, check_mixed),
    ("don't", "hey dj don't skip this one, I love it", PLAYING, check_dont),
    ("remove last two", "hey dj take the last two songs off the queue", PLAYING,
     check_remove_last_two),
    ("second half paraphrase", "hey dj queue the back half of pray for the wicked", PLAYING,
     check_second_half),
    ("skip the next song", "hey dj skip the next song", PLAYING, check_skip_next),
    ("five then move", "hey dj queue i write hymns not travesties by yung gravy five times and "
                       "then move one of those to the front", PLAYING, check_five_then_move),
]


def heard_cases(path: str) -> list:
    """The recordings as the bot's own recogniser heard them (a
    scripts/voice_replay.py --out file): the same checks as the reference
    transcripts, so recognition errors and planning are tested together."""
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    out = []
    for r in rows:
        n = int("".join(c for c in os.path.splitext(r["file"])[0] if c.isdigit()) or 0)
        if n in _REC_CHECK and r.get("heard"):
            out.append((f"recording {n} (as heard)", r["heard"], PLAYING, _REC_CHECK[n]))
    return out


def catalogue_lookup():
    """The bot's own planner lookup (cogs/voicedj.py VoiceDJ._plan_lookup:
    the real Deezer/iTunes catalogue, read-only), so a model that asks for an
    album's tracks first gets the same facts it would live."""
    import functools

    from cogs.voicedj import VoiceDJ
    return functools.partial(VoiceDJ._plan_lookup, None)


async def run(models, rounds, key, out, only=None, extra=(), lookup=None):
    results = {}
    for model in models:
        router = LLMRouter(key, model=model, fallbacks=(), timeout=45.0, per_minute=1000)
        rows = []
        for r in range(rounds):
            for name, said, situation, check in [*CASES, *extra]:
                if only and only not in name:
                    continue
                ctx = plan_context(current=situation["current"], upcoming=situation["upcoming"],
                                   player=situation.get("player", ""),
                                   recent=situation.get("recent", ()))
                text = said.split(",", 1)[1].strip() if said.lower().startswith("hey dj,") else (
                    said[6:].strip() if said.lower().startswith("hey dj ") else said)
                t0 = time.monotonic()
                plan = await router.plan(text, ctx, timeout=45.0, lookup=lookup)
                secs = time.monotonic() - t0
                if plan is None:
                    checks = {"usable answer": False}
                    desc = (f"(no plan: {router.last_error[:160]}) RAW: "
                            + getattr(router, "last_raw", "")[:3000].replace("\n", " "))
                else:
                    try:
                        checks = check(plan)
                    except Exception as e:
                        checks = {f"check crashed: {type(e).__name__}": False}
                    desc = vp.describe(plan)
                ok = all(checks.values())
                row = {"model": model, "round": r + 1, "case": name, "ok": ok,
                       "checks": checks, "seconds": round(secs, 2),
                       "cost": round(plan.cost, 6) if plan else 0.0, "plan": desc}
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{model:32} r{r + 1} {name:26} {'PASS' if ok else 'FAIL'} {secs:5.2f}s "
                      + ("" if ok else "✗ " + ", ".join(k for k, v in checks.items() if not v)),
                      flush=True)
        await router.close()
        secs = [x["seconds"] for x in rows]
        results[model] = (sum(x["ok"] for x in rows), len(rows),
                          statistics.median(secs) if secs else 0,
                          sorted(secs)[int(0.9 * (len(secs) - 1))] if secs else 0,
                          sum(x["cost"] for x in rows))
    print("\nmodel                             pass   median   p90    cost")
    for m, (ok, n, med, p90, cost) in results.items():
        print(f"{m:32} {ok:3}/{n:<3} {med:6.2f}s {p90:6.2f}s ${cost:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="a file with OPENROUTER_API_KEY=…")
    ap.add_argument("--pid", type=int, default=None,
                    help="…or the running bot's process id: its environment has the key")
    ap.add_argument("--models", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", default="voice_plan_bench.jsonl")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-lookups", action="store_true",
                    help="don't answer the model's catalogue lookups (default: the real catalogue)")
    ap.add_argument("--heard", default=None,
                    help="also plan what the recogniser heard (scripts/voice_replay.py --out)")
    a = ap.parse_args()
    key = _key(a.env, a.pid)
    with open(a.out, "a", encoding="utf-8") as out:
        asyncio.run(run([m.strip() for m in a.models.split(",") if m.strip()], a.rounds, key,
                        out, a.only, heard_cases(a.heard) if a.heard else (),
                        None if a.no_lookups else catalogue_lookup()))


if __name__ == "__main__":
    main()
