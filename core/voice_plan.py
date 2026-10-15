"""
Voice DJ plans: what one spoken request asks for, as explicit steps.

The planner (core/voice_llm.py) turns a request into a Plan — numbered
steps, each one operation from the capability registry (core/voice_tools.py)
with its arguments, a repeat count, and the steps it must wait for — and the
executor (core/voice_exec.py) runs it against the live queue. This module is
the pure part in between: parsing and checking a plan, the order to run it
in, and what queue references mean.

Queue references ("target" / "to") are small strings, so the model can name
exactly which entry it means without positions going stale:

  pos:N        the N-th upcoming song (1 = the next one; the playing song is
               not in the queue)        last / last:N   the last (N) upcoming
  first        = pos:1                  range:A-B       positions A..B
  current      the song playing now     entry:qN        one queued occurrence,
                                                        by the id shown to the model
  step:K       what step K added (all of it) · step:K#J  the J-th of those
  match:TEXT   the queued entry whose title/artist best fits TEXT ("that song
               called Wax by Alte One Den")
  mine:last    the latest entry this speaker added
  Destinations: front · end · pos:N · after:REF · before:REF

Positions are resolved when the step runs, against the queue as it is then —
never all at the start (core/voice_exec.py also checks nobody changed the
queue under the request in a way that makes them point elsewhere).

Tests: tests/test_voice_plan.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from . import music_catalog as mc

#: Most steps one request may have, and most queue entries it may add —
#: beyond these the plan is refused with a reason, never silently cut.
MAX_STEPS = 40
MAX_REPEAT = 25
MAX_NEW_ENTRIES = 120


@dataclass(frozen=True)
class Step:
    id: int
    op: str
    title: str = ""
    artist: str = ""
    heard_title: str = ""            # the title as the recogniser wrote it
    heard_artist: str = ""           # the words said after "by", as written
    album: str = ""
    tracks: str = ""                 # album selector (core/music_catalog.py)
    target: str = ""                 # queue reference
    to: str = ""                     # destination / second reference
    value: str = ""
    count: int = 1                   # repeat (adds), how many (skip/top/latest)
    when: str = ""                   # songs: "" (end of the queue) | next | now
    after: tuple[int, ...] = ()      # steps that must be done first
    took_skip: bool = False          # a "skip this" folded into these songs' "now"

    @property
    def label(self) -> str:
        return f"step {self.id} ({self.op})"


@dataclass(frozen=True)
class Plan:
    """kind: plan | answer | other | lookup (the model wants facts first)."""
    kind: str
    steps: tuple[Step, ...] = ()
    reply: str = ""                  # a spoken answer (answer), or a note
    ask: str = ""                    # a focused question for what it couldn't resolve
    lookups: tuple[dict, ...] = ()
    problems: tuple[str, ...] = ()   # why steps were refused, in words
    model: str = ""
    seconds: float = 0.0
    cost: float = 0.0
    raw: str = field(default="", repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.problems


# ── parsing what the model wrote ─────────────────────────────────────────────

_TEXT_LIMIT = 300


def _s(v, limit: int = _TEXT_LIMIT) -> str:
    return " ".join(str(v or "").split())[:limit]


def _spilled(*texts: str) -> bool:
    """The model's JSON spilling into a string ('X},{') — not a name."""
    return any("{" in t or "}" in t for t in texts)


def parse_plan(content: str, *, ops) -> Plan | None:
    """The model's JSON → Plan, or None if it isn't usable JSON at all.
    Steps with unknown operations, broken references or impossible counts
    are kept out *and listed in `problems`* — the caller reports them."""
    if not isinstance(content, str) or not content.strip():
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
    kind = _s(data.get("kind"), 20).lower()
    reply = _s(data.get("reply"), 600)
    ask = _s(data.get("ask"), 300)
    if kind == "lookup":
        looks = tuple(x for x in (data.get("lookups") or []) if isinstance(x, dict))[:4]
        return Plan("lookup", lookups=looks, raw=content) if looks else None
    if kind in ("answer", "question"):
        return Plan("answer", reply=reply, ask=ask, raw=content) if (reply or ask) else Plan(
            "other", raw=content)
    if kind != "plan":
        return Plan("other", reply=reply, raw=content)
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        return None
    problems: list[str] = []
    steps: list[Step] = []
    if len(raw_steps) > MAX_STEPS:
        problems.append(f"that's {len(raw_steps)} steps — I can do at most {MAX_STEPS} "
                        f"in one request")
        return Plan("plan", (), reply, ask, problems=tuple(problems), raw=content)
    seen_ids: set[int] = set()
    for n, item in enumerate(raw_steps, 1):
        if not isinstance(item, dict):
            problems.append(f"step {n} wasn't readable")
            continue
        try:
            sid = int(item.get("id") or n)
        except (TypeError, ValueError):
            sid = n
        if sid in seen_ids:
            sid = max(seen_ids) + 1
        seen_ids.add(sid)
        op = _s(item.get("op"), 40).lower()
        fields = {k: _s(item.get(k)) for k in ("title", "artist", "heard_title", "heard_artist",
                                               "album", "tracks", "target", "to", "value")}
        for k in ("target", "to"):                     # "pos:1-3" is the range 1-3
            fields[k] = re.sub(r"(?i)^pos:(\d+)\s*-\s*(\d+)$", r"range:\1-\2", fields[k])
            # an entry id as the context shows it ("q23", "[q23]"): the entry
            fields[k] = re.sub(r"(?i)^((?:after|before):)?\[?(q\d+)\]?$", r"\1entry:\2", fields[k])
        if _spilled(*fields.values()):
            problems.append(f"step {sid} came back garbled")
            continue
        try:
            count = int(item.get("count") or 1)
        except (TypeError, ValueError):
            count = 1
        when = _s(item.get("when"), 10).lower()
        when = when if when in ("next", "now") else ""
        after = []
        for a in item.get("after") or []:
            try:
                after.append(int(a))
            except (TypeError, ValueError):
                continue
        if op not in ops:
            problems.append(f"step {sid} asked for something I can't do (“{op or '?'}”)")
            continue
        if count < 1:
            count = 1
        after = [a for a in dict.fromkeys(after) if a != sid]      # (a step can't wait for itself)
        if op == "skip" and fields["target"] and fields["target"].lower() not in ("current", "this"):
            # "skip the next song" with a queue position: that upcoming song
            # is taken off the queue — the one playing isn't touched.
            op, count = "remove", 1
        steps.append(Step(sid, op, count=count, when=when, after=tuple(after), **fields))
    plan = Plan("plan", tuple(steps), reply, ask, problems=tuple(problems), raw=content)
    return check_plan(plan, ops=ops)


# ── checking a plan ──────────────────────────────────────────────────────────

_REF = re.compile(r"^(?:pos:\d+|\d+|last(?::\d+)?|first|next|current|this|range:\d+-\d+|"
                  r"entry:q\d+|step:\d+(?:#\d+)?|match:.+|mine:last)$", re.I)
_DEST = re.compile(r"^(?:front|top|end|back|pos:\d+|\d+|(?:after|before):.+)$", re.I)


def ref_ok(ref: str, kind: str = "ref") -> bool:
    """Is `ref` a reference (kind "ref") / destination (kind "dest") we can follow?"""
    r = (ref or "").strip()
    if kind == "dest":
        low = r.lower()
        for word in ("after:", "before:"):
            if low.startswith(word):
                return bool(_REF.match(r[len(word):]))
        return bool(_DEST.match(r))
    return bool(_REF.match(r))


def ref_steps(ref: str) -> list[int]:
    """Step ids a reference (or destination) points at."""
    return [int(x) for x in re.findall(r"step:(\d+)", ref or "")]


def check_plan(plan: Plan, *, ops) -> Plan:
    """Everything that can be checked before running: required arguments,
    references (to steps that exist and come first), dependency cycles, and
    the size limits. Bad steps are dropped with a reason; steps that depend
    on a dropped one are dropped too (they'd act on the wrong thing)."""
    problems = list(plan.problems)
    ids = {s.id for s in plan.steps}
    good: dict[int, Step] = {}
    dropped: set[int] = set()
    new_entries = 0
    for s in plan.steps:
        spec = ops[s.op]
        missing = [f for f in spec.required if not getattr(s, f)]
        why = None
        if missing:
            why = f"step {s.id} ({spec.say}) is missing its {', '.join(missing)}"
        for name in ("target", "to"):
            ref = getattr(s, name)
            if ref and name in spec.refs and not ref_ok(ref, spec.refs[name]):
                why = why or f"step {s.id} points at something I can't follow (“{ref}”)"
        for k in ref_steps(s.target) + ref_steps(s.to):
            if k == s.id:
                why = why or (f"step {s.id} points at what it adds itself — I can't tell which "
                              f"song that means")
            elif k not in ids:
                why = why or f"step {s.id} points at a step {k} that isn't there"
            elif k in dropped:
                why = why or f"step {s.id} acts on what step {k} adds, and I couldn't use step {k}"
        if any(a not in ids or a in dropped for a in s.after):
            s = replace(s, after=tuple(a for a in s.after if a in ids and a not in dropped))
        if s.count > MAX_REPEAT and spec.adds:
            why = why or (f"step {s.id} asks for {s.count} copies — at most {MAX_REPEAT} "
                          f"at a time")
        if why:
            problems.append(why)
            dropped.add(s.id)
            continue
        if spec.adds:
            new_entries += s.count if s.op not in ("add_album", "add_top", "add_latest") else 0
        good[s.id] = s
    if new_entries > MAX_NEW_ENTRIES:
        problems.append(f"that would add {new_entries} songs — at most {MAX_NEW_ENTRIES} "
                        f"in one request")
        return replace(plan, steps=(), problems=tuple(problems))
    order = run_order(list(good.values()))
    if order is None:
        problems.append("the steps wait for each other in a circle, so none can go first")
        return replace(plan, steps=(), problems=tuple(problems))
    return replace(plan, steps=tuple(order), problems=tuple(problems))


def deps(step: Step) -> set[int]:
    """Steps this one waits for (ordering): explicit "after" and the steps
    whose results it points at."""
    return set(step.after) | set(ref_steps(step.target)) | set(ref_steps(step.to))


def needs(step: Step) -> set[int]:
    """Steps whose *results* this one acts on ("move what step 4 added"): if
    one of those didn't work, this one can't either. (An "after" alone only
    orders: "…, but only after you've queued X" still happens if X can't be
    found — reported.)"""
    return set(ref_steps(step.target)) | set(ref_steps(step.to))


def run_order(steps: list[Step]) -> list[Step] | None:
    """The order to run steps in: as listed (the order said), except that a
    step waits for everything it depends on — "…, but only after you have
    queued X five times" makes a step listed earlier wait for the later
    ones. Stable: among steps that are ready, the first listed goes first.
    None if the dependencies form a cycle."""
    by_id = {s.id: s for s in steps}
    pending = list(steps)
    done: set[int] = set()
    out: list[Step] = []
    while pending:
        for i, s in enumerate(pending):
            if all(d in done or d not in by_id for d in deps(s)):
                out.append(s)
                done.add(s.id)
                pending.pop(i)
                break
        else:
            return None
    return out


# ── references against a queue ───────────────────────────────────────────────

class RefError(ValueError):
    """A reference that doesn't point at anything in the queue (now)."""


def _qid(e) -> str | None:
    return e.get("_qid") if isinstance(e, dict) else None


def entry_label(e: dict | None) -> str:
    if not e:
        return "?"
    return _s(e.get("title") or "?", 90)


#: "that song called X", "the one by Y": words about the song, not its name.
_SAID_FILLER = re.compile(r"^(?:(?:that|the|this)\s+)?(?:song|one|track|tune)\s+"
                          r"(?:(?:that'?s|which is)\s+)?(?:called|named|titled)\s+", re.IGNORECASE)

_DISPLAY_NOISE = re.compile(
    r"[\(\[][^\)\]]*(official|lyric|lyrics|audio|video|visualizer|hd|hq|4k|explicit)"
    r"[^\)\]]*[\)\]]", re.IGNORECASE)


def match_score(text: str, entry: dict) -> float:
    """How well spoken `text` ("Wax by Alte One Den") names a queue entry —
    its display title ("WAX - ALTE WUNDEN"), with its artist/channel, by
    spelling and by sound, in either order ("X by Y" for "Y - X"), ignoring
    the "(Official Audio)" kind of noise uploads carry."""
    title = _DISPLAY_NOISE.sub("", entry.get("title") or "").strip()
    artist = re.sub(r"\s*-\s*topic$", "", entry.get("artist") or entry.get("uploader") or "",
                    flags=re.IGNORECASE)
    said = " ".join((text or "").split())
    said = _SAID_FILLER.sub("", said)
    by = re.split(r"\s+by\s+", said, maxsplit=1, flags=re.IGNORECASE)
    # Both halves of "X by Y" must fit — never just one ("Wax" alone would
    # pick any song by an artist called Wax).
    variants = [" ".join(by)] if len(by) == 2 else [said]
    if len(by) == 2:
        variants.append(f"{by[1]} {by[0]}")
    cands = [title, f"{artist} {title}", f"{title} {artist}"]
    if " - " in title:
        a, b = title.split(" - ", 1)
        cands += [f"{b} {a}"]
        if len(by) != 2:
            cands.append(b)                     # "remove Mr Brightside": the title alone
    return max(mc.title_score(v, c) for v in variants for c in cands if c.strip() and v.strip())


def resolve_target(ref: str, upcoming: list[dict], *, current: dict | None = None,
                   step_entries: dict[int, list[str]] | None = None,
                   mine: list[str] | None = None) -> list[str]:
    """The occurrence ids `ref` means in `upcoming` (queue order), or the
    special id "current". Raises RefError when it points at nothing."""
    r = (ref or "").strip()
    low = r.lower()
    n = len(upcoming)
    ids = [_qid(e) for e in upcoming]
    if low in ("current", "this"):
        if current is None:
            raise RefError("nothing is playing")
        return ["current"]
    if low in ("first", "next"):
        low = "pos:1"
    if re.fullmatch(r"\d+", low):
        low = f"pos:{low}"
    if low.startswith("pos:"):
        p = int(low[4:])
        if not 1 <= p <= n:
            raise RefError(f"there's no song {p} in the queue (it has {n})" if n
                           else "the queue is empty")
        return [ids[p - 1]]
    if low == "last" or low.startswith("last:"):
        k = int(low[5:]) if ":" in low else 1
        if n == 0:
            raise RefError("the queue is empty")
        return ids[max(0, n - k):]
    if low.startswith("range:"):
        a, b = sorted(int(x) for x in low[6:].split("-"))
        if n == 0:
            raise RefError("the queue is empty")
        if a > n:
            raise RefError(f"there's no song {a} in the queue (it has {n})")
        return ids[max(1, a) - 1:min(b, n)]
    if low.startswith("entry:"):
        q = r.split(":", 1)[1]
        if current is not None and _qid(current) == q:
            return ["current"]
        if q not in ids:
            raise RefError("that song isn't in the queue any more")
        return [q]
    if low.startswith("step:"):
        m = re.fullmatch(r"step:(\d+)(?:#(\d+))?", low)
        made = (step_entries or {}).get(int(m.group(1)), [])
        if m.group(2):
            j = int(m.group(2))
            made = made[j - 1:j] if 1 <= j <= len(made) else []
        live = [q for q in made if q in ids]
        if current is not None and _qid(current) in made and not live:
            return ["current"]
        if not live:
            raise RefError("what that step added isn't in the queue any more")
        return live
    if low == "mine:last":
        for q in reversed(mine or []):
            if q in ids:
                return [q]
            if current is not None and _qid(current) == q:
                return ["current"]
        raise RefError("nothing you added is still queued")
    if low.startswith("match:"):
        text = r.split(":", 1)[1]
        scored = sorted(((match_score(text, e), i) for i, e in enumerate(upcoming)),
                        key=lambda x: -x[0])
        if not scored or scored[0][0] < MATCH_MIN:
            raise RefError(f"nothing in the queue sounds like “{text}”")
        top = scored[0][0]
        close = [i for s, i in scored if s >= top - MATCH_TIE]
        labels = {mc.norm(entry_label(upcoming[i])) for i in close}
        if len(labels) > 1:
            a, b = (entry_label(upcoming[i]) for i in close[:2])
            raise RefError(f"“{text}” could be {a} or {b}")
        # Several copies of the same song: the one this person added last, or
        # else the newest — "that song" is rarely the oldest copy.
        own = [i for i in close if ids[i] in set(mine or ())]
        pick = max(own or close, key=lambda i: _qid_number(ids[i]))
        return [ids[pick]]
    raise RefError(f"I can't follow “{ref}”")


#: A queued song is only taken to be the one named if its title (and
#: artist/channel) fit this well — by spelling or by sound.
MATCH_MIN = 0.6
#: Names scoring within this of the best are "just as good": the same song
#: (copies — the newest is taken) or a real ambiguity (the speaker is asked).
MATCH_TIE = 0.03


def _qid_number(entry_id: str | None) -> int:
    m = re.fullmatch(r"q(\d+)", entry_id or "")
    return int(m.group(1)) if m else -1


def resolve_destination(to: str, upcoming: list[dict], *, view: list[dict] | None = None,
                        **kw) -> int:
    """0-based index in `upcoming` (the live queue with the entries being
    moved already taken out) that a destination means. "after:REF" /
    "before:REF" are resolved like targets — against `view`, the request's
    own picture of the queue, when given — and then found in `upcoming`."""
    low = (to or "").strip().lower()
    n = len(upcoming)
    if low in ("front", "top", "", "pos:1", "1", "next", "first"):
        return 0
    if low in ("end", "back", "last", "bottom"):
        return n
    if re.fullmatch(r"(?:pos:)?\d+", low):
        p = int(low.split(":")[-1])
        return max(0, min(p - 1, n))
    for word, off in (("after:", 1), ("before:", 0)):
        if low.startswith(word):
            ids = resolve_target(to[len(word):], view if view is not None else upcoming, **kw)
            if ids == ["current"]:
                return 0
            pos = [i for i, e in enumerate(upcoming) if _qid(e) in ids]
            if not pos:
                raise RefError("the song it should go next to isn't in the queue any more")
            return (max(pos) + 1) if off else min(pos)
    raise RefError(f"I can't follow “{to}”")


#: Operations that act on whatever is queued or playing — after a request's
#: songs failed to go in, they'd act on the wrong songs ("…, shuffle, skip the
#: first one, loop the rest").
_ACTS_ON_QUEUE = frozenset({"skip", "shuffle", "reverse", "balance", "dedupe", "loop",
                            "restart", "skipto", "seek"})


def uses_positions(step: Step) -> bool:
    """Does what this step acts on depend on where things are in the queue,
    or on what's playing (rather than on one named song)? Those are the steps
    that would hit the wrong song if an earlier song of the request didn't
    make it into the queue."""
    refs = [step.target, step.to]
    return any(re.match(r"(?i)^(pos:|\d|last|first|next|range:|front|top|end|back|current)",
                        r or "") for r in refs) or step.op in _ACTS_ON_QUEUE


# ── the grammar's (and old-style) actions as a plan ──────────────────────────

def from_intents(intents) -> Plan:
    """core/voice_intent.py Intents (the offline grammar, or an old-style
    "commands" answer) → a Plan with the same meaning."""
    steps: list[Step] = []
    for n, i in enumerate(intents, 1):
        a, when = i.action, (i.when or "")
        if a == "play":
            steps.append(Step(n, "add_song", title=i.arg or "", artist=i.artist or "",
                              heard_title=i.heard or "", when=when))
        elif a == "find":
            steps.append(Step(n, "find_song", value=i.arg or "", artist=i.artist or "", when=when))
        elif a == "album":
            steps.append(Step(n, "add_album", album=i.arg or "", artist=i.artist or "",
                              tracks=i.count or "all", when=when))
        elif a in ("top", "latest"):
            steps.append(Step(n, f"add_{a}", artist=i.artist or i.arg or "",
                              count=mc.parse_count(i.count, 5 if a == "top" else 1, mc.MAX_TOP),
                              when=when))
        elif a == "skip":
            steps.append(Step(n, "skip", count=int(i.count) if (i.count or "").isdigit() else 1))
        elif a == "volume":
            steps.append(Step(n, "volume", value=i.arg or ""))
        elif a == "loop":
            steps.append(Step(n, "loop", value=i.arg or "song"))
        elif a in ("pause", "resume", "restart", "stop", "clear", "shuffle"):
            steps.append(Step(n, a))
    return Plan("plan", tuple(steps))


def fold_skips(plan: Plan) -> Plan:
    """A plain "skip" right before songs ("skip this and play X", "skip this
    song and instead play …") — or right after songs wanted "next"/"now" —
    makes those songs "now": they go in front, then one skip. Done as said,
    the skip started whatever was queued next and the songs went to the end
    of the queue (the model gave a reported sentence that way twice on
    2026-10-12). Only a bare skip (one song, no target) next to an add."""
    steps = list(plan.steps)
    took: set[int] = set()
    drop: set[int] = set()
    adds = {"add_song", "find_song", "add_album", "add_top", "add_latest"}
    for i, s in enumerate(steps):
        if s.op != "skip" or s.count > 1 or s.target or s.after:
            continue
        for j, any_when in ((i + 1, True), (i - 1, False)):
            if (0 <= j < len(steps) and j not in took and steps[j].op in adds
                    and (any_when or steps[j].when in ("next", "now"))):
                steps[j] = replace(steps[j], when="now", took_skip=True)
                took.add(j)
                drop.add(i)
                break
    if not drop:
        return plan
    gone = {steps[i].id for i in drop}
    kept = [replace(s, after=tuple(a for a in s.after if a not in gone))
            for i, s in enumerate(steps) if i not in drop]
    return replace(plan, steps=tuple(kept))


def from_reply(reply) -> Plan:
    """An old-style LLMReply (core/voice_llm.py: commands / question /
    other) as a Plan."""
    if reply.kind == "commands":
        return replace(from_intents(reply.intents), model=reply.model, seconds=reply.seconds,
                       cost=reply.cost)
    if reply.kind == "question":
        return Plan("answer", reply=reply.reply, model=reply.model, seconds=reply.seconds)
    return Plan("other", model=reply.model, seconds=reply.seconds)


def describe(plan: Plan) -> str:
    """A plan in one line, for logs (no user ids, just what it does)."""
    if plan.kind != "plan":
        return f"{plan.kind}: {plan.reply[:120]!r}" + (f" ask {plan.ask!r}" if plan.ask else "")
    parts = []
    for s in plan.steps:
        bits = [f"{k}={getattr(s, k)!r}" for k in ("title", "artist", "heard_title",
                                                     "heard_artist", "album", "tracks",
                                                     "target", "to", "value") if getattr(s, k)]
        if s.count != 1:
            bits.append(f"count={s.count}")
        if s.when:
            bits.append(f"when={s.when}")
        if s.after:
            bits.append(f"after={list(s.after)}")
        parts.append(f"{s.id}:{s.op}(" + ", ".join(bits) + ")")
    out = "; ".join(parts)
    if plan.problems:
        out += " | problems: " + "; ".join(plan.problems)
    if plan.ask:
        out += f" | ask: {plan.ask!r}"
    return out
