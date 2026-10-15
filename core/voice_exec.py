"""
The voice DJ's executor: one spoken request's plan (core/voice_plan.py),
step by step, against the live queue.

What it guarantees, and how:

  * **Steps run in the planned order** (voice_plan.run_order: as said, except
    that "…, but only after X" makes X go first) and each finishes — songs
    looked up and in the queue, an idle player started on the first one —
    before the next begins.
  * **References are resolved when the step runs**, never all at the start.
    The request keeps a *view* of the queue: what the planner was shown, then
    the queue exactly as each of its own steps left it. "The second song in
    the queue" is the view's second entry; the step then acts on that entry
    *by its occurrence id* (core/state.py) in the live queue. If someone else
    changed the queue meanwhile — another person's request, a typed command,
    the song simply ending — the entry is either still there (and that is
    the one acted on, wherever it is now) or it isn't, and then nothing is
    moved or removed in its place: the step is reported as a conflict.
  * **Songs this request adds** are referred to by the step that added them
    ("step:4" — "move The Phoenix to the front" right after adding it), each
    copy with its own id, so older copies of the same song are never
    mistaken for the new one.
  * **Nothing is silently dropped.** Every step ends as done, no-op, failed,
    conflict, skipped (it depended on a step that failed) or not done (the
    request was stopped), and the reply lists what happened to each.
  * **No stale plan resumes.** "stop"/"clear" — typed or said — cancel the
    request between steps (and its lookup in flight); a stop or clear by
    someone else between two steps ends it; the speaker opting out, leaving,
    or the bot moving ends it.

Lookups (song searches, catalogue) are awaited outside any critical section;
each queue change is a synchronous find-check-change on the event loop,
which nothing else can interleave with.

Tests: tests/test_voice_exec.py (the 13-effect request from the round-8 brief
among them), tests/test_voice_plan.py.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import queue_ops, voice_plan as vp
from .logging_setup import get_logger
from .state import get_state, qid
from .voice_intent import VOICE_VOLUME_MAX, Intent
from .voice_tools import OPS

log = get_logger(__name__)

#: Song operations and the Intent action that carries them through the cog.
SONG_OPS = {"add_song": "play", "find_song": "find", "add_album": "album",
            "add_top": "top", "add_latest": "latest"}
#: Player controls the cog already implements as Intents (cogs/voicedj.py
#: _execute_intent): the same code the offline grammar's commands run.
CONTROL_OPS = {"skip", "pause", "resume", "restart", "stop", "clear", "volume", "loop"}
QUEUE_REF_OPS = {"remove", "move", "swap", "skipto"}
WHOLE_QUEUE_OPS = {"shuffle", "reverse", "balance", "dedupe"}
INFO_OPS = {"now_playing", "queue_info", "history", "answer"}

#: Final states that count as "it happened".
OK = ("done", "noop")
#: What acts on whatever is queued or playing (not on one named song).
_ACTS_ON_CONTENT = frozenset({"skip", "shuffle", "reverse", "balance", "dedupe", "loop",
                              "restart", "seek", "skipto"})

#: How to ask for an input a command needs that the request didn't give (by
#: the command's parameter name, in words).
_MISSING_ASK = {
    ("save_queue", "name"): "What should I call the playlist?",
    ("search", "query"): "What should I search for?",
    "when": "When should I remind you?",
    "name": "Which playlist?",
    "query": "Which song?",
    "position": "Where in the song — what time?",
    "reminder id": "Which reminder — what number?",
}


@dataclass
class StepResult:
    step: vp.Step
    status: str                    # done | noop | failed | conflict | skipped | not_done | cancelled
    message: str = ""              # for the reply ("" = nothing to say)
    added: tuple[str, ...] = ()    # occurrence ids put in the queue
    removed: tuple[str, ...] = ()
    moved: tuple[str, ...] = ()
    answer: str = ""               # a spoken answer (questions about the state, or the model's)
    why: str = ""                  # sanitized, for the log
    settle: bool = False           # it started an idle player: wait for its first song
    ask: str = ""                  # a question for the speaker (what it couldn't resolve)

    @property
    def ok(self) -> bool:
        return self.status in OK


@dataclass
class View:
    """The queue as this request believes it to be (see the module doc)."""
    current: dict | None
    upcoming: list[dict] = field(default_factory=list)

    @classmethod
    def capture(cls, state) -> "View":
        return cls(state.current, list(state.queue))

    def ids(self) -> list[str | None]:
        return [qid(e) for e in self.upcoming]


#: Steps whose own effect is a new order of the queue: after them the
#: request's view takes the live order (of the entries it knows about).
_REORDERS = frozenset({"shuffle", "reverse", "balance", "move", "swap", "dedupe", "skipto"})
#: Steps that, when they work, change what is playing on purpose.
_CHANGES_CURRENT = frozenset({"skip", "skipto", "stop", "back"})


def next_view(prev: View, state, step: vp.Step, result: StepResult) -> View:
    """The request's view after one of its own steps: the previous view with
    *this step's* effect applied — and nobody else's. What others added
    meanwhile isn't in it; what vanished without this request's doing (the
    song it named started playing, someone removed it) stays where it was, as
    the entry the speaker meant — so "the second song in the queue" keeps
    meaning the same song, and acting on it reports that it's gone instead of
    hitting the one that moved up into its place."""
    live = list(state.queue)
    live_ids = [qid(e) for e in live]
    by_id = {qid(e): e for e in prev.upcoming}
    by_id.update({qid(e): e for e in live})
    worked = result.status in OK
    ours_current = worked and (step.op in _CHANGES_CURRENT or result.settle
                               or (step.op in SONG_OPS and step.when == "now"))
    current = state.current if ours_current else prev.current
    cur_id = qid(state.current) if ours_current else None
    known = [i for i in prev.ids() if i not in set(result.removed)]
    added = [i for i in result.added if i != cur_id]
    keep = set(known) | set(added)
    if worked and step.op in _REORDERS:
        order = [i for i in live_ids if i in keep]
        present = set(order)
        reference = known + added
        for n, i in enumerate(reference):          # vanished meanwhile: back where it was
            if i in present:
                continue
            later = next((j for j in reference[n + 1:] if j in present), None)
            order.insert(order.index(later) if later is not None else len(order), i)
            present.add(i)
    else:
        order = [i for i in known if i != cur_id]
        for n, i in enumerate(live_ids):            # this step's new entries, where they went
            if i not in set(added) or i in order:
                continue
            before = next((j for j in reversed(live_ids[:n]) if j in order), None)
            order.insert(order.index(before) + 1 if before is not None else 0, i)
        order += [i for i in added if i not in order]   # added, and gone again already
    return View(current, [by_id[i] for i in order if i in by_id])


def describe_step(s: vp.Step) -> str:
    """One step in a few words ("remove the 2nd song in the queue")."""
    spec = OPS.get(s.op)
    by = f" by {s.artist or s.heard_artist}" if (s.artist or s.heard_artist) else ""
    copies = f" ×{s.count}" if s.count > 1 and s.op in ("add_song", "find_song", "add_album") else ""
    if s.op == "add_song":
        return f"play {s.title or s.heard_title}{by}{copies}"
    if s.op == "find_song":
        return f"find {s.value}{copies}"
    if s.op == "add_album":
        return f"play {s.tracks or 'all'} of {s.album}{by}{copies}"
    if s.op == "add_top":
        return f"play {s.count} of {s.artist}'s most popular songs"
    if s.op == "add_latest":
        return f"play {s.artist}'s newest " + ("song" if s.count == 1 else f"{s.count} songs")
    if s.op == "remove":
        return f"remove {ref_words(s.target)}"
    if s.op == "move":
        return f"move {ref_words(s.target)} to {dest_words(s.to)}"
    if s.op == "swap":
        return f"swap {ref_words(s.target)} and {ref_words(s.to)}"
    if s.op == "skipto":
        return f"jump to {ref_words(s.target)}"
    if s.op == "skip":
        return f"skip {s.count} songs" if s.count > 1 else "skip"
    if s.op == "volume":
        v = s.value.replace("%", "")
        return (f"set the volume to {v}%" if v.isdigit() else
                f"turn the volume {'up' if v.startswith('+') or v == 'up' else 'down'}"
                + (f" by {v.lstrip('+-')}%" if v.lstrip("+-").isdigit() else ""))
    if s.op == "loop":
        return {"queue": "loop the queue", "off": "turn the loop off"}.get(s.value, "loop the song")
    if s.op == "clear":
        return "clear the whole queue"
    if s.op == "answer":
        return "answer"
    return spec.say if spec else s.op


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def ref_words(ref: str) -> str:
    """A queue reference in words, for messages."""
    low = (ref or "").strip().lower()
    if low in ("current", "this"):
        return "the song playing now"
    if low in ("first", "next", "pos:1", "1"):
        return "the next song"
    m = re.fullmatch(r"(?:pos:)?(\d+)", low)
    if m:
        return f"the {_ordinal(int(m.group(1)))} song in the queue"
    if low == "last" or low == "last:1":
        return "the last song in the queue"
    m = re.fullmatch(r"last:(\d+)", low)
    if m:
        return f"the last {m.group(1)} songs"
    m = re.fullmatch(r"range:(\d+)-(\d+)", low)
    if m:
        return f"songs {m.group(1)}–{m.group(2)} of the queue"
    if low.startswith("step:"):
        return "the song(s) I just added"
    if low == "mine:last":
        return "the last song you added"
    if low.startswith("match:"):
        return f"“{ref.split(':', 1)[1]}”"
    if low.startswith("entry:"):
        return "that song"
    return f"“{ref}”"


def dest_words(to: str) -> str:
    low = (to or "").strip().lower()
    if low in ("front", "top", "", "pos:1", "1", "next"):
        return "the front of the queue"
    if low in ("end", "back", "last"):
        return "the end of the queue"
    m = re.fullmatch(r"(?:pos:)?(\d+)", low)
    if m:
        return f"position {m.group(1)}"
    for word in ("after:", "before:"):
        if low.startswith(word):
            return f"{word[:-1]} {ref_words(to[len(word):])}"
    return f"“{to}”"


def _title(e: dict | None) -> str:
    return vp.entry_label(e)


class PlanRun:
    """Runs one Plan for one person's request. `cog` is the VoiceDJ cog
    (cogs/voicedj.py): it supplies the checks (consent, session, staleness),
    the song and catalogue lookups, the player controls and the replies.
    `act` is the request's lane entry (its job, sequence number, queue epoch
    and cancellation flag)."""

    def __init__(self, cog, act, plan: vp.Plan, *, view: View | None = None,
                 mine: list[str] | None = None, rng: random.Random | None = None):
        self.cog = cog
        self.act = act
        self.plan = plan
        self.job = act.job
        self.gid = act.job.guild_id
        self.view = view if view is not None else View.capture(get_state(self.gid))
        self.mine = list(mine or [])
        self.rng = rng
        self.results: dict[int, StepResult] = {}
        self.step_entries: dict[int, list[str]] = {}
        #: The view after each step (step id → (current id, upcoming ids)),
        #: kept for tests and the debug log.
        self.trace: list[tuple[int, str | None, list[str | None]]] = []
        self.stopped: str = ""      # why the rest wasn't done, if it wasn't

    @property
    def state(self):
        return get_state(self.gid)

    # ── the run ───────────────────────────────────────────────────────────────

    async def run(self) -> list[StepResult]:
        cog, act = self.cog, self.act
        guild = cog.bot.get_guild(self.gid)
        member = guild.get_member(self.job.user_id) if guild else None
        epoch = act.queue_epoch
        steps = list(self.plan.steps)
        for n, step in enumerate(steps):
            if act.cancelled is not None:
                self._rest(steps[n:], "cancelled", "")         # the canceller said why
                break
            st = self.state
            if epoch is not None and st.queue_epoch != epoch:
                changed = ("the queue was cleared" if st.queue_change_reason == "cleared"
                           else "the music was stopped")
                self.stopped = f"{changed} meanwhile"
                self._rest(steps[n:], "not_done", "")
                break
            gone = cog._job_valid(self.job) if member is not None else "speaker_left"
            if gone:
                self.stopped = "" if gone == "revoked" else gone
                self._rest(steps[n:], "cancelled", "")
                break
            blocked = self._blocked(step, steps[:n])
            if blocked:
                result = StepResult(step, "skipped", why=blocked)
            else:
                try:
                    result = await self._run_step(step, guild, member)
                except Exception as e:                      # one step's bug never kills the rest
                    log.exception("voice plan step %s failed", step.label)
                    result = StepResult(step, "failed", f"❌ Couldn't {describe_step(step)} "
                                                        f"— something went wrong.",
                                        why=f"{type(e).__name__}: {e}"[:160])
            if act.cancelled is not None and result.status == "cancelled":
                self.results[step.id] = result
                self._rest(steps[n + 1:], "cancelled", "")
                break
            self.results[step.id] = result
            if result.added:
                self.step_entries[step.id] = list(result.added)
                self.mine.extend(result.added)
            if result.settle:
                await cog._settle_advance(self.gid)
            self.view = next_view(self.view, self.state, step, result)
            self.trace.append((step.id, qid(self.view.current), self.view.ids()))
            if step.op in ("stop", "clear") and result.status in OK:
                epoch = self.state.queue_epoch              # our own stop/clear moves it on
            # (anyone else's — even one landing while this step awaited a
            # lookup — stops the rest at the next step)
        return [self.results[s.id] for s in steps if s.id in self.results]

    def _rest(self, steps, status: str, message: str) -> None:
        for s in steps:
            if s.id not in self.results:
                self.results[s.id] = StepResult(s, status, message)

    def _blocked(self, step: vp.Step, before: list[vp.Step]) -> str:
        """Why `step` must not run (it would act on the wrong thing), or ""."""
        for k in sorted(vp.needs(step)):
            r = self.results.get(k)
            if r is None or not r.ok or not r.added:
                return (f"it acts on what step {k} adds, and that "
                        + ("didn't work" if r is not None and not r.ok else "added nothing"))
        failed_adds = [s for s in before if s.op in SONG_OPS
                       and s.id in self.results and not self.results[s.id].ok]
        if not failed_adds:
            return ""
        # After songs that didn't make it in, whatever acts on "the queue" or
        # "the song playing" would act on other songs than the ones meant
        # ("…, shuffle, skip the first one, loop the rest"). A step about one
        # particular queued song (by its id or name), the volume, a pause, a
        # question… doesn't depend on them and still happens.
        positional = bool(re.match(r"(?i)^(pos:|\d|last|first|next|range:|current|this)",
                                   step.target or ""))
        if step.op in _ACTS_ON_CONTENT or (step.op in QUEUE_REF_OPS and positional):
            return (f"{describe_step(failed_adds[-1])} didn't work, so it would have "
                    f"acted on other songs")
        return ""

    # ── one step ──────────────────────────────────────────────────────────────

    async def _run_step(self, step: vp.Step, guild, member) -> StepResult:
        op = step.op
        if op in SONG_OPS:
            return await self._song(step)
        if op in CONTROL_OPS or op == "shuffle":
            return await self._control(step)
        if op in QUEUE_REF_OPS:
            return self._queue_ref(step, guild)
        if op in WHOLE_QUEUE_OPS:
            return self._whole_queue(step)
        if op in INFO_OPS:
            return self._info(step)
        if op == "typed_only":
            return await self._typed_only(step, guild)
        spec = OPS.get(op)
        if spec is not None and spec.command:
            missing = self.cog._bridged_missing(spec, step)
            if missing:
                # Asked for, not run with a gap and not reported as done: the
                # answer comes back to the planner with this question.
                q = (_MISSING_ASK.get((op, missing)) or _MISSING_ASK.get(missing)
                     or f"To {spec.say}, what's the {missing}?")
                return StepResult(step, "failed", f"❓ {q}", why=f"no {missing} given", ask=q)
            status, message = await self.cog._run_bridged(guild, member, spec, step,
                                                          self.act.text)
            return StepResult(step, status, message)
        return StepResult(step, "failed", f"❌ I can't {describe_step(step)} by voice.")

    async def _through_cog(self, step: vp.Step, intent: Intent, **extra) -> StepResult:
        """A step the cog's own command path carries out (cogs/voicedj.py
        _dispatch_one → _execute_intent): the same checks, lookups and
        replies as a command from the grammar."""
        sub = self.cog._plan_sub(self.act, intent, step, **extra)
        self.act.current = sub                     # "stop" calls off its lookup too
        try:
            res = await self.cog._dispatch_one(sub, deliver=False)
        finally:
            self.act.current = None
        if res is None:
            return StepResult(step, "noop")
        return StepResult(step, res.status, res.message or "",
                          added=tuple(getattr(res, "added", ()) or ()),
                          why=getattr(res, "why", "") or "", settle=bool(sub.started_idle),
                          ask=getattr(res, "ask", "") or "")

    async def _song(self, step: vp.Step) -> StepResult:
        a = SONG_OPS[step.op]
        if a == "play":
            intent = Intent("play", step.title or step.heard_title, artist=step.artist or None,
                            heard=step.heard_title or None, when=step.when or None)
        elif a == "find":
            intent = Intent("find", step.value, artist=step.artist or None, when=step.when or None)
        elif a == "album":
            intent = Intent("album", step.album, artist=step.artist or None,
                            count=step.tracks or "all", when=step.when or None)
        else:
            intent = Intent(a, None, artist=step.artist or None, count=str(step.count),
                            when=step.when or None)
        copies = step.count if a in ("play", "find", "album") else 1
        return await self._through_cog(step, intent, copies=copies)

    async def _control(self, step: vp.Step) -> StepResult:
        st = self.state
        op = step.op
        if op == "skip":
            # "Skip this" means the song they heard: if it isn't the one
            # playing any more (it ended, someone skipped it), skipping now
            # would skip the next one instead — say so rather than do that.
            seen, live = qid(self.view.current), qid(st.current)
            if seen is not None and live is not None and seen != live:
                return StepResult(step, "conflict",
                                  f"↪️ Didn't skip — **{_title(self.view.current)}** already "
                                  f"ended, and **{_title(st.current)}** is playing now.")
        if op == "skip":
            intent = Intent("skip", count=str(step.count) if step.count > 1 else None)
        elif op == "volume":
            level = self._volume_level(step.value)
            if level is None:
                return StepResult(step, "failed", "❌ Didn't catch the volume.")
            intent = Intent("volume", level)
        elif op == "loop":
            mode = (step.value or "song").lower()
            mode = {"on": "song", "one": "song", "track": "song", "all": "queue",
                    "everything": "queue", "none": "off", "stop": "off"}.get(mode, mode)
            if mode not in ("song", "queue", "off"):
                mode = "song"
            intent = Intent("loop", mode)
        else:
            intent = Intent(op)
        return await self._through_cog(step, intent)

    def _volume_level(self, value: str) -> str | None:
        """The planner's volume ("50", "50%", "+10", "-20", "up", "down") as
        what the cog's volume control takes: "up"/"down" or a level."""
        v = (value or "").strip().lower().replace("%", "").replace(" ", "")
        if v in ("up", "down"):
            return v
        if re.fullmatch(r"[+-]\d+", v):
            now = round(self.state.volume * 100)
            level = now + int(v)
            return str(max(0, min(level, max(now, VOICE_VOLUME_MAX))))
        if v.isdigit():
            return v
        return None

    # ── the queue, by reference ────────────────────────────────────────────────

    def _resolve(self, ref: str) -> tuple[list[str], str]:
        """(occurrence ids `ref` means, "") or ([], why not). Resolved against
        the view, then checked against the live queue."""
        try:
            ids = vp.resolve_target(ref, self.view.upcoming, current=self.view.current,
                                    step_entries=self.step_entries, mine=self.mine)
        except vp.RefError as e:
            return [], str(e)
        live = set(self.state.queue.ids())
        cur = qid(self.state.current)
        out = []
        for i in ids:
            if i == "current":
                out.append(i)
                continue
            if i in live:
                out.append(i)
            elif i == cur:
                out.append("current")
            else:
                gone = next((e for e in self.view.upcoming if qid(e) == i), None)
                return [], (f"**{_title(gone)}** isn't in the queue any more" if gone
                            else "that song isn't in the queue any more")
        return out, ""

    def _queue_ref(self, step: vp.Step, guild) -> StepResult:
        st = self.state
        ids, why = self._resolve(step.target)
        if not ids:
            return StepResult(step, "conflict" if "any more" in why else "failed",
                              f"↪️ Didn't {describe_step(step)} — {why}.", why=why)
        by_id = {qid(e): e for e in st.queue}
        if step.op == "remove":
            if "current" in ids:
                playing = _title(st.current)
                ids = [i for i in ids if i != "current"]
                if not ids:
                    return StepResult(step, "conflict",
                                      f"↪️ Didn't remove **{playing}** — it's playing now "
                                      f"(say skip to skip it).")
            removed = queue_ops.remove_ids(st, ids)
            names = [_title(e) for e in removed]
            return StepResult(step, "done" if removed else "noop",
                              "🗑️ Removed " + _names(names) + " from the queue."
                              if removed else "Nothing to remove.",
                              removed=tuple(qid(e) for e in removed))
        if step.op == "skipto":
            if ids == ["current"]:
                return StepResult(step, "noop", f"**{_title(st.current)}** is already playing.")
            target = ids[0]
            dropped = queue_ops.drop_before(st, target)
            entry = st.queue[0]
            vc = guild.voice_client
            music = self.cog.bot.get_cog("Music")
            if not queue_ops.skip_current(st, vc, music, guild):
                if music is not None and vc is not None and not vc.is_playing() and not vc.is_paused():
                    music._kickstart(guild, vc)
            return StepResult(step, "done", f"⏭️ Jumping to **{_title(entry)}**"
                              + (f" (dropped {len(dropped)} before it)." if dropped else "."),
                              removed=tuple(qid(e) for e in dropped), moved=(target,),
                              settle=True)
        if step.op == "move":
            if "current" in ids:
                return StepResult(step, "noop", f"**{_title(st.current)}** is playing now — "
                                                "it's already ahead of the queue.")
            try:
                moved = self._move(st, ids, step.to)
            except (vp.RefError, queue_ops.QueueOpError) as e:
                return StepResult(step, "failed", f"↪️ Didn't {describe_step(step)} — {e}.")
            names = [_title(by_id.get(i)) for i in ids]
            where = dest_words(step.to)
            return StepResult(step, "done", f"↕️ Moved {_names(names)} to {where}.",
                              moved=tuple(moved))
        if step.op == "swap":
            other, why2 = self._resolve(step.to)
            if not other:
                return StepResult(step, "conflict" if "any more" in why2 else "failed",
                                  f"↪️ Didn't {describe_step(step)} — {why2}.")
            a, b = ids[0], other[0]
            if "current" in (a, b):
                return StepResult(step, "failed", "↪️ Can't swap the song playing now — "
                                                  "only songs in the queue.")
            try:
                ea, eb = queue_ops.swap_ids(st, a, b)
            except queue_ops.QueueOpError as e:
                return StepResult(step, "failed", f"↪️ Didn't swap them — {e}.")
            return StepResult(step, "done", f"🔄 Swapped **{_title(ea)}** and **{_title(eb)}**.",
                              moved=(a, b))
        return StepResult(step, "failed", f"❌ I can't {describe_step(step)}.")

    def _move(self, st, ids: list[str], to: str) -> list[str]:
        """Move these entries (in their queue order) as a block to `to`."""
        order = [i for i in st.queue.ids() if i in set(ids)]
        others = [e for e in st.queue if qid(e) not in set(order)]
        index = vp.resolve_destination(to, others, view=self.view.upcoming,
                                       current=self.view.current,
                                       step_entries=self.step_entries, mine=self.mine)
        for n, i in enumerate(order):
            queue_ops.move_id(st, i, index + n)
        return order

    def _whole_queue(self, step: vp.Step) -> StepResult:
        st = self.state
        if step.op == "reverse":
            if len(st.queue) < 2:
                return StepResult(step, "noop", "Nothing to reverse.")
            return StepResult(step, "done", f"🔃 Reversed the queue ({queue_ops.reverse(st)} songs).")
        if step.op == "balance":
            if not st.queue:
                return StepResult(step, "noop", "The queue is empty.")
            queue_ops.balance(st)
            return StepResult(step, "done", "⚖️ The queue now takes turns between requesters.")
        removed = queue_ops.dedupe(st)
        return StepResult(step, "done" if removed else "noop",
                          f"🧹 Removed {len(removed)} duplicate{'s' if len(removed) != 1 else ''}."
                          if removed else "✨ No duplicates in the queue.",
                          removed=tuple(qid(e) for e in removed))

    async def _typed_only(self, step: vp.Step, guild) -> StepResult:
        """Something that has to be typed: say how, instead of pretending."""
        from .voice_tools import typed_only_reason
        name = " ".join(step.value.lower().lstrip("-/!").split())
        why = typed_only_reason(name)
        prefix = await self.cog._prefix(guild)
        if why is None:
            return StepResult(step, "failed", f"❌ I don't have a `{name}` command.")
        return StepResult(step, "noop", f"⌨️ `{prefix}{name}` has to be typed — {why}.")

    # ── questions about what's going on ───────────────────────────────────────

    def _info(self, step: vp.Step) -> StepResult:
        st = self.state
        if step.op == "answer":
            return StepResult(step, "done", answer=step.value)
        if step.op == "now_playing":
            cur = st.current
            if cur is None:
                return StepResult(step, "done", answer="Nothing's playing right now.")
            who = cur.get("requester")
            return StepResult(step, "done", answer=f"This is {_spoken_title(cur)}"
                              + (f", requested by {who}." if who else "."))
        if step.op == "queue_info":
            items = list(st.queue)
            if not items:
                return StepResult(step, "done", answer="The queue is empty.")
            n = max(1, min(step.count if step.count > 1 else 5, 15))
            shown = ", ".join(_spoken_title(e) for e in items[:n])
            more = f", and {len(items) - n} more" if len(items) > n else ""
            return StepResult(step, "done", answer=f"{len(items)} song{'s' if len(items) != 1 else ''} "
                                                   f"up next: {shown}{more}.")
        hist = [t for t in reversed(st.history)][:5]
        if not hist:
            return StepResult(step, "done", answer="Nothing has played yet.")
        return StepResult(step, "done", answer="Recently played: "
                          + ", ".join(_spoken_title(t) for t in hist) + ".")


def _spoken_title(e: dict) -> str:
    t = (e.get("title") or "an unknown song").replace("—", "-").replace("–", "-")
    return re.sub(r"\s*[\(\[](official|lyric|audio|music video|visualizer)[^\)\]]*[\)\]]", "",
                  t, flags=re.IGNORECASE).strip()


def _names(names: list[str]) -> str:
    names = [f"**{n}**" for n in names]
    if len(names) <= 3:
        return " and ".join(names) if len(names) <= 2 else f"{names[0]}, {names[1]} and {names[2]}"
    return f"{', '.join(names[:3])} and {len(names) - 3} more"


def summarize(run: PlanRun) -> str:
    """What the request did, in one line, for the speaker's recent context
    (what the planner is shown next time): ids added/removed, songs named."""
    parts = []
    for s in run.plan.steps:
        r = run.results.get(s.id)
        if r is None:
            continue
        if r.added:
            entries = {qid(e): e for e in run.state.queue}
            cur = run.state.current
            if cur is not None:
                entries[qid(cur)] = cur
            named = [f"[{i}] {_title(entries[i])}" for i in r.added if i in entries][:6]
            parts.append(f"{describe_step(s)} → added " + (", ".join(named) or f"{len(r.added)} songs"))
        elif r.status in OK:
            parts.append(describe_step(s))
        else:
            parts.append(f"{describe_step(s)} ({r.status})")
    return "; ".join(parts)[:400]


def compose_reply(run: PlanRun) -> tuple[list[str], list[str], bool]:
    """(reply lines, spoken answers, everything worked) for a finished run.
    Lines follow the order the steps ran; then what wasn't done and why; then
    the plan's own problems and question."""
    lines: list[str] = []
    answers: list[str] = []
    skipped: dict[str, list[str]] = {}         # why → what wasn't done for that reason
    not_done: list[str] = []
    for s in run.plan.steps:
        r = run.results.get(s.id)
        if r is None:
            not_done.append(describe_step(s))
            continue
        if r.answer:
            answers.append(r.answer)
        if r.status == "skipped":
            skipped.setdefault(r.why, []).append(describe_step(s))
        elif r.status in ("not_done", "cancelled"):
            if r.status == "not_done" or run.stopped:
                not_done.append(describe_step(s))
        elif r.message:
            lines.append(r.message)
        if s.took_skip and not r.ok and r.status not in ("not_done", "cancelled"):
            skipped.setdefault("", []).append("skip this song")   # "skip this and play X": no X, no skip
    for why, what in skipped.items():
        lines.append("↪️ So I didn't " + ", ".join(what) + (f" — {why}" if why else "") + ".")
    if not_done:
        why = f" — {run.stopped}" if run.stopped else ""
        lines.append("✖ Not done" + why + ": " + ", ".join(not_done) + ".")
    for p in run.plan.problems:
        lines.append(f"⚠️ {p[:1].upper()}{p[1:]}.")
    if run.plan.ask:
        lines.append(f"❓ {run.plan.ask}")
    ok = (all(r.ok for r in run.results.values()) and not skipped and not not_done
          and not run.plan.problems)
    return lines, answers, ok


__all__ = ["PlanRun", "StepResult", "View", "compose_reply", "describe_step", "summarize",
           "ref_words", "dest_words"]
