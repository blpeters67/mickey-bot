"""
Voice-command pipeline plumbing — everything between "a segment of speech"
and "an action", minus Discord. Unit-tested in tests/test_voice_pipeline.py.

  * :class:`JobQueue` — the speech-to-text backlog. Bounded per speaker and
    overall; confirmations and "armed" follow-ups jump the queue; every drop
    is counted with a reason. (The old queue held 4 jobs, dropped the rest
    silently, and served them strictly in order.)
  * :class:`AsrRunner` — one inference at a time on one dedicated thread. A
    timed-out inference can't be killed (Python can't cancel a running
    thread), so the runner refuses new work while one is still running rather
    than stacking more CPU-bound calls behind it.
  * :class:`ModelManager` — a single, shared load: concurrent callers wait on
    the same task, a caller timing out doesn't cancel the load, and the
    result is kept. (Previously a load timeout cancelled the waiter before the
    model was stored, so the next utterance started a second load.) Switches
    are serialized and commit atomically.
  * :class:`VoiceMetrics` — counters, drop reasons, timings and a short event
    log per guild, for `voice diag`.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .logging_setup import get_logger

log = get_logger(__name__)

PRIORITY_CONFIRM = 0      # someone owes a yes/no
PRIORITY_ARMED = 1        # follows a wake-only utterance
PRIORITY_NORMAL = 2


@dataclass(eq=False)
class Job:
    """One segment of speech waiting for (or going through) recognition."""
    guild_id: int
    user_id: int
    audio: Any                 # float32 mono 16 kHz
    start: float               # capture time (monotonic) of the first sample
    end: float                 # …of the last speech sample
    speech_ms: int
    session: int               # guild's voice session when captured
    epoch: int                 # speaker's consent epoch when captured
    priority: int = PRIORITY_NORMAL
    channel_id: int | None = None
    overlap_ms: int = 0
    segment_reason: str = ""
    created: float = field(default_factory=time.monotonic)
    #: When speech actually began (VAD onset); `start` includes pre-roll.
    speech_start: float | None = None
    #: One utterance the front end split at its length cap comes as several
    #: jobs: the same `utt`, `part` 0, 1, 2… (the cog joins their words).
    utt: int = 0
    part: int = 0
    #: The whole of a long utterance, its parts' audio joined (overlaps taken
    #: out): transcribed once more, as one, before it is understood.
    whole: bool = False

    @property
    def key(self) -> tuple[int, int]:
        return (self.guild_id, self.user_id)

    @property
    def onset(self) -> float:
        return self.start if self.speech_start is None else self.speech_start


class JobQueue:
    """Priority queue with admission limits.

    Order: priority, then oldest capture — and the parts of one long
    utterance in their order. Limits: `per_speaker` waiting jobs per person
    (a new one supersedes their oldest — they've moved on; the parts of one
    utterance count once), and `total` overall (the oldest lowest-priority
    job goes). `on_drop(job, reason)` is told about everything that's
    discarded."""

    def __init__(self, *, per_speaker: int = 2, total: int = 8,
                 on_drop: Callable[[Job, str], None] | None = None):
        self.per_speaker = per_speaker
        self.total = total
        self._on_drop = on_drop or (lambda job, reason: None)
        self._jobs: list[Job] = []
        self._event = asyncio.Event()

    def __len__(self) -> int:
        return len(self._jobs)

    @staticmethod
    def _one_utterance(a: Job, b: Job) -> bool:
        """Parts of the same long utterance (cut at the length cap): one thing
        said, not someone moving on."""
        return (bool(a.utt) and a.utt == b.utt and a.key == b.key
                and not a.whole and not b.whole)

    def put(self, job: Job) -> None:
        same = [j for j in self._jobs if self._one_utterance(j, job)]
        if job.part and same:
            # The start of a long utterance is as urgent as its rest (and
            # must be heard before it): never left behind at a lower priority.
            for j in same:
                j.priority = min(j.priority, job.priority)
        mine = [j for j in self._jobs if j.key == job.key and not self._one_utterance(j, job)]
        while len(mine) >= self.per_speaker:
            victim = min(mine, key=lambda j: (-j.priority, j.end))
            if victim.priority < job.priority:
                self._on_drop(job, "speaker_backlog")
                return
            self._jobs.remove(victim)
            mine.remove(victim)
            self._on_drop(victim, "superseded")
        if len(self._jobs) >= self.total:
            victim = max(self._jobs, key=lambda j: (j.priority, -j.end))
            if victim.priority < job.priority:
                self._on_drop(job, "queue_full")
                return
            self._jobs.remove(victim)
            self._on_drop(victim, "queue_full")
        self._jobs.append(job)
        self._event.set()

    def _pop_best(self) -> Job | None:
        if not self._jobs:
            return None
        best = min(self._jobs, key=lambda j: (j.priority, j.end))
        earlier = [j for j in self._jobs
                   if self._one_utterance(j, best) and j.part < best.part]
        if earlier:
            best = min(earlier, key=lambda j: j.part)      # an utterance's parts in order
        self._jobs.remove(best)
        return best

    async def get(self) -> Job:
        while True:
            job = self._pop_best()
            if job is not None:
                return job
            self._event.clear()
            await self._event.wait()

    def snapshot(self) -> list[Job]:
        """The jobs waiting right now (a copy)."""
        return list(self._jobs)

    def purge(self, predicate: Callable[[Job], bool], reason: str) -> int:
        doomed = [j for j in self._jobs if predicate(j)]
        for j in doomed:
            self._jobs.remove(j)
            self._on_drop(j, reason)
        return len(doomed)

    def oldest_age(self, now: float | None = None) -> float:
        if not self._jobs:
            return 0.0
        now = time.monotonic() if now is None else now
        return now - min(j.end for j in self._jobs)


class AsrBusy(Exception):
    """The previous (timed-out) inference is still running."""


class AsrTimeout(Exception):
    """This inference exceeded its time limit (it keeps running in its thread)."""


def _wrap(fut: concurrent.futures.Future) -> asyncio.Future:
    """asyncio view of a thread future whose outcome is always "retrieved", so
    an abandoned (timed-out) call that later fails doesn't log noise."""
    afut = asyncio.wrap_future(fut)
    afut.add_done_callback(lambda f: f.cancelled() or f.exception())
    return afut


class AsrRunner:
    """Run blocking inference on one dedicated thread, one call at a time."""

    def __init__(self, *, timeout: float = 30.0, busy_wait: float = 2.0):
        self.timeout = timeout
        self.busy_wait = busy_wait
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="voice-asr")
        self._inflight: concurrent.futures.Future | None = None
        self.busy_since: float | None = None
        self.timeouts = 0
        self.last_seconds: float | None = None
        self._durations: deque = deque(maxlen=50)

    @property
    def stuck_for(self) -> float:
        """Seconds the current call has been running past its timeout, or 0."""
        if self._inflight is None or self._inflight.done() or self.busy_since is None:
            return 0.0
        return max(0.0, time.monotonic() - self.busy_since - self.timeout)

    def median_seconds(self) -> float | None:
        if not self._durations:
            return None
        d = sorted(self._durations)
        return d[len(d) // 2]

    async def run(self, fn: Callable, *args):
        if self._inflight is not None and not self._inflight.done():
            # A timed-out call is still burning the CPU. Wait briefly; if it's
            # still going, refuse rather than queue more work behind it.
            try:
                await asyncio.wait_for(asyncio.shield(_wrap(self._inflight)), self.busy_wait)
            except Exception:
                pass
            if not self._inflight.done():
                raise AsrBusy()
        started = time.monotonic()
        self.busy_since = started
        fut = self._pool.submit(fn, *args)
        self._inflight = fut
        try:
            result = await asyncio.wait_for(asyncio.shield(_wrap(fut)), self.timeout)
        except asyncio.TimeoutError:
            self.timeouts += 1
            raise AsrTimeout() from None
        self.last_seconds = time.monotonic() - started
        self._durations.append(self.last_seconds)
        self.busy_since = None
        return result

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class ModelManager:
    """Loads one speech model at a time and keeps it.

    ``loader(name)`` blocks (it runs on a private thread); ``warmup(model)``
    optionally runs once after loading so the first real command doesn't pay
    first-inference costs."""

    def __init__(self, loader: Callable[[str], Any], name: str, *,
                 warmup: Callable[[Any], None] | None = None,
                 retry_after: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self._loader = loader
        self._warmup = warmup
        self.name = name
        self.model = None
        self.state = "unloaded"        # unloaded | loading | ready | failed
        self.error: str | None = None
        self.loaded_at: float | None = None
        self.load_seconds: float | None = None
        self._retry_after = retry_after
        self._clock = clock
        self._failed_at: float | None = None
        self._task: asyncio.Task | None = None
        self._switch_lock = asyncio.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="voice-model")

    def _load_blocking(self, name: str):
        model = self._loader(name)
        if self._warmup is not None:
            try:
                self._warmup(model)
            except Exception:
                log.warning("voice model warm-up failed (continuing)", exc_info=True)
        return model

    async def _load(self, name: str) -> None:
        self.state = "loading"
        started = self._clock()
        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(self._pool, self._load_blocking, name)
        except Exception as e:
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"[:300]
            self._failed_at = self._clock()
            log.error("Couldn't load voice model %r: %s", name, self.error)
            return
        if name != self.name:            # a switch happened meanwhile; ours is stale
            return
        self.model, self.state, self.error = model, "ready", None
        self.loaded_at = self._clock()
        self.load_seconds = self.loaded_at - started
        log.info("Voice model %r ready (%.1fs).", name, self.load_seconds)

    def prefetch(self) -> None:
        """Start loading in the background (no-op if loaded or loading)."""
        if self.model is not None or (self._task is not None and not self._task.done()):
            return
        if (self.state == "failed" and self._failed_at is not None
                and self._clock() - self._failed_at < self._retry_after):
            return
        self._task = asyncio.create_task(self._load(self.name))

    async def get(self, timeout: float):
        """The model, waiting up to `timeout` for a load in progress. Timing out
        does NOT cancel the load — it finishes and is kept."""
        if self.model is not None:
            return self.model
        self.prefetch()
        if self._task is None:
            return None
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except asyncio.TimeoutError:
            pass
        return self.model

    async def switch(self, name: str) -> None:
        """Load `name` and swap it in. Serialized; the current model keeps
        serving until the new one is ready; raises (and changes nothing) on
        failure."""
        async with self._switch_lock:
            if name == self.name and self.model is not None:
                return
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(self._pool, self._load_blocking, name)
            self.model, self.name, self.state, self.error = model, name, "ready", None
            self.loaded_at = self._clock()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


#: The life of a recognised command, in order. A command that was heard but
#: failed must never be counted as done: each stage is counted separately.
ACTION_STAGES = ("recognized", "admitted", "started", "completed", "noop",
                 "failed", "cancelled", "stale")


@dataclass
class GuildMetrics:
    counts: Counter = field(default_factory=Counter)
    drops: Counter = field(default_factory=Counter)
    parse: Counter = field(default_factory=Counter)
    events: deque = field(default_factory=lambda: deque(maxlen=12))
    asr_seconds: deque = field(default_factory=lambda: deque(maxlen=50))
    latency_seconds: deque = field(default_factory=lambda: deque(maxlen=50))
    #: End of speech → the first audio of a voice-requested song (complete
    #: response latency for "play", unlike `latency_seconds`).
    audible_seconds: deque = field(default_factory=lambda: deque(maxlen=50))
    #: Whisper's own quality signals per transcribed clip: (avg_logprob,
    #: no_speech_prob, audio_seconds, outcome). Diagnostics only — these are
    #: not calibrated probabilities of being right.
    asr_quality: deque = field(default_factory=lambda: deque(maxlen=50))
    #: action → Counter(stage) — see ACTION_STAGES.
    actions: dict = field(default_factory=dict)
    last_error: str | None = None
    #: core/voice_mix.CueTiming of the last wake-word cues (filled in as
    #: each one plays: its first audible frame is stamped on the audio thread).
    cue_timings: deque = field(default_factory=lambda: deque(maxlen=50))


def _median(xs) -> float | None:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _p95(xs) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else None


def percentile(xs, q: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def cue_summary(timings) -> dict:
    """Median / p90 / p95 (ms) of each stage of the wake cues that played,
    by the instant wake word's own clock: from the detector's frame to the
    cue's first audible frame leaving the mixer — and the stages between."""
    out = {}
    done = [t for t in timings if t.first_frame is not None and t.detected is not None]
    for name, a, b in (("total", "detected", "first_frame"), ("to_bot", "detected", "received"),
                       ("check", "verify_start", "verify_end"),
                       ("ask_to_mixer", "scheduled", "attached"),
                       ("mixer_to_frame", "attached", "first_frame")):
        xs = [x for x in (t.ms(a, b) for t in done) if x is not None]
        if xs:
            out[name] = (percentile(xs, 0.5), percentile(xs, 0.9), percentile(xs, 0.95), len(xs))
    out["tiers"] = Counter(t.tier or "?" for t in done)
    return out


class VoiceMetrics:
    """Reason codes and numbers only — transcripts are never stored here."""

    def __init__(self):
        self._g: dict[int, GuildMetrics] = {}

    def guild(self, guild_id: int) -> GuildMetrics:
        return self._g.setdefault(guild_id, GuildMetrics())

    def count(self, guild_id: int, name: str, n: int = 1) -> None:
        self.guild(guild_id).counts[name] += n

    def drop(self, guild_id: int, reason: str) -> None:
        self.guild(guild_id).drops[reason] += 1

    def event(self, guild_id: int, user_id: int, what: str) -> None:
        self.guild(guild_id).events.append((time.time(), user_id, what))

    def stage(self, guild_id: int, action: str, stage: str) -> None:
        """Count one step of a command's life (see ACTION_STAGES)."""
        self.guild(guild_id).actions.setdefault(action, Counter())[stage] += 1

    def asr(self, guild_id: int, result, outcome: str) -> None:
        """Keep whisper's quality numbers with what the clip turned out to be."""
        try:
            self.guild(guild_id).asr_quality.append(
                (float(result.avg_logprob), float(result.no_speech_prob),
                 float(result.audio_seconds), outcome))
        except (AttributeError, TypeError, ValueError):
            pass

    def summary(self, guild_id: int) -> dict:
        g = self.guild(guild_id)
        q = list(g.asr_quality)
        commands = [r for r in q if r[3].startswith("→")]
        other = [r for r in q if not r[3].startswith("→")]
        return {
            "counts": dict(g.counts), "drops": dict(g.drops), "parse": dict(g.parse),
            "asr_median": _median(g.asr_seconds), "asr_p95": _p95(g.asr_seconds),
            "latency_median": _median(g.latency_seconds),
            "latency_p95": _p95(g.latency_seconds),
            "audible_median": _median(g.audible_seconds),
            "audible_p95": _p95(g.audible_seconds),
            "actions": {a: dict(c) for a, c in g.actions.items()},
            "quality_commands": (_median(r[0] for r in commands),
                                 _median(r[1] for r in commands), len(commands)),
            "quality_other": (_median(r[0] for r in other),
                              _median(r[1] for r in other), len(other)),
            "quality_recent": q[-5:],
            "events": list(g.events), "last_error": g.last_error,
        }
