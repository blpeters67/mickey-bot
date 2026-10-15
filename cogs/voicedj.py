"""
Hands-free voice control for the music player — the "talk to the DJ" feature.

Say a wake word in the voice channel ("hey DJ, skip") and the bot acts on it.

Privacy gates (unchanged): an admin turns it on per server (`voice enable`),
and each member opts in (`voice optin`). Only opted-in members' audio is ever
processed — and only audio that arrives *after* they're allowed in: the
listener taps live frames from the recorder instead of reading its 5-minute
clip buffer, so turning it on can't sweep up earlier conversation.

Pipeline (2026-10 rewrite, from the voice review):

  recorder tap ─▶ VoiceFrontEnd (thread) ─▶ JobQueue ─▶ AsrRunner (thread)
  (decrypted Opus,   decode+FEC/PLC,          priority,     one inference at
   RTP seq/ts)       anti-aliased 16 kHz,     admission,    a time, can't
                     Silero VAD, segments     expiry        stack on timeout
                                                                │
       per-guild dispatch ◀── parse (exact grammar, yes/no, wake windows)
       (ordered, bounded song lookup, rechecks consent/session/channel)

Every job carries its capture time, speech onset, session and consent epoch
— stamped on the packets when they were admitted, never "whatever is current
when the segment reaches the event loop" — and is re-checked before
transcription and again before it acts: opting out, disabling, the bot
moving/reconnecting, the speaker leaving, or the speech simply being too old
all stop it. Drops are counted by reason (`voice diag`); transcripts are only
logged while `voice debug` is switched on.

Actions run in two lanes per server: playback controls (skip/pause/resume/
volume/loop/restart/stop/clear) run immediately in the order heard; song
requests queue behind each other for their lookup. A slow search can no longer
make a "skip" behind it go stale, and "stop"/"clear" cancel the song requests
heard before them, so queue edits still happen in the order they were said.

Smart tier (2026-10, optional): song requests, and whatever the grammar can't
handle after the wake word — free phrasing, several commands at once, a
question — go as text (never audio) to an LLM via OpenRouter
(core/voice_llm.py), which works out what was meant: the actual song, an
album's tracks or an artist's popular/newest songs (looked up in a music
catalogue, core/music_catalog.py), controls. Its commands pass the same checks
as the grammar's (consent, staleness, yes/no for stop/clear/very loud); a
request with songs and other steps runs as one plan, strictly in the order
said. Questions are answered out loud with TTS and posted.
Off without OPENROUTER_API_KEY, or per server with `voice ai off`. While a
request is being worked on, the reply channel shows "<bot> is typing…" —
started early by a quick look at the first ~2 s of a long utterance for the
wake word (only when the recogniser is idle).

Core modules: core/voice_frontend.py (audio), core/voice_pipeline.py (queue,
models, metrics), core/voice_intent.py (grammar), core/voice_llm.py (LLM
tier), core/whisper.py (ASR call).
"""
from __future__ import annotations

import asyncio
import io
import random
import re
import time
import wave
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional
from urllib.parse import quote_plus

import discord
from discord.ext import commands, tasks

from core import audio, db, music_catalog, queue_ops, voice_mix
from core import voice_exec as vx
from core import voice_plan as vp
from core.config import settings
from core.formatters import chunk_lines
from core.logging_setup import get_logger
from core.state import clear_queue, get_state, qid, requeue_copy, reset_state
from core.voice_intent import (
    SONG_ACTIONS, VOICE_VOLUME_MAX, WAKE_WINDOW_SECONDS, Intent, ParseResult, interpret,
    normalize_wake, parse_confirmation, parse_detailed, text_after_wake,
)
from core.voice_llm import LLMRouter, plan_context
from core.voice_tools import OPS
from core.voice_pipeline import (
    ACTION_STAGES, PRIORITY_ARMED, PRIORITY_CONFIRM, PRIORITY_NORMAL, AsrBusy,
    AsrRunner, AsrTimeout, Job, JobQueue, ModelManager, VoiceMetrics, cue_summary,
)
from core.voice_mix import CueTiming
from core.whisper import KNOWN_MODELS, AsrOptions, load_model, transcribe_command

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tunables (seconds unless noted)
# ──────────────────────────────────────────────────────────────────────────────

_MIN_COMMAND_SPEECH_MS = 350   # with no window open, shorter can't be "hey DJ <cmd>"
_ASR_MAX_AGE = 8.0             # don't start transcribing a normal command older than this
_DISPATCH_MAX_AGE = 20.0       # never act on speech older than this
_CONFIRM_SECONDS = 12.0        # a yes/no reply must START within this (after the question is out)
_ARM_SECONDS = WAKE_WINDOW_SECONDS   # "hey DJ" … (pause) … "skip": must start within this
_CONTEXT_GRACE = 20.0          # …but a reply captured in time may take this long to recognise
_WINDOW_END_SILENCE_MS = 450   # quicker end-of-speech while a window is open
_ECHO_TAIL = 0.8               # bot speech can echo back this long after it ends
_MODEL_WAIT = 5.0              # a job waits this long for a loading model
_PLAY_LOOKUP_TIMEOUT = 25.0    # bound on the song search for "play …"
_FIND_LOOKUP_TIMEOUT = 40.0    # …and for a described song (AI pick, web lookup, then search)
_CATALOG_TIMEOUT = 15.0        # bound on an album / top / newest-songs catalogue lookup
_SETTLE_SECONDS = 2.0          # a plan waits this long for an idle player to pick its first song
_DEDUPE_SECONDS = 3.0          # same command twice from a split segment = once
_TRANSCRIBE_TIMEOUT = 30.0
_REFRESH_EVERY = 30.0          # safety-net re-read of settings (changes apply instantly)
_TYPING_MAX = 25.0             # "typing…" never outlives this, answer or not
_TYPING_EVERY = 8.0            # Discord shows one typing ping for ~10 s
_PROBE_SECONDS = 2.2           # audio given to the early wake check (incl. pre-roll)
_PROBE_MAX_ASR = 1.5           # median recognition slower than this: no early checks

#: app_settings key prefix — the voice channel the bot was connected to, per
#: guild. Written on every join/move/leave. Memory cannot survive os.execv or a
#: crash, so this is how a restart finds its way back to where people already
#: are (see _rejoin_voice).
_VOICE_CHANNEL_KEY = "voice_channel:"

#: Two reports of the same utterance (the early wake probe and the finished
#: transcript) carry the same VAD onset, so anything within this window is the
#: same "hey DJ" and gets one cue, not two.
_CUE_SAME_UTTERANCE = 1.0

#: The text posted in the reply channel when the wake word is heard.
_CUE_MESSAGE = ":ear:*Now listening...*"

#: A fire still waiting for its check after this long isn't cued from the
#: fire any more (the transcript's cue backs it up).
_WAKE_CHECK_STALE = 1.5
#: The check heard "hey…" only: hand the utterance over again this much later.
_WAKE_RECHECK_AFTER = 0.3


def _wake_cut_short(heard: str, wakes) -> bool:
    """Did the check hear just the start of a wake phrase ("hey", "hey d")?"""
    words = normalize_wake(heard or "") or ""
    if not words:
        return False
    for w in wakes:
        full = normalize_wake(w) or ""
        if full and full != words and full.startswith(words):
            return True
    return False

# How the bot acknowledges a recognised command. Stored per-user (NULL → guild
# default). Keep these keys in sync with the DB and the help text.
FEEDBACK_MODES: dict[str, str] = {
    "text":   "Reply with a text message (e.g. “🎙️ heard ‘skip’ → skipping”).",
    "voice":  "Say the result out loud in the voice channel (needs TTS on the host).",
    "silent": "Just do it — no acknowledgement (questions are still asked in text).",
    "chime":  "Text reply plus a short confirmation sound.",
}


# Tier 2 — the LLM (core/voice_llm.py) — is asked about song requests and
# whatever the grammar couldn't handle after a wake word; see _wants_llm.


@dataclass(eq=False)
class _Typing:
    """"<bot> is typing…" for one person's request, in the reply channel.
    Runs until the reply is posted or `until` (later work can extend it)."""
    until: float
    task: asyncio.Task | None = None


@dataclass(eq=False)
class _Pending:
    """An open "are you sure?" for one person.

    Two deadlines, deliberately separate. CAPTURE: a reply counts only if the
    person started speaking after we began asking and by `deadline` — judged
    by the speech onset (VAD), not the padded clip start, and never extended
    for speech captured late. PROCESSING: a reply captured in time keeps the
    question alive while it waits for / goes through recognition, up to
    `process_by` — a slow CPU used to lose a timely "yes"."""
    intent: Intent
    asked_at: float          # we began delivering the question
    deadline: float          # replies must start by this (re-anchored once delivered)
    session: int
    then: tuple = ()         # the rest of a multi-part request, run in order on "yes"
    plan: Optional["vp.Plan"] = None     # …or the whole request as a plan
    view: Optional["vx.View"] = None     # the queue as the request saw it

    @property
    def confirm_action(self) -> str:
        """The action word a "yes" may repeat ("yes, clear it")."""
        if self.plan is not None:
            for st in self.plan.steps:
                if st.op in ("clear", "stop") or (
                        st.op == "volume" and st.value.replace("%", "").strip().isdigit()
                        and int(st.value.replace("%", "").strip()) > VOICE_VOLUME_MAX):
                    return st.op
            return "plan"
        return next((i.action for i in (self.intent, *self.then) if i.needs_confirmation),
                    self.intent.action)

    def accepts(self, job: Job) -> bool:
        return self.accepts_onset(job.onset)

    def accepts_onset(self, onset: float) -> bool:
        return self.asked_at <= onset <= self.deadline

    @property
    def process_by(self) -> float:
        return self.deadline + _CONTEXT_GRACE


@dataclass(eq=False)
class _Armed:
    """"hey DJ" said on its own: the next utterance is the command. Same two
    deadlines as _Pending: the command must START by `until`; a command
    captured in time survives slow recognition up to `process_by`."""
    since: float             # end of the wake-only utterance
    until: float
    session: int
    #: We asked them something ("Kaimara by who?"): what they say next, wake
    #: word or not, is the answer — the planner is told the question.
    question: str = ""

    def accepts(self, job: Job) -> bool:
        return self.accepts_onset(job.onset)

    def accepts_onset(self, onset: float) -> bool:
        return self.since - 0.1 <= onset <= self.until

    @property
    def process_by(self) -> float:
        return self.until + _CONTEXT_GRACE


@dataclass(eq=False)
class _Parts:
    """The first parts of one long utterance that is addressed to us (it
    opened with the wake word, or answers a question), waiting for the rest:
    their transcripts are joined (core/voice_intent.stitch takes the words
    the parts' overlap repeats once) and the whole is one request."""
    utt: int
    jobs: list
    texts: list
    since: float


#: Most parts one utterance may have (~2 minutes of speech): past this the
#: request is taken as it is so far, and that is said.
_MAX_PARTS = 12
#: A part that should be followed by another (it was cut at the length cap)
#: but isn't, within this long: what came is taken, and that is said.
_PARTS_WAIT = 14.0


@dataclass(frozen=True)
class ActionResult:
    """What executing a command did. `status` drives the metrics and the
    chime — a returned "❌ …" string is never counted as success again.
    done: it took effect · noop: nothing to do · failed: tried, couldn't ·
    cancelled: invalidated before it could act (message may be None)."""
    status: str
    message: Optional[str] = None
    #: Occurrence ids (core/state.py) of the entries it put in the queue.
    added: tuple = ()
    #: Sanitized reasoning (which song was accepted or rejected, and why).
    why: str = ""
    #: A question for the speaker (what it couldn't resolve); their next
    #: words answer it.
    ask: str = ""


_STATUS_STAGE = {"done": "completed", "noop": "noop", "failed": "failed",
                 "cancelled": "cancelled", "conflict": "conflict", "skipped": "skipped",
                 "not_done": "cancelled"}


@dataclass(frozen=True)
class _SongFound:
    """What looking a song up came to. status: matched (catalogue song and
    its upload) · upload_only (not in the catalogue; a checked search found
    it) · ambiguous (several fit, or only another artist's) · not_found ·
    catalog_down · no_upload (known song, no upload that is it) · incomplete
    · error. `why` is the sanitized evidence, for the log."""
    entry: Optional[dict]
    known: object
    failure: Optional[str]
    why: str
    status: str
    ask: str = ""


def _fresh_copy(track: dict) -> dict:
    """Another queue entry for the same song ("queue it five times"): the
    song, not this entry — no id (it gets its own), no playback state."""
    return {k: v for k, v in track.items()
            if not k.startswith("_") or k in ("_needs_resolve", "_search_query", "_http_headers")}


@dataclass(eq=False)
class _Action:
    """A recognised command waiting for (or going through) execution."""
    job: Job
    intent: Intent
    text: str
    seq: int                                  # order heard, per cog
    cancelled: Optional[str] = None           # why it was called off
    task: Optional[asyncio.Future] = None     # in-flight song lookup
    enqueued: float = 0.0                     # when it was recognised and queued
    queue_epoch: int | None = None             # admission, not the later lookup start
    steps: tuple = ()                         # a plan: one request's actions, run in order
    current: Optional["_Action"] = None       # the plan's step being run right now
    in_plan: bool = False                     # a step of a plan (aged from the plan's start)
    started_idle: bool = False                # its songs got an idle player going
    parent: Optional["_Action"] = None        # the plan this step belongs to
    front_last: Optional[dict] = None         # last song this request put at the front
    now_done: bool = False                    # this request already made way for a "now" song
    plan: Optional["vp.Plan"] = None          # a planned request (core/voice_exec.py runs it)
    view: Optional["vx.View"] = None          # …and the queue it was planned against
    step: Optional["vp.Step"] = None          # the plan step this action carries out
    copies: int = 1                           # separate queue entries to add (songs)


_WHY_INVALID = {
    "session_changed": "I moved or reconnected meanwhile",
    "disconnected": "I'm not in a voice channel any more",
    "speaker_left": "you left the voice channel",
}


# ── typed commands run by voice ───────────────────────────────────────────────

class _VoiceMessage:
    """What a typed command expects of ``ctx.message``, for one run by voice:
    the speaker as author, the voice reply channel, no attachments, nothing
    replied to. Replies and reactions go to the channel (there is no message
    to react to)."""

    def __init__(self, guild, author, channel, content: str):
        self.guild, self.author, self.channel = guild, author, channel
        self.content = self.clean_content = content
        self.attachments, self.embeds, self.stickers = [], [], []
        self.mentions, self.role_mentions, self.channel_mentions = [], [], []
        self.raw_mentions, self.raw_role_mentions, self.raw_channel_mentions = [], [], []
        self.mention_everyone = False
        self.reference = None
        self.interaction = None
        self.webhook_id = None
        self.type = discord.MessageType.default
        self.id = discord.utils.time_snowflake(discord.utils.utcnow())
        self.created_at = discord.utils.utcnow()
        self._state = getattr(channel, "_state", None)

    async def add_reaction(self, *_a, **_kw) -> None:
        return None

    async def remove_reaction(self, *_a, **_kw) -> None:
        return None

    async def reply(self, content=None, **kw):
        kw.pop("mention_author", None)
        kw.pop("reference", None)
        return await self.channel.send(content, **kw)

    async def delete(self, *_a, **_kw) -> None:
        return None

    @property
    def jump_url(self) -> str:
        return getattr(self.channel, "jump_url", "")


class _NullTyping:
    def __init__(self, coro=None):
        self._coro = coro

    async def __aenter__(self):
        if self._coro is not None:
            try:
                await self._coro
            except Exception:
                pass
        return self

    async def __aexit__(self, *exc):
        return False


class _VoiceCommandContext(commands.Context):
    """A command context for a command run by voice. Sends go straight to
    the reply channel (there's no interaction or message to answer)."""

    async def send(self, content=None, **kwargs):
        for k in ("ephemeral", "reference", "mention_author", "suppress_embeds", "silent"):
            kwargs.pop(k, None)
        return await self.channel.send(content, **kwargs)

    async def reply(self, content=None, **kwargs):
        return await self.send(content, **kwargs)

    def typing(self, *, ephemeral: bool = False):
        t = getattr(self.channel, "typing", None)
        if t is None:
            return _NullTyping()
        got = t()
        return got if hasattr(got, "__aenter__") else _NullTyping(got)

    async def defer(self, *, ephemeral: bool = False) -> None:
        return None


def _voice_context(bot, guild, member, channel, cmd, prefix: str, heard: str):
    from discord.ext.commands.view import StringView
    msg = _VoiceMessage(guild, member, channel, f"{prefix}{cmd.qualified_name}")
    ctx = _VoiceCommandContext(message=msg, bot=bot, view=StringView(""), prefix=prefix,
                               command=cmd, invoked_with=cmd.name)
    ctx.voice_heard = heard              # for a command that wants to know it was voice
    return ctx


def _bridged_args(cmd, spec, step) -> tuple[dict, str]:
    """The command's arguments from a plan step, by the registry's mapping
    (parameter ← step field), converted to what the parameter is declared as
    (int / bool / text). ({kwargs}, "") or ({}, what's wrong)."""
    import typing
    kwargs = {}
    for pname, param in cmd.clean_params.items():
        field_name = spec.params.get(pname)
        if field_name is None:
            if param.required:
                return {}, f"it needs a {pname}, which voice can't give"
            continue
        raw = getattr(step, field_name, "")
        if isinstance(raw, int) and field_name == "count":
            raw = "" if (raw == 1 and not param.required and pname != "reminder_id") else str(raw)
        raw = " ".join(str(raw or "").split())
        ann = param.annotation
        origin = typing.get_origin(ann)
        if origin is typing.Union or str(origin) == "types.UnionType":
            inner = [a for a in typing.get_args(ann) if a is not type(None)]
            ann = inner[0] if len(inner) == 1 else str
        if not raw:
            if param.required:
                return {}, f"say the {pname.replace('_', ' ')}"
            continue
        if ann is int:
            try:
                kwargs[pname] = int(raw)
            except ValueError:
                return {}, f"“{raw}” isn't a number"
        elif ann is bool:
            kwargs[pname] = raw.lower() in ("on", "yes", "true", "enable", "1")
        elif ann in (str, inspect_empty()) or isinstance(ann, str):
            kwargs[pname] = raw
        else:
            return {}, f"its {pname} can't be given by voice"
    return kwargs, ""


def inspect_empty():
    import inspect
    return inspect.Parameter.empty


def _warmup(model) -> None:
    """One tiny inference so the first real command doesn't pay setup costs."""
    try:
        import numpy as np
        list(model.transcribe(np.zeros(8000, dtype=np.float32), language="en",
                              beam_size=1, temperature=0.0, without_timestamps=True)[0])
    except Exception:
        log.debug("voice model warm-up inference failed", exc_info=True)


def _model_loader(name: str):
    return load_model(name, device=settings.voice_device,
                      compute_type=settings.voice_compute_type,
                      cpu_threads=settings.voice_cpu_threads)


class VoiceDJ(commands.Cog, name="Voice DJ"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Tier 2: set up in cog_load when an OpenRouter key is configured.
        self.llm: LLMRouter | None = None
        self._typing: dict[tuple[int, int], _Typing] = {}
        self._typing_generation: dict[tuple[int, int], int] = {}
        self._cue_onset: dict[tuple[int, int], float] = {}   # last utterance greeted
        self._cue_msgs: dict[tuple[int, int], tuple[float, object]] = {}   # (onset, its message)
        # The instant wake word (core/wake_spotter.py): what it fired for,
        # until speech-to-text confirms it or takes it back.
        self._fast_wakes: dict[tuple[int, int], object] = {}
        self._wake_confirmed: dict[tuple[int, int], float] = {}  # onset of the utterance cued
        # Fires waiting for the recogniser's check, one per speaker, first
        # come first served (insertion order), and the task working them off.
        self._wake_queue: dict[tuple[int, int], tuple] = {}
        self._wake_worker: asyncio.Task | None = None
        self._instant_cued: dict[tuple[int, int], float] = {}   # onset cued before its check
        self.wake_model = None
        self.wake_verifier = None
        self._probe_task: asyncio.Task | None = None
        self._whisper_ok = False
        self._front = None                                    # VoiceFrontEnd
        # Consent, read on the receive thread: guild → frozenset of user ids.
        self._auth: dict[int, frozenset] = {}
        self._cfg: dict[int, dict] = {}                       # guild → settings snapshot
        self._epoch: dict[tuple[int, int], int] = {}          # bumps when consent is revoked
        self._session: dict[int, int] = {}                    # bumps on move/disconnect/receiver change
        self._pending: dict[tuple[int, int], _Pending] = {}
        self._armed: dict[tuple[int, int], _Armed] = {}
        self._last_action: dict[tuple[int, int], tuple[Intent, float]] = {}
        self._last_audio: dict[tuple[int, int], tuple[float, bytes]] = {}
        self._debug_until: dict[int, float] = {}
        self._debug_log: dict[int, list] = {}
        self.metrics = VoiceMetrics()
        self._jobs = JobQueue(on_drop=self._on_job_dropped)
        self._asr = AsrRunner(timeout=_TRANSCRIBE_TIMEOUT)
        self._models = ModelManager(_model_loader, settings.voice_whisper_model,
                                    warmup=_warmup)
        self._dispatch_q: dict[int, asyncio.Queue] = {}     # control lane, per guild
        self._dispatchers: dict[int, asyncio.Task] = {}
        self._plays: dict[int, deque] = {}                  # song-request lane, per guild
        self._play_workers: dict[int, asyncio.Task] = {}
        self._action_seq = 0
        self._current_job: Job | None = None                # being recognised right now
        self._decisions: dict[tuple[int, int], deque] = {}
        self._decision_workers: dict[tuple[int, int], asyncio.Task] = {}
        self._bg: set[asyncio.Task] = set()                 # strong refs to fire-and-forget work
        self._worker: asyncio.Task | None = None
        self._last_refresh = 0.0
        self._subscribed_to = None
        # Conversation context, per speaker and voice session: their last few
        # requests and what came of them (the planner sees these, so "the one
        # I just added" has a referent), and the queue entries they added.
        self._recent: dict[tuple[int, int], deque] = {}
        self._mine: dict[tuple[int, int], deque] = {}
        # One long utterance arrives as several segments (the front end splits
        # at its length cap, with overlap): their transcripts are joined and
        # understood as one request. key → _Parts while more is coming.
        self._parts: dict[tuple[int, int], "_Parts"] = {}
        # key → (utterance no., its last part's end, why that part ended, part no.)
        self._utt: dict[tuple[int, int], tuple] = {}
        self._utt_seq = 0
        self._ignored_utts: dict[tuple[int, int], int] = {}        # utterance not for us
        self._whole_fallback: dict[tuple, str] = {}                # (key, utt) → joined words
        # Shuffles go through this (tests seed it); None = the random module.
        self._rng: random.Random | None = None
        self._last_run: dict[int, "vx.PlanRun"] = {}           # guild → its last plan run
        # Requests already carried out (speaker, onset): the same speech
        # reaching the planner twice must not queue its songs twice.
        self._done_requests: deque = deque(maxlen=64)

    @property
    def _model_name(self) -> str:
        return self._models.name

    async def cog_load(self) -> None:
        # Listening needs faster-whisper (and with it numpy + the Silero VAD
        # model). Without it, stay idle and quiet — `-transcribe` does the same.
        try:
            import faster_whisper  # noqa: F401  (availability probe)
            from core.voice_frontend import SegmenterConfig, VoiceFrontEnd
            self._whisper_ok = True
        except Exception:
            log.warning("Voice control: faster-whisper isn't installed, so the "
                        "listener is idle. `pip install faster-whisper` to enable.")
        try:
            self._models.name = await db.get_app_setting(
                "voice_whisper_model", settings.voice_whisper_model) or settings.voice_whisper_model
        except Exception:
            pass
        if settings.openrouter_api_key and settings.voice_llm:
            self.llm = LLMRouter(settings.openrouter_api_key, model=settings.voice_llm_model,
                                 fallbacks=settings.voice_llm_fallbacks,
                                 base_url=settings.voice_llm_url,
                                 timeout=settings.voice_llm_timeout,
                                 web=settings.voice_web_lookup,
                                 web_per_hour=settings.voice_web_lookups_per_hour)
            log.info("Voice DJ smart tier on: %s (fallback: %s); planner %s", settings.voice_llm_model,
                     ", ".join(settings.voice_llm_fallbacks) or "none",
                     settings.voice_plan_model or "the same")
        if self._whisper_ok:
            probe_ms = settings.voice_probe_ms if settings.voice_typing else 0
            cfg = SegmenterConfig(threshold=settings.voice_vad_threshold,
                                  neg_threshold=max(0.05, settings.voice_vad_threshold - 0.15),
                                  end_silence_ms=settings.voice_end_silence_ms,
                                  dtx_timeout_ms=settings.voice_end_silence_ms,
                                  probe_ms=probe_ms)
            if settings.voice_fast_wake:
                self.wake_model = await self._load_wake_model()
                if self.wake_model is not None and settings.voice_fast_wake_verify:
                    self._spawn(self._load_wake_verifier())
            self._front = VoiceFrontEnd(self._segment_from_thread, cfg=cfg,
                                        on_probe=self._probe_from_thread if probe_ms else None,
                                        on_wake=self._wake_from_thread,
                                        wake_model=self.wake_model,
                                        wake_threshold=settings.voice_fast_wake_threshold or None)
            self._front.start()
            self._worker = asyncio.create_task(self._asr_worker(), name="voice-asr-worker")
        self._maintenance.start()

    async def cog_unload(self) -> None:
        self._maintenance.cancel()
        for key in list(self._typing):
            self._typing_stop(key)
        if self.llm is not None:
            await self.llm.close()
        if self._worker is not None:
            self._worker.cancel()
        for task in [*self._dispatchers.values(), *self._play_workers.values(),
                     *self._decision_workers.values(), *self._bg]:
            task.cancel()
        rec = self.bot.get_cog("Recording")
        if rec is not None:
            rec.unsubscribe(self._on_frame, self._on_receiver_session)
        if self._front is not None:
            self._front.stop()
        if self.wake_verifier is not None:
            self.wake_verifier.shutdown()
        self._asr.shutdown()
        self._models.shutdown()

    async def cog_check(self, ctx: commands.Context) -> bool:
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _wake_words(self, guild_id: int) -> list[str]:
        """All accepted wake phrases for a guild, falling back to the host
        default when none are configured. Never empty."""
        wakes = await db.get_voice_wakes(guild_id)
        return wakes or [settings.voice_wake_word]

    @staticmethod
    def _fmt_wakes(wakes: list[str]) -> str:
        """Human display for one or many wake phrases."""
        return " / ".join(f"“{w}”" for w in wakes)

    async def _feedback_for(self, guild_id: int, user_id: int) -> str:
        """The effective feedback style for a member: their own choice, else the
        guild default."""
        user = await db.get_voice_user(guild_id, user_id)
        if user["feedback"]:
            return user["feedback"]
        return (await db.get_voice_settings(guild_id))["feedback_default"]

    def _debug_on(self, guild_id: int) -> bool:
        return self._debug_until.get(guild_id, 0.0) > time.monotonic()

    # ── config group ─────────────────────────────────────────────────────────

    @commands.hybrid_group(
        name="voice", invoke_without_command=True,
        description="Hands-free voice control of the music player",
        help=(
            "Control the music player by talking in the voice channel — say the "
            "wake word then a command, e.g. “hey DJ, skip”. You can also pause "
            "after the wake word: “hey DJ” … “play mr brightside”.\n\n"
            "**Setup**\n"
            "`voice enable` / `voice disable` — turn it on/off for the server "
            "*(Manage Server)*\n"
            "`voice wake [add|remove|set|clear] <phrase>` — manage wake words "
            "*(Manage Server)*\n"
            "`voice default <style>` — set the server's default reply style "
            "*(Manage Server)*\n"
            "`voice channel [#channel|clear]` — where replies are posted "
            "*(Manage Server)*\n"
            "`voice model [name]` — show/change the speech model *(bot owner)*\n"
            "`voice ai [on|off]` · `voice ai chat on|off` — the smart tier: several "
            "songs at once, mixed commands, and spoken answers to questions "
            "*(Manage Server)*\n"
            "`voice cue [on|off]` — the sound played when the wake word is heard "
            "*(Manage Server)*\n\n"
            "**For you**\n"
            "`voice optin [style]` — opt in (only opted-in people are listened to)\n"
            "`voice optout` — opt back out\n"
            "`voice feedback <style>` — how the bot replies to *you*\n"
            "`voice status` — settings, your opt-in, and whether it's working\n\n"
            "**Troubleshooting**\n"
            "`voice diag` — what the pipeline is doing *(Manage Server)*\n"
            "`voice debug on|off [minutes]` — temporarily log what it hears "
            "*(Manage Server)*\n"
            "`voice clip` — DMs you the last thing of yours it transcribed\n"
            "`voice doctor` — host self-check *(bot owner)*\n\n"
            "Reply styles: `text`, `voice`, `silent`, `chime`. Commands: skip/next, "
            "pause/hold up, resume, start over, stop, clear the queue, loop on/off, "
            "louder/quieter, volume 50, shuffle, loop the queue, play <song>.\n"
            "With the smart tier, say what you want in your own words and I work out "
            "what you mean: “play the first three songs off Vices and Virtues by "
            "Panic! At The Disco, turn it down to 50, shuffle, skip the first one and "
            "loop the rest”, “play the song from Top Gun”, “play Owl City's most "
            "popular songs” — done in the order you said. Or just ask a question "
            "(“who sings this?”) and I'll answer out loud."
        ),
    )
    async def voice(self, ctx: commands.Context):
        await self._send_status(ctx)

    @voice.command(name="enable",
                   description="Turn on voice control for this server (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_enable(self, ctx: commands.Context):
        await db.set_voice_enabled(ctx.guild.id, True)
        await self._refresh_guild(ctx.guild.id)
        self._prefetch_model()
        wakes = await self._wake_words(ctx.guild.id)
        word_label = "Wake word" if len(wakes) == 1 else "Wake words"
        await ctx.send(
            f"🎙️ **Voice control enabled.** {word_label}: **{self._fmt_wakes(wakes)}**.\n"
            f"🔴 Heads up — while I'm in a voice channel I listen for spoken "
            f"commands from people who've opted in. Each person must run "
            f"`{ctx.prefix}voice optin` first; nobody else is transcribed, and "
            f"nothing said before someone opts in is ever processed.\n"
            f"I have to be **in your voice channel** to hear you — pull me in with "
            f"`{ctx.prefix}play` or `{ctx.prefix}join`. "
            f"Manage wake words with `{ctx.prefix}voice wake add/remove <phrase>`."
        )

    @voice.command(name="disable",
                   description="Turn off voice control for this server (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_disable(self, ctx: commands.Context):
        await db.set_voice_enabled(ctx.guild.id, False)
        await self._refresh_guild(ctx.guild.id)       # revokes + purges immediately
        await ctx.send("🔇 Voice control disabled for this server.")

    @voice.group(name="wake", invoke_without_command=True,
                 description="Manage wake words (Manage Server)",
                 help=(
                     "The bot acts on a command only after a wake phrase. A "
                     "server can have several.\n\n"
                     "`voice wake` — list the current wake words\n"
                     "`voice wake add <phrase>` — add one *(Manage Server)*\n"
                     "`voice wake remove <phrase>` — remove one *(Manage Server)*\n"
                     "`voice wake clear` — reset to the default *(Manage Server)*\n"
                     "`voice wake set <phrase>` — replace all with one *(Manage Server)*"
                 ))
    async def voice_wake(self, ctx: commands.Context):
        wakes = await self._wake_words(ctx.guild.id)
        configured = await db.get_voice_wakes(ctx.guild.id)
        suffix = "" if configured else " *(default — none set)*"
        await ctx.send(
            f"🗣️ Wake word{'s' if len(wakes) != 1 else ''}: "
            f"**{self._fmt_wakes(wakes)}**{suffix}\n"
            f"Add more with `{ctx.prefix}voice wake add <phrase>`."
        )

    @staticmethod
    def _validate_wake(phrase: str) -> str | None:
        """Normalise + validate a wake phrase with the SAME normaliser the
        matcher uses (core.voice_intent.normalize_wake). The old check allowed
        any Unicode letters, which the matcher then stripped — a phrase that
        normalised to nothing opened the gate for all speech. Returns the
        canonical phrase ("Hey D.J." → "hey dj"), or None if unusable."""
        return normalize_wake(phrase)

    _BAD_WAKE = ("❌ Wake phrase must be 1–4 words of plain Latin letters "
                 "(32 characters max), e.g. `hey dj`, `computer`, `yo bot` — "
                 "the speech model writes English, so other scripts never match.")

    @staticmethod
    def _wake_tip(phrase: str) -> str:
        if len(phrase.replace(" ", "")) <= 3 or any(
            len(w) <= 2 for w in phrase.split()
        ):
            return ("\n-# 💡 Short words and letters (like “dj”) are hard for "
                    "speech-to-text — a normal word like `computer` or `jarvis` "
                    "is recognised far more reliably.")
        return ""

    @voice_wake.command(name="add", description="Add a wake word (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_wake_add(self, ctx: commands.Context, *, phrase: str):
        clean = self._validate_wake(phrase)
        if clean is None:
            return await ctx.send(self._BAD_WAKE)
        before = await db.get_voice_wakes(ctx.guild.id)
        if clean in before:
            return await ctx.send(f"ℹ️ **“{clean}”** is already a wake word.")
        wakes = await db.add_voice_wake(ctx.guild.id, clean)
        await self._refresh_guild(ctx.guild.id)
        await ctx.send(
            f"✅ Added wake word **“{clean}”**. Now: **{self._fmt_wakes(wakes)}**."
            f"{self._wake_tip(clean)}"
        )

    @voice_wake.command(name="remove", aliases=["rm", "delete"],
                        description="Remove a wake word (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_wake_remove(self, ctx: commands.Context, *, phrase: str):
        clean = self._validate_wake(phrase) or " ".join(phrase.lower().split())
        before = await db.get_voice_wakes(ctx.guild.id)
        if clean not in before:
            return await ctx.send(
                f"❌ **“{clean}”** isn't a configured wake word. "
                f"Current: **{self._fmt_wakes(await self._wake_words(ctx.guild.id))}**."
            )
        wakes = await db.remove_voice_wake(ctx.guild.id, clean)
        await self._refresh_guild(ctx.guild.id)
        if wakes:
            await ctx.send(
                f"✅ Removed **“{clean}”**. Now: **{self._fmt_wakes(wakes)}**."
            )
        else:
            await ctx.send(
                f"✅ Removed **“{clean}”**. No custom wake words left — "
                f"falling back to the default **“{settings.voice_wake_word}”**."
            )

    @voice_wake.command(name="set", description="Replace all wake words with one (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_wake_set(self, ctx: commands.Context, *, phrase: str):
        clean = self._validate_wake(phrase)
        if clean is None:
            return await ctx.send(self._BAD_WAKE)
        await db.set_voice_wakes(ctx.guild.id, [clean])
        await self._refresh_guild(ctx.guild.id)
        await ctx.send(f"✅ Wake word set to **“{clean}”**.{self._wake_tip(clean)}")

    @voice_wake.command(name="clear", description="Reset wake words to the default (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_wake_clear(self, ctx: commands.Context):
        await db.set_voice_wakes(ctx.guild.id, [])
        await self._refresh_guild(ctx.guild.id)
        await ctx.send(
            f"✅ Wake words reset to the default **“{settings.voice_wake_word}”**."
        )

    @voice.command(name="default",
                   description="Set the server's default reply style (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_default(self, ctx: commands.Context, style: str):
        style = style.lower()
        if style not in FEEDBACK_MODES:
            return await ctx.send(self._bad_style_msg())
        await db.set_voice_feedback_default(ctx.guild.id, style)
        await ctx.send(
            f"✅ Default reply style set to **{style}** — "
            f"{FEEDBACK_MODES[style]} *(for anyone who hasn't picked their own)*"
        )

    @voice.command(name="channel",
                   description="Choose where voice-command replies go (Manage Server)",
                   help=("`voice channel #music` — always reply there\n"
                         "`voice channel clear` — reply where I was last summoned, "
                         "else where music was last used (remembered across "
                         "restarts) — the default\n"
                         "With none of those, replies fall back to the server's "
                         "system channel — never to a random channel."))
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_channel(self, ctx: commands.Context, *, target: Optional[str] = None):
        if target and target.strip().lower() in ("clear", "reset", "default", "none"):
            await db.set_voice_feedback_channel(ctx.guild.id, None)
            return await ctx.send("✅ Voice-command replies go where I was last "
                                  "summoned, else where music was last used "
                                  "(or the system channel).")
        channel = ctx.channel
        if target:
            try:
                channel = await commands.TextChannelConverter().convert(ctx, target.strip())
            except commands.BadArgument:
                return await ctx.send("❌ Name a text channel, e.g. `voice channel #music`.")
        if not channel.permissions_for(ctx.guild.me).send_messages:
            return await ctx.send(f"❌ I can't send messages in {channel.mention}.")
        await db.set_voice_feedback_channel(ctx.guild.id, channel.id)
        await ctx.send(f"✅ Voice-command replies will go to {channel.mention}.")

    @voice.command(
        name="ai",
        description="The smart tier: complex commands and spoken answers",
        help=(
            "Song requests, and anything the built-in voice commands don't cover, "
            "go to an AI (via OpenRouter) that works out what you meant: the actual "
            "song (“the one from Top Gun”), an album's tracks (“the first three off "
            "…”, looked up in a music catalogue), an artist's most popular or newest "
            "songs, several commands in one go — run in the order you said them — or "
            "a question I answer out loud. Plain controls (“skip”, “louder”) stay "
            "instant and offline.\n\n"
            "`voice ai` — status, model and what it has cost\n"
            "`voice ai on` / `voice ai off` — for this server *(Manage Server)*\n"
            "`voice ai chat on` / `voice ai chat off` — answering questions "
            "*(Manage Server)*\n\n"
            "Needs `OPENROUTER_API_KEY` in the bot's `.env` (host)."
        ),
    )
    async def voice_ai(self, ctx: commands.Context, what: Optional[str] = None,
                       value: Optional[str] = None):
        gid = ctx.guild.id
        what = (what or "").strip().lower()
        value = (value or "").strip().lower()
        if not what or what == "status":
            return await ctx.send(f"🧠 **Smart voice tier:** {self._ai_line(gid)}")
        manager = (ctx.author.guild_permissions.manage_guild
                   or await self.bot.is_owner(ctx.author))
        if what in ("on", "off") or (what == "chat" and value in ("on", "off")):
            if not manager:
                return await ctx.send("❌ You need **Manage Server** to change that.")
            if what == "chat":
                await db.set_app_setting(f"voice_chat:{gid}", value)
                msg = ("✅ I'll answer questions out loud." if value == "on"
                       else "✅ I won't answer questions — commands only.")
            else:
                await db.set_app_setting(f"voice_ai:{gid}", what)
                msg = ("✅ Smart tier on for this server." if what == "on"
                       else "✅ Smart tier off — only the built-in voice commands.")
            await self._refresh_guild(gid)
            if self.llm is None and (what == "on" or value == "on"):
                msg += ("\n⚠️ It still won't run: there's no `OPENROUTER_API_KEY` in the "
                        "bot's `.env` (or `VOICE_LLM` is off). The host needs to add it "
                        "and restart me.")
            return await ctx.send(msg)
        await ctx.send(f"Usage: `{ctx.prefix}voice ai [on|off]` · "
                       f"`{ctx.prefix}voice ai chat on|off`")

    @voice.command(
        name="cue",
        description="The \"I'm listening\" cue when the wake word is heard",
        help=(
            "When I hear the wake word I play a short sound in the voice channel "
            "and post `:ear:*Now listening...*` in the reply channel — the audio "
            "version of a smart speaker's light.\\n\\n"
            "`voice cue` — show the setting\\n"
            "`voice cue on` / `voice cue off` — the sound effect *(Manage Server)*\\n\\n"
            "The message always posts. On the host, `VOICE_CUE=false` turns the "
            "whole cue off and `VOICE_CUE_SOUND=false` mutes just the sound."
        ),
    )
    async def voice_cue(self, ctx: commands.Context, value: Optional[str] = None):
        gid = ctx.guild.id
        value = (value or "").strip().lower()
        if not value or value == "status":
            return await ctx.send(f"🔔 **Wake-word cue:** {self._cue_line(gid)}")
        if value not in ("on", "off"):
            return await ctx.send(f"Usage: `{ctx.prefix}voice cue [on|off]`")
        if not (ctx.author.guild_permissions.manage_guild
                or await self.bot.is_owner(ctx.author)):
            return await ctx.send("❌ You need **Manage Server** to change that.")
        await db.set_app_setting(f"voice_cue:{gid}", value)
        await self._refresh_guild(gid)
        if value == "on":
            msg = "🔔 Cue sound on — you'll hear it the moment I catch the wake word."
        else:
            msg = ("🔕 Cue sound off — you'll still get the "
                   "`:ear:*Now listening...*` message.")
        if settings.voice_cue and settings.voice_cue_sound:
            return await ctx.send(msg)
        return await ctx.send(msg + "\n⚠️ It still won't play: the host has the cue "
                                    "sound switched off (`VOICE_CUE_SOUND`/`VOICE_CUE`).")

    def _fast_wake_line(self, guild_id: int) -> str:
        if self.wake_model is None:
            return ("off (`VOICE_FAST_WAKE`)" if not settings.voice_fast_wake
                    else "unavailable — the cue comes from speech-to-text only")
        if not self._fast_wake_on(guild_id):
            return "not for this server's wake words (it only knows “hey DJ”)"
        v = self.wake_verifier
        med = v.median_seconds() if v is not None else None
        c = self.metrics.guild(guild_id).counts
        thr = f"{self._front.wake_threshold:.2f}" if self._front else "—"
        return (f"on · threshold {thr}"
                + (f" · check {med:.2f}s median" if med is not None else
                   " · checker loading" if settings.voice_fast_wake_verify and v is None else "")
                + " · " + ", ".join(f"{k.replace('fast_wake', '').strip('_') or 'cued'} {c.get(k, 0)}"
                                    for k in ("fast_wake_fired", "fast_wake", "fast_wake_rejected",
                                              "fast_wake_confirmed", "fast_wake_false",
                                              "fast_wake_missed")))

    def _ai_line(self, guild_id: int, *, short: bool = False) -> str:
        cfg = self._cfg.get(guild_id, {})
        if self.llm is None:
            return ("off — no OpenRouter key on the host" if short else
                    "⚪ off — the host has no `OPENROUTER_API_KEY` in `.env` "
                    "(or `VOICE_LLM` is off)")
        if not cfg.get("ai", True):
            return "⚪ off for this server" + ("" if short else " (`voice ai on`)")
        llm = self.llm
        chat = "answers questions" if cfg.get("chat", True) else "commands only"
        if short:
            return f"🟢 on · {chat}"
        med = llm.median_seconds()
        web = (f"on ({llm.web_per_hour}/hour max)" if llm.web else "off (`VOICE_WEB_LOOKUP`)")
        out = (f"🟢 on · {chat}\n**Model:** `{llm.model}`"
               + (f" (fallback `{', '.join(llm.fallbacks)}`)" if llm.fallbacks else "")
               + (f"\n**Planner:** `{settings.voice_plan_model}`"
                  + (f" (fallback `{', '.join(settings.voice_plan_fallbacks)}`)"
                     if settings.voice_plan_fallbacks else "")
                  if settings.voice_plan_model else "")
               + f"\n**Web lookups** for songs described, not named: {web}"
               + f"\n**Since restart:** {llm.calls} asked"
               + (f", median {med:.2f}s" if med is not None else "")
               + (" · " + ", ".join(f"{k} {v}" for k, v in sorted(llm.kinds.items()))
                  if llm.kinds else "")
               + f" · cost ${llm.cost:.4f}")
        if llm.failures:
            last = llm.last_error.replace("`", "'")
            out += f"\n**Failures:** {llm.failures} · last: `{last}`"
        return out

    @voice.command(
        name="model",
        description="Show or change the speech model for voice commands (bot owner)",
        help=(
            "The whisper model used to recognise spoken commands. Bigger = more "
            "accurate but slower. It's host-wide — one shared model for every "
            "server — so only the bot owner can change it. The choice persists "
            "across restarts.\n\n"
            "Measure before switching: `scripts/voice_eval.py` compares models on "
            "real recordings. English-only `.en` variants are the most accurate here.\n\n"
            "**Examples**\n"
            "`voice model` — show the current model and the options\n"
            "`voice model small.en` — switch to small.en"
        ),
    )
    async def voice_model(self, ctx: commands.Context, name: Optional[str] = None):
        opts = ", ".join(f"`{m}`" for m in KNOWN_MODELS)
        if name is None:
            return await ctx.send(
                f"🧠 Voice model: **{self._models.name}** ({self._models.state}).\n"
                f"Change with `{ctx.prefix}voice model <name>` *(bot owner, "
                f"host-wide, persists across restarts)*.\n**Options:** {opts}"
            )
        if not await self.bot.is_owner(ctx.author):
            return await ctx.send("❌ Only the bot owner can change the speech model "
                                  "— it's shared by every server this bot is in.")
        name = name.strip()
        if name not in KNOWN_MODELS:
            return await ctx.send(f"❌ Unknown model `{name}`.\n**Options:** {opts}")
        if not self._whisper_ok:
            return await ctx.send(
                "❌ faster-whisper isn't installed on the host — there's no speech "
                "model to switch. Run `pip install faster-whisper` and restart."
            )
        if name == self._models.name and self._models.model is not None:
            return await ctx.send(f"🧠 Already using **{name}**.")
        status = await ctx.send(
            f"⏳ Loading **{name}**… the first use downloads it, which can take a "
            f"minute. The current model keeps working until it's ready."
        )
        try:
            await self._models.switch(name)          # serialized; atomic on success
        except Exception as e:
            log.exception("voice model switch failed")
            return await status.edit(content=f"❌ Couldn't load `{name}`: {e}")
        await db.set_app_setting("voice_whisper_model", name)
        await status.edit(
            content=f"✅ Voice model is now **{name}**. (Saved — survives restarts.)"
        )

    @voice.command(name="optin",
                   description="Opt into voice control (optionally set your reply style)")
    async def voice_optin(self, ctx: commands.Context, style: Optional[str] = None):
        if style is not None:
            style = style.lower()
            if style not in FEEDBACK_MODES:
                return await ctx.send(self._bad_style_msg())
        await db.set_voice_optin(ctx.guild.id, ctx.author.id, True)
        if style is not None:
            await db.set_voice_user_feedback(ctx.guild.id, ctx.author.id, style)
        await self._refresh_guild(ctx.guild.id)

        s = await db.get_voice_settings(ctx.guild.id)
        eff = await self._feedback_for(ctx.guild.id, ctx.author.id)
        wakes = await self._wake_words(ctx.guild.id)
        msg = (f"✅ You're opted into voice control. Replies to you: **{eff}**.\n"
               f"Say **{self._fmt_wakes(wakes)}** then a command "
               f"(e.g. “…skip”, “…play mr brightside”). Only what you say from "
               f"now on is listened to.")
        if not s["enabled"]:
            msg += (f"\n-# Note: an admin hasn't run `{ctx.prefix}voice enable` "
                    f"yet, so nothing's listening server-wide right now.")
        if self.llm is not None and self._cfg.get(ctx.guild.id, {}).get("ai", True):
            msg += ("\n-# The smart tier is on here: song requests, and anything the built-in "
                    "commands don't cover, are sent as text (never audio) to an AI service "
                    "(OpenRouter) to work out what you meant.")
        await ctx.send(msg)

    @voice.command(name="optout", description="Opt out of voice control")
    async def voice_optout(self, ctx: commands.Context):
        await db.set_voice_optin(ctx.guild.id, ctx.author.id, False)
        await self._refresh_guild(ctx.guild.id)       # immediate: purges queued audio
        await ctx.send("🚪 You're opted out — I've stopped listening to you, and "
                       "anything of yours still in the queue was discarded.")

    @voice.command(name="feedback",
                   description="Set how the bot replies to you (text/voice/silent/chime)")
    async def voice_feedback(self, ctx: commands.Context, style: str):
        style = style.lower()
        if style not in FEEDBACK_MODES:
            return await ctx.send(self._bad_style_msg())
        await db.set_voice_user_feedback(ctx.guild.id, ctx.author.id, style)
        note = ""
        if style == "voice" and self._tts() is None:
            note = "\n-# TTS isn't available on the host, so this falls back to text."
        await ctx.send(f"✅ I'll reply to you with **{style}** — "
                       f"{FEEDBACK_MODES[style]}{note}")

    @voice.command(name="status", description="Show voice-control settings")
    async def voice_status(self, ctx: commands.Context):
        await self._send_status(ctx)

    def _health_line(self, guild: discord.Guild) -> str:
        """One honest line: is it actually able to hear and understand?"""
        if not self._whisper_ok:
            return "🔴 not working — faster-whisper isn't installed on the host"
        vc = guild.voice_client
        if vc is None or not vc.is_connected():
            return "⚪ idle — I'm not in a voice channel"
        rec = self.bot.get_cog("Recording")
        h = (rec.health.get(guild.id) if rec else None) or {}
        if h.get("state") != "listening":
            return f"🟠 receiver {h.get('state', 'not attached')} — can't hear anyone right now"
        if self._models.state == "failed":
            return f"🔴 speech model failed to load: {self._models.error}"
        if self._models.state != "ready":
            return f"🟡 speech model {self._models.state} — commands may be slow at first"
        if self._asr.stuck_for > 0:
            return "🟠 recogniser is stuck on a slow clip — commands are being dropped"
        return "🟢 listening"

    async def _send_status(self, ctx: commands.Context):
        s = await db.get_voice_settings(ctx.guild.id)
        wakes = await self._wake_words(ctx.guild.id)
        me = await db.get_voice_user(ctx.guild.id, ctx.author.id)
        optins = await db.list_voice_optins(ctx.guild.id)
        eff = me["feedback"] or s["feedback_default"]
        chan = ctx.guild.get_channel(s["feedback_channel_id"]) if s["feedback_channel_id"] else None

        embed = discord.Embed(
            title="🎙️ Voice control",
            color=discord.Color.blurple() if s["enabled"] else discord.Color.greyple(),
            description=(
                f"**Status:** {'🟢 enabled' if s['enabled'] else '⚪ disabled'}\n"
                f"**Working:** {self._health_line(ctx.guild) if s['enabled'] else '—'}\n"
                f"**Wake word{'s' if len(wakes) != 1 else ''}:** {self._fmt_wakes(wakes)}\n"
                f"**Model:** {self._models.name} ({self._models.state})\n"
                f"**Default reply style:** {s['feedback_default']}\n"
                f"**Replies go to:** {chan.mention if chan else 'where I was summoned'}\n"
                f"**Smart tier:** {self._ai_line(ctx.guild.id, short=True)}\n"
                f"**Wake cue:** {self._cue_line(ctx.guild.id, short=True)}\n"
                f"**Opted-in members:** {len(optins)}"
            ),
        )
        embed.add_field(
            name="You",
            value=(f"{'✅ opted in' if me['opted_in'] else '⬜ not opted in'} · "
                   f"reply style: **{eff}**"),
            inline=False,
        )
        embed.set_footer(text=f"{ctx.prefix}voice — full command list · "
                              f"{ctx.prefix}voice diag for details")
        await ctx.send(embed=embed)

    def _bad_style_msg(self) -> str:
        opts = "\n".join(f"• `{k}` — {v}" for k, v in FEEDBACK_MODES.items())
        return f"❌ Unknown reply style. Pick one of:\n{opts}"

    # ── diagnostics ──────────────────────────────────────────────────────────

    @voice.command(name="diag", description="What the voice pipeline is doing (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_diag(self, ctx: commands.Context):
        await self._send_report(ctx, self._diag_text(ctx.guild), "voice-diag.txt")

    def _diag_text(self, g: discord.Guild) -> str:
        m = self.metrics.summary(g.id)
        rec = self.bot.get_cog("Recording")
        h = (rec.health.get(g.id) if rec else None) or {}
        r = rec.recorders.get(g.id) if rec else None
        fmt = lambda x: "—" if x is None else f"{x:.2f}s"   # noqa: E731
        lines = [
            f"**Health:** {self._health_line(g)}",
            f"**Receiver:** {h.get('state', 'n/a')} · restarts {h.get('restarts', 0)}"
            + (f" · last error `{h['error']}`" if h.get("error") else ""),
        ]
        if r is not None:
            lines.append(f"**Frames:** {r.frames} received · DAVE not-ready "
                         f"{r.dave_not_ready} · decrypt failures {r.dave_failures}")
        fe_stats = self._front.stats(g.id) if self._front else {}
        if fe_stats:
            rows = []
            for (_gid, uid), st in sorted(fe_stats.items()):
                member = g.get_member(uid)
                name = member.display_name if member else str(uid)
                rows.append(f"`{name[:16]:16}` pkts {st['packets']} · lost→concealed "
                            f"{st['concealed']} · decode err {st['decode_errors']} · "
                            f"speech {st['speech_windows']}/{st['vad_windows']} win · "
                            f"segments {st['segments']} (+{st['dropped_short']} blips)")
            lines.append("**Speakers (front end):**\n" + "\n".join(rows[:25]))
        lines.append(
            f"**Model:** {self._models.name} — {self._models.state}"
            + (f" (load {self._models.load_seconds:.1f}s)" if self._models.load_seconds else "")
            + (f" · error `{self._models.error}`" if self._models.error else ""))
        lines.append(f"**Recognition (ASR only):** median {fmt(m['asr_median'])}, p95 {fmt(m['asr_p95'])} · "
                     f"queue {len(self._jobs)} (oldest {self._jobs.oldest_age():.1f}s) · "
                     f"timeouts {self._asr.timeouts}")
        lines.append("**Smart tier (LLM):** " + self._ai_line(g.id))
        lines.append("**Instant wake word:** " + self._fast_wake_line(g.id))
        cues = cue_summary(self.metrics.guild(g.id).cue_timings)
        if cues.get("total"):
            def ms(k):
                v = cues.get(k)
                return "—" if not v else f"{v[0]:.0f} ms (p90 {v[1]:.0f}, p95 {v[2]:.0f})"
            lines.append(f"**Wake cue, server side** (last {cues['total'][3]}; "
                         + ", ".join(f"{k} {n}" for k, n in sorted(cues["tiers"].items()))
                         + f"): detector frame → first sound frame out {ms('total')} · "
                         f"check {ms('check')} · mixer → frame {ms('mixer_to_frame')}. "
                         "(What a listener hears also waits on Discord's transport and their "
                         "client — not measured here.)")
        lines.append(f"**End of speech → action done:** median {fmt(m['latency_median'])}, "
                     f"p95 {fmt(m['latency_p95'])} · **→ first audio of a requested song:** "
                     f"median {fmt(m['audible_median'])}, p95 {fmt(m['audible_p95'])}")
        qc, qo = m["quality_commands"], m["quality_other"]

        def q(t):
            lp, ns, n = t
            return "—" if not n else f"logprob {lp:.2f} · no-speech {ns:.2f} (n={n})"
        lines.append(f"**ASR quality (median):** commands {q(qc)} · other speech {q(qo)}")
        if m["actions"]:
            rows = []
            for action, c in sorted(m["actions"].items()):
                stages = [f"{st} {c[st]}" for st in (*ACTION_STAGES, "queued", "playback_failed")
                          if c.get(st)]
                rows.append(f"`{action:8}` " + " · ".join(stages))
            lines.append("**Commands:**\n" + "\n".join(rows))
        if m["counts"]:
            lines.append("**Counts:** " + ", ".join(f"{k} {v}" for k, v in sorted(m["counts"].items())))
        if m["parse"]:
            lines.append("**Parser:** " + ", ".join(f"{k} {v}" for k, v in sorted(m["parse"].items())))
        if m["drops"]:
            lines.append("**Dropped:** " + ", ".join(f"{k} {v}" for k, v in
                                                     sorted(m["drops"].items(), key=lambda kv: -kv[1])))
        if m["last_error"]:
            lines.append(f"**Last dispatch error:** `{m['last_error']}`")
        if m["events"]:
            ev = [f"<t:{int(t)}:T> {g.get_member(u).display_name if g.get_member(u) else u}: {w}"
                  for t, u, w in m["events"][-6:]]
            lines.append("**Recent:**\n" + "\n".join(ev))
        if self._debug_on(g.id):
            left = int(self._debug_until[g.id] - time.monotonic())
            heard = self._debug_log.get(g.id, [])[-5:]
            quality = m["quality_recent"]
            lines.append(f"**Debug mode on** ({left // 60}m {left % 60}s left) — last heard:\n"
                         + ("\n".join(f"• {n}: “{t}” → {res}" for n, t, res in heard) or "• (nothing yet)"))
            if quality:
                lines.append("**Last clips (logprob / no-speech / seconds → outcome):**\n"
                             + "\n".join(f"• {lp:.2f} / {ns:.2f} / {sec:.1f}s → {out}"
                                          for lp, ns, sec, out in quality))
        return "\n".join(lines)

    async def _send_report(self, ctx: commands.Context, text: str, filename: str) -> None:
        """Send a report that may be long. Discord rejects message content
        over 2000 characters, so a long report goes out as its first part plus
        the whole thing attached as a text file — never as a failed send."""
        chunks = chunk_lines(text, 1900)
        if len(chunks) <= 1:
            await ctx.send(text or "—", allowed_mentions=discord.AllowedMentions.none())
            return
        head = chunks[0]
        if len(head) > 1850:
            head = head[:1850]
        await ctx.send(head + "\n-# …the full report is attached.",
                       file=discord.File(io.BytesIO(text.encode("utf-8")), filename=filename),
                       allowed_mentions=discord.AllowedMentions.none())

    @voice.command(name="debug",
                   description="Temporarily log what voice control hears (Manage Server)",
                   help=("`voice debug on [minutes]` — for the next 10 minutes (max 60), "
                         "what opted-in people say is written to the bot's log and "
                         "shown in `voice diag`, so you can see exactly what whisper "
                         "heard and why a command did or didn't fire.\n"
                         "`voice debug off` — stop now and forget it.\n"
                         "Off by default: ordinary conversation is not logged."))
    @commands.has_guild_permissions(manage_guild=True)
    async def voice_debug(self, ctx: commands.Context, state: str = "on", minutes: int = 10):
        if state.lower() in ("off", "stop", "0", "false"):
            self._debug_until.pop(ctx.guild.id, None)
            self._debug_log.pop(ctx.guild.id, None)
            return await ctx.send("🔕 Voice debug off — transcripts are no longer logged.")
        minutes = max(1, min(60, minutes))
        self._debug_until[ctx.guild.id] = time.monotonic() + minutes * 60
        await ctx.send(f"🔎 Voice debug **on for {minutes} min** — what opted-in "
                       f"people say is logged and shown in `{ctx.prefix}voice diag`. "
                       f"`{ctx.prefix}voice debug off` to stop early.")

    @voice.command(name="clip",
                   description="DM yourself the last thing of yours voice control transcribed")
    async def voice_clip(self, ctx: commands.Context):
        entry = self._last_audio.get((ctx.guild.id, ctx.author.id))
        if entry is None:
            return await ctx.send("❌ I haven't transcribed anything of yours lately.")
        captured, wav = entry
        try:
            await ctx.author.send(
                f"🎙️ This is exactly what reached the speech model from you "
                f"(16 kHz mono, {int(time.monotonic() - captured)}s ago). It's only "
                f"kept in memory, and only your own audio is ever sent to you.",
                file=discord.File(io.BytesIO(wav), filename="voice-command.wav"))
        except discord.HTTPException:
            return await ctx.send("❌ I couldn't DM you — check your privacy settings.")
        await ctx.send("📬 Sent it to your DMs.")

    @voice.command(name="doctor", description="Host self-check for voice control (bot owner)")
    @commands.is_owner()
    async def voice_doctor(self, ctx: commands.Context):
        from core import preflight
        async with ctx.typing():
            report = await asyncio.to_thread(preflight.run, load_model_name=self._models.name
                                             if self._models.model is None else None)
        text = preflight.format_report(report, markdown=True)
        await self._send_report(ctx, text, "voice-doctor.txt")

    # ──────────────────────────────────────────────────────────────────────────
    # Consent, sessions, maintenance
    # ──────────────────────────────────────────────────────────────────────────

    def _prefetch_model(self) -> None:
        if self._whisper_ok and any(self._auth.values()):
            self._models.prefetch()

    async def _refresh_guild(self, guild_id: int) -> None:
        """Re-read a guild's settings and apply them NOW. Anyone no longer
        allowed is revoked: their partial speech, queued jobs, open questions
        and wake windows are all discarded."""
        s = await db.get_voice_settings(guild_id)
        optins = await db.list_voice_optins(guild_id) if s["enabled"] else set()
        wakes = await db.get_voice_wakes(guild_id) or [settings.voice_wake_word]
        allowed = frozenset(optins) if (s["enabled"] and self._whisper_ok) else frozenset()
        for uid in self._auth.get(guild_id, frozenset()) - allowed:
            self._revoke(guild_id, uid, "revoked")
        self._auth[guild_id] = allowed
        ai = (await db.get_app_setting(f"voice_ai:{guild_id}", "on")) != "off"
        chat = (await db.get_app_setting(f"voice_chat:{guild_id}", "on")) != "off"
        cue = (await db.get_app_setting(f"voice_cue:{guild_id}", "on")) != "off"
        self._cfg[guild_id] = {"enabled": s["enabled"], "wake": wakes,
                               "channel_id": s["feedback_channel_id"],
                               "ai": ai, "chat": chat, "cue": cue}

    async def _refresh_all(self) -> None:
        try:
            enabled = await db.list_voice_enabled_guilds()
        except Exception:
            log.exception("voice: couldn't list enabled guilds")
            return
        for gid in enabled | set(self._auth):
            try:
                await self._refresh_guild(gid)
            except Exception:
                log.exception("voice: refresh failed for guild %s", gid)
        self._last_refresh = time.monotonic()

    def _log_restored(self) -> None:
        """Say out loud what the database just restored, so a restart that
        loses voice control is visible in the log instead of guessed at."""
        enabled = sorted(g for g, cfg in self._cfg.items() if cfg.get("enabled"))
        members = sum(len(self._auth.get(g, ())) for g in enabled)
        if enabled:
            log.info("Voice control: restored %d enabled guild(s) [%s] with %d "
                     "opted-in member(s) from the database",
                     len(enabled), ", ".join(str(g) for g in enabled), members)
        else:
            log.info("Voice control: no guild has it enabled in the database")

    async def _remember_voice_channel(self, guild_id: int, channel) -> None:
        """Persist — or forget, when the bot left — the channel it is in."""
        try:
            await db.set_app_setting(
                f"{_VOICE_CHANNEL_KEY}{guild_id}", str(channel.id) if channel else "")
        except Exception:
            log.debug("voice: couldn't remember the voice channel", exc_info=True)

    async def _rejoin_voice(self) -> None:
        """Put the bot back where it was before it went down.

        A voice connection cannot survive os.execv or a crash, so every restart
        used to drop the bot out of the channel — and autojoin (which only fires
        when somebody *enters* voice) never brought it back for the people who
        were already sitting there. That reads as "the bot forgot voice control
        is on even though I'm opted in". Only for guilds that have voice control
        enabled, and only while somebody is still in the channel."""
        music = self.bot.get_cog("Music")
        if music is None or not hasattr(music, "autojoin"):
            return
        try:
            saved = await db.list_app_settings(_VOICE_CHANNEL_KEY)
        except Exception:
            log.exception("voice: couldn't read the remembered voice channels")
            return
        for key, value in saved.items():
            try:
                gid, cid = int(key[len(_VOICE_CHANNEL_KEY):]), int(value)
            except ValueError:
                continue
            if not self._cfg.get(gid, {}).get("enabled"):
                continue                    # voice control is off there now
            guild = self.bot.get_guild(gid)
            if guild is None or guild.voice_client is not None:
                continue                    # gone, or already connected elsewhere
            channel = guild.get_channel(cid)
            if channel is None or channel not in guild.voice_channels:
                continue                    # deleted, or not a plain voice channel
            if channel == guild.afk_channel:
                continue
            perms = channel.permissions_for(guild.me)
            if not (perms.connect and perms.speak):
                continue                    # can't go back in there
            member = next((m for m in channel.members if not m.bot), None)
            if member is None:
                log.info("Voice control: not rejoining %s in %s — nobody is in there",
                         channel.name, guild.name)
                continue
            try:
                if await music.autojoin(member, channel):
                    log.info("Voice control: rejoined %s in %s after the restart",
                             channel.name, guild.name)
            except Exception:
                log.exception("voice: rejoin into %s failed",
                              getattr(channel, "name", "?"))

    def _revoke(self, guild_id: int, user_id: int, reason: str) -> None:
        key = (guild_id, user_id)
        self._epoch[key] = self._epoch.get(key, 0) + 1
        if self._front is not None:
            self._front.drop_speaker(guild_id, user_id)
            self._front.set_end_silence(guild_id, user_id, None)
        self._jobs.purge(lambda j: j.key == key, reason)
        self._pending.pop(key, None)
        self._armed.pop(key, None)
        self._last_audio.pop(key, None)
        self._parts.pop(key, None)
        self._recent.pop(key, None)
        self._mine.pop(key, None)
        self._typing_stop(key)
        # Their song requests still being looked up are void too — silently:
        # they asked us to stop listening to them.
        self._cancel_plays(guild_id, reason=reason, user_id=user_id, notify=False)

    def _new_session(self, guild_id: int, reason: str) -> None:
        """The bot moved or left voice: nothing captured before this may act
        after it, and song requests still being looked up are called off — out
        loud. (A receive hiccup while still connected isn't this; see
        _on_receiver_session.)"""
        self._session[guild_id] = self._session.get(guild_id, 0) + 1
        if self._front is not None:
            self._front.drop_guild(guild_id)
        self._jobs.purge(lambda j: j.guild_id == guild_id, "session_changed")
        for key in [k for k in self._pending if k[0] == guild_id]:
            self._pending.pop(key, None)
        for key in [k for k in self._armed if k[0] == guild_id]:
            self._armed.pop(key, None)
        for store in (self._parts, self._recent, self._mine):
            for key in [k for k in store if k[0] == guild_id]:
                store.pop(key, None)
        for key in [k for k in self._typing if k[0] == guild_id]:
            self._typing_stop(key)
        # Say so: a song request that silently never happens is the worst kind.
        self._cancel_plays(guild_id, reason="session_changed")
        log.info("voice session %s → %d (%s)", guild_id, self._session[guild_id], reason)

    def _on_receiver_session(self, guild_id: int, event: str) -> None:
        """The voice *receiver* (audio in) started, stopped or recovered.

        With the bot still connected where it was, that's a receive hiccup —
        Discord's voice encryption (DAVE) drops one now and then and the
        recorder brings it back within a second or two. Speech still being
        captured is lost, but what was already heard stands: clips waiting for
        recognition, requests being worked out or carried out (song lookups,
        plans), open questions and wake windows. That used to start a new
        session, which silently called off every song lookup in flight — a
        "skip this and play …" came back as just the skip. A real change of
        place (the bot moved, left or was disconnected) still starts a new
        session, from on_voice_state_update / here when it's no longer
        connected."""
        if event not in ("stopped", "recovered", "started"):
            return
        guild = self.bot.get_guild(guild_id)
        vc = guild.voice_client if guild else None
        if vc is not None and vc.is_connected():
            if self._front is not None:
                self._front.drop_guild(guild_id)          # half-heard speech only
            self.metrics.count(guild_id, f"receiver_{event}")
            log.info("voice receiver %s in %s while connected — keeping what was "
                     "already heard", event, guild_id)
            return
        self._new_session(guild_id, f"receiver {event}")

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after) -> None:
        guild = member.guild
        if self.bot.user is not None and member.id == self.bot.user.id:
            if before.channel != after.channel:
                self._new_session(guild.id, "bot moved" if after.channel else "bot left")
                # Remember where we are: a restart can't carry the voice
                # connection, but it can put the bot back on its feet (see
                # _rejoin_voice).
                self._spawn(self._remember_voice_channel(guild.id, after.channel))
            return
        key = (guild.id, member.id)
        vc = guild.voice_client
        if vc is not None and before.channel == vc.channel and after.channel != vc.channel:
            # A speaker left the bot's channel: their half-said command and any
            # open question/window go with them.
            self._pending.pop(key, None)
            self._armed.pop(key, None)
            if self._front is not None:
                self._front.drop_speaker(guild.id, member.id)

    @tasks.loop(seconds=2.0)
    async def _maintenance(self) -> None:
        now = time.monotonic()
        rec = self.bot.get_cog("Recording")
        if rec is not None and self._subscribed_to is not rec:
            rec.subscribe(self._on_frame, self._on_receiver_session)
            self._subscribed_to = rec
        if now - self._last_refresh >= _REFRESH_EVERY:
            await self._refresh_all()
            self._prefetch_model()
        # Expire open questions / wake windows — but not while a reply that was
        # captured in time is still queued or being recognised (bounded by
        # process_by), so a slow CPU can't turn a timely "yes" into "no answer".
        for key, p in list(self._pending.items()):
            if now <= p.deadline + 1.0:
                continue
            waiting = self._timely_work(key, p)
            if waiting and now <= p.process_by:
                continue
            if self._pending.get(key) is not p:
                continue
            self._pending.pop(key, None)
            self._restore_end_silence(key)
            guild = self.bot.get_guild(key[0])
            member = guild.get_member(key[1]) if guild else None
            if member is not None:
                msg = ("⌛ I couldn't make out your answer in time — cancelled."
                       if waiting else "⌛ No answer — cancelled.")
                await self._deliver_feedback(guild, member, "", msg, force=True)
        for key, a in list(self._armed.items()):
            if now <= a.until + 1.0:
                continue
            if now <= a.process_by and self._timely_work(key, a):
                continue
            self._armed.pop(key, None)
            self._restore_end_silence(key)
        self._flush_stale_parts()
        for key, ev in list(self._fast_wakes.items()):
            if now - ev.at > 60:
                self._fast_wakes.pop(key, None)          # never transcribed (dropped, too short…)
        for key, onset in list(self._wake_confirmed.items()):
            if now - onset > 120:
                self._wake_confirmed.pop(key, None)
        for key, (at, _msg) in list(self._cue_msgs.items()):
            if now - at > 120:
                self._cue_msgs.pop(key, None)
        for gid in list(self._debug_until):
            if self._debug_until[gid] <= now:
                self._debug_until.pop(gid, None)
                self._debug_log.pop(gid, None)
        # Self-heal: voice on, bot connected, but nothing receiving.
        if rec is not None and self._whisper_ok:
            for gid, allowed in self._auth.items():
                if not allowed:
                    continue
                guild = self.bot.get_guild(gid)
                vc = guild.voice_client if guild else None
                if vc is None or not vc.is_connected() or not hasattr(vc, "is_listening"):
                    continue
                state = (rec.health.get(gid) or {}).get("state")
                if not vc.is_listening() and state not in ("recovering",):
                    try:
                        await rec.attach(vc, None)
                    except Exception:
                        log.exception("voice: receiver re-attach failed in %s", gid)

    @_maintenance.before_loop
    async def _before_maintenance(self) -> None:
        await self.bot.wait_until_ready()
        await self._refresh_all()
        self._log_restored()
        self._prefetch_model()
        await self._rejoin_voice()

    @_maintenance.error
    async def _maintenance_error(self, exc: BaseException) -> None:
        log.exception("voice maintenance loop crashed — restarting", exc_info=exc)
        try:
            self._maintenance.restart()
        except Exception:
            log.exception("voice maintenance restart failed")

    def _timely_work(self, key, context) -> bool:
        """Is speech this question/window accepts (by capture time) still
        being captured, queued or recognised?"""
        active = getattr(self._front, "active_speech", None)
        speech = active(key) if active is not None else None
        if speech is not None:
            onset, tag = speech
            identity = (context.session, self._epoch.get(key, 0))
            if tag == identity and context.accepts_onset(onset):
                return True
        cur = self._current_job
        if cur is not None and cur.key == key and context.accepts(cur):
            return True
        return any(j.key == key and context.accepts(j)
                   for j in [*self._jobs.snapshot(), *self._decision_jobs()])

    def _restore_end_silence(self, key) -> None:
        if self._front is not None and key not in self._pending and key not in self._armed:
            self._front.set_end_silence(key[0], key[1], None)

    # ──────────────────────────────────────────────────────────────────────────
    # Audio in → jobs
    # ──────────────────────────────────────────────────────────────────────────

    def _on_frame(self, guild_id, user_id, opus, seq, ts, ssrc, arrival) -> None:
        """RECEIVE THREAD. Consent is checked here, per frame: audio from
        anyone not currently opted in is never even decoded. The packet is
        stamped with the session and consent epoch it was admitted under; the
        segment it ends up in carries that stamp all the way to the job."""
        allowed = self._auth.get(guild_id)
        if not allowed or user_id not in allowed:
            return
        front = self._front
        if front is not None:
            tag = (self._session.get(guild_id, 0), self._epoch.get((guild_id, user_id), 0))
            front.feed(guild_id, user_id, opus, seq, ts, ssrc, arrival, tag=tag)

    def _segment_from_thread(self, seg) -> None:
        """FRONT-END THREAD → event loop."""
        try:
            self.bot.loop.call_soon_threadsafe(self._on_segment, seg)
        except RuntimeError:
            pass

    def _bot_speech_overlaps(self, guild_id: int, start: float, end: float) -> bool:
        tts_cog = self.bot.get_cog("TTS")
        if tts_cog is None:
            return False
        return any(start < b + _ECHO_TAIL and end > a
                   for a, b in tts_cog.speaking_windows(guild_id))

    def _on_segment(self, seg) -> None:
        gid, uid = seg.guild_id, seg.user_id
        key = (gid, uid)
        self.metrics.count(gid, "segments")
        if uid not in self._auth.get(gid, ()):
            self.metrics.drop(gid, "not_authorized")
            return
        current = (self._session.get(gid, 0), self._epoch.get(key, 0))
        tag = getattr(seg, "tag", None)
        session, epoch = tag if tag is not None else current
        # Judge the segment by the identity it was captured under. A segment
        # emitted just before the bot moved (or the speaker opted out) used to
        # be stamped with the NEW session when the loop got to it.
        if session != current[0]:
            self.metrics.drop(gid, "session_changed")
            self._typing_drop(key)
            return
        if epoch != current[1]:
            self.metrics.drop(gid, "revoked")
            self._typing_drop(key)
            return
        job = Job(gid, uid, seg.audio, seg.start, seg.end, seg.speech_ms,
                  session=session, epoch=epoch, overlap_ms=seg.overlap_ms,
                  segment_reason=seg.reason,
                  speech_start=getattr(seg, "speech_start", None))
        # One long utterance comes as several parts (cut at the length cap,
        # each overlapping the last): number them, so they're understood
        # together (see _collect_part).
        prev = self._utt.get(key)
        if (seg.overlap_ms and prev is not None and prev[2] == "max_length"
                and seg.start <= prev[1] + 1.0):
            job.utt, job.part = prev[0], prev[3] + 1
        else:
            self._utt_seq += 1
            job.utt, job.part = self._utt_seq, 0
        self._utt[key] = (job.utt, seg.end, seg.reason, job.part)
        if job.part and self._ignored_utts.get(key) == job.utt:
            self.metrics.drop(gid, "not_addressed")      # the rest of something not for us
            return
        pending = self._pending.get(key)
        armed = self._armed.get(key)
        if pending and pending.session == session and pending.accepts(job):
            job.priority = PRIORITY_CONFIRM
        elif armed and armed.session == session and armed.accepts(job):
            job.priority = PRIORITY_ARMED
        elif job.part:
            job.priority = PRIORITY_ARMED      # the rest of a sentence: never superseded
        else:
            job.priority = PRIORITY_NORMAL
        if (job.priority != PRIORITY_NORMAL and not job.part
                and self._bot_speech_overlaps(gid, job.onset, job.end)):
            # No wake word is needed in these windows, so a reply captured while
            # the bot was talking could be the bot itself, echoing through
            # someone's speakers. (Judged from the speech onset: the clip's
            # pre-roll can reach back into the bot's last words.)
            self.metrics.drop(gid, "bot_speaking")
            self._typing_drop(key)
            return
        if (job.priority == PRIORITY_NORMAL and seg.speech_ms < _MIN_COMMAND_SPEECH_MS
                and not job.part and not self._may_follow_a_wake(key, job.onset)):
            self.metrics.drop(gid, "too_short")
            return
        self.metrics.count(gid, "queued")
        self._jobs.put(job)

    def _may_follow_a_wake(self, key, onset: float) -> bool:
        """A short utterance ("skip") right after the same person's previous
        one, while that one is still being recognised: it may be the command
        after a wake-only "hey DJ" whose window doesn't exist yet. Keep it;
        it's judged against the window by capture time once that's known."""
        jobs = self._jobs.snapshot()
        jobs.extend(self._decision_jobs())
        if self._current_job is not None:
            jobs.append(self._current_job)
        # (A little negative slack: segment edges are estimates and can touch.)
        return any(j.key == key and -0.25 <= onset - j.end <= _ARM_SECONDS for j in jobs)

    def _on_job_dropped(self, job: Job, reason: str) -> None:
        self.metrics.drop(job.guild_id, reason)
        if job.whole:
            # The whole couldn't be transcribed once more: go by its parts'
            # joined words instead of losing the request.
            stitched = self._whole_fallback.pop((job.key, job.utt), None)
            if stitched is not None:
                self._decide_later(job, stitched)
                return
        self._typing_drop(job.key, job)

    @staticmethod
    def _max_age(job: Job) -> float:
        """How old speech may be when it's acted on. A long request whose end
        never came was held by us, waiting for it — not by them."""
        if job.segment_reason == "joined_incomplete":
            return _DISPATCH_MAX_AGE + _PARTS_WAIT
        return _DISPATCH_MAX_AGE

    def _decide_later(self, job: Job, text: str) -> None:
        """Understand `text` for `job` in this person's decision order."""
        from core.whisper import AsrResult
        queue = self._decisions.setdefault(job.key, deque())
        queue.append((job, AsrResult(text, 0.0, 0.0, max(0.0, job.end - job.start))))
        worker = self._decision_workers.get(job.key)
        if worker is None or worker.done():
            self._decision_workers[job.key] = self._spawn(self._decide_for_speaker(job.key))

    # ──────────────────────────────────────────────────────────────────────────
    # Jobs → text
    # ──────────────────────────────────────────────────────────────────────────

    def _job_valid(self, job: Job) -> str | None:
        """None if the job may proceed, else the reason it may not."""
        if job.user_id not in self._auth.get(job.guild_id, ()):
            return "revoked"
        if self._epoch.get(job.key, 0) != job.epoch:
            return "revoked"
        if self._session.get(job.guild_id, 0) != job.session:
            return "session_changed"
        guild = self.bot.get_guild(job.guild_id)
        vc = guild.voice_client if guild else None
        if vc is None or not vc.is_connected():
            return "disconnected"
        member = guild.get_member(job.user_id)
        if member is None or member.voice is None or member.voice.channel != vc.channel:
            return "speaker_left"
        return None

    def _asr_options(self, guild_id: int) -> AsrOptions:
        prompt = None
        if settings.voice_asr_hints:
            wakes = self._cfg.get(guild_id, {}).get("wake") or [settings.voice_wake_word]
            prompt = ", ".join(wakes) + "."
        return AsrOptions(language=settings.voice_language or None,
                          beam_size=settings.voice_beam_size, prompt=prompt)

    async def _asr_worker(self) -> None:
        while True:
            job = await self._jobs.get()
            self._current_job = job
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("voice job failed")
                self.metrics.count(job.guild_id, "errors")
            finally:
                self._current_job = None

    async def _run_job(self, job: Job) -> None:
        gid = job.guild_id
        if job.part and self._ignored_utts.get(job.key) == job.utt:
            self.metrics.drop(gid, "not_addressed")
            return
        reason = self._job_valid(job)
        if reason:
            self.metrics.drop(gid, reason)
            self._typing_drop(job.key, job)
            return
        if job.priority == PRIORITY_NORMAL and time.monotonic() - job.end > _ASR_MAX_AGE:
            self.metrics.drop(gid, "stale_before_asr")
            self._typing_drop(job.key, job)
            return
        model = await self._models.get(_MODEL_WAIT)
        if model is None:
            self._on_job_dropped(job, f"model_{self._models.state}")
            return
        probe = self._probe_task
        if probe is not None and not probe.done():
            # An early wake check is using the recogniser: wait for it (it's
            # short) rather than have the runner refuse this as "busy".
            try:
                await asyncio.wait_for(asyncio.shield(probe), _TRANSCRIBE_TIMEOUT)
            except Exception:
                pass
        try:
            result = await self._asr.run(transcribe_command, model, job.audio,
                                         self._asr_options(gid))
        except AsrBusy:
            self._on_job_dropped(job, "asr_busy")        # (a long request's whole: its parts' words)
            return
        except AsrTimeout:
            self._on_job_dropped(job, "asr_timeout")
            return
        except Exception:
            log.exception("voice recognition failed")
            self.metrics.count(gid, "errors")
            self._on_job_dropped(job, "asr_error")
            return
        self.metrics.guild(gid).asr_seconds.append(self._asr.last_seconds or 0.0)
        self.metrics.count(gid, "transcribed")
        self._remember_audio(job)
        reason = self._job_valid(job)          # consent/session again, after ASR
        if reason:
            self.metrics.drop(gid, reason)
            self.metrics.asr(gid, result, f"dropped:{reason}")
            self._typing_drop(job.key, job)
            return
        # AI and TTS can wait on the network. Release the single recogniser
        # now; another speaker's skip must not wait for a chat answer.
        queue = self._decisions.setdefault(job.key, deque())
        if len(queue) >= 3 or sum(map(len, self._decisions.values())) >= 8:
            self.metrics.drop(gid, "decision_queue_full")
            self.metrics.asr(gid, result, "dropped:decision_queue_full")
            self._typing_drop(job.key, job)
            return
        queue.append((job, result))
        worker = self._decision_workers.get(job.key)
        if worker is None or worker.done():
            self._decision_workers[job.key] = self._spawn(self._decide_for_speaker(job.key))

    def _decision_jobs(self) -> list[Job]:
        return [job for queue in self._decisions.values() for job, _ in queue]

    async def _decide_for_speaker(self, key) -> None:
        """Keep one person's wake/reply order, independently of other speakers."""
        queue = self._decisions[key]
        try:
            while queue:
                job, result = queue[0]
                try:
                    reason = self._job_valid(job)
                    if reason:
                        self.metrics.drop(job.guild_id, reason)
                        outcome = f"dropped:{reason}"
                        self._typing_drop(key, job)
                    else:
                        outcome = None
                        for whole, text in self._collect_part(job, result.text):
                            outcome = await self._handle_transcript(whole, text)
                        if outcome is None:
                            outcome = "part"                   # more of it is coming
                    self.metrics.asr(job.guild_id, result, outcome or "?")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("voice decision failed")
                    self.metrics.count(job.guild_id, "errors")
                    self._typing_stop(key)
                finally:
                    queue.popleft()
        finally:
            self._decisions.pop(key, None)
            self._decision_workers.pop(key, None)

    def _addressed(self, job: Job, text: str) -> bool:
        """Is this (first part of an) utterance for us: does it open with the
        wake word, or come while we wait for this person's answer?"""
        wakes = self._cfg.get(job.guild_id, {}).get("wake") or [settings.voice_wake_word]
        if text_after_wake(text, wakes) is not None:
            return True
        for ctx in (self._pending.get(job.key), self._armed.get(job.key)):
            if ctx is not None and ctx.session == job.session and ctx.accepts(job):
                return True
        return False

    def _collect_part(self, job: Job, text: str) -> list:
        """[(job, text) to understand now]. A part cut at the length cap that
        is addressed to us is held until the utterance ends; then its parts
        are understood as one request, so a later "…, but only after X"
        still governs the beginning, and the overlap's repeated words aren't
        taken twice. (Not for us: understood — i.e. ignored — as before, and
        its later parts aren't even transcribed.)"""
        key = job.key
        out: list = []
        if job.part and not job.whole and self._ignored_utts.get(key) == job.utt:
            self.metrics.drop(job.guild_id, "not_addressed")     # the rest of something not for us
            return out
        if job.whole:
            stitched = self._whole_fallback.pop((key, job.utt), None)
            if stitched is not None and (not text.strip() or (
                    not self._addressed(job, text) and self._addressed(job, stitched))):
                # Heard once more, the opening came out differently ("and DJ"
                # for "hey DJ") or not at all: the parts were for us, so go by
                # their words rather than ignore the request.
                self.metrics.count(job.guild_id, "long_request_whole_unused")
                return [(job, stitched)]
            return [(job, text)]
        parts = self._parts.get(key)
        if parts is not None and job.utt != parts.utt:
            out.append(self._join_parts(key, complete=False))   # its last part never came
            parts = None
        continues = job.segment_reason == "max_length"
        if parts is None:
            if not continues:
                out.append((job, text))
                return out
            if not self._addressed(job, text):
                self._ignored_utts[key] = job.utt
                out.append((job, text))
                return out
            self._parts[key] = _Parts(job.utt, [job], [text], time.monotonic())
            self.metrics.count(job.guild_id, "long_request_parts")
            self._spawn(self._greet_long_request(job, text))
            return out
        parts.jobs.append(job)
        parts.texts.append(text)
        parts.since = time.monotonic()
        if continues and len(parts.jobs) < _MAX_PARTS:
            return out
        whole, stitched = self._join_parts(key, complete=not continues)
        if self._transcribe_whole(whole, stitched):
            return out                      # understood when its own transcript is back
        out.append((whole, stitched))
        return out

    def _transcribe_whole(self, whole: Job, stitched: str) -> bool:
        """Queue one more recognition of a long request's whole audio. Each
        part was transcribed on its own, cut at the length cap: a name said
        across the cut comes out twice, differently ("the Phoenix by fall off?
        Phoenix by Fallout Boy" in a real recording) — the whole, at once,
        doesn't have that seam. The joined parts' words are kept to fall back
        on if this can't run."""
        if whole.audio is None or len(whole.audio) < 16_000:
            return False
        whole.whole = True
        whole.priority = PRIORITY_CONFIRM
        self._whole_fallback[(whole.key, whole.utt)] = stitched
        self.metrics.count(whole.guild_id, "long_request_rejoined")
        self._jobs.put(whole)
        return True

    def _join_parts(self, key, *, complete: bool) -> tuple:
        from core.voice_intent import stitch
        parts = self._parts.pop(key)
        text = parts.texts[0]
        for more in parts.texts[1:]:
            text = stitch(text, more)
        first, last = parts.jobs[0], parts.jobs[-1]
        audio = None
        try:
            import numpy as np
            chunks = [first.audio] + [j.audio[int(j.overlap_ms * 16):] for j in parts.jobs[1:]]
            if all(c is not None for c in chunks):
                audio = np.concatenate(chunks)
        except Exception:
            audio = None
        whole = Job(first.guild_id, first.user_id, audio, first.start, last.end,
                    sum(j.speech_ms for j in parts.jobs), session=first.session,
                    epoch=first.epoch, priority=first.priority, overlap_ms=0,
                    segment_reason="joined" if complete else "joined_incomplete",
                    speech_start=first.speech_start, utt=first.utt, part=0)
        self.metrics.count(first.guild_id, "long_requests" if complete
                           else "long_requests_incomplete")
        return whole, text

    async def _greet_long_request(self, job: Job, text: str) -> None:
        """The first part of a long request: the cue (if the instant wake
        word hasn't given it already) and "typing…" while they finish."""
        wakes = self._cfg.get(job.guild_id, {}).get("wake") or [settings.voice_wake_word]
        if text_after_wake(text, wakes) is not None:
            self._spawn(self._wake_cue(job.guild_id, job.user_id, job.speech_start))
        self._settle_fast_wake(job, text, wakes)
        guild = self.bot.get_guild(job.guild_id)
        member = guild.get_member(job.user_id) if guild else None
        if member is not None:
            await self._typing_start(guild, member, reason="listening", limit=_PARTS_WAIT + 10)

    def _flush_stale_parts(self) -> None:
        """A long request whose next part never came (dropped, or the speaker
        left): understood as far as it got — through the speaker's own
        decision queue, so it stays in order."""
        now = time.monotonic()
        for key, parts in list(self._parts.items()):
            if now - parts.since < _PARTS_WAIT:
                continue
            whole, text = self._join_parts(key, complete=False)
            self._decide_later(whole, text)

    def _remember_audio(self, job: Job) -> None:
        """Keep the requester's own last clip (in memory) for `voice clip`."""
        try:
            import numpy as np
            pcm = (np.clip(job.audio, -1, 1) * 32767).astype("<i2").tobytes()
        except Exception:
            return
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16_000)
            wf.writeframes(pcm)
        self._last_audio[job.key] = (job.end, buf.getvalue())

    # ──────────────────────────────────────────────────────────────────────────
    # Text → decision
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_transcript(self, job: Job, text: str) -> str:
        """Decide what a transcript means. Returns the outcome label (also
        used for the ASR-quality diagnostics)."""
        gid, key = job.guild_id, job.key
        guild = self.bot.get_guild(gid)
        member = guild.get_member(job.user_id) if guild else None
        if member is None:
            self._typing_stop(key)
            return "no_member"
        wakes = self._cfg.get(gid, {}).get("wake") or [settings.voice_wake_word]
        debug = self._debug_on(gid)

        def note(result: str) -> str:
            self.metrics.guild(gid).parse[result] += 1
            if debug:
                log.info("voice heard from %s: %r → %s", member.display_name, text, result)
                self._debug_log.setdefault(gid, []).append((member.display_name, text, result))
                del self._debug_log[gid][:-20]
            return result

        if not text:
            self._typing_stop(key)
            return note("empty")

        # Backstop for the early probe: whenever an utterance turns out to open
        # with the wake word, greet it. The probe is skipped when the box is
        # busy or recognition is slow, and it only sees the first couple of
        # seconds — filler-heavy openings ("um, so, hey DJ …") can fall outside
        # it. Spawned so the cue can never delay what the command does.
        if text_after_wake(text, wakes) is not None:
            self._spawn(self._wake_cue(gid, job.user_id, job.speech_start))
        self._settle_fast_wake(job, text, wakes)

        pending = self._pending.get(key)
        if (pending and pending.session == job.session and pending.accepts(job)
                and time.monotonic() <= pending.process_by):
            answer = parse_confirmation(text, wakes, action=pending.confirm_action)
            if answer == "yes":
                self._pending.pop(key, None)
                self._restore_end_silence(key)
                label = note(f"confirmed {pending.confirm_action}")
                if pending.plan is not None:
                    self._enqueue_plan(job, pending.plan, text, view=pending.view)
                else:
                    self._enqueue_steps(job, (pending.intent, *pending.then), text)
                return label
            if answer == "no":
                self._pending.pop(key, None)
                self._restore_end_silence(key)
                label = note("declined")
                await self._deliver_feedback(guild, member, text, "✖ Cancelled.",
                                             chime="cancel")
                return label
            # Neither: the question stays open; still parse it as a command.

        armed = self._armed.get(key)
        in_window = bool(armed and armed.session == job.session and armed.accepts(job)
                         and time.monotonic() <= armed.process_by)
        question = armed.question if (in_window and armed is not None) else ""
        result = interpret(text, wakes, in_window=in_window)
        if self._wants_llm(gid, result, in_window) or (question and result.reason != "wake_only"):
            state = get_state(gid)
            queue_epoch = state.queue_epoch
            view = vx.View.capture(state)                  # what the planner is shown
            plan = await self._ask_planner(guild, member, text, wakes, result, view, question)
            gone = self._job_valid(job)        # that took a moment: still allowed, still here?
            if not gone and not self._cfg.get(gid, {}).get("ai", True):
                gone = "ai_disabled"
            if not gone and time.monotonic() - job.end > self._max_age(job):
                gone = "stale_after_ai"
            if gone:
                self.metrics.drop(gid, gone)
                self._typing_stop(key)
                if gone == "stale_after_ai":
                    await self._deliver_feedback(
                        guild, member, text, "⌛ That took me too long to work out to act on it "
                        "safely — say it again.", chime="cancel")
                return note(f"dropped:{gone}")
            acts = plan is not None and plan.kind == "plan" and bool(plan.steps)
            if (state.queue_epoch != queue_epoch
                    and (acts or (plan is None and result.intent is not None))):
                # Someone stopped the music or cleared the queue while this
                # was being worked out: don't undo that — but say so.
                self.metrics.drop(gid, "queue_changed")
                changed = ("the queue was cleared" if state.queue_change_reason == "cleared"
                           else "the music was stopped")
                await self._deliver_feedback(
                    guild, member, text, f"✖ Didn't do that — {changed} while I was working "
                    "it out. Say it again if you still want it.", chime="cancel")
                return note("dropped:queue_changed")
            if plan is not None and plan.kind == "plan" and (plan.steps or plan.ask
                                                             or plan.problems):
                self._armed.pop(key, None)
                self._restore_end_silence(key)
                label = note("→ plan " + (" + ".join(st.op for st in plan.steps) or "(ask)"))
                for st in plan.steps:
                    self.metrics.stage(gid, st.op, "recognized")
                await self._start_plan(job, plan, text, view, guild, member)
                return label
            if plan is not None and plan.kind == "answer" and result.intent is not None:
                plan = None          # a command the grammar heard is never turned into chat
            if plan is not None and plan.kind == "answer":
                self._armed.pop(key, None)
                self._restore_end_silence(key)
                if self._cfg.get(gid, {}).get("chat", True) or plan.ask:
                    label = note("→ question")
                    await self._answer(guild, member, text, plan.reply or plan.ask)
                    if plan.ask:
                        self._open_question(job, plan.ask)
                    return label
                result = ParseResult(None, "chat_off", result.wake, result.wake_at_start,
                                     result.command)
            elif plan is not None:
                if result.intent is None or result.list_like:
                    result = ParseResult(None, "not_for_me", result.wake, result.wake_at_start,
                                         result.command)
                # (a control the grammar was sure of, which the planner found
                # nothing in: the grammar's reading stands)
            elif result.intent is None:
                label = note("llm_failed")
                await self._deliver_feedback(
                    guild, member, text, "🤷 I didn't get that, and the AI couldn't answer "
                    "just now — try again.", chime="cancel")
                return label
        label = note(result.reason if result.intent is None
                     else "→ " + " + ".join(i.action for i in result.intents))

        if result.reason == "wake_only":
            self._armed[key] = _Armed(since=job.end, until=job.end + _ARM_SECONDS,
                                      session=job.session)
            if self._front is not None:
                self._front.set_end_silence(gid, job.user_id, _WINDOW_END_SILENCE_MS)
            # Only as long as the window: a "hey DJ" with nothing after it
            # shouldn't leave "typing…" up. A follow-up extends it.
            await self._typing_start(guild, member, reason="listening",
                                     limit=_ARM_SECONDS + 2.0)
            await self._deliver_feedback(guild, member, text, None, chime="listening")
            return label
        intent = result.intent
        if intent is None:
            self._typing_stop(key)
            return label
        for step in result.intents:
            self.metrics.stage(gid, step.action, "recognized")
        self._armed.pop(key, None)
        self._restore_end_silence(key)

        last = self._last_action.get(key)
        if (job.overlap_ms and last and last[0] == intent
                and job.start - last[1] < _DEDUPE_SECONDS):
            self.metrics.drop(gid, "duplicate_from_split")
            self._typing_stop(key)
            return label
        steps = tuple(i for i in result.intents if i.action not in ("confirm", "cancel"))
        if not steps:
            self._typing_stop(key)
            return label                              # nothing was asked
        if any(i.needs_confirmation for i in steps):
            asked = time.monotonic()
            pending = _Pending(steps[0], asked_at=asked, deadline=asked + _CONFIRM_SECONDS,
                               session=job.session, then=steps[1:])
            self._pending[key] = pending
            if self._front is not None:
                self._front.set_end_silence(gid, job.user_id, _WINDOW_END_SILENCE_MS)
            what = ", then ".join(d if n == 0 else d[0].lower() + d[1:]
                                  for n, d in enumerate(self._describe(i) for i in steps))
            # Always asked in text — even to someone on `silent`.
            await self._deliver_feedback(guild, member, text,
                                         f"⚠️ {what}? Say “yes” or “no”.", force=True)
            # The reply window runs from when the question was actually out,
            # not from before a possibly slow send.
            if self._pending.get(key) is pending:
                pending.deadline = max(pending.deadline, time.monotonic() + _CONFIRM_SECONDS)
            return label
        self._last_action[key] = (intent, job.end)
        self._enqueue_steps(job, steps, text)           # song lookups show "typing…"
        return label

    # ── tier 2: the LLM ──────────────────────────────────────────────────────

    def _wants_llm(self, guild_id: int, result: ParseResult, in_window: bool) -> bool:
        """Only after a wake word (or in the "hey DJ" … window), for what
        needs understanding rather than matching: no command recognised, a
        "don't"/"stop" the grammar won't guess about ("stop the music and
        clear the queue"), and every song request — "play Panic! At The
        Disco's first three songs off Vices and Virtues" must become an album
        lookup, not a YouTube search for that sentence. Controls the grammar
        is sure of ("skip", "louder") stay instant and offline."""
        if self.llm is None or not self._cfg.get(guild_id, {}).get("ai", True):
            return False
        if result.intent is not None:
            return result.list_like or any(i.action in SONG_ACTIONS for i in result.intents)
        return (result.reason in ("no_command", "negated")
                or (in_window and result.reason == "no_wake"))

    async def _ask_planner(self, guild, member, text: str, wakes, result: ParseResult,
                           view: "vx.View", question: str = "") -> "vp.Plan | None":
        """The planner (core/voice_llm.py LLMRouter.plan): the words after
        the wake word, what is playing and queued (with entry ids it can point
        at), this person's recent requests, and a question we asked them."""
        said = text_after_wake(text, wakes) if result.wake_at_start else None
        if said is None:
            said = text            # a follow-up in the "hey DJ" … window has no wake word
        if not said.strip():
            return None
        await self._typing_start(guild, member, reason="ai")
        self.metrics.count(guild.id, "llm_asked")
        key = (guild.id, member.id)
        ctx = plan_context(current=view.current, upcoming=view.upcoming,
                           player=self._player_line(guild), recent=self._recent_lines(key),
                           question=question)
        plan = await self.llm.plan(said, ctx, lookup=self._plan_lookup,
                                   model=settings.voice_plan_model or None,
                                   fallbacks=settings.voice_plan_fallbacks or None,
                                   timeout=settings.voice_plan_timeout)
        self.metrics.count(guild.id, "llm_failed" if plan is None else f"llm_{plan.kind}")
        if plan is not None and self._debug_on(guild.id):
            log.info("voice plan for %s: %s", member.display_name, vp.describe(plan))
        return plan

    def _player_line(self, guild) -> str:
        """The player in a few words, for the planner."""
        state = get_state(guild.id)
        vc = getattr(guild, "voice_client", None)
        if vc is not None and voice_mix.music_is_paused(vc):
            what = "paused"
        elif state.current is not None:
            what = "playing"
        else:
            what = "idle"
        loop = queue_ops.loop_mode(state)
        return (f"{what}, volume {round(state.volume * 100)}%, loop "
                + {"song": "this song", "queue": "the whole queue", "off": "off"}[loop])

    async def _plan_lookup(self, tool: str, title: str, artist: str) -> str:
        """Facts the planner asked for before planning (read-only): an album's
        track list, or an artist's songs."""
        try:
            if tool == "album":
                album = await asyncio.wait_for(music_catalog.find_album(title, artist),
                                               _CATALOG_TIMEOUT)
                if album is None:
                    return f"album {title!r} by {artist or '?'}: not found in the catalogue"
                return (f"album {album.title!r} by {album.artist} ({len(album.tracks)} tracks): "
                        + "; ".join(f"{t.position or n}. {t.title}"
                                    for n, t in enumerate(album.tracks, 1)))
            if tool == "songs":
                songs = await asyncio.wait_for(music_catalog.artist_songs(artist), _CATALOG_TIMEOUT)
                if not songs:
                    return f"songs by {artist!r}: none found in the catalogue"
                return f"songs by {songs[0].artist} (most popular first): " + "; ".join(
                    t.title for t in songs[:40])
        except (music_catalog.CatalogError, asyncio.TimeoutError) as e:
            return f"{tool} lookup failed ({type(e).__name__}): the catalogue isn't answering"
        return f"unknown lookup {tool!r}"

    # ── conversation context: what this person asked recently ────────────────

    _RECENT_MAX = 5
    _RECENT_SECONDS = 600.0

    def _recent_lines(self, key) -> list[str]:
        now = time.monotonic()
        out = []
        for at, said, what in self._recent.get(key, ()):
            if now - at <= self._RECENT_SECONDS:
                out.append(f"{int(now - at)} s ago: “{said[:160]}” → {what}")
        return out

    def _remember_request(self, key, said: str, what: str) -> None:
        d = self._recent.setdefault(key, deque(maxlen=self._RECENT_MAX))
        d.append((time.monotonic(), " ".join(said.split()), what or "nothing"))

    def _open_question(self, job: Job, question: str, seconds: float = 25.0) -> None:
        """We asked them something: their next words, wake word or not, are
        the answer — for `seconds`, and the planner is told the question."""
        now = time.monotonic()
        self._armed[job.key] = _Armed(since=now - 0.1, until=now + seconds,
                                      session=job.session, question=question)
        if self._front is not None:
            self._front.set_end_silence(job.guild_id, job.user_id, _WINDOW_END_SILENCE_MS)

    def _now_playing(self, guild_id: int) -> Optional[str]:
        """What's playing, as context for the LLM, in plain words: "Fireflies
        by Owl City". (With an em dash here the fallback model was seen
        derailing into broken JSON on 2026-10-11.) YouTube's "- Topic" channel
        suffix is dropped."""
        cur = get_state(guild_id).current
        if not cur:
            return None
        title = (cur.get("title") or "").replace("—", "-").replace("–", "-").strip()
        by = cur.get("artist") or cur.get("uploader") or ""
        by = re.sub(r"\s*-\s*Topic$", "", by.replace("—", "-").replace("–", "-")).strip()
        if by and by.lower() not in title.lower():
            return f"{title} by {by}"
        return title or None

    async def _answer(self, guild, member, heard: str, reply: str) -> None:
        """A question, answered: said out loud (over the music, ducked) and
        posted, so it can be read too."""
        tts_cog = self._tts()
        spoken = False
        if tts_cog is not None:
            try:
                spoken = await tts_cog.say(guild, reply)
            except Exception:
                log.debug("voice answer TTS failed", exc_info=True)
        self.metrics.count(guild.id, "answers_spoken" if spoken else "answers_text_only")
        await self._post_text(guild, member, heard, f"💬 {reply}")

    # ── "typing…" while a request is being worked on ─────────────────────────

    def _typing_active(self, key) -> _Typing | None:
        t = self._typing.get(key)
        return t if t is not None and t.task is not None and not t.task.done() else None

    # ── the wake-word cue: "I heard you" ─────────────────────────────────────

    def _cue_on(self, guild_id: int) -> bool:
        """Is the cue active at all here? (host master switch — the message)"""
        return bool(settings.voice_cue)

    def _cue_sound_on(self, guild_id: int) -> bool:
        """Is the *sound* part of the cue on? (host + per-server switch)"""
        return (bool(settings.voice_cue) and bool(settings.voice_cue_sound)
                and bool(self._cfg.get(guild_id, {}).get("cue", True)))

    def _cue_line(self, guild_id: int, *, short: bool = False) -> str:
        """Human-readable state of the cue, for `voice cue` and `voice status`."""
        if not settings.voice_cue:
            return "off — host" if short else "⚪ off — the host has `VOICE_CUE` disabled"
        if not self._cfg.get(guild_id, {}).get("cue", True):
            return ("message only — sound off here" if short else
                    "🔕 message only — the sound is off for this server "
                    "(`voice cue on` to bring it back)")
        if not settings.voice_cue_sound:
            return ("message only — host" if short else
                    "🔕 message only — the sound is off on the host "
                    "(`VOICE_CUE_SOUND=false`)")
        return "🔔 sound + message" if short else "🔔 on — a short sound plus the message"

    async def _wake_cue(self, guild_id: int, user_id: int, onset: float | None, *,
                        timing: CueTiming | None = None) -> None:
        """The wake word was just heard: tell the speaker so, the way a smart
        speaker's light turns on — a short sound in the voice channel (unless
        it's turned off) and `:ear:*Now listening...*` in the reply channel.

        Once per utterance: the early probe and the finished transcript both
        report the same utterance and carry the same VAD onset, so whichever
        gets here second is swallowed."""
        key = (guild_id, user_id)
        at = time.monotonic() if onset is None else onset
        last = self._cue_onset.get(key)
        if last is not None and abs(at - last) <= _CUE_SAME_UTTERANCE:
            return
        self._cue_onset[key] = at
        if not self._cue_on(guild_id):
            return
        guild = self.bot.get_guild(guild_id)
        if guild is None:
            return
        self.metrics.count(guild_id, "wake_cue")
        if self._cue_sound_on(guild_id):
            # Deliberately not self._tts(): a sound effect doesn't need the
            # speech engine (edge-tts) to be available.
            tts_cog = self.bot.get_cog("TTS")
            if tts_cog is not None and hasattr(tts_cog, "cue"):
                try:
                    timing = timing or CueTiming(onset, tier="transcript")
                    if tts_cog.cue(guild, settings.voice_cue_file,
                                   volume=settings.voice_cue_volume, timing=timing):
                        self.metrics.guild(guild_id).cue_timings.append(timing)
                except Exception:
                    log.debug("voice: wake cue sound failed", exc_info=True)
        try:
            channel = await self._echo_channel(guild)
            if channel is not None:
                msg = await channel.send(_CUE_MESSAGE)
                if msg is not None:
                    self._cue_msgs[key] = (at, msg)     # a false alarm takes it back
                # Discord clears "<bot> is typing…" when the bot posts; if this
                # person's request is still being worked on, put it back now
                # rather than at the next 8 s ping.
                if self._typing_active(key) is not None and hasattr(channel, "typing"):
                    await channel.typing()
        except Exception:
            log.debug("voice: wake cue message failed", exc_info=True)

    async def _typing_start(self, guild, member, *, reason: str,
                            limit: float = _TYPING_MAX) -> None:
        """Show "<bot> is typing…" in the reply channel until this person's
        answer is posted, for at most `limit` seconds from now (a running
        indicator is extended instead). Only where a text reply follows."""
        if not settings.voice_typing or guild is None or member is None:
            return
        key = (guild.id, member.id)
        identity = (self._session.get(guild.id, 0), self._epoch.get(key, 0),
                    self._typing_generation.get(key, 0))

        def allowed():
            vc = guild.voice_client
            return (member.id in self._auth.get(guild.id, ())
                    and identity == (self._session.get(guild.id, 0), self._epoch.get(key, 0),
                                     self._typing_generation.get(key, 0))
                    and vc is not None and vc.is_connected()
                    and member.voice is not None and member.voice.channel == vc.channel)

        if not allowed():
            return
        until = time.monotonic() + limit
        current = self._typing_active(key)
        if current is not None:
            current.until = max(current.until, until)
            return
        try:
            mode = await self._feedback_for(guild.id, member.id)
        except Exception:
            mode = "text"
        if mode not in ("text", "chime") and reason != "ai":
            return                       # "ai": a question's answer is posted for everyone
        channel = await self._echo_channel(guild)
        if not allowed() or channel is None or not hasattr(channel, "typing"):
            return
        current = self._typing_active(key)              # re-check after the awaits
        if current is not None:
            current.until = max(current.until, until)
            return
        entry = _Typing(until)
        self._typing[key] = entry
        entry.task = self._spawn(self._typing_loop(key, entry, channel))
        self.metrics.count(guild.id, f"typing_{reason}")
        await asyncio.sleep(0)            # the first ping goes out before the slow part starts

    async def _typing_loop(self, key, entry: _Typing, channel) -> None:
        try:
            while time.monotonic() < entry.until:
                try:
                    await channel.typing()             # one ping shows for ~10 s
                except Exception:
                    return
                await asyncio.sleep(min(_TYPING_EVERY, max(0.0, entry.until - time.monotonic())))
        finally:
            if self._typing.get(key) is entry:
                self._typing.pop(key, None)

    def _typing_stop(self, key) -> None:
        self._typing_generation[key] = self._typing_generation.get(key, 0) + 1
        t = self._typing.pop(key, None)
        if t is not None and t.task is not None and not t.task.done():
            t.task.cancel()

    def _typing_drop(self, key, job: Job | None = None) -> None:
        """Speech was dropped before it was understood: end its "typing…"
        unless more from the same person is still on its way."""
        if key not in self._typing:
            return
        cur = self._current_job
        if ((cur is not None and cur is not job and cur.key == key)
                or any(j.key == key for j in [*self._jobs.snapshot(), *self._decision_jobs()]
                       if j is not job)):
            return
        self._typing_stop(key)

    # ── the instant wake word: a tiny model that only knows "hey DJ" ────────

    @staticmethod
    async def _load_wake_model():
        """core/wake_spotter.py's models, or None (then only speech-to-text
        spots the wake word, as before)."""
        try:
            from core.wake_spotter import WakeModel
            if not WakeModel.available():
                log.warning("Voice control: instant wake word unavailable (assets/wakeword "
                            "missing) — the cue comes from speech-to-text only")
                return None

            def load():
                m = WakeModel()
                m.initial_state()                      # warm: the first stream needs it
                return m
            model = await asyncio.to_thread(load)
            log.info("Voice control: instant wake word ready (threshold %.2f)",
                     settings.voice_fast_wake_threshold or model.threshold)
            return model
        except Exception:
            log.exception("Voice control: instant wake word failed to load")
            return None

    async def _load_wake_verifier(self) -> None:
        """The speech recogniser's second opinion on each fire — its own
        instance of the listener's model (VOICE_FAST_WAKE_VERIFY_MODEL to pick
        another), two threads, so it never waits behind anyone's sentence."""
        name = settings.voice_fast_wake_verify_model or self._models.name
        try:
            from core.wake_spotter import WakeVerifier

            def build():
                import numpy as np
                model = load_model(name, device=settings.voice_device,
                                   compute_type=settings.voice_compute_type, cpu_threads=2)
                from core.wake_spotter import PHRASE, check_words
                window = settings.voice_fast_wake_verify_window
                lang = settings.voice_language or None

                def heard(m, audio):
                    return check_words(m, audio, window=window, language=lang,
                                       full=settings.voice_fast_wake_verify_full,
                                       accept=lambda t: text_after_wake(t, [PHRASE]) is not None)
                for _ in range(2):
                    heard(model, np.zeros(16000, np.float32))       # warm it up
                return WakeVerifier(model, heard)
            self.wake_verifier = await asyncio.to_thread(build)
            w = settings.voice_fast_wake_verify_window
            log.info("Voice control: instant wake word verifier ready (%s, %s)", name,
                     (f"{w:g} s window" + (", then 10 s" if w < 10 else "")
                      + (", then the 30 s pass" if settings.voice_fast_wake_verify_full else ""))
                     if w else "30 s pass")
        except Exception:
            log.exception("Voice control: the instant wake word's verifier failed to load "
                          "— the cue comes from speech-to-text only")

    def _fast_wake_on(self, guild_id: int) -> bool:
        """The bundled model knows "hey DJ" only: not for servers that changed
        their wake words away from it."""
        from core.wake_spotter import PHRASE
        wakes = self._cfg.get(guild_id, {}).get("wake") or [settings.voice_wake_word]
        return self.wake_model is not None and PHRASE in {normalize_wake(w) for w in wakes}

    def _wake_from_thread(self, event) -> None:
        """FRONT-END THREAD → event loop."""
        try:
            self.bot.loop.call_soon_threadsafe(self._on_fast_wake, event)
        except RuntimeError:
            pass

    def _fast_wake_ok(self, event) -> bool:
        gid, uid = event.guild_id, event.user_id
        if uid not in self._auth.get(gid, ()):
            return False
        tag = getattr(event, "tag", None)
        if tag is not None and tuple(tag) != (self._session.get(gid, 0),
                                              self._epoch.get((gid, uid), 0)):
            return False
        return self._fast_wake_on(gid)

    def _on_fast_wake(self, event) -> None:
        """A fire (or a recheck) from the instant wake word, on the loop."""
        key = (event.guild_id, event.user_id)
        if not self._fast_wake_ok(event):
            return
        done = self._wake_confirmed.get(key)
        if done is not None and abs(done - event.onset) <= 0.5:
            return                                   # this utterance was cued already
        gid = event.guild_id
        recheck = getattr(event, "kind", "fire") == "recheck"
        self.metrics.count(gid, "fast_wake_recheck" if recheck else "fast_wake_fired")
        timing = CueTiming(event.at, score=event.score)
        timing.received = time.monotonic()
        instant = settings.voice_fast_wake_instant
        if (instant and not recheck and event.score >= instant
                and self._instant_cued.get(key) != event.onset):
            # Tier 1: confident enough to cue at once. The check still runs;
            # a "no" takes the text cue back (the sound can't be).
            timing.tier = "instant"
            self._instant_cued[key] = event.onset
            self.metrics.count(gid, "fast_wake_instant")
            self._spawn(self._wake_cue(gid, event.user_id, event.onset, timing=timing))
        elif not timing.tier:
            timing.tier = "checked"
        # Fair across speakers: one waiting fire per person, in the order they
        # came; a newer fire of the same person replaces theirs in place.
        self._wake_queue[key] = (event, timing)
        if self._wake_worker is None or self._wake_worker.done():
            self._wake_worker = self._spawn(self._wake_check_worker())

    async def _wake_check_worker(self) -> None:
        while self._wake_queue:
            key = next(iter(self._wake_queue))
            event, timing = self._wake_queue.pop(key)
            if time.monotonic() - (timing.received or 0.0) > _WAKE_CHECK_STALE:
                # Waited too long behind other speakers' checks: a cue now
                # would be late anyway — the transcript cues it instead.
                self.metrics.count(event.guild_id, "fast_wake_stale")
                continue
            try:
                await self._check_fast_wake(event, timing)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("voice: wake check failed")

    async def _check_fast_wake(self, event, timing: CueTiming | None = None) -> None:
        """Ask the recogniser (WakeVerifier) about a fire; cue it if it agrees."""
        gid, uid = event.guild_id, event.user_id
        key = (gid, uid)
        timing = timing or CueTiming(event.at, score=event.score)
        verifier = self.wake_verifier
        wakes = self._cfg.get(gid, {}).get("wake") or [settings.voice_wake_word]
        if verifier is not None and event.audio is not None:
            timing.verify_start = time.monotonic()
            try:
                heard, took = await verifier.heard(event.audio)
            except Exception:
                log.debug("voice: wake verification failed", exc_info=True)
                return
            timing.verify_end = time.monotonic()
            if self._debug_on(gid):
                log.info("voice instant wake: score %.2f (%s), the recogniser heard %r in %.2fs",
                         event.score, getattr(event, "kind", "fire"), heard, took)
            if text_after_wake(heard, wakes) is None:
                self.metrics.count(gid, "fast_wake_rejected")
                if timing.tier == "instant":
                    self.metrics.count(gid, "fast_wake_false_cue")
                    self._spawn(self._retract_cue(gid, uid, event.onset))
                elif (getattr(event, "kind", "fire") == "fire" and self._front is not None
                      and _wake_cut_short(heard, wakes)):
                    # It heard "hey…": the fire came a moment before the rest
                    # of the wake word. Look again with a little more audio.
                    self.metrics.count(gid, "fast_wake_recheck_asked")
                    self._front.request_wake_recheck(gid, uid, event.at + _WAKE_RECHECK_AFTER)
                return
        elif settings.voice_fast_wake_verify and timing.tier != "instant":
            self.metrics.count(gid, "fast_wake_unverified")    # verifier not loaded (yet)
            return
        if not self._fast_wake_ok(event):
            return                                   # opted out / moved meanwhile
        done = self._wake_confirmed.get(key)
        if done is not None and abs(done - event.onset) <= 0.5:
            return
        self._wake_confirmed[key] = event.onset
        self._fast_wakes[key] = event
        guild = self.bot.get_guild(gid)
        member = guild.get_member(uid) if guild else None
        if timing.tier == "instant":
            self.metrics.count(gid, "fast_wake")
        else:
            self.metrics.count(gid, "fast_wake")
            # The sound first — it's the "light" people notice — then "typing…".
            self._spawn(self._wake_cue(gid, uid, event.onset, timing=timing))
        if member is not None:
            await self._typing_start(guild, member, reason="heard")

    def _settle_fast_wake(self, job: Job, text: str, wakes) -> None:
        """Speech-to-text has the words of an utterance the instant wake word
        fired for (or didn't): count it, and take back a false alarm's
        ":ear: Now listening..." (the sound can't be taken back)."""
        key = job.key
        fast = self._fast_wakes.get(key)
        same = (fast is not None and job.speech_start is not None
                and abs(fast.onset - job.speech_start) <= 0.5)
        heard = text_after_wake(text, wakes) is not None
        if same:
            self._fast_wakes.pop(key, None)
            if heard:
                self.metrics.count(job.guild_id, "fast_wake_confirmed")
            else:
                self.metrics.count(job.guild_id, "fast_wake_false")
                self._spawn(self._retract_cue(job.guild_id, job.user_id, fast.onset))
        elif heard and self._fast_wake_on(job.guild_id) and not job.overlap_ms:
            self.metrics.count(job.guild_id, "fast_wake_missed")

    async def _retract_cue(self, guild_id: int, user_id: int, onset: float) -> None:
        key = (guild_id, user_id)
        entry = self._cue_msgs.get(key)
        if entry is None or abs(entry[0] - onset) > _CUE_SAME_UTTERANCE:
            return
        self._cue_msgs.pop(key, None)
        try:
            await entry[1].delete()
        except Exception:
            log.debug("voice: couldn't delete a false-alarm cue message", exc_info=True)

    # ── the early look for the wake word, while someone is still talking ────

    def _probe_from_thread(self, probe) -> None:
        """FRONT-END THREAD → event loop."""
        try:
            self.bot.loop.call_soon_threadsafe(self._on_probe, probe)
        except RuntimeError:
            pass

    def _on_probe(self, probe) -> None:
        gid, uid = probe.guild_id, probe.user_id
        key = (gid, uid)
        if uid not in self._auth.get(gid, ()) or key in self._typing:
            return
        tag = getattr(probe, "tag", None)
        if tag is not None and tuple(tag) != (self._session.get(gid, 0), self._epoch.get(key, 0)):
            return
        # Never in the way of real recognition: only when the recogniser is
        # free, and not on a box where recognition is slow anyway.
        if (len(self._jobs) or self._current_job is not None
                or (self._probe_task is not None and not self._probe_task.done())):
            self.metrics.count(gid, "probe_skipped_busy")
            return
        recent = sorted(self.metrics.guild(gid).asr_seconds)
        if len(recent) >= 3 and recent[len(recent) // 2] > _PROBE_MAX_ASR:
            self.metrics.count(gid, "probe_skipped_slow")
            return
        self._probe_task = self._spawn(self._run_probe(probe))

    async def _run_probe(self, probe) -> None:
        gid, uid = probe.guild_id, probe.user_id
        model = await self._models.get(0.2)
        if model is None:
            return
        audio = probe.audio[: int(_PROBE_SECONDS * 16_000)]
        try:
            result = await self._asr.run(transcribe_command, model, audio,
                                         self._asr_options(gid))
        except (AsrBusy, AsrTimeout):
            return
        self.metrics.count(gid, "probes")
        wakes = self._cfg.get(gid, {}).get("wake") or [settings.voice_wake_word]
        if text_after_wake(getattr(result, "text", "") or "", wakes) is None:
            return
        guild = self.bot.get_guild(gid)
        member = guild.get_member(uid) if guild else None
        if member is None or uid not in self._auth.get(gid, ()):
            return
        if (probe.tag is not None and tuple(probe.tag) !=
                (self._session.get(gid, 0), self._epoch.get((gid, uid), 0))):
            return
        self.metrics.count(gid, "probe_wake")
        await self._typing_start(guild, member, reason="heard")
        # …plus the audible "I'm listening" cue. Spawned: this runs on the
        # recogniser's thread, which must not wait on a message send.
        self._spawn(self._wake_cue(gid, uid, getattr(probe, "onset", None)))

    # ──────────────────────────────────────────────────────────────────────────
    # Decision → action
    #
    # Two lanes per guild, both in the order things were said:
    #   * controls (skip, pause, resume, volume, loop, restart, stop, clear) —
    #     quick, and never stuck behind a song search;
    #   * song requests — one lookup at a time, so songs join the queue in the
    #     order they were asked for.
    # Explicit rules instead of reordering: "stop" and "clear" cancel the song
    # requests heard BEFORE them (so a search that finishes afterwards can't
    # undo them); a typed stop invalidates them through the queue epoch; and a
    # command that has gone stale says so instead of vanishing.
    # ──────────────────────────────────────────────────────────────────────────

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        task.add_done_callback(lambda t: t.cancelled() or t.exception() is None
                               or log.error("voice background task failed", exc_info=t.exception()))
        return task

    def _enqueue_steps(self, job: Job, steps, text: str) -> None:
        """The grammar's actions for one request. A single action, or controls
        only, take the usual lanes (controls run at once, in order). Songs
        together with anything else run as one plan (core/voice_exec.py),
        strictly in order, each step waiting for the one before — "play these
        three, shuffle the queue, skip the first, loop the rest" needs the
        songs queued before the shuffle."""
        steps = tuple(steps)
        if len(steps) <= 1 or not any(s.action in SONG_ACTIONS for s in steps):
            for step in steps:
                self._enqueue_action(job, step, text)
            return
        plan = vp.fold_skips(vp.from_intents(steps))
        self._enqueue_plan(job, plan, text, view=vx.View.capture(get_state(job.guild_id)))

    def _enqueue_plan(self, job: Job, plan: "vp.Plan", text: str, *, view=None) -> None:
        """A planned request. With a lookup in it (songs, the catalogue, or
        another cog's command) it waits its turn in the song lane, behind
        earlier song requests; one made only of controls and queue edits runs
        in the control lane, in the order heard."""
        self._action_seq += 1
        act = _Action(job, Intent("plan"), text, self._action_seq, enqueued=time.monotonic(),
                      queue_epoch=get_state(job.guild_id).queue_epoch, plan=plan,
                      view=view if view is not None else vx.View.capture(get_state(job.guild_id)))
        gid = job.guild_id
        if any(OPS[st.op].lookup or OPS[st.op].command for st in plan.steps if st.op in OPS):
            self._plays.setdefault(gid, deque()).append(act)
            task = self._play_workers.get(gid)
            if task is None or task.done():
                self._play_workers[gid] = asyncio.create_task(
                    self._play_worker(gid), name=f"voice-play:{gid}")
            return
        q = self._dispatch_q.setdefault(gid, asyncio.Queue())
        q.put_nowait(act)
        task = self._dispatchers.get(gid)
        if task is None or task.done():
            self._dispatchers[gid] = asyncio.create_task(
                self._dispatcher(gid), name=f"voice-dispatch:{gid}")

    def _plan_needs_yes(self, plan: "vp.Plan") -> bool:
        """Like a grammar command: stop, clear and a very loud volume need a
        spoken "yes" first — once for the whole request."""
        for st in plan.steps:
            spec = OPS.get(st.op)
            if spec is not None and spec.confirm:
                return True
            if st.op == "volume":
                v = st.value.replace("%", "").strip()
                if v.isdigit() and int(v) > VOICE_VOLUME_MAX:
                    return True
        return False

    async def _start_plan(self, job: Job, plan: "vp.Plan", text: str, view, guild,
                          member) -> None:
        """A plan from the planner: asked about first if it needs a yes; run;
        or — with nothing to run — its problems and question, said."""
        key = job.key
        if job.segment_reason == "joined_incomplete":
            plan = replace(plan, problems=plan.problems + (
                "I may have missed the end of what you said",))
        if plan.steps and self._plan_needs_yes(plan):
            asked = time.monotonic()
            pending = _Pending(Intent("plan"), asked_at=asked, deadline=asked + _CONFIRM_SECONDS,
                               session=job.session, plan=plan, view=view)
            self._pending[key] = pending
            if self._front is not None:
                self._front.set_end_silence(job.guild_id, job.user_id, _WINDOW_END_SILENCE_MS)
            what = ", then ".join(vx.describe_step(st) for st in plan.steps)
            await self._deliver_feedback(guild, member, text,
                                         f"⚠️ {what[:1].upper()}{what[1:]}? Say “yes” or “no”.",
                                         force=True)
            if self._pending.get(key) is pending:
                pending.deadline = max(pending.deadline, time.monotonic() + _CONFIRM_SECONDS)
            return
        if plan.steps:
            self._last_action[key] = (Intent("plan"), job.end)
            self._enqueue_plan(job, plan, text, view=view)
            return
        lines = [f"⚠️ {p[:1].upper()}{p[1:]}." for p in plan.problems]
        if plan.ask:
            lines.append(f"❓ {plan.ask}")
        await self._deliver_feedback(guild, member, text, "\n".join(lines) or "🤷 Nothing to do.",
                                     force=bool(plan.ask), chime="cancel")
        if plan.ask:
            self._open_question(job, plan.ask)

    def _enqueue_action(self, job: Job, intent: Intent, text: str, *, steps=()) -> None:
        self._action_seq += 1
        act = _Action(job, intent, text, self._action_seq, enqueued=time.monotonic(),
                      queue_epoch=get_state(job.guild_id).queue_epoch, steps=tuple(steps))
        gid = job.guild_id
        if intent.action in SONG_ACTIONS or act.steps:
            self._plays.setdefault(gid, deque()).append(act)
            task = self._play_workers.get(gid)
            if task is None or task.done():
                self._play_workers[gid] = asyncio.create_task(
                    self._play_worker(gid), name=f"voice-play:{gid}")
            return
        q = self._dispatch_q.setdefault(gid, asyncio.Queue())
        q.put_nowait(act)
        task = self._dispatchers.get(gid)
        if task is None or task.done():
            self._dispatchers[gid] = asyncio.create_task(
                self._dispatcher(gid), name=f"voice-dispatch:{gid}")

    async def _dispatcher(self, guild_id: int) -> None:
        q = self._dispatch_q[guild_id]
        while not q.empty():
            await self._run_action(guild_id, q.get_nowait())

    async def _play_worker(self, guild_id: int) -> None:
        plays = self._plays[guild_id]
        while plays:
            act = plays[0]                 # stays visible to _cancel_plays while running
            try:
                if act.cancelled is None:
                    await self._run_action(guild_id, act)
            finally:
                if plays and plays[0] is act:
                    plays.popleft()

    async def _run_action(self, guild_id: int, act: _Action) -> None:
        try:
            await self._dispatch_one(act)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("voice dispatch failed")
            self.metrics.guild(guild_id).last_error = f"{type(e).__name__}: {e}"[:200]
            self.metrics.count(guild_id, "errors")
            self.metrics.stage(guild_id, act.intent.action, "failed")

    def _cancel_plays(self, guild_id: int, *, reason: str, before_seq: int | None = None,
                      user_id: int | None = None, notify: bool = True) -> int:
        """Call off song requests still waiting or being looked up."""
        n = 0
        for act in list(self._plays.get(guild_id, ())):
            if act.cancelled is not None:
                continue
            if before_seq is not None and act.seq >= before_seq:
                continue
            if user_id is not None and act.job.user_id != user_id:
                continue
            act.cancelled = reason
            n += 1
            for part in (act, act.current):          # a plan: also the step under way
                if part is None:
                    continue
                part.cancelled = reason
                if part.task is not None and not part.task.done():
                    part.task.cancel()
            self.metrics.stage(guild_id, act.intent.action, "cancelled")
            if notify:
                guild = self.bot.get_guild(guild_id)
                member = guild.get_member(act.job.user_id) if guild else None
                if member is not None:
                    what = {"stop": "playback was stopped",
                            "clear": "the queue was cleared",
                            "session_changed": "I moved, left or reconnected"}.get(reason, reason)
                    if act.steps or act.plan is not None:
                        msg = f"✖ Didn't finish your request — {what} after you asked."
                    elif act.intent.action in ("play", "find"):
                        msg = f"✖ Didn't queue “{act.intent.arg}” — {what} after you asked."
                    else:
                        msg = (f"✖ Didn't {self._lower_first(self._describe(act.intent))} — "
                               f"{what} after you asked.")
                    self._spawn(self._deliver_feedback(guild, member, act.text, msg,
                                                       chime="cancel"))
        return n

    async def _dispatch_one(self, act: _Action, *, deliver: bool = True
                            ) -> Optional[ActionResult]:
        """Run one action (or a plan) and reply. With `deliver=False` — a step
        of a plan — the reply is returned instead, for one combined message."""
        if act.plan is not None:
            return await self._run_plan(act)
        job, intent, text = act.job, act.intent, act.text
        gid, action = job.guild_id, intent.action
        reason = self._job_valid(job)
        if reason:
            self.metrics.drop(gid, reason)
            self.metrics.stage(gid, action, "cancelled")
            self._typing_stop(job.key)
            return ActionResult("cancelled", None)
        guild = self.bot.get_guild(gid)
        member = guild.get_member(job.user_id)
        # Speech too old to act on safely is refused — but time a song request
        # (or a step of a plan) spent queued behind another song's lookup is
        # our delay, not theirs: its age is judged as of when it was recognised.
        aged_by_queueing = action in SONG_ACTIONS or act.in_plan
        now = act.enqueued if (aged_by_queueing and act.enqueued) else time.monotonic()
        if now - job.end > self._max_age(job):
            # Recognised, but too late to act on safely. Say so — a command
            # that silently never happens is what makes voice control maddening.
            self.metrics.drop(gid, "stale_before_action")
            self.metrics.stage(gid, action, "stale")
            what = self._describe(intent)
            msg = (f"⌛ I understood “{what[0].lower() + what[1:]}” too late to act on "
                   f"it safely — say it again.")
            if deliver:
                await self._deliver_feedback(guild, member, text, msg, chime="cancel")
            return ActionResult("failed", msg)
        self.metrics.stage(gid, action, "admitted")
        if action in ("stop", "clear"):
            self._cancel_plays(gid, reason=action, before_seq=act.seq)
        if action in SONG_ACTIONS:
            # Each song's lookup takes a moment; an earlier reply (of a
            # several-part request) may have ended the indicator.
            await self._typing_start(guild, member, reason="search")
            if act.cancelled is not None:
                return ActionResult("cancelled", None)  # _cancel_plays told the requester
        vc = guild.voice_client
        old_epoch = get_state(gid).queue_epoch
        result = await self._execute_intent(guild, member, vc, intent, act=act)
        if action in ("stop", "clear") and result.status == "done":
            # Later steps already admitted after this control (e.g. "clear,
            # then play X") belong to the new queue. Earlier work stays invalid.
            for later in self._plays.get(gid, ()):
                if later.seq > act.seq and later.queue_epoch == old_epoch:
                    later.queue_epoch = get_state(gid).queue_epoch
        if act.cancelled is not None:
            return ActionResult("cancelled", None)   # _cancel_plays already told the requester
        self.metrics.stage(gid, action, "queued" if (action in SONG_ACTIONS and result.status == "done")
                           else _STATUS_STAGE.get(result.status, result.status))
        if result.status == "done":
            self.metrics.guild(gid).latency_seconds.append(time.monotonic() - job.end)
            self.metrics.event(gid, job.user_id, action)
        if result.message is None or not deliver:
            return result
        await self._deliver_feedback(guild, member, text, result.message,
                                     chime="ok" if result.status == "done" else "cancel")
        return result

    async def _run_plan(self, act: _Action) -> ActionResult:
        """One planned request (core/voice_exec.py PlanRun): its steps in
        order, each against the live queue, then one reply listing what
        happened to every step — done, failed, not done and why."""
        job, gid = act.job, act.job.guild_id
        key = job.key
        guild = self.bot.get_guild(gid)
        member = guild.get_member(job.user_id) if guild else None
        plan = act.plan
        # The same speech must not be carried out twice (a retry, a segment
        # heard twice): intentional repeats are a step's count, not this.
        rid = (gid, job.user_id, round(job.onset, 2), " ".join(act.text.split()))
        if rid in self._done_requests:
            self.metrics.drop(gid, "duplicate_request")
            self._typing_stop(key)
            return ActionResult("noop", None)
        self._done_requests.append(rid)
        if any(st.op in vx.SONG_OPS for st in plan.steps):
            await self._typing_start(guild, member, reason="search")
        run = vx.PlanRun(self, act, plan, view=act.view, mine=list(self._mine.get(key, ())),
                         rng=self._rng)
        self._last_run[gid] = run                    # `voice diag` and tests read it
        results = await run.run()
        added = [i for r in results for i in r.added]
        if added:
            self._mine.setdefault(key, deque(maxlen=50)).extend(added)
        self._remember_request(key, act.text, vx.summarize(run))
        for r in results:
            self.metrics.stage(gid, r.step.op, _STATUS_STAGE.get(r.status, r.status))
            if r.why and r.status not in vx.OK:
                log.info("voice plan step %s in %s: %s — %s", r.step.label, gid, r.status,
                         " ".join(str(r.why).split())[:300])
        if act.cancelled is not None and not any(r.ok for r in results):
            return ActionResult("cancelled", None)       # the canceller already said why
        lines, answers, ok = vx.compose_reply(run)
        question = plan.ask or next((r.ask for r in results if r.ask), "")
        if member is not None and answers:
            await self._answer(guild, member, act.text, " ".join(answers))
        if member is not None and lines:
            await self._deliver_feedback(guild, member, act.text, "\n".join(lines),
                                         chime="ok" if ok else "cancel", force=bool(question))
        elif member is not None and not answers:
            self._typing_stop(key)
        if question and member is not None:
            self._open_question(job, question)
        if ok:
            self.metrics.guild(gid).latency_seconds.append(time.monotonic() - job.end)
            self.metrics.event(gid, job.user_id, "plan")
        return ActionResult("done" if ok else "failed", None, added=tuple(added))

    def _plan_sub(self, act: _Action, intent: Intent, step, *, copies: int = 1) -> _Action:
        """A plan step as an action for the cog's own command path."""
        return _Action(act.job, intent, act.text, act.seq, enqueued=act.enqueued,
                       queue_epoch=get_state(act.job.guild_id).queue_epoch, in_plan=True,
                       parent=act, step=step, copies=max(1, copies))

    async def _prefix(self, guild) -> str:
        try:
            return await db.get_prefix(guild.id)
        except Exception:
            return settings.default_prefix

    def _bridged_missing(self, spec, step) -> str:
        """The input a bridged command needs that the request didn't give
        (in words, e.g. "when"), or "" — so it's asked for, not failed."""
        name = spec.command
        if spec.choices:
            name = spec.choices.get(" ".join((step.value or "").lower().split())) or \
                spec.choices.get("") or spec.command
        cmd = self.bot.get_command(name) if name else None
        if cmd is None:
            return ""
        for pname, param in cmd.clean_params.items():
            field_name = spec.params.get(pname)
            if field_name is None or not param.required:
                continue
            raw = getattr(step, field_name, "")
            if field_name != "count" and not " ".join(str(raw or "").split()):
                return pname.replace("_", " ")
        return ""

    async def _run_bridged(self, guild, member, spec, step, heard: str) -> tuple[str, str]:
        """Another cog's command, run by voice exactly as if typed: looked up
        by name, its own checks (cog check, permission decorators, global
        checks) run against the speaker, its parameters given typed values
        from the plan step (never a command string), its replies posted in
        the voice reply channel. Returns (status, message for the reply)."""
        name = spec.command
        if spec.choices:
            choice = " ".join((step.value or "").lower().split())
            name = spec.choices.get(choice) or spec.choices.get("") or spec.command
            if choice and choice not in spec.choices:
                opts = ", ".join(sorted(k for k in spec.choices if k))
                return "failed", f"❌ {spec.say[:1].upper()}{spec.say[1:]}: “{step.value}”? ({opts})"
        cmd = self.bot.get_command(name)
        if cmd is None:
            return "failed", f"❌ `{name}` isn't available on this bot right now."
        channel = await self._echo_channel(guild)
        if channel is None:
            return "failed", "❌ I have nowhere to post that."
        kwargs, problem = _bridged_args(cmd, spec, step)
        prefix = await self._prefix(guild)
        if problem:
            return "failed", f"❌ `{prefix}{name}`: {problem}"
        ctx = _voice_context(self.bot, guild, member, channel, cmd, prefix, heard)
        try:
            for parent in reversed(cmd.parents):
                if not await parent.can_run(ctx):
                    raise commands.CheckFailure(f"not allowed: {parent.qualified_name}")
            if not await cmd.can_run(ctx):
                raise commands.CheckFailure(f"not allowed: {name}")
        except commands.CheckFailure as e:
            self.metrics.count(guild.id, "bridged_refused")
            why = ("you don't have permission for that" if not str(e) or "permission" in str(e).lower()
                   or "not allowed" in str(e) else str(e))
            return "failed", f"❌ `{prefix}{name}`: {why}."[:300]
        except commands.CommandError as e:
            return "failed", f"❌ `{prefix}{name}`: {e}"[:300]
        self.metrics.count(guild.id, "bridged_" + name.replace(" ", "_"))
        try:
            await ctx.invoke(cmd, **kwargs)
        except commands.CommandError as e:
            return "failed", f"❌ `{prefix}{name}`: {e}"[:300]
        except Exception as e:
            log.exception("voice: bridged command %s failed", name)
            return "failed", f"❌ `{prefix}{name}` failed ({type(e).__name__})."
        return "done", ""                   # the command posted its own reply

    def _player_busy(self, guild_id: int) -> bool:
        guild = self.bot.get_guild(guild_id)
        music = self.bot.get_cog("Music")
        return bool(guild is not None and music is not None and music._busy(guild))

    def _place_songs(self, guild, vc, music, act, entries: list) -> tuple[bool, bool]:
        """New songs into the queue: at the end, or — "next"/"now" — at the
        front, after any this same request already put there ("X now, then Y
        next" keeps X before Y). "now" also makes way: the current song is
        skipped, once per request. Returns (the player was idle, one was
        skipped)."""
        state = get_state(guild.id)
        when = (act.intent.when if act is not None else None) or ""
        holder = (act.parent or act) if act is not None else None
        idle = music is None or not music._busy(guild)
        if when in ("next", "now"):
            last = holder.front_last if holder is not None else None
            at = next((i + 1 for i, t in enumerate(state.queue) if t is last), 0) if last else 0
            for n, entry in enumerate(entries):
                state.queue.insert(at + n, entry)
            if holder is not None and entries:
                holder.front_last = entries[-1]
        else:
            state.queue.extend(entries)
        skipped = False
        if when == "now" and not idle and not (holder is not None and holder.now_done):
            # A spoken line playing on its own (nothing else on) isn't a song
            # to skip: the songs start when it ends (cogs/tts.py).
            speech_only = isinstance(getattr(vc, "source", None), voice_mix.SpeechSource)
            if (vc.is_playing() or vc.is_paused()) and not speech_only:
                if state.loop_queue and state.current is not None:
                    state.queue.append(requeue_copy(state.current))
                state.current = None            # don't let loop re-queue it
                vc.stop()                       # its end advances to the new song
                skipped = True
            elif music is not None and music._abandon_starting(state):
                skipped = True
            if holder is not None:
                holder.now_done = True
        if music is not None and not vc.is_playing() and not vc.is_paused():
            music._kickstart(guild, vc)
        return idle, skipped

    async def _settle_advance(self, guild_id: int, timeout: float = _SETTLE_SECONDS) -> None:
        """Songs were just queued into an idle player: give it a moment to
        take the first one, so the next step sees it as "the current song"."""
        state = get_state(guild_id)
        deadline = time.monotonic() + timeout
        while state.current is None and state.queue and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

    @staticmethod
    def _lower_first(text: str) -> str:
        return text[:1].lower() + text[1:] if text else text

    # ── intent execution against the music player ─────────────────────────────

    async def _execute_intent(self, guild, member, vc, intent: Intent, *,
                              act: _Action | None = None) -> ActionResult:
        if vc is None or not vc.is_connected():
            return ActionResult("failed", "❌ I'm not in a voice channel.")
        state = get_state(guild.id)
        music = self.bot.get_cog("Music")
        a = intent.action

        if a == "skip":
            # "skip two songs": the current one plus the next one(s) in line.
            extra = max(0, min(10, int(intent.count)) - 1) if (intent.count or "").isdigit() else 0
            dropped = 0
            for _ in range(extra):
                if not state.queue:
                    break
                gone = state.queue.popleft()
                if state.loop_queue:
                    state.queue.append(requeue_copy(gone))   # queue loop: comes round again
                dropped += 1
            said = "⏭️ Skipped." if not dropped else f"⏭️ Skipped {dropped + 1} songs."
            if vc.is_playing() or vc.is_paused():
                if state.loop_queue and state.current is not None:
                    state.queue.append(requeue_copy(state.current))
                state.current = None        # don't let loop re-queue it
                vc.stop()
                return ActionResult("done", said)
            if music is not None and music._abandon_starting(state):
                music._kickstart(guild, vc)     # it hadn't made a sound yet
                return ActionResult("done", said)
            if dropped:
                return ActionResult("done", f"⏭️ Skipped {dropped} queued song{'s' if dropped != 1 else ''}.")
            return ActionResult("noop", "Nothing's playing.")
        if a == "pause":
            # voice_mix variants: see the text pause/resume commands.
            if voice_mix.music_is_playing(vc):
                # Same clock bookkeeping as the text `pause` command — without
                # it, pausing by voice desyncs lyrics and the progress bar.
                if state.clock is not None:
                    state.clock.pause(vc)
                vc.pause()
                if state.clock is not None:
                    state.track_started_at = state.clock.as_started_at(vc)
                return ActionResult("done", "⏸️ Paused.")
            return ActionResult("noop", "Nothing to pause.")
        if a == "resume":
            if voice_mix.music_is_paused(vc):
                voice_mix.resume_music(vc)
                if state.clock is not None:
                    state.clock.resume(vc)
                    state.track_started_at = state.clock.as_started_at(vc)
                return ActionResult("done", "▶️ Resumed.")
            return ActionResult("noop", "Nothing is paused.")
        if a == "loop":
            # Explicit modes are idempotent — hearing "loop on" twice (or a
            # duplicate recognition) must not flip it back off. "on" (the
            # grammar's word) is the song; "queue" loops everything queued.
            before = "song" if state.loop else "queue" if state.loop_queue else "off"
            mode = {"on": "song", "song": "song", "queue": "queue", "off": "off"}.get(
                intent.arg or "on", "song")
            if intent.arg == "toggle":
                mode = "off" if before != "off" else "song"
            state.loop, state.loop_queue = mode == "song", mode == "queue"
            n = len(state.queue) + (1 if state.current else 0)
            label = {"song": "🔁 Loop on — this song.",
                     "queue": f"🔁 Looping the queue ({n} song{'s' if n != 1 else ''}).",
                     "off": "➡️ Loop off."}[mode]
            changed = mode != before
            return ActionResult("done" if changed else "noop",
                                label if changed else label.replace(".", " (already).", 1))
        if a == "shuffle":
            n = len(state.queue)
            if n < 2:
                return ActionResult("noop", "🔀 Nothing to shuffle." if not n
                                    else "🔀 There's only one song in the queue.")
            queue_ops.shuffle(state, self._rng)
            return ActionResult("done", f"🔀 Shuffled {n} songs.")
        if a == "restart":
            return await self._do_restart(guild, vc, state, music)
        if a == "stop":
            reset_state(guild.id)
            if vc.is_playing() or vc.is_paused():
                vc.stop()
            return ActionResult("done", "⏹️ Stopped.")   # stay connected so voice control keeps working
        if a == "clear":
            n = clear_queue(guild.id)
            return ActionResult("done", f"🗑️ Cleared {n} from the queue.")
        if a == "volume":
            return self._do_volume(state, vc, intent.arg)
        if a in ("play", "find"):
            return await self._do_play(guild, vc, member, intent, music, act=act)
        if a in ("album", "top", "latest"):
            return await self._do_catalog(guild, vc, member, intent, music, act=act)
        return ActionResult("noop", None)

    def _do_volume(self, state, vc, arg: Optional[str]) -> ActionResult:
        cap = VOICE_VOLUME_MAX / 100
        if arg == "up":
            # Steps stop at the voice cap but never go *down*: at 300% (set by
            # the typed command) "louder" used to drop it to 200%.
            if state.volume >= cap:
                return ActionResult("noop", f"🔊 Already at {int(state.volume * 100)}% — that's as "
                                            f"loud as voice control goes. Use the `volume` command "
                                            f"for more.")
            state.volume = min(cap, state.volume + 0.25)
        elif arg == "down":
            state.volume = max(0.0, state.volume - 0.25)
        else:
            try:
                # Levels above the voice cap were confirmed before we got here.
                state.volume = max(0, min(1000, int(arg))) / 100
            except (TypeError, ValueError):
                return ActionResult("failed", "❌ Didn't catch the volume.")
        if vc.source:
            vc.source.volume = state.volume
        return ActionResult("done", f"🔊 Volume {int(state.volume * 100)}%.")

    async def _do_restart(self, guild, vc, state, music) -> ActionResult:
        """Back to the start of the current track (same path as `seek 0`)."""
        cur = state.current
        if music is None or not cur or not (vc.is_playing() or vc.is_paused()):
            return ActionResult("noop", "Nothing's playing.")
        gen = state.playback_gen
        try:
            await audio.resolve_stream_url(cur)
        except Exception as e:
            log.warning("voice restart: re-resolve failed: %s", audio.friendly_error_msg(e))
            return ActionResult("failed", "❌ Couldn't restart the track.")
        if state.current is not cur or state.playback_gen != gen:
            return ActionResult("cancelled", "✖ The track changed meanwhile — not restarted.")
        if not music._begin_playback(guild, vc, state, cur, position=0.0):
            return ActionResult("failed", None)          # the music cog already said why
        return ActionResult("done", "⏮️ From the top.")

    @staticmethod
    def _describe(intent: Intent) -> str:
        """An action in words — the question before one that needs a yes,
        and what a plan didn't get to."""
        a, by = intent.action, (f" by {intent.artist}" if intent.artist else "")
        if intent.when in ("next", "now") and a in SONG_ACTIONS:
            base = VoiceDJ._describe(replace(intent, when=None))
            return f"{base} {intent.when}"
        if a == "stop":
            return "Stop the music and clear the queue"
        if a == "clear":
            return "Clear the whole queue"
        if a == "volume":
            return f"Set the volume to {intent.arg}%" if (intent.arg or "").isdigit() else \
                f"Turn the volume {intent.arg}"
        if a == "play" and intent.arg:
            return f"Play {intent.arg}{by}"
        if a == "find" and intent.arg:
            return f"Find and play {intent.arg}"
        if a == "album":
            sel = (intent.count or "all").strip()
            which = ("all of" if sel.lower() in ("", "all") else
                     f"track {sel} of" if sel.isdigit() else f"tracks {sel} of")
            return f"Play {which} {intent.arg}{by}"
        if a == "top":
            return f"Play {intent.count or 5} of {intent.artist}'s most popular songs"
        if a == "latest":
            return f"Play {intent.artist}'s newest " + (
                "song" if (intent.count or "1") == "1" else f"{intent.count} songs")
        if a == "shuffle":
            return "Shuffle the queue"
        if a == "loop":
            return {"queue": "Loop the queue", "off": "Turn the loop off"}.get(
                intent.arg or "", "Loop the song")
        if a == "skip" and intent.count:
            return f"Skip {intent.count} songs"
        if a == "restart":
            return "Restart the song"
        return a.capitalize()

    async def _do_play(self, guild, vc, member, intent: Intent, music, *,
                       act: _Action | None = None) -> ActionResult:
        """One song — named ("play X by Y") or described ("find": that Yung
        Gravy song that samples Panic! At The Disco) — worked out with the
        music catalogue, then found on YouTube, every step checked against
        what was asked (_find_song): a miss, an ambiguity, a catalogue outage
        or a missing upload is said as what it is, never "the first YouTube
        result" instead. `act.copies` separate entries ("five times" = 5),
        each with its own occurrence id."""
        query = intent.search if intent.action == "play" else intent.arg
        if not query:
            return ActionResult("failed", "❌ Play what?")
        state = get_state(guild.id)
        epoch = act.queue_epoch if act is not None and act.queue_epoch is not None else state.queue_epoch
        if epoch != state.queue_epoch:
            return ActionResult("cancelled", "✖ Didn't queue that request — playback was stopped or the queue was cleared after you asked.")
        echo = await self._echo_channel(guild)
        # Bounded and cancellable: a slow search used to hold up ALL
        # recognition, then every command in this server. It now has its own
        # lane, gives up eventually, and "stop"/"clear" call it off.
        lookup = asyncio.ensure_future(self._find_song(guild, intent, act))
        if act is not None:
            act.task = lookup
        timeout = _FIND_LOOKUP_TIMEOUT if intent.action == "find" else _PLAY_LOOKUP_TIMEOUT
        try:
            done, _ = await asyncio.wait({lookup}, timeout=timeout)
        except asyncio.CancelledError:
            lookup.cancel()
            raise
        if act is not None and act.cancelled:
            lookup.cancel()
            return ActionResult("cancelled", None)       # the canceller already said why
        if not done:
            lookup.cancel()
            return ActionResult("failed", f"❌ Searching for “{query}” took too long — try again.",
                                why="lookup timed out")
        try:
            found = lookup.result()
        except asyncio.CancelledError:
            return ActionResult("cancelled", None)
        except Exception as e:
            log.warning("voice play lookup failed for %r: %s", query, audio.friendly_error_msg(e))
            found = _SongFound(None, None, f"❌ Couldn't look up “{query}” just now — try again.",
                               f"lookup error {type(e).__name__}", "error")
        # The evidence for (or against) the song — catalogue names, scores and
        # the uploads weighed — no speaker, no transcript; bounded.
        log.info("voice song in %s: %s → %s", guild.id, found.status,
                 " ".join(str(found.why).split())[:400])
        if found.entry is None:
            return ActionResult("failed", found.failure or f"❌ Couldn't find “{query}”.",
                                why=found.why, ask=found.ask)
        track, known = found.entry, found.known
        title = track.get("title") or query
        # The search awaited the network: re-check everything that could have
        # changed, right before touching the queue.
        if member.id not in self._auth.get(guild.id, frozenset()):
            return ActionResult("cancelled", None)       # opted out meanwhile: say nothing
        if act is not None:
            reason = self._job_valid(act.job)
            if reason == "revoked":
                return ActionResult("cancelled", None)   # they opted out: say nothing
            if reason:
                return ActionResult("cancelled", f"✖ Didn't queue **{title}** — "
                                                 f"{_WHY_INVALID.get(reason, reason)}.")
        if state.queue_epoch != epoch:
            changed = "The queue was cleared" if state.queue_change_reason == "cleared" else "Playback was stopped"
            return ActionResult("cancelled", f"⏹️ {changed} while I was searching "
                                             f"— didn't queue **{title}**.")
        vc = guild.voice_client
        if vc is None or not vc.is_connected():
            return ActionResult("cancelled", f"✖ Didn't queue **{title}** — "
                                             f"{_WHY_INVALID['disconnected']}.")
        copies = max(1, act.copies) if act is not None else 1
        entries = [track] + [_fresh_copy(track) for _ in range(copies - 1)]
        for n, e in enumerate(entries):
            e["requester"] = member.display_name
            e["requester_id"] = member.id      # so voice-queued plays count in stats
            if echo is not None:
                e["request_channel_id"] = echo.id    # where a playback failure is reported
            if act is not None and n == 0:
                e["_voice_job_end"] = act.job.end    # end of speech → first audio, for diag
        idle, skipped = self._place_songs(guild, vc, music, act, entries)
        added = tuple(qid(e) for e in entries)
        if act is not None:
            act.started_idle = (idle or skipped) and music is not None   # an advance is coming
        when = (act.intent.when if act is not None else None) or ""
        found_msg = (f"🔎 That's **{known.title}** by {known.artist}. "
                     if intent.action == "find" and known is not None else "")
        times = f" ×{copies}" if copies > 1 else ""
        # "Queued" is all we know here; the music cog reports a failure to
        # actually play, and "Starting…" never claims sound came out.
        if skipped:
            msg = f"{found_msg}⏭️ Skipped — playing **{title}**{times} now."
        elif when in ("next", "now") and not idle:
            msg = f"{found_msg}✅ **{title}**{times} is up next."
        else:
            msg = f"{found_msg}✅ Queued **{title}**{times}." + (" Starting…" if idle else "")
        return ActionResult("done", msg, added=added, why=found.why)

    async def _find_song(self, guild, intent: Intent, act: _Action | None) -> "_SongFound":
        """Which song, and its upload — with what was found at each stage.

        1. The catalogue (core/music_catalog.find_song) with the title and
           artist as heard and as the planner read them. Only a song by the
           artist named counts; a similar-sounding more famous one ("Prince"
           for "Priest") is only ever offered back as a question.
        2. Unsure between close candidates: the AI picks among those real
           songs (only), or the speaker is asked which.
        3. The catalogue's song on YouTube, the upload checked against title,
           artist and length (core/audio.upload_score).
        4. A song the catalogue doesn't know (or can't be asked about right
           now): a YouTube search whose result must show the title — and the
           artist, if one was named — or nothing is queued and they're asked.
        """
        gid = guild.id
        said = act.text if act is not None else (intent.heard or intent.arg or "")
        step = act.step if act is not None else None
        heard_artist = (step.heard_artist if step is not None else "") or ""
        if intent.action == "find":
            known = await self._identify_song(gid, said, intent)
            if known is None:
                self.metrics.count(gid, "find_failed")
                return _SongFound(None, None, f"❌ I couldn't work out which song “{intent.arg}” "
                                              f"is — try saying its name.", "described: not identified",
                                  "not_found")
            why, ask, status = f"described → {known.artist} - {known.title}", None, "matched"
        else:
            if not (intent.arg or "").strip():
                return _SongFound(None, None, "❌ Play what?", "no title", "incomplete")
            known, why, ask, status = await self._match_song(gid, said, intent, heard_artist)
        if known is not None:
            for q in (known.query, f"{known.artist} {known.title} audio"):
                try:
                    entry, up = await audio.find_upload_checked(
                        q, title=known.title, artist=known.artist, duration=known.duration)
                except Exception as e:
                    log.warning("voice: YouTube search for %r failed: %s", q,
                                audio.friendly_error_msg(e))
                    entry, up = None, f"search failed ({type(e).__name__})"
                if entry is not None:
                    self.metrics.count(gid, "song_found")
                    return _SongFound(entry, known, None, f"{why}; upload: {up}", "matched")
            self.metrics.count(gid, "upload_not_found")
            return _SongFound(None, known, f"❌ Couldn't find **{known.title}** by {known.artist} "
                                           f"on YouTube.", f"{why}; upload: {up}", "no_upload")
        # Not identified by the catalogue: a checked search, or nothing.
        title = (intent.arg or "").strip()
        names = [n for n in dict.fromkeys(x.strip() for x in (heard_artist, intent.artist or ""))
                 if n]
        if len(names) > 1 and not music_catalog.plausible_correction(names[0], names[1]):
            names = names[:1]            # the words said, not another artist's name
        tried = []
        for artist in names or [""]:
            q = f"{artist} - {title}" if artist else title
            need = (audio.UPLOAD_ACCEPT_UNKNOWN if artist else audio.UPLOAD_ACCEPT_UNKNOWN_NO_ARTIST)
            try:
                entry, up = await audio.find_upload_checked(q, title=title, artist=artist,
                                                            min_score=need)
            except Exception as e:
                log.warning("voice: YouTube search for %r failed: %s", q, audio.friendly_error_msg(e))
                entry, up = None, f"search failed ({type(e).__name__})"
            tried.append(up)
            if entry is not None:
                self.metrics.count(gid, "song_found_by_search")
                return _SongFound(entry, None, None, f"{why}; search: {up}", "upload_only")
        self.metrics.count(gid, "song_not_found")
        by = f" by {names[0]}" if names else ""
        if ask:
            return _SongFound(None, None, f"❓ {ask}", f"{why}; search: {' | '.join(tried)}",
                              "ambiguous", ask=ask)
        down = status == "catalog_down"
        question = (f"I couldn't find **{title}**{by}"
                    + (" (the music catalogue isn't answering right now)" if down else "")
                    + " — can you say the title or artist again, or spell it?")
        return _SongFound(None, None, f"❓ {question}", f"{why}; search: {' | '.join(tried)}",
                          "catalog_down" if down else "not_found", ask=question)

    def _ai_on(self, guild_id: int) -> bool:
        return self.llm is not None and bool(self._cfg.get(guild_id, {}).get("ai", True))

    async def _match_song(self, gid: int, said: str, intent: Intent, heard_artist: str = ""):
        """(the catalogue's song or None, why, a question to ask or None,
        status) for a "play X (by Y)" — by spelling and by sound, against the
        artist's own songs. None without a question means the catalogue
        doesn't know it (or can't be asked): the caller searches, checked."""
        wanted = f"{intent.arg}" + (f" by {heard_artist or intent.artist}"
                                    if (heard_artist or intent.artist) else "")
        try:
            match = await music_catalog.find_song(intent.arg or "", intent.artist or "",
                                                  heard=intent.heard or "",
                                                  heard_artist=heard_artist)
        except music_catalog.CatalogError as e:
            log.info("voice: catalogue song lookup failed: %s", e)
            self.metrics.count(gid, "catalog_failed")
            return None, f"catalogue unavailable: {e}"[:200], None, "catalog_down"
        if match is None:
            self.metrics.count(gid, "song_unknown")
            return None, f"catalogue: nothing close to {wanted!r}", None, "not_found"
        if not match.matched:
            # Only songs by *other* artists with a similar name: Priest isn't
            # Prince. Offered back as a question, never played.
            self.metrics.count(gid, "song_artist_unmatched")
            t = match.track
            named = heard_artist or intent.artist or "them"
            return None, match.why, (f"I couldn't find **{intent.arg}** by {named} — did you mean "
                                     f"**{t.title}** by {t.artist}?"), "ambiguous"
        if match.sure:
            self.metrics.count(gid, "song_matched")
            return match.track, match.why, None, "matched"
        choices = [match.track, *match.others][:8]
        if self._ai_on(gid):
            i = await self.llm.choose_song(said, wanted, choices)
            self.metrics.count(gid, "song_asked_ai")
            if i is not None:
                c = choices[i]
                return c, f"{match.why}; AI picked {c.artist} - {c.title}", None, "matched"
        alts = [c for n, c in enumerate(choices) if all(
            music_catalog.norm(c.title) != music_catalog.norm(o.title) or c.artist != o.artist
            for o in choices[:n])][:2]
        question = ("Which one did you mean — "
                    + " or ".join(f"**{c.title}** by {c.artist}" for c in alts) + "?")
        self.metrics.count(gid, "song_ambiguous")
        return None, match.why, question, "ambiguous"

    async def _identify_song(self, gid: int, said: str, intent: Intent):
        """A song they described: picked by the AI from the artist's songs if
        they named one, otherwise (or if none fits) looked up online, then
        checked against the catalogue. None if it can't be worked out."""
        if not self._ai_on(gid):
            return None
        desc, artist = intent.arg or "", intent.artist or ""
        if artist:
            try:
                songs = await music_catalog.artist_songs(artist)
            except music_catalog.CatalogError as e:
                log.info("voice: catalogue artist lookup failed: %s", e)
                songs = []
            if songs:
                i = await self.llm.choose_song(said, f"a song they described: {desc}", songs)
                self.metrics.count(gid, "find_from_artist" if i is not None else "find_not_in_artist")
                if i is not None:
                    return songs[i]
        found = await self.llm.lookup_song(desc, artist)
        self.metrics.count(gid, "find_web" if found else "find_web_none")
        if found is None:
            return None
        title, who = found
        try:
            match = await music_catalog.find_song(title, who or artist)
        except music_catalog.CatalogError:
            match = None
        if match is not None and match.score >= 0.75:
            return match.track
        return music_catalog.CatalogTrack(title, who or artist)

    async def _do_catalog(self, guild, vc, member, intent: Intent, music, *,
                          act: _Action | None = None) -> ActionResult:
        """album / top / latest. The LLM named the album or the artist; a
        music catalogue (core/music_catalog.py) knows the tracks and their
        order. They're queued like a Spotify playlist: each one is searched
        on YouTube as "Artist - Title" when its turn comes, so a whole album
        is in the queue at once — ready to be shuffled, skipped or looped."""
        state = get_state(guild.id)
        epoch = (act.queue_epoch if act is not None and act.queue_epoch is not None
                 else state.queue_epoch)
        if epoch != state.queue_epoch:
            return ActionResult("cancelled", "✖ Didn't queue that — playback was stopped or "
                                             "the queue was cleared after you asked.")
        a = intent.action
        if a == "album":
            coro = music_catalog.find_album(intent.arg or "", intent.artist or "")
        elif a == "top":
            coro = music_catalog.top_tracks(
                intent.artist or "", music_catalog.parse_count(intent.count, 5, music_catalog.MAX_TOP))
        else:
            coro = music_catalog.latest_tracks(
                intent.artist or "", music_catalog.parse_count(intent.count, 1, 5))
        echo = await self._echo_channel(guild)
        self.metrics.count(guild.id, f"catalog_{a}")
        lookup = asyncio.ensure_future(coro)
        if act is not None:
            act.task = lookup
        try:
            done, _ = await asyncio.wait({lookup}, timeout=_CATALOG_TIMEOUT)
        except asyncio.CancelledError:
            lookup.cancel()
            raise
        if act is not None and act.cancelled:
            lookup.cancel()
            return ActionResult("cancelled", None)       # the canceller already said why
        if not done:
            lookup.cancel()
            self.metrics.count(guild.id, "catalog_failed")
            return ActionResult("failed", "❌ The music catalogue took too long — try again.")
        try:
            found = lookup.result()
        except asyncio.CancelledError:
            return ActionResult("cancelled", None)
        except music_catalog.CatalogError as e:
            log.warning("voice catalogue lookup failed: %s", e)
            self.metrics.count(guild.id, "catalog_failed")
            return ActionResult("failed", "❌ Couldn't reach the music catalogue just now — "
                                          "try again in a bit.")
        except Exception:
            log.exception("voice catalogue lookup crashed")
            self.metrics.count(guild.id, "catalog_failed")
            return ActionResult("failed", "❌ Something went wrong looking that up.")
        by = f" by {intent.artist}" if intent.artist else ""
        note = ""
        if a == "album":
            if found is None:
                return ActionResult("failed", f"❌ Couldn't find the album “{intent.arg}”{by}.",
                                    why="album not in the catalogue")
            sel = intent.count or "all"
            try:
                picked = music_catalog.select(found.tracks, sel, rng=self._rng)
            except music_catalog.SelectionError as e:
                question = (f"**{found.title}** has {len(found.tracks)} songs — which ones? "
                            f"(I couldn't follow {e})")
                return ActionResult("failed", f"❓ {question}", ask=question,
                                    why=f"selector {sel!r}: {e}")
            tracks, note = list(picked.tracks), picked.note
            if not tracks:
                return ActionResult("noop", f"That leaves no songs of **{found.title}** "
                                            f"({len(found.tracks)} tracks) to play.")
            source = f"from **{found.title}** by {found.artist}"
            cover = found.cover
        else:
            tracks = list(found or [])
            if not tracks:
                return ActionResult("failed", f"❌ Couldn't find songs{by} in the catalogue.")
            source = (f"— **{tracks[0].artist}**'s most popular" if a == "top"
                      else f"— **{tracks[0].artist}**'s newest")
            cover = None
        # The lookup awaited the network: re-check everything that could have
        # changed, right before touching the queue (same as _do_play).
        if member.id not in self._auth.get(guild.id, frozenset()):
            return ActionResult("cancelled", None)       # opted out meanwhile: say nothing
        if act is not None:
            reason = self._job_valid(act.job)
            if reason == "revoked":
                return ActionResult("cancelled", None)
            if reason:
                return ActionResult("cancelled", f"✖ Didn't queue the songs {source} — "
                                                 f"{_WHY_INVALID.get(reason, reason)}.")
        if state.queue_epoch != epoch:
            changed = ("The queue was cleared" if state.queue_change_reason == "cleared"
                       else "Playback was stopped")
            return ActionResult("cancelled", f"⏹️ {changed} while I was looking — didn't "
                                             f"queue the songs {source}.")
        vc = guild.voice_client
        if vc is None or not vc.is_connected():
            return ActionResult("cancelled", f"✖ Didn't queue the songs {source} — "
                                             f"{_WHY_INVALID['disconnected']}.")
        copies = max(1, act.copies) if (act is not None and a == "album") else 1
        tracks = tracks * copies
        entries = []
        for n, t in enumerate(tracks):
            entry = {
                "title":          t.query,
                "stream_url":     None,             # found on YouTube when its turn comes
                # A link that shows what will be played, until it's found.
                "webpage_url":    ("https://www.youtube.com/results?search_query="
                                   + quote_plus(t.query)),
                "duration":       t.duration,
                "thumbnail":      cover,
                "requester":      member.display_name,
                "requester_id":   member.id,
                "_needs_resolve": True,
                "_search_query":  t.query,
            }
            if echo is not None:
                entry["request_channel_id"] = echo.id
            if n == 0 and act is not None:
                entry["_voice_job_end"] = act.job.end
            entries.append(entry)
        idle, skipped = self._place_songs(guild, vc, music, act, entries)
        added = tuple(qid(e) for e in entries)
        if act is not None:
            act.started_idle = (idle or skipped) and music is not None   # an advance is coming
        when = (act.intent.when if act is not None else None) or ""
        names = ", ".join(t.title for t in tracks[:5]) + (
            f" and {len(tracks) - 5} more" if len(tracks) > 5 else "")
        count = f"{len(tracks)} song{'s' if len(tracks) != 1 else ''}"
        tail = f" {note}" if note else ""
        if a == "album":
            why = (f"album {found.title!r} by {found.artist} ({len(found.tracks)} tracks, "
                   f"{found.source}); selection {intent.count or 'all'!r} → "
                   + ",".join(str(t.position) for t in tracks[:40]))
        else:
            why = f"{a} for {intent.artist!r}: {len(tracks)} songs"
        if skipped:
            return ActionResult("done", f"⏭️ Skipped — playing {count} {source} now: {names}.{tail}",
                                added=added, why=why)
        if when in ("next", "now") and not idle:
            return ActionResult("done", f"✅ Up next, {count} {source}: {names}.{tail}",
                                added=added, why=why)
        return ActionResult("done", f"✅ Queued {count} {source}: {names}."
                                    + (" Starting…" if idle else "") + tail, added=added, why=why)

    # ── what happened to songs requested by voice ─────────────────────────────

    @commands.Cog.listener()
    async def on_track_audible(self, guild, track) -> None:
        end = track.pop("_voice_job_end", None)
        if end is None:
            return
        self.metrics.stage(guild.id, "play", "started")
        self.metrics.guild(guild.id).audible_seconds.append(time.monotonic() - end)

    @commands.Cog.listener()
    async def on_track_failed(self, guild, track, category) -> None:
        if track.pop("_voice_job_end", None) is not None:
            self.metrics.stage(guild.id, "play", "playback_failed")

    # ── feedback delivery ─────────────────────────────────────────────────────

    def _tts(self):
        tts_cog = self.bot.get_cog("TTS")
        if tts_cog is None:
            return None
        try:
            from core import tts as tts_engine
            return tts_cog if tts_engine.available() else None
        except Exception:
            return None

    async def _deliver_feedback(self, guild, member, heard: str, result: Optional[str],
                                *, force: bool = False, chime: str | None = None) -> None:
        """Reply in the member's chosen style. `force` (questions) always sends
        text, whatever the style. `chime` names the sound for chime mode; a
        `result` of None sends only that sound (e.g. the "listening" ping)."""
        try:
            mode = await self._feedback_for(guild.id, member.id)
        except Exception:
            mode = "text"
        if result is None:
            if mode == "chime" and chime:
                tts_cog = self.bot.get_cog("TTS")
                if tts_cog is not None:
                    tts_cog.chime(guild, chime)
            return
        if mode == "silent" and not force:
            self._typing_stop((guild.id, member.id))
            return
        if mode == "voice":
            tts_cog = self._tts()
            spoken = tts_cog is not None and await tts_cog.say(guild, result)
            if spoken and not force:
                self._typing_stop((guild.id, member.id))
                return                               # spoken instead of posted
            # Questions (force) are spoken AND posted. A spoken "yes or no?"
            # echoing back through someone's speakers can't answer itself:
            # replies captured while the bot talks are ignored.
        if mode == "chime" and chime:
            tts_cog = self.bot.get_cog("TTS")
            if tts_cog is not None:
                tts_cog.chime(guild, chime)
        await self._post_text(guild, member, heard, result)

    async def _post_text(self, guild, member, heard: str, result: str) -> None:
        """Post one reply line. Sending ends the "typing…" indicator."""
        self._typing_stop((guild.id, member.id))
        channel = await self._echo_channel(guild)
        if channel is None:
            return
        shown = f"“{heard}” → " if heard else ""
        text = f"🎙️ {member.display_name}: {shown}{result}"
        try:
            await channel.send(text[:2000], allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            log.debug("voice feedback send failed", exc_info=True)

    async def _echo_channel(self, guild):
        """Where voice-command replies go: the configured channel, else where
        the bot was last summoned by a typed command, else where music was
        last used (remembered across restarts), else the system channel. Never
        "the first channel I can write in" — that put transcripts somewhere
        unexpected."""
        cid = self._cfg.get(guild.id, {}).get("channel_id")
        if cid is None:
            try:
                cid = (await db.get_voice_settings(guild.id))["feedback_channel_id"]
            except Exception:
                cid = None
        if cid:
            ch = guild.get_channel(cid)
            if ch is not None and ch.permissions_for(guild.me).send_messages:
                return ch
        rec = self.bot.get_cog("Recording")
        ch = rec.text_channels.get(guild.id) if rec else None
        # Never a voice channel's own chat, even if that's where the bot was
        # summoned from: replies and the wake cue post on their own, and
        # some servers treat any post there as spam.
        if ch is not None and not isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
            return ch
        # Not summoned by a typed command (autojoin, or since a restart): use
        # where music was last controlled — remembered across restarts.
        last = get_state(guild.id).text_channel_id
        get_channel = getattr(self.bot, "get_channel", None)
        ch = get_channel(last) if (last and get_channel) else None
        if ch is not None and not isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
            return ch
        sysch, me = getattr(guild, "system_channel", None), getattr(guild, "me", None)
        if sysch is not None and me is not None and sysch.permissions_for(me).send_messages:
            return sysch
        return None

    # Text-level entry point (used by tests and as the LLM-tier seam): turn one
    # transcribed line into an Intent, applying the enable + opt-in gates.
    async def handle_utterance(
        self, guild: discord.Guild, member: discord.Member, text: str,
    ) -> Optional[Intent]:
        s = await db.get_voice_settings(guild.id)
        if not s["enabled"]:
            return None
        if not (await db.get_voice_user(guild.id, member.id))["opted_in"]:
            return None
        wake = await self._wake_words(guild.id)
        result = parse_detailed(text, wake)
        intent = result.intent
        if intent is None and result.reason == "no_command" and self.llm is not None:
            reply = await self.llm.ask(text_after_wake(text, wake) or text,
                                       now_playing=self._now_playing(guild.id))
            if reply is not None and reply.kind == "commands":
                intent = reply.intents[0]
        return intent


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(VoiceDJ(bot))
