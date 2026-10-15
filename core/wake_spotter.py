"""
Instant wake word: hear "hey DJ" the moment it's said, like a smart speaker.

Speech-to-text can't do this: it only learns that someone said "hey DJ" after
transcribing them, a second or more after they started talking — longer when
it's busy with someone else. A smart speaker lights up within a fraction of a
second because a tiny model that knows only its wake word listens all the
time; the heavy recognition runs afterwards. This is that tiny model.

It runs on every opted-in speaker's audio as it arrives (in the voice front
end's worker thread, after the 48 → 16 kHz decimator) and scores the last
two seconds every 80 ms. A score over the threshold, near the start of an
utterance, is reported at once with the audio so far. That alone is only
"probably": on held-out synthetic voices it hears ~95 % of "hey DJ"s but
also fires on similar sounds ("hey BJ", "that DJ", "hey, did you…") several
times an hour of speech. So before anything is shown, WakeVerifier asks the
same speech recogniser the bot uses — on just those 1-3 s, on its own thread
and model instance so nobody else's speech is in the way — whether it says
"hey DJ" too. Only then do the cue and "typing…" come, about half a second
after the words, instead of after the whole sentence was transcribed.

How it hears (models in assets/wakeword/):
  * melspectrogram.onnx + embedding_model.onnx — openWakeWord's audio
    features (github.com/dscripka/openWakeWord, v0.5.1 release, Apache-2.0;
    the embedding is Google's speech_embedding model, Apache-2.0): 32 mel
    bands every 10 ms → one 96-number embedding every 80 ms.
  * hey_dj.npz — our classifier over the last 16 embeddings (1.28 s): a small
    two-layer network, trained for this bot on synthetic voices (Piper
    multi-speaker TTS, ~1,000 speakers, put through noise, reverb, mic
    filtering and Discord's Opus codec) against confusable phrases ("hey
    Jay", "hey, did you…", "the DJ"), real speech (LibriSpeech) and general
    audio (ACAV100M features). See assets/wakeword/README.md.

Pure numpy + onnxruntime (both already here for Silero VAD / faster-whisper).
Tests: tests/test_wake_spotter.py.
"""
from __future__ import annotations

import os
import threading
import asyncio
import concurrent.futures
import time
from dataclasses import dataclass, field

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "assets", "wakeword")
CHUNK = 1280                  # 80 ms at 16 kHz: one embedding per chunk
_MEL_CONTEXT = 480            # the mel model's frames overlap the previous chunk this much
_MEL_WINDOWS = 76             # mel frames per embedding (~0.78 s)
_EMB_WINDOWS = 16             # embeddings per decision (1.28 s)
#: The phrase the bundled model knows (as core.voice_intent normalises it).
PHRASE = "hey dj"


@dataclass(frozen=True)
class WakeEvent:
    """The wake word was (probably) just said."""
    guild_id: int
    user_id: int
    at: float                 # capture time of the end of the audio that fired it
    score: float
    onset: float              # when the utterance started (the VAD's speech onset)
    tag: tuple | None = None  # the packets' admission identity (session, consent epoch)
    audio: np.ndarray | None = field(default=None, repr=False, compare=False)
    # ↑ float32 16 kHz: the utterance so far (with pre-roll), for the verifier
    #: "fire" (the score crossed the threshold) or "recheck" (the same
    #: utterance a moment later, asked for when the check heard only "hey…").
    kind: str = field(default="fire", compare=False)
    #: When the front end handed it over (monotonic), for the cue's timing.
    emitted: float | None = field(default=None, compare=False)


class WakeModel:
    """The three models, loaded once and shared by every speaker's stream
    (onnxruntime sessions are safe to run from one thread at a time each;
    the front end has one worker thread)."""

    def __init__(self, directory: str = ASSETS, head: str = "hey_dj.npz"):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        prov = ["CPUExecutionProvider"]
        self._mel = ort.InferenceSession(os.path.join(directory, "melspectrogram.onnx"),
                                         sess_options=opts, providers=prov)
        self._emb = ort.InferenceSession(os.path.join(directory, "embedding_model.onnx"),
                                         sess_options=opts, providers=prov)
        p = np.load(os.path.join(directory, head))
        self._w = [np.ascontiguousarray(p[f"net.{i}.weight"], np.float32) for i in (1, 2, 4, 5, 7)]
        self._b = [np.ascontiguousarray(p[f"net.{i}.bias"], np.float32) for i in (1, 2, 4, 5, 7)]
        self.threshold = round(float(p["threshold"]), 4) if "threshold" in p else 0.5
        self._lock = threading.Lock()
        self._initial: tuple[np.ndarray, np.ndarray] | None = None

    @staticmethod
    def available(directory: str = ASSETS, head: str = "hey_dj.npz") -> bool:
        try:
            import onnxruntime  # noqa: F401  (availability probe)
        except Exception:
            return False
        return all(os.path.isfile(os.path.join(directory, f))
                   for f in ("melspectrogram.onnx", "embedding_model.onnx", head))

    # ── the three stages ─────────────────────────────────────────────────────

    def melspec(self, samples: np.ndarray) -> np.ndarray:
        """16 kHz samples at int16 scale → (frames, 32), openWakeWord's scaling."""
        out = self._mel.run(None, {"input": samples[None, :].astype(np.float32, copy=False)})[0]
        return out.reshape(-1, 32) / 10.0 + 2.0

    def embed(self, mel76: np.ndarray) -> np.ndarray:
        """(76, 32) → (96,)"""
        x = mel76[None, :, :, None].astype(np.float32, copy=False)
        return self._emb.run(None, {"input_1": x})[0].reshape(96)

    def score(self, window: np.ndarray) -> float:
        """(16, 96) embeddings → wake-word probability."""
        h = window.reshape(1, -1).astype(np.float32, copy=False)
        w, b = self._w, self._b
        h = _layer_norm(h @ w[0].T + b[0], w[1], b[1])
        h = np.maximum(h, 0.0)
        h = _layer_norm(h @ w[2].T + b[2], w[3], b[3])
        h = np.maximum(h, 0.0)
        z = float((h @ w[4].T + b[4])[0, 0])
        return 1.0 / (1.0 + np.exp(-z))

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Mel and embedding history as after two and a half seconds of
        silence — what a stream "heard" before its first packet (the
        training clips had quiet before the wake word too)."""
        with self._lock:
            if self._initial is None:
                mel = np.ones((_MEL_WINDOWS, 32), np.float32)
                emb = np.zeros((_EMB_WINDOWS, 96), np.float32)
                hist = np.zeros(_MEL_CONTEXT, np.float32)
                zeros = np.zeros(CHUNK, np.float32)
                for _ in range(32):                  # 2.56 s
                    spec = self.melspec(np.concatenate([hist, zeros]))
                    mel = np.vstack([mel, spec])[-_MEL_WINDOWS:]
                    emb = np.vstack([emb[1:], self.embed(mel)[None]])
                self._initial = (mel, emb)
            return self._initial[0].copy(), self._initial[1].copy()


def _layer_norm(h: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    m = h.mean(-1, keepdims=True)
    v = h.var(-1, keepdims=True)
    return (h - m) / np.sqrt(v + eps) * g + b


class WakeStream:
    """One speaker's running state. Not thread-safe: one owner thread."""

    def __init__(self, model: WakeModel):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self._mel, self._emb = self.model.initial_state()
        self._hist = np.zeros(_MEL_CONTEXT, np.float32)
        self._pending = np.zeros(0, np.float32)

    def push(self, mono16k: np.ndarray) -> list[tuple[int, float]]:
        """Feed float audio in [-1, 1) at 16 kHz. Returns (samples of this
        input consumed up to the scored chunk's end, score) for every 80 ms
        chunk completed — the caller turns the offset into a capture time."""
        x = np.asarray(mono16k, np.float32) * 32767.0
        before = len(self._pending)
        buf = np.concatenate([self._pending, x]) if before else x
        out: list[tuple[int, float]] = []
        n = len(buf) // CHUNK
        for i in range(n):
            chunk = buf[i * CHUNK:(i + 1) * CHUNK]
            spec = self.model.melspec(np.concatenate([self._hist, chunk]))
            self._hist = chunk[-_MEL_CONTEXT:]
            self._mel = np.vstack([self._mel, spec])[-_MEL_WINDOWS:]
            self._emb = np.vstack([self._emb[1:], self.model.embed(self._mel)[None]])
            out.append(((i + 1) * CHUNK - before, self.model.score(self._emb)))
        self._pending = buf[n * CHUNK:]
        return out


def check_words(model, audio, *, window: float = 5.0, language: str | None = "en",
                accept=None, full: bool = True) -> str:
    """What the recogniser hears at the start of a fire's audio — the check
    the cue waits for. Looks, cheapest first, until one hears the wake word
    (`accept(text) -> bool`):

    1. core/whisper.transcribe_short on a short encoder window (the clip
       padded to `window` s): ~0.14 s here instead of the 30-second pass's
       ~0.95 s — most real wakes are accepted at this step;
    2. the same with a 10-second window (a real recording: 5 s heard "and
       DJ" where 8 s and up heard "Hey DJ");
    3. (`full`) the old 30-second pass — the short windows are less sure of
       themselves in noise: on synthetic voices at 20 dB SNR steps 1-2
       accepted 23/40 wake sentences, the 30-second pass 26/40, all three
       29/40; at 10 dB 7, 14 and 15 of 40 (2026-10-14, scripts/
       wake_eval_synth.py). So a hard one still gets the thorough look —
       at the old speed — instead of no fast cue.

    Look-alikes stay rare either way (clean: 6, 8 and 9 of 100). Returns the
    first accepted transcript, else the first look's. window 0 = only the
    30-second pass."""
    from .whisper import AsrOptions, transcribe_command, transcribe_short
    clip = np.asarray(audio, np.float32)[-int(WakeVerifier.MAX_SECONDS * 16000):]
    if not window:
        return transcribe_command(model, clip, AsrOptions(language=language, beam_size=1)).text
    text = transcribe_short(model, clip, language=language, minimum=window, max_tokens=12)
    if accept is None or accept(text):
        return text
    if window < 10:
        again = transcribe_short(model, clip, language=language, minimum=10, max_tokens=12)
        if accept(again):
            return again
    if full:
        last = transcribe_command(model, clip, AsrOptions(language=language, beam_size=1)).text
        if accept(last):
            return last
    return text


class WakeVerifier:
    """Second opinion on a wake-word event: the bot's own speech recogniser
    (faster-whisper; by default the same model as the listener's, as its own
    instance) on just the audio that fired it. Measured on synthetic held-out
    voices (2026-10-13): it accepted none of 60 confusable openings ("hey BJ",
    "hey Jay", "hey, did you…") and about the same share of real "hey DJ"s as
    it does when the whole sentence is transcribed — so the cue and what the
    bot does agree. How much audio after the words it gets barely matters
    (-40 ms … +2 s: 45/80 … 44/80 on deliberately hard clips), so it runs on
    what fired, at once."""

    MAX_SECONDS = 3.0

    def __init__(self, model, transcribe, *, threads: int = 2):
        self._model = model
        self._transcribe = transcribe           # (model, audio) -> text
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="wake-verify")
        self.busy = False
        self.checks = 0
        self.last_seconds: float | None = None
        self.seconds: list[float] = []

    async def heard(self, audio: np.ndarray) -> tuple[str, float]:
        """(what the recogniser heard, seconds it took). One check at a time
        (the cog queues the rest, fairly across speakers); a call while one
        is running waits for its own turn on the verifier's thread."""
        self.busy = True
        started = time.monotonic()
        try:
            clip = np.asarray(audio, np.float32)[-int(self.MAX_SECONDS * 16000):]
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(self._pool, self._transcribe, self._model, clip)
        finally:
            self.busy = False
        took = time.monotonic() - started
        self.checks += 1
        self.last_seconds = took
        self.seconds = (self.seconds + [took])[-50:]
        return text or "", took

    def median_seconds(self) -> float | None:
        if not self.seconds:
            return None
        s = sorted(self.seconds)
        return s[len(s) // 2]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
