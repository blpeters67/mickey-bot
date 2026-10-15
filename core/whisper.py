"""Shared faster-whisper helpers — the model list and loader used by both the
voice-command listener (cogs/voicedj.py) and `-transcribe` (cogs/downloads.py),
plus the voice-command transcription call itself.

Kept import-light: faster-whisper is imported lazily, so neither cog needs it
installed just to load.
"""
from __future__ import annotations

from dataclasses import dataclass

# Sizes offered by the `voice model` / `transcribemodel` commands. English-only
# `.en` variants are the most accurate when the language is known; bigger =
# more accurate but slower.
KNOWN_MODELS = (
    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
    "distil-small.en", "medium.en", "medium", "distil-medium.en",
    "large-v3", "large-v3-turbo",
)


def load_model(name: str, *, device: str = "cpu", compute_type: str = "int8",
               cpu_threads: int = 0):
    """Load a faster-whisper model. Blocks — run it off the event loop.

    Defaults (CPU, int8, CTranslate2's own thread count) are the baseline the
    voice review recommends measuring from; the voice listener passes the
    VOICE_DEVICE / VOICE_COMPUTE_TYPE / VOICE_CPU_THREADS settings through."""
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=device, compute_type=compute_type,
                        cpu_threads=cpu_threads)


@dataclass(frozen=True)
class AsrOptions:
    """How voice commands are transcribed. Explicit on purpose — the old call
    relied on faster-whisper defaults (temperature fallback ladder, context
    conditioning) that only add latency or hallucination risk to 1–3 s clips."""
    language: str | None = "en"
    beam_size: int = 1
    prompt: str | None = None        # vocabulary hint; opt-in (see VOICE_ASR_HINTS)


@dataclass(frozen=True)
class AsrResult:
    text: str
    avg_logprob: float               # mean over segments; diagnostic, not a probability
    no_speech_prob: float            # max over segments
    audio_seconds: float


def transcribe_command(model, audio, opts: AsrOptions) -> AsrResult:
    """Transcribe one short voice-command clip (float32 mono 16 kHz).

    * temperature 0 only — deterministic, and no retries at higher
      temperatures that mostly produce confident nonsense on short clips;
    * condition_on_previous_text off and no timestamps — single window;
    * whisper's own VAD off — the front end already cut the clip to speech,
      and trimming twice eats word edges;
    * whisper's standard silence rule kept (no_speech > 0.6 with low log-prob
      → the segment is dropped), and the numbers are returned for diagnostics.
    """
    segments, _info = model.transcribe(
        audio,
        language=opts.language,
        task="transcribe",
        beam_size=opts.beam_size,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=False,
        initial_prompt=opts.prompt or None,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    if segs:
        avg = sum(s.avg_logprob for s in segs) / len(segs)
        nsp = max(s.no_speech_prob for s in segs)
    else:
        avg, nsp = float("-inf"), 1.0
    return AsrResult(text, avg, nsp, len(audio) / 16_000)


# ── a few words, fast: the wake word's second opinion ─────────────────────────

_TOKENIZERS: dict[int, object] = {}


def _tokenizer(model, language: str | None):
    key = id(model)
    tk = _TOKENIZERS.get(key)
    if tk is None:
        from faster_whisper.tokenizer import Tokenizer
        multilingual = bool(model.model.is_multilingual)
        tk = Tokenizer(model.hf_tokenizer, multilingual, task="transcribe",
                       language=(language or "en") if multilingual else None)
        _TOKENIZERS[key] = tk
    return tk


def short_window_seconds(audio_seconds: float, *, minimum: float = 5.0,
                         pad: float = 2.0) -> int:
    """The encoder window for a clip: the clip plus `pad` seconds of silence
    (Whisper expects to hear the end of speech), at least `minimum`, whole
    seconds, at most Whisper's own 30."""
    import math
    return int(min(30, max(minimum, math.ceil(audio_seconds + pad))))


def transcribe_short(model, audio, *, language: str | None = "en", max_tokens: int = 16,
                     minimum: float = 5.0, pad: float = 2.0) -> str:
    """The first words of a short clip (float32 mono 16 kHz, ≤ ~10 s), fast.

    faster-whisper always runs the encoder over a 30-second window, padding a
    3-second clip with 27 seconds of silence — ~0.9 s on the bot's host for
    the wake word's second opinion. CTranslate2's Whisper encoder takes a
    shorter window: the clip plus a little silence (≥ 5 s) costs ~0.05 s, and
    only `max_tokens` tokens are decoded. Greedy, no timestamps, no prompt.
    Measured against the 30-second window in assets/wakeword/README.md: the
    same words on the wake-word clips and recordings checked; not meant for
    whole sentences (accuracy beyond a few words wasn't the question)."""
    import ctranslate2
    import numpy as np
    x = np.asarray(audio, dtype=np.float32)
    secs = short_window_seconds(len(x) / 16_000, minimum=minimum, pad=pad)
    padded = np.zeros(secs * 16_000, dtype=np.float32)
    padded[:min(len(x), len(padded))] = x[:len(padded)]
    feats = model.feature_extractor(padded)[:, :secs * 100]
    enc = model.model.encode(ctranslate2.StorageView.from_array(
        np.ascontiguousarray(feats[None], dtype=np.float32)), to_cpu=False)
    tk = _tokenizer(model, language)
    prompt = list(tk.sot_sequence) + [tk.no_timestamps]
    res = model.model.generate(enc, [prompt], beam_size=1, max_length=len(prompt) + max_tokens,
                               suppress_blank=True)
    ids = [t for t in res[0].sequences_ids[0] if t < tk.eot]
    return tk.decode(ids).strip()
