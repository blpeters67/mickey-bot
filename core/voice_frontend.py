"""
Voice-command audio front end: Opus packets in → speech segments out.

This replaces "an utterance is whatever arrived until packets paused for
600 ms", which failed in both directions: an open mic (noise, music bleed,
keyboard) never paused, so after 8 s the whole chunk — command included — was
thrown away; and two separate bursts merged whenever the 0.4 s poll missed the
gap between them, pushing the wake word off the front of the transcript.

Per speaker, every 20 ms frame goes through:

  1. **A persistent Opus decoder** (one per speaker, reset when their SSRC
     changes). Lost packets — detected from RTP sequence numbers — are
     concealed with in-band FEC / Opus PLC instead of being skipped, so time
     isn't compressed and syllables aren't silently dropped. A pause in
     transmission (Discord's DTX) becomes real silence, sized from the RTP
     timestamp, so pauses between words survive.
  2. **Downmix + a proper anti-aliased 48→16 kHz decimator** (windowed-sinc
     FIR, streaming state). `audioop.ratecv` had no low-pass filter, so 8–24
     kHz energy folded back into the speech band.
  3. **Voice activity detection** on the decoded audio: Silero VAD v6, run
     streaming through onnxruntime from the model file faster-whisper already
     ships (no extra dependency), with an energy-based fallback.
  4. **A segmenter**: pre-roll before speech, hysteresis, an end-of-speech
     silence, and at the length cap a deliberate split with overlap rather
     than discarding the audio.

Decrypted Opus comes from our own sink (the recorder decrypts DAVE), so this
never relies on voice-recv's decoder. Pure Python + numpy; no Discord objects —
the cog owns threading, consent and transcription. Tests: tests/test_voice_frontend.py.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

IN_RATE = 48_000
OUT_RATE = 16_000
FRAME_SAMPLES = 960                  # 20 ms at 48 kHz
FRAME_SECONDS = 0.02
VAD_WINDOW = 512                     # 32 ms at 16 kHz (Silero's window)
VAD_WINDOW_S = VAD_WINDOW / OUT_RATE


# ──────────────────────────────────────────────────────────────────────────────
# 48 kHz → 16 kHz decimator
# ──────────────────────────────────────────────────────────────────────────────

def _design_lowpass(numtaps: int = 127, cutoff_hz: float = 7200.0,
                    rate: float = IN_RATE) -> np.ndarray:
    """Blackman-windowed sinc low-pass, unity DC gain. ~-74 dB stopband: with
    this cutoff everything that would alias into 0–8 kHz is gone."""
    fc = cutoff_hz / rate
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(numtaps)
    return (h / h.sum()).astype(np.float64)


_LOWPASS = _design_lowpass()


class Decimator:
    """Streaming 3:1 decimator (48 → 16 kHz) with filter state carried across
    calls, so chunk boundaries are seamless."""

    def __init__(self, taps: np.ndarray = _LOWPASS, factor: int = 3):
        self.h = taps
        self.factor = factor
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros(len(self.h) - 1)
        self._phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if not len(x):
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate([self._hist, x.astype(np.float64, copy=False)])
        y = np.convolve(buf, self.h, mode="valid")          # len(x) outputs
        out = y[self._phase::self.factor]
        self._phase = (self._phase - len(x)) % self.factor
        self._hist = buf[-(len(self.h) - 1):]
        return out.astype(np.float32)


def pcm48_stereo_to_mono_float(pcm: bytes) -> np.ndarray:
    """Discord's s16le 48 kHz stereo → mono float32 in [-1, 1)."""
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if a.size % 2:
        a = a[:-1]
    return (a[0::2] + a[1::2]) * (0.5 / 32768.0)


# ──────────────────────────────────────────────────────────────────────────────
# Voice activity detection
# ──────────────────────────────────────────────────────────────────────────────

class SileroVAD:
    """Streaming Silero VAD v6 over onnxruntime. One instance per speaker — it
    carries LSTM state and the 64-sample context between 512-sample windows,
    which is what makes window-by-window results match whole-file results."""

    CONTEXT = 64
    _session = None
    _session_lock = threading.Lock()

    def __init__(self):
        self._session_get()
        self.reset()

    @classmethod
    def model_path(cls) -> str | None:
        try:
            from faster_whisper.utils import get_assets_path
        except Exception:
            return None
        for name in ("silero_vad_v6.onnx", "silero_vad.onnx"):
            path = os.path.join(get_assets_path(), name)
            if os.path.isfile(path):
                return path
        return None

    @classmethod
    def available(cls) -> bool:
        try:
            import onnxruntime  # noqa: F401  (availability probe)
        except Exception:
            return False
        return cls.model_path() is not None

    @classmethod
    def _session_get(cls):
        with cls._session_lock:
            if cls._session is None:
                import onnxruntime
                path = cls.model_path()
                if path is None:
                    raise RuntimeError("Silero VAD model not found (faster-whisper missing?)")
                opts = onnxruntime.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                opts.log_severity_level = 4
                cls._session = onnxruntime.InferenceSession(
                    path, providers=["CPUExecutionProvider"], sess_options=opts)
                names = [i.name for i in cls._session.get_inputs()]
                if names[:3] != ["input", "h", "c"]:
                    cls._session = None
                    raise RuntimeError(f"unexpected Silero model inputs: {names}")
            return cls._session

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)

    def window(self, w: np.ndarray) -> float:
        """Speech probability for one 512-sample window at 16 kHz."""
        inp = np.concatenate([self._context, w.astype(np.float32, copy=False)])[None, :]
        out, self._h, self._c = self._session.run(
            None, {"input": inp, "h": self._h, "c": self._c})
        self._context = inp[0, -self.CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Fallback when Silero isn't available: frame energy against an adaptive
    noise floor, squashed into a 0–1 "probability". Much weaker than Silero
    (it can't tell speech from music bleed), but it still gives proper
    segment boundaries."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._floor_db = -60.0

    def window(self, w: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(w, dtype=np.float64)))) if len(w) else 0.0
        db = 20 * np.log10(rms + 1e-9)
        # The floor follows quiet stretches quickly and loud ones slowly.
        rate = 0.05 if db < self._floor_db else 0.002
        self._floor_db += (db - self._floor_db) * rate
        self._floor_db = min(max(self._floor_db, -80.0), -25.0)
        return float(np.clip((db - self._floor_db - 9.0) / 9.0, 0.0, 1.0))


def make_vad():
    """Silero if possible, else the energy fallback."""
    if SileroVAD.available():
        try:
            return SileroVAD()
        except Exception as e:                                # pragma: no cover
            log.warning("Silero VAD unavailable (%s) — using energy VAD", e)
    return EnergyVAD()


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmenterConfig:
    """Starting points, not tuned optima — see the review's "tuning proposals".
    Adjust from `voice diag` numbers on the real microphones."""
    threshold: float = 0.5          # speech starts at/above this probability
    neg_threshold: float = 0.35     # …and continues until it drops below this
    preroll_ms: int = 320           # audio kept from before speech started
    end_silence_ms: int = 600       # this much quiet ends an utterance
    trail_ms: int = 160             # quiet kept after the last speech window
    min_speech_ms: int = 160        # shorter blips are dropped (clicks, "uh")
    max_segment_ms: int = 10_000    # split (not drop) anything longer…
    overlap_ms: int = 800           # …carrying this much into the next part
    dtx_timeout_ms: int = 600       # no packets this long mid-speech = ended
    #: Once an utterance has run this long, hand the speech so far to the
    #: probe callback — once per utterance — so the wake word can be spotted
    #: while the person is still talking (the "typing…" indicator). 0 = off.
    probe_ms: int = 0


@dataclass
class Probe:
    """The first part of an utterance that's still going (see probe_ms)."""
    guild_id: int
    user_id: int
    audio: np.ndarray               # float32 mono 16 kHz, from the padded start
    start: float                    # capture time of the first sample
    onset: float                    # when speech began
    tag: tuple | None = None


@dataclass
class Segment:
    """One utterance from one speaker, ready for speech-to-text."""
    guild_id: int
    user_id: int
    audio: np.ndarray               # float32 mono 16 kHz
    start: float                    # monotonic capture time of the first sample (incl. pre-roll)
    end: float                      # …of the last speech sample
    speech_ms: int
    reason: str                     # silence | max_length | timeout | flush
    overlap_ms: int = 0             # leading audio repeated from the previous part
    seq: int = 0                    # per-speaker counter
    #: When the VAD says speech actually began. `start` includes up to
    #: `preroll_ms` of audio from BEFORE that (kept so whisper hears the first
    #: syllable whole) — timing decisions ("was this said after the question?")
    #: must use the onset, not the padded start.
    speech_start: float | None = None
    #: Immutable identity stamped on the packets when they were admitted
    #: (the cog uses (voice session, consent epoch)). A segment carries its
    #: packets' tag, never "whatever is current when it reaches the loop".
    tag: tuple | None = None

    @property
    def duration(self) -> float:
        return len(self.audio) / OUT_RATE

    @property
    def onset(self) -> float:
        return self.start if self.speech_start is None else self.speech_start


class Segmenter:
    """Turns a stream of (window, probability, capture time) into Segments.

    ``emit(audio, start, end, speech_ms, reason, overlap_ms, onset)`` — `start`
    is the padded audio start, `onset` the first speech window's start."""

    def __init__(self, cfg: SegmenterConfig,
                 emit: Callable[[np.ndarray, float, float, int, str, int, float], None],
                 probe: Callable[[np.ndarray, float, float], None] | None = None):
        self.cfg = cfg
        self._emit = emit
        self._probe = probe
        self._win_ms = VAD_WINDOW_S * 1000
        self.dropped_short = 0
        self.reset()

    def reset(self) -> None:
        self._pre: deque = deque(maxlen=max(1, round(self.cfg.preroll_ms / self._win_ms)))
        self._seg: list[np.ndarray] = []
        self._seg_start = 0.0
        self._onset_t = 0.0
        self._last_speech_t = 0.0
        self._last_speech_idx = -1
        self._speech_windows = 0
        self._silence_run = 0
        self._in_speech = False
        self._overlap_ms = 0
        self._probed = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, w: np.ndarray, prob: float, t: float) -> None:
        cfg = self.cfg
        if not self._in_speech:
            if prob >= cfg.threshold:
                self._in_speech = True
                self._seg = list(self._pre) + [w]
                self._seg_start = t - (len(self._pre) + 1) * VAD_WINDOW_S
                self._onset_t = t - VAD_WINDOW_S      # this window is the first speech
                self._pre.clear()
                self._speech_windows = 1
                self._silence_run = 0
                self._last_speech_idx = len(self._seg) - 1
                self._last_speech_t = t
            else:
                self._pre.append(w)
            return

        self._seg.append(w)
        if (self._probe is not None and cfg.probe_ms and not self._probed
                and (t - self._onset_t) * 1000 >= cfg.probe_ms):
            self._probed = True                       # once per utterance
            self._probe(np.concatenate(self._seg), self._seg_start, self._onset_t)
        if prob >= cfg.neg_threshold:
            self._speech_windows += 1
            self._silence_run = 0
            self._last_speech_idx = len(self._seg) - 1
            self._last_speech_t = t
        else:
            self._silence_run += 1
            if self._silence_run * self._win_ms >= cfg.end_silence_ms:
                self._finish("silence")
                return
        if len(self._seg) * self._win_ms >= cfg.max_segment_ms:
            self._split()

    def timeout(self) -> None:
        """No packets for a while (DTX or a network stall) mid-speech: end it.
        Network liveness is separate from VAD — this only fires while in speech."""
        if self._in_speech:
            self._finish("timeout")

    def flush(self) -> None:
        if self._in_speech:
            self._finish("flush")
        self.reset()

    def _finish(self, reason: str) -> None:
        keep = self._last_speech_idx + 1 + round(self.cfg.trail_ms / self._win_ms)
        windows = self._seg[:keep]
        speech_ms = int(self._speech_windows * self._win_ms)
        if windows and speech_ms >= self.cfg.min_speech_ms:
            self._emit(np.concatenate(windows), self._seg_start, self._last_speech_t,
                       speech_ms, reason, self._overlap_ms, self._onset_t)
        else:
            self.dropped_short += 1
        tail = self._seg[keep:]
        self.reset()
        for w in tail[-self._pre.maxlen:]:
            self._pre.append(w)

    def _split(self) -> None:
        """At the length cap: emit what we have and keep going, starting the
        next part with an overlap so a wake word straddling the cut survives.
        (The cog de-duplicates a command heard in both parts.)"""
        windows = self._seg
        speech_ms = int(self._speech_windows * self._win_ms)
        self._emit(np.concatenate(windows), self._seg_start, self._last_speech_t,
                   speech_ms, "max_length", self._overlap_ms, self._onset_t)
        n_overlap = max(1, round(self.cfg.overlap_ms / self._win_ms))
        carry = windows[-n_overlap:]
        self._seg = list(carry)
        self._seg_start = self._last_speech_t - (len(carry) - 1) * VAD_WINDOW_S
        self._onset_t = self._seg_start           # the carried windows are speech
        self._speech_windows = len(carry)
        self._last_speech_idx = len(self._seg) - 1
        self._silence_run = 0
        self._overlap_ms = int(len(carry) * self._win_ms)


# ──────────────────────────────────────────────────────────────────────────────
# Per-speaker stream: packets → 16 kHz windows → segmenter
# ──────────────────────────────────────────────────────────────────────────────

_MAX_CONCEAL = 10        # conceal up to 200 ms of lost packets; more = discontinuity
_MAX_DTX_FILL = 75       # cap silence inserted for a transmission pause (1.5 s)


@dataclass
class StreamStats:
    packets: int = 0
    concealed: int = 0           # lost packets filled by FEC/PLC
    discontinuities: int = 0     # gaps too big to conceal
    decode_errors: int = 0
    dtx_fill_frames: int = 0
    vad_windows: int = 0
    speech_windows: int = 0
    segments: int = 0
    last_packet: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


#: The instant wake word (core/wake_spotter.py) only counts near the start of
#: an utterance — where the grammar accepts it — and fires once per this long.
WAKE_LEAD_S = 4.0
WAKE_AFTER_END_S = 0.7
#: Short: an early fire on "hey D…" is turned down by the verifier, and the
#: right one a moment later must still get through. (The cog acts on one
#: confirmed fire per utterance.)
WAKE_REFRACTORY_S = 0.4


class SpeakerStream:
    """Front end for one speaker. Not thread-safe: one owner thread."""

    def __init__(self, guild_id: int, user_id: int, cfg: SegmenterConfig,
                 emit: Callable[[Segment], None], *,
                 decoder_factory: Callable[[], object], vad_factory: Callable[[], object],
                 tag: tuple | None = None,
                 probe: Callable[[Probe], None] | None = None,
                 wake: Callable[[object], None] | None = None,
                 wake_model=None, wake_threshold: float = 0.5):
        self.guild_id, self.user_id = guild_id, user_id
        self.tag = tag
        self.stats = StreamStats()
        self._decoder_factory = decoder_factory
        self._vad_factory = vad_factory
        self._emit_segment = emit
        self._emit_probe = probe
        self._emit_wake = wake if wake_model is not None else None
        self._wake_model = wake_model
        self.wake_threshold = wake_threshold
        self.wake_scores = 0
        self.wake_fired = 0
        self._last_wake = -1e9
        self._last_score = 0.0
        #: Stream time after which to hand the utterance over once more (the
        #: check heard only "hey…": the rest of the wake word was still coming).
        self.recheck_after: float | None = None
        self._last_utt: tuple[float, float] | None = None     # (onset, end) of the last segment
        self._last_utt_audio: np.ndarray | None = None
        self._seq_counter = 0
        self.segmenter = Segmenter(cfg, self._emit, self._probe if probe else None)
        self._reset_stream(None)

    # ── state ────────────────────────────────────────────────────────────────

    def _reset_stream(self, ssrc) -> None:
        self._ssrc = ssrc
        self._decoder = self._decoder_factory()
        self._dec48 = Decimator()
        self._vad = self._vad_factory()
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_seq: int | None = None
        self._last_ts: int | None = None
        self._last_arrival: float | None = None
        self._spotter = None
        if self._emit_wake is not None:
            from .wake_spotter import WakeStream
            self._spotter = WakeStream(self._wake_model)

    def _probe(self, audio, start, onset) -> None:
        self._emit_probe(Probe(self.guild_id, self.user_id, audio, start, onset, self.tag))

    def _emit(self, audio, start, end, speech_ms, reason, overlap_ms, onset=None) -> None:
        self._seq_counter += 1
        self.stats.segments += 1
        self._last_utt = (start if onset is None else onset, end)
        self._last_utt_audio = audio if self._emit_wake is not None else None
        self._emit_segment(Segment(self.guild_id, self.user_id, audio, start, end,
                                   speech_ms, reason, overlap_ms, self._seq_counter,
                                   speech_start=onset, tag=self.tag))

    def _utterance_onset(self, t: float) -> float | None:
        """The onset of the utterance a wake word heard at `t` would open, or
        None if `t` isn't near the start of one (the grammar only takes the
        wake word at the start, so a cue there could only be wrong)."""
        seg = self.segmenter
        if seg.in_speech:
            return seg._onset_t if t - seg._onset_t <= WAKE_LEAD_S else None
        if self._last_utt is not None:
            onset, end = self._last_utt
            if t - end <= WAKE_AFTER_END_S and end - onset <= WAKE_LEAD_S:
                return onset
        return None

    def _spot(self, x16: np.ndarray, t_end: float) -> None:
        if self.recheck_after is not None and t_end >= self.recheck_after:
            self.recheck_after = None
            self._recheck(t_end)
        for consumed, score in self._spotter.push(x16):
            self.wake_scores += 1
            self._last_score = score
            if score < self.wake_threshold:
                continue
            t = t_end - (len(x16) - consumed) / OUT_RATE
            if t - self._last_wake < WAKE_REFRACTORY_S:
                continue
            onset = self._utterance_onset(t)
            if onset is None:
                continue
            self._last_wake = t
            self.wake_fired += 1
            seg = self.segmenter
            audio = (np.concatenate(seg._seg) if seg.in_speech and seg._seg
                     else self._last_utt_audio)
            from .wake_spotter import WakeEvent
            self._emit_wake(WakeEvent(self.guild_id, self.user_id, t, score, onset, self.tag,
                                      audio=audio))

    def _recheck(self, t: float) -> None:
        """The same utterance once more, with the audio since the last fire."""
        onset = self._utterance_onset(t)
        if onset is None:
            return
        seg = self.segmenter
        audio = (np.concatenate(seg._seg) if seg.in_speech and seg._seg
                 else self._last_utt_audio)
        if audio is None:
            return
        from .wake_spotter import WakeEvent
        self._emit_wake(WakeEvent(self.guild_id, self.user_id, t, self._last_score, onset,
                                  self.tag, audio=audio, kind="recheck"))

    # ── input ────────────────────────────────────────────────────────────────

    def push(self, opus: bytes, seq: int | None, ts: int | None, ssrc,
             arrival: float) -> None:
        st = self.stats
        if ssrc is not None and ssrc != self._ssrc:
            if self._ssrc is not None:
                self.segmenter.flush()           # a new stream — don't splice audio
            self._reset_stream(ssrc)

        lost = 0
        dtx_frames = 0
        if self._last_seq is not None and seq is not None:
            gap = (seq - self._last_seq) & 0xFFFF
            if gap == 0 or gap > 0x8000:
                return                            # duplicate / stale reordering
            lost = gap - 1
            if lost > _MAX_CONCEAL:
                st.discontinuities += 1
                lost = 0
                self.segmenter.timeout()          # the speech we were in is over
        if self._last_ts is not None and ts is not None and lost == 0:
            step = (ts - self._last_ts) & 0xFFFFFFFF
            frames = step // FRAME_SAMPLES
            if 1 < frames <= 50 * 60:
                dtx_frames = frames - 1           # a transmission pause, per RTP
        # Without RTP timestamps, fall back to arrival spacing. With them,
        # trust the sender's clock: arrival gaps are mostly network jitter,
        # and padding those with "silence" would split words apart.
        if ts is None and lost == 0 and self._last_arrival is not None:
            gap_s = arrival - self._last_arrival
            if gap_s > 0.08:
                dtx_frames = int(round(gap_s / FRAME_SECONDS)) - 1

        if dtx_frames:
            n = min(dtx_frames, _MAX_DTX_FILL)
            st.dtx_fill_frames += n
            self._feed(np.zeros(FRAME_SAMPLES * n, dtype=np.float32),
                       arrival - (n + 1) * FRAME_SECONDS)

        frames: list[np.ndarray] = []
        if lost:
            st.concealed += lost
            for k in range(lost):
                try:
                    if k == lost - 1:              # in-band FEC rebuilds the last one
                        pcm = self._decoder.decode(opus, fec=True)
                    else:
                        pcm = self._decoder.decode(None, fec=False)   # PLC
                    frames.append(pcm48_stereo_to_mono_float(pcm))
                except Exception:
                    st.decode_errors += 1
                    frames.append(np.zeros(FRAME_SAMPLES, dtype=np.float32))
        try:
            frames.append(pcm48_stereo_to_mono_float(self._decoder.decode(opus, fec=False)))
        except Exception:
            # Keep time continuous: a failed frame becomes 20 ms of silence
            # (and is counted) instead of vanishing.
            st.decode_errors += 1
            frames.append(np.zeros(FRAME_SAMPLES, dtype=np.float32))

        st.packets += 1
        st.last_packet = arrival
        self._last_seq = seq if seq is not None else self._last_seq
        self._last_ts = ts if ts is not None else self._last_ts
        self._last_arrival = arrival
        start_t = arrival - (len(frames) - 1) * FRAME_SECONDS
        self._feed(np.concatenate(frames), start_t)

    def tick(self, now: float, timeout_s: float) -> None:
        """Called regularly: ends speech when packets stop arriving."""
        if (self.segmenter.in_speech and self._last_arrival is not None
                and now - self._last_arrival >= timeout_s):
            self.segmenter.timeout()
            self._pending = np.zeros(0, dtype=np.float32)

    def _feed(self, mono48: np.ndarray, t0: float) -> None:
        x = self._dec48.process(mono48)
        buf = np.concatenate([self._pending, x]) if len(self._pending) else x
        n = len(buf) // VAD_WINDOW
        st = self.stats
        # Capture time of each window ≈ time of its last sample.
        t_end = t0 + len(mono48) / IN_RATE
        for i in range(n):
            w = buf[i * VAD_WINDOW:(i + 1) * VAD_WINDOW]
            prob = self._vad.window(w)
            st.vad_windows += 1
            if prob >= self.segmenter.cfg.neg_threshold:
                st.speech_windows += 1
            t = t_end - (len(buf) - (i + 1) * VAD_WINDOW) / OUT_RATE
            self.segmenter.push(w, prob, t)
        self._pending = buf[n * VAD_WINDOW:]
        if self._spotter is not None and len(x):
            self._spot(x, t_end)


# ──────────────────────────────────────────────────────────────────────────────
# The front end: one worker thread for every speaker
# ──────────────────────────────────────────────────────────────────────────────

def _discord_decoder():
    from discord.opus import Decoder
    return Decoder()


class VoiceFrontEnd:
    """Owns a worker thread that turns queued packets into Segments.

    `feed` is called on voice-recv's receive thread and only enqueues (cheap);
    decoding, resampling and VAD happen on the worker. `on_segment` is called
    on the worker thread — the cog hops it onto the event loop.

    Consent is enforced by the caller deciding what to feed; `drop_speaker` /
    `drop_guild` purge a person's state *and* anything of theirs still queued
    (via epochs), so revoked consent can't leak through the queue. The guild
    generation is separate from the per-speaker epochs: a first packet still
    waiting in the queue has no speaker state yet, and dropping the guild must
    invalidate it too.

    `feed(..., tag=…)` attaches the caller's identity for the packet (the voice
    cog passes (session, consent epoch) read at admission). A speaker's stream
    never mixes tags — a packet with a different tag starts a fresh stream —
    and every Segment carries the tag of the packets it was made from.
    """

    def __init__(self, on_segment: Callable[[Segment], None], *,
                 on_probe: Callable[[Probe], None] | None = None,
                 on_wake: Callable[[object], None] | None = None,
                 wake_model=None, wake_threshold: float | None = None,
                 cfg: SegmenterConfig | None = None,
                 decoder_factory: Callable[[], object] = _discord_decoder,
                 vad_factory: Callable[[], object] = make_vad,
                 clock: Callable[[], float] = time.monotonic,
                 max_queue: int = 3000):
        self.cfg = cfg or SegmenterConfig()
        self._on_segment = on_segment
        self._on_probe = on_probe
        # The instant wake word (core/wake_spotter.py): only with both a
        # callback and a loaded model.
        self._on_wake = on_wake if wake_model is not None else None
        self.wake_model = wake_model
        self.wake_threshold = (wake_threshold if wake_threshold is not None
                               else getattr(wake_model, "threshold", 0.5))
        self._decoder_factory = decoder_factory
        self._vad_factory = vad_factory
        self._clock = clock
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._streams: dict[tuple[int, int], SpeakerStream] = {}
        self._epochs: dict[tuple[int, int], int] = {}
        self._guild_gen: dict[int, int] = {}
        self._overrides: dict[tuple[int, int], dict] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.queue_overflows = 0
        self.errors = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="voice-frontend", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── called from other threads ────────────────────────────────────────────

    def feed(self, guild_id: int, user_id: int, opus: bytes, seq, ts, ssrc,
             arrival: float | None = None, *, tag: tuple | None = None) -> None:
        key = (guild_id, user_id)
        item = (key, self._epochs.get(key, 0), self._guild_gen.get(guild_id, 0), tag,
                opus, seq, ts, ssrc, self._clock() if arrival is None else arrival)
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.queue_overflows += 1

    def drop_speaker(self, guild_id: int, user_id: int) -> None:
        """Forget a speaker now: partial speech and queued packets are discarded
        and nothing more is emitted for audio fed before this call."""
        key = (guild_id, user_id)
        with self._lock:
            self._epochs[key] = self._epochs.get(key, 0) + 1
            self._streams.pop(key, None)

    def drop_guild(self, guild_id: int) -> None:
        """Forget everything in a guild, including packets of speakers the
        worker hasn't seen yet (they carry the old guild generation)."""
        with self._lock:
            self._guild_gen[guild_id] = self._guild_gen.get(guild_id, 0) + 1
            for key in [k for k in set(self._streams) | set(self._epochs) if k[0] == guild_id]:
                self._epochs[key] = self._epochs.get(key, 0) + 1
                self._streams.pop(key, None)

    def set_end_silence(self, guild_id: int, user_id: int, ms: int | None) -> None:
        """Shorter end-of-speech while someone owes a yes/no or just said the
        wake word; None restores the default."""
        key = (guild_id, user_id)
        with self._lock:
            stream = self._streams.get(key)
            if ms is None:
                self._overrides.pop(key, None)
                if stream:
                    stream.segmenter.cfg = self.cfg
            else:
                self._overrides[key] = {"end_silence_ms": ms}
                if stream:
                    stream.segmenter.cfg = SegmenterConfig(**{**self.cfg.__dict__,
                                                              "end_silence_ms": ms})

    def stats(self, guild_id: int | None = None) -> dict[tuple[int, int], dict]:
        with self._lock:
            out = {}
            for k, s in self._streams.items():
                if guild_id is None or k[0] == guild_id:
                    d = s.stats.as_dict()
                    d["dropped_short"] = s.segmenter.dropped_short
                    d["in_speech"] = s.segmenter.in_speech
                    d["wake_scores"] = s.wake_scores
                    d["wake_fired"] = s.wake_fired
                    out[k] = d
            return out

    def request_wake_recheck(self, guild_id: int, user_id: int, at: float) -> None:
        """Hand this speaker's utterance to the wake callback once more when
        their audio reaches stream time `at` (see SpeakerStream._recheck)."""
        with self._lock:
            stream = self._streams.get((guild_id, user_id))
            if stream is not None:
                stream.recheck_after = at

    def active_speech(self, key) -> tuple[float, tuple | None] | None:
        """Snapshot the onset and admission identity of unfinished speech."""
        with self._lock:
            stream = self._streams.get(key)
            if stream is not None and stream.segmenter.in_speech:
                return stream.segmenter._onset_t, stream.tag
        return None

    # ── worker ───────────────────────────────────────────────────────────────

    def _stream_for(self, key, tag=None) -> SpeakerStream:
        stream = self._streams.get(key)
        if stream is not None and stream.tag != tag:
            # New identity (the session or this person's consent changed):
            # never splice their audio across it — start over.
            stream = None
        if stream is None:
            cfg = self.cfg
            if key in self._overrides:
                cfg = SegmenterConfig(**{**self.cfg.__dict__, **self._overrides[key]})
            epoch = self._epochs.get(key, 0)
            ggen = self._guild_gen.get(key[0], 0)
            probe = None
            if self._on_probe is not None:
                probe = (lambda pr, _key=key, _epoch=epoch, _g=ggen:
                         self._deliver_probe(_key, _epoch, _g, pr))
            wake = None
            if self._on_wake is not None:
                wake = (lambda ev, _key=key, _epoch=epoch, _g=ggen:
                        self._deliver_wake(_key, _epoch, _g, ev))
            stream = SpeakerStream(
                key[0], key[1], cfg,
                lambda seg, _key=key, _epoch=epoch, _g=ggen: self._deliver(_key, _epoch, _g, seg),
                decoder_factory=self._decoder_factory, vad_factory=self._vad_factory,
                tag=tag, probe=probe, wake=wake, wake_model=self.wake_model,
                wake_threshold=self.wake_threshold)
            self._streams[key] = stream
        return stream

    def _valid(self, key, epoch, ggen) -> bool:
        return (self._epochs.get(key, 0) == epoch
                and self._guild_gen.get(key[0], 0) == ggen)

    def _deliver_probe(self, key, epoch, ggen, probe: Probe) -> None:
        if not self._valid(key, epoch, ggen):
            return
        try:
            self._on_probe(probe)
        except Exception:
            self.errors += 1
            log.exception("voice front end: probe callback failed")

    def _deliver_wake(self, key, epoch, ggen, event) -> None:
        if not self._valid(key, epoch, ggen):
            return
        try:
            if getattr(event, "emitted", 0) is None:
                import dataclasses
                event = dataclasses.replace(event, emitted=self._clock())
            self._on_wake(event)
        except Exception:
            self.errors += 1
            log.exception("voice front end: wake callback failed")

    def _deliver(self, key, epoch, ggen, seg: Segment) -> None:
        if not self._valid(key, epoch, ggen):
            return                                  # consent revoked / guild reset meanwhile
        try:
            self._on_segment(seg)
        except Exception:
            self.errors += 1
            log.exception("voice front end: segment callback failed")

    def _run(self) -> None:
        last_tick = 0.0
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.05)
            except queue.Empty:
                item = None
            now = self._clock()
            if item is not None:
                key, epoch, ggen, tag, opus, seq, ts, ssrc, arrival = item
                with self._lock:
                    if self._valid(key, epoch, ggen):
                        try:
                            self._stream_for(key, tag).push(opus, seq, ts, ssrc, arrival)
                        except Exception:
                            self.errors += 1
                            log.exception("voice front end: packet processing failed")
            if now - last_tick >= 0.05:
                last_tick = now
                timeout_s = self.cfg.dtx_timeout_ms / 1000
                with self._lock:
                    for stream in list(self._streams.values()):
                        try:
                            stream.tick(now, timeout_s)
                        except Exception:
                            self.errors += 1
                            log.exception("voice front end: tick failed")


# ──────────────────────────────────────────────────────────────────────────────
# Offline use (scripts/voice_eval.py): a whole recording → segments
# ──────────────────────────────────────────────────────────────────────────────

def segment_audio(mono48: np.ndarray, cfg: SegmenterConfig | None = None,
                  vad=None) -> list[Segment]:
    """Run the same decimator → VAD → segmenter chain over a finished
    recording (mono float32 at 48 kHz). Times are seconds from its start."""
    cfg = cfg or SegmenterConfig()
    vad = vad or make_vad()
    out: list[Segment] = []

    def emit(audio, start, end, speech_ms, reason, overlap_ms, onset=None):
        out.append(Segment(0, 0, audio, start, end, speech_ms, reason, overlap_ms, len(out) + 1,
                           speech_start=onset))

    seg = Segmenter(cfg, emit)
    x16 = Decimator().process(mono48.astype(np.float32, copy=False))
    for i in range(len(x16) // VAD_WINDOW):
        w = x16[i * VAD_WINDOW:(i + 1) * VAD_WINDOW]
        seg.push(w, vad.window(w), (i + 1) * VAD_WINDOW_S)
    seg.flush()
    return out
