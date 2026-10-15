"""Tests for the voice-command audio front end (core/voice_frontend.py).

Three layers:
  * pure DSP — the 48→16 kHz decimator against tones (and against the old
    audioop.ratecv, which had no anti-alias filter);
  * the segmenter, driven by synthetic speech probabilities so every boundary
    rule is deterministic;
  * the real thing — speech synthesised by ffmpeg's flite voice, encoded to
    genuine Opus packets with libopus, run through decoding, FEC/PLC, Silero
    VAD and segmentation. Skips itself when libopus, ffmpeg-flite or the
    Silero model (faster-whisper) isn't installed.
"""
from __future__ import annotations

import shutil
import subprocess
import time
import warnings

import pytest

np = pytest.importorskip("numpy")    # voice control's stack; absent → skip, don't fail

from core import voice_frontend as fe  # noqa: E402
from core.voice_frontend import (Decimator, EnergyVAD, Segmenter,  # noqa: E402
                                 SegmenterConfig, SileroVAD, SpeakerStream, VoiceFrontEnd)


def tone(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(48_000 * seconds)) / 48_000
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def level_db(x):
    x = np.asarray(x, dtype=np.float64)[2000:-2000]
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


# ── decimator ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("freq", [300, 1000, 3000, 6000])
def test_decimator_passes_speech_band(freq):
    x = tone(freq)
    y = Decimator().process(x)
    assert len(y) == len(x) // 3
    assert abs(level_db(y) - level_db(x)) < 0.5


@pytest.mark.parametrize("freq", [8500, 10_000, 15_000, 20_000])
def test_decimator_removes_what_would_alias(freq):
    y = Decimator().process(tone(freq))
    assert level_db(y) < -80          # input is -9 dBFS


def test_old_ratecv_path_aliased():
    """Documents why the resampler was replaced: audioop.ratecv folds a 15 kHz
    tone straight into the band whisper listens to."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop
    s16 = (tone(15_000) * 32767).astype("<i2").tobytes()
    out, _ = audioop.ratecv(s16, 2, 1, 48_000, 16_000, None)
    aliased = np.frombuffer(out, dtype="<i2").astype(np.float32) / 32767
    assert level_db(aliased) > -20


def test_decimator_is_seamless_across_chunks():
    x = tone(1234, 0.5) + tone(9000, 0.5, 0.2)
    whole = Decimator().process(x)
    d = Decimator()
    sizes = [960, 7, 960, 1000, 13, 2000]
    parts, pos = [], 0
    for n in sizes + [len(x)]:
        parts.append(d.process(x[pos:pos + n]))
        pos += n
    assert np.array_equal(whole, np.concatenate(parts))


def test_stereo_downmix():
    pcm = np.array([1000, 3000, -2000, -2000], dtype="<i2").tobytes()
    mono = fe.pcm48_stereo_to_mono_float(pcm)
    assert np.allclose(mono * 32768, [2000, -2000])


# ── segmenter (synthetic probabilities) ───────────────────────────────────────

W = np.zeros(fe.VAD_WINDOW, dtype=np.float32)
WIN_S = fe.VAD_WINDOW_S


def run_segmenter(probs, cfg=None, start=100.0):
    out = []
    seg = Segmenter(cfg or SegmenterConfig(),
                    lambda audio, s, e, ms, reason, ov, *_: out.append((s, e, ms, reason, ov, len(audio))))
    t = start
    for p in probs:
        t += WIN_S
        seg.push(W, p, t)
    seg.flush()
    return out, seg


def windows(ms):
    return round(ms / (WIN_S * 1000))


def test_two_bursts_with_a_gap_are_two_segments():
    # Old bug: bursts 700 ms apart merged whenever the poll missed the gap.
    probs = [0.0] * 10 + [0.9] * windows(1000) + [0.0] * windows(700) + [0.9] * windows(800) + [0.0] * 30
    out, _ = run_segmenter(probs)
    assert len(out) == 2
    assert all(r == "silence" for _s, _e, _ms, r, _ov, _n in out)


def test_short_pause_inside_speech_does_not_split():
    probs = [0.9] * windows(800) + [0.1] * windows(300) + [0.9] * windows(800) + [0.0] * 30
    out, _ = run_segmenter(probs)
    assert len(out) == 1


def test_preroll_is_included():
    probs = [0.0] * 20 + [0.9] * windows(500) + [0.0] * 30
    out, _ = run_segmenter(probs)
    (_s, _e, _ms, _r, _ov, n), = out
    cfg = SegmenterConfig()
    expect = windows(cfg.preroll_ms) + windows(500) + windows(cfg.trail_ms)
    assert n == expect * fe.VAD_WINDOW


def test_blips_are_dropped_and_counted():
    probs = [0.0] * 5 + [0.9] * 2 + [0.0] * 40          # 64 ms "click"
    out, seg = run_segmenter(probs)
    assert out == [] and seg.dropped_short == 1


def test_continuous_speech_is_split_not_discarded():
    # Old bug: speech that never paused for 8 s was thrown away whole.
    cfg = SegmenterConfig(max_segment_ms=3000, overlap_ms=640)
    probs = [0.9] * windows(7500) + [0.0] * 40
    out, _ = run_segmenter(probs, cfg)
    reasons = [r for _s, _e, _ms, r, _ov, _n in out]
    assert reasons[:2] == ["max_length", "max_length"] and reasons[-1] == "silence"
    assert all(ov > 0 for _s, _e, _ms, _r, ov, _n in out[1:])      # overlap carried
    total = sum(n for *_x, n in out) - sum(ov for *_y, ov, _n in out) * 16
    assert total >= windows(7500) * fe.VAD_WINDOW                   # nothing lost


def test_hysteresis_keeps_quiet_syllables():
    probs = [0.9] * 5 + [0.4] * 10 + [0.9] * 5 + [0.0] * 30        # 0.4: below start, above end
    out, _ = run_segmenter(probs)
    assert len(out) == 1


def test_timeout_ends_speech_when_packets_stop():
    seg_out = []
    seg = Segmenter(SegmenterConfig(), lambda *a: seg_out.append(a[4]))
    for i in range(15):
        seg.push(W, 0.9, 100 + i * WIN_S)
    seg.timeout()
    assert seg_out == ["timeout"] and not seg.in_speech


def test_energy_vad_separates_tone_bursts_from_silence():
    vad = EnergyVAD()
    quiet = [vad.window(np.zeros(512, np.float32)) for _ in range(30)]
    loud = [vad.window(0.3 * np.sin(np.arange(512) / 3).astype(np.float32)) for _ in range(5)]
    assert max(quiet) < 0.2 and min(loud) > 0.8


# ── real Opus + Silero ────────────────────────────────────────────────────────

def _opus_ready():
    try:
        import discord
        if not discord.opus.is_loaded():
            import ctypes.util
            discord.opus.load_opus(ctypes.util.find_library("opus") or "libopus.so.0")
        return discord.opus.is_loaded()
    except Exception:
        return False


def _flite_ready():
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return " flite " in out.stdout


needs_audio_stack = pytest.mark.skipif(
    not (_opus_ready() and _flite_ready() and SileroVAD.available()),
    reason="needs libopus, ffmpeg with flite, and faster-whisper's Silero model")


def speech(text):
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"flite=text='{text}':voice=slt", "-ar", "48000", "-ac", "2",
         "-f", "s16le", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2)


def silence(seconds):
    return np.zeros((int(48_000 * seconds), 2), dtype="<i2")


def noise(seconds, dbfs):
    rng = np.random.default_rng(7)
    w = np.convolve(rng.standard_normal(int(48_000 * seconds)), np.ones(8) / 8, "same")
    w = w / np.sqrt(np.mean(w ** 2)) * 10 ** (dbfs / 20) * 32767
    return np.stack([w, w], 1).astype("<i2")


def stream_through(timeline, *, dtx=True, drop=()):
    """Encode `timeline` into real 20 ms Opus packets (DTX: send nothing while
    silent, as Discord clients do) and push them through a SpeakerStream."""
    from discord.opus import Encoder
    enc = Encoder()
    segs = []
    st = SpeakerStream(1, 2, SegmenterConfig(), segs.append,
                       decoder_factory=fe._discord_decoder, vad_factory=fe.make_vad)
    seq, ts, t = 100, 5000, 1000.0
    data = timeline.reshape(-1).tobytes()
    step = fe.FRAME_SAMPLES * 4
    for i in range(0, len(data) - step + 1, step):
        chunk = data[i:i + step]
        t += 0.02
        if not (dtx and not np.any(np.frombuffer(chunk, dtype="<i2"))):
            if seq not in drop:
                st.push(enc.encode(chunk, fe.FRAME_SAMPLES), seq, ts, 4242, t)
            seq = (seq + 1) & 0xFFFF
        ts += fe.FRAME_SAMPLES
        st.tick(t, 0.6)
    st.tick(t + 2, 0.6)
    st.segmenter.flush()
    return segs, st


@needs_audio_stack
@pytest.mark.parametrize("dtx", [True, False])
def test_chatter_then_command_are_separate_segments(dtx):
    chatter = speech("so anyway I was saying that the game was good")
    command = speech("hey D J skip")
    tl = np.concatenate([silence(0.5), chatter, silence(0.7), command, silence(1.0)])
    segs, _ = stream_through(tl, dtx=dtx)
    assert len(segs) == 2
    assert 1.0 < segs[1].duration < 2.5          # just the command


@needs_audio_stack
def test_open_mic_noise_does_not_swallow_the_command():
    # The old listener needed 600 ms with NO packets; an open mic never gave it.
    chatter = speech("so anyway I was saying that the game was good")
    command = speech("hey D J skip")
    tl = np.concatenate([silence(0.5), chatter, silence(0.7), command, silence(1.0)])
    segs, _ = stream_through(np.clip(tl.astype(np.int32) + noise(len(tl) / 48_000, -30),
                                     -32768, 32767).astype("<i2"), dtx=False)
    assert len(segs) == 2


@needs_audio_stack
def test_short_yes_survives():
    segs, _ = stream_through(np.concatenate([silence(0.5), speech("yes"), silence(1.0)]))
    assert len(segs) == 1 and segs[0].speech_ms >= 200


@needs_audio_stack
def test_lost_packets_are_concealed_not_skipped():
    tl = np.concatenate([silence(0.5), speech("hey D J skip"), silence(1.0)])
    clean, _ = stream_through(tl)
    lossy, st = stream_through(tl, drop={110, 111, 112})
    assert st.stats.concealed == 3 and st.stats.decode_errors == 0
    # Same length of audio: lost time is filled, not compressed away.
    assert abs(lossy[0].duration - clean[0].duration) < 0.05


@needs_audio_stack
def test_pause_in_transmission_becomes_real_silence():
    tl = np.concatenate([silence(0.5), speech("hey D J"), silence(0.9),
                         speech("play mister brightside"), silence(1.0)])
    segs, st = stream_through(tl)
    assert len(segs) == 2 and st.stats.dtx_fill_frames > 0


# ── the threaded front end ────────────────────────────────────────────────────

class FakeDecoder:
    """Turns a 1-byte 'packet' into a 20 ms frame of constant level."""
    def decode(self, data, fec=False):
        level = 0 if not data else data[0] * 100
        return np.full(fe.FRAME_SAMPLES * 2, level, dtype="<i2").tobytes()


class ScriptedVAD:
    """Speech iff the window isn't silent."""
    def window(self, w):
        return 0.9 if np.abs(w).max() > 0.001 else 0.0


def make_frontend(sink):
    return VoiceFrontEnd(sink.append, decoder_factory=FakeDecoder, vad_factory=ScriptedVAD)


def feed_burst(front, user, n, t0, *, seq0=0, level=50):
    for i in range(n):
        front.feed(1, user, bytes([level]), seq0 + i, (seq0 + i) * 960, 99, t0 + i * 0.02)


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_frontend_emits_segments_from_its_thread():
    got = []
    front = make_frontend(got)
    front.start()
    try:
        now = time.monotonic()
        feed_burst(front, 7, 40, now)                       # 800 ms of "speech"
        feed_burst(front, 7, 40, now + 0.8, seq0=40, level=0)   # then silence
        assert wait_for(lambda: len(got) == 1)
        assert got[0].user_id == 7 and got[0].reason == "silence"
    finally:
        front.stop()


def test_dropping_a_speaker_discards_queued_and_partial_audio():
    """Opt-out mid-utterance: nothing fed before the drop may come out after."""
    got = []
    front = make_frontend(got)
    now = time.monotonic()
    feed_burst(front, 7, 40, now)                    # queued, worker not started yet
    front.drop_speaker(1, 7)
    front.start()
    try:
        feed_burst(front, 8, 40, now)                # someone else, still allowed
        feed_burst(front, 8, 40, now + 0.8, seq0=40, level=0)
        assert wait_for(lambda: len(got) == 1)
        time.sleep(0.2)
        assert [s.user_id for s in got] == [8]
    finally:
        front.stop()


def test_queue_overflow_is_counted_not_fatal():
    front = VoiceFrontEnd(lambda s: None, decoder_factory=FakeDecoder,
                          vad_factory=ScriptedVAD, max_queue=5)
    for i in range(20):
        front.feed(1, 1, b"\x01", i, i * 960, 1, 0.0)
    assert front.queue_overflows == 15
