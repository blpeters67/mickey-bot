"""
Replay recordings through the bot's real voice front end, as Discord would
deliver them, and measure the wake word's path and what the recogniser hears.

    venv/bin/python scripts/voice_replay.py rec/*.mp3 --ref ref.json [--threads 2]
        [--load asr] [--speakers 2] [--out replay.json]

Each recording is resampled to 48 kHz stereo, encoded to Opus in 20 ms
packets (64 kbps, like a Discord voice channel) and pushed packet by packet
into core/voice_frontend.SpeakerStream — the bot's own Opus decoder,
anti-aliased 16 kHz decimation, Silero VAD, segmenter and instant wake-word
spotter (core/wake_spotter.py). Then:

  * every wake event, against the *annotated* end of "DJ" (--ref: word
    timings from a strong reference model, refined to the end of the voiced
    sound) — not packet arrival;
  * the wake check on the event's audio, both ways (the short window the bot
    now uses, and the old 30-second pass): what each heard, accepted or not,
    how long it took on this machine;
  * the server-side cue estimate: wake end → detection + check + the next
    audio frame (the mixer attach itself is synchronous; tests/
    test_wake_fast_path.py shows the first frame goes out on the next 20 ms
    tick) — a replay estimate, NOT microphone-to-listener latency, which
    also includes the speaker's client, Discord's transport and the
    listener's jitter buffer;
  * the segments (with the length cap's split and overlap) transcribed by
    the bot's model and options, joined like the cog joins a long request
    (core/voice_intent.stitch), and compared with the reference words.

--load asr runs the bot's main recogniser on another clip at the same time as
each check (heavy recognition load). --speakers 2 replays each recording as
two people talking at once (the second 50 ms later): fires queue for the one
checker, as in the cog.

Nothing here touches Discord or the bot's database. The recordings are read,
never written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DISCORD_TOKEN", "replay")


def load(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a.astype(np.float32), sr


def to_48k(a: np.ndarray, sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly
    if sr == 48_000:
        return a
    g = gcd(48_000, sr)
    return resample_poly(a, 48_000 // g, sr // g).astype(np.float32)


def to_16k(a: np.ndarray, sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(16_000, sr)
    return resample_poly(a, 16_000 // g, sr // g).astype(np.float32)


def opus_packets(mono48: np.ndarray, bitrate_kbps: int = 64) -> list[bytes]:
    import discord.opus as dopus
    if not dopus.is_loaded():
        dopus._load_default()
    enc = dopus.Encoder()
    enc.set_bitrate(bitrate_kbps)
    pcm = np.clip(mono48, -1, 1)
    st = (np.repeat(pcm[:, None], 2, axis=1) * 32767).astype("<i2")
    frame = 960
    out = []
    pad = (-len(st)) % frame
    st = np.concatenate([st, np.zeros((pad, 2), "<i2")])
    for i in range(0, len(st), frame):
        out.append(enc.encode(st[i:i + frame].tobytes(), frame))
    return out


def wake_end(ref: dict | None, a16: np.ndarray) -> tuple[float | None, float | None]:
    """(reference model's end of "DJ", refined to where the voiced sound of
    it ends) in seconds from the start of the recording."""
    if not ref:
        return None, None
    words = ref.get("words") or []
    end = None
    for w, _s, e, _p in words[:6]:
        if re.sub(r"[^a-z]", "", w.lower()) in ("dj", "deejay", "dee", "jay", "d"):
            end = e
            break
    if end is None:
        return None, None
    # energy in 10 ms hops; the last hop within ±250 ms of `end` that is still
    # within 25 dB of the loudest part of the wake phrase
    hop = 160
    n = len(a16) // hop
    e = np.array([float(np.sqrt(np.mean(np.square(a16[i * hop:(i + 1) * hop])) + 1e-12))
                  for i in range(n)])
    db = 20 * np.log10(e + 1e-9)
    lo, hi = max(0, int((end - 0.25) * 100)), min(n - 1, int((end + 0.25) * 100))
    peak = db[:hi + 1].max() if hi > 0 else db.max()
    voiced = [i for i in range(lo, hi + 1) if db[i] >= peak - 25]
    refined = (max(voiced) + 1) / 100 if voiced else end
    return end, refined


def wer(ref: str, hyp: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()   # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def replay_stream(packets, *, t0: float, speakers: int = 1, offset: float = 0.05):
    """Push the packets into SpeakerStreams; returns (segments, events) per
    speaker. Stream time: t0 + 20 ms per packet."""
    from core.voice_frontend import SegmenterConfig, SpeakerStream, make_vad
    from core.config import settings
    from core.wake_spotter import WakeModel
    import discord.opus as dopus
    model = WakeModel()
    cfg = SegmenterConfig(threshold=settings.voice_vad_threshold,
                          neg_threshold=max(0.05, settings.voice_vad_threshold - 0.15),
                          end_silence_ms=settings.voice_end_silence_ms,
                          dtx_timeout_ms=settings.voice_end_silence_ms)
    outs = []
    for k in range(speakers):
        segs, events = [], []
        stream = SpeakerStream(1, 100 + k, cfg, segs.append, decoder_factory=dopus.Decoder,
                               vad_factory=make_vad, wake=events.append, wake_model=model,
                               wake_threshold=settings.voice_fast_wake_threshold or model.threshold)
        outs.append((stream, segs, events))
    for i, pkt in enumerate(packets + [None] * 75):           # + 1.5 s of silence after
        for k, (stream, _s, _e) in enumerate(outs):
            t = t0 + k * offset + i * 0.02
            if pkt is None:
                stream.tick(t, 0.6)
                continue
            stream.push(pkt, i & 0xFFFF, (i * 960) & 0xFFFFFFFF, 1000 + k, t)
    for stream, _s, _e in outs:
        stream.segmenter.flush()
    return [(segs, events) for _st, segs, events in outs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default=None, help="reference transcripts + word timings (json)")
    ap.add_argument("--model", default=None, help="speech model (default: the bot's)")
    ap.add_argument("--threads", type=int, default=2, help="the checker's CPU threads")
    ap.add_argument("--load", choices=["none", "asr"], default="none")
    ap.add_argument("--speakers", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from core.config import settings
    from core.voice_intent import stitch, text_after_wake
    from core.whisper import AsrOptions, load_model, transcribe_command
    name = a.model or settings.voice_whisper_model
    print(f"model {name} · checker threads {a.threads} · load {a.load} · speakers {a.speakers}",
          flush=True)
    checker = load_model(name, device=settings.voice_device, compute_type=settings.voice_compute_type,
                         cpu_threads=a.threads)
    listener = load_model(name, device=settings.voice_device,
                          compute_type=settings.voice_compute_type,
                          cpu_threads=settings.voice_cpu_threads)
    wakes = [settings.voice_wake_word]
    opts = AsrOptions(language=settings.voice_language or None, beam_size=settings.voice_beam_size,
                      prompt=(", ".join(wakes) + ".") if settings.voice_asr_hints else None)
    from core.wake_spotter import PHRASE, check_words
    short = lambda x: check_words(checker, x, window=settings.voice_fast_wake_verify_window,   # noqa: E731
                                  accept=lambda t: text_after_wake(t, [PHRASE]) is not None)
    full = lambda x: transcribe_command(checker, x, AsrOptions(beam_size=1)).text  # noqa: E731
    z = np.zeros(16000, np.float32)
    for _ in range(2):
        short(z), full(z), transcribe_command(listener, z, opts)
    refs = json.load(open(a.ref)) if a.ref else {}
    rows = []
    for path in a.files:
        n = re.sub(r"\D", "", os.path.splitext(os.path.basename(path))[0]) or path
        audio, sr = load(path)
        a16 = to_16k(audio, sr)
        ref = refs.get(str(n))
        w_end, w_end_v = wake_end(ref, a16)
        packets = opus_packets(to_48k(audio, sr))
        t0 = 1000.0
        per_speaker = replay_stream(packets, t0=t0, speakers=a.speakers)
        segs, events = per_speaker[0]
        row = {"file": os.path.basename(path), "seconds": round(len(a16) / 16000, 2),
               "wake_end_ref": w_end, "wake_end": w_end_v, "fires": [], "segments": []}
        # ── the wake path ─────────────────────────────────────────────────────
        load_thread = None
        if a.load == "asr":
            busy = threading.Event()

            def hog():
                clip = a16[:16000 * 8]
                while not busy.is_set():
                    transcribe_command(listener, clip, opts)
            load_thread = threading.Thread(target=hog, daemon=True)
            load_thread.start()
            time.sleep(0.2)
        queue_wait = 0.0
        all_events = [(k, e) for k, (_s, evs) in enumerate(per_speaker) for e in evs]
        all_events.sort(key=lambda ke: ke[1].at)
        clock = None                                 # when the one checker is next free
        for k, ev in all_events:
            fired_at = ev.at - t0 - k * 0.05         # seconds into the recording
            start_at = fired_at if clock is None else max(fired_at, clock)
            t = time.perf_counter()
            heard_s = short(ev.audio[-int(3.0 * 16000):])
            took_s = time.perf_counter() - t
            clock = start_at + took_s
            t = time.perf_counter()
            heard_f = full(ev.audio[-int(3.0 * 16000):])
            took_f = time.perf_counter() - t
            ok_s = text_after_wake(heard_s, wakes) is not None
            ok_f = text_after_wake(heard_f, wakes) is not None
            row["fires"].append({
                "speaker": k, "kind": ev.kind, "score": round(ev.score, 3),
                "fire_s": round(fired_at, 3),
                "vs_wake_end_ms": None if w_end_v is None else round((fired_at - w_end_v) * 1000),
                "waited_ms": round((start_at - fired_at) * 1000),
                "short": {"heard": heard_s, "ok": ok_s, "ms": round(took_s * 1000)},
                "full30": {"heard": heard_f, "ok": ok_f, "ms": round(took_f * 1000)},
            })
            queue_wait = max(queue_wait, start_at - fired_at)
        if load_thread is not None:
            busy.set()
            load_thread.join(timeout=30)
        # the cue: first accepted fire (per speaker), + its check, + ≤ 20 ms
        for k in range(a.speakers):
            fires = [f for f in row["fires"] if f["speaker"] == k and f["short"]["ok"]]
            if fires and w_end_v is not None:
                f = fires[0]
                row.setdefault("cue_ms", []).append(f["vs_wake_end_ms"] + f["waited_ms"]
                                                    + f["short"]["ms"] + 10)
        # ── what the recogniser hears ─────────────────────────────────────────
        texts = []
        for s in segs:
            r = transcribe_command(listener, s.audio, opts)
            texts.append(r.text)
            row["segments"].append({"reason": s.reason, "seconds": round(len(s.audio) / 16000, 2),
                                    "overlap_ms": s.overlap_ms, "text": r.text})
        joined = ""
        for s, txt in zip(segs, texts):
            joined = stitch(joined, txt) if (joined and s.overlap_ms) else (
                (joined + " " + txt).strip() if joined else txt)
        row["heard_stitched"] = joined
        if len(segs) > 1:
            # what the cog does with a long request: its parts' audio joined
            # (overlaps out), transcribed once more as one
            audio_all = np.concatenate([segs[0].audio] + [s.audio[int(s.overlap_ms * 16):]
                                                          for s in segs[1:]])
            t = time.perf_counter()
            joined = transcribe_command(listener, audio_all, opts).text
            row["whole_ms"] = round((time.perf_counter() - t) * 1000)
        row["heard"] = joined
        if ref:
            row["reference"] = ref.get("text", "")
            row["wer"] = round(wer(ref.get("text", ""), joined), 3)
            row["wer_stitched"] = round(wer(ref.get("text", ""), row["heard_stitched"]), 3)
        rows.append(row)
        first = row["fires"][0] if row["fires"] else None
        print(f"{row['file']:>7} {row['seconds']:5.1f}s fires {len(row['fires'])} "
              + (f"first {first['vs_wake_end_ms']:+d} ms score {first['score']:.2f} "
                 f"short {first['short']['ms']} ms {'✓' if first['short']['ok'] else '✗'} "
                 f"{first['short']['heard']!r} | 30 s {first['full30']['ms']} ms "
                 f"{'✓' if first['full30']['ok'] else '✗'}" if first and first['vs_wake_end_ms'] is not None
                 else "NO FIRE")
              + f" · cue {row.get('cue_ms')} · segs {len(segs)} · WER {row.get('wer')}", flush=True)
    # ── summary ──────────────────────────────────────────────────────────────
    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None
    cues = [c for r in rows for c in r.get("cue_ms", [])]
    firsts = [r["fires"][0] for r in rows if r["fires"]]
    det = [f["vs_wake_end_ms"] for f in firsts if f["vs_wake_end_ms"] is not None]
    sh = [f["short"]["ms"] for r in rows for f in r["fires"]]
    fu = [f["full30"]["ms"] for r in rows for f in r["fires"]]
    missed = sum(1 for r in rows if not any(f["short"]["ok"] for f in r["fires"]))
    agree = sum(1 for r in rows for f in r["fires"] if f["short"]["ok"] == f["full30"]["ok"])
    total = sum(len(r["fires"]) for r in rows)
    summary = {
        "recordings": len(rows), "speakers": a.speakers, "load": a.load,
        "no_accepted_fire": missed,
        "detection_vs_wake_end_ms": {"median": pct(det, .5), "p90": pct(det, .9), "max": max(det) if det else None},
        "check_short_ms": {"median": pct(sh, .5), "p90": pct(sh, .9), "p95": pct(sh, .95)},
        "check_30s_ms": {"median": pct(fu, .5), "p90": pct(fu, .9)},
        "short_vs_30s_agree": f"{agree}/{total}",
        "cue_estimate_ms": {"median": pct(cues, .5), "p90": pct(cues, .9), "p95": pct(cues, .95),
                            "n": len(cues)},
        "wer_median": statistics.median([r["wer"] for r in rows if "wer" in r]) if refs else None,
        "wer_stitched_median": (statistics.median([r["wer_stitched"] for r in rows
                                                   if "wer_stitched" in r]) if refs else None),
        "wer_by_file": {r["file"]: (r.get("wer_stitched"), r.get("wer")) for r in rows},
        "max_queue_wait_ms": None,
    }
    print(json.dumps(summary, indent=1))
    if a.out:
        json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
