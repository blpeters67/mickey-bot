"""
Wake word on voices other than the owner's: synthetic speakers saying "hey
DJ …" and phrases that sound like it, at several speeds, clean and with noise,
through the same path as scripts/voice_replay.py (Opus packets → the bot's
decoder, VAD and instant spotter → the short-window check).

    venv/bin/python scripts/wake_eval_synth.py [--engine flite] [--out eval.json]

Reports, per condition: wake phrases detected / accepted by the check (a
missed wake), and look-alikes that fired the detector / that the check let
through (a false cue). The phrases are generic test sentences, nothing taken
from anyone's recordings.

What this is and isn't: flite's five voices are old concatenative/statistical
synthesis — robotic, unlike a real microphone. A miss here can mean the voice
is unnatural rather than the detector being weak, and a pass doesn't prove a
real accent works. It does test other voices, speeds, noise and confusable
phrases, which the owner's recordings can't. Nothing touches Discord.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DISCORD_TOKEN", "replay")

from scripts.voice_replay import opus_packets, replay_stream  # noqa: E402

WAKE = [
    "Hey DJ, play some music.",
    "Hey DJ, skip this song.",
    "Hey DJ.",
    "Hey DJ, what's playing right now?",
    "Hey DJ, turn it up a little.",
    "Hey DJ, can you queue up something by Queen?",
    "Hey, DJ, pause it.",
    "Hey DJ, shuffle the queue and then play the next one.",
]
LOOKALIKES = [
    "Hey Jay, play some music.",
    "Hey TJ, how are you?",
    "Hey BJ, come over here.",
    "Hey PJ, what's up?",
    "Hey CJ, did you see that?",
    "Hey G, let's go.",
    "Hey Dee, are you there?",
    "Hey, did you hear that?",
    "Hey Judge, one question.",
    "Hey Didi, it's me.",
    "The DJ played my song last night.",
    "A DJ set at the club.",
    "DJ Khaled is on the radio.",
    "Hey there, how's it going?",
    "Hey guys, what are we doing?",
    "Hey Siri, what time is it?",
    "Okay Google, play music.",
    "Heyday of the band was the nineties.",
    "They deejayed the whole party.",
    "Hey Deja, are you coming?",
]
FLITE_VOICES = ["slt", "kal16", "awb", "rms", "kal"]
SPEEDS = [0.85, 1.0, 1.2]


def flite(text: str, voice: str, speed: float) -> np.ndarray:
    """48 kHz mono float32."""
    safe = text.replace("'", "").replace(":", " ")
    chain = f"atempo={speed}" if speed != 1.0 else "anull"
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"flite=text='{safe}':voice={voice}", "-af", chain,
         "-ar", "48000", "-ac", "1", "-f", "f32le", "pipe:1"],
        capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    peak = float(np.max(np.abs(a))) or 1.0
    return a / peak * 0.5                                  # a normal speaking level


def babble(n: int, rng: np.random.Generator) -> np.ndarray:
    """A noise bed: pink-ish noise plus a slow 'music' of chords — a crude
    stand-in for a room with music in it, not a real recording."""
    white = rng.standard_normal(n).astype(np.float32)
    k = np.fft.rfftfreq(n, 1 / 48000)
    spec = np.fft.rfft(white) / np.maximum(1.0, np.sqrt(k / 50))
    pink = np.fft.irfft(spec, n).astype(np.float32)
    t = np.arange(n) / 48000
    chords = sum(np.sin(2 * np.pi * f * t) for f in (220, 277.2, 329.6)) * (
        0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    bed = pink / (np.std(pink) + 1e-9) + 0.5 * chords / (np.std(chords) + 1e-9)
    return (bed / (np.std(bed) + 1e-9)).astype(np.float32)


def with_noise(speech: np.ndarray, snr_db: float | None, rng) -> np.ndarray:
    lead, tail = np.zeros(int(0.6 * 48000), np.float32), np.zeros(int(1.2 * 48000), np.float32)
    clip = np.concatenate([lead, speech, tail])
    if snr_db is None:
        return clip
    active = speech[np.abs(speech) > 0.01]
    p_speech = float(np.mean(np.square(active))) if len(active) else 1e-4
    noise = babble(len(clip), rng)
    noise *= np.sqrt(p_speech / (10 ** (snr_db / 10)))
    return np.clip(clip + noise, -1, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=",".join(FLITE_VOICES))
    ap.add_argument("--speeds", default=",".join(map(str, SPEEDS)))
    ap.add_argument("--snr", default="none,10", help="comma list: none (clean) or dB")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--compare30", action="store_true",
                    help="also run the old 30-second check on every fire (before/after)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from core.config import settings
    from core.voice_intent import text_after_wake
    from core.wake_spotter import PHRASE, check_words
    from core.whisper import AsrOptions, load_model, transcribe_command
    checker = load_model(settings.voice_whisper_model, device=settings.voice_device,
                         compute_type=settings.voice_compute_type, cpu_threads=a.threads)
    accept = lambda t: text_after_wake(t, [PHRASE]) is not None            # noqa: E731
    check = lambda x: check_words(checker, x, window=settings.voice_fast_wake_verify_window,  # noqa: E731
                                  accept=accept)
    check(np.zeros(16000, np.float32))
    rng = np.random.default_rng(7)
    voices = a.voices.split(",")
    speeds = [float(s) for s in a.speeds.split(",")]
    snrs = [None if s == "none" else float(s) for s in a.snr.split(",")]
    rows = []
    t_start = time.time()
    for snr in snrs:
        for voice in voices:
            for speed in speeds:
                for kind, phrases in (("wake", WAKE), ("lookalike", LOOKALIKES)):
                    for text in phrases:
                        clip = with_noise(flite(text, voice, speed), snr, rng)
                        (segs, events), = replay_stream(opus_packets(clip), t0=1000.0)
                        fires = []
                        for ev in events:
                            t = time.perf_counter()
                            heard = check(ev.audio[-int(3.0 * 16000):])
                            fire = {"at": round(ev.at - 1000.0, 2), "score": round(ev.score, 3),
                                    "heard": heard, "ok": accept(heard),
                                    "ms": round((time.perf_counter() - t) * 1000)}
                            if a.compare30:
                                t = time.perf_counter()
                                old = transcribe_command(checker, ev.audio[-int(3.0 * 16000):],
                                                         AsrOptions(beam_size=1)).text
                                fire["old30"] = {"heard": old, "ok": accept(old),
                                                 "ms": round((time.perf_counter() - t) * 1000)}
                            fires.append(fire)
                        rows.append({"kind": kind, "text": text, "voice": voice, "speed": speed,
                                     "snr": snr, "fired": bool(fires),
                                     "accepted": any(f["ok"] for f in fires),
                                     "accepted_old30": (any(f["old30"]["ok"] for f in fires)
                                                        if a.compare30 else None),
                                     "fires": fires})
                        mark = ("✓" if rows[-1]["accepted"] else ("·" if fires else "✗")) \
                            if kind == "wake" else ("FALSE CUE" if rows[-1]["accepted"]
                                                    else ("fired" if fires else "-"))
                        print(f"{str(snr):>4} {voice:5} x{speed:<4} {kind:9} {mark:9} {text!r}"
                              + (f" heard {fires[0]['heard']!r}" if fires else ""), flush=True)

    def rate(sel, key):
        return f"{sum(r[key] for r in sel)}/{len(sel)}"
    summary = {"seconds": round(time.time() - t_start), "conditions": {}}
    for snr in snrs:
        for voice in voices:
            sel = [r for r in rows if r["snr"] == snr and r["voice"] == voice]
            w = [r for r in sel if r["kind"] == "wake"]
            lk = [r for r in sel if r["kind"] == "lookalike"]
            summary["conditions"][f"{voice} snr={snr}"] = {
                "wake_detected": rate(w, "fired"), "wake_accepted": rate(w, "accepted"),
                "lookalike_fired": rate(lk, "fired"), "lookalike_false_cue": rate(lk, "accepted")}
    w = [r for r in rows if r["kind"] == "wake"]
    lk = [r for r in rows if r["kind"] == "lookalike"]
    summary["all"] = {"wake_detected": rate(w, "fired"), "wake_accepted": rate(w, "accepted"),
                      "lookalike_fired": rate(lk, "fired"),
                      "lookalike_false_cue": rate(lk, "accepted"),
                      "false_cues": sorted({(r["text"], r["fires"][0]["heard"]) for r in lk
                                            if r["accepted"]})}
    if a.compare30:
        for snr in snrs:
            w_ = [r for r in rows if r["kind"] == "wake" and r["snr"] == snr]
            lk_ = [r for r in rows if r["kind"] == "lookalike" and r["snr"] == snr]
            summary[f"short_vs_old30 snr={snr}"] = {
                "wake_accepted_short": rate(w_, "accepted"),
                "wake_accepted_old30": rate(w_, "accepted_old30"),
                "lookalike_accepted_short": rate(lk_, "accepted"),
                "lookalike_accepted_old30": rate(lk_, "accepted_old30")}
        old = sorted(f["old30"]["ms"] for r in rows for f in r["fires"])
        if old:
            summary["old30_ms"] = {"median": old[len(old) // 2], "p90": old[int(0.9 * (len(old) - 1))]}
    ms = sorted(f["ms"] for r in rows for f in r["fires"])
    if ms:
        summary["check_ms"] = {"median": ms[len(ms) // 2], "p90": ms[int(0.9 * (len(ms) - 1))]}
    print(json.dumps(summary, indent=1, default=str))
    if a.out:
        json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
