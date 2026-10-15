#!/usr/bin/env python3
"""
voice_eval.py — measure voice-command recognition on YOUR recordings.

The voice review's point: don't pick a model (or beam size, or thread count)
from its label — measure command accuracy and delay on the real microphones
and the real CPU. This runs the bot's own pipeline (optional VAD segmentation
→ faster-whisper with the bot's options → the bot's grammar) over a labelled
set of clips and compares configurations side by side.

Eval set layout
---------------
    eval/
      labels.csv          file,expected[,note]
      001.wav             anything ffmpeg can read (any rate, mono or stereo)

`expected` is one of:
    skip | pause | resume | restart | stop | clear | loop:on | loop:off |
    volume:up | volume:down | volume:<n> | play:<song> | wake_only | none
`none` means "must NOT do anything" (ordinary chat, music bleed, coughs) —
false actions are what make voice control unbearable, so include plenty.

Songs: `play:<title>` or `play:<title> by <artist>` (the LAST " by " splits;
use `play:<title> | <artist>` when the title itself contains " by "). A play
result is correct only if the title's important words AND every artist word
are in what was heard — "the pretender by foo fighters" is WRONG when you said
Infected Mushroom. Exact query matches are reported separately.

Optional third column `transcript`: what was actually said. If present, exact
(normalised) transcript accuracy is reported too.

Build it from real use: `voice clip` DMs you exactly what reached the speech
model for your last utterance. Record wake variants, fast/slow commands,
"hey DJ … (pause) … skip", artist names, spoken volumes, quiet speech,
keyboard noise, music bleed, two people talking, and refusals.

Usage
-----
    python scripts/voice_eval.py eval/                          # base.en, beam 1
    python scripts/voice_eval.py eval/ --models base.en small.en --beams 1 3 5
    python scripts/voice_eval.py eval/ --threads 2 4            # cpu_threads
    python scripts/voice_eval.py eval/ --segment                # long clips: VAD first
    python scripts/voice_eval.py eval/ --hints --csv out.csv    # wake words as prompt

Keep clips used to tune thresholds separate from the ones you judge by.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

os.environ.setdefault("DISCORD_TOKEN", "voice-eval-does-not-connect")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from core.voice_intent import WAKE_WINDOW_SECONDS, _levenshtein, interpret  # noqa: E402


# ── loading ──────────────────────────────────────────────────────────────────

def load_audio(path: str, rate: int) -> np.ndarray:
    """Any audio file → mono float32 at `rate` (via ffmpeg)."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "1",
         "-ar", str(rate), "-f", "s16le", "pipe:1"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class Clip:
    file: str
    expected: str
    note: str = ""
    transcript: str = ""          # optional: what was really said


def read_labels(folder: str) -> list[Clip]:
    path = os.path.join(folder, "labels.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clips = []
    for r in rows:
        if not r.get("file") or not r.get("expected"):
            continue
        clips.append(Clip(r["file"].strip(), r["expected"].strip().lower(),
                          (r.get("note") or "").strip(), (r.get("transcript") or "").strip()))
    return clips


# ── scoring ──────────────────────────────────────────────────────────────────

def _words(s: str) -> list[str]:
    return re.findall(r"\w+", s.casefold())


def outcome_label(result) -> str:
    """ParseResult → the same vocabulary as labels.csv."""
    if result.intent is None:
        return "wake_only" if result.reason == "wake_only" else "none"
    i = result.intent
    return f"{i.action}:{i.arg}" if i.arg else i.action


# Words that don't identify a song: they may be heard or not without changing
# which track the search finds.
_FILLER = frozenset({"the", "a", "an", "by", "of", "and", "feat", "ft", "featuring",
                     "song", "track", "music", "please", "to", "on", "in", "my"})


def _content(text: str) -> list[str]:
    words = _words(text)
    return [w for w in words if w not in _FILLER] or words


def split_song(query: str) -> tuple[str, str | None]:
    """'title by artist' / 'title | artist' → (title, artist or None)."""
    if "|" in query:
        title, _, artist = query.partition("|")
        return title.strip(), (artist.strip() or None)
    m = re.match(r"^(.*\S)\s+by\s+(\S.*)$", query.strip(), flags=re.IGNORECASE)
    if m:                                   # greedy: splits at the LAST " by "
        return m.group(1), m.group(2)
    return query.strip(), None


def _heard(word: str, have: set[str]) -> bool:
    if word in have:
        return True
    # one-letter slips on long words ("mushrooms") still name the same thing
    return len(word) >= 5 and any(len(h) >= 4 and _levenshtein(word, h) <= 1 for h in have)


def song_match(expected: str, got: str) -> dict:
    """Score a heard play query against the expected song, entity by entity:
    share of the title's important words heard, and whether EVERY word of the
    artist (if one was named) was heard."""
    title, artist = split_song(expected)
    have = set(_words(got))
    t = _content(title)
    title_score = sum(_heard(w, have) for w in t) / len(t) if t else 1.0
    artist_ok = all(_heard(w, have) for w in _content(artist)) if artist else True
    return {"title": title_score, "artist": artist_ok,
            "exact": _words(expected) == _words(got)}


def judge(expected: str, got: str) -> str:
    """correct | false_action | missed | wrong"""
    if expected == got:
        return "correct"
    if expected.startswith("play:") and got.startswith("play:"):
        m = song_match(expected[5:], got[5:])
        if m["title"] >= 0.8 and m["artist"]:
            return "correct"
        return "wrong"
    if expected in ("loop", "loop:on") and got in ("loop", "loop:on"):
        return "correct"
    if expected in ("none", "wake_only"):
        return "false_action" if got not in ("none", "wake_only") else "correct"
    if got in ("none", "wake_only"):
        return "missed"
    return "wrong"


@dataclass
class ConfigResult:
    label: str
    load_seconds: float = 0.0
    rows: list = field(default_factory=list)       # (clip, heard, got, verdict, asr_s, audio_s)

    def summary(self) -> dict:
        verdicts = [r[3] for r in self.rows]
        acts = [r for r in self.rows if r[0].expected not in ("none", "wake_only")]
        quiet = [r for r in self.rows if r[0].expected in ("none", "wake_only")]
        plays = [r for r in self.rows if r[0].expected.startswith("play:")]
        said = [r for r in self.rows if r[0].transcript]
        asr = sorted(r[4] for r in self.rows)
        audio = sum(r[5] for r in self.rows) or 1e-9
        pick = lambda q: asr[min(len(asr) - 1, int(round(q * (len(asr) - 1))))] if asr else 0.0  # noqa: E731
        return {
            "clips": len(self.rows),
            "command accuracy": sum(r[3] == "correct" for r in acts) / len(acts) if acts else None,
            "false actions": sum(v == "false_action" for v in verdicts),
            "quiet clips kept quiet": (sum(r[3] == "correct" for r in quiet) / len(quiet)) if quiet else None,
            "missed": sum(v == "missed" for v in verdicts),
            "wrong action": sum(v == "wrong" for v in verdicts),
            # Songs, two ways: the exact query, and title+artist entities.
            "play exact query": (sum(r[2].startswith("play:") and _words(r[2][5:]) == _words(r[0].expected[5:])
                                     for r in plays) / len(plays)) if plays else None,
            "play entity match": (sum(r[3] == "correct" for r in plays) / len(plays)) if plays else None,
            "transcript exact": (sum(_words(r[1].replace("|", " ")) == _words(r[0].transcript)
                                     for r in said) / len(said)) if said else None,
            # Recognition time only — NOT end-of-speech-to-sound latency (that
            # adds end-of-speech detection, queueing, lookup and FFmpeg start;
            # `voice diag` measures those on the bot).
            "asr-only median s": pick(0.5), "asr-only p95 s": pick(0.95),
            "asr-only real-time factor": sum(asr) / audio,
            "model load s": self.load_seconds,
        }


def decide(timeline: list[tuple[str, float, float]], wakes: list[str]) -> str:
    """Replay (text, speech onset, speech end) utterances through the bot's own
    decision rules: `interpret` (wake at the start, or the wake-then-command
    window), and a window that opens at the end of a wake-only utterance and
    accepts speech STARTING within WAKE_WINDOW_SECONDS — not "for the rest of
    the file". The first action decides."""
    got = "none"
    window_until: float | None = None
    for text, onset, end in timeline:
        in_window = window_until is not None and onset <= window_until
        o = outcome_label(interpret(text, wakes, in_window=in_window))
        if o == "wake_only":
            window_until = end + WAKE_WINDOW_SECONDS
            if got == "none":
                got = "wake_only"
            continue
        if o != "none":
            return o
    return got


def evaluate(clips: list[Clip], folder: str, transcribe, *, wakes: list[str],
             segment: bool, label: str, load_seconds: float = 0.0) -> ConfigResult:
    """`transcribe(audio16k) -> text`. With `segment`, each clip is cut with
    the bot's VAD first and its segments are replayed with their timestamps
    through `decide`."""
    res = ConfigResult(label, load_seconds)
    for clip in clips:
        path = os.path.join(folder, clip.file)
        if segment:
            from core.voice_frontend import segment_audio
            pieces = [(s.audio, s.onset, s.end) for s in segment_audio(load_audio(path, 48_000))]
        else:
            audio = load_audio(path, 16_000)
            pieces = [(audio, 0.0, len(audio) / 16_000)]
        heard, asr_s, timeline = [], 0.0, []
        for audio, onset, end in pieces:
            t0 = time.perf_counter()
            text = transcribe(audio)
            asr_s += time.perf_counter() - t0
            heard.append(text)
            timeline.append((text, onset, end))
        got = decide(timeline, wakes)
        audio_s = sum(len(p[0]) for p in pieces) / 16_000
        res.rows.append((clip, " | ".join(heard), got, judge(clip.expected, got), asr_s, audio_s))
    return res


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--models", nargs="+", default=["base.en"])
    ap.add_argument("--beams", nargs="+", type=int, default=[1])
    ap.add_argument("--threads", nargs="+", type=int, default=[0], help="cpu_threads (0 = auto)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--language", default="en")
    ap.add_argument("--wake", nargs="+", default=["hey dj"])
    ap.add_argument("--segment", action="store_true", help="cut long clips with the bot's VAD first")
    ap.add_argument("--hints", action="store_true", help="put the wake words in whisper's prompt")
    ap.add_argument("--csv", help="write per-clip results here")
    args = ap.parse_args(argv)

    from core.whisper import AsrOptions, load_model, transcribe_command

    clips = read_labels(args.folder)
    if not clips:
        print("No clips in labels.csv"); return 1
    print(f"{len(clips)} clips; wake words: {args.wake}\n")
    results: list[ConfigResult] = []
    for model_name in args.models:
        for threads in args.threads:
            t0 = time.perf_counter()
            model = load_model(model_name, device=args.device,
                               compute_type=args.compute_type, cpu_threads=threads)
            load_s = time.perf_counter() - t0
            transcribe_command(model, np.zeros(16000, dtype=np.float32), AsrOptions())  # warm-up
            for beam in args.beams:
                opts = AsrOptions(language=args.language or None, beam_size=beam,
                                  prompt=(", ".join(args.wake) + ".") if args.hints else None)
                label = f"{model_name} beam={beam} threads={threads or 'auto'}"
                print(f"… {label}")
                results.append(evaluate(
                    clips, args.folder, lambda a: transcribe_command(model, a, opts).text,
                    wakes=args.wake, segment=args.segment, label=label, load_seconds=load_s))

    keys = list(results[0].summary())
    width = max(len(r.label) for r in results)
    print("\n" + "config".ljust(width) + "  " + "  ".join(k for k in keys))
    for r in results:
        s = r.summary()
        cells = []
        for k in keys:
            v = s[k]
            pct = any(w in k for w in ("accuracy", "quiet", "exact", "match"))
            cells.append(("—" if v is None else f"{v:.0%}" if pct
                          else f"{v:.2f}" if isinstance(v, float) else str(v)).ljust(len(k)))
        print(r.label.ljust(width) + "  " + "  ".join(cells))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["config", "file", "expected", "heard", "got", "verdict", "asr_s"])
            for r in results:
                for clip, heard, got, verdict, asr_s, _a in r.rows:
                    w.writerow([r.label, clip.file, clip.expected, heard, got, verdict, f"{asr_s:.3f}"])
        print(f"\nPer-clip results: {args.csv}")
    print("\nLook at false actions first, then command accuracy, then p95 delay.\n"
          "Timings are ASR-only: the bot's real response also waits for end of speech, "
          "the queue, the song lookup and FFmpeg — `voice diag` shows end-of-speech → "
          "first audio for songs requested by voice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
