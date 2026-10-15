"""
Host self-check: is THIS interpreter able to do voice?

The voice review found the project folder carried a venv built on Linux
(/usr/bin/python3.12) while the machine it sat on was Windows with Python
3.14 — so nothing in the folder proved what the running bot actually had.
This answers that from inside the process: interpreter, platform, package
versions, Opus, DAVE, voice receive, ffmpeg, the VAD, the speech-model
runtime and (optionally) a real model load with a timed inference.

Used three ways:
  * at startup (quick: no model load) — problems are logged as warnings;
  * `voice doctor` (bot owner) — the same report in Discord;
  * `python bot.py --preflight` — full check including a model load, printed,
    exit code 1 on any failure.
"""
from __future__ import annotations

import importlib
import importlib.util
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _check_python() -> list[Check]:
    venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    out = [Check("interpreter", INFO,
                 f"{sys.executable} — Python {platform.python_version()} "
                 f"({'venv' if venv else 'no venv'})"),
           Check("platform", INFO, f"{platform.system()} {platform.release()} "
                                   f"{platform.machine()}")]
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import audioop  # noqa: F401
        out.append(Check("audioop", OK, "available"))
    except Exception:
        out.append(Check("audioop", FAIL, "missing — on Python 3.13+ install `audioop-lts`"))
    return out


_PACKAGES = [
    # (distribution, needed for, required?)
    ("discord.py", "everything", True),
    ("PyNaCl", "voice", True),
    ("davey", "voice (DAVE end-to-end encryption, mandatory since 2026-03)", True),
    ("discord-ext-voice-recv", "voice receive (-clip, voice control)", True),
    ("yt-dlp", "playback", True),
    ("edge-tts", "-tts and spoken voice-control replies", False),
    ("faster-whisper", "voice control, -transcribe", False),
    ("ctranslate2", "voice control (whisper runtime)", False),
    ("onnxruntime", "voice control (Silero VAD)", False),
    ("numpy", "voice control (audio front end)", False),
]


def _check_packages() -> list[Check]:
    out = []
    for dist, why, required in _PACKAGES:
        v = _version(dist)
        if v:
            out.append(Check(dist, OK, v))
        else:
            out.append(Check(dist, FAIL if required else WARN, f"not installed — needed for {why}"))
    return out


def _check_opus() -> Check:
    try:
        import ctypes.util

        import discord
        if not discord.opus.is_loaded():
            for name in filter(None, [ctypes.util.find_library("opus"), "libopus.so.0",
                                      "libopus.so", "opus", "libopus-0", "libopus-0.x64"]):
                try:
                    discord.opus.load_opus(name)
                    break
                except Exception:
                    continue
        if discord.opus.is_loaded():
            return Check("opus", OK, "libopus loaded")
        hint = ("sudo apt install libopus0" if sys.platform.startswith("linux")
                else "put opus.dll (64-bit) next to the bot or on PATH")
        return Check("opus", FAIL, f"libopus not loadable — {hint}")
    except Exception as e:
        return Check("opus", FAIL, f"{type(e).__name__}: {e}")


def _check_voice_modules() -> list[Check]:
    out = []
    try:
        import davey
        out.append(Check("DAVE", OK, f"davey importable ({getattr(davey, '__version__', 'ok')})"))
    except Exception as e:
        out.append(Check("DAVE", FAIL, f"davey not importable: {e}"))
    try:
        from discord.ext import voice_recv
        out.append(Check("receiver", OK, f"voice_recv.VoiceRecvClient "
                                         f"({'ok' if hasattr(voice_recv, 'VoiceRecvClient') else '?'})"))
    except Exception as e:
        out.append(Check("receiver", FAIL, f"discord.ext.voice_recv not importable: {e}"))
    return out


def _check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    if not path:
        return Check("ffmpeg", FAIL, "not on PATH")
    try:
        first = subprocess.run([path, "-version"], capture_output=True, text=True,
                               timeout=10).stdout.splitlines()[0]
    except Exception:
        first = path
    return Check("ffmpeg", OK, first[:80])


def _check_youtube_solver() -> Check:
    """yt-dlp 2025.11+ needs a JS runtime and its solver scripts for
    YouTube's player challenge; without them it works with fewer formats —
    a likely source of intermittent 403s."""
    try:
        from core.ytdiag import solver_fix_steps, solver_status
        st = solver_status()
    except Exception as e:
        return Check("YouTube solver", WARN, f"couldn't check ({type(e).__name__}: {e})")
    if st.ok:
        rt = ", ".join(f"{n} {st.found[n]}" for n in st.usable)
        return Check("YouTube solver", OK, f"{rt}; yt-dlp-ejs {st.ejs}")
    return Check("YouTube solver", WARN,
                 "; ".join(st.problems()) + " — a likely cause of intermittent "
                 "YouTube 403s.\n"
                 + solver_fix_steps())


def _check_vad() -> Check:
    try:
        import numpy as np

        from core.voice_frontend import SileroVAD
        if not SileroVAD.available():
            return Check("VAD", WARN, "Silero model not found — energy-based fallback in use")
        v = SileroVAD()
        t0 = time.perf_counter()
        for _ in range(10):
            v.window(np.zeros(512, dtype=np.float32))
        ms = (time.perf_counter() - t0) * 100
        return Check("VAD", OK, f"Silero v6 via onnxruntime ({ms:.2f} ms per 32 ms window)")
    except Exception as e:
        return Check("VAD", WARN, f"unavailable ({type(e).__name__}: {e})")


def _check_asr_runtime(device: str, compute_type: str) -> list[Check]:
    out = []
    try:
        import ctranslate2
    except Exception:
        return [Check("ASR runtime", WARN, "ctranslate2 missing (faster-whisper not installed)")]
    try:
        cpu = sorted(ctranslate2.get_supported_compute_types("cpu"))
        out.append(Check("CPU compute types", INFO, ", ".join(cpu)))
        n = ctranslate2.get_cuda_device_count()
        if n:
            cuda = sorted(ctranslate2.get_supported_compute_types("cuda"))
            out.append(Check("CUDA", INFO, f"{n} device(s); compute types: {', '.join(cuda)}"))
        else:
            out.append(Check("CUDA", INFO, "no usable CUDA device (CPU only)"))
        if device == "cuda" and not n:
            out.append(Check("VOICE_DEVICE", FAIL, "set to cuda but no CUDA device is usable"))
        else:
            supported = set(ctranslate2.get_supported_compute_types(device if device != "auto" else "cpu"))
            out.append(Check("VOICE_COMPUTE_TYPE",
                             OK if compute_type in supported or compute_type in ("auto", "default") else FAIL,
                             f"{compute_type} on {device}"))
    except Exception as e:
        out.append(Check("ASR runtime", WARN, f"{type(e).__name__}: {e}"))
    return out


def _check_model_load(name: str, device: str, compute_type: str, threads: int) -> list[Check]:
    try:
        import numpy as np

        from core.whisper import AsrOptions, load_model, transcribe_command
    except Exception as e:
        return [Check("model load", FAIL, f"can't import: {e}")]
    t0 = time.perf_counter()
    try:
        model = load_model(name, device=device, compute_type=compute_type, cpu_threads=threads)
    except Exception as e:
        return [Check("model load", FAIL, f"{name}: {type(e).__name__}: {e}"[:200])]
    load_s = time.perf_counter() - t0
    out = [Check("model load", OK, f"{name} on {device}/{compute_type} in {load_s:.1f}s")]
    try:
        clip = np.zeros(16000 * 2, dtype=np.float32)
        transcribe_command(model, clip, AsrOptions())       # warm-up
        t1 = time.perf_counter()
        transcribe_command(model, clip, AsrOptions())
        out.append(Check("inference", OK, f"2 s clip in {time.perf_counter() - t1:.2f}s "
                                          f"(silence; real speech takes longer)"))
    except Exception as e:
        out.append(Check("inference", FAIL, f"{type(e).__name__}: {e}"[:200]))
    return out


def run(*, load_model_name: str | None = None) -> list[Check]:
    """All checks. Pass a model name to also load it and time an inference
    (slow: may download the model the first time)."""
    from core.config import settings
    checks: list[Check] = []
    checks += _check_python()
    checks += _check_packages()
    checks.append(_check_opus())
    checks += _check_voice_modules()
    checks.append(_check_ffmpeg())
    if importlib.util.find_spec("yt_dlp") is not None:
        checks.append(_check_youtube_solver())
    if importlib.util.find_spec("numpy") is not None:
        checks.append(_check_vad())
    checks += _check_asr_runtime(settings.voice_device, settings.voice_compute_type)
    db_dir = os.path.dirname(os.path.abspath(settings.db_path)) or "."
    checks.append(Check("database dir", OK if os.access(db_dir, os.W_OK) else FAIL, db_dir))
    if load_model_name and importlib.util.find_spec("faster_whisper") is not None:
        checks += _check_model_load(load_model_name, settings.voice_device,
                                    settings.voice_compute_type, settings.voice_cpu_threads)
    return checks


def format_report(checks: list[Check], *, markdown: bool = False) -> str:
    icon = {OK: "✅", WARN: "⚠️", FAIL: "❌", INFO: "ℹ️"}
    lines = []
    for c in checks:
        if markdown:
            lines.append(f"{icon.get(c.status, '•')} **{c.name}** — {c.detail}")
        else:
            lines.append(f"[{c.status.upper():4}] {c.name:20} {c.detail}")
    worst = (FAIL if any(c.status == FAIL for c in checks)
             else WARN if any(c.status == WARN for c in checks) else OK)
    summary = {OK: "All good.", WARN: "Works, with warnings.",
               FAIL: "Something required is missing — see ❌ / FAIL lines."}[worst]
    lines.append(("**" + summary + "**") if markdown else summary)
    return "\n".join(lines)


def problems(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.status in (WARN, FAIL)]
