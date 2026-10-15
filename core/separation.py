"""
AI vocal/instrumental separation, for karaoke.

Why not the `karaoke` audio filter
----------------------------------
`pan=stereo|c0=0.5*c0+-0.5*c1|c1=0.5*c1+-0.5*c0` is naive centre-channel
cancellation. It doesn't identify vocals; it subtracts one channel from the
other, which removes *everything panned centre* — so the kick, snare and bass
go with the vocal, and the stereo image collapses to mono. That's why it sounds
awful, and no amount of tuning fixes it: it's the ceiling of the technique.

Real separation uses a trained model. This module shells out to
`audio-separator` (the UVR model zoo — MDX-Net, Demucs v4, BS/Mel-Roformer),
which is MIT-licensed and built for exactly this use case.

The three things that actually matter on a CPU-only host
--------------------------------------------------------
1. **It is not realtime.** Roughly 1x realtime on a fast desktop CPU, and
   slower than that on a shared VPS. So separation can never sit in the
   playback path; it runs ahead of time and the result is cached.

2. **It must not starve the audio thread.** discord.py plays audio from a
   Python thread that has to hand the socket a 20 ms frame every 20 ms. A
   torch inference job that saturates every core will make that thread miss
   its deadline, and *every* song stutters — including ones nobody is
   karaokeing. So the job runs in a separate process, at the lowest possible
   scheduling priority, with its thread pools capped to leave a core free.
   This is the single most important thing in this file.

3. **Cache or die.** Minutes of CPU per track is only acceptable once. Results
   are cached on disk, keyed by the upload and the model, and survive restarts.

Everything here degrades gracefully when `audio-separator` isn't installed —
`available()` returns False and callers report that rather than crashing, the
same pattern faster-whisper and rembg already use.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from core.config import settings
from core.logging_setup import get_logger

log = get_logger(__name__)

__all__ = [
    "MODELS", "SeparationError", "SeparationResult",
    "available", "model_filename", "separate_to_instrumental",
    "throughput", "record_throughput", "recommended_lookahead",
    "estimate_seconds", "cache_path_for",
]


class SeparationError(RuntimeError):
    """Separation failed for a reason worth showing a user."""


# ── model ladder ──────────────────────────────────────────────────────────────

# Quality vs speed. The SDR figures are the instrumental-stem scores published
# with the UVR models; for reference, the old `pan=` filter has no meaningful
# SDR at all because it isn't separating anything.
MODELS: dict[str, tuple[str, str]] = {
    # key:      (model filename,                              human description)
    "fast": ("UVR-MDX-NET-Inst_HQ_3.onnx",
             "MDX-Net HQ3 — ONNX, the practical choice on CPU"),
    "good": ("model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
             "MelBand Roformer — noticeably cleaner, ~2-3x slower"),
    "best": ("model_bs_roformer_ep_317_sdr_12.9755.ckpt",
             "BS-Roformer (17.0 dB instrumental) — GPU territory"),
}
DEFAULT_MODEL_KEY = "fast"

#: Rough multiples of realtime, CPU, relative to `fast`. Only used to warn
#: about a choice that won't keep up; real numbers come from measurement.
_RELATIVE_COST = {"fast": 1.0, "good": 2.5, "best": 4.0}


def model_filename(key: str | None = None) -> tuple[str, str]:
    """Resolve a quality key to (model filename, description)."""
    k = (key or settings.separation_model or DEFAULT_MODEL_KEY).strip().lower()
    if k not in MODELS:
        log.warning("Unknown SEPARATION_MODEL %r — falling back to %r",
                    k, DEFAULT_MODEL_KEY)
        k = DEFAULT_MODEL_KEY
    return MODELS[k]


# ── availability ──────────────────────────────────────────────────────────────

_available: bool | None = None


def available() -> bool:
    """True if `audio-separator` can be run. Cached after the first check."""
    global _available
    if _available is None:
        _available = _find_executable() is not None
    return _available


#: Module paths the CLI has lived at. Tried in order when the console script
#: isn't on PATH — which is the normal situation under systemd, where the
#: venv's bin/ often isn't in the unit's PATH.
_CLI_MODULES = ("audio_separator.utils.cli", "audio_separator.cli")


def _find_executable() -> list[str] | None:
    """The command that runs audio-separator, or None.

    Prefers the console script. The `python -m` fallback checks that the
    module actually resolves rather than assuming a path — the CLI has moved
    within the package before, and a wrong guess would surface as a confusing
    "no module named" at karaoke time instead of a clean "not installed".
    """
    exe = shutil.which("audio-separator")
    if exe:
        return [exe]
    try:
        import importlib.util
        if importlib.util.find_spec("audio_separator") is None:
            return None
        for mod in _CLI_MODULES:
            try:
                if importlib.util.find_spec(mod) is not None:
                    return [sys.executable, "-m", mod]
            except (ImportError, ValueError, ModuleNotFoundError):
                continue
    except (ImportError, ValueError):
        pass
    return None


def reset_availability_cache() -> None:
    global _available
    _available = None


# ── CPU budgeting ─────────────────────────────────────────────────────────────

def thread_budget() -> int:
    """How many threads separation may use.

    Always leaves at least one core for the event loop and the audio player
    thread. On a 1-core box we still return 1 and accept the contention — but
    the `nice` level below means the audio thread wins any actual fight.
    """
    if settings.separation_threads > 0:
        return max(1, settings.separation_threads)
    cores = os.cpu_count() or 2
    return max(1, cores - 1)


def _child_env() -> dict[str, str]:
    """Environment that stops the maths libraries grabbing every core.

    torch, ONNX Runtime, OpenMP and MKL each independently default to
    "all cores". Capping them here is what keeps playback smooth while a
    separation job runs.
    """
    n = str(thread_budget())
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n,
        "TORCH_NUM_THREADS": n,
        # Keeps HF/torch from printing progress bars into our log capture.
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
    })
    return env


def _preexec():
    """Drop the child to the lowest scheduling priority (POSIX only).

    Without this, a separation job competes with the audio player thread on
    equal terms and playback stutters. With it, the kernel hands the CPU back
    to the player whenever it's runnable, and separation soaks up the leftovers.
    """
    try:
        os.nice(19)
    except (OSError, AttributeError):
        pass


# ── throughput tracking ───────────────────────────────────────────────────────

# Exponentially weighted mean of (audio seconds processed / wall seconds).
# > 1.0 means faster than realtime. Seeded pessimistically so the first
# lookahead decision errs toward preparing early.
_throughput: float = 0.6
_throughput_samples: int = 0
_EWMA_ALPHA = 0.35


def throughput() -> float:
    """Measured speed as a multiple of realtime (0.5 = half speed)."""
    return _throughput


def record_throughput(audio_seconds: float, wall_seconds: float) -> float:
    """Fold one completed job into the running estimate."""
    global _throughput, _throughput_samples
    if audio_seconds <= 0 or wall_seconds <= 0:
        return _throughput
    sample = audio_seconds / wall_seconds
    # Ignore absurd samples (a cache hit misreported, a clock jump).
    if not (0.01 <= sample <= 100):
        return _throughput
    if _throughput_samples == 0:
        _throughput = sample
    else:
        _throughput = (1 - _EWMA_ALPHA) * _throughput + _EWMA_ALPHA * sample
    _throughput_samples += 1
    return _throughput


def estimate_seconds(audio_seconds: float) -> float:
    """How long separating a track of this length is expected to take."""
    return audio_seconds / max(0.01, _throughput)


def recommended_lookahead() -> int:
    """How many queue entries ahead the prep worker should run.

    At 1x realtime, preparing the next track while the current one plays just
    keeps pace, so one is enough. At half speed you need two songs of runway,
    and so on. Capped so a very slow box doesn't try to separate the entire
    queue at once.
    """
    return max(1, min(4, int(round(1.0 / max(0.05, _throughput))) + 1))


# ── cache paths ───────────────────────────────────────────────────────────────

def cache_path_for(source_id: str, model_file: str) -> str:
    """Deterministic cache location for this upload + model pair.

    Keyed by both: switching models must not silently serve the old model's
    output, and two uploads of the same song legitimately separate differently.
    """
    digest = hashlib.sha256(
        f"{source_id}\x00{model_file}".encode("utf-8")).hexdigest()[:24]
    return os.path.join(settings.stem_cache_dir, f"{digest}.mp3")


# ── the job ───────────────────────────────────────────────────────────────────

@dataclass
class SeparationResult:
    path: str
    cached: bool
    wall_seconds: float
    model_file: str


async def _download_audio(stream_url: str, dest: str,
                          duration: float = 0.0) -> None:
    """Pull the stream down to a local file for the separator to read.

    audio-separator wants a path, and handing a model a live CDN URL that can
    expire mid-job is asking for a half-processed track. WAV because the
    separator decodes to PCM anyway — re-encoding here would just add a lossy
    generation for no benefit.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-reconnect", "1", "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5", "-rw_timeout", "30000000",
        "-i", stream_url, "-vn", "-ac", "2", "-ar", "44100", dest,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    timeout = max(120.0, (duration or 300.0) * 2)
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise SeparationError("timed out downloading the audio")
    if proc.returncode != 0:
        msg = (err or b"").decode("utf-8", "replace").strip().splitlines()
        raise SeparationError(
            f"couldn't fetch the audio ({msg[-1] if msg else 'ffmpeg failed'})")


#: One job at a time, process-wide. On CPU these are not parallelisable in any
#: useful sense — two concurrent jobs each run at half speed and double the
#: pressure on the audio thread.
_job_lock = asyncio.Lock()


def busy() -> bool:
    return _job_lock.locked()


async def separate_to_instrumental(
    *, source_id: str, stream_url: str, duration: float = 0.0,
    model_key: str | None = None, on_start=None,
) -> SeparationResult:
    """Produce (or fetch from cache) the instrumental for one track.

    `source_id` is the upload's watch URL — the cache key. `stream_url` must
    already be resolved. `on_start` is called if actual work begins, so callers
    can tell the difference between "here it is" and "this'll be a few minutes".
    """
    if not available():
        raise SeparationError(
            "AI separation isn't installed on my host. Owner: "
            "`pip install \"audio-separator[cpu]\"`")

    model_file, _desc = model_filename(model_key)
    dest = cache_path_for(source_id, model_file)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return SeparationResult(dest, cached=True, wall_seconds=0.0,
                                model_file=model_file)

    if on_start is not None:
        try:
            res = on_start()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            log.debug("separation on_start callback failed", exc_info=True)

    async with _job_lock:
        # Re-check: another waiter may have produced it while we queued.
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return SeparationResult(dest, cached=True, wall_seconds=0.0,
                                    model_file=model_file)
        return await _run_job(source_id, stream_url, duration, model_file, dest)


async def _run_job(source_id: str, stream_url: str, duration: float,
                   model_file: str, dest: str) -> SeparationResult:
    import tempfile

    os.makedirs(settings.stem_cache_dir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="musicbot-sep-", dir=settings.download_tmp)
    started = time.monotonic()
    try:
        src = os.path.join(workdir, "input.wav")
        await _download_audio(stream_url, src, duration)

        cmd = _find_executable()
        if cmd is None:                       # vanished between checks
            raise SeparationError("audio-separator disappeared from the host")
        cmd = cmd + [
            src,
            "--model_filename", model_file,
            "--output_dir", workdir,
            "--model_file_dir", os.path.join(settings.stem_cache_dir, "models"),
            # Only the instrumental is wanted; not writing the vocal stem saves
            # a decode, an encode and a file.
            "--single_stem", "Instrumental",
            "--output_format", "MP3",
            "--output_bitrate", settings.stem_bitrate or "192k",
            "--custom_output_names", '{"Instrumental": "instrumental"}',
            "--log_level", "warning",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workdir, env=_child_env(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            preexec_fn=_preexec if os.name == "posix" else None,
        )
        # Generous ceiling: a slow box on a long track legitimately takes a
        # while, and killing a nearly-finished job wastes everything.
        budget = max(900.0, estimate_seconds(duration or 300.0) * 4)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=budget)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise SeparationError(
                f"separation took longer than {budget/60:.0f} minutes and was "
                f"stopped — try the `fast` model")
        if proc.returncode != 0:
            tail = (out or b"").decode("utf-8", "replace").strip().splitlines()
            detail = tail[-1] if tail else f"exit {proc.returncode}"
            log.error("audio-separator failed: %s", "\n".join(tail[-15:]))
            raise SeparationError(f"separation failed ({detail})")

        produced = _locate_output(workdir)
        if produced is None:
            raise SeparationError("separation produced no instrumental file")

        os.replace(produced, dest)
        wall = time.monotonic() - started
        if duration:
            record_throughput(duration, wall)
        log.info("Separated %r in %.0fs (%.2fx realtime, model %s)",
                 source_id, wall, (duration / wall) if wall and duration else 0,
                 model_file)
        return SeparationResult(dest, cached=False, wall_seconds=wall,
                                model_file=model_file)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")


def _locate_output(workdir: str) -> str | None:
    """Find the instrumental the separator wrote.

    `--custom_output_names` should make this trivial, but model-specific
    suffixes have changed across releases, so fall back to "the biggest audio
    file that isn't our input".
    """
    candidates = []
    for name in os.listdir(workdir):
        if name == "input.wav":
            continue
        path = os.path.join(workdir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in _AUDIO_EXTS:
            continue
        candidates.append(path)
    if not candidates:
        return None
    named = [p for p in candidates
             if "instrumental" in os.path.basename(p).lower()]
    pool = named or candidates
    return max(pool, key=os.path.getsize)
