"""
Dependency updates, driven from Discord or a shell.

Exists because yt-dlp breaks on a schedule set by YouTube, not by us — an
extractor stops working, playback dies, and the fix is always "upgrade
yt-dlp". Needing SSH for that is a bad trade when the person who notices is
usually in Discord already.

The one thing that makes this genuinely useful rather than a footgun:
**upgrading a package on disk does not change the running process.** Python
imported yt-dlp at startup and holds those module objects until it exits.
Reloading is not a reliable fix either — `core.audio` holds a direct reference,
and half the package's submodules would stay stale. So an update that isn't
followed by a restart looks exactly like an update that didn't work. Every
function here reports the version *before* and *after* so the difference is
visible, and the caller is expected to offer a restart.

No Discord imports, so the logic is testable on its own.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field

from core.logging_setup import get_logger

log = get_logger(__name__)

__all__ = [
    "UpdateResult", "CORE_WATCH", "OPTIONAL_DISTS", "YTDLP_SPEC", "dist_name",
    "installed_version", "versions_of", "installed_optional_dists",
    "pip_upgrade", "upgrade_requirements", "upgrade_packages",
]

#: Distributions worth reporting a version for after any update. These are the
#: ones whose breakage is visible to users. yt-dlp-ejs is yt-dlp's YouTube
#: challenge solver; yt-dlp pins an exact version of it.
CORE_WATCH = ("yt-dlp", "yt-dlp-ejs", "discord.py", "spotipy", "aiohttp", "PyNaCl")

#: How to install yt-dlp: with the `default` extra, which brings the matching
#: yt-dlp-ejs. A bare `pip install -U yt-dlp` leaves the solver missing (or at
#: the previous version's pin), and YouTube links start failing with 403s.
YTDLP_SPEC = "yt-dlp[default]"

#: Optional extras: (import name, distribution/pip spec). Keyed by import name
#: because that's the only reliable way to tell whether one is actually usable.
OPTIONAL_DISTS = {
    "audio_separator": "audio-separator",
    "faster_whisper": "faster-whisper",
    "argostranslate": "argostranslate",
    "langdetect": "langdetect",
    "rembg": "rembg",
    "qrcode": "qrcode",
    "zxingcpp": "zxing-cpp",
}

#: pip can sit on a slow index for a long time; bound it so a Discord command
#: can't hang forever.
DEFAULT_TIMEOUT = 900.0


@dataclass
class UpdateResult:
    ok: bool
    #: dist name -> (before, after). `before` is None if it wasn't installed.
    changed: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    output: str = ""
    error: str = ""

    @property
    def anything_changed(self) -> bool:
        return bool(self.changed)


def installed_version(dist: str) -> str | None:
    """Version of an installed distribution, or None.

    Read fresh from the metadata on every call rather than cached: the whole
    point is to observe a change that happened seconds ago, in this same
    process, on disk.
    """
    # An empty name is not "not found" — importlib.metadata happily matches
    # *something* and hands back a version, which would report a phantom
    # package as installed.
    if not dist or not dist.strip():
        return None
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                       # pragma: no cover - py<3.8
        return None
    try:
        return version(dist)
    except PackageNotFoundError:
        return None
    except Exception:
        log.debug("version lookup failed for %s", dist, exc_info=True)
        return None


def versions_of(dists) -> dict[str, str | None]:
    return {d: installed_version(d) for d in dists}


def dist_name(spec: str) -> str:
    """'yt-dlp[default]' → 'yt-dlp' (extras off, for version lookups)."""
    return spec.split("[", 1)[0].strip()


def installed_optional_dists() -> list[str]:
    """Pip names of the optional extras that are actually installed.

    Probing by import name matters: a half-removed package can leave metadata
    behind, and upgrading something that can't be imported helps nobody.
    """
    import importlib.util

    out = []
    for module, dist in OPTIONAL_DISTS.items():
        try:
            if importlib.util.find_spec(module) is not None:
                out.append(dist)
        except (ImportError, ValueError):
            continue
    return out


def running_version(module: str = "yt_dlp") -> str | None:
    """Version of the module *currently loaded in this process*.

    Deliberately different from `installed_version`, which reads the metadata
    on disk. When they disagree, someone upgraded and didn't restart — the
    single most confusing state to debug, because every tool reports the new
    version while the bot keeps running the old code.
    """
    try:
        import importlib
        mod = importlib.import_module(module)
    except Exception:
        return None
    ver = getattr(mod, "version", None)
    return (getattr(ver, "__version__", None)
            or getattr(mod, "__version__", None))


def same_version(a: str, b: str) -> bool:
    """Do two version strings name the same release?

    Compares the numeric release part only: yt-dlp's own `__version__` is
    zero-padded ("2026.08.19") while its package metadata is normalised
    ("2026.8.19"), and nightlies add ".dev0" to the metadata alone. A plain
    string compare called every yt-dlp with a one-digit month or day
    "upgraded but not restarted"."""
    def release(v: str):
        m = re.match(r"\s*v?(\d+(?:\.\d+)*)", v or "")
        return tuple(int(x) for x in m.group(1).split(".")) if m else (v or "").strip()
    return release(a) == release(b)


def restart_pending(dist: str = "yt-dlp", module: str = "yt_dlp") -> bool:
    """True when the installed version differs from the one in memory."""
    on_disk, in_memory = installed_version(dist), running_version(module)
    if not on_disk or not in_memory:
        return False
    return not same_version(on_disk, in_memory)


def _pip_base() -> list[str]:
    """Always the *running interpreter's* pip.

    `sys.executable` is the venv's python when the bot runs from a venv, which
    is what makes this land in the right place. A bare `pip` on PATH would
    happily install into system Python and change nothing the bot can see.
    """
    return [sys.executable, "-m", "pip", "install", "--upgrade",
            "--disable-pip-version-check", "--no-input"]


async def pip_upgrade(args: list[str], *, timeout: float = DEFAULT_TIMEOUT,
                      cwd: str | None = None) -> tuple[bool, str]:
    """Run one pip upgrade. Returns (ok, combined output)."""
    cmd = _pip_base() + args
    log.info("Running: %s", " ".join(cmd))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
    except OSError as e:
        return False, f"couldn't start pip: {e}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"pip timed out after {timeout / 60:.0f} minutes"
    text = (out or b"").decode("utf-8", "replace")
    return proc.returncode == 0, text


def _diff(before: dict[str, str | None],
          after: dict[str, str | None]) -> tuple[dict, list]:
    changed, unchanged = {}, []
    for dist, old in before.items():
        new = after.get(dist)
        if old != new:
            changed[dist] = (old, new)
        elif new is not None:
            unchanged.append(dist)
    return changed, unchanged


async def upgrade_packages(dists: list[str], *, channel: str = "stable",
                           timeout: float = DEFAULT_TIMEOUT) -> UpdateResult:
    """Upgrade specific distributions, reporting what actually moved.

    `channel="nightly"` allows pre-releases. That matters for yt-dlp
    specifically: YouTube breaks it on YouTube's schedule, and fixes land in
    nightly days to weeks before they reach a stable release. Running a
    six-week-old stable against a site that changed last Tuesday is the normal
    way to end up with an inexplicable 403.
    """
    if not dists:
        return UpdateResult(ok=True, output="nothing to do")
    args = list(dists)
    # Specs may carry extras ("yt-dlp[default]"); versions are per distribution.
    names = [dist_name(d) for d in dists]
    if "yt-dlp" in names and "yt-dlp-ejs" not in names:
        names.append("yt-dlp-ejs")              # report the solver moving too
    if channel == "nightly":
        args = ["--pre", *args]
    elif channel == "stable":
        # Coming *back* from a nightly needs force: the installed dev build
        # sorts higher than the newest stable, so a plain upgrade is a no-op.
        if any(_looks_prerelease(installed_version(dist_name(d))) for d in dists):
            args = ["--force-reinstall", *args]
    before = versions_of(names)
    ok, out = await pip_upgrade(args, timeout=timeout)
    after = versions_of(names)
    changed, unchanged = _diff(before, after)
    return UpdateResult(ok=ok, changed=changed, unchanged=unchanged, output=out,
                        error="" if ok else _last_error_line(out))


async def upgrade_requirements(requirements: str = "requirements.txt", *,
                               extras: bool = False,
                               timeout: float = DEFAULT_TIMEOUT) -> UpdateResult:
    """Upgrade everything in requirements.txt, and optionally the extras.

    Deliberately *not* `--upgrade-strategy eager` (which `update.sh` does use).
    Eager also drags every transitive dependency forward, which is fine when
    you're sitting at a terminal watching it, and a poor idea against a bot
    that's currently streaming audio to people.
    """
    watch = list(CORE_WATCH)
    extra_dists = installed_optional_dists() if extras else []
    watch += [d for d in extra_dists if d not in watch]

    before = versions_of(watch)
    ok, out = await pip_upgrade(["-r", requirements], timeout=timeout)
    combined = out

    if ok and extra_dists:
        ok2, out2 = await pip_upgrade(extra_dists, timeout=timeout)
        combined += "\n" + out2
        ok = ok and ok2

    after = versions_of(watch)
    changed, unchanged = _diff(before, after)
    return UpdateResult(ok=ok, changed=changed, unchanged=unchanged,
                        output=combined,
                        error="" if ok else _last_error_line(combined))


def _looks_prerelease(version: str | None) -> bool:
    """yt-dlp nightlies are versioned `2026.8.19.233000.dev0`."""
    if not version:
        return False
    v = version.lower()
    return any(m in v for m in (".dev", "a", "b", "rc")) and ".dev" in v


def _last_error_line(output: str) -> str:
    """The most useful line from a failed pip run, for a Discord message."""
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.lower().startswith("error"):
            return ln[:400]
    return lines[-1][:400] if lines else "pip failed with no output"
