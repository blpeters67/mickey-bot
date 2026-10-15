"""Tests for the dependency updater.

The interesting behaviour isn't "does pip run" — it's the reporting around it.
An upgrade that doesn't change a version, or that changes one without a
restart, both look like success to a user, so the before/after diff is the
whole product here.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

from core import updater


# ── version probing ───────────────────────────────────────────────────────────

def test_installed_version_finds_a_real_package():
    assert updater.installed_version("pytest") is not None


def test_installed_version_returns_none_for_absent_package():
    assert updater.installed_version("definitely-not-a-real-package-xyz") is None


def test_installed_version_survives_a_junk_name():
    assert updater.installed_version("") is None


def test_versions_of_maps_every_name():
    out = updater.versions_of(["pytest", "not-real-at-all"])
    assert set(out) == {"pytest", "not-real-at-all"}
    assert out["not-real-at-all"] is None


def test_optional_dists_are_import_names_mapped_to_pip_names():
    """Probing by import name is the point — metadata can linger after a
    half-removed package, and upgrading something unimportable helps nobody."""
    assert updater.OPTIONAL_DISTS["audio_separator"] == "audio-separator"
    assert updater.OPTIONAL_DISTS["zxingcpp"] == "zxing-cpp"


def test_installed_optional_dists_returns_pip_names(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name in ("audio_separator", "langdetect"):
            return object()
        if name in updater.OPTIONAL_DISTS:
            return None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    out = updater.installed_optional_dists()
    assert "audio-separator" in out and "langdetect" in out
    assert "rembg" not in out


# ── pip invocation ────────────────────────────────────────────────────────────

def test_pip_uses_the_running_interpreter():
    """A bare `pip` on PATH would install into system Python and change
    nothing the bot can actually see."""
    base = updater._pip_base()
    assert base[0] == sys.executable
    assert base[1:4] == ["-m", "pip", "install"]
    assert "--upgrade" in base


@pytest.mark.asyncio
async def test_pip_upgrade_reports_failure_without_raising(monkeypatch):
    async def fake_exec(*cmd, **kw):
        class P:
            returncode = 1
            async def communicate(self):
                return (b"ERROR: Could not find a version\n", b"")
            def kill(self): pass
            async def wait(self): pass
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ok, out = await updater.pip_upgrade(["nope"])
    assert ok is False
    assert "ERROR" in out


@pytest.mark.asyncio
async def test_pip_upgrade_times_out_cleanly(monkeypatch):
    killed = {"v": False}

    async def fake_exec(*cmd, **kw):
        class P:
            returncode = None
            async def communicate(self):
                await asyncio.sleep(10)
            def kill(self): killed["v"] = True
            async def wait(self): pass
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ok, out = await updater.pip_upgrade(["slow"], timeout=0.05)
    assert ok is False
    assert "timed out" in out
    assert killed["v"], "the hung pip process wasn't killed"


@pytest.mark.asyncio
async def test_pip_upgrade_handles_pip_being_unlaunchable(monkeypatch):
    async def boom(*a, **k):
        raise OSError("no such file")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    ok, out = await updater.pip_upgrade(["x"])
    assert ok is False and "couldn't start pip" in out


# ── change reporting ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upgrade_reports_a_version_change(monkeypatch):
    seq = iter([{"yt-dlp": "2024.1.1"}, {"yt-dlp": "2026.8.1"}])
    monkeypatch.setattr(updater, "versions_of", lambda d: next(seq))

    async def fake(args, **kw):
        return True, "Successfully installed yt-dlp-2026.8.1"

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    res = await updater.upgrade_packages(["yt-dlp"])
    assert res.ok and res.anything_changed
    assert res.changed["yt-dlp"] == ("2024.1.1", "2026.8.1")


@pytest.mark.asyncio
async def test_upgrade_reports_no_change(monkeypatch):
    monkeypatch.setattr(updater, "versions_of", lambda d: {"yt-dlp": "2026.8.1"})

    async def fake(args, **kw):
        return True, "Requirement already satisfied"

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    res = await updater.upgrade_packages(["yt-dlp"])
    assert res.ok
    assert not res.anything_changed
    assert "yt-dlp" in res.unchanged


@pytest.mark.asyncio
async def test_newly_installed_package_shows_as_new(monkeypatch):
    seq = iter([{"thing": None}, {"thing": "1.0"}])
    monkeypatch.setattr(updater, "versions_of", lambda d: next(seq))

    async def fake(args, **kw):
        return True, ""

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    res = await updater.upgrade_packages(["thing"])
    assert res.changed["thing"] == (None, "1.0")


@pytest.mark.asyncio
async def test_empty_package_list_is_a_noop():
    res = await updater.upgrade_packages([])
    assert res.ok and not res.anything_changed


@pytest.mark.asyncio
async def test_failed_upgrade_surfaces_the_error_line(monkeypatch):
    monkeypatch.setattr(updater, "versions_of", lambda d: {"x": "1.0"})

    async def fake(args, **kw):
        return False, "Collecting x\nERROR: No matching distribution found for x\n"

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    res = await updater.upgrade_packages(["x"])
    assert res.ok is False
    assert "No matching distribution" in res.error


@pytest.mark.asyncio
async def test_requirements_upgrade_skips_extras_by_default(monkeypatch):
    calls = []

    async def fake(args, **kw):
        calls.append(args)
        return True, ""

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    monkeypatch.setattr(updater, "installed_optional_dists",
                        lambda: ["audio-separator"])
    monkeypatch.setattr(updater, "versions_of", lambda d: {x: "1" for x in d})

    await updater.upgrade_requirements(extras=False)
    assert calls == [["-r", "requirements.txt"]]


@pytest.mark.asyncio
async def test_requirements_upgrade_includes_extras_when_asked(monkeypatch):
    calls = []

    async def fake(args, **kw):
        calls.append(args)
        return True, ""

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    monkeypatch.setattr(updater, "installed_optional_dists",
                        lambda: ["audio-separator"])
    monkeypatch.setattr(updater, "versions_of", lambda d: {x: "1" for x in d})

    await updater.upgrade_requirements(extras=True)
    assert ["-r", "requirements.txt"] in calls
    assert ["audio-separator"] in calls


@pytest.mark.asyncio
async def test_requirements_upgrade_is_not_eager(monkeypatch):
    """`update.sh` uses --upgrade-strategy eager because a human is watching.
    Dragging every transitive dep forward under a live bot is a worse trade."""
    seen = {}

    async def fake_exec(*cmd, **kw):
        seen["cmd"] = cmd
        class P:
            returncode = 0
            async def communicate(self):
                return (b"", b"")
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await updater.pip_upgrade(["-r", "requirements.txt"])
    assert "eager" not in " ".join(seen["cmd"])


def test_error_line_extraction():
    assert "ERROR: boom" in updater._last_error_line("a\nb\nERROR: boom\ntrailing")
    assert updater._last_error_line("") == "pip failed with no output"
    assert updater._last_error_line("just one line") == "just one line"


# ── nightly channel + restart detection ───────────────────────────────────────

@pytest.mark.parametrize("version,expected", [
    ("2026.8.19", False),
    ("2026.7.4", False),
    ("2026.8.19.233000.dev0", True),
    (None, False),
    ("", False),
])
def test_prerelease_detection(version, expected):
    assert updater._looks_prerelease(version) is expected


@pytest.mark.asyncio
async def test_nightly_channel_passes_pre(monkeypatch):
    """YouTube fixes reach nightly days-to-weeks before a stable release, so
    this flag is the whole point of the feature."""
    seen = {}

    async def fake(args, **kw):
        seen["args"] = args
        return True, ""

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    monkeypatch.setattr(updater, "versions_of", lambda d: {x: "1" for x in d})
    await updater.upgrade_packages(["yt-dlp"], channel="nightly")
    assert seen["args"][0] == "--pre"
    assert "yt-dlp" in seen["args"]


@pytest.mark.asyncio
async def test_stable_channel_does_not_pass_pre(monkeypatch):
    seen = {}

    async def fake(args, **kw):
        seen["args"] = args
        return True, ""

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    monkeypatch.setattr(updater, "versions_of", lambda d: {x: "1" for x in d})
    monkeypatch.setattr(updater, "installed_version", lambda d: "2026.8.19")
    await updater.upgrade_packages(["yt-dlp"], channel="stable")
    assert "--pre" not in seen["args"]
    assert "--force-reinstall" not in seen["args"]


@pytest.mark.asyncio
async def test_returning_to_stable_from_nightly_forces_reinstall(monkeypatch):
    """A dev build sorts *higher* than the newest stable, so a plain upgrade
    would silently do nothing and leave you on nightly."""
    seen = {}

    async def fake(args, **kw):
        seen["args"] = args
        return True, ""

    monkeypatch.setattr(updater, "pip_upgrade", fake)
    monkeypatch.setattr(updater, "versions_of", lambda d: {x: "1" for x in d})
    monkeypatch.setattr(updater, "installed_version",
                        lambda d: "2026.8.19.233000.dev0")
    await updater.upgrade_packages(["yt-dlp"], channel="stable")
    assert "--force-reinstall" in seen["args"]


def test_running_version_reads_the_loaded_module():
    assert updater.running_version("pytest") is not None


def test_running_version_of_absent_module_is_none():
    assert updater.running_version("no_such_module_xyz") is None


def test_restart_pending_false_when_versions_agree(monkeypatch):
    monkeypatch.setattr(updater, "installed_version", lambda d: "2026.7.4")
    monkeypatch.setattr(updater, "running_version", lambda m: "2026.7.4")
    assert updater.restart_pending() is False


def test_restart_pending_true_when_disk_is_newer(monkeypatch):
    """The exact trap: upgraded on disk, old code still in memory. Every tool
    reports the new version while the bot runs the old one."""
    monkeypatch.setattr(updater, "installed_version", lambda d: "2026.8.19")
    monkeypatch.setattr(updater, "running_version", lambda m: "2026.7.4")
    assert updater.restart_pending() is True


@pytest.mark.parametrize("on_disk,in_memory", [
    ("2026.8.19", "2026.08.19"),                     # metadata vs yt_dlp.version
    ("2026.8.19.233000.dev0", "2026.08.19.233000"),  # a nightly
])
def test_restart_pending_ignores_how_the_version_is_spelled(monkeypatch, on_disk, in_memory):
    # yt-dlp's __version__ is zero-padded, its metadata isn't: same release.
    monkeypatch.setattr(updater, "installed_version", lambda d: on_disk)
    monkeypatch.setattr(updater, "running_version", lambda m: in_memory)
    assert updater.restart_pending() is False


def test_restart_pending_is_false_when_unknown(monkeypatch):
    monkeypatch.setattr(updater, "installed_version", lambda d: None)
    monkeypatch.setattr(updater, "running_version", lambda m: "2026.7.4")
    assert updater.restart_pending() is False


@pytest.mark.asyncio
async def test_ytdlp_is_upgraded_with_its_solver(monkeypatch):
    calls = []
    versions = {"yt-dlp": "2026.7.1", "yt-dlp-ejs": "0.7.0"}
    monkeypatch.setattr(updater, "installed_version", lambda d: versions.get(d))

    async def bump(args, timeout=None):
        calls.append(args)
        versions.update({"yt-dlp": "2026.8.19", "yt-dlp-ejs": "0.8.0"})
        return True, "ok"
    monkeypatch.setattr(updater, "pip_upgrade", bump)
    result = await updater.upgrade_packages([updater.YTDLP_SPEC])
    assert calls == [["yt-dlp[default]"]]
    assert result.changed == {"yt-dlp": ("2026.7.1", "2026.8.19"),
                              "yt-dlp-ejs": ("0.7.0", "0.8.0")}


def test_dist_name_strips_extras():
    assert updater.dist_name("yt-dlp[default]") == "yt-dlp"
    assert updater.dist_name("discord.py") == "discord.py"


def test_requirements_install_ytdlp_with_its_solver():
    import pathlib
    req = (pathlib.Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
    line = next(ln for ln in req.splitlines() if ln.strip().startswith("yt-dlp"))
    assert line.split()[0].startswith("yt-dlp[default]")
