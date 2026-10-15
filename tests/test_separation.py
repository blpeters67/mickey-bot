"""Tests for the separation engine and the stem cache.

The model itself isn't exercised here — that needs torch and minutes of CPU.
What is exercised is everything around it that decides whether karaoke is
usable: the CPU budgeting that keeps playback smooth, the throughput estimate
that drives how far ahead the prep worker runs, and the cache that stops the
whole thing being paid for twice.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from core import db, separation
from core.config import settings


def _with_settings(monkeypatch, **overrides):
    """Swap in a modified Settings. It's a frozen dataclass, so the only way
    to override a field is to build a new one and rebind the module's name."""
    patched = dataclasses.replace(settings, **overrides)
    monkeypatch.setattr(separation, "settings", patched)
    return patched


@pytest.fixture(autouse=True)
def _reset_throughput():
    separation._throughput = 0.6
    separation._throughput_samples = 0
    yield
    separation._throughput = 0.6
    separation._throughput_samples = 0


# ── CPU budgeting ─────────────────────────────────────────────────────────────

def test_thread_budget_leaves_a_core_free(monkeypatch):
    """The audio player thread must hand the socket a frame every 20 ms. If
    separation takes every core, it can't, and every song stutters."""
    _with_settings(monkeypatch, separation_threads=0)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert separation.thread_budget() == 7


def test_thread_budget_never_zero(monkeypatch):
    _with_settings(monkeypatch, separation_threads=0)
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert separation.thread_budget() == 1


def test_thread_budget_respects_explicit_setting(monkeypatch):
    _with_settings(monkeypatch, separation_threads=3)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert separation.thread_budget() == 3


def test_child_env_caps_every_maths_library(monkeypatch):
    """torch, ONNX, OpenMP and MKL each independently default to 'all cores'.
    Missing one of them is enough to saturate the box."""
    _with_settings(monkeypatch, separation_threads=2)
    env = separation._child_env()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        assert env[var] == "2", f"{var} not capped"


def test_child_env_preserves_the_rest_of_the_environment(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keepme")
    assert separation._child_env()["SOME_UNRELATED_VAR"] == "keepme"


# ── model ladder ──────────────────────────────────────────────────────────────

def test_model_keys_resolve(monkeypatch):
    for key in ("fast", "good", "best"):
        filename, desc = separation.model_filename(key)
        assert filename and desc


def test_unknown_model_falls_back_to_fast(monkeypatch):
    filename, _ = separation.model_filename("nonsense")
    assert filename == separation.MODELS[separation.DEFAULT_MODEL_KEY][0]


def test_model_key_is_case_insensitive():
    assert separation.model_filename("FAST")[0] == separation.MODELS["fast"][0]


# ── throughput and lookahead ──────────────────────────────────────────────────

def test_throughput_records_a_sample():
    # 240s of audio separated in 240s of wall time = exactly realtime.
    assert separation.record_throughput(240, 240) == pytest.approx(1.0)


def test_throughput_smooths_across_samples():
    separation.record_throughput(240, 240)      # 1.0x
    first = separation.throughput()
    separation.record_throughput(240, 480)      # 0.5x
    second = separation.throughput()
    assert second < first
    assert second > 0.5, "a single slow track shouldn't dominate the estimate"


def test_throughput_ignores_absurd_samples():
    separation.record_throughput(240, 240)
    before = separation.throughput()
    separation.record_throughput(240, 0.001)    # 240000x — a cache hit misread
    assert separation.throughput() == pytest.approx(before)


def test_throughput_ignores_nonpositive():
    before = separation.throughput()
    separation.record_throughput(0, 10)
    separation.record_throughput(10, 0)
    separation.record_throughput(-5, -5)
    assert separation.throughput() == pytest.approx(before)


def test_estimate_scales_with_measured_speed():
    separation.record_throughput(240, 480)      # half realtime
    # A 4-minute track at half speed takes about 8 minutes.
    assert separation.estimate_seconds(240) == pytest.approx(480, rel=0.25)


@pytest.mark.parametrize("rate,minimum", [
    (2.0, 1),     # faster than realtime — one song of runway is plenty
    (1.0, 2),     # exactly realtime — need a little slack
    (0.5, 3),     # half speed — two songs of runway
])
def test_lookahead_grows_as_the_host_slows(rate, minimum):
    separation._throughput = rate
    separation._throughput_samples = 5
    assert separation.recommended_lookahead() >= minimum


def test_lookahead_is_capped():
    """A very slow box mustn't try to separate the whole queue at once."""
    separation._throughput = 0.01
    separation._throughput_samples = 5
    assert separation.recommended_lookahead() <= 4


def test_lookahead_is_at_least_one():
    separation._throughput = 100.0
    separation._throughput_samples = 5
    assert separation.recommended_lookahead() >= 1


# ── cache keying ──────────────────────────────────────────────────────────────

def test_cache_path_is_deterministic():
    a = separation.cache_path_for("https://y/watch?v=x", "model.onnx")
    b = separation.cache_path_for("https://y/watch?v=x", "model.onnx")
    assert a == b


def test_cache_path_separates_models():
    """Switching models must not serve the previous model's output."""
    a = separation.cache_path_for("https://y/watch?v=x", "model_a.onnx")
    b = separation.cache_path_for("https://y/watch?v=x", "model_b.ckpt")
    assert a != b


def test_cache_path_separates_uploads():
    a = separation.cache_path_for("https://y/watch?v=one", "m.onnx")
    b = separation.cache_path_for("https://y/watch?v=two", "m.onnx")
    assert a != b


# ── the on-disk cache index ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stem_round_trips(tmp_path):
    await db.init_db()
    f = tmp_path / "inst.mp3"
    f.write_bytes(b"x" * 2048)
    url = "https://example.test/watch?v=stem1"

    assert await db.get_stem(url, "m.onnx") is None
    await db.put_stem(url, "m.onnx", str(f), duration=200.0, title="Song")
    assert await db.get_stem(url, "m.onnx") == str(f)
    await db.clear_stem_cache()


@pytest.mark.asyncio
async def test_missing_file_drops_the_row(tmp_path):
    """A user clearing the cache directory must not leave rows pointing at
    nothing — ffmpeg would be handed a dead path."""
    await db.init_db()
    f = tmp_path / "gone.mp3"
    f.write_bytes(b"x" * 100)
    url = "https://example.test/watch?v=stem2"
    await db.put_stem(url, "m.onnx", str(f))
    os.remove(f)

    assert await db.get_stem(url, "m.onnx") is None
    # And the row is gone, not merely ignored.
    stats = await db.stem_cache_stats()
    assert stats["count"] == 0


@pytest.mark.asyncio
async def test_eviction_removes_least_recently_used(tmp_path):
    await db.init_db()
    await db.clear_stem_cache()

    paths = []
    for i in range(4):
        f = tmp_path / f"s{i}.mp3"
        f.write_bytes(b"x" * 1000)
        paths.append(f)
        await db.put_stem(f"https://example.test/v={i}", "m.onnx", str(f))

    # Touch the newest two so they're most-recently-used.
    await db.get_stem("https://example.test/v=2", "m.onnx")
    await db.get_stem("https://example.test/v=3", "m.onnx")

    freed_files, _ = await db.evict_stems(2000)     # room for two
    assert freed_files == 2
    assert await db.get_stem("https://example.test/v=2", "m.onnx") is not None
    assert await db.get_stem("https://example.test/v=3", "m.onnx") is not None
    assert await db.get_stem("https://example.test/v=0", "m.onnx") is None
    # And the files are actually gone from disk, not just de-indexed.
    assert not paths[0].exists()
    await db.clear_stem_cache()


@pytest.mark.asyncio
async def test_eviction_is_a_noop_under_the_cap(tmp_path):
    await db.init_db()
    await db.clear_stem_cache()
    f = tmp_path / "small.mp3"
    f.write_bytes(b"x" * 100)
    await db.put_stem("https://example.test/v=small", "m.onnx", str(f))
    assert await db.evict_stems(10_000) == (0, 0)
    await db.clear_stem_cache()


@pytest.mark.asyncio
async def test_put_stem_is_idempotent(tmp_path):
    await db.init_db()
    await db.clear_stem_cache()
    f = tmp_path / "again.mp3"
    f.write_bytes(b"x" * 500)
    url = "https://example.test/v=twice"
    await db.put_stem(url, "m.onnx", str(f))
    await db.put_stem(url, "m.onnx", str(f))
    stats = await db.stem_cache_stats()
    assert stats["count"] == 1
    await db.clear_stem_cache()


@pytest.mark.asyncio
async def test_empty_source_id_ignored():
    await db.init_db()
    assert await db.get_stem("", "m.onnx") is None
    await db.put_stem("", "m.onnx", "/nonexistent")     # must not raise


@pytest.mark.asyncio
async def test_separate_reports_missing_dependency(monkeypatch):
    """Without audio-separator the command must explain, not crash."""
    monkeypatch.setattr(separation, "available", lambda: False)
    with pytest.raises(separation.SeparationError, match="isn't installed"):
        await separation.separate_to_instrumental(
            source_id="https://y/x", stream_url="http://example/a.mp3")


# ── CLI contract ──────────────────────────────────────────────────────────────

# These pin the exact audio-separator flags the engine depends on. If upstream
# renames one, this fails loudly here rather than as a cryptic runtime error in
# the middle of someone's karaoke night.

_REQUIRED_FLAGS = {
    "--model_filename", "--output_dir", "--model_file_dir",
    "--single_stem", "--output_format", "--output_bitrate",
    "--custom_output_names", "--log_level",
}


@pytest.mark.asyncio
async def test_command_line_uses_only_documented_flags(tmp_path, monkeypatch):
    """Capture the argv the engine builds and check every flag is one we
    expect, and that the JSON rename argument is valid JSON."""
    import json
    import dataclasses as _dc

    captured = {}

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env") or {}
        raise RuntimeError("stop here — argv is all we needed")

    patched = _dc.replace(settings, stem_cache_dir=str(tmp_path / "c"),
                          download_tmp=str(tmp_path / "w"))
    os.makedirs(patched.download_tmp, exist_ok=True)
    monkeypatch.setattr(separation, "settings", patched)
    monkeypatch.setattr(separation, "_available", True)
    monkeypatch.setattr(separation, "_find_executable", lambda: ["audio-separator"])

    async def fake_download(stream_url, dest, duration=0.0):
        with open(dest, "wb") as fh:
            fh.write(b"\0" * 16)

    monkeypatch.setattr(separation, "_download_audio", fake_download)
    monkeypatch.setattr(separation.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(Exception):
        await separation.separate_to_instrumental(
            source_id="https://example.test/watch?v=argv",
            stream_url="http://example/a.mp3", duration=120.0)

    cmd = captured["cmd"]
    flags = {a for a in cmd if a.startswith("--")}
    assert flags <= _REQUIRED_FLAGS, f"undocumented flag(s): {flags - _REQUIRED_FLAGS}"
    assert "--single_stem" in flags and cmd[cmd.index("--single_stem") + 1] == "Instrumental"
    rename = cmd[cmd.index("--custom_output_names") + 1]
    assert json.loads(rename) == {"Instrumental": "instrumental"}


def test_cli_module_fallbacks_are_plausible():
    """The console script is the normal path; the module fallback must at
    least name real-looking module paths and never be empty."""
    assert separation._CLI_MODULES
    assert all(m.startswith("audio_separator") for m in separation._CLI_MODULES)


def test_find_executable_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(separation.shutil, "which", lambda _: None)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    assert separation._find_executable() is None
