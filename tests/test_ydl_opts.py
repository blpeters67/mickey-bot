"""Guards against yt-dlp option drift.

The bug these exist for: `source_address` (which forces IPv4) lived only in
`YDL_BASE`, while cogs/downloads.py hand-rolled its own option dicts. So
playback forced IPv4 and downloads used IPv6 — and YouTube 403s a lot of
datacenter IPv6 space. Same video, same host, same second: `play` worked,
`download` returned 403 Forbidden.

Nothing about that is visible when reading either file on its own, which is
why it survived so long. These tests make the invariant explicit: every
yt-dlp call in the project inherits the shared networking and auth settings.
"""
from __future__ import annotations

import ast
import os
import pathlib

import pytest

from core import audio

REPO = pathlib.Path(__file__).resolve().parent.parent


# ── the shared base ───────────────────────────────────────────────────────────

def test_ydl_net_forces_ipv4_by_default():
    assert audio.YDL_NET.get("source_address") == "0.0.0.0"


def test_ydl_base_inherits_the_network_settings():
    for key, value in audio.YDL_NET.items():
        assert audio.YDL_BASE.get(key) == value, f"YDL_BASE dropped {key}"


def test_ydl_base_still_skips_download():
    """It's the metadata path — actually downloading here would be a
    catastrophic regression, not a subtle one."""
    assert audio.YDL_BASE["skip_download"] is True


def test_ydl_opts_carries_the_shared_settings():
    opts = audio.ydl_opts(format="bestaudio/best")
    for key, value in audio.YDL_NET.items():
        assert opts.get(key) == value, f"ydl_opts() dropped {key}"


def test_ydl_opts_never_leaks_skip_download():
    """A download built on this must not inherit skip_download from the
    metadata config — that would silently produce no file."""
    assert "skip_download" not in audio.ydl_opts(format="best")


def test_overrides_win():
    assert audio.ydl_opts(socket_timeout=120)["socket_timeout"] == 120


def test_ydl_opts_returns_a_fresh_dict_each_time():
    """Callers mutate these (progress_hooks, download_ranges). Handing back a
    shared reference would leak one command's hooks into the next."""
    a = audio.ydl_opts()
    b = audio.ydl_opts()
    assert a is not b
    a["poisoned"] = True
    assert "poisoned" not in audio.ydl_opts()
    assert "poisoned" not in audio.YDL_NET


def test_source_address_can_be_disabled(monkeypatch):
    """Some hosts have the opposite problem — their IPv4 is the blocked one."""
    import dataclasses
    import importlib
    from core.config import settings as real

    monkeypatch.setattr("core.config.settings",
                        dataclasses.replace(real, ytdlp_source_address=""))
    reloaded = importlib.reload(audio)
    try:
        assert "source_address" not in reloaded.YDL_NET
    finally:
        monkeypatch.undo()
        importlib.reload(audio)


# ── nothing bypasses the base ─────────────────────────────────────────────────

def _yt_dlp_call_sites(path: pathlib.Path) -> list[int]:
    """Line numbers of `yt_dlp.YoutubeDL(...)` constructions in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = ""
        if isinstance(f, ast.Attribute):
            name = f.attr
        elif isinstance(f, ast.Name):
            name = f.id
        if name == "YoutubeDL":
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("relpath", ["cogs/downloads.py", "core/audio.py"])
def test_no_handrolled_option_literals(relpath):
    """Every YoutubeDL(...) must be handed options built from the shared base.

    Catching this structurally rather than by review is the point — the
    original drift was four separate dicts that each looked fine in isolation.
    """
    path = REPO / relpath
    src = path.read_text(encoding="utf-8")
    assert _yt_dlp_call_sites(path), f"no YoutubeDL calls found in {relpath}"

    # The old smell: spreading cookie opts into a literal instead of going
    # through ydl_opts()/YDL_BASE, which is how source_address got lost.
    assert "**audio.ytdlp_cookie_opts()" not in src, (
        f"{relpath} builds yt-dlp options by hand — use audio.ydl_opts() so "
        f"networking settings can't drift")


def test_downloads_cog_uses_the_shared_builder():
    src = (REPO / "cogs/downloads.py").read_text(encoding="utf-8")
    assert "audio.ydl_opts(" in src


# ── 403 handling ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    "HTTP Error 403: Forbidden",
    "urllib.error.HTTPError: HTTP Error 403: Forbidden",
])
def test_403_is_treated_as_retryable(msg):
    """A 403 on a media fetch is normally an expired signed URL. Retrying
    re-extracts and mints a fresh one, so it recovers."""
    assert audio._is_transient_network_error(Exception(msg))


CLOUDFLARE = ("ERROR: [generic] Got HTTP Error 403 caused by Cloudflare anti-bot challenge; "
              "see  https://github.com/yt-dlp/yt-dlp#impersonation  for how to install the "
              "required impersonation dependency, and try again with  --extractor-args "
              "\"generic:impersonate\"")     # yt-dlp's words, as the bot logged them


def test_a_cloudflare_block_is_not_retried_as_a_dead_link():
    """A browser check answers every retry the same way: retrying only made
    the user wait 15 s for a wrong "connection dropped"."""
    assert not audio._is_transient_network_error(Exception(CLOUDFLARE))


@pytest.mark.asyncio
async def test_a_cloudflare_block_fails_at_once_and_says_why(monkeypatch):
    calls = []

    def blocked():
        calls.append(1)
        raise Exception(CLOUDFLARE)

    monkeypatch.setattr(audio.asyncio, "sleep", lambda *_a: pytest.fail("retried"))
    with pytest.raises(Exception) as err:
        await audio._with_retry(blocked)
    assert len(calls) == 1
    msg = audio.friendly_error_msg(err.value)
    assert "anti-bot" in msg and "connection dropped" not in msg
    assert "github.com" not in msg and "impersonat" not in msg


def test_genuine_permanent_errors_are_not_retried():
    for msg in ("Video unavailable", "Private video",
                "This video has been removed by the uploader"):
        assert not audio._is_transient_network_error(Exception(msg))


@pytest.mark.asyncio
async def test_force_bypasses_the_freshness_check(monkeypatch):
    """A 403 means the URL died early, so 'resolved 3 minutes ago' is exactly
    the answer we must not trust."""
    import time as _t

    calls = {"n": 0}

    async def fake_retry(fn, **kw):
        calls["n"] += 1
        return {"url": "https://fresh/media", "webpage_url": "https://y/x"}

    monkeypatch.setattr(audio, "_with_retry", fake_retry)
    track = {
        "webpage_url": "https://y/x",
        "stream_url": "https://stale/media",
        "_resolved_at": _t.time(),      # brand new, well inside the TTL
    }

    await audio.resolve_stream_url(track)
    assert calls["n"] == 0, "a fresh URL should short-circuit"

    await audio.resolve_stream_url(track, force=True)
    assert calls["n"] == 1, "force should re-resolve regardless of the TTL"
    assert track["stream_url"] == "https://fresh/media"


def test_youtube_challenge_runtimes_are_enabled():
    # Deno is yt-dlp's default; Node 22+ is used only if enabled here.
    assert set(audio.YDL_NET["js_runtimes"]) == {"deno", "node"}


def test_runtimes_on_path_are_left_to_ytdlp(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert audio._js_runtimes() == {"deno": {}, "node": {}}


def test_deno_off_path_is_found_where_its_installer_puts_it(monkeypatch, tmp_path):
    # A bot started before Deno was installed doesn't have ~/.deno/bin on its
    # PATH, and `restart` (execv) keeps that PATH — so look there too.
    deno = tmp_path / ".deno" / "bin" / ("deno.exe" if os.name == "nt" else "deno")
    deno.parent.mkdir(parents=True)
    deno.write_text("stand-in")
    deno.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("DENO_INSTALL", raising=False)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    assert audio._js_runtimes() == {"deno": {"path": str(deno)}, "node": {}}


def test_deno_install_dir_is_honoured(monkeypatch, tmp_path):
    custom = tmp_path / "opt-deno"
    deno = custom / "bin" / ("deno.exe" if os.name == "nt" else "deno")
    deno.parent.mkdir(parents=True)
    deno.write_text("stand-in")
    deno.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "nohome"))
    monkeypatch.setenv("DENO_INSTALL", str(custom))
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    assert audio._js_runtimes()["deno"] == {"path": str(deno)}


def test_no_runtime_anywhere_leaves_ytdlp_to_report_it(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("DENO_INSTALL", raising=False)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    assert audio._js_runtimes() == {"deno": {}, "node": {}}


def test_ytdlp_warnings_reach_the_log_once(caplog):
    logger = audio._YtDlpLog()
    msg = ("[youtube] dQw4w9WgXcQ: n challenge solving failed: Some formats may be "
           "missing. https://rr1.googlevideo.com/videoplayback?sig=SECRET")
    with caplog.at_level("WARNING"):
        logger.warning(msg)
        logger.warning(msg.replace("dQw4w9WgXcQ", "aaaaaaaaaaa"))   # same kind, other video
        logger.debug("[youtube] Extracting URL")
    lines = [r.getMessage() for r in caplog.records if "yt-dlp:" in r.getMessage()]
    assert len(lines) == 1 and "n challenge solving failed" in lines[0]
    assert "SECRET" not in lines[0]


def test_downloads_explain_a_cloudflare_block_the_same_way():
    from cogs.downloads import _clean_ytdlp_err
    msg = _clean_ytdlp_err(Exception(CLOUDFLARE))
    assert "anti-bot" in msg and "impersonat" not in msg
