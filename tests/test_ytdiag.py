"""Tests for the yt-dlp diagnostic.

Mostly guarding the failure-classification heuristics, because those are the
part that quietly gives confident wrong answers. The first version tested
`"age" in err`, which matched the "age" inside "Unable to download API page"
and reported a proxy outage as an age-restricted video.
"""
from __future__ import annotations

import os

import pytest

from core import ytdiag


# ── failure classification ────────────────────────────────────────────────────

def test_proxy_error_is_not_mistaken_for_an_age_gate():
    err = ("DownloadError: ERROR: [youtube] abc: Unable to download API page: "
           "('Unable to connect to proxy', OSError('Tunnel connection failed: "
           "403 Forbidden'))")
    hint = ytdiag._explain_extract_failure(err, {"js": ["node"]})
    assert "network" in hint.lower()
    assert "age" not in hint.lower()


def test_dns_failure_reads_as_network():
    hint = ytdiag._explain_extract_failure(
        "Temporary failure in name resolution", {"js": ["node"]})
    assert "network" in hint.lower()


@pytest.mark.parametrize("err", [
    "ERROR: Sign in to confirm your age",
    "This video is age-restricted",
    "ERROR: Private video. Sign in if you've been granted access",
    "Join this channel to get access to members-only content",
])
def test_real_auth_gates_are_detected(err):
    hint = ytdiag._explain_extract_failure(err, {"js": ["node"]})
    assert "cookie" in hint.lower()


def test_geo_block_detected():
    hint = ytdiag._explain_extract_failure(
        "The uploader has not made this video available in your country",
        {"js": ["node"]})
    assert "region" in hint.lower()


def test_missing_js_runtime_is_suggested_when_nothing_else_matches():
    hint = ytdiag._explain_extract_failure("something inscrutable", {"js": []})
    assert "deno" in hint.lower()


def test_generic_failure_falls_back_to_update_advice():
    hint = ytdiag._explain_extract_failure("something inscrutable",
                                           {"js": ["node"]})
    assert "update" in hint.lower()


def test_classification_survives_empty_input():
    assert ytdiag._explain_extract_failure("", {"js": ["node"]})
    assert ytdiag._explain_extract_failure(None, {"js": []})


# ── cookie domain reading ─────────────────────────────────────────────────────

def test_cookie_domains_parsed_from_netscape_file(tmp_path):
    f = tmp_path / "cookies.txt"
    f.write_text(
        "# Netscape HTTP Cookie File\n"
        "\n"
        ".instagram.com\tTRUE\t/\tTRUE\t0\tsessionid\tabc\n"
        ".instagram.com\tTRUE\t/\tTRUE\t0\tds_user_id\t123\n"
        ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\txyz\n"
    )
    domains = ytdiag._cookie_domains(str(f))
    assert domains == {"instagram.com", "youtube.com"}


def test_cookie_domains_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("# comment\n\n\n.example.com\tTRUE\t/\tFALSE\t0\tk\tv\n")
    assert ytdiag._cookie_domains(str(f)) == {"example.com"}


def test_cookie_domains_survives_a_missing_file():
    assert ytdiag._cookie_domains("/nonexistent/cookies.txt") == set()


# ── report assembly ───────────────────────────────────────────────────────────

def test_report_text_includes_verdict_and_hint():
    rep = ytdiag.Report()
    rep.add("line one")
    rep.verdict = "the verdict"
    rep.hint = "the hint"
    out = rep.text()
    assert "line one" in out and "the verdict" in out and "the hint" in out


def test_report_without_verdict_is_still_readable():
    rep = ytdiag.Report()
    rep.add("just this")
    assert rep.text().strip() == "just this"


def test_js_runtimes_list_covers_what_ytdlp_supports():
    assert set(ytdiag.JS_RUNTIMES) >= {"deno", "node"}


@pytest.mark.asyncio
async def test_probe_media_reports_a_missing_url():
    status, note = await ytdiag._probe_media({})
    assert status is None
    assert "no URL" in note


# ── YouTube challenge solver (the usual cause of intermittent 403s) ──────────

def _solver(monkeypatch, *, found, ejs, pin):
    from core import audio, updater
    # Hermetic: on a host with Deno in ~/.deno/bin, the real config would
    # carry its path and bypass the PATH lookup stubbed below.
    monkeypatch.setitem(audio.YDL_NET, "js_runtimes", {"deno": {}, "node": {}})
    monkeypatch.setattr(ytdiag.shutil, "which", lambda name: f"/bin/{name}" if name in found else None)
    monkeypatch.setattr(ytdiag, "_runtime_version", lambda exe: found[exe.rsplit("/", 1)[-1]])
    monkeypatch.setattr(updater, "installed_version", lambda d: ejs if d == "yt-dlp-ejs" else "1")
    monkeypatch.setattr(ytdiag, "_ejs_pin", lambda: pin)
    return ytdiag.solver_status()


def test_solver_ok_with_deno_and_matching_ejs(monkeypatch):
    st = _solver(monkeypatch, found={"deno": "deno 2.5.6 (stable)"}, ejs="0.8.0", pin="0.8.0")
    assert st.ok and st.usable == ["deno"] and st.problems() == []


@pytest.mark.parametrize("found,ejs,pin,needle", [
    ({}, "0.8.0", "0.8.0", "no javascript runtime"),
    ({"deno": "deno 2.1.0"}, "0.8.0", "0.8.0", "too old"),
    ({"node": "v20.11.0"}, "0.8.0", "0.8.0", "too old"),
    ({"deno": "deno 2.5.6"}, None, "0.8.0", "isn't installed"),
    ({"deno": "deno 2.5.6"}, "0.7.0", "0.8.0", "needs 0.8.0"),
])
def test_solver_problems_are_named(monkeypatch, found, ejs, pin, needle):
    st = _solver(monkeypatch, found=found, ejs=ejs, pin=pin)
    assert not st.ok and needle in " ".join(st.problems()).lower()


def test_node_22_counts_when_deno_is_missing(monkeypatch):
    st = _solver(monkeypatch, found={"node": "v22.11.0"}, ejs="0.8.0", pin="0.8.0")
    assert st.ok and st.usable == ["node"]


@pytest.mark.skipif(os.name == "nt", reason="a shell script stands in for deno")
def test_solver_checks_the_deno_ytdlp_will_actually_use(monkeypatch, tmp_path):
    # Deno installed after the bot started: not on its PATH, but core.audio
    # points yt-dlp at ~/.deno/bin/deno — the check must test that same file.
    from core import audio, updater
    deno = tmp_path / "deno"
    deno.write_text("#!/bin/sh\necho 'deno 2.9.7 (stable, release)'\n")
    deno.chmod(0o755)
    monkeypatch.setitem(audio.YDL_NET, "js_runtimes", {"deno": {"path": str(deno)}, "node": {}})
    monkeypatch.setattr(ytdiag.shutil, "which", lambda name: None)
    monkeypatch.setattr(updater, "installed_version", lambda d: "0.8.0" if d == "yt-dlp-ejs" else "1")
    monkeypatch.setattr(ytdiag, "_ejs_pin", lambda: "0.8.0")
    st = ytdiag.solver_status()
    assert st.ok and st.usable == ["deno"]
    assert "not on PATH" in st.found["deno"]


@pytest.mark.parametrize("installed,running,pending", [
    ("2026.8.19", "2026.08.19", False),     # same release, spelled two ways
    ("2026.9.2", "2026.08.19", True),       # really upgraded, not restarted
])
def test_restart_check_compares_releases_not_strings(monkeypatch, installed, running, pending):
    from core import updater
    monkeypatch.setattr(updater, "running_version", lambda m: running)
    monkeypatch.setattr(updater, "installed_version", lambda d: installed)
    monkeypatch.setattr(ytdiag, "solver_status",
                        lambda: ytdiag.SolverStatus({"deno": "deno 2.9.7"}, ["deno"], "0.8.0", "0.8.0"))
    rep = ytdiag.Report()
    facts = ytdiag._env_section(rep)
    assert bool(facts.get("restart_pending")) is pending
    assert ("RESTART REQUIRED" in rep.text()) is pending


def test_fix_steps_are_for_this_os(monkeypatch):
    monkeypatch.setattr(ytdiag.sys, "platform", "win32")
    steps = ytdiag.solver_fix_steps()
    assert "winget install --id=DenoLand.Deno" in steps and 'yt-dlp[default]' in steps
    monkeypatch.setattr(ytdiag.sys, "platform", "linux")
    assert "deno.land/install.sh" in ytdiag.solver_fix_steps()


def _fake_run(monkeypatch, *, warnings, statuses, solver_ok=True, ipv4=None):
    """Drive diagnose() with scripted extraction/probe results."""
    st = ytdiag.SolverStatus({"deno": "deno 2.5.6"} if solver_ok else {},
                             ["deno"] if solver_ok else [], "0.8.0", "0.8.0")
    monkeypatch.setattr(ytdiag, "solver_status", lambda: st)
    monkeypatch.setattr(ytdiag, "_extract", lambda url: ({"title": "T", "formats": [{}]}, "", warnings))
    monkeypatch.setattr(ytdiag, "_pick", lambda info, sel: {"url": "https://h/v", "format_id": "251"})
    calls = iter(statuses)

    async def probe(fmt, *, ipv4=False):
        if ipv4:
            return (ipv4_status, "")
        return (next(calls), "")
    ipv4_status = ipv4
    monkeypatch.setattr(ytdiag, "_probe_media", probe)


def test_unsolved_challenge_is_named_as_the_cause(monkeypatch):
    import asyncio
    _fake_run(monkeypatch, statuses=[403, 403, 403], warnings=[
        "[youtube] abcdefghijk: n challenge solving failed: Some formats may be missing. "
        "Ensure you have a supported JavaScript runtime and challenge solver script "
        "distribution installed."])
    rep = asyncio.run(ytdiag.diagnose("https://youtu.be/abcdefghijk"))
    assert "player challenge" in rep.verdict and "Deno" in rep.hint
    assert "n challenge solving failed" in rep.text()


def test_intermittent_case_still_points_at_the_solver(monkeypatch):
    import asyncio
    _fake_run(monkeypatch, statuses=[200, 200, 200], warnings=[], solver_ok=False)
    rep = asyncio.run(ytdiag.diagnose("https://youtu.be/x"))
    assert "isn't reproducing" in rep.verdict and "Deno" in rep.hint


def test_ipv6_mismatch_is_detected(monkeypatch):
    import asyncio
    _fake_run(monkeypatch, statuses=[403, 403, 403], warnings=[], ipv4=200)
    rep = asyncio.run(ytdiag.diagnose("https://youtu.be/x"))
    assert "Address mismatch" in rep.verdict and "YTDLP_SOURCE_ADDRESS" in rep.hint


def test_preflight_warns_when_the_solver_is_missing(monkeypatch):
    from core import preflight
    st = ytdiag.SolverStatus({}, [], None, "0.8.0")
    monkeypatch.setattr(ytdiag, "solver_status", lambda: st)
    check = preflight._check_youtube_solver()
    assert check.status == preflight.WARN and "403" in check.detail
