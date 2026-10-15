"""
yt-dlp diagnostics — why did *this* URL fail?

A 403 on YouTube has half a dozen causes that look identical from the outside,
and the usual debugging loop is "run these six commands and read the output".
This does all six in one pass and says which one it is.

The important design choice: it uses the bot's own `YDL_BASE`, cookies and
format string. A diagnosis run with different options tells you nothing about
why the bot failed — it just tells you that some *other* configuration works.

The decisive step is `_probe_media`: extraction succeeding proves nothing,
because a 403 happens later, when the media URL is actually fetched. So this
resolves a format the way the bot would and then issues a 1 KB ranged GET
against the real CDN URL, with the headers yt-dlp would have used. That
reproduces the failure in a second instead of downloading a whole track.

No discord imports — runs from the terminal too:
    venv/bin/python -m core.ytdiag "https://youtu.be/..."
"""
from __future__ import annotations

import asyncio
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field

from core.logging_setup import get_logger

log = get_logger(__name__)

__all__ = ["Report", "diagnose", "JS_RUNTIMES", "SolverStatus", "solver_status",
           "solver_fix_steps"]

#: Runtimes yt-dlp can use for YouTube's player challenge. Only the ones the
#: bot enables (core.audio.YDL_NET["js_runtimes"]) count — yt-dlp ignores a
#: Node or Bun on PATH unless it's enabled.
JS_RUNTIMES = ("deno", "node", "bun", "qjs")

#: Oldest runtime versions yt-dlp's solver supports (yt-dlp EJS wiki).
_MIN_RUNTIME = {"deno": (2, 3, 0), "node": (22, 0, 0)}


def _vtuple(text: str | None) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def _runtime_version(exe: str) -> str | None:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return None
    first = (out or "").strip().splitlines()
    return first[0].strip() if first else None


def _ejs_pin() -> str | None:
    """The yt-dlp-ejs version this yt-dlp pins (its `default` extra)."""
    try:
        from importlib.metadata import requires
        for req in requires("yt-dlp") or []:
            m = re.match(r"\s*yt-dlp-ejs\s*==\s*([\w.]+)", req)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


@dataclass
class SolverStatus:
    """Can yt-dlp solve YouTube's JavaScript challenge on this machine?"""
    found: dict[str, str]          # enabled runtime -> version line (+ where, if off PATH)
    usable: list[str]              # found and new enough
    ejs: str | None                # installed yt-dlp-ejs
    ejs_wanted: str | None         # what the installed yt-dlp pins

    @property
    def ok(self) -> bool:
        return bool(self.usable) and bool(self.ejs) and (
            not self.ejs_wanted or self.ejs == self.ejs_wanted)

    def problems(self) -> list[str]:
        out = []
        if not self.usable:
            old = [f"{n} {v}" for n, v in self.found.items()]
            out.append("no JavaScript runtime yt-dlp can use (needs Deno 2.3+ or Node 22+)"
                       + (f" — found only {', '.join(old)}, too old" if old else ""))
        if not self.ejs:
            out.append("yt-dlp-ejs (YouTube's challenge solver) isn't installed")
        elif self.ejs_wanted and self.ejs != self.ejs_wanted:
            out.append(f"yt-dlp-ejs {self.ejs} is installed but this yt-dlp needs "
                       f"{self.ejs_wanted}")
        return out


def solver_status() -> SolverStatus:
    from core import updater
    from core.audio import YDL_NET
    enabled = YDL_NET.get("js_runtimes") or {"deno": {}}
    found, usable = {}, []
    for name, cfg in enabled.items():
        # The same executable yt-dlp will use: a configured path (a Deno that
        # isn't on PATH — core.audio._js_runtimes), else a PATH lookup.
        exe = (cfg or {}).get("path") or shutil.which("qjs" if name == "quickjs" else name)
        if not exe:
            continue
        ver = _runtime_version(exe)
        found[name] = (ver or "unknown version") + (
            f" — {exe}, not on PATH" if (cfg or {}).get("path") else "")
        tup = _vtuple(ver)
        if tup is None or tup >= _MIN_RUNTIME.get(name, (0, 0, 0)):
            usable.append(name)
    return SolverStatus(found, usable, updater.installed_version("yt-dlp-ejs"), _ejs_pin())


def solver_fix_steps() -> str:
    """How to give yt-dlp what it needs, for this OS."""
    if sys.platform.startswith("win"):
        deno = "winget install --id=DenoLand.Deno"
    elif sys.platform == "darwin":
        deno = "brew install deno"
    else:
        deno = "curl -fsSL https://deno.land/install.sh | sh"
    return (f"1) Install Deno:  {deno}\n"
            f"2) Update yt-dlp with its solver:  -update ytdlp   (or in the bot's "
            f"environment: python -m pip install -U \"yt-dlp[default]\")\n"
            f"3) Restart me (-restart). I find Deno on PATH or in ~/.deno/bin; "
            f"if it's somewhere else, start me from a new terminal instead.")

_PROBE_BYTES = 1024
_PROBE_TIMEOUT = 20


def _looks_prerelease(version: str | None) -> bool:
    return bool(version) and ".dev" in version.lower()


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    verdict: str = ""
    hint: str = ""

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        out = "\n".join(self.lines)
        if self.verdict:
            out += f"\n\n=== VERDICT ===\n{self.verdict}"
        if self.hint:
            out += f"\n{self.hint}"
        return out


def _env_section(rep: Report) -> dict:
    facts = {}

    from core import updater

    running = updater.running_version("yt_dlp")
    on_disk = updater.installed_version("yt-dlp")
    facts["ytdlp"] = running or "unknown"
    facts["ytdlp_disk"] = on_disk

    rep.add(f"yt-dlp        {running or 'unknown'} (running)")
    if on_disk and running and not updater.same_version(on_disk, running):
        # The trap: `update` upgrades the files, but Python keeps the modules
        # it imported at startup. Every tool then reports the new version
        # while the bot keeps executing the old code, so the upgrade looks
        # like it did nothing.
        facts["restart_pending"] = True
        rep.add(f"              {on_disk} INSTALLED — RESTART REQUIRED")
        rep.add("              !! the upgrade hasn't taken effect yet")
    elif on_disk:
        rep.add(f"              {on_disk} on disk (matches — no restart needed)")
    if _looks_prerelease(running):
        rep.add("              (nightly build)")
    rep.add(f"python        {sys.version.split()[0]}")

    solver = solver_status()
    facts["solver"] = solver
    facts["js"] = solver.usable
    rt = ", ".join(f"{n} ({v})" for n, v in solver.found.items()) or "NONE FOUND"
    rep.add(f"js runtime    {rt}")
    rep.add(f"ejs solver    {solver.ejs or 'NOT INSTALLED'}"
            + (f" (this yt-dlp wants {solver.ejs_wanted})"
               if solver.ejs_wanted and solver.ejs != solver.ejs_wanted else ""))
    for problem in solver.problems():
        rep.add(f"  !! {problem}")

    from core.audio import _COOKIE_OPTS
    if _COOKIE_OPTS:
        kind = next(iter(_COOKIE_OPTS))
        rep.add(f"cookies       {kind}")
        path = _COOKIE_OPTS.get("cookiefile")
        if path:
            domains = _cookie_domains(path)
            facts["cookie_domains"] = domains
            shown = ", ".join(sorted(domains)[:8]) or "(none readable)"
            rep.add(f"cookie domains {shown}")
            leaky = {d for d in domains
                     if "google" in d or "youtube" in d}
            if leaky:
                rep.add(f"  !! contains Google/YouTube cookies: {sorted(leaky)}")
                facts["leaky_cookies"] = True
    else:
        rep.add("cookies       none configured")
    return facts


def _cookie_domains(path: str) -> set[str]:
    domains = set()
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if parts and parts[0]:
                    domains.add(parts[0].lstrip("."))
    except OSError as e:
        log.debug("could not read cookie file: %s", e)
    return domains


class _Capture:
    """yt-dlp logger that keeps the warnings — they name the cause ("n
    challenge solving failed", "missing a url", PO tokens…)."""

    def __init__(self):
        self.warnings: list[str] = []

    def debug(self, msg):
        pass

    info = debug

    def warning(self, msg):
        if str(msg) not in self.warnings:
            self.warnings.append(str(msg))

    def error(self, msg):
        pass


def _extract(url: str) -> tuple[dict | None, str, list[str]]:
    """Run extraction with the bot's real options. Returns (info, error,
    yt-dlp's warnings)."""
    import yt_dlp
    from core.audio import YDL_BASE

    capture = _Capture()
    opts = {**YDL_BASE, "quiet": True, "no_warnings": True,
            "skip_download": True, "noplaylist": True, "logger": capture}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False), "", capture.warnings
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", capture.warnings


def _pick(info: dict, selector: str) -> dict | None:
    """The format the bot's selector would choose."""
    import yt_dlp
    from core.audio import YDL_BASE

    opts = {**YDL_BASE, "format": selector, "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            picked = ydl.process_ie_result(dict(info), download=False)
        # Merged selections expose the pieces under requested_formats.
        reqs = picked.get("requested_formats")
        return (reqs[0] if reqs else picked) if picked else None
    except Exception as e:
        log.debug("format selection failed for %r: %s", selector, e)
        return None


async def _probe_media(fmt: dict, *, ipv4: bool = False) -> tuple[int | None, str]:
    """Ranged GET against the real media URL. (status, note).

    This is the step that actually reproduces a 403: extraction can succeed
    while the CDN still refuses the media fetch. By default it connects the
    way FFmpeg does (the OS picks IPv6 or IPv4); `ipv4` forces IPv4, which is
    what yt-dlp itself uses here (YTDLP_SOURCE_ADDRESS) — YouTube's links
    are bound to the address that fetched them.
    """
    import aiohttp

    url = fmt.get("url")
    if not url:
        return None, "format had no URL"
    headers = dict(fmt.get("http_headers") or {})
    headers["Range"] = f"bytes=0-{_PROBE_BYTES - 1}"
    try:
        timeout = aiohttp.ClientTimeout(total=_PROBE_TIMEOUT)
        connector = aiohttp.TCPConnector(family=socket.AF_INET) if ipv4 else None
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as sess:
            async with sess.get(url, headers=headers,
                                allow_redirects=True) as resp:
                await resp.content.read(_PROBE_BYTES)
                return resp.status, resp.reason or ""
    except asyncio.TimeoutError:
        return None, "timed out"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# Matched as whole phrases, not loose substrings. An earlier version tested
# `"age" in err`, which matched the "age" inside "Unable to download API page"
# and confidently diagnosed a proxy failure as an age gate.
_AUTH_PHRASES = (
    "sign in to confirm your age",
    "confirm your age",
    "age-restricted",
    "age restricted",
    "sign in to confirm you're not a bot",
    "login required",
    "private video",
    "members-only",
)
_NETWORK_PHRASES = (
    "unable to connect to proxy",
    "tunnel connection failed",
    "name or service not known",
    "temporary failure in name resolution",
    "connection refused",
    "connection reset",
    "timed out",
)
# Taken from yt-dlp's actual wording. "not available in your country" looks
# right but never matches — the real message is "has not made this video
# available in your country", so the negation sits several words earlier.
_GEO_PHRASES = (
    "available in your country",
    "available from your location",
    "blocked it in your country",
    "geo restricted",
    "geo-restricted",
    "not available in your location",
)


def _explain_extract_failure(err: str, facts: dict) -> str:
    low = (err or "").lower()
    if any(p in low for p in _NETWORK_PHRASES):
        return ("This is a network/proxy failure reaching YouTube at all — not a "
                "yt-dlp or format problem. Check the host's outbound connectivity.")
    if any(p in low for p in _AUTH_PHRASES):
        return ("Age/sign-in gated. That needs YouTube cookies specifically; "
                "Instagram-only cookies don't help here.")
    if any(p in low for p in _GEO_PHRASES):
        return "Region-locked away from wherever this server is hosted."
    solver = facts.get("solver")
    if (solver is not None and not solver.ok) or (solver is None and not facts.get("js")):
        why = "; ".join(solver.problems()) if solver is not None else "no JavaScript runtime found"
        return (f"yt-dlp can't solve YouTube's player challenge here ({why}).\n"
                + solver_fix_steps())
    return "Try `-update ytdlp`, restart, and re-run this."


_SOLVER_WARNINGS = ("challenge solving failed", "javascript runtime",
                    "challenge solver", "ejs")


def _solver_trouble(warnings: list[str]) -> list[str]:
    return [w for w in warnings if any(k in w.lower() for k in _SOLVER_WARNINGS)]


def _describe(fmt: dict) -> str:
    return (f"id={fmt.get('format_id')} ext={fmt.get('ext')} "
            f"acodec={fmt.get('acodec')} proto={fmt.get('protocol')} "
            f"abr={fmt.get('abr')}")


async def diagnose(url: str) -> Report:
    """Full pass over one URL. Safe to call from the event loop."""
    rep = Report()
    rep.add("── environment ──")
    facts = await asyncio.to_thread(_env_section, rep)
    rep.add()

    rep.add("── extraction ──")
    info, err, warnings = await asyncio.to_thread(_extract, url)
    from core.formatters import redact_secrets
    for w in warnings[:8]:
        rep.add(f"yt-dlp warned: {redact_secrets(w)[:300]}")
    facts["solver_warnings"] = _solver_trouble(warnings)
    if info is None:
        rep.add(f"FAILED: {err}")
        rep.verdict = "Extraction itself failed, so this never got as far as a media fetch."
        rep.hint = _explain_extract_failure(err, facts)
        return rep

    rep.add(f"title         {info.get('title')}")
    rep.add(f"duration      {info.get('duration')}s")
    rep.add(f"availability  {info.get('availability')}")
    rep.add(f"live          {info.get('is_live')}")
    fmts = info.get("formats") or []
    rep.add(f"formats       {len(fmts)} available")
    rep.add()

    rep.add("── media probe (this is where a 403 shows up) ──")
    results = {}
    for label, selector in (("bot's selector", "bestaudio/best"),
                            ("plain bestaudio", "bestaudio"),
                            ("yt-dlp default", None)):
        sel = selector or info.get("format_id") or "best"
        fmt = await asyncio.to_thread(_pick, info, sel)
        if not fmt:
            rep.add(f"{label:<16} could not select a format")
            continue
        status, note = await _probe_media(fmt)
        results[label] = status
        flag = "OK " if status and 200 <= status < 400 else "!! "
        rep.add(f"{flag}{label:<16} {status or 'no response'} {note}")
        rep.add(f"   {_describe(fmt)}")
        if status == 403 and label == "bot's selector":
            # FFmpeg connects however the OS prefers; yt-dlp is pinned to IPv4.
            v4, v4note = await _probe_media(fmt, ipv4=True)
            facts["ipv4_rescues"] = bool(v4 and 200 <= v4 < 400)
            rep.add(f"   same URL over IPv4 only: {v4 or 'no response'} {v4note}")

    rep.add()
    ok = [k for k, v in results.items() if v and 200 <= v < 400]
    bad = [k for k, v in results.items() if v == 403]
    solver = facts.get("solver")
    solver_broken = facts.get("solver_warnings") or (solver is not None and not solver.ok)

    if facts.get("ipv4_rescues"):
        rep.verdict = ("Address mismatch: YouTube's link is bound to the IPv4 address "
                       "yt-dlp used, but this machine reaches YouTube over IPv6 by "
                       "default — which is what FFmpeg does. That's the 403.")
        from core.config import settings
        rep.hint = ("Put YTDLP_SOURCE_ADDRESS= (empty) in .env so yt-dlp uses the same "
                    "address family as FFmpeg, then restart." if settings.ytdlp_source_address
                    else "Something on this network routes the two differently — "
                         "try YTDLP_SOURCE_ADDRESS=0.0.0.0 in .env and restart.")
    elif solver_broken and (bad or not results):
        rep.verdict = ("yt-dlp can't solve YouTube's player challenge on this machine, so "
                       "it's working with fewer formats than it should. "
                       "That's the most likely cause of these 403s — fix it first, "
                       "then re-run this.")
        rep.hint = solver_fix_steps()
    elif not results:
        rep.verdict = "No format could be selected at all — likely a yt-dlp/extractor problem."
        rep.hint = "Run `-update ytdlp` and restart."
    elif ok and bad:
        rep.verdict = (f"Format-specific. These 403: {', '.join(bad)}. "
                       f"These work: {', '.join(ok)}.")
        rep.hint = ("The bot's format string is the problem for this video, not "
                    "your server. Loosening it in core/audio.py YDL_BASE fixes it.")
    elif bad and not ok:
        rep.verdict = "Every format 403s — this is environmental, not video-specific."
        if facts.get("leaky_cookies"):
            rep.hint = ("Your cookies.txt carries Google/YouTube cookies. Using a "
                        "logged-in session from a datacenter IP gets 403s — "
                        "re-export scoped to Instagram only.")
        else:
            rep.hint = "Try `-update ytdlp`, restart, and re-run this."
    elif ok:
        rep.verdict = "Everything fetched fine right now — the failure isn't reproducing."
        if solver_broken:
            rep.hint = ("But yt-dlp can't solve YouTube's player challenge here, so it's "
                        "working with fewer formats than it should — the most likely "
                        "source of on-and-off 403s. Fix it:\n" + solver_fix_steps())
        elif facts.get("restart_pending"):
            rep.hint = ("But note the version mismatch above: a newer yt-dlp is "
                        "installed and NOT loaded. Restart me and try again — "
                        "you're testing the old code.")
        else:
            rep.hint = ("Likely a stale/expired media URL at play time rather than "
                        "a config problem. If it only fails for queued tracks, "
                        "that's the stream URL going stale before playback.\n"
                        "If it keeps failing, try `update ytdlp nightly` — YouTube "
                        "fixes land in nightly well before a stable release.")
    return rep


async def _main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m core.ytdiag <url>")
        return 2
    rep = await diagnose(sys.argv[1])
    print(rep.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
