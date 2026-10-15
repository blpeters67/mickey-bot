"""
Download / resize cog.

Routes downloads to the right host based on size:

    ≤ 10 MB    → Discord attachment (native inline preview)
    ≤ 200 MB   → catbox.moe (permanent public link)
    ≤ 1 GB     → litterbox.catbox.moe (link expires in 72 hours)
    ≤ 3 GB     → re-encoded down to ~950 MB, then litterbox
    > 3 GB     → refused (compression would mangle quality)

`-dl <query>` downloads as-is. `-resize <link> <size>` always re-encodes to the
target. Both show a status message with a Cancel button while running.

Two neighbouring commands are easy to mix up:
  * `-convert` changes the **container/format** (mp4 → mkv, video → mp3,
    png → jpg) and remuxes instantly where the codecs already fit.
  * `-transcode` changes the **video codec** inside it (av01 → h264 and back).
    That's always a real re-encode, so it always shows a progress bar.

Bug fixes vs the original auto-delete design:
  * No more `setdelete`, upload IDs, or scheduled deletes — files either
    stay forever (catbox) or expire on their own (litterbox).
  * Cancel button kills the running ffmpeg / yt-dlp process and cleans up.
  * `-resize` defaults to mp4 — it never silently strips video unless you
    pass `mp3` explicitly. (Was a bug: yt-dlp would sometimes pick audio-only
    format when given a search query rather than a direct video link.)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.parse

import aiohttp
import discord
import yt_dlp
from discord import app_commands
from discord.ext import commands

from core import audio, db, translate
from core.config import settings
from core.formatters import fmt_duration, fmt_size, parse_seek, parse_size, unwrap_url
from core.logging_setup import get_logger
from core.whisper import KNOWN_MODELS, load_model

log = get_logger(__name__)


class _PlaylistAborted(Exception):
    """Sentinel raised to stop a whole-playlist loop after a single-item cancel."""


# yt-dlp errors come pre-decorated with ANSI color codes (red ERROR: prefix,
# etc.) intended for terminal output. They render as garbage in Discord.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _clean_ytdlp_err(e: Exception) -> str:
    """Strip ANSI codes and the redundant 'ERROR: ' prefix from yt-dlp errors."""
    # Collapse login-gated failures (Instagram, private videos) to one line
    # instead of dumping yt-dlp's cookies instructions at the user.
    hint = audio.auth_required_hint(e)
    if hint:
        return hint
    msg = _ANSI_RE.sub("", str(e)).strip()
    # yt-dlp prefixes every error with "ERROR: " — we have a ❌ already.
    if msg.startswith("ERROR:"):
        msg = msg[len("ERROR:"):].strip()
    # Truncate so we don't blow Discord's 2000-char message limit on long ones.
    if len(msg) > 500:
        msg = msg[:497] + "…"
    return msg or "Unknown error"


# ──────────────────────────────────────────────────────────────────────────────
# Tier thresholds
# ──────────────────────────────────────────────────────────────────────────────

DISCORD_LIMIT    = 10 * 1024 * 1024
CATBOX_MAX       = 200 * 1024 * 1024
LITTERBOX_MAX    = 1024 * 1024 * 1024
RESIZE_CAP       = 3 * 1024 * 1024 * 1024
LITTERBOX_TARGET = int(LITTERBOX_MAX * 0.95)

CATBOX_URL       = "https://catbox.moe/user/api.php"
LITTERBOX_URL    = "https://litterbox.catbox.moe/resources/internals/api.php"
LITTERBOX_EXPIRY = "72h"


# ──────────────────────────────────────────────────────────────────────────────
# -convert support — target formats and ffmpeg command construction
# ──────────────────────────────────────────────────────────────────────────────

CONVERT_IMAGE_TARGETS = {"png", "jpg", "webp", "bmp", "gif"}
CONVERT_AUDIO_TARGETS = {"mp3", "wav", "flac", "ogg", "opus", "m4a"}
CONVERT_VIDEO_TARGETS = {"mp4", "mkv", "webm", "mov"}
CONVERT_TARGETS = CONVERT_IMAGE_TARGETS | CONVERT_AUDIO_TARGETS | CONVERT_VIDEO_TARGETS
_TARGET_ALIASES = {"jpeg": "jpg"}


async def _probe_media(path: str) -> dict:
    """ffprobe a local file → stream/duration facts the convert path needs.

    Returns {'video': bool, 'attached_pic': bool, 'audio': bool,
    'duration': float, 'vcodec': str, 'acodec': str, 'pix_fmt': str,
    'width': int, 'height': int}.
    `attached_pic` (album art in an mp3/m4a) is deliberately separated from
    real video: it can be extracted as an image, but converting to mp4 from
    it would produce nonsense.

    `acodec` / `pix_fmt` exist for -transcode: the first decides whether the
    audio track can ride along as a stream-copy, the second catches 10-bit
    sources (very common for AV1) that have to be dithered down to 8-bit for
    an H.264 target.
    """
    def _probe():
        empty = {"video": False, "attached_pic": False, "audio": False,
                 "duration": 0.0, "vcodec": "", "acodec": "", "pix_fmt": "",
                 "width": 0, "height": 0}
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(r.stdout)
        except Exception:
            return empty
        has_video = attached = has_audio = False
        vcodec = acodec = pix_fmt = ""
        width = height = 0
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                if s.get("disposition", {}).get("attached_pic"):
                    attached = True
                else:
                    has_video = True
                    vcodec  = vcodec  or (s.get("codec_name") or "")
                    pix_fmt = pix_fmt or (s.get("pix_fmt") or "")
                    width   = width   or int(s.get("width") or 0)
                    height  = height  or int(s.get("height") or 0)
            elif s.get("codec_type") == "audio":
                has_audio = True
                acodec = acodec or (s.get("codec_name") or "")
        try:
            dur = float(data.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError):
            dur = 0.0
        return {"video": has_video, "attached_pic": attached,
                "audio": has_audio, "duration": dur, "vcodec": vcodec,
                "acodec": acodec, "pix_fmt": pix_fmt,
                "width": width, "height": height}
    return await asyncio.to_thread(_probe)


# Source codecs where a video stream-copy would "succeed" but produce a file
# players can't handle (a gif stream inside an mkv, say) — always re-encode.
_NO_COPY_VCODECS = {"gif", "png", "mjpeg", "webp", "bmp", "tiff"}


def _build_convert_cmds(src: str, dst: str, target: str, *,
                        allow_copy: bool = True) -> list[list[str]]:
    """Candidate ffmpeg commands for one conversion, tried in order.

    Container-only video changes (mp4/mkv/mov) get a stream-copy attempt
    first — instant and lossless when the codecs fit the new container —
    with a full re-encode as the fallback. `allow_copy=False` skips the
    copy attempt (image-codec sources must always re-encode).
    """
    base = ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
            "-progress", "pipe:1", "-i", src]

    # Images. -frames:v 1 also lets video sources produce a first-frame
    # snapshot and pulls cover art out of tagged audio files.
    if target in ("png", "bmp"):
        return [base + ["-frames:v", "1", dst]]
    if target == "jpg":
        return [base + ["-frames:v", "1", "-q:v", "2", dst]]
    if target == "webp":
        return [base + ["-frames:v", "1", "-q:v", "80", dst]]
    if target == "gif":
        # Two-stage palette in one command — a global 256-color palette from
        # palettegen looks far better than ffmpeg's default gif dithering.
        flt = ("[0:v]fps=12,scale=min(480\\,iw):-1:flags=lanczos,split[a][b];"
               "[a]palettegen[p];[b][p]paletteuse")
        return [base + ["-filter_complex", flt, dst]]

    # Audio. -vn strips video/cover art so the output is purely audio.
    if target == "mp3":
        return [base + ["-vn", "-c:a", "libmp3lame", "-q:a", "2", dst]]
    if target == "wav":
        return [base + ["-vn", "-c:a", "pcm_s16le", dst]]
    if target == "flac":
        return [base + ["-vn", "-c:a", "flac", dst]]
    if target == "ogg":
        return [base + ["-vn", "-c:a", "libvorbis", "-q:a", "5", dst]]
    if target == "opus":
        return [base + ["-vn", "-c:a", "libopus", "-b:a", "128k", dst]]
    if target == "m4a":
        return [base + ["-vn", "-c:a", "aac", "-b:a", "192k", dst]]

    # Video. The scale filter rounds dimensions down to even numbers and
    # pix_fmt pins yuv420p — gif/png sources decode to palette/RGB frames
    # (often with odd dimensions), which makes a bare libx264 encode either
    # fail or produce yuv444 files many players can't show.
    vsafe = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-pix_fmt", "yuv420p"]

    # webm only allows vp8/vp9/av1 + vorbis/opus, so it always re-encodes;
    # cpu-used 4 trades a little quality for sane speed.
    if target == "webm":
        return [base + vsafe + ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0",
                                "-row-mt", "1", "-cpu-used", "4",
                                "-c:a", "libopus", "-b:a", "128k", dst]]

    # mp4 / mkv / mov: remux first, re-encode as fallback.
    faststart = ["-movflags", "+faststart"] if target in ("mp4", "mov") else []
    copy_cmd = base + ["-c", "copy"] + faststart + [dst]
    enc_cmd  = base + vsafe + ["-c:v", "libx264", "-preset", "medium",
                               "-crf", "20", "-c:a", "aac", "-b:a", "192k",
                               "-sn"] + faststart + [dst]
    return [copy_cmd, enc_cmd] if allow_copy else [enc_cmd]


# ──────────────────────────────────────────────────────────────────────────────
# -transcode support — video codec targets, encoder discovery, ffmpeg commands
#
# -convert changes the *container* (and remuxes when it can). -transcode
# changes the *codec* inside it, which always means a real re-encode. The
# common case is an AV1 (av01) file that half the world can't play being
# turned into H.264, and the reverse when you want the file small.
# ──────────────────────────────────────────────────────────────────────────────

TRANSCODE_CODECS = ("h264", "h265", "vp9", "av1")

# ffprobe reports one spelling, people type a dozen others. Everything on the
# right collapses to a canonical key on the left. ffprobe's own names are in
# here too ('hevc', 'av1') so a probe result can go through the same lookup.
_CODEC_ALIASES = {
    "avc": "h264", "avc1": "h264", "h.264": "h264", "x264": "h264",
    "264": "h264", "mpeg4part10": "h264",
    "hevc": "h265", "h.265": "h265", "x265": "h265", "265": "h265",
    "hvc1": "h265", "hev1": "h265",
    "vp09": "vp9", "vp-9": "vp9", "vp9.0": "vp9",
    "av01": "av1", "av-1": "av1", "aom": "av1", "aomenc": "av1",
    "svtav1": "av1", "svt-av1": "av1", "libaom": "av1",
}

_CODEC_LABEL = {
    "h264": "H.264 / AVC",
    "h265": "H.265 / HEVC",
    "vp9":  "VP9",
    "av1":  "AV1",
}

# Shown on the picker buttons' message so nobody has to know codec trivia.
_CODEC_BLURB = {
    "h264": "plays on everything ever made — the safe pick",
    "h265": "~30% smaller than H.264, ideal for Apple devices",
    "vp9":  "open and browser-native, good for web embeds",
    "av1":  "smallest files by far, but slow to encode and newer players only",
}

# ffmpeg's stream/encoder stem for each canonical codec ("h265" is "hevc" to
# ffmpeg, which is why this table exists at all).
_CODEC_FFNAME = {"h264": "h264", "h265": "hevc", "vp9": "vp9", "av1": "av1"}

# Containers each codec is legal in. First entry is the default output; an
# mkv source keeps mkv, since mkv holds all four.
_CODEC_CONTAINERS = {
    "h264": ("mp4", "mkv", "mov"),
    "h265": ("mp4", "mkv", "mov"),
    "vp9":  ("webm", "mkv"),
    "av1":  ("mp4", "mkv", "webm"),
}

TRANSCODE_QUALITIES = ("high", "balanced", "small")
_QUALITY_ALIASES = {
    "best": "high", "hq": "high", "max": "high", "quality": "high",
    "medium": "balanced", "mid": "balanced", "default": "balanced",
    "normal": "balanced", "med": "balanced",
    "low": "small", "tiny": "small", "smallest": "small", "compact": "small",
}

# CRF per codec per tier. These scales are NOT comparable across codecs —
# libx264/libx265 run 0–51, libvpx-vp9 and SVT-AV1 run 0–63 — so each row is
# tuned on its own for roughly the same perceived quality.
_CRF = {
    "h264": {"high": 18, "balanced": 21, "small": 25},
    "h265": {"high": 20, "balanced": 24, "small": 28},
    "vp9":  {"high": 28, "balanced": 32, "small": 37},
    "av1":  {"high": 26, "balanced": 32, "small": 38},
}

# Software encoders, best first. Every codec must have at least one — the
# hardware attempts above them are optional extras.
_SW_ENCODERS = {
    "h264": ("libx264",),
    "h265": ("libx265",),
    "vp9":  ("libvpx-vp9",),
    "av1":  ("libsvtav1", "libaom-av1"),
}

# Hardware encoding is offered for H.264/H.265 only. Their quality knobs
# (-cq / -global_quality / -qp) all share libx264's 0–51 scale, so the CRF
# table maps straight across. The VP9 and AV1 hardware encoders use different
# vendor-specific ranges that these same numbers would silently mis-set —
# producing a file that encodes fine and looks wrong — so those always take
# the software path.
_HW_ORDER  = ("nvenc", "qsv", "videotoolbox", "vaapi")
_HW_CODECS = ("h264", "h265")
_VAAPI_DEVICE = "/dev/dri/renderD128"


def _hw_rate_control(family: str, q: int) -> list[str]:
    """Rate-control flags putting `family` at roughly CRF `q`."""
    if family == "nvenc":
        return ["-rc", "vbr", "-cq", str(q), "-b:v", "0", "-preset", "p5"]
    if family == "qsv":
        return ["-global_quality", str(q), "-preset", "medium"]
    if family == "vaapi":
        return ["-rc_mode", "CQP", "-qp", str(q)]
    if family == "videotoolbox":
        # VideoToolbox inverts the scale: -q:v is 1–100, higher is better.
        return ["-q:v", str(max(1, min(100, int(round((51 - q) / 51 * 100)))))]
    return []


# Audio codecs each container carries as-is. Anything else gets re-encoded to
# the container's fallback rather than failing the mux.
_CONTAINER_AUDIO_COPY = {
    "mp4":  {"aac", "mp3", "ac3", "eac3", "alac"},
    "mov":  {"aac", "mp3", "ac3", "eac3", "alac"},
    "mkv":  {"aac", "mp3", "ac3", "eac3", "alac", "opus", "vorbis", "flac"},
    "webm": {"opus", "vorbis"},
}
_CONTAINER_AUDIO_ENCODE = {
    "mp4":  ["-c:a", "aac", "-b:a", "192k"],
    "mov":  ["-c:a", "aac", "-b:a", "192k"],
    "mkv":  ["-c:a", "aac", "-b:a", "192k"],
    "webm": ["-c:a", "libopus", "-b:a", "128k"],
}

# `ffmpeg -encoders` rows look like " V....D h264_nvenc   NVIDIA NVENC ...".
# Leading V = video encoder. Requiring the name to start alphanumerically
# skips the header legend ("V..... = Video"), which otherwise parses as an
# encoder named "=".
_ENCODER_RE = re.compile(r"^\s*V[.A-Z]{5}\s+([A-Za-z0-9][\w.-]*)")

_encoders_cache: set[str] | None = None


def _available_encoders(*, refresh: bool = False) -> set[str]:
    """Video encoder names this ffmpeg build accepts. Cached after first call.

    Falls back to an empty set if ffmpeg can't be run, which makes
    `_pick_video_encoders` return nothing and the caller report a clean
    "no encoder available" instead of dying inside a subprocess.
    """
    global _encoders_cache
    if _encoders_cache is not None and not refresh:
        return _encoders_cache
    found: set[str] = set()
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=30)
        for line in r.stdout.splitlines():
            m = _ENCODER_RE.match(line)
            if m:
                found.add(m.group(1))
    except Exception:
        log.warning("Couldn't list ffmpeg encoders — transcode falls back to "
                    "software defaults.")
    _encoders_cache = found
    return found


def normalise_codec(name: str | None) -> str | None:
    """'AV01', 'h.265', 'x264' → 'av1', 'h265', 'h264'. None if unrecognised."""
    if not name:
        return None
    key = name.strip().lower().lstrip(".").replace("_", "")
    key = _CODEC_ALIASES.get(key, key)
    return key if key in TRANSCODE_CODECS else None


def normalise_quality(name: str | None) -> str | None:
    """'best' → 'high', 'tiny' → 'small'. None if unrecognised."""
    if not name:
        return None
    key = name.strip().lower()
    key = _QUALITY_ALIASES.get(key, key)
    return key if key in TRANSCODE_QUALITIES else None


def _split_preset_arg(
    preset: str | None, url: str | None, presets: dict, aliases: dict,
    default: str, label: str,
) -> tuple[str, str | None, str | None]:
    """Sort a `<preset> [url]` positional pair. Returns (preset, url, error).

    `-normalize https://…` puts the URL in the preset slot, exactly like
    `-convert` and `-transcode` — so the token is classified by what it is.
    """
    if not preset:
        return default, url, None
    key = aliases.get(preset.strip().lower(), preset.strip().lower())
    if key in presets:
        return key, url, None
    if unwrap_url(preset).startswith(("http://", "https://")) and not url:
        return default, unwrap_url(preset), None
    return default, url, (
        f"❌ Don't know the {label} `{preset}`.\n"
        f"**Options:** {', '.join(f'`{k}`' for k in presets)}\n"
        f"*(or leave it out for `{default}`)*"
    )


def parse_transcode_args(
    tokens, url: str | None,
) -> tuple[str | None, str, str | None, str | None]:
    """Sort -transcode's loose positional tokens into (codec, quality, url, err).

    Prefix invocations can put any of a codec, a quality and a URL in any of
    the positional slots — `transcode <url>`, `transcode small h264`,
    `transcode h264 <url>` — so each token is classified by what it *is*
    rather than where it landed. `err` is a message for the user when a token
    is none of the three; the other fields are meaningless in that case.
    """
    codec: str | None = None
    quality: str | None = None
    for token in (t for t in tokens if t):
        as_codec   = normalise_codec(token)
        as_quality = normalise_quality(token)
        if as_codec:
            codec = as_codec
        elif as_quality:
            quality = as_quality
        elif unwrap_url(token).startswith(("http://", "https://")) and not url:
            url = unwrap_url(token)
        else:
            return None, "balanced", None, (
                f"❌ Don't know what `{token}` means.\n"
                f"**Codecs:** {', '.join(TRANSCODE_CODECS)}\n"
                f"**Quality:** {', '.join(TRANSCODE_QUALITIES)}\n"
                "*(or omit both and I'll offer buttons)*"
            )
    return codec, quality or "balanced", unwrap_url(url) if url else None, None


def _pick_video_encoders(codec: str, *, allow_hw: bool = True) -> list[str]:
    """Encoder names to try for `codec`, fastest/best first, software last.

    The list is a fallback chain, not a promise: a GPU encoder can be present
    in the build and still fail at init (no driver, no free session, headless
    box), so the software encoder always sits at the end as the one that will
    actually work.
    """
    have = _available_encoders()
    out: list[str] = []
    if allow_hw and codec in _HW_CODECS:
        stem = _CODEC_FFNAME[codec]
        for fam in _HW_ORDER:
            name = f"{stem}_{fam}"
            if name not in have:
                continue
            if fam == "vaapi" and not os.path.exists(_VAAPI_DEVICE):
                continue
            out.append(name)
    out.extend(enc for enc in _SW_ENCODERS[codec] if enc in have)
    return out


def _sw_video_args(encoder: str, crf: int) -> list[str]:
    """Quality/speed flags for a software encoder at the given CRF."""
    if encoder == "libx264":
        # profile high + yuv420p is what makes the output play on old phones,
        # smart TVs and Discord's own preview — the whole point of →h264.
        return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-profile:v", "high"]
    if encoder == "libx265":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", str(crf)]
    if encoder == "libvpx-vp9":
        # crf + -b:v 0 is VP9's constant-quality mode; without -b:v 0 the crf
        # is treated as a ceiling on a bitrate-targeted encode instead.
        return ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0",
                "-row-mt", "1", "-cpu-used", "3"]
    if encoder == "libsvtav1":
        # SVT-AV1 preset runs 0–13, higher = faster. 8 keeps a 10-minute clip
        # in the minutes rather than the hours libaom would take.
        return ["-c:v", "libsvtav1", "-crf", str(crf), "-preset", "8"]
    if encoder == "libaom-av1":
        return ["-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0",
                "-cpu-used", "6", "-row-mt", "1"]
    return ["-c:v", encoder, "-crf", str(crf)]


def _target_pix_fmt(codec: str, src_pix_fmt: str) -> str:
    """Pixel format for the output.

    H.264 is pinned to 8-bit on purpose: libx264 will happily encode 10-bit,
    but that's High 10 profile, which most hardware decoders refuse — and
    "make it play everywhere" is exactly why someone picks H.264. AV1 sources
    are frequently 10-bit, so this path gets hit constantly. The other three
    codecs keep 10-bit when the source has it.
    """
    if codec == "h264":
        return "yuv420p"
    return "yuv420p10le" if "10" in (src_pix_fmt or "") else "yuv420p"


def pick_container(codec: str, src_ext: str) -> str:
    """Output container for `codec`, preserving mkv sources."""
    src_ext = (src_ext or "").lower().lstrip(".")
    allowed = _CODEC_CONTAINERS[codec]
    return src_ext if src_ext in allowed else allowed[0]


def _build_transcode_cmds(
    src: str, dst: str, codec: str, *, quality: str, container: str,
    probe: dict, allow_hw: bool = True,
) -> list[tuple[str, list[str]]]:
    """(encoder_name, ffmpeg_cmd) candidates for one transcode, in try order.

    Hardware attempts come first when available, with the software encoder
    last — the caller walks the list until one succeeds. The encoder name
    rides along so the caller can log/report which one actually did the work
    without having to recompute the same preference list.
    """
    crf      = _CRF[codec][quality]
    encoders = _pick_video_encoders(codec, allow_hw=allow_hw)
    pix_fmt  = _target_pix_fmt(codec, probe.get("pix_fmt", ""))

    # Even dimensions: yuv420p chroma subsampling can't represent an odd
    # width or height, and several encoders hard-fail rather than round.
    scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    acodec = (probe.get("acodec") or "").lower()
    if not probe.get("audio"):
        audio_args = ["-an"]
    elif acodec in _CONTAINER_AUDIO_COPY.get(container, set()):
        audio_args = ["-c:a", "copy"]
    else:
        audio_args = list(_CONTAINER_AUDIO_ENCODE.get(
            container, ["-c:a", "aac", "-b:a", "192k"]))

    # First video + first audio only. Multi-track mkv rips carry commentary
    # tracks and subtitles that mp4 either can't hold or would reorder.
    maps = ["-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]

    container_args: list[str] = []
    if container in ("mp4", "mov"):
        container_args += ["-movflags", "+faststart"]
        if codec == "h265":
            # Without hvc1, QuickTime/Safari/iOS show audio and a black frame.
            container_args += ["-tag:v", "hvc1"]

    cmds: list[tuple[str, list[str]]] = []
    for enc in encoders:
        head = ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
                "-progress", "pipe:1"]
        family = enc.rsplit("_", 1)[-1] if "_" in enc else ""

        if family == "vaapi":
            # VAAPI encodes on GPU surfaces, so frames have to be converted
            # and uploaded before they reach the encoder — and the scale has
            # to happen on the CPU side, before hwupload.
            head += ["-vaapi_device", _VAAPI_DEVICE]
            vf   = f"{scale},format=nv12,hwupload"
            vid  = ["-c:v", enc] + _hw_rate_control(family, crf)
        elif family in _HW_ORDER:
            vf  = scale
            vid = ["-c:v", enc, "-pix_fmt", pix_fmt] + _hw_rate_control(family, crf)
        else:
            vf  = scale
            vid = _sw_video_args(enc, crf) + ["-pix_fmt", pix_fmt]

        cmds.append((enc, head + ["-i", src, "-vf", vf] + maps + vid
                          + audio_args + container_args + [dst]))
    return cmds


# ──────────────────────────────────────────────────────────────────────────────
# -normalize / -desilence — audio cleanup
#
# Both are ffmpeg filter passes, but they differ in one important way: loudness
# normalisation leaves the timeline alone, so a video keeps its video track
# untouched (stream-copied). Silence removal *shortens* the audio, which would
# desync any video riding along with it — so that one always produces audio.
# ──────────────────────────────────────────────────────────────────────────────

# Integrated loudness / true-peak / loudness-range targets, in LUFS and dBTP.
# -16 is the de-facto podcast and voice-chat level; -14 is what Spotify,
# YouTube and Apple Music normalise to; -23 is the EBU R128 broadcast spec.
LOUDNESS_PRESETS = {
    "speech":    {"I": -16.0, "TP": -1.5, "LRA": 11.0},
    "music":     {"I": -14.0, "TP": -1.0, "LRA": 11.0},
    "broadcast": {"I": -23.0, "TP": -2.0, "LRA": 7.0},
}
_LOUDNESS_ALIASES = {
    "voice": "speech", "podcast": "speech", "talk": "speech", "default": "speech",
    "streaming": "music", "spotify": "music", "youtube": "music", "song": "music",
    "ebu": "broadcast", "r128": "broadcast", "tv": "broadcast",
}

# Silence trimming aggressiveness. `threshold` is the level below which audio
# counts as silence, `min_silence` how long a quiet patch must be before it's
# cut at all, and `keep` how much of it survives — cutting silence to zero
# makes speech sound like it's been spliced with a razor blade.
SILENCE_PRESETS = {
    "gentle":     {"threshold": -50, "min_silence": 1.0,  "keep": 0.5},
    "normal":     {"threshold": -45, "min_silence": 0.5,  "keep": 0.3},
    "aggressive": {"threshold": -40, "min_silence": 0.25, "keep": 0.1},
}
_SILENCE_ALIASES = {
    "light": "gentle", "soft": "gentle", "safe": "gentle",
    "default": "normal", "medium": "normal", "balanced": "normal",
    "hard": "aggressive", "tight": "aggressive", "max": "aggressive",
}

_LOUDNORM_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


async def _measure_loudness(path: str, target: dict) -> dict | None:
    """Run loudnorm's analysis pass and return its measurements.

    Two-pass loudnorm is the difference between "roughly the right volume"
    and actually hitting the target: the single-pass filter works from a
    running estimate and drifts, most audibly in the first few seconds.
    Returns None if the JSON block can't be found, and the caller falls back
    to a single pass.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", path,
        "-af", (f"loudnorm=I={target['I']}:TP={target['TP']}:"
                f"LRA={target['LRA']}:print_format=json"),
        "-f", "null", "-",
    ]

    def _run():
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        # loudnorm prints its JSON to stderr, after everything else.
        m = _LOUDNORM_JSON_RE.search(r.stderr or "")
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    try:
        return await asyncio.to_thread(_run)
    except Exception:
        log.warning("loudnorm analysis pass failed for %s", path, exc_info=True)
        return None


def _build_normalize_cmd(src: str, dst: str, target: dict, *,
                         measured: dict | None, keep_video: bool,
                         audio_args: list[str]) -> list[str]:
    """Second-pass (or single-pass) loudnorm command."""
    flt = f"loudnorm=I={target['I']}:TP={target['TP']}:LRA={target['LRA']}"
    if measured:
        # linear=true asks for a single gain change across the whole file
        # rather than dynamic compression — it preserves the original
        # dynamics, which is what you want for music and speech alike.
        flt += (f":measured_I={measured['input_i']}"
                f":measured_TP={measured['input_tp']}"
                f":measured_LRA={measured['input_lra']}"
                f":measured_thresh={measured['input_thresh']}"
                f":offset={measured.get('target_offset', 0)}"
                f":linear=true:print_format=summary")
    cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
           "-progress", "pipe:1", "-i", src, "-af", flt]
    if keep_video:
        # The filter only touches audio, so the video can ride along as-is.
        cmd += ["-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-sn", "-dn"]
    # loudnorm resamples to 192 kHz internally; pin the output rate so the
    # encoder doesn't inherit something absurd.
    return cmd + audio_args + ["-ar", "48000", dst]


def _build_desilence_cmd(src: str, dst: str, preset: dict,
                         audio_args: list[str]) -> list[str]:
    """Strip silence from the start, the end, and everywhere between."""
    thr  = preset["threshold"]
    mind = preset["min_silence"]
    keep = preset["keep"]
    # stop_periods=-1 means "every silent stretch", not just the first.
    # detection=rms tracks perceived level, so room tone and breath noise
    # register as silence where a peak detector would call them audio.
    flt = (f"silenceremove="
           f"start_periods=1:start_duration=0:start_threshold={thr}dB:"
           f"start_silence={keep}:"
           f"stop_periods=-1:stop_duration={mind}:stop_threshold={thr}dB:"
           f"stop_silence={keep}:detection=rms")
    return (["ffmpeg", "-y", "-nostats", "-loglevel", "error",
             "-progress", "pipe:1", "-i", src, "-af", flt, "-vn"]
            + audio_args + [dst])


SPEED_MIN = 0.25
SPEED_MAX = 4.0


def parse_speed(text: str | None) -> float | None:
    """'1.5' / '150%' / '1.5x' / 'double' → a speed multiplier, or None."""
    if not text:
        return None
    t = text.strip().lower().rstrip("×")
    named = {"double": 2.0, "twice": 2.0, "half": 0.5, "normal": 1.0,
             "faster": 1.25, "slower": 0.8}
    if t in named:
        return named[t]
    if t.endswith("x"):
        t = t[:-1]
    try:
        value = float(t[:-1]) / 100 if t.endswith("%") else float(t)
    except ValueError:
        return None
    return value if SPEED_MIN <= value <= SPEED_MAX else None


def _atempo_chain(factor: float) -> str:
    """ffmpeg's atempo only accepts 0.5–2.0, so bigger changes are chained.

    Chaining is also *better* than a single wide stretch would be: each stage
    works within the range the algorithm was tuned for, so a 4× speed-up
    stays intelligible instead of turning to warble.
    """
    parts: list[str] = []
    remaining = factor
    while remaining > 2.0 + 1e-9:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        parts.append("atempo=0.5")
        remaining *= 2.0
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def _build_speed_cmd(src: str, dst: str, factor: float, *,
                     has_video: bool, has_audio: bool,
                     audio_args: list[str]) -> list[str]:
    """Change playback speed. Pitch is preserved — atempo time-stretches
    rather than resampling, so a 1.5× podcast doesn't sound like a chipmunk.
    """
    cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
           "-progress", "pipe:1", "-i", src]
    if has_video:
        # setpts rewrites presentation timestamps; 2× speed = half the PTS.
        cmd += ["-vf", f"setpts={1 / factor:.6f}*PTS",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    else:
        cmd += ["-vn"]
    if has_audio:
        cmd += ["-af", _atempo_chain(factor)] + audio_args
    else:
        cmd += ["-an"]
    return cmd + [dst]


def _audio_output_args(src_ext: str) -> tuple[list[str], str]:
    """Encoder flags and file extension for a cleaned-up audio file.

    Lossless in stays lossless; everything else lands on a high-quality mp3
    rather than re-encoding, say, a 96 kbps opus into a 320 kbps mp3 and
    pretending that helped.
    """
    ext = (src_ext or "").lower().lstrip(".")
    if ext in ("wav", "aiff", "aif"):
        return ["-c:a", "pcm_s16le"], "wav"
    if ext == "flac":
        return ["-c:a", "flac"], "flac"
    if ext in ("m4a", "aac", "mp4", "mov", "mkv", "webm"):
        return ["-c:a", "aac", "-b:a", "192k"], "m4a"
    return ["-c:a", "libmp3lame", "-q:a", "2"], "mp3"


# ──────────────────────────────────────────────────────────────────────────────
# -ocr — text out of screenshots
#
# Optional, like faster-whisper: without the `tesseract` binary the command
# says so and everything else keeps working.
# ──────────────────────────────────────────────────────────────────────────────

# Tesseract does much better on large, clean text. Screenshots pasted into
# Discord are often small, so anything under this gets upscaled first.
OCR_MIN_WIDTH   = 1000
OCR_MAX_PIXELS  = 40_000_000     # refuse absurd inputs before Pillow decodes them
OCR_INLINE_CHARS = 1500          # longer results go out as a .txt attachment

_ocr_langs_cache: set[str] | None = None


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _ocr_languages(*, refresh: bool = False) -> set[str]:
    """Language packs installed alongside tesseract. Cached."""
    global _ocr_langs_cache
    if _ocr_langs_cache is not None and not refresh:
        return _ocr_langs_cache
    langs: set[str] = set()
    if tesseract_available():
        try:
            r = subprocess.run(["tesseract", "--list-langs"],
                               capture_output=True, text=True, timeout=30)
            # First line is a header ("List of available languages (4):").
            for line in (r.stdout or "").splitlines()[1:]:
                line = line.strip()
                if line and " " not in line:
                    langs.add(line)
        except Exception:
            log.warning("couldn't list tesseract languages", exc_info=True)
    _ocr_langs_cache = langs
    return langs


def _prep_for_ocr(src: str, dst: str) -> str:
    """Grayscale + upscale a source image to give tesseract a fair chance.

    Returns the path to use — the prepared copy, or the original if Pillow
    isn't installed or the image can't be opened, since tesseract can read
    the raw file perfectly well most of the time.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return src
    try:
        with Image.open(src) as im:
            if im.width * im.height > OCR_MAX_PIXELS:
                return src
            im = ImageOps.exif_transpose(im)
            im = im.convert("L")
            if im.width < OCR_MIN_WIDTH:
                # Cap the enlargement so a 20px sliver doesn't become a wall
                # of interpolated mush — but 6× is enough for a small
                # screenshot to actually reach the target width.
                scale = min(6.0, OCR_MIN_WIDTH / max(1, im.width))
                im = im.resize((int(im.width * scale), int(im.height * scale)),
                               Image.LANCZOS)
            # autocontrast rescues pale-grey-on-white UI text, which is
            # otherwise the most common OCR failure on screenshots.
            im = ImageOps.autocontrast(im)
            im.save(dst)
        return dst
    except Exception:
        log.debug("OCR preprocessing failed, using the original", exc_info=True)
        return src


async def _run_ocr(path: str, lang: str = "eng", *,
                   canceller: JobCanceller | None = None) -> str:
    """Return the text tesseract finds in an image file."""
    cmd = ["tesseract", path, "stdout", "-l", lang, "--psm", "3"]

    def _run():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if canceller is not None:
            canceller.register_proc(proc)
        try:
            out, err = proc.communicate(timeout=300)
        finally:
            if canceller is not None:
                canceller.unregister_proc(proc)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=b"", stderr=(err or "").encode())
        return out

    return await asyncio.to_thread(_run)


def _tidy_ocr_text(raw: str) -> str:
    """Collapse the ragged whitespace tesseract emits into readable lines."""
    lines = [ln.rstrip() for ln in (raw or "").splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            # Tesseract loves emitting runs of empty lines between blocks.
            if blank <= 1 and out:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


# ──────────────────────────────────────────────────────────────────────────────
# -chapters — split a video on its own chapter markers
# ──────────────────────────────────────────────────────────────────────────────

MAX_SPLIT_CHAPTERS = 20      # each one is a separate encode *and* upload
MAX_LISTED_CHAPTERS = 25     # Discord's hard cap on select-menu options


def _clean_chapter_title(title: str, index: int) -> str:
    """Chapter titles come from user-typed descriptions — sanitise for a name."""
    name = re.sub(r"[\\/:*?\"<>|]+", "-", (title or "").strip())
    name = re.sub(r"\s+", " ", name).strip(" .-")
    return name[:80] or f"Chapter {index}"


def extract_chapters(info: dict) -> list[dict]:
    """Normalise yt-dlp's chapter list, filling in any missing end times.

    Some extractors give `start_time` only, so each chapter's end is taken
    from the next one's start (and the last from the video duration). Entries
    without a usable start are dropped rather than guessed at.
    """
    raw = info.get("chapters") or []
    total = float(info.get("duration") or 0)
    out: list[dict] = []
    for i, ch in enumerate(raw, 1):
        try:
            start = float(ch.get("start_time"))
        except (TypeError, ValueError):
            continue
        out.append({
            "index": len(out) + 1,
            "start": max(0.0, start),
            "end":   ch.get("end_time"),
            "title": _clean_chapter_title(ch.get("title") or "", i),
        })
    for i, ch in enumerate(out):
        if ch["end"] is None:
            nxt = out[i + 1]["start"] if i + 1 < len(out) else total
            ch["end"] = nxt
        try:
            ch["end"] = float(ch["end"])
        except (TypeError, ValueError):
            ch["end"] = ch["start"]
    # A zero- or negative-length chapter is a metadata bug, not a clip.
    kept = [c for c in out if c["end"] > c["start"] + 0.5]
    # Renumber so the listing and the dropdown run 1..N with no gaps — the
    # index is what the user picks by, not a reference to the source metadata.
    for n, ch in enumerate(kept, 1):
        ch["index"] = n
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Cancellation token — passed through every long-running step so the Cancel
# button can ask anything in flight to bail.
# ──────────────────────────────────────────────────────────────────────────────

class JobCanceller:
    """Cooperative cancel signal for long-running download/encode jobs."""
    def __init__(self):
        self._event = asyncio.Event()
        self._ffmpeg_procs: list[subprocess.Popen] = []

    def cancel(self) -> None:
        self._event.set()
        # Send SIGTERM to any running ffmpeg child. Doing this from outside
        # the encoder thread is safe — subprocess.Popen is thread-safe for
        # send_signal / terminate.
        for p in self._ffmpeg_procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass

    def register_proc(self, proc: subprocess.Popen) -> None:
        self._ffmpeg_procs.append(proc)

    def unregister_proc(self, proc: subprocess.Popen) -> None:
        try:
            self._ffmpeg_procs.remove(proc)
        except ValueError:
            pass

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


_OCR_MISSING = (
    "❌ OCR isn't available — the host doesn't have **tesseract** installed.\n"
    "-# The bot owner can enable it with `sudo apt install tesseract-ocr` "
    "(plus `tesseract-ocr-deu` etc. for other languages), then restart."
)

_TRANSLATE_MISSING = (
    "❌ Translation isn't enabled on this bot.\n"
    "-# Owner: `pip install argostranslate langdetect` on the host, then "
    "restart. Language models are downloaded per pair with "
    "`translate install <from> <to>`."
)


class _ChapterView(discord.ui.View):
    """Chapter picker: a dropdown for one, a button for all of them.

    Anyone can browse the list, but only the person who ran the command can
    trigger a download — otherwise a chapter list sitting in a busy channel
    becomes a button anybody can use to spend the bot's bandwidth.
    """

    def __init__(self, cog, ctx: commands.Context, url: str, title: str,
                 chapters: list[dict], *, forced: str | None):
        super().__init__(timeout=600)
        self.cog      = cog
        self.ctx      = ctx
        self.url      = url
        self.title    = title
        self.chapters = chapters
        self.forced   = forced
        self.message: discord.Message | None = None

        options = [
            discord.SelectOption(
                label=f"{c['index']}. {c['title']}"[:100],
                description=(f"{fmt_duration(int(c['start']))}–"
                             f"{fmt_duration(int(c['end']))} · "
                             f"{fmt_duration(int(c['end'] - c['start']))}")[:100],
                value=str(c["index"]),
            )
            for c in chapters[:MAX_LISTED_CHAPTERS]
        ]
        select = discord.ui.Select(placeholder="Download one chapter…",
                                   options=options, row=0)
        select.callback = self._on_select
        self.add_item(select)

        if len(chapters) > 1:
            n = min(len(chapters), MAX_SPLIT_CHAPTERS)
            btn = discord.ui.Button(
                label=f"Split all ({n})", emoji="✂️",
                style=discord.ButtonStyle.primary, row=1)
            btn.callback = self._on_split_all
            self.add_item(btn)

        close = discord.ui.Button(label="Close", emoji="✖",
                                  style=discord.ButtonStyle.secondary, row=1)
        close.callback = self._on_close
        self.add_item(close)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user != self.ctx.author:
            await interaction.response.send_message(
                "❌ Only the person who ran the command can download from this "
                "list — run `chapters` yourself for your own.", ephemeral=True)
            return False
        return True

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass

    async def _on_select(self, interaction: discord.Interaction):
        idx = int(interaction.data["values"][0])
        chapter = next((c for c in self.chapters if c["index"] == idx), None)
        await interaction.response.defer()
        if chapter is None:
            return
        await self.cog._download_chapter(self.ctx, self.url, chapter,
                                         forced=self.forced)

    async def _on_split_all(self, interaction: discord.Interaction):
        await interaction.response.defer()
        # One job at a time — the button would otherwise start a second full
        # download on top of the first.
        for item in self.children:
            item.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass
        self.stop()
        await self.cog._split_all_chapters(self.ctx, self.url, self.title,
                                           self.chapters, forced=self.forced)

    async def _on_close(self, interaction: discord.Interaction):
        self.stop()
        try:
            await interaction.message.delete()
        except discord.HTTPException:
            await interaction.response.defer()


class JobView(discord.ui.View):
    """One-button status view with a Cancel that only the requester can press."""
    def __init__(self, requester: discord.abc.User, canceller: JobCanceller):
        super().__init__(timeout=None)
        self.requester = requester
        self.canceller = canceller

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, emoji="✖")
    async def cancel_btn(self, interaction: discord.Interaction, _button: discord.ui.Button):
        if interaction.user.id != self.requester.id:
            return await interaction.response.send_message(
                "❌ Only the requester can cancel this.", ephemeral=True,
            )
        self.canceller.cancel()
        button = self.children[0]
        if isinstance(button, discord.ui.Button):
            button.disabled = True
            button.label    = "Cancelling…"
        await interaction.response.edit_message(view=self)


# ──────────────────────────────────────────────────────────────────────────────
# Anonymous upload helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _upload_anon(
    *, url: str, filepath: str,
    extra_fields: dict[str, str] | None = None,
    timeout_seconds: int = 1800,
) -> str:
    filename = os.path.basename(filepath)
    timeout  = aiohttp.ClientTimeout(total=timeout_seconds)
    with open(filepath, "rb") as fh:
        form = aiohttp.FormData()
        form.add_field("reqtype", "fileupload")
        for k, v in (extra_fields or {}).items():
            form.add_field(k, v)
        form.add_field("fileToUpload", fh, filename=filename,
                       content_type="application/octet-stream")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=form) as resp:
                resp.raise_for_status()
                body = (await resp.text()).strip()
    if not body.startswith("http"):
        raise RuntimeError(f"upload returned an error: {body!r}")
    return body


async def _upload_to_catbox(filepath: str) -> str:
    return await _upload_anon(url=CATBOX_URL, filepath=filepath)


async def _upload_to_litterbox(filepath: str, *, expiry: str = LITTERBOX_EXPIRY) -> str:
    return await _upload_anon(
        url=LITTERBOX_URL, filepath=filepath,
        extra_fields={"time": expiry},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared progress bar — same look for downloads and re-encodes.
# ──────────────────────────────────────────────────────────────────────────────

def _bar(pct: float, *, width: int = 20) -> str:
    """0–100 → '████████░░░░░░░░░░░░' style bar."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct * width / 100))
    return "█" * filled + "░" * (width - filled)


class _ThrottledStatusUpdater:
    """Edit a status message at most once every `interval` seconds, from any
    thread. Coalesces rapid updates so we don't hammer Discord's rate limit.

    Edits are fire-and-forget (scheduled on the loop, not awaited), so a caller
    that writes a *final* message afterwards must call ``aclose()`` first —
    otherwise a still-queued progress edit can execute later and overwrite the
    final result (the intermittent transcribe "stuck at 100%" was exactly this)."""

    def __init__(self, status: discord.Message, loop: asyncio.AbstractEventLoop,
                 *, interval: float = 2.0):
        self.status   = status
        self.loop     = loop
        self.interval = interval
        self._last_t  = 0.0
        self._pending = None
        self._stopped = False
        self._last_future = None      # most recent scheduled edit, awaited by aclose()
        self._lock    = __import__("threading").Lock()

    def update(self, content: str) -> None:
        """Schedule an edit. If called more often than `interval`, the most
        recent text wins and intermediate ones are dropped. No-op once stopped."""
        import time as _t
        now = _t.time()
        with self._lock:
            if self._stopped:
                return
            self._pending = content
            if now - self._last_t < self.interval:
                return
            self._last_t = now
            content_to_send = self._pending
            self._pending = None
            # Track the future inside the lock so aclose() can await whatever
            # was scheduled last (edits to one message serialize, so the last
            # one finishing means the earlier ones did too).
            self._last_future = asyncio.run_coroutine_threadsafe(
                self.status.edit(content=content_to_send), self.loop,
            )

    async def aclose(self) -> None:
        """Stop accepting updates and wait for the last in-flight edit to land,
        so a queued progress edit can't overwrite a final message written after
        this returns. Best-effort and timeout-guarded."""
        with self._lock:
            self._stopped = True
            fut = self._last_future
            self._last_future = None
        if fut is not None:
            try:
                await asyncio.wait_for(asyncio.wrap_future(fut), timeout=10)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Cancellable ffmpeg runner. Uses Popen so we can SIGTERM mid-encode and
# reads ffmpeg's `-progress pipe:1` output to drive a progress bar. Shared by
# the re-encode (-resize / shrink-to-fit) and -convert paths.
# ──────────────────────────────────────────────────────────────────────────────

async def _run_ffmpeg(
    cmd: list[str], *, duration: float = 0.0,
    canceller: JobCanceller | None = None,
    on_progress=None,
) -> None:
    """Run one ffmpeg command to completion.

    `on_progress` (if given) is called with a float 0–100 as the encode
    advances — requires `duration` > 0 and `-progress pipe:1` in the command.
    Called from a thread, so the callback must be threadsafe.

    Raises CalledProcessError on failure, asyncio.CancelledError if the
    canceller fired.
    """
    def _run():
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        if canceller is not None:
            canceller.register_proc(proc)
        try:
            # ffmpeg writes progress lines as `key=value\n`. We care about
            # `out_time_us=` (microseconds into the output). Dividing by the
            # known duration gives us a clean percentage.
            duration_us = duration * 1_000_000
            for line in proc.stdout:
                if not line:
                    break
                if line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        continue
                    if on_progress is not None and duration_us > 0:
                        pct = (us / duration_us) * 100
                        try:
                            on_progress(pct)
                        except Exception:
                            pass
                elif line.startswith("progress=end"):
                    break
            stderr = proc.stderr.read() or ""
            proc.wait()
        finally:
            if canceller is not None:
                canceller.unregister_proc(proc)
        if proc.returncode != 0:
            if canceller is not None and canceller.cancelled:
                raise asyncio.CancelledError("ffmpeg cancelled by user")
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=b"", stderr=stderr.encode(),
            )

    await asyncio.to_thread(_run)


async def _reencode_to_target(
    filepath: str, *, target_bytes: int, has_video: bool, duration: float,
    canceller: JobCanceller | None = None,
    on_progress=None,
) -> str:
    """Re-encode `filepath` toward `target_bytes`.

    If `on_progress` is provided, it gets called with a float 0–100 as the
    encode advances. Called from a thread, so the callback must be threadsafe.
    """
    if duration <= 0:
        raise ValueError("Can't re-encode without a known duration.")
    if target_bytes < 100 * 1024:
        raise ValueError("Target too small (minimum 100 KB).")

    ext    = "mp4" if has_video else "mp3"
    shrunk = filepath.rsplit(".", 1)[0] + "_shrunk." + ext

    # Headroom (×0.97) accounts for muxer overhead.
    total_kbps = max(8, int((target_bytes * 8) / (duration * 1000) * 0.97))

    # `-progress pipe:1` makes ffmpeg write machine-readable progress lines
    # to stdout: `out_time_us=12345678` etc. We parse those for the bar.
    if has_video:
        audio_kbps = min(128, max(48, total_kbps // 8))
        video_kbps = max(80, total_kbps - audio_kbps)
        cmd = [
            "ffmpeg", "-y", "-nostats", "-loglevel", "error",
            "-progress", "pipe:1",
            "-i", filepath,
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", f"{video_kbps}k",
            "-maxrate", f"{int(video_kbps * 1.5)}k",
            "-bufsize", f"{video_kbps * 2}k",
            "-c:a", "aac", "-b:a", f"{audio_kbps}k",
            "-movflags", "+faststart", shrunk,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-nostats", "-loglevel", "error",
            "-progress", "pipe:1",
            "-i", filepath,
            "-c:a", "libmp3lame", "-b:a", f"{total_kbps}k", shrunk,
        ]

    await _run_ffmpeg(cmd, duration=duration, canceller=canceller,
                      on_progress=on_progress)
    return shrunk


# ──────────────────────────────────────────────────────────────────────────────
# The cog
# ──────────────────────────────────────────────────────────────────────────────

class DownloadsCog(commands.Cog, name="Downloads"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        os.makedirs(settings.download_tmp, exist_ok=True)
        self._sweep_stale_workdirs()
        # Whisper model for -transcribe. Separate from the voice-command model
        # (different trade-off: accuracy over latency); loaded lazily, swappable
        # live via `transcribemodel`.
        self._tmodel = None
        self._tmodel_name = settings.whisper_model

        # Right-click → Apps → … on any message. Context menus can't be
        # declared with a decorator inside a cog (there's no `self` at class
        # definition time), so they're built here and registered on the tree.
        self._ctx_menus = [
            app_commands.ContextMenu(name="Transcribe",
                                     callback=self._ctx_transcribe),
            app_commands.ContextMenu(name="Extract text",
                                     callback=self._ctx_ocr),
            app_commands.ContextMenu(name="Convert format",
                                     callback=self._ctx_convert),
            app_commands.ContextMenu(name="Re-encode codec",
                                     callback=self._ctx_transcode),
        ]
        for menu in self._ctx_menus:
            bot.tree.add_command(menu)

    async def cog_load(self) -> None:
        # Restore a transcribe model picked at runtime (else env/config default).
        try:
            self._tmodel_name = await db.get_app_setting(
                "transcribe_whisper_model", settings.whisper_model,
            )
        except Exception:
            pass

    async def cog_unload(self) -> None:
        # Without this a `reload downloads` would leave the old menus on the
        # tree and the re-add would fail with "command already registered".
        for menu in self._ctx_menus:
            self.bot.tree.remove_command(menu.name, type=menu.type)

    def _sweep_stale_workdirs(self) -> None:
        """Delete orphaned musicbot-* job dirs from the download tmp.

        Every job removes its workdir in a finally block, but a crash or a
        console `restart` mid-download skips that, leaking the partial file.
        Only dirs older than a day are swept, so a `reload downloads` can't
        yank the workdir out from under a job that's still running.
        """
        cutoff = time.time() - 24 * 3600
        try:
            for name in os.listdir(settings.download_tmp):
                if not name.startswith("musicbot-"):
                    continue
                path = os.path.join(settings.download_tmp, name)
                try:
                    if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                        shutil.rmtree(path, ignore_errors=True)
                        log.info("Swept stale download workdir %s", name)
                except OSError:
                    continue
        except OSError:
            pass

    # ── Public commands ──────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="download", aliases=["dl"],
        description="Download audio/video and post it here",
        help=(
            "Pulls audio/video from a URL or search query. The bot picks where "
            "to put it based on size: ≤10 MB → Discord attachment (inline "
            "preview), ≤200 MB → catbox.moe (permanent), ≤1 GB → "
            "litterbox.catbox.moe (72h expiry).\n\n"
            "**Force a format** by putting `mp3` or `mp4` before the URL:\n"
            "`dl mp3 rick astley never gonna give you up`\n"
            "`dl mp4 https://soundcloud.com/...`\n\n"
            "**Resize while downloading** with `--resize <size>` — re-encodes "
            "to hit a target size before posting. Useful when a file is just "
            "over the 10 MB Discord cutoff and you want it as a real attachment "
            "instead of a catbox link.\n"
            "Accepts the same formats as `-resize`: `9mb`, `500m`, `1g`, `0.5gb`, "
            "or a bare number (treated as MB). Max 1 GB.\n"
            "`dl --resize 9mb <url>` — force under 10 MB for Discord\n"
            "`dl --resize 100mb <url>` — shrink to ~100 MB\n"
            "`dl mp4 --resize 500m <url>` — combine with format\n\n"
            "**Plain examples**\n"
            "`dl https://youtu.be/dQw4w9WgXcQ`\n"
            "`dl never gonna give you up` — search if it's not a URL\n\n"
            "While running, click ✖ Cancel to stop the operation."
        ),
    )
    async def download(self, ctx: commands.Context, *, query: str):
        # Parse tokens left-to-right: optional `mp3|mp4`, optional `--resize <size>`,
        # then the rest is the query. Order is flexible: `dl mp4 --resize 9mb URL`
        # and `dl --resize 9mb mp4 URL` both work.
        forced       = None
        resize_target = None
        tokens = query.strip().split()

        i = 0
        while i < len(tokens):
            tok = tokens[i].lower()
            if tok in ("mp3", "mp4") and forced is None:
                forced = tok
                i += 1
            elif tok in ("--resize", "-r") and resize_target is None:
                if i + 1 >= len(tokens):
                    return await ctx.send(
                        "❌ `--resize` needs a size. e.g. `--resize 9mb` or `--resize 500m`."
                    )
                resize_target = parse_size(tokens[i + 1])
                if resize_target is None:
                    return await ctx.send(
                        f"❌ Couldn't parse resize target `{tokens[i + 1]}`. "
                        "Try `9mb`, `500m`, `1g`, etc."
                    )
                if resize_target < 100 * 1024:
                    return await ctx.send("❌ Resize target too small — minimum 100 KB.")
                if resize_target > LITTERBOX_MAX:
                    return await ctx.send(
                        f"❌ Resize target {fmt_size(resize_target)} exceeds the "
                        "1 GB upload ceiling. Pick something smaller."
                    )
                i += 2
            else:
                break

        query = unwrap_url(" ".join(tokens[i:]))
        if not query:
            return await ctx.send("❌ Provide a URL or search query.")

        # Playlist URLs need a different code path. Detect them BEFORE handing
        # off to yt-dlp's probe, which on a long playlist would block in a
        # thread we can't cancel from the button.
        if audio.is_any_playlist(query):
            await self._handle_dl_playlist(
                ctx, query, forced=forced, resize_target=resize_target,
            )
            return

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send("🔍 Fetching info…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_download(
                ctx, status, query, forced=forced, resize_target=resize_target,
                workdir=workdir, canceller=canceller, view=view,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()   # timeout=None views stay registered until stopped

    # ── Playlist picker for -dl ──────────────────────────────────────────────

    async def _handle_dl_playlist(
        self, ctx: commands.Context, url: str, *,
        forced: str | None, resize_target: int | None,
    ) -> None:
        """Ask the user what to do with a playlist URL, then dispatch.

        We use a button view (same idiom as the music cog's playlist prompt)
        rather than reactions — buttons are dismissable, scoped to the
        requester, and don't depend on emoji permissions.
        """
        label = audio.playlist_source_label(url)

        class PlaylistDLPrompt(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=30)
                self.choice: str | None = None

            async def interaction_check(self, i: discord.Interaction) -> bool:
                if i.user != ctx.author:
                    await i.response.send_message(
                        "❌ Only the requester can choose.", ephemeral=True,
                    )
                    return False
                return True

            @discord.ui.button(label="First only", style=discord.ButtonStyle.primary, emoji="🎵")
            async def one_btn(self, i: discord.Interaction, _b):
                self.choice = "one";  self.stop();  await i.response.defer()

            @discord.ui.button(label="Whole playlist", style=discord.ButtonStyle.success, emoji="📋")
            async def all_btn(self, i: discord.Interaction, _b):
                self.choice = "all";  self.stop();  await i.response.defer()

            @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary, emoji="✖")
            async def cancel_btn(self, i: discord.Interaction, _b):
                self.choice = "cancel";  self.stop();  await i.response.defer()

        view = PlaylistDLPrompt()
        prompt = await ctx.send(
            f"📋 That's a **{label}** playlist. What should I download?\n"
            "*First only* grabs just the first video. *Whole playlist* downloads "
            "and uploads each entry one by one — can take a while.",
            view=view,
        )
        await view.wait()

        if view.choice in (None, "cancel"):
            return await prompt.edit(content="❌ Cancelled.", view=None)

        await prompt.edit(content="⏳ Loading playlist…", view=None)

        # extract_flat: one cheap HTTP request that returns just titles + URLs,
        # no per-video probing. This is what fetch_playlist() already does.
        try:
            entries = await audio.fetch_playlist(url)
        except Exception as e:
            log.exception("playlist fetch failed for %s", url)
            return await prompt.edit(content=f"❌ Couldn't load playlist: {_clean_ytdlp_err(e)}")

        if not entries:
            return await prompt.edit(content="❌ Playlist is empty or private.")

        if view.choice == "one":
            # Just one item — point the regular download path at the first
            # entry's individual video URL.
            first_url = entries[0].get("webpage_url") or entries[0].get("_search_query")
            if not first_url:
                return await prompt.edit(content="❌ Couldn't resolve the first track's URL.")
            await prompt.delete()
            await self._download_one(
                ctx, first_url, forced=forced, resize_target=resize_target,
            )
            return

        # Whole-playlist mode: iterate, posting each as its own job/status msg.
        await prompt.edit(content=f"📋 Found **{len(entries)} tracks**. Starting downloads…")
        for idx, entry in enumerate(entries, 1):
            track_url = entry.get("webpage_url") or entry.get("_search_query")
            title     = entry.get("title") or "?"
            if not track_url:
                await ctx.send(f"⚠️ [{idx}/{len(entries)}] Skipping **{title}** — no resolvable URL.")
                continue

            try:
                await self._download_one(
                    ctx, track_url, forced=forced, resize_target=resize_target,
                    prefix=f"[{idx}/{len(entries)}] ",
                )
            except _PlaylistAborted:
                await ctx.send(f"✖ Playlist download cancelled at [{idx}/{len(entries)}].")
                return

        await ctx.send(f"✅ Finished playlist ({len(entries)} tracks).")

    async def _download_one(
        self, ctx: commands.Context, url: str, *,
        forced: str | None, resize_target: int | None,
        prefix: str = "",
    ) -> None:
        """Wrap _do_download with its own canceller, workdir, and status."""
        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send(f"{prefix}🔍 Fetching info…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_download(
                ctx, status, url, forced=forced, resize_target=resize_target,
                workdir=workdir, canceller=canceller, view=view,
            )
            if canceller.cancelled:
                # Propagate up so the playlist loop knows to stop.
                raise _PlaylistAborted()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    @commands.hybrid_command(
        name="resize",
        description="Download a video and re-encode it to a target size (e.g. 500mb, 1g)",
        help=(
            "Downloads then re-encodes a video to fit within a target size. "
            "Smart parser accepts many forms — `500mb`, `1g`, `5GB`, `0.5gb`, "
            "or just `500` (treated as MB). Maximum target is 1 GB.\n\n"
            "Defaults to keeping video (mp4). Pass `mp3` as a third argument to "
            "force audio-only output.\n\n"
            "**Examples**\n"
            "`resize https://youtu.be/... 500mb`\n"
            "`resize https://youtu.be/... 1g`\n"
            "`resize https://youtu.be/... 50mb mp3`\n\n"
            "While running, click ✖ Cancel to stop the operation."
        ),
    )
    async def resize(self, ctx: commands.Context, link: str, size: str,
                     fmt: str | None = None):
        link   = unwrap_url(link)
        target = parse_size(size)
        if target is None:
            return await ctx.send(
                "❌ Couldn't parse size. Examples: `500mb`, `1g`, `5GB`, `0.5gb`, "
                "or just `500` (treated as MB)."
            )
        if target < 100 * 1024:
            return await ctx.send("❌ Target too small — minimum is 100 KB.")
        if target > LITTERBOX_MAX:
            return await ctx.send(
                f"❌ Target {fmt_size(target)} exceeds the 1 GB upload ceiling. "
                "Pick something smaller."
            )

        # Default to video unless explicitly asked for mp3.
        forced = "mp3" if (fmt and fmt.lower() == "mp3") else "mp4"
        if fmt and fmt.lower() not in ("mp3", "mp4"):
            return await ctx.send("❌ Format must be `mp3` or `mp4` if provided.")

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send(f"🔍 Fetching `{link}`…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_resize(
                ctx, status, link, target,
                forced=forced, workdir=workdir,
                canceller=canceller, view=view,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    # ── -share: clean meme-friendly repost ───────────────────────────────────

    @commands.hybrid_command(
        name="share", aliases=["repost"],
        description="Download and repost cleanly (deletes your message, no embed/text)",
        help=(
            "Designed for sharing memes from YouTube Shorts / Twitter / etc.\n"
            "  • Deletes your original message so the channel stays tidy "
            "*(needs the **Manage Messages** permission)*\n"
            "  • Reposts the file **as you** — your name and avatar, via a "
            "webhook *(needs **Manage Webhooks**; falls back to a small "
            "'shared by' credit without it)*\n"
            "  • Auto-resizes to fit Discord if the source is too big\n\n"
            "Slash invocations can't delete a 'message' (there isn't one), so "
            "in that case the bot just posts the file directly.\n\n"
            "**Examples**\n"
            "`share https://youtube.com/shorts/abc123`\n"
            "`share https://twitter.com/x/status/123` *(if the URL is downloadable)*\n\n"
            "If the file can't be made small enough for Discord without trashing "
            "quality, the bot will say so and bail. Use `-dl` for big files that "
            "should go to catbox/litterbox."
        ),
    )
    async def share(self, ctx: commands.Context, *, url: str):
        url = unwrap_url(url)
        if not url:
            return await ctx.send("❌ Provide a URL.", delete_after=10)
        if audio.is_any_playlist(url):
            return await ctx.send(
                "❌ Playlists aren't supported for `share`. Use `-dl` for those.",
                delete_after=15,
            )

        # Delete the invoking message so the channel stays clean. Only applies
        # to prefix-command invocations — slash interactions don't have a
        # user message in the same sense. Deleting someone else's message
        # needs Manage Messages; failing that used to be silent, which read
        # as "share is broken" — now it's surfaced at the end.
        delete_failed = False
        if ctx.interaction is None:
            try:
                await ctx.message.delete()
            except discord.Forbidden:
                delete_failed = True
            except discord.NotFound:
                pass  # already gone; harmless

        # silent=True suppresses push notifications — same Discord feature as
        # "Send Silently". The status message is still visible, just unobtrusive.
        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        try:
            status = await ctx.channel.send(
                "⏳ Downloading…", view=view, silent=True,
            )
        except TypeError:
            # Older discord.py without `silent=` — degrade gracefully.
            status = await ctx.channel.send("⏳ Downloading…", view=view)

        workdir = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_share(
                ctx, status, url,
                workdir=workdir, canceller=canceller,
                delete_failed=delete_failed,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    async def _send_as_user(self, ctx: commands.Context, filepath: str) -> bool:
        """Post a file through a webhook wearing the sharer's name and avatar.

        This is how 'post as the user' is done on Discord — webhook messages
        carry a per-message username/avatar override. (Renaming the bot
        itself would be globally visible and rate-limited to ~2 changes/hour.)

        Returns False when webhooks can't be used here (DM, missing the
        Manage Webhooks permission, …) so the caller can fall back.
        """
        if ctx.guild is None:
            return False
        channel = ctx.channel
        thread  = discord.utils.MISSING
        if isinstance(channel, discord.Thread):
            # Webhooks live on the parent channel and target threads per-send.
            thread, channel = channel, channel.parent
            if channel is None:
                return False
        if not channel.permissions_for(ctx.guild.me).manage_webhooks:
            return False

        hooks = await channel.webhooks()
        hook  = next((w for w in hooks if w.user == self.bot.user and w.token), None)
        if hook is None:
            hook = await channel.create_webhook(
                name="musicbot share",
                reason="share command: repost with the sharer's name/avatar",
            )
        await hook.send(
            file=discord.File(filepath, filename=os.path.basename(filepath)),
            username=ctx.author.display_name,
            avatar_url=ctx.author.display_avatar.url,
            thread=thread,
        )
        return True

    async def _do_share(
        self, ctx: commands.Context, status: discord.Message, url: str, *,
        workdir: str, canceller: JobCanceller, delete_failed: bool = False,
    ) -> None:
        result = await self._fetch_via_ytdlp(
            url, forced=None, workdir=workdir, status=status, canceller=canceller,
        )
        if result is None:
            # _fetch_via_ytdlp already edited status with an error. Have it
            # auto-delete so the channel doesn't accumulate failure messages.
            try:
                await status.edit(view=None, delete_after=15)
            except Exception:
                pass
            return
        filepath, _title, duration, has_video, _height = result

        # Discord's real per-guild file size limit. Most servers have 25 MB
        # (boost tier 1 = 25, tier 2 = 50, tier 3 = 100 MB).
        max_bytes    = ctx.guild.filesize_limit if ctx.guild else 25 * 1024 * 1024
        target_bytes = int(max_bytes * 0.95)
        actual       = os.path.getsize(filepath)

        if actual > max_bytes:
            if duration <= 0:
                await status.edit(
                    content="❌ Can't shrink — duration unknown.",
                    view=None, delete_after=10,
                )
                return
            await status.edit(content="⚙️ Shrinking to fit Discord…")

            updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

            def on_progress(pct: float):
                updater.update(
                    f"⚙️ Shrinking to fit Discord\n"
                    f"`[{_bar(pct)}]` {pct:.0f}%"
                )

            try:
                filepath = await _reencode_to_target(
                    filepath, target_bytes=target_bytes,
                    has_video=has_video, duration=duration,
                    canceller=canceller, on_progress=on_progress,
                )
            except asyncio.CancelledError:
                await updater.aclose()
                await status.edit(content="✖ Cancelled.", view=None, delete_after=10)
                return
            except (subprocess.CalledProcessError, ValueError):
                log.exception("share re-encode failed")
                await updater.aclose()
                await status.edit(
                    content="❌ Couldn't shrink to fit cleanly.",
                    view=None, delete_after=10,
                )
                return
            await updater.aclose()

            actual = os.path.getsize(filepath)
            if actual > max_bytes:
                await status.edit(
                    content="❌ Re-encode missed the size target.",
                    view=None, delete_after=10,
                )
                return

        if canceller.cancelled:
            await status.edit(content="✖ Cancelled.", view=None, delete_after=10)
            return

        # Post as the sharer (their name + avatar, via webhook) so the channel
        # still shows who shared it after their original message was deleted.
        # Without webhook permission, fall back to a small text credit.
        posted = False
        try:
            posted = await self._send_as_user(ctx, filepath)
        except discord.HTTPException as e:
            log.warning("webhook share failed (%s) — falling back to plain post", e)
        except Exception:
            log.exception("webhook share failed — falling back to plain post")

        if not posted:
            try:
                await ctx.channel.send(
                    content=f"-# 📤 shared by **{ctx.author.display_name}**",
                    file=discord.File(filepath, filename=os.path.basename(filepath)),
                )
            except discord.HTTPException as e:
                log.warning("share upload failed: %s", e)
                return await status.edit(
                    content=f"❌ Upload failed: {e}", view=None, delete_after=15,
                )

        if delete_failed:
            await status.edit(
                content=("-# ⚠️ I couldn't delete the original message — "
                         "`share` needs the **Manage Messages** permission for that."),
                view=None, delete_after=20,
            )
        else:
            await status.delete()

    # ── -convert: file format conversion ─────────────────────────────────────

    @commands.hybrid_command(
        name="convert",
        description="Convert a file to another format (png→jpg, mp4→mkv, …)",
        help=(
            "Converts an attached file or URL to a different format and posts "
            "the result. Output goes wherever its size fits: ≤10 MB → Discord "
            "attachment, ≤200 MB → catbox.moe, ≤1 GB → litterbox (72h).\n\n"
            "**Images:** png, jpg, webp, bmp, gif\n"
            "**Audio:** mp3, wav, flac, ogg, opus, m4a\n"
            "**Video:** mp4, mkv, webm, mov\n\n"
            "Cross-type conversions that make sense work too: video → mp3 "
            "extracts the audio, video → gif makes an animated gif, video or "
            "tagged audio → png/jpg grabs a frame / the cover art.\n"
            "mp4 ↔ mkv ↔ mov is usually a lossless remux (instant); anything "
            "else re-encodes with a progress bar and a ✖ Cancel button.\n\n"
            "**The format is optional.** `convert <url>` (or `convert` with "
            "just an attachment) downloads the file, detects what it is, and "
            "shows buttons with the sensible targets to pick from.\n\n"
            "**Examples**\n"
            "`convert jpg` *(with a .png attached)*\n"
            "`convert` *(with any file attached — buttons ask what you want)*\n"
            "`convert https://cdn.example.com/video.mp4` — same, for a link\n"
            "`convert mkv https://cdn.example.com/video.mp4`\n"
            "`convert mp3 https://youtu.be/dQw4w9WgXcQ` — yt-dlp links work too"
        ),
    )
    async def convert(
        self, ctx: commands.Context, target: str | None = None,
        attachment: discord.Attachment | None = None,
        *, url: str | None = None,
    ):
        # `convert <url>` puts the URL in the target slot — detect and shift.
        if target is not None:
            t = _TARGET_ALIASES.get(target.lower().lstrip("."),
                                    target.lower().lstrip("."))
            if t in CONVERT_TARGETS:
                target = t
            elif unwrap_url(target).startswith(("http://", "https://")) and not url:
                url, target = unwrap_url(target), None
            else:
                cats = (
                    f"**Images:** {', '.join(sorted(CONVERT_IMAGE_TARGETS))}\n"
                    f"**Audio:** {', '.join(sorted(CONVERT_AUDIO_TARGETS))}\n"
                    f"**Video:** {', '.join(sorted(CONVERT_VIDEO_TARGETS))}"
                )
                return await ctx.send(
                    f"❌ Can't convert to `{target}`. Supported:\n{cats}\n"
                    f"*(or omit the format and I'll offer buttons)*"
                )
        url = unwrap_url(url) if url else url

        # Prefix invocations put the file on the message; slash invocations
        # use the attachment option. discord.py's Attachment converter covers
        # the prefix case, but only if the file is the *first* attachment —
        # fall back to the raw list to be safe.
        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]
        await self._start_convert(ctx, target, attachment=attachment, url=url)

    async def _start_convert(
        self, ctx: commands.Context, target: str | None, *,
        attachment: discord.Attachment | None, url: str | None,
    ) -> None:
        """Shared by `-convert` and the Convert format context menu."""
        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]
        if attachment is None and not url:
            return await ctx.send("❌ Attach a file or provide a URL to convert.")

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send("⬇️ Fetching source…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_convert(
                ctx, status, target, attachment=attachment, url=url,
                workdir=workdir, canceller=canceller, view=view,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    async def _ask_convert_target(
        self, ctx: commands.Context, status: discord.Message,
        probe: dict, src_ext: str, src_name: str, canceller: JobCanceller,
    ) -> str | None:
        """Show target-format buttons for a probed source. Returns the chosen
        target, or None if cancelled/timed out (status already updated)."""
        if probe["video"]:
            kind = "image" if probe["duration"] <= 0.1 else "video"
        elif probe["audio"]:
            kind = "audio"
        elif probe["attached_pic"]:
            kind = "image"
        else:
            await status.edit(content="❌ Couldn't identify that file as "
                                      "audio, video, or an image.", view=None)
            return None

        if kind == "video":
            offers = ["mp4", "mkv", "webm", "mov", "gif", "mp3", "jpg"]
        elif kind == "audio":
            offers = ["mp3", "wav", "flac", "ogg", "opus", "m4a"]
        else:
            offers = ["png", "jpg", "webp", "gif", "bmp"]
        offers = [t for t in offers if t != src_ext]

        picker = discord.ui.View(timeout=60)
        chosen: list[str | None] = []

        async def _check(interaction: discord.Interaction) -> bool:
            if interaction.user != ctx.author:
                await interaction.response.send_message(
                    "❌ Only the requester can choose.", ephemeral=True,
                )
                return False
            return True
        picker.interaction_check = _check

        def _mk_callback(value: str | None):
            async def cb(interaction: discord.Interaction):
                chosen.append(value)
                picker.stop()
                await interaction.response.defer()
            return cb

        for idx, t in enumerate(offers):
            btn = discord.ui.Button(label=f".{t}",
                                    style=discord.ButtonStyle.primary,
                                    row=idx // 4)
            btn.callback = _mk_callback(t)
            picker.add_item(btn)
        cancel = discord.ui.Button(label="Cancel", emoji="✖",
                                   style=discord.ButtonStyle.secondary,
                                   row=(len(offers) - 1) // 4 + 1)
        cancel.callback = _mk_callback(None)
        picker.add_item(cancel)

        kind_label = {"video": "a video 🎬", "audio": "audio 🎵",
                      "image": "an image 🖼️"}[kind]
        await status.edit(
            content=f"📄 **{src_name}** is {kind_label} — convert it to what?",
            view=picker,
        )
        await picker.wait()
        if not chosen or chosen[0] is None or canceller.cancelled:
            await status.edit(content="❌ Cancelled.", view=None)
            return None
        return chosen[0]

    async def _download_direct(
        self, url: str, *, workdir: str, status: discord.Message,
        canceller: JobCanceller,
    ) -> str | None:
        """Stream a direct file URL to disk with a progress bar.

        Returns the saved path, or None if the URL serves an HTML page (the
        caller should hand those to yt-dlp instead). Raises ValueError for
        oversized files and aiohttp errors for network failures.
        """
        timeout = aiohttp.ClientTimeout(total=1800, sock_connect=30)
        updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)
        # try/finally so the last in-flight progress edit always lands before
        # the caller writes its next status — otherwise a queued edit could
        # overwrite it (same race as the transcribe "stuck at 100%").
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(
                    url, headers={"User-Agent": "DiscordMusicBot/2.0"},
                ) as resp:
                    resp.raise_for_status()
                    if "text/html" in (resp.content_type or "").lower():
                        return None
                    total = resp.content_length or 0
                    if total > RESIZE_CAP:
                        raise ValueError(
                            f"File is {fmt_size(total)} — too big to convert "
                            f"(max {fmt_size(RESIZE_CAP)})."
                        )
                    name = os.path.basename(
                        urllib.parse.urlparse(url).path.rstrip("/")
                    ) or "file"
                    path = os.path.join(workdir, name)
                    got  = 0
                    with open(path, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(1 << 18):
                            if canceller.cancelled:
                                return path   # caller re-checks cancelled
                            fh.write(chunk)
                            got += len(chunk)
                            if got > RESIZE_CAP:
                                raise ValueError(
                                    "File exceeded the "
                                    f"{fmt_size(RESIZE_CAP)} convert limit mid-download."
                                )
                            if total:
                                pct = got / total * 100
                                updater.update(
                                    f"⬇️ Downloading source…\n`[{_bar(pct)}]` {pct:.0f}%"
                                )
                            else:
                                updater.update(f"⬇️ Downloading source… {fmt_size(got)}")
            return path
        finally:
            await updater.aclose()

    async def _materialise_source(
        self, *, attachment: discord.Attachment | None, url: str | None,
        workdir: str, status: discord.Message, canceller: JobCanceller,
        label: str,
    ) -> tuple[str, str] | None:
        """Get the job's input onto disk in workdir, from an attachment or URL.

        Returns (path, filename), or None if it failed — in which case the
        status message already explains why. Shared by -convert and -transcode
        so both handle Discord attachments, direct file links and yt-dlp pages
        identically.
        """
        if attachment is not None:
            src_name = os.path.basename(attachment.filename or "") or "file"
            src = os.path.join(workdir, src_name)
            try:
                await attachment.save(src)
            except discord.HTTPException as e:
                await status.edit(
                    content=f"❌ Couldn't fetch the attachment: {e}", view=None,
                )
                return None
            return src, src_name

        try:
            src = await self._download_direct(
                url, workdir=workdir, status=status, canceller=canceller,
            )
        except ValueError as e:
            await status.edit(content=f"❌ {e}", view=None)
            return None
        except Exception as e:
            log.exception("%s: direct download failed for %s", label, url)
            await status.edit(content=f"❌ Download failed: {e}", view=None)
            return None
        if src is None:
            # The URL is a web page, not a file — let yt-dlp deal with it
            # (YouTube, SoundCloud, Twitter, …). raw_media: take whatever
            # streams exist as-is; guessing mp3/mp4 here broke on gif
            # pages, which have no audio for the mp3 postprocessor.
            result = await self._fetch_via_ytdlp(
                url, forced=None, workdir=workdir,
                status=status, canceller=canceller, raw_media=True,
            )
            if result is None:
                return None
            src = result[0]
        return src, os.path.basename(src)

    async def _do_convert(
        self, ctx: commands.Context, status: discord.Message,
        target: str | None, *,
        attachment: discord.Attachment | None, url: str | None,
        workdir: str, canceller: JobCanceller, view: JobView,
    ) -> None:
        # 1. Materialise the source file in workdir.
        got = await self._materialise_source(
            attachment=attachment, url=url, workdir=workdir,
            status=status, canceller=canceller, label="convert",
        )
        if got is None:
            return
        src, src_name = got

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        stem, src_ext = os.path.splitext(src_name)
        src_ext = _TARGET_ALIASES.get(src_ext.lstrip(".").lower(),
                                      src_ext.lstrip(".").lower())

        # 2. Probe; ask for the target if none was given.
        probe = await _probe_media(src)
        if target is None:
            target = await self._ask_convert_target(
                ctx, status, probe, src_ext, src_name, canceller,
            )
            if target is None:
                return
            await status.edit(view=view)   # picker gone — Cancel button back

        if src_ext == target:
            return await status.edit(
                content=f"❌ That file is already `.{target}`.", view=None,
            )

        # 3. Sanity-check the source/target combination.
        visual = probe["video"] or probe["attached_pic"]
        if target in CONVERT_AUDIO_TARGETS and not probe["audio"]:
            return await status.edit(
                content="❌ The source has no audio track.", view=None,
            )
        if target in CONVERT_IMAGE_TARGETS and not visual:
            return await status.edit(
                content="❌ The source has no image or video to convert.", view=None,
            )
        if target in CONVERT_VIDEO_TARGETS and not probe["video"]:
            hint = " (only album art)" if probe["attached_pic"] else ""
            return await status.edit(
                content=(f"❌ The source has no video track{hint} — "
                         "try an audio format like `mp3` instead."),
                view=None,
            )

        # 4. Convert. A/V targets get the progress bar; stills are instant.
        duration = probe["duration"]
        is_av    = target not in CONVERT_IMAGE_TARGETS or target == "gif"
        await status.edit(content=f"⚙️ Converting **{src_name}** → `.{target}`…")

        updater = None
        on_progress = None
        if is_av and duration > 0:
            updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

            def on_progress(pct: float):
                updater.update(
                    f"⚙️ Converting **{src_name}** → `.{target}`\n"
                    f"`[{_bar(pct)}]` {pct:.0f}%"
                )

        async def _updater_done():
            # Let the last in-flight progress edit land before a final edit,
            # so it can't overwrite the result (same race as transcribe's).
            if updater is not None:
                await updater.aclose()

        out  = os.path.join(workdir, f"{stem}.{target}")
        cmds = _build_convert_cmds(
            src, out, target,
            allow_copy=probe["vcodec"] not in _NO_COPY_VCODECS,
        )
        for i, cmd in enumerate(cmds):
            try:
                await _run_ffmpeg(cmd, duration=duration,
                                  canceller=canceller, on_progress=on_progress)
                break
            except asyncio.CancelledError:
                await _updater_done()
                return await status.edit(content="✖ Cancelled.", view=None)
            except subprocess.CalledProcessError as e:
                if i + 1 < len(cmds):
                    continue   # stream-copy attempt failed → try the re-encode
                err = (e.stderr or b"").decode(errors="replace").strip()
                log.warning("convert %s → %s failed: %s", src_name, target, err)
                line = err.splitlines()[-1][:300] if err else "unknown ffmpeg error"
                await _updater_done()
                return await status.edit(
                    content=f"❌ Conversion failed: `{line}`", view=None,
                )
        await _updater_done()

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            return await status.edit(
                content="❌ Conversion produced no output.", view=None,
            )

        await self._route_upload(
            ctx, status, out,
            title=f"{stem}.{target}",
            duration=duration if is_av else 0,
            has_video=target in CONVERT_VIDEO_TARGETS,
            canceller=canceller,
            emoji="🖼️" if target in CONVERT_IMAGE_TARGETS else None,
        )

    # ── -transcode: change the video codec, keep everything else ─────────────

    @commands.hybrid_command(
        name="transcode",
        aliases=["reencode", "recode"],
        description="Re-encode a video to another codec (av1→h264, h264→av1, …)",
        help=(
            "Changes the **video codec** of a file and posts the result. Use "
            "this when a video plays badly or not at all — an AV1 (`av01`) "
            "clip that stutters on an older phone becomes H.264 and just "
            "works. Same size routing as `dl`: ≤10 MB → Discord attachment, "
            "≤200 MB → catbox.moe, ≤1 GB → litterbox (72h).\n\n"
            "**Codecs**\n"
            "`h264` — plays on everything ever made; the safe pick\n"
            "`h265` — ~30% smaller than H.264, ideal for Apple devices\n"
            "`vp9`  — open and browser-native, good for web embeds\n"
            "`av1`  — smallest files by far, slow to encode, newer players only\n\n"
            "**Quality** *(optional, defaults to `balanced`)*\n"
            "`high` · `balanced` · `small` — shifts the quality/size tradeoff.\n\n"
            "The container follows the codec automatically (h264/h265/av1 → "
            "mp4, vp9 → webm; an `.mkv` source stays `.mkv`), the audio track "
            "rides along untouched whenever the container allows it, and GPU "
            "encoding is used for h264/h265 when the host has it.\n\n"
            "**The codec is optional.** `transcode <url>` (or `transcode` "
            "with just an attachment) fetches the file, reports what codec "
            "it's already in, and offers buttons.\n\n"
            "*Changing only the container (mp4↔mkv↔mov) is `convert` — it "
            "remuxes instantly instead of re-encoding.*\n\n"
            "**Examples**\n"
            "`transcode h264` *(with an AV1 video attached)*\n"
            "`transcode` *(with any video attached — buttons ask)*\n"
            "`transcode av1 small https://cdn.example.com/clip.mp4`\n"
            "`transcode h265 high https://youtu.be/dQw4w9WgXcQ`"
        ),
    )
    async def transcode(
        self, ctx: commands.Context,
        codec: str | None = None, quality: str | None = None,
        attachment: discord.Attachment | None = None,
        *, url: str | None = None,
    ):
        codec, quality, url, err = parse_transcode_args((codec, quality), url)
        if err:
            return await ctx.send(err)

        await self._start_transcode(ctx, codec, quality,
                                    attachment=attachment, url=url)

    async def _start_transcode(
        self, ctx: commands.Context, codec: str | None, quality: str, *,
        attachment: discord.Attachment | None, url: str | None,
    ) -> None:
        """Shared by `-transcode` and the Re-encode codec context menu."""
        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]
        if attachment is None and not url:
            return await ctx.send("❌ Attach a video or provide a URL to transcode.")

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send("⬇️ Fetching source…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_transcode(
                ctx, status, codec, quality, attachment=attachment, url=url,
                workdir=workdir, canceller=canceller, view=view,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    async def _ask_transcode_codec(
        self, ctx: commands.Context, status: discord.Message,
        src_codec: str | None, src_name: str, canceller: JobCanceller,
    ) -> str | None:
        """Show codec buttons for a probed video. Returns the chosen codec, or
        None if cancelled/timed out (status already updated)."""
        offers = [c for c in TRANSCODE_CODECS if c != src_codec]

        picker = discord.ui.View(timeout=60)
        chosen: list[str | None] = []

        async def _check(interaction: discord.Interaction) -> bool:
            if interaction.user != ctx.author:
                await interaction.response.send_message(
                    "❌ Only the requester can choose.", ephemeral=True,
                )
                return False
            return True
        picker.interaction_check = _check

        def _mk_callback(value: str | None):
            async def cb(interaction: discord.Interaction):
                chosen.append(value)
                picker.stop()
                await interaction.response.defer()
            return cb

        for idx, c in enumerate(offers):
            btn = discord.ui.Button(
                label=_CODEC_LABEL[c],
                # H.264 is the answer most people fetching this menu want.
                style=(discord.ButtonStyle.success if c == "h264"
                       else discord.ButtonStyle.primary),
                row=idx // 4,
            )
            btn.callback = _mk_callback(c)
            picker.add_item(btn)
        cancel = discord.ui.Button(label="Cancel", emoji="✖",
                                   style=discord.ButtonStyle.secondary,
                                   row=(len(offers) - 1) // 4 + 1)
        cancel.callback = _mk_callback(None)
        picker.add_item(cancel)

        now  = (f"currently **{_CODEC_LABEL[src_codec]}**" if src_codec
                else "in a codec I don't re-encode from by name")
        menu = "\n".join(f"`{c}` — {_CODEC_BLURB[c]}" for c in offers)
        await status.edit(
            content=f"🎬 **{src_name}** is {now} — re-encode it to what?\n{menu}",
            view=picker,
        )
        await picker.wait()
        if not chosen or chosen[0] is None or canceller.cancelled:
            await status.edit(content="❌ Cancelled.", view=None)
            return None
        return chosen[0]

    async def _do_transcode(
        self, ctx: commands.Context, status: discord.Message,
        codec: str | None, quality: str, *,
        attachment: discord.Attachment | None, url: str | None,
        workdir: str, canceller: JobCanceller, view: JobView,
    ) -> None:
        # 1. Materialise the source file in workdir.
        got = await self._materialise_source(
            attachment=attachment, url=url, workdir=workdir,
            status=status, canceller=canceller, label="transcode",
        )
        if got is None:
            return
        src, src_name = got

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        stem, src_ext = os.path.splitext(src_name)
        src_ext = src_ext.lstrip(".").lower()

        # 2. Probe. Unlike -convert this needs a real video stream: there is
        #    no such thing as transcoding album art to H.265.
        probe = await _probe_media(src)
        if not probe["video"]:
            hint = " (only album art)" if probe["attached_pic"] else ""
            return await status.edit(
                content=(f"❌ **{src_name}** has no video track{hint} — "
                         "`transcode` only changes video codecs. For audio "
                         "formats use `convert`."),
                view=None,
            )
        src_codec = normalise_codec(probe["vcodec"])

        # 3. Ask for the target codec if none was given.
        if codec is None:
            codec = await self._ask_transcode_codec(
                ctx, status, src_codec, src_name, canceller,
            )
            if codec is None:
                return
            await status.edit(view=view)   # picker gone — Cancel button back

        if src_codec == codec:
            return await status.edit(
                content=(f"❌ **{src_name}** is already **{_CODEC_LABEL[codec]}**. "
                         "To change just the container, use `convert`."),
                view=None,
            )

        encoders = _pick_video_encoders(codec)
        if not encoders:
            return await status.edit(
                content=(f"❌ This ffmpeg build has no encoder for "
                         f"**{_CODEC_LABEL[codec]}**. Expected one of: "
                         f"`{'`, `'.join(_SW_ENCODERS[codec])}`."),
                view=None,
            )

        # 4. Transcode. Always a full re-encode, so always a progress bar.
        container = pick_container(codec, src_ext)
        duration  = probe["duration"]
        out       = os.path.join(workdir, f"{stem}_{codec}.{container}")
        if os.path.abspath(out) == os.path.abspath(src):
            out = os.path.join(workdir, f"{stem}_{codec}_out.{container}")

        arrow = (f"{_CODEC_LABEL[src_codec]} → {_CODEC_LABEL[codec]}"
                 if src_codec else f"→ {_CODEC_LABEL[codec]}")
        head  = f"⚙️ Transcoding **{src_name}**\n{arrow} · *{quality}*"
        # AV1 has no hardware path here and is genuinely slow — say so up
        # front rather than letting a quiet progress bar look like a hang.
        if codec == "av1":
            head += "\n🐌 AV1 encodes are CPU-bound — this can take a while."
        await status.edit(content=head)

        updater = None
        on_progress = None
        if duration > 0:
            updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

            def on_progress(pct: float):
                updater.update(f"{head}\n`[{_bar(pct)}]` {pct:.0f}%")

        async def _updater_done():
            # Let the last in-flight progress edit land before a final edit,
            # so it can't overwrite the result (same race as transcribe's).
            if updater is not None:
                await updater.aclose()

        cmds = _build_transcode_cmds(
            src, out, codec, quality=quality, container=container, probe=probe,
        )
        used = None
        for i, (enc, cmd) in enumerate(cmds):
            used = enc
            try:
                await _run_ffmpeg(cmd, duration=duration,
                                  canceller=canceller, on_progress=on_progress)
                break
            except asyncio.CancelledError:
                await _updater_done()
                return await status.edit(content="✖ Cancelled.", view=None)
            except subprocess.CalledProcessError as e:
                if i + 1 < len(cmds):
                    # A GPU encoder present in the build can still fail at
                    # init — no driver, no free session, headless host. Drop
                    # to the next candidate, which ends at software.
                    log.info("transcode: %s failed, trying %s",
                             enc, cmds[i + 1][0])
                    # Clear the partial file the failed attempt left behind,
                    # so the empty-output check below can't read it as success.
                    if os.path.exists(out):
                        try:
                            os.remove(out)
                        except OSError:
                            pass
                    continue
                err = (e.stderr or b"").decode(errors="replace").strip()
                log.warning("transcode %s → %s failed: %s", src_name, codec, err)
                line = err.splitlines()[-1][:300] if err else "unknown ffmpeg error"
                await _updater_done()
                return await status.edit(
                    content=f"❌ Transcode failed: `{line}`", view=None,
                )
        await _updater_done()

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            return await status.edit(
                content="❌ Transcode produced no output.", view=None,
            )
        log.info("transcode: %s (%s) → %s via %s [%s]",
                 src_name, src_codec or "unknown", codec, used, quality)

        # A CRF encode targets quality, not size, so a long high-bitrate
        # source can land past the biggest host we have. _route_upload would
        # report that as "the re-encode missed its target", which is wrong
        # here and doesn't tell you the fix.
        size = os.path.getsize(out)
        if size > LITTERBOX_MAX:
            cheaper = {"high": "balanced", "balanced": "small"}.get(quality)
            fix = (f"Try `{ctx.clean_prefix}transcode {codec} {cheaper}`"
                   if cheaper else
                   f"Try `{ctx.clean_prefix}resize` to hit a size target instead")
            return await status.edit(
                content=(f"❌ The **{quality}** {_CODEC_LABEL[codec]} encode came "
                         f"out at **{fmt_size(size)}**, past the "
                         f"{fmt_size(LITTERBOX_MAX)} upload ceiling.\n{fix}."),
                view=None,
            )

        await self._route_upload(
            ctx, status, out,
            title=f"{stem}_{codec}.{container}",
            duration=duration,
            has_video=True,
            canceller=canceller,
        )

    # ── -snip: cut a section out of a video/audio ────────────────────────────

    @commands.hybrid_command(
        name="snip",
        description="Cut a section from a video/audio (start–end timestamps)",
        help=(
            "Cuts the section between two timestamps and posts it — same size "
            "routing as `dl` (≤10 MB → Discord, bigger → catbox/litterbox).\n\n"
            "Timestamps accept `1:30`, `1:23:45`, or plain seconds. The end "
            "time can also be relative to the start: `+30` = 30 seconds after.\n"
            "For YouTube-style links only the clipped section is downloaded, "
            "so clipping 30s out of a 3-hour video is fast.\n\n"
            "Keeps video by default; pass `mp3` at the end for audio-only.\n\n"
            "**Examples**\n"
            "`snip https://youtu.be/... 1:00 1:30`\n"
            "`snip https://youtu.be/... 12:05 +20`\n"
            "`snip https://youtu.be/... 0:45 1:10 mp3`\n\n"
            "While running, click ✖ Cancel to stop the operation.\n\n"
            "*(Renamed from `clip` — that command now clips voice chat.)*"
        ),
    )
    async def snip(self, ctx: commands.Context, url: str, start: str, end: str,
                   fmt: str | None = None):
        url = unwrap_url(url)
        if start.strip().startswith(("+", "-")):
            return await ctx.send("❌ Start must be absolute: `1:30`, `90`, `1:23:45`.")
        s = parse_seek(start, 0.0)
        if s is None:
            return await ctx.send("❌ Bad start time. Use `1:30`, `90`, or `1:23:45`.")
        e = parse_seek(end, s)   # `+30` here means 30s after the start
        if e is None:
            return await ctx.send("❌ Bad end time. Use `1:45`, `105`, or `+15`.")
        if e <= s:
            return await ctx.send("❌ End must be after start.")
        if fmt and fmt.lower() not in ("mp3", "mp4"):
            return await ctx.send("❌ Format must be `mp3` or `mp4` if provided.")

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send("🔍 Fetching info…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_snip(
                ctx, status, url, s, e,
                forced=fmt.lower() if fmt else None,
                workdir=workdir, canceller=canceller,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _do_snip(
        self, ctx: commands.Context, status: discord.Message, url: str,
        start: float, end: float, *,
        forced: str | None, workdir: str, canceller: JobCanceller,
    ) -> None:
        want = end - start
        result = await self._fetch_via_ytdlp(
            url, forced=forced, workdir=workdir, status=status,
            canceller=canceller, clip_range=(start, end),
        )
        if result is None:
            return
        filepath, title, _full_dur, has_video, _height = result
        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        # If the ranged download worked, the file is already the clip. If the
        # extractor fell back to a full download, the duration gives it away
        # — trim locally with ffmpeg.
        file_dur = await audio.get_ffprobe_duration(filepath)
        if file_dur > want + 10:
            if file_dur and start >= file_dur:
                return await status.edit(
                    content=f"❌ Start time is past the end of the source "
                            f"({fmt_duration(int(file_dur))}).",
                    view=None,
                )
            await status.edit(content=f"✂️ Trimming **{title}**…")
            base, ext = os.path.splitext(filepath)
            out  = f"{base}_clip{ext}"
            head = ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
                    "-progress", "pipe:1",
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", filepath]
            if has_video:
                # Re-encode for frame-exact cuts — stream copy snaps to
                # keyframes, which can shift the start by several seconds.
                cmds = [head + ["-c:v", "libx264", "-preset", "medium",
                                "-crf", "20", "-c:a", "aac", "-b:a", "192k",
                                "-movflags", "+faststart", out]]
            else:
                # Audio frames are ~26 ms — copy is effectively exact.
                cmds = [head + ["-c", "copy", out],
                        head + ["-c:a", "libmp3lame", "-q:a", "2", out]]

            updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

            def on_progress(pct: float):
                updater.update(f"✂️ Trimming **{title}**\n`[{_bar(pct)}]` {pct:.0f}%")

            for i, cmd in enumerate(cmds):
                try:
                    await _run_ffmpeg(cmd, duration=want, canceller=canceller,
                                      on_progress=on_progress)
                    break
                except asyncio.CancelledError:
                    return await status.edit(content="✖ Cancelled.", view=None)
                except subprocess.CalledProcessError as exc:
                    if i + 1 < len(cmds):
                        continue
                    err = (exc.stderr or b"").decode(errors="replace").strip()
                    log.warning("clip trim failed: %s", err)
                    return await status.edit(content="❌ Trim failed.", view=None)
            filepath = out

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return await status.edit(content="❌ Clip came out empty — check the "
                                             "timestamps against the video length.",
                                     view=None)

        clip_dur = await audio.get_ffprobe_duration(filepath) or want
        await self._route_upload(
            ctx, status, filepath,
            title=f"{title} (clip {fmt_duration(int(start))}–{fmt_duration(int(end))})",
            duration=clip_dur, has_video=has_video, canceller=canceller,
        )

    # ── Right-click context menus ────────────────────────────────────────────

    async def _ctx_media(
        self, interaction: discord.Interaction, message: discord.Message,
        kinds: tuple[str, ...], noun: str,
    ) -> tuple[discord.Attachment | None, str | None] | None:
        """Find something on a right-clicked message worth acting on.

        Returns (attachment, url) — exactly one of which is set — or None
        after telling the user (privately) that there's nothing usable. The
        ephemeral reply matters here: a failed right-click shouldn't leave
        noise in the channel.
        """
        for att in message.attachments:
            ext = os.path.splitext(att.filename or "")[1].lower().lstrip(".")
            ctype = (att.content_type or "").lower()
            if ext in kinds or any(ctype.startswith(k) for k in
                                   ("image/",) if "image" in kinds):
                return att, None
            if any(ctype.startswith(f"{k}/") for k in ("audio", "video")
                   if k in kinds):
                return att, None
        # Fall back to the first link in the message body.
        m = re.search(r"https?://\S+", message.content or "")
        if m:
            return None, unwrap_url(m.group(0).rstrip(").,>"))

        await interaction.response.send_message(
            f"❌ No {noun} on that message — no matching attachment, and no "
            "link in the text.", ephemeral=True)
        return None

    async def _ctx_context(self, interaction: discord.Interaction) -> commands.Context:
        """A Context for the interaction, with the 3-second clock stopped.

        Deferring first is what lets these jobs take minutes: without it
        Discord marks the interaction failed before the download even starts.
        """
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=False)
        return await commands.Context.from_interaction(interaction)

    async def _ctx_transcribe(self, interaction: discord.Interaction,
                              message: discord.Message):
        found = await self._ctx_media(
            interaction, message,
            ("mp3", "wav", "ogg", "oga", "m4a", "flac", "opus", "webm", "mp4",
             "mkv", "mov", "aac", "wma"),
            "audio or video")
        if found is None:
            return
        att, url = found
        ctx = await self._ctx_context(interaction)
        await self._run_transcribe_job(ctx, attachment=att, url=url)

    async def _ctx_ocr(self, interaction: discord.Interaction,
                       message: discord.Message):
        found = await self._ctx_media(
            interaction, message,
            ("png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff", "image",
             "mp4", "mkv", "mov", "webm"),
            "image")
        if found is None:
            return
        att, url = found
        if not tesseract_available():
            return await interaction.response.send_message(_OCR_MISSING,
                                                           ephemeral=True)
        ctx = await self._ctx_context(interaction)
        await self._run_ocr_job(ctx, "eng", attachment=att, url=url)

    async def _ctx_convert(self, interaction: discord.Interaction,
                           message: discord.Message):
        found = await self._ctx_media(
            interaction, message,
            ("png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff", "image",
             "mp3", "wav", "ogg", "oga", "m4a", "flac", "opus",
             "mp4", "mkv", "mov", "webm", "avi"),
            "convertible file")
        if found is None:
            return
        att, url = found
        ctx = await self._ctx_context(interaction)
        # No target given, so the existing picker asks with buttons — which is
        # exactly the right flow when you've arrived here by right-clicking.
        await self._start_convert(ctx, None, attachment=att, url=url)

    async def _ctx_transcode(self, interaction: discord.Interaction,
                             message: discord.Message):
        found = await self._ctx_media(
            interaction, message,
            ("mp4", "mkv", "mov", "webm", "avi", "m4v", "video"),
            "video")
        if found is None:
            return
        att, url = found
        ctx = await self._ctx_context(interaction)
        await self._start_transcode(ctx, None, "balanced",
                                    attachment=att, url=url)

    # ── -ocr: text out of an image ───────────────────────────────────────────

    @commands.hybrid_command(
        name="ocr", aliases=["readtext", "extracttext"],
        description="Pull the text out of a screenshot or image",
        help=(
            "Reads the text in an attached image (or one at a URL) and posts "
            "it back as selectable, searchable, copy-pasteable text — for the "
            "screenshot of an error message, a receipt, a slide, or a wall of "
            "text someone posted as a picture.\n\n"
            "Short results come back inline; long ones as a `.txt` file. The "
            "image is upscaled and contrast-corrected first, which is what "
            "makes small screenshots readable.\n\n"
            "Point it at a **video** and it reads the first frame.\n\n"
            "**Languages** — pass a code to read non-English text: "
            "`ocr deu` (German), `ocr fra`, `ocr jpn`. Run `ocr langs` to see "
            "what's installed on the host.\n\n"
            "Runs locally via [tesseract](https://github.com/tesseract-ocr/tesseract); "
            "the bot owner enables it with `apt install tesseract-ocr` (plus "
            "`tesseract-ocr-<lang>` for other languages).\n\n"
            "**Examples**\n"
            "`ocr` *(with a screenshot attached)*\n"
            "`ocr https://cdn.example.com/receipt.png`\n"
            "`ocr deu` *(with a German image attached)*"
        ),
    )
    async def ocr(
        self, ctx: commands.Context, lang: str | None = None,
        attachment: discord.Attachment | None = None,
        *, url: str | None = None,
    ):
        if lang and lang.strip().lower() in ("langs", "languages", "list"):
            langs = _ocr_languages()
            if not tesseract_available():
                return await ctx.send(_OCR_MISSING)
            return await ctx.send(
                f"🔤 Installed OCR languages ({len(langs)}): "
                + ", ".join(f"`{l}`" for l in sorted(langs))
                + "\n-# More with `apt install tesseract-ocr-<lang>` on the host."
            )

        # `ocr <url>` puts the URL in the language slot — same shuffle as convert.
        if lang is not None:
            key = lang.strip().lower()
            if key in _ocr_languages() or (not _ocr_languages() and key.isalpha()
                                           and len(key) == 3):
                lang = key
            elif unwrap_url(lang).startswith(("http://", "https://")) and not url:
                url, lang = unwrap_url(lang), None
            else:
                langs = _ocr_languages()
                return await ctx.send(
                    f"❌ No OCR language pack `{lang}` on this host.\n"
                    f"**Installed:** {', '.join(f'`{l}`' for l in sorted(langs)) or '*(none)*'}"
                )
        await self._run_ocr_job(ctx, lang or "eng",
                                attachment=attachment, url=url)

    async def _run_ocr_job(
        self, ctx: commands.Context, lang: str, *,
        attachment: discord.Attachment | None, url: str | None,
    ) -> None:
        if not tesseract_available():
            return await ctx.send(_OCR_MISSING)
        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]
        if attachment is None and not url:
            return await ctx.send("❌ Attach an image or provide a URL to read.")

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send("⬇️ Fetching image…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_ocr(ctx, status, lang, attachment=attachment, url=url,
                               workdir=workdir, canceller=canceller)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    async def _do_ocr(
        self, ctx: commands.Context, status: discord.Message, lang: str, *,
        attachment: discord.Attachment | None, url: str | None,
        workdir: str, canceller: JobCanceller,
    ) -> None:
        got = await self._materialise_source(
            attachment=attachment, url=url, workdir=workdir,
            status=status, canceller=canceller, label="ocr",
        )
        if got is None:
            return
        src, src_name = got
        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        # A video (or anything with a duration) → read its first frame.
        probe = await _probe_media(src)
        if probe["video"] and probe["duration"] > 0.1:
            frame = os.path.join(workdir, "frame.png")
            try:
                await _run_ffmpeg(
                    ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
                     "-i", src, "-frames:v", "1", frame],
                    canceller=canceller)
                src = frame
            except (subprocess.CalledProcessError, asyncio.CancelledError):
                return await status.edit(
                    content="❌ Couldn't grab a frame from that video.", view=None)

        await status.edit(content=f"🔤 Reading **{src_name}**…")
        prepped = _prep_for_ocr(src, os.path.join(workdir, "ocr_prep.png"))
        try:
            raw = await _run_ocr(prepped, lang, canceller=canceller)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode(errors="replace").strip()
            log.warning("ocr failed for %s: %s", src_name, err)
            line = err.splitlines()[-1][:300] if err else "unknown tesseract error"
            return await status.edit(content=f"❌ OCR failed: `{line}`", view=None)
        except Exception as e:
            log.exception("ocr crashed")
            return await status.edit(content=f"❌ OCR failed: {e}", view=None)

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        text = _tidy_ocr_text(raw)
        if not text:
            return await status.edit(
                content=(f"🔤 No text found in **{src_name}**.\n"
                         "-# Low-contrast, very small, or handwritten text often "
                         "won't read. A bigger or sharper image usually helps."),
                view=None,
            )

        words = len(text.split())
        head  = (f"🔤 **{src_name}** — {words:,} word{'s' if words != 1 else ''}"
                 + (f" *({lang})*" if lang != "eng" else ""))
        if len(text) <= OCR_INLINE_CHARS:
            await status.delete()
            # No code fence: the point is text you can select and copy, and a
            # fence makes Discord's mobile client select the whole block.
            return await ctx.send(
                f"{head}\n>>> {text}",
                allowed_mentions=discord.AllowedMentions.none(),
            )

        out = os.path.join(workdir, f"{os.path.splitext(src_name)[0]}.txt")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        await status.delete()
        await ctx.send(f"{head} — too long to inline, so here it is as a file:",
                       file=discord.File(out, filename=os.path.basename(out)))

    # ── -translate: local machine translation ────────────────────────────────

    @commands.hybrid_command(
        name="translate", aliases=["tr", "tl"],
        description="Translate text, or the text in an image, into English",
        help=(
            "Translates text into English by default, or any language you "
            "name. Runs entirely on the host — no API key, no cost, and the "
            "text isn't sent anywhere.\n\n"
            "**Text**\n"
            "`translate hola, ¿qué tal?` — into English\n"
            "`translate de good morning` — into German\n"
            "`translate french how are you`\n\n"
            "**Images** — attach a screenshot and it reads the text with OCR "
            "first, so a foreign error message or menu becomes English in one "
            "step.\n\n"
            "**Setup**\n"
            "`translate langs` — what's installed, and what's available\n"
            "`translate install <from> <to>` — add a language pair "
            "*(bot owner)*; each is a ~100 MB download.\n\n"
            "Models are trained to and from English, so other combinations "
            "hop through it — `de` → `fr` becomes de→en→fr, which costs a "
            "little accuracy.\n\n"
            "Owner setup: `pip install argostranslate langdetect`.\n\n"
            "**Examples**\n"
            "`translate guten morgen`\n"
            "`translate es hello everyone`\n"
            "`translate` *(with a screenshot of foreign text attached)*"
        ),
    )
    async def translate(self, ctx: commands.Context, *, text: str | None = None):
        body = (text or "").strip()
        first, _, rest = body.partition(" ")
        lowered = first.lower()

        if lowered in ("langs", "languages", "list", "models"):
            return await self._translate_langs(ctx)
        if lowered == "install":
            return await self._translate_install(ctx, rest)

        if not translate.available():
            return await ctx.send(_TRANSLATE_MISSING)

        # A leading language name/code is the target: "translate de <text>".
        target = "en"
        if lowered:
            code = translate.normalise_language(first)
            if code and rest.strip():
                target, body = code, rest.strip()

        attachment = (ctx.message.attachments[0]
                      if ctx.message and ctx.message.attachments else None)
        if attachment is None and not body:
            return await ctx.send(
                "❌ Give me some text to translate, or attach an image with "
                "text in it.")

        if attachment is not None and not body:
            return await self._translate_image(ctx, attachment, target)
        await self._translate_and_send(ctx, body, target, source_note="")

    async def _translate_langs(self, ctx: commands.Context) -> None:
        if not translate.available():
            return await ctx.send(_TRANSLATE_MISSING)
        status = await ctx.send("🌐 Checking installed models…")

        installed = await asyncio.to_thread(translate.installed_pairs)
        if not installed:
            body = ("No language models are installed yet.\n"
                    f"The owner can add one with "
                    f"`{ctx.prefix}translate install de en`.")
        else:
            lines = [
                f"`{a}` → `{b}`  ·  {translate.language_name(a)} → "
                f"{translate.language_name(b)}"
                for a, b in installed
            ]
            body = "**Installed**\n" + "\n".join(lines[:40])

        avail = await asyncio.to_thread(translate.available_pairs)
        if avail:
            others = sorted({a for a, b in avail if b == "en"}
                            - {a for a, b in installed})
            if others:
                body += ("\n\n**Available to install** *(to/from English)*\n"
                         + ", ".join(f"`{c}`" for c in others[:60]))

        if not translate.detection_available():
            body += ("\n\n-# `langdetect` isn't installed, so I can't guess "
                     "the source language — I'll assume it from what's "
                     "installed.")
        await status.edit(content=body[:4000])

    @commands.is_owner()
    async def _translate_install_impl(self, ctx, src: str, dst: str) -> None:
        status = await ctx.send(
            f"⬇️ Installing **{translate.language_name(src)} → "
            f"{translate.language_name(dst)}**… this downloads ~100 MB, so "
            "give it a minute.")
        try:
            await asyncio.to_thread(translate.install_pair, src, dst)
        except translate.TranslateError as e:
            return await status.edit(content=f"❌ {e}")
        except Exception as e:
            log.exception("argos pair install failed %s→%s", src, dst)
            return await status.edit(content=f"❌ Install failed: {e}")
        await status.edit(
            content=f"✅ Installed **{translate.language_name(src)} → "
                    f"{translate.language_name(dst)}**.")

    async def _translate_install(self, ctx: commands.Context, rest: str) -> None:
        if not translate.available():
            return await ctx.send(_TRANSLATE_MISSING)
        if not await self.bot.is_owner(ctx.author):
            return await ctx.send(
                "❌ Only the bot owner can install language models — each one "
                "is a ~100 MB download onto the host.")
        parts = rest.split()
        if len(parts) != 2:
            return await ctx.send(
                f"❌ Usage: `{ctx.prefix}translate install <from> <to>` — "
                f"e.g. `{ctx.prefix}translate install de en`.")
        src = translate.normalise_language(parts[0])
        dst = translate.normalise_language(parts[1])
        if not src or not dst:
            return await ctx.send(
                f"❌ Don't recognise `{parts[0]}` or `{parts[1]}` as a "
                f"language. `{ctx.prefix}translate langs` lists what's available.")
        await self._translate_install_impl(ctx, src, dst)

    async def _translate_image(self, ctx: commands.Context,
                               attachment: discord.Attachment,
                               target: str) -> None:
        """OCR an attached image, then translate what it found."""
        if not tesseract_available():
            return await ctx.send(
                "❌ I'd need OCR to read text out of that image, and "
                "tesseract isn't installed on this host.\n"
                "-# Paste the text instead and I'll translate it.")

        canceller = JobCanceller()
        status  = await ctx.send("🔤 Reading the image…")
        workdir = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            got = await self._materialise_source(
                attachment=attachment, url=None, workdir=workdir,
                status=status, canceller=canceller, label="translate")
            if got is None:
                return
            src, src_name = got
            prepped = _prep_for_ocr(src, os.path.join(workdir, "prep.png"))
            try:
                raw = await _run_ocr(prepped, "eng", canceller=canceller)
            except Exception as e:
                log.warning("translate: OCR failed on %s: %s", src_name, e)
                return await status.edit(content="❌ Couldn't read that image.")

            found = _tidy_ocr_text(raw)
            if not found:
                return await status.edit(
                    content=f"🔤 No text found in **{src_name}**.")
            await status.delete()
            await self._translate_and_send(
                ctx, found, target, source_note=f" *(read from {src_name})*")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _translate_and_send(self, ctx: commands.Context, body: str,
                                  target: str, *, source_note: str) -> None:
        if len(body) > translate.MAX_CHARS:
            return await ctx.send(
                f"❌ That's {len(body):,} characters — the limit is "
                f"{translate.MAX_CHARS:,}.")

        installed = await asyncio.to_thread(translate.installed_pairs)
        if not installed:
            return await ctx.send(
                "❌ No translation models are installed yet.\n"
                f"-# The owner can add one: `{ctx.prefix}translate install de en`.")

        detected = await asyncio.to_thread(translate.detect_language, body)
        if detected is None:
            # No detector (or too short a sample) — infer from what's
            # installed. With one non-target language available there's only
            # one thing it can be.
            candidates = {a for a, b in installed if b == target and a != target}
            if len(candidates) == 1:
                detected = candidates.pop()
            else:
                return await ctx.send(
                    "❌ I can't tell what language that is.\n"
                    + ("-# `langdetect` isn't installed on the host, so name "
                       "the language yourself — but the target slot only takes "
                       "the *destination*. Ask the owner to "
                       "`pip install langdetect`."
                       if not translate.detection_available()
                       else "-# Try a longer sample — very short text is "
                            "guesswork.")
                )

        if detected == target:
            return await ctx.send(
                f"🌐 That already looks like **{translate.language_name(target)}**.",
                allowed_mentions=discord.AllowedMentions.none())

        if not translate.can_translate(detected, target):
            return await ctx.send(
                f"❌ No **{translate.language_name(detected)} → "
                f"{translate.language_name(target)}** model is installed.\n"
                f"-# Owner: `{ctx.prefix}translate install {detected} {target}`"
                + (f" (and `{detected} en` + `en {target}` also works — "
                   "models route through English)" if target != "en" else "")
            )

        async with ctx.typing():
            try:
                result = await asyncio.to_thread(
                    translate.translate_text, body, detected, target)
            except translate.TranslateError as e:
                return await ctx.send(f"❌ {e}")
            except Exception:
                log.exception("translation failed")
                return await ctx.send("❌ Translation failed.")

        head = (f"🌐 **{translate.language_name(detected)} → "
                f"{translate.language_name(target)}**{source_note}")
        hop = ("\n-# routed through English — accuracy takes a small hit"
               if (detected, target) not in set(installed) and target != "en"
               and detected != "en" else "")

        if len(result) <= 1800:
            return await ctx.send(f"{head}\n>>> {result}{hop}",
                                  allowed_mentions=discord.AllowedMentions.none())
        workdir = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            out = os.path.join(workdir, f"translated_{target}.txt")
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(result)
            await ctx.send(f"{head} — too long to inline:{hop}",
                           file=discord.File(out, filename=os.path.basename(out)))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ── -chapters: split a video on its chapter markers ──────────────────────

    @commands.hybrid_command(
        name="chapters", aliases=["chapter", "split"],
        description="List a video's chapters and download any of them",
        help=(
            "Reads the chapter markers a video already has and lists them with "
            "timestamps and lengths — then lets you pull one out, or split the "
            "whole thing, without working out the timecodes yourself.\n\n"
            "Pick a single chapter from the dropdown, or **Split all** to get "
            f"every chapter as its own file (up to {MAX_SPLIT_CHAPTERS}). Each "
            "one is routed like `dl`: ≤10 MB → Discord, bigger → catbox / "
            "litterbox.\n\n"
            "Add `mp3` to get audio instead of video — handy for pulling one "
            "track off an album upload or one section out of a long podcast.\n\n"
            "Only sources that *publish* chapters have them (most long-form "
            "YouTube videos do). If there are none, the bot says so and points "
            "you at `snip` for a manual range.\n\n"
            "**Examples**\n"
            "`chapters https://youtu.be/dQw4w9WgXcQ`\n"
            "`chapters mp3 https://youtu.be/dQw4w9WgXcQ`"
        ),
    )
    async def chapters(self, ctx: commands.Context, fmt: str | None = None,
                       *, url: str | None = None):
        # `chapters <url>` puts the URL in the format slot.
        if fmt is not None:
            key = fmt.strip().lower()
            if key in ("mp3", "audio", "mp4", "video"):
                fmt = "mp3" if key in ("mp3", "audio") else "mp4"
            elif unwrap_url(fmt).startswith(("http://", "https://")) and not url:
                url, fmt = unwrap_url(fmt), None
            else:
                return await ctx.send(
                    f"❌ `{fmt}` isn't a format. Use `mp3` or `mp4`, or just "
                    f"give the link: `{ctx.prefix}chapters <url>`")
        url = unwrap_url(url) if url else None
        if not url:
            return await ctx.send(
                f"❌ Give me a link. `{ctx.prefix}chapters https://youtu.be/…`")

        status = await ctx.send("🔍 Looking for chapters…")
        try:
            info = await audio.fetch_info(url)
        except Exception as e:
            log.warning("chapters: info fetch failed for %s", url)
            return await status.edit(content=f"❌ {_clean_ytdlp_err(e)}")
        if not info:
            return await status.edit(
                content="❌ Couldn't read anything about that link.")

        chapters = extract_chapters(info)
        title    = info.get("title") or "video"
        if not chapters:
            return await status.edit(
                content=(f"📄 **{title}** doesn't publish any chapters.\n"
                         f"-# `{ctx.prefix}snip <url> <start> <end>` cuts a "
                         "range by hand instead."))

        await self._show_chapters(ctx, status, url, title, chapters,
                                  forced=fmt, duration=float(info.get("duration") or 0))

    async def _show_chapters(
        self, ctx: commands.Context, status: discord.Message, url: str,
        title: str, chapters: list[dict], *, forced: str | None, duration: float,
    ) -> None:
        shown = chapters[:MAX_LISTED_CHAPTERS]
        lines = [
            f"`{c['index']:>2}.` **{c['title']}** — "
            f"{fmt_duration(int(c['start']))}–{fmt_duration(int(c['end']))} "
            f"*({fmt_duration(int(c['end'] - c['start']))})*"
            for c in shown
        ]
        extra = (f"\n…and {len(chapters) - len(shown)} more"
                 if len(chapters) > len(shown) else "")
        embed = discord.Embed(
            title=f"📑 {title[:240]}",
            description="\n".join(lines) + extra,
            color=discord.Color.blurple(),
        )
        embed.set_footer(
            text=f"{len(chapters)} chapters · "
                 f"{fmt_duration(int(duration))} total"
                 + ("  ·  audio (mp3)" if forced == "mp3" else "")
        )

        view = _ChapterView(self, ctx, url, title, chapters, forced=forced)
        await status.edit(content=None, embed=embed, view=view)
        view.message = status

    async def _download_chapter(
        self, ctx: commands.Context, url: str, chapter: dict, *,
        forced: str | None,
    ) -> None:
        """Pull one chapter out via the existing snip path."""
        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send(
            f"✂️ Grabbing **{chapter['title']}**…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_snip(
                ctx, status, url, chapter["start"], chapter["end"],
                forced=forced, workdir=workdir, canceller=canceller,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    async def _split_all_chapters(
        self, ctx: commands.Context, url: str, title: str, chapters: list[dict],
        *, forced: str | None,
    ) -> None:
        """Download once, then cut every chapter locally.

        Downloading the source per chapter would be N times the bandwidth and
        N times the chance of a rate limit — one fetch and N local cuts is the
        only sane way to do this.
        """
        todo = chapters[:MAX_SPLIT_CHAPTERS]
        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send(
            f"⬇️ Downloading **{title}** once, then splitting into "
            f"{len(todo)} files…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            result = await self._fetch_via_ytdlp(
                url, forced=forced, workdir=workdir, status=status,
                canceller=canceller,
            )
            if result is None:
                return
            filepath, _t, _dur, has_video, _h = result
            if canceller.cancelled:
                return await status.edit(content="✖ Cancelled.", view=None)

            _base, ext = os.path.splitext(filepath)
            done = 0
            for ch in todo:
                if canceller.cancelled:
                    await status.edit(content=f"✖ Stopped after {done} chapters.",
                                      view=None)
                    return
                out = os.path.join(
                    workdir, f"{ch['index']:02d} - {ch['title']}{ext}")
                await status.edit(
                    content=(f"✂️ Cutting **{ch['index']}/{len(todo)}** — "
                             f"{ch['title']}"))
                cmd = ["ffmpeg", "-y", "-nostats", "-loglevel", "error",
                       "-ss", f"{ch['start']:.3f}", "-to", f"{ch['end']:.3f}",
                       "-i", filepath, "-c", "copy"]
                if has_video:
                    cmd += ["-movflags", "+faststart"]
                cmd.append(out)
                try:
                    await _run_ffmpeg(cmd, canceller=canceller)
                except asyncio.CancelledError:
                    return await status.edit(content="✖ Cancelled.", view=None)
                except subprocess.CalledProcessError:
                    log.warning("chapter split failed for %r", ch["title"])
                    await ctx.send(f"⚠️ Skipped **{ch['title']}** — couldn't cut it.")
                    continue
                if not os.path.exists(out) or os.path.getsize(out) == 0:
                    await ctx.send(f"⚠️ Skipped **{ch['title']}** — came out empty.")
                    continue
                # Each chapter gets its own message so links stay attributable.
                part = await ctx.send(f"📄 **{ch['index']}. {ch['title']}**")
                await self._route_upload(
                    ctx, part, out,
                    title=f"{ch['index']:02d} - {ch['title']}{ext}",
                    duration=ch["end"] - ch["start"],
                    has_video=has_video, canceller=canceller,
                )
                done += 1

            skipped = len(chapters) - len(todo)
            tail = (f" *(first {MAX_SPLIT_CHAPTERS} of {len(chapters)} — "
                    f"{skipped} not split)*" if skipped > 0 else "")
            await status.edit(content=f"✅ Split **{title}** into {done} files.{tail}",
                              view=None)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    # ── -normalize / -desilence: audio cleanup ───────────────────────────────

    @commands.hybrid_command(
        name="normalize", aliases=["normalise", "loudnorm", "volumefix"],
        description="Even out the loudness of an audio/video file (EBU R128)",
        help=(
            "Brings a file to a consistent loudness — the fix for a recording "
            "that's too quiet, or a clip where one person is inaudible and the "
            "next is deafening.\n\n"
            "Uses two-pass EBU R128 loudness normalisation: the first pass "
            "measures, the second corrects. That's what makes it hit the "
            "target instead of drifting, and `linear` mode keeps the original "
            "dynamics rather than squashing them.\n\n"
            "**Targets** *(optional, defaults to `speech`)*\n"
            "`speech` — −16 LUFS, the podcast/voice-chat level\n"
            "`music` — −14 LUFS, what Spotify and YouTube normalise to\n"
            "`broadcast` — −23 LUFS, the EBU R128 spec\n\n"
            "Video files keep their video track untouched (stream-copied) — "
            "only the audio is rewritten.\n\n"
            "**Examples**\n"
            "`normalize` *(with a quiet voice memo attached)*\n"
            "`normalize music https://cdn.example.com/song.mp3`\n"
            "`normalize broadcast` *(with a video attached)*"
        ),
    )
    async def normalize(
        self, ctx: commands.Context, preset: str | None = None,
        attachment: discord.Attachment | None = None,
        *, url: str | None = None,
    ):
        preset, url, err = _split_preset_arg(
            preset, url, LOUDNESS_PRESETS, _LOUDNESS_ALIASES, "speech", "target")
        if err:
            return await ctx.send(err)
        await self._run_audio_job(ctx, "normalize", preset,
                                  attachment=attachment, url=url)

    @commands.hybrid_command(
        name="desilence", aliases=["trimsilence", "unsilence", "cutsilence"],
        description="Cut the silent gaps out of a recording",
        help=(
            "Removes silence from the start, the end, and every gap in "
            "between — the one-command version of topping and tailing a voice "
            "memo, or tightening up a recording with long pauses.\n\n"
            "**Strength** *(optional, defaults to `normal`)*\n"
            "`gentle` — only cuts pauses over 1s, leaves 0.5s of breathing room\n"
            "`normal` — cuts pauses over 0.5s, leaves 0.3s\n"
            "`aggressive` — cuts pauses over 0.25s, leaves 0.1s\n\n"
            "None of them cut silence to zero: a gap trimmed to nothing makes "
            "speech sound spliced. The bot reports how much time it removed.\n\n"
            "**Output is always audio.** Removing silence shortens the "
            "timeline, which would put any video track out of sync — so a "
            "video source comes back as an audio file.\n\n"
            "**Examples**\n"
            "`desilence` *(with a recording attached)*\n"
            "`desilence aggressive`\n"
            "`desilence gentle https://cdn.example.com/interview.wav`"
        ),
    )
    async def desilence(
        self, ctx: commands.Context, preset: str | None = None,
        attachment: discord.Attachment | None = None,
        *, url: str | None = None,
    ):
        preset, url, err = _split_preset_arg(
            preset, url, SILENCE_PRESETS, _SILENCE_ALIASES, "normal", "strength")
        if err:
            return await ctx.send(err)
        await self._run_audio_job(ctx, "desilence", preset,
                                  attachment=attachment, url=url)

    @commands.hybrid_command(
        name="speed", aliases=["tempo", "speedup"],
        description="Speed up or slow down an audio/video file",
        help=(
            "Changes playback speed **without changing pitch** — 1.5× a "
            "podcast and everyone still sounds like themselves. Works on "
            "audio and video; video is re-encoded in step so it stays in "
            "sync.\n\n"
            "Speeds run from `0.25` to `4`. Accepts `1.5`, `150%`, `1.5x`, "
            "or `double` / `half`.\n\n"
            "Pairs with `normalize` and `desilence` — those three turn a raw "
            "recording into something listenable.\n\n"
            "*(For the chipmunk/nightcore effect on music playback, that's "
            "`filter nightcore`.)*\n\n"
            "**Examples**\n"
            "`speed 1.5` *(with a recording attached)*\n"
            "`speed 0.75` — slow something down to follow it\n"
            "`speed 2x https://cdn.example.com/lecture.mp4`"
        ),
    )
    async def speed(
        self, ctx: commands.Context, factor: str | None = None,
        attachment: discord.Attachment | None = None,
        *, url: str | None = None,
    ):
        # `speed <url>` puts the URL in the factor slot.
        if factor and parse_speed(factor) is None and \
                unwrap_url(factor).startswith(("http://", "https://")) and not url:
            url, factor = unwrap_url(factor), None
        if not factor:
            return await ctx.send(
                f"❌ How fast? `{ctx.prefix}speed 1.5`, `{ctx.prefix}speed 0.75`, "
                f"`{ctx.prefix}speed 2x`.")
        value = parse_speed(factor)
        if value is None:
            return await ctx.send(
                f"❌ `{factor}` isn't a speed I can use. Give me a number "
                f"between **{SPEED_MIN:g}** and **{SPEED_MAX:g}** — `1.5`, "
                "`150%`, `2x`, `half`.")
        if abs(value - 1.0) < 1e-6:
            return await ctx.send("❌ That's just… the same speed.")
        await self._run_audio_job(ctx, "speed", f"{value:.6f}",
                                  attachment=attachment, url=url)

    async def _run_audio_job(
        self, ctx: commands.Context, job: str, option: str, *,
        attachment: discord.Attachment | None, url: str | None,
    ) -> None:
        """Shared shell for -normalize, -desilence and -speed."""
        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]
        if attachment is None and not url:
            return await ctx.send(
                f"❌ Attach a file or provide a URL to `{job}`.")

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send("⬇️ Fetching source…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_audio_job(
                ctx, status, job, option, attachment=attachment, url=url,
                workdir=workdir, canceller=canceller,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    async def _do_audio_job(
        self, ctx: commands.Context, status: discord.Message,
        job: str, option: str, *,
        attachment: discord.Attachment | None, url: str | None,
        workdir: str, canceller: JobCanceller,
    ) -> None:
        got = await self._materialise_source(
            attachment=attachment, url=url, workdir=workdir,
            status=status, canceller=canceller, label=job,
        )
        if got is None:
            return
        src, src_name = got
        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        stem, src_ext = os.path.splitext(src_name)
        probe = await _probe_media(src)
        # -speed is the one that works on a silent video; the other two are
        # about the audio by definition.
        if not probe["audio"] and job != "speed":
            return await status.edit(
                content=f"❌ **{src_name}** has no audio track.", view=None)
        if job == "speed" and not (probe["audio"] or probe["video"]):
            return await status.edit(
                content=f"❌ **{src_name}** has nothing to speed up.", view=None)

        duration = probe["duration"]
        audio_args, out_ext = _audio_output_args(src_ext)
        # -normalize and -speed keep video; -desilence can't, since removing
        # silence shortens the timeline and desyncs the picture.
        keep_video = job in ("normalize", "speed") and probe["video"]
        if keep_video:
            out_ext = "mp4" if src_ext.lower().lstrip(".") != "mkv" else "mkv"
            audio_args = ["-c:a", "aac", "-b:a", "192k"]
        out = os.path.join(workdir, f"{stem}_{job}.{out_ext}")

        updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

        async def _updater_done():
            await updater.aclose()

        if job == "normalize":
            target = LOUDNESS_PRESETS[option]
            await status.edit(
                content=f"📊 Measuring loudness of **{src_name}**…\n"
                        f"-# pass 1 of 2 · target {target['I']:g} LUFS ({option})")
            measured = await _measure_loudness(src, target)
            if canceller.cancelled:
                await _updater_done()
                return await status.edit(content="✖ Cancelled.", view=None)

            head = (f"🔊 Normalising **{src_name}** → {target['I']:g} LUFS "
                    f"*({option})*")
            if measured is None:
                head += "\n-# analysis pass didn't report — using a single pass"
            await status.edit(content=head)
            cmd = _build_normalize_cmd(src, out, target, measured=measured,
                                       keep_video=keep_video, audio_args=audio_args)
        elif job == "speed":
            factor = float(option)
            head = (f"⏩ {'Speeding up' if factor > 1 else 'Slowing down'} "
                    f"**{src_name}** — **{factor:g}×**")
            if keep_video:
                head += "\n-# video is re-encoded to stay in sync"
            await status.edit(content=head)
            cmd = _build_speed_cmd(src, out, factor, has_video=keep_video,
                                   has_audio=probe["audio"],
                                   audio_args=audio_args)
            # The progress bar reads output time, which is now compressed.
            duration = duration / factor if duration else 0
        else:
            preset = SILENCE_PRESETS[option]
            head = f"✂️ Trimming silence from **{src_name}** *({option})*"
            await status.edit(content=head)
            cmd = _build_desilence_cmd(src, out, preset, audio_args)

        def on_progress(pct: float):
            updater.update(f"{head}\n`[{_bar(pct)}]` {pct:.0f}%")

        try:
            await _run_ffmpeg(cmd, duration=duration, canceller=canceller,
                              on_progress=on_progress)
        except asyncio.CancelledError:
            await _updater_done()
            return await status.edit(content="✖ Cancelled.", view=None)
        except subprocess.CalledProcessError as e:
            err  = (e.stderr or b"").decode(errors="replace").strip()
            log.warning("%s failed for %s: %s", job, src_name, err)
            line = err.splitlines()[-1][:300] if err else "unknown ffmpeg error"
            await _updater_done()
            return await status.edit(content=f"❌ `{job}` failed: `{line}`",
                                     view=None)
        await _updater_done()

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            return await status.edit(
                content=f"❌ `{job}` produced no output.", view=None)

        out_dur = await audio.get_ffprobe_duration(out) or 0
        if job == "desilence" and duration > 0 and out_dur > 0:
            cut = max(0.0, duration - out_dur)
            pct = cut / duration * 100
            if cut < 0.5:
                note = ("nothing to cut — no silence matched at this "
                        f"strength. Try `{ctx.prefix}desilence aggressive`.")
            else:
                note = (f"removed **{fmt_duration(int(cut))}** of silence "
                        f"({pct:.0f}% shorter)")
            await ctx.send(f"✂️ {note}")
        elif job == "speed" and probe["duration"] > 0 and out_dur > 0:
            saved = probe["duration"] - out_dur
            verb  = "saves" if saved > 0 else "adds"
            await ctx.send(
                f"⏩ {fmt_duration(int(probe['duration']))} → "
                f"**{fmt_duration(int(out_dur))}** — {verb} "
                f"**{fmt_duration(int(abs(saved)))}**.")

        await self._route_upload(
            ctx, status, out,
            title=f"{stem}_{job}.{out_ext}",
            duration=out_dur or duration,
            has_video=keep_video,
            canceller=canceller,
        )

    # ── -transcribe: speech-to-text ──────────────────────────────────────────

    async def _ensure_whisper(self):
        """Load (once) and return the faster-whisper model for -transcribe."""
        if self._tmodel is None:
            self._tmodel = await asyncio.to_thread(load_model, self._tmodel_name)
            log.info("Loaded transcribe whisper model %r", self._tmodel_name)
        return self._tmodel

    async def _switch_whisper(self, name: str) -> None:
        model = await asyncio.to_thread(load_model, name)
        self._tmodel = model
        self._tmodel_name = name
        log.info("Switched transcribe whisper model to %r", name)

    @commands.hybrid_command(
        name="transcribemodel", aliases=["tmodel"],
        description="Show or change the model -transcribe uses (Manage Server)",
        help=(
            "The whisper model used by `-transcribe`. Bigger = more accurate but "
            "slower; it's host-wide (one shared model) and the choice persists "
            "across restarts. Separate from the voice-command model (`voice "
            "model`).\n\n"
            "Run with no argument to see the current model and options. "
            "English-only `.en` variants are the most accurate when the audio is "
            "English.\n\n"
            "**Examples**\n"
            "`transcribemodel` — show the current model and options\n"
            "`transcribemodel small.en` — switch to small.en"
        ),
    )
    @commands.has_guild_permissions(manage_guild=True)
    async def transcribemodel(self, ctx: commands.Context, name: str | None = None):
        opts = ", ".join(f"`{m}`" for m in KNOWN_MODELS)
        if name is None:
            return await ctx.send(
                f"🧠 `-transcribe` model: **{self._tmodel_name}**.\n"
                f"Change with `{ctx.prefix}transcribemodel <name>` *(host-wide, "
                f"persists)*. Separate from `{ctx.prefix}voice model`.\n"
                f"**Options:** {opts}"
            )
        name = name.strip()
        if name not in KNOWN_MODELS:
            return await ctx.send(f"❌ Unknown model `{name}`.\n**Options:** {opts}")
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return await ctx.send(
                "❌ faster-whisper isn't installed on the host. Run "
                "`pip install faster-whisper` and restart."
            )
        if name == self._tmodel_name and self._tmodel is not None:
            return await ctx.send(f"🧠 Already using **{name}**.")
        status = await ctx.send(
            f"⏳ Loading **{name}**… the first use downloads it, which can take a "
            f"minute. The current model keeps working until it's ready."
        )
        try:
            await self._switch_whisper(name)
        except Exception as e:
            log.exception("transcribe model switch failed")
            return await status.edit(content=f"❌ Couldn't load `{name}`: {e}")
        await db.set_app_setting("transcribe_whisper_model", name)
        await status.edit(
            content=f"✅ `-transcribe` model is now **{name}**. (Saved — survives restarts.)"
        )

    @commands.hybrid_command(
        name="transcribe",
        description="Transcribe speech from an audio/video file or URL",
        help=(
            "Speech-to-text for an attached audio/video file or a URL "
            "(YouTube links work). Posts the transcript with timestamps — "
            "inline if it's short, as a `.txt` attachment if it's long.\n\n"
            "Runs locally via [faster-whisper](https://github.com/SYSTRAN/faster-whisper); "
            "the bot owner enables it with `pip install faster-whisper`. The "
            "model is set with `WHISPER_MODEL` in `.env` (default `base`; "
            "`small`/`medium` are more accurate but slower). The first run "
            "downloads the model, so it may take a minute.\n\n"
            "Sources longer than 2 hours are refused.\n\n"
            "**Examples**\n"
            "`transcribe` *(with a voice memo attached)*\n"
            "`transcribe https://youtu.be/dQw4w9WgXcQ`"
        ),
    )
    async def transcribe(
        self, ctx: commands.Context,
        attachment: discord.Attachment | None = None,
        *, url: str | None = None,
    ):
        await self._run_transcribe_job(
            ctx, attachment=attachment, url=unwrap_url(url) if url else url)

    async def _run_transcribe_job(
        self, ctx: commands.Context, *,
        attachment: discord.Attachment | None, url: str | None,
    ) -> None:
        """Shared by `-transcribe` and the Transcribe context menu."""
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return await ctx.send(
                "❌ Transcription isn't enabled on this bot. Owner: run "
                "`pip install faster-whisper` on the host and restart."
            )
        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]
        if attachment is None and not url:
            return await ctx.send("❌ Attach an audio/video file or provide a URL.")

        canceller = JobCanceller()
        view      = JobView(ctx.author, canceller)
        status    = await ctx.send("⬇️ Fetching source…", view=view)
        workdir   = tempfile.mkdtemp(prefix="musicbot-", dir=settings.download_tmp)
        try:
            await self._do_transcribe(
                ctx, status, attachment=attachment, url=url,
                workdir=workdir, canceller=canceller,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            view.stop()

    async def _do_transcribe(
        self, ctx: commands.Context, status: discord.Message, *,
        attachment: discord.Attachment | None, url: str | None,
        workdir: str, canceller: JobCanceller,
    ) -> None:
        # Materialise the source — same three-way split as -convert.
        if attachment is not None:
            src_name = os.path.basename(attachment.filename or "") or "file"
            src = os.path.join(workdir, src_name)
            try:
                await attachment.save(src)
            except discord.HTTPException as e:
                return await status.edit(
                    content=f"❌ Couldn't fetch the attachment: {e}", view=None,
                )
        else:
            try:
                src = await self._download_direct(
                    url, workdir=workdir, status=status, canceller=canceller,
                )
            except ValueError as e:
                return await status.edit(content=f"❌ {e}", view=None)
            except Exception as e:
                log.exception("transcribe: direct download failed for %s", url)
                return await status.edit(content=f"❌ Download failed: {e}", view=None)
            if src is None:
                # Web page → yt-dlp. Audio-only: whisper doesn't need video.
                result = await self._fetch_via_ytdlp(
                    url, forced="mp3", workdir=workdir,
                    status=status, canceller=canceller,
                )
                if result is None:
                    return
                src = result[0]
            src_name = os.path.basename(src)

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        duration = await audio.get_ffprobe_duration(src)
        if duration > 2 * 3600:
            return await status.edit(
                content=f"❌ Source is {fmt_duration(int(duration))} long — "
                        "the limit for transcription is 2 hours.",
                view=None,
            )

        await status.edit(content="🧠 Loading speech model… "
                                  "(first run downloads it — may take a minute)")
        try:
            model = await self._ensure_whisper()
        except Exception as e:
            log.exception("whisper model load failed")
            return await status.edit(content=f"❌ Couldn't load the speech model: {e}",
                                     view=None)

        await status.edit(content=f"📝 Transcribing **{src_name}**…")
        # 4s (not 2s) between progress edits: fewer edits to the same message
        # makes wedging its per-message edit rate-limit far less likely.
        updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=4.0)

        def _work():
            segments, info = model.transcribe(src, vad_filter=True)
            lines: list[tuple[float, str]] = []
            for seg in segments:   # generator — transcription happens here
                if canceller.cancelled:
                    raise asyncio.CancelledError()
                text = seg.text.strip()
                if text:
                    lines.append((seg.start, text))
                if duration > 0:
                    pct = min(100.0, seg.end / duration * 100)
                    updater.update(
                        f"📝 Transcribing **{src_name}**…\n"
                        f"`[{_bar(pct)}]` {pct:.0f}%"
                    )
            return lines, info

        error = None
        try:
            lines, info = await asyncio.to_thread(_work)
        except asyncio.CancelledError:
            error = "✖ Cancelled."
        except Exception as e:
            log.exception("transcription failed")
            error = f"❌ Transcription failed: {e}"

        # Stop progress edits and let the last in-flight one land BEFORE we
        # write the final message. Otherwise a queued progress edit (e.g. the
        # 100% one) can run *after* the result edit and overwrite it — that race
        # is the intermittent "stuck at 100%" (the thread dump showed the bot
        # idle, so nothing was hung; the transcript was just being clobbered).
        await updater.aclose()

        if error is not None:
            return await self._settle_status(ctx, status, error)
        if not lines:
            return await self._settle_status(ctx, status, "❌ No speech detected.")

        lang = ""
        if getattr(info, "language", None):
            lang = f" · language: {info.language}"
        header = f"📝 **Transcript of {src_name}**{lang}"
        body   = "\n".join(f"`[{fmt_duration(int(t))}]` {text}" for t, text in lines)

        if len(header) + len(body) <= 1900:
            return await self._settle_status(ctx, status, f"{header}\n{body}")

        # Too long for a message — attach as a text file instead.
        txt_path = os.path.join(workdir, "transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"Transcript of {src_name}{lang}\n\n")
            for t, text in lines:
                fh.write(f"[{fmt_duration(int(t))}] {text}\n")
        stem = os.path.splitext(src_name)[0]
        try:
            await ctx.send(
                content=f"{header} — {len(lines)} segments",
                file=discord.File(txt_path, filename=f"{stem}.txt"),
            )
        except discord.HTTPException as e:
            return await self._settle_status(ctx, status, f"❌ Upload failed: {e}")
        # Result already delivered as a fresh message; drop the stale progress
        # message best-effort (timeout-guarded so a wedged edit bucket can't
        # hang us here either).
        try:
            await asyncio.wait_for(status.delete(), timeout=10)
        except Exception:
            pass

    async def _settle_status(self, ctx: commands.Context, status: discord.Message,
                             content: str) -> None:
        """Replace the progress message with the final result — hang-proof.

        Editing a status message many times during a slow transcription can
        wedge its per-message edit rate-limit bucket, making a plain
        `status.edit` hang forever — that's the transcribe-while-talking freeze
        (talking slows the job, so more progress edits pile up against the limit).
        We cap the edit with a timeout and, if it doesn't land, deliver via a
        fresh `ctx.send` — a *different* bucket the wedged one can't block — then
        drop the stale progress message. Cancelling the timed-out edit also
        releases discord.py's bucket lock, so this self-heals rather than masks.
        """
        try:
            await asyncio.wait_for(
                status.edit(content=content, view=None), timeout=10,
            )
            return
        except Exception:
            log.debug("transcribe: status edit didn't land — sending fresh",
                      exc_info=True)
        try:
            await ctx.send(content)
            await asyncio.wait_for(status.delete(), timeout=10)
        except Exception:
            log.debug("transcribe: fresh-send fallback issue", exc_info=True)

    # ── Shared yt-dlp fetcher ────────────────────────────────────────────────

    async def _fetch_via_ytdlp(
        self, query: str, *, forced, workdir: str, status: discord.Message,
        canceller: JobCanceller, clip_range: tuple[float, float] | None = None,
        raw_media: bool = False,
    ):
        """Probe + download. Returns (filepath, title, duration, has_video, height)
        or None on failure (status message is updated with an error).

        `clip_range=(start, end)` asks yt-dlp to download only that section
        (much faster for clipping a long video). Extractors that can't do
        partial downloads fall back to the full file — the caller is expected
        to verify the result's duration and trim locally if needed.

        `raw_media=True` downloads the source as-is: no mp3 extraction, no
        mp4 merge assumptions. Used by -convert, where guessing breaks on
        sources like gif pages (no audio stream → the FFmpegExtractAudio
        postprocessor dies with 'unable to obtain file audio codec')."""

        if canceller.cancelled:
            return None

        preset_title    = None
        preset_duration = 0

        if audio.is_spotify_url(query):
            if not audio.spotify_ready():
                await status.edit(content="❌ Spotify isn't configured.")
                return None
            if forced == "mp4":
                await status.edit(content="❌ Spotify is audio-only — `mp4` not supported.")
                return None
            try:
                tracks = await audio.fetch_spotify_tracks(query)
            except Exception as e:
                log.exception("spotify fetch failed")
                await status.edit(content=f"❌ {e}")
                return None
            if not tracks or len(tracks) > 1:
                await status.edit(content="❌ Provide a single Spotify track, not a playlist.")
                return None
            sp_track        = tracks[0]
            query           = sp_track["_search_query"]
            forced          = "mp3"
            preset_title    = sp_track["title"]
            preset_duration = sp_track["duration"]

        def _probe():
            # Built from the shared base so this can't drift away from the
            # playback path again — see the note in core/audio.py.
            opts = audio.ydl_opts(
                skip_download=True,
                # Same slow-host concern as the download path below — without
                # this, a direct-hosted URL like litterbox times out at 20s.
                socket_timeout=60,
                retries=2,
            )
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
                if "entries" in info and info["entries"]:
                    info = info["entries"][0]
            return info

        try:
            info = await asyncio.to_thread(_probe)
        except Exception as e:
            log.exception("probe failed")
            await status.edit(content=f"❌ Couldn't fetch info: {_clean_ytdlp_err(e)}")
            return None

        if canceller.cancelled:
            return None

        title     = preset_title    or info.get("title", "audio")
        duration  = preset_duration or info.get("duration", 0) or 0
        height    = info.get("height") or 0
        probe_has_video = bool(info.get("vcodec") and info.get("vcodec") != "none")

        # forced=None means "use what the source has". An explicit mp3/mp4
        # always wins over the probe result.
        if   forced == "mp3": has_video = False
        elif forced == "mp4": has_video = True
        else:                 has_video = probe_has_video

        fmt_type = "mp4" if has_video else "mp3"
        fmt_note = "" if raw_media else f" as `{fmt_type}`"
        await status.edit(content=f"⬇️ Downloading **{title}**{fmt_note}…")

        # Two-second throttled bar — same look as the re-encode bar below.
        dl_updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

        def hook(d):
            if canceller.cancelled:
                # Raising here actually cancels yt-dlp's download — it catches
                # everything as DownloadError below.
                raise yt_dlp.utils.DownloadError("cancelled by user")
            if d.get("status") != "downloading":
                return
            try:
                pct = float(d.get("_percent_str", "0%").strip().replace("%", ""))
            except ValueError:
                return
            speed = d.get("_speed_str", "").strip()
            eta   = d.get("_eta_str", "").strip()
            extras = (f"  {speed}" if speed else "") + (f"  ETA {eta}" if eta else "")
            line  = (
                f"⬇️ Downloading **{title}**{fmt_note}…\n"
                f"`[{_bar(pct)}]` {pct:.0f}%{extras}"
            )
            dl_updater.update(line)

        if raw_media:
            # Take whatever the source has, no postprocessing. mkv as the
            # merge container because it accepts any codec combination —
            # the converter re-containers afterward anyway.
            opts = audio.ydl_opts(
                format="bestvideo+bestaudio/best",
                outtmpl=f"{workdir}/%(title)s.%(ext)s",
                merge_output_format="mkv",
                progress_hooks=[hook],
                socket_timeout=120,
            )
        elif has_video:
            opts = audio.ydl_opts(
                format="bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                outtmpl=f"{workdir}/%(title)s.%(ext)s",
                merge_output_format="mp4",
                progress_hooks=[hook],
                # Direct-hosted servers (catbox, litterbox, random CDNs) can be
                # slow to start a transfer. Default is 20s which trips on
                # bigger files. Bumped + retried.
                socket_timeout=120,
            )
        else:
            opts = audio.ydl_opts(
                format="bestaudio/best",
                outtmpl=f"{workdir}/%(title)s.%(ext)s",
                progress_hooks=[hook],
                socket_timeout=120,
                postprocessors=[{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3", "preferredquality": "192",
                }],
            )

        if clip_range is not None:
            from yt_dlp.utils import download_range_func
            opts["download_ranges"]         = download_range_func(None, [clip_range])
            opts["force_keyframes_at_cuts"] = True   # re-encode at cuts = exact times

        def _download():
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=True)
                if "entries" in info and info["entries"]:
                    info = info["entries"][0]
                # yt-dlp records the real post-merge/post-processed path in
                # requested_downloads. Guessing `base.mp4` breaks when the
                # source only offers a single non-mp4 format (e.g. webm-only),
                # because merge_output_format never kicks in for those.
                for rd in info.get("requested_downloads") or []:
                    path = rd.get("filepath")
                    if path and os.path.exists(path):
                        return path
                if raw_media:
                    return ydl.prepare_filename(info)
                base = ydl.prepare_filename(info).rsplit(".", 1)[0]
                return base + (".mp4" if has_video else ".mp3")

        try:
            filepath = await asyncio.to_thread(_download)
        except Exception as e:
            # Some extractors can't do partial downloads — drop the range and
            # grab the whole file; the clip caller trims locally afterward.
            if (clip_range is not None and not canceller.cancelled
                    and opts.pop("download_ranges", None) is not None):
                opts.pop("force_keyframes_at_cuts", None)
                log.info("Ranged download failed (%s) — retrying full download",
                         _clean_ytdlp_err(e))
                try:
                    filepath = await asyncio.to_thread(_download)
                except Exception as e2:
                    e, filepath = e2, None
            else:
                filepath = None
            if filepath is None:
                await dl_updater.aclose()   # let the last progress edit land first
                if canceller.cancelled:
                    await status.edit(content="✖ Cancelled.", view=None)
                    return None
                log.exception("download failed")
                await status.edit(content=f"❌ Download failed: {_clean_ytdlp_err(e)}")
                return None

        await dl_updater.aclose()           # ditto for the success path
        if canceller.cancelled:
            await status.edit(content="✖ Cancelled.", view=None)
            return None
        if not os.path.exists(filepath):
            await status.edit(content="❌ File missing after download.")
            return None

        # Direct-hosted files (catbox, litterbox, arbitrary CDNs, plain mp4
        # URLs) don't carry metadata that yt-dlp can extract. The duration
        # field comes back as 0 or None. ffprobe on the downloaded file
        # always works, since by this point the bytes are on disk.
        if not duration:
            try:
                duration = await audio.get_ffprobe_duration(filepath)
            except Exception:
                log.exception("ffprobe fallback failed for %s", filepath)
                duration = 0

        return filepath, title, duration, has_video, height

    # ── -dl handler ──────────────────────────────────────────────────────────

    async def _do_download(
        self, ctx: commands.Context, status: discord.Message, query: str, *,
        forced, resize_target: int | None,
        workdir: str, canceller: JobCanceller, view: JobView,
    ) -> None:
        result = await self._fetch_via_ytdlp(
            query, forced=forced, workdir=workdir,
            status=status, canceller=canceller,
        )
        if result is None:
            return
        filepath, title, duration, has_video, _height = result

        actual = os.path.getsize(filepath)

        # Explicit --resize target wins over automatic litterbox shrinking.
        if resize_target is not None:
            if duration <= 0:
                return await status.edit(
                    content="❌ Couldn't determine duration — can't compute resize bitrate.",
                    view=None,
                )
            await status.edit(
                content=(
                    f"⚙️ Got **{title}** ({fmt_size(actual)}) — "
                    f"re-encoding to ~{fmt_size(resize_target)}…"
                )
            )
            updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

            def on_progress(pct: float):
                updater.update(
                    f"⚙️ Re-encoding **{title}** → {fmt_size(resize_target)}\n"
                    f"`[{_bar(pct)}]` {pct:.0f}%"
                )

            try:
                filepath = await _reencode_to_target(
                    filepath, target_bytes=resize_target,
                    has_video=has_video, duration=duration,
                    canceller=canceller, on_progress=on_progress,
                )
            except asyncio.CancelledError:
                await updater.aclose()
                return await status.edit(content="✖ Cancelled.", view=None)
            except subprocess.CalledProcessError:
                log.exception("ffmpeg failed during --resize")
                await updater.aclose()
                return await status.edit(content="❌ Re-encode failed.", view=None)
            except ValueError as e:
                await updater.aclose()
                return await status.edit(content=f"❌ {e}", view=None)
            await updater.aclose()

        elif actual > RESIZE_CAP:
            return await status.edit(
                content=(
                    f"❌ Source is **{fmt_size(actual)}** — too big to re-encode "
                    "down to a hostable size without significant quality loss. "
                    "Try a lower-quality source or use `-resize` with an explicit target."
                ),
                view=None,
            )

        elif actual > LITTERBOX_MAX:
            await status.edit(
                content=(
                    f"⚙️ File is **{fmt_size(actual)}** — re-encoding to fit "
                    "litterbox (~950 MB)…"
                )
            )
            updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

            def on_progress(pct: float):
                updater.update(
                    f"⚙️ Re-encoding **{title}** to fit litterbox\n"
                    f"`[{_bar(pct)}]` {pct:.0f}%"
                )

            try:
                filepath = await _reencode_to_target(
                    filepath, target_bytes=LITTERBOX_TARGET,
                    has_video=has_video, duration=duration,
                    canceller=canceller, on_progress=on_progress,
                )
            except asyncio.CancelledError:
                await updater.aclose()
                return await status.edit(content="✖ Cancelled.", view=None)
            except subprocess.CalledProcessError:
                log.exception("ffmpeg failed during -dl shrink")
                await updater.aclose()
                return await status.edit(content="❌ Re-encode failed.", view=None)
            except ValueError as e:
                await updater.aclose()
                return await status.edit(content=f"❌ {e}", view=None)
            await updater.aclose()

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        await self._route_upload(
            ctx, status, filepath,
            title=title, duration=duration, has_video=has_video,
            canceller=canceller,
        )

    # ── -resize handler ──────────────────────────────────────────────────────

    async def _do_resize(
        self, ctx: commands.Context, status: discord.Message, link: str,
        target_bytes: int, *, forced: str, workdir: str,
        canceller: JobCanceller, view: JobView,
    ) -> None:
        result = await self._fetch_via_ytdlp(
            link, forced=forced, workdir=workdir,
            status=status, canceller=canceller,
        )
        if result is None:
            return
        filepath, title, duration, has_video, _height = result

        if duration <= 0:
            return await status.edit(
                content="❌ Couldn't determine duration — can't compute target bitrate.",
                view=None,
            )

        actual = os.path.getsize(filepath)
        await status.edit(
            content=(
                f"⚙️ Got **{title}** ({fmt_size(actual)}) — "
                f"re-encoding to ~{fmt_size(target_bytes)}…"
            )
        )

        updater = _ThrottledStatusUpdater(status, self.bot.loop, interval=2.0)

        def on_progress(pct: float):
            updater.update(
                f"⚙️ Re-encoding **{title}** → {fmt_size(target_bytes)}\n"
                f"`[{_bar(pct)}]` {pct:.0f}%"
            )

        try:
            shrunk = await _reencode_to_target(
                filepath, target_bytes=target_bytes,
                has_video=has_video, duration=duration,
                canceller=canceller, on_progress=on_progress,
            )
        except asyncio.CancelledError:
            await updater.aclose()
            return await status.edit(content="✖ Cancelled.", view=None)
        except subprocess.CalledProcessError:
            log.exception("ffmpeg failed during -resize")
            await updater.aclose()
            return await status.edit(content="❌ Re-encode failed.", view=None)
        except ValueError as e:
            await updater.aclose()
            return await status.edit(content=f"❌ {e}", view=None)
        await updater.aclose()

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        await self._route_upload(
            ctx, status, shrunk,
            title=f"{title} (resized)",
            duration=duration, has_video=has_video,
            canceller=canceller,
        )

    # ── Tiered upload ────────────────────────────────────────────────────────

    async def _route_upload(
        self, ctx: commands.Context, status: discord.Message, filepath: str, *,
        title: str, duration: float, has_video: bool,
        canceller: JobCanceller, emoji: str | None = None,
    ) -> None:
        actual = os.path.getsize(filepath)
        if emoji is None:
            emoji = "🎬" if has_video else "🎵"
        # No duration chip for stills (convert png→jpg etc.) — "[??:??]" on
        # an image looks broken.
        dur_part = f" [{fmt_duration(int(duration))}]" if duration > 0 else ""
        head   = f"{emoji} **{title}**{dur_part} `{fmt_size(actual)}`"
        asker  = f"requested by *{ctx.author.display_name}*"

        if canceller.cancelled:
            return await status.edit(content="✖ Cancelled.", view=None)

        if actual <= DISCORD_LIMIT:
            try:
                await ctx.send(
                    content=f"{head} — {asker}",
                    file=discord.File(filepath, filename=os.path.basename(filepath)),
                )
                await status.delete()
            except discord.HTTPException as e:
                log.warning("Discord upload failed: %s", e)
                await status.edit(content=f"❌ Discord upload failed: {e}", view=None)
            return

        if actual <= CATBOX_MAX:
            await status.edit(content=f"☁️ Uploading **{fmt_size(actual)}** to catbox.moe…")
            try:
                link = await _upload_to_catbox(filepath)
            except Exception as e:
                log.exception("catbox upload failed")
                return await status.edit(content=f"❌ Catbox upload failed: {e}", view=None)
            await ctx.send(
                f"{head} — {asker}\n☁️ {link}\n"
                "ℹ️ Hosted on catbox.moe — the link is permanent."
            )
            await status.delete()
            return

        if actual <= LITTERBOX_MAX:
            await status.edit(content=f"⏳ Uploading **{fmt_size(actual)}** to litterbox…")
            try:
                link = await _upload_to_litterbox(filepath, expiry=LITTERBOX_EXPIRY)
            except Exception as e:
                log.exception("litterbox upload failed")
                return await status.edit(content=f"❌ Litterbox upload failed: {e}", view=None)
            await ctx.send(
                f"{head} — {asker}\n⏳ {link}\n"
                f"⚠️ Link expires in {LITTERBOX_EXPIRY} — save it now if you want to keep it."
            )
            await status.delete()
            return

        # Shouldn't reach here after the -dl re-encode or -resize target check.
        await status.edit(
            content=(
                f"❌ File ended up at **{fmt_size(actual)}**, bigger than the "
                "1 GB litterbox ceiling. The re-encode missed its target."
            ),
            view=None,
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(DownloadsCog(bot))
