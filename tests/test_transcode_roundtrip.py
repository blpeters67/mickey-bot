"""End-to-end -transcode checks that actually run ffmpeg.

Skipped automatically when ffmpeg/ffprobe aren't on PATH, or when the build
lacks the encoder a given case needs. These are slower than the rest of the
suite (a few seconds each) because they encode real frames — that's the
point: they catch flag combinations that only fail once ffmpeg parses them.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess

import pytest

from cogs import downloads as d

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed",
)

# Deliberately odd dimensions: the even-dimension scale filter has to fix
# these or yuv420p encoders reject the frames. libaom accepts odd input;
# libx264 does not, which is itself the reason that filter exists — so h264
# sources are generated at an even size instead.
_W, _H, _SECS = 321, 241, 2
_EVEN = (320, 240)


def _have(encoder: str) -> bool:
    return encoder in d._available_encoders()


def _make_source(path: str, *, vcodec: str, acodec: str, pix_fmt: str) -> None:
    """Synthesise a short test clip with a known codec/pixel format."""
    venc = {"av1": "libaom-av1", "h264": "libx264"}[vcodec]
    vargs = ["-c:v", venc, "-pix_fmt", pix_fmt]
    if venc == "libaom-av1":
        w, h = _W, _H
        vargs += ["-cpu-used", "8", "-crf", "50", "-b:v", "0"]
    else:
        w, h = _EVEN
        vargs += ["-preset", "ultrafast", "-crf", "35"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=15:duration={_SECS}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={_SECS}",
         *vargs, "-c:a", acodec, "-shortest", path],
        check=True, capture_output=True, timeout=300,
    )


def _run_chain(cmds) -> str:
    """Walk the candidate list the way _do_transcode does. Returns the encoder
    that succeeded, or raises the last error."""
    last = None
    for i, (enc, cmd) in enumerate(cmds):
        try:
            asyncio.run(d._run_ffmpeg(cmd, duration=float(_SECS)))
            return enc
        except subprocess.CalledProcessError as e:
            last = e
            out = cmd[-1]
            if os.path.exists(out):
                os.remove(out)
            if i + 1 == len(cmds):
                raise
    raise last


def _probe(path: str) -> dict:
    return asyncio.run(d._probe_media(path))


# ── the headline case: AV1 in, widely-supported codec out ────────────────────

@pytest.mark.parametrize("codec", ["h264", "h265", "vp9"])
def test_av1_source_transcodes_out(tmp_path, codec):
    if not _have(d._SW_ENCODERS[codec][0]):
        pytest.skip(f"{d._SW_ENCODERS[codec][0]} not in this ffmpeg build")
    if not _have("libaom-av1"):
        pytest.skip("no AV1 encoder to build the source with")

    src = str(tmp_path / "src.mkv")
    _make_source(src, vcodec="av1", acodec="libopus", pix_fmt="yuv420p")
    probe = _probe(src)
    assert d.normalise_codec(probe["vcodec"]) == "av1"

    container = d.pick_container(codec, "mkv")
    dst = str(tmp_path / f"out.{container}")
    used = _run_chain(d._build_transcode_cmds(
        src, dst, codec, quality="small", container=container, probe=probe))

    out = _probe(dst)
    assert d.normalise_codec(out["vcodec"]) == codec, f"via {used}"
    assert out["audio"], "audio track was dropped"
    # Odd source dimensions must come out even.
    assert out["width"] % 2 == 0 and out["height"] % 2 == 0
    assert out["width"] == _W - 1 and out["height"] == _H - 1
    assert abs(out["duration"] - _SECS) < 0.5


def test_h264_source_transcodes_to_av1(tmp_path):
    """The reverse direction — shrink a widely-supported file down to AV1."""
    if not any(_have(e) for e in d._SW_ENCODERS["av1"]):
        pytest.skip("no AV1 encoder in this ffmpeg build")

    src = str(tmp_path / "src.mp4")
    _make_source(src, vcodec="h264", acodec="aac", pix_fmt="yuv420p")
    probe = _probe(src)
    assert d.normalise_codec(probe["vcodec"]) == "h264"

    container = d.pick_container("av1", "mp4")
    dst = str(tmp_path / f"out.{container}")
    _run_chain(d._build_transcode_cmds(
        src, dst, "av1", quality="small", container=container, probe=probe))

    out = _probe(dst)
    assert d.normalise_codec(out["vcodec"]) == "av1"
    assert out["audio"]


# ── the 10-bit trap ──────────────────────────────────────────────────────────

def test_10bit_av1_becomes_8bit_h264(tmp_path):
    """AV1 rips are routinely 10-bit; H.264 High 10 won't play on the devices
    someone is asking for H.264 to reach."""
    if not (_have("libaom-av1") and _have("libx264")):
        pytest.skip("needs libaom-av1 and libx264")

    src = str(tmp_path / "src10.mkv")
    _make_source(src, vcodec="av1", acodec="libopus", pix_fmt="yuv420p10le")
    probe = _probe(src)
    assert "10" in probe["pix_fmt"], "source wasn't actually 10-bit"

    dst = str(tmp_path / "out.mp4")
    _run_chain(d._build_transcode_cmds(
        src, dst, "h264", quality="small", container="mp4", probe=probe))

    out = _probe(dst)
    assert out["pix_fmt"] == "yuv420p", f"got {out['pix_fmt']}"


# ── audio handling ───────────────────────────────────────────────────────────

def test_opus_audio_survives_the_move_into_mp4(tmp_path):
    """Copying opus into mp4 would mux but not play — it must be re-encoded."""
    if not (_have("libaom-av1") and _have("libx264")):
        pytest.skip("needs libaom-av1 and libx264")

    src = str(tmp_path / "src.mkv")
    _make_source(src, vcodec="av1", acodec="libopus", pix_fmt="yuv420p")
    probe = _probe(src)
    assert probe["acodec"] == "opus"

    dst = str(tmp_path / "out.mp4")
    _run_chain(d._build_transcode_cmds(
        src, dst, "h264", quality="small", container="mp4", probe=probe))

    assert _probe(dst)["acodec"] == "aac"


def test_silent_source_produces_a_silent_file(tmp_path):
    if not _have("libx264"):
        pytest.skip("needs libx264")

    src = str(tmp_path / "silent.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size=160x120:rate=15:duration={_SECS}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", src],
        check=True, capture_output=True, timeout=120,
    )
    probe = _probe(src)
    assert not probe["audio"]

    dst = str(tmp_path / "out.webm")
    if not _have("libvpx-vp9"):
        pytest.skip("needs libvpx-vp9")
    _run_chain(d._build_transcode_cmds(
        src, dst, "vp9", quality="small", container="webm", probe=probe))

    out = _probe(dst)
    assert out["video"] and not out["audio"]


# ── the fallback chain ───────────────────────────────────────────────────────

def test_falls_back_to_software_when_the_gpu_encoder_fails(tmp_path):
    """A build can advertise h264_nvenc on a box with no NVIDIA card at all —
    the chain has to survive that, not just skip it at listing time."""
    if not _have("libx264"):
        pytest.skip("needs libx264")

    src = str(tmp_path / "src.mp4")
    _make_source(src, vcodec="h264", acodec="aac", pix_fmt="yuv420p")
    probe = _probe(src)

    # Force a first candidate that cannot possibly initialise.
    real = d._available_encoders
    d._available_encoders = lambda **kw: {"h264_nvenc", "h264_qsv", "libx264"}
    try:
        cmds = d._build_transcode_cmds(
            src, str(tmp_path / "out.mp4"), "h264", quality="small",
            container="mp4", probe=probe)
        assert [e for e, _ in cmds][0] != "libx264", "no GPU attempt to fail"
        used = _run_chain(cmds)
    finally:
        d._available_encoders = real

    assert used == "libx264"
    assert d.normalise_codec(_probe(str(tmp_path / "out.mp4"))["vcodec"]) == "h264"
