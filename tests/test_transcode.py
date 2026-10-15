"""Tests for the -transcode command's pure helpers.

Command-building and codec/quality parsing are all deterministic, so they're
tested directly. The end-to-end ffmpeg round-trip lives in
`test_transcode_roundtrip.py`, which is skipped unless ffmpeg is present.
"""
from __future__ import annotations

import pytest

from cogs import downloads as d


# ── codec / quality parsing ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("h264", "h264"), ("H264", "h264"), ("avc", "h264"), ("x264", "h264"),
    ("h.264", "h264"), (".h264", "h264"), ("  H.264  ", "h264"),
    ("h265", "h265"), ("hevc", "h265"), ("HEVC", "h265"), ("x265", "h265"),
    ("hvc1", "h265"),
    ("vp9", "vp9"), ("VP09", "vp9"), ("vp-9", "vp9"),
    ("av1", "av1"), ("av01", "av1"), ("AV01", "av1"), ("svt-av1", "av1"),
    ("libaom", "av1"),
])
def test_normalise_codec_accepts_aliases(raw, expect):
    assert d.normalise_codec(raw) == expect


@pytest.mark.parametrize("raw", ["", None, "mp4", "mkv", "h266", "vp8",
                                 "https://example.com/a.mp4", "gibberish"])
def test_normalise_codec_rejects_non_codecs(raw):
    assert d.normalise_codec(raw) is None


def test_ffprobe_codec_names_normalise():
    """A probe result must feed straight back through normalise_codec."""
    for probed, expect in [("h264", "h264"), ("hevc", "h265"),
                           ("vp9", "vp9"), ("av1", "av1")]:
        assert d.normalise_codec(probed) == expect


@pytest.mark.parametrize("raw,expect", [
    ("high", "high"), ("best", "high"), ("HQ", "high"),
    ("balanced", "balanced"), ("medium", "balanced"), ("default", "balanced"),
    ("small", "small"), ("tiny", "small"), ("low", "small"),
])
def test_normalise_quality_accepts_aliases(raw, expect):
    assert d.normalise_quality(raw) == expect


@pytest.mark.parametrize("raw", ["", None, "h264", "huge", "42"])
def test_normalise_quality_rejects_junk(raw):
    assert d.normalise_quality(raw) is None


def test_codec_and_quality_namespaces_do_not_overlap():
    """`transcode <token>` classifies by identity, so no token may be both."""
    for c in d.TRANSCODE_CODECS:
        assert d.normalise_quality(c) is None
    for q in d.TRANSCODE_QUALITIES:
        assert d.normalise_codec(q) is None


# ── loose positional argument parsing ────────────────────────────────────────

_URL = "https://cdn.example.com/clip.mp4"


@pytest.mark.parametrize("tokens,url,expect", [
    # (codec, quality, url)
    ((None, None),        None,  (None,   "balanced", None)),
    (("h264", None),      None,  ("h264", "balanced", None)),
    (("h264", "small"),   None,  ("h264", "small",    None)),
    (("small", "h264"),   None,  ("h264", "small",    None)),   # either order
    (("av01", "best"),    None,  ("av1",  "high",     None)),   # both aliased
    ((_URL, None),        None,  (None,   "balanced", _URL)),   # bare URL
    (("h264", _URL),      None,  ("h264", "balanced", _URL)),
    (("h264", "small"),   _URL,  ("h264", "small",    _URL)),   # url kwarg
    ((None, None),        _URL,  (None,   "balanced", _URL)),
])
def test_parse_transcode_args(tokens, url, expect):
    codec, quality, got_url, err = d.parse_transcode_args(tokens, url)
    assert err is None
    assert (codec, quality, got_url) == expect


def test_parse_transcode_args_rejects_junk():
    _, _, _, err = d.parse_transcode_args(("banana", None), None)
    assert err and "banana" in err


def test_parse_transcode_args_rejects_a_container_name():
    """`transcode mp4` is a convert request — say so rather than guessing."""
    _, _, _, err = d.parse_transcode_args(("mp4", None), None)
    assert err and "mp4" in err


def test_parse_transcode_args_never_leaves_a_url_in_the_codec_slot():
    codec, _, url, err = d.parse_transcode_args((_URL, None), None)
    assert err is None
    assert codec is None and url == _URL


def test_parse_transcode_args_unwraps_angle_bracketed_urls():
    """Discord users paste <https://…> to suppress the embed."""
    _, _, url, err = d.parse_transcode_args((f"<{_URL}>", None), None)
    assert err is None and url == _URL


# ── container selection ──────────────────────────────────────────────────────

@pytest.mark.parametrize("codec,src_ext,expect", [
    ("h264", "webm", "mp4"),    # webm can't hold h264
    ("h264", "mkv",  "mkv"),    # mkv source keeps mkv
    ("h264", "mov",  "mov"),
    ("h265", "webm", "mp4"),
    ("vp9",  "mp4",  "webm"),   # mp4 vp9 exists but webm is the safe default
    ("vp9",  "mkv",  "mkv"),
    ("av1",  "avi",  "mp4"),
    ("av1",  "webm", "webm"),
    ("h264", "",     "mp4"),    # extensionless source
    ("h264", ".MKV", "mkv"),    # case and leading dot
])
def test_pick_container(codec, src_ext, expect):
    assert d.pick_container(codec, src_ext) == expect


def test_every_codec_has_a_default_container():
    for codec in d.TRANSCODE_CODECS:
        assert d._CODEC_CONTAINERS[codec][0] in d._CONTAINER_AUDIO_ENCODE


# ── table completeness ───────────────────────────────────────────────────────

def test_tables_cover_every_codec():
    for codec in d.TRANSCODE_CODECS:
        assert codec in d._CODEC_LABEL
        assert codec in d._CODEC_BLURB
        assert codec in d._CODEC_FFNAME
        assert codec in d._CODEC_CONTAINERS
        assert codec in d._SW_ENCODERS and d._SW_ENCODERS[codec]
        assert set(d._CRF[codec]) == set(d.TRANSCODE_QUALITIES)


def test_crf_tiers_are_ordered_high_to_small():
    """Higher CRF = worse quality = smaller file, for all four codecs."""
    for codec, tiers in d._CRF.items():
        assert tiers["high"] < tiers["balanced"] < tiers["small"], codec


# ── encoder discovery ────────────────────────────────────────────────────────

def test_encoder_regex_ignores_the_header_legend():
    """`ffmpeg -encoders` prints 'V..... = Video' above the table."""
    sample = """Encoders:
 V..... = Video
 A..... = Audio
 ------
 V..... libx264              libx264 H.264 / AVC
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder
 A..... aac                  AAC (Advanced Audio Coding)
"""
    found = {m.group(1) for m in
             (d._ENCODER_RE.match(ln) for ln in sample.splitlines()) if m}
    assert found == {"libx264", "h264_nvenc"}
    assert "=" not in found


def test_software_encoder_is_always_last(monkeypatch):
    """The fallback chain must end at software — GPU encoders can fail late."""
    monkeypatch.setattr(d, "_available_encoders",
                        lambda **kw: {"h264_nvenc", "h264_qsv", "libx264"})
    chain = d._pick_video_encoders("h264")
    assert chain[-1] == "libx264"
    assert chain[0] == "h264_nvenc"        # preference order honoured


def test_hardware_is_never_used_for_vp9_or_av1(monkeypatch):
    """Their HW quality knobs use vendor scales our CRF numbers would misset."""
    monkeypatch.setattr(d, "_available_encoders",
                        lambda **kw: {"vp9_qsv", "vp9_vaapi", "libvpx-vp9",
                                      "av1_nvenc", "av1_qsv", "libsvtav1"})
    assert d._pick_video_encoders("vp9") == ["libvpx-vp9"]
    assert d._pick_video_encoders("av1") == ["libsvtav1"]


def test_vaapi_skipped_without_a_render_node(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders",
                        lambda **kw: {"h264_vaapi", "libx264"})
    monkeypatch.setattr(d.os.path, "exists", lambda p: False)
    assert d._pick_video_encoders("h264") == ["libx264"]


def test_no_encoders_when_build_lacks_them(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: set())
    assert d._pick_video_encoders("h264") == []


# ── command building ─────────────────────────────────────────────────────────

def _probe(**over):
    base = {"video": True, "attached_pic": False, "audio": True,
            "duration": 10.0, "vcodec": "av1", "acodec": "opus",
            "pix_fmt": "yuv420p10le", "width": 1920, "height": 1080}
    base.update(over)
    return base


def _cmd_for(codec, monkeypatch, *, encoders=None, **probe_over):
    monkeypatch.setattr(
        d, "_available_encoders",
        lambda **kw: set(encoders or d._SW_ENCODERS[codec]))
    cmds = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.mp4", codec, quality="balanced",
        container=d._CODEC_CONTAINERS[codec][0], probe=_probe(**probe_over),
    )
    return cmds[0][1]


def test_h264_forces_8bit_from_a_10bit_av1_source(monkeypatch):
    """High 10 profile is what most hardware decoders choke on."""
    cmd = _cmd_for("h264", monkeypatch, pix_fmt="yuv420p10le")
    assert "yuv420p" in cmd
    assert "yuv420p10le" not in cmd
    assert cmd[cmd.index("-profile:v") + 1] == "high"


def test_av1_and_h265_keep_10bit(monkeypatch):
    for codec in ("h265", "av1"):
        cmd = _cmd_for(codec, monkeypatch, pix_fmt="yuv420p10le")
        assert "yuv420p10le" in cmd, codec


def test_8bit_source_stays_8bit(monkeypatch):
    for codec in d.TRANSCODE_CODECS:
        cmd = _cmd_for(codec, monkeypatch, pix_fmt="yuv420p")
        assert "yuv420p10le" not in cmd, codec


def test_h265_in_mp4_gets_the_hvc1_tag(monkeypatch):
    """Without it QuickTime/Safari play the audio over a black frame."""
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"libx265"})
    cmd = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.mp4", "h265", quality="balanced",
        container="mp4", probe=_probe())[0][1]
    assert cmd[cmd.index("-tag:v") + 1] == "hvc1"


def test_h265_in_mkv_has_no_hvc1_tag(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"libx265"})
    cmd = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.mkv", "h265", quality="balanced",
        container="mkv", probe=_probe())[0][1]
    assert "-tag:v" not in cmd
    assert "-movflags" not in cmd


def test_audio_is_copied_when_the_container_allows_it(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"libx264"})
    cmd = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.mp4", "h264", quality="balanced",
        container="mp4", probe=_probe(acodec="aac"))[0][1]
    assert cmd[cmd.index("-c:a") + 1] == "copy"


def test_opus_audio_is_reencoded_into_mp4(monkeypatch):
    """mp4 + opus is legal on paper and broken in half the players alive."""
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"libx264"})
    cmd = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.mp4", "h264", quality="balanced",
        container="mp4", probe=_probe(acodec="opus"))[0][1]
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_opus_audio_is_copied_into_webm(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"libvpx-vp9"})
    cmd = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.webm", "vp9", quality="balanced",
        container="webm", probe=_probe(acodec="opus"))[0][1]
    assert cmd[cmd.index("-c:a") + 1] == "copy"


def test_aac_audio_is_reencoded_into_webm(monkeypatch):
    """webm only carries opus/vorbis — anything else has to be converted."""
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"libvpx-vp9"})
    cmd = d._build_transcode_cmds(
        "/tmp/in.mp4", "/tmp/out.webm", "vp9", quality="balanced",
        container="webm", probe=_probe(acodec="aac"))[0][1]
    assert cmd[cmd.index("-c:a") + 1] == "libopus"


def test_silent_source_gets_an(monkeypatch):
    cmd = _cmd_for("h264", monkeypatch, audio=False, acodec="")
    assert "-an" in cmd
    assert "-c:a" not in cmd


def test_every_command_scales_to_even_dimensions(monkeypatch):
    """yuv420p can't represent odd width/height; some encoders hard-fail."""
    for codec in d.TRANSCODE_CODECS:
        cmd = _cmd_for(codec, monkeypatch)
        assert "trunc(iw/2)*2:trunc(ih/2)*2" in cmd[cmd.index("-vf") + 1], codec


def test_every_command_takes_one_video_and_one_audio_track(monkeypatch):
    """Multi-track mkv rips carry commentary and subs mp4 can't hold."""
    for codec in d.TRANSCODE_CODECS:
        cmd = _cmd_for(codec, monkeypatch)
        assert cmd[cmd.index("-map") + 1] == "0:v:0", codec
        assert "0:a:0?" in cmd, codec
        assert "-sn" in cmd, codec


def test_every_command_reports_progress(monkeypatch):
    """The status bar reads ffmpeg's -progress pipe; without it, it's dead."""
    for codec in d.TRANSCODE_CODECS:
        cmd = _cmd_for(codec, monkeypatch)
        assert cmd[cmd.index("-progress") + 1] == "pipe:1", codec


def test_quality_tier_changes_the_crf(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"libx264"})
    seen = []
    for q in d.TRANSCODE_QUALITIES:
        cmd = d._build_transcode_cmds(
            "/tmp/in.mkv", "/tmp/out.mp4", "h264", quality=q,
            container="mp4", probe=_probe())[0][1]
        seen.append(int(cmd[cmd.index("-crf") + 1]))
    assert seen == sorted(seen)          # high → small = rising CRF
    assert len(set(seen)) == 3


def test_hardware_and_software_commands_are_both_emitted(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders",
                        lambda **kw: {"h264_nvenc", "libx264"})
    cmds = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.mp4", "h264", quality="balanced",
        container="mp4", probe=_probe())
    assert [enc for enc, _ in cmds] == ["h264_nvenc", "libx264"]
    nvenc = cmds[0][1]
    assert nvenc[nvenc.index("-cq") + 1] == str(d._CRF["h264"]["balanced"])


def test_vaapi_command_uploads_frames_to_the_gpu(monkeypatch):
    monkeypatch.setattr(d, "_available_encoders", lambda **kw: {"h264_vaapi"})
    monkeypatch.setattr(d.os.path, "exists", lambda p: True)
    cmd = d._build_transcode_cmds(
        "/tmp/in.mkv", "/tmp/out.mp4", "h264", quality="balanced",
        container="mp4", probe=_probe())[0][1]
    assert "-vaapi_device" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    # The scale has to run on the CPU, before the upload.
    assert vf.index("scale=") < vf.index("hwupload")
    assert "format=nv12" in vf


def test_videotoolbox_quality_is_inverted(monkeypatch):
    """VideoToolbox's -q:v is 1–100 with higher = better, unlike CRF."""
    monkeypatch.setattr(d, "_available_encoders",
                        lambda **kw: {"h264_videotoolbox", "libx264"})
    hi = d._hw_rate_control("videotoolbox", d._CRF["h264"]["high"])
    lo = d._hw_rate_control("videotoolbox", d._CRF["h264"]["small"])
    assert int(hi[1]) > int(lo[1])
    assert 1 <= int(lo[1]) <= 100


def test_output_path_is_last_and_input_is_after_i(monkeypatch):
    for codec in d.TRANSCODE_CODECS:
        monkeypatch.setattr(d, "_available_encoders",
                            lambda **kw: set(d._SW_ENCODERS[codec]))
        cmds = d._build_transcode_cmds(
            "/tmp/in.mkv", "/tmp/out.bin", codec, quality="balanced",
            container="mp4", probe=_probe())
        for _, cmd in cmds:
            assert cmd[-1] == "/tmp/out.bin", codec
            assert cmd[cmd.index("-i") + 1] == "/tmp/in.mkv", codec
