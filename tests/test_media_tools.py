"""Tests for -ocr, -chapters, -normalize and -desilence.

The pure parts (chapter normalisation, filter construction, preset parsing)
always run. The parts that shell out to ffmpeg or tesseract are marked and
skip when the binary isn't installed.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from cogs import downloads as d

_HAS_FFMPEG    = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
_HAS_TESSERACT = bool(shutil.which("tesseract"))


# ── chapter extraction ───────────────────────────────────────────────────────

def test_chapters_are_normalised_and_indexed():
    info = {"duration": 300, "chapters": [
        {"start_time": 0,   "end_time": 100, "title": "Intro"},
        {"start_time": 100, "end_time": 220, "title": "Main bit"},
        {"start_time": 220, "end_time": 300, "title": "Outro"},
    ]}
    out = d.extract_chapters(info)
    assert [c["index"] for c in out] == [1, 2, 3]
    assert [c["title"] for c in out] == ["Intro", "Main bit", "Outro"]
    assert out[1]["start"] == 100 and out[1]["end"] == 220


def test_missing_end_times_come_from_the_next_chapter():
    """Several extractors only give start times."""
    info = {"duration": 300, "chapters": [
        {"start_time": 0,   "title": "One"},
        {"start_time": 120, "title": "Two"},
        {"start_time": 240, "title": "Three"},
    ]}
    out = d.extract_chapters(info)
    assert [(c["start"], c["end"]) for c in out] == [
        (0, 120), (120, 240), (240, 300)]


def test_last_chapter_without_an_end_uses_the_video_duration():
    info = {"duration": 500, "chapters": [{"start_time": 10, "title": "Only"}]}
    assert d.extract_chapters(info)[0]["end"] == 500


def test_zero_length_chapters_are_dropped():
    """Duplicate markers are a metadata bug, not a clip worth cutting."""
    info = {"duration": 100, "chapters": [
        {"start_time": 0,  "end_time": 50, "title": "Real"},
        {"start_time": 50, "end_time": 50, "title": "Empty"},
        {"start_time": 50, "end_time": 100, "title": "Also real"},
    ]}
    out = d.extract_chapters(info)
    assert [c["title"] for c in out] == ["Real", "Also real"]
    assert [c["index"] for c in out] == [1, 2]


def test_chapters_without_a_usable_start_are_skipped():
    info = {"duration": 100, "chapters": [
        {"title": "No start"},
        {"start_time": "banana", "title": "Bad start"},
        {"start_time": 0, "end_time": 100, "title": "Fine"},
    ]}
    assert [c["title"] for c in d.extract_chapters(info)] == ["Fine"]


@pytest.mark.parametrize("info", [
    {}, {"chapters": None}, {"chapters": []}, {"duration": 100},
])
def test_no_chapters_is_an_empty_list_not_a_crash(info):
    assert d.extract_chapters(info) == []


@pytest.mark.parametrize("raw,expect", [
    ("Intro", "Intro"),
    ("  spaced  out  ", "spaced out"),
    ("with/slashes:and*stars", "with-slashes-and-stars"),
    ("trailing dots...", "trailing dots"),
    ("", "Chapter 1"),
    ("   ", "Chapter 1"),
])
def test_chapter_titles_are_safe_as_filenames(raw, expect):
    assert d._clean_chapter_title(raw, 1) == expect


def test_long_chapter_titles_are_truncated():
    assert len(d._clean_chapter_title("x" * 500, 1)) <= 80


# ── preset argument splitting ────────────────────────────────────────────────

_URL = "https://cdn.example.com/clip.mp3"


@pytest.mark.parametrize("arg,url,expect", [
    (None,        None, ("speech", None)),
    ("music",     None, ("music", None)),
    ("broadcast", None, ("broadcast", None)),
    ("spotify",   None, ("music", None)),        # alias
    ("podcast",   None, ("speech", None)),       # alias
    (_URL,        None, ("speech", _URL)),       # bare URL in the preset slot
    ("music",     _URL, ("music", _URL)),
])
def test_split_preset_arg(arg, url, expect):
    preset, got_url, err = d._split_preset_arg(
        arg, url, d.LOUDNESS_PRESETS, d._LOUDNESS_ALIASES, "speech", "target")
    assert err is None
    assert (preset, got_url) == expect


def test_split_preset_arg_rejects_junk():
    _p, _u, err = d._split_preset_arg(
        "banana", None, d.LOUDNESS_PRESETS, d._LOUDNESS_ALIASES,
        "speech", "target")
    assert err and "banana" in err


def test_loudness_presets_are_ordered_loud_to_quiet():
    assert (d.LOUDNESS_PRESETS["music"]["I"]
            > d.LOUDNESS_PRESETS["speech"]["I"]
            > d.LOUDNESS_PRESETS["broadcast"]["I"])


def test_silence_presets_get_progressively_more_aggressive():
    g, n, a = (d.SILENCE_PRESETS[k] for k in ("gentle", "normal", "aggressive"))
    assert g["min_silence"] > n["min_silence"] > a["min_silence"]
    assert g["keep"] > n["keep"] > a["keep"]
    assert g["threshold"] < n["threshold"] < a["threshold"]


def test_every_silence_preset_keeps_some_silence():
    """Cutting a gap to zero makes speech sound spliced."""
    for name, p in d.SILENCE_PRESETS.items():
        assert p["keep"] > 0, name


# ── filter construction ──────────────────────────────────────────────────────

def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_normalize_single_pass_when_measurement_failed():
    cmd = d._build_normalize_cmd(
        "/tmp/in.mp3", "/tmp/out.mp3", d.LOUDNESS_PRESETS["speech"],
        measured=None, keep_video=False, audio_args=["-c:a", "libmp3lame"])
    af = _flag(cmd, "-af")
    assert "loudnorm=I=-16" in af
    assert "measured_I" not in af


def test_normalize_second_pass_carries_the_measurements():
    measured = {"input_i": "-27.1", "input_tp": "-8.4", "input_lra": "5.2",
                "input_thresh": "-37.6", "target_offset": "0.3"}
    cmd = d._build_normalize_cmd(
        "/tmp/in.mp3", "/tmp/out.mp3", d.LOUDNESS_PRESETS["music"],
        measured=measured, keep_video=False, audio_args=["-c:a", "aac"])
    af = _flag(cmd, "-af")
    # loudnorm's parameter names are case-sensitive: measured_I, not measured_i.
    for key, token in (("input_i", "measured_I"), ("input_tp", "measured_TP"),
                       ("input_lra", "measured_LRA"),
                       ("input_thresh", "measured_thresh"),
                       ("target_offset", "offset")):
        assert f"{token}={measured[key]}" in af, key
    # linear=true is what preserves dynamics instead of compressing them.
    assert "linear=true" in af


def test_normalize_keeps_video_as_a_stream_copy():
    """Loudness doesn't touch the timeline, so re-encoding video is waste."""
    cmd = d._build_normalize_cmd(
        "/tmp/in.mp4", "/tmp/out.mp4", d.LOUDNESS_PRESETS["speech"],
        measured=None, keep_video=True, audio_args=["-c:a", "aac"])
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert "0:v:0" in cmd


def test_normalize_audio_only_has_no_video_mapping():
    cmd = d._build_normalize_cmd(
        "/tmp/in.mp3", "/tmp/out.mp3", d.LOUDNESS_PRESETS["speech"],
        measured=None, keep_video=False, audio_args=["-c:a", "libmp3lame"])
    assert "-c:v" not in cmd


def test_desilence_strips_video():
    """Shortening the audio would desync any video riding along."""
    cmd = d._build_desilence_cmd(
        "/tmp/in.mp4", "/tmp/out.mp3", d.SILENCE_PRESETS["normal"],
        ["-c:a", "libmp3lame"])
    assert "-vn" in cmd


def test_desilence_removes_every_gap_not_just_the_first():
    cmd = d._build_desilence_cmd(
        "/tmp/in.mp3", "/tmp/out.mp3", d.SILENCE_PRESETS["normal"],
        ["-c:a", "libmp3lame"])
    af = _flag(cmd, "-af")
    assert "stop_periods=-1" in af      # -1 = all of them
    assert "detection=rms" in af        # perceived level, not peak
    assert "stop_silence=0.3" in af     # leaves breathing room


def test_both_filters_report_progress():
    for cmd in (
        d._build_normalize_cmd("/i", "/o", d.LOUDNESS_PRESETS["speech"],
                               measured=None, keep_video=False,
                               audio_args=["-c:a", "aac"]),
        d._build_desilence_cmd("/i", "/o", d.SILENCE_PRESETS["normal"],
                               ["-c:a", "aac"]),
    ):
        assert _flag(cmd, "-progress") == "pipe:1"
        assert cmd[-1] == "/o"


@pytest.mark.parametrize("ext,expect_ext", [
    ("wav", "wav"), (".wav", "wav"), ("flac", "flac"),
    ("mp3", "mp3"), ("ogg", "mp3"), ("opus", "mp3"),
    ("m4a", "m4a"), ("mp4", "m4a"), ("mkv", "m4a"),
    ("", "mp3"),
])
def test_audio_output_format_follows_the_source(ext, expect_ext):
    _args, out_ext = d._audio_output_args(ext)
    assert out_ext == expect_ext


def test_lossless_sources_stay_lossless():
    assert d._audio_output_args("flac")[0] == ["-c:a", "flac"]
    assert d._audio_output_args("wav")[0] == ["-c:a", "pcm_s16le"]


# ── OCR helpers ──────────────────────────────────────────────────────────────

def test_tidy_collapses_tesseracts_blank_line_runs():
    raw = "Line one\n\n\n\nLine two\n\n\n\n\nLine three\n\n\n"
    assert d._tidy_ocr_text(raw) == "Line one\n\nLine two\n\nLine three"


def test_tidy_strips_trailing_whitespace_per_line():
    assert d._tidy_ocr_text("hello   \nworld\t\n") == "hello\nworld"


@pytest.mark.parametrize("raw", ["", "   ", "\n\n\n", None])
def test_tidy_of_nothing_is_empty(raw):
    assert d._tidy_ocr_text(raw) == ""


def test_ocr_availability_matches_the_binary():
    assert d.tesseract_available() == _HAS_TESSERACT


# ── real ffmpeg round-trips ──────────────────────────────────────────────────

pytestmark_ff = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")


def _duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True, timeout=60)
    try:
        return float((r.stdout or "0").strip())
    except ValueError:
        return 0.0


def _loudness(path: str) -> float:
    """Integrated LUFS, measured by the same filter that sets it."""
    measured = asyncio.run(_measure(path))
    return float(measured["input_i"]) if measured else 0.0


async def _measure(path):
    return await d._measure_loudness(path, d.LOUDNESS_PRESETS["speech"])


@pytestmark_ff
def test_normalize_actually_moves_the_loudness(tmp_path):
    """A deliberately quiet file should come out at the target level."""
    quiet = str(tmp_path / "quiet.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=300:duration=5", "-af", "volume=-30dB",
         "-c:a", "pcm_s16le", quiet], check=True, capture_output=True, timeout=120)

    before = _loudness(quiet)
    assert before < -30, f"test fixture wasn't quiet ({before} LUFS)"

    target   = d.LOUDNESS_PRESETS["speech"]
    measured = asyncio.run(d._measure_loudness(quiet, target))
    assert measured is not None, "analysis pass produced no JSON"

    out = str(tmp_path / "loud.wav")
    cmd = d._build_normalize_cmd(quiet, out, target, measured=measured,
                                 keep_video=False,
                                 audio_args=["-c:a", "pcm_s16le"])
    asyncio.run(d._run_ffmpeg(cmd, duration=5.0))

    after = _loudness(out)
    assert abs(after - target["I"]) < 2.0, \
        f"wanted ~{target['I']} LUFS, got {after}"
    assert after > before


@pytestmark_ff
def test_normalize_keeps_the_video_stream(tmp_path):
    src = str(tmp_path / "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", src],
        check=True, capture_output=True, timeout=120)

    out = str(tmp_path / "out.mp4")
    cmd = d._build_normalize_cmd(
        src, out, d.LOUDNESS_PRESETS["speech"], measured=None,
        keep_video=True, audio_args=["-c:a", "aac", "-b:a", "192k"])
    asyncio.run(d._run_ffmpeg(cmd, duration=3.0))

    probe = asyncio.run(d._probe_media(out))
    assert probe["video"] and probe["audio"]
    assert abs(_duration(out) - 3.0) < 0.5


@pytestmark_ff
def test_desilence_actually_shortens_a_padded_file(tmp_path):
    """2s tone + 3s silence + 2s tone should lose most of the middle."""
    src = str(tmp_path / "padded.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-filter_complex",
         "sine=frequency=440:duration=2[a];"
         "anullsrc=r=44100:cl=mono:duration=3[b];"
         "sine=frequency=440:duration=2[c];"
         "[a][b][c]concat=n=3:v=0:a=1[out]",
         "-map", "[out]", "-c:a", "pcm_s16le", src],
        check=True, capture_output=True, timeout=120)
    before = _duration(src)
    assert before > 6.5, f"fixture was {before}s, expected ~7"

    out = str(tmp_path / "tight.wav")
    cmd = d._build_desilence_cmd(src, out, d.SILENCE_PRESETS["normal"],
                                 ["-c:a", "pcm_s16le"])
    asyncio.run(d._run_ffmpeg(cmd, duration=before))

    after = _duration(out)
    assert after < before - 2, f"only trimmed {before - after:.1f}s"
    assert after > 3.5, "cut into the actual audio"


@pytestmark_ff
def test_desilence_leaves_a_gapless_file_alone(tmp_path):
    """Nothing to cut must mean nothing cut — not a mangled file."""
    src = str(tmp_path / "solid.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=4", "-c:a", "pcm_s16le", src],
        check=True, capture_output=True, timeout=120)

    out = str(tmp_path / "same.wav")
    cmd = d._build_desilence_cmd(src, out, d.SILENCE_PRESETS["normal"],
                                 ["-c:a", "pcm_s16le"])
    asyncio.run(d._run_ffmpeg(cmd, duration=4.0))
    assert abs(_duration(out) - 4.0) < 0.6


# ── real OCR ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract not installed")
def test_ocr_reads_rendered_text(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw
    from core.imageops import _load_font

    img = Image.new("RGB", (900, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 60), "HELLO WORLD", fill="black", font=_load_font(90))
    src = str(tmp_path / "text.png")
    img.save(src)

    text = asyncio.run(d._run_ocr(src, "eng"))
    assert "HELLO" in text.upper()
    assert "WORLD" in text.upper()


@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract not installed")
def test_ocr_preprocessing_upscales_a_small_screenshot(tmp_path):
    """The whole point of the prep pass is making small text readable."""
    pytest.importorskip("PIL")
    from PIL import Image

    src = str(tmp_path / "small.png")
    Image.new("RGB", (200, 80), "white").save(src)
    dst = str(tmp_path / "prep.png")

    out = d._prep_for_ocr(src, dst)
    assert out == dst
    with Image.open(out) as im:
        assert im.width >= d.OCR_MIN_WIDTH
        assert im.mode == "L"          # grayscale


@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract not installed")
def test_ocr_on_a_blank_image_returns_nothing(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    src = str(tmp_path / "blank.png")
    Image.new("RGB", (600, 400), "white").save(src)
    assert d._tidy_ocr_text(asyncio.run(d._run_ocr(src, "eng"))) == ""


@pytest.mark.skipif(not _HAS_TESSERACT, reason="tesseract not installed")
def test_installed_languages_include_english():
    assert "eng" in d._ocr_languages()


def test_ocr_prep_falls_back_to_the_original_on_a_bad_file(tmp_path):
    src = str(tmp_path / "junk.png")
    with open(src, "wb") as fh:
        fh.write(b"not an image")
    assert d._prep_for_ocr(src, str(tmp_path / "out.png")) == src


# ── -speed ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("1.5", 1.5), ("0.5", 0.5), ("2", 2.0), ("2x", 2.0), ("1.25X", 1.25),
    ("150%", 1.5), ("75%", 0.75), ("double", 2.0), ("half", 0.5),
    ("twice", 2.0), ("0.25", 0.25), ("4", 4.0),
])
def test_parse_speed(raw, expect):
    assert d.parse_speed(raw) == pytest.approx(expect)


@pytest.mark.parametrize("raw", [
    "", None, "banana", "0", "-1", "10", "0.1", "5x", "500%",
])
def test_parse_speed_rejects_out_of_range_and_junk(raw):
    assert d.parse_speed(raw) is None


def test_speed_bounds_are_inclusive():
    assert d.parse_speed(str(d.SPEED_MIN)) == d.SPEED_MIN
    assert d.parse_speed(str(d.SPEED_MAX)) == d.SPEED_MAX


@pytest.mark.parametrize("factor", [0.25, 0.4, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0])
def test_atempo_chain_multiplies_out_to_the_factor(factor):
    """Each stage must stay inside atempo's 0.5–2.0 window, and the product
    of all of them must be the speed the user asked for."""
    chain = d._atempo_chain(factor)
    stages = [float(part.split("=")[1]) for part in chain.split(",")]
    product = 1.0
    for s in stages:
        assert 0.5 - 1e-6 <= s <= 2.0 + 1e-6, f"{s} is outside atempo's range"
        product *= s
    assert product == pytest.approx(factor, rel=1e-4)


def test_atempo_uses_one_stage_when_it_can():
    assert d._atempo_chain(1.5) == "atempo=1.500000"
    assert "," not in d._atempo_chain(2.0)


def test_atempo_chains_beyond_the_single_stage_range():
    assert d._atempo_chain(4.0).count("atempo") == 2
    assert d._atempo_chain(0.25).count("atempo") == 2


def test_speed_command_rewrites_video_timestamps():
    cmd = d._build_speed_cmd("/i.mp4", "/o.mp4", 2.0, has_video=True,
                             has_audio=True, audio_args=["-c:a", "aac"])
    assert cmd[cmd.index("-vf") + 1] == "setpts=0.500000*PTS"
    assert "atempo=2.000000" in cmd[cmd.index("-af") + 1]


def test_speed_on_audio_only_drops_the_video_stream():
    cmd = d._build_speed_cmd("/i.mp3", "/o.mp3", 1.5, has_video=False,
                             has_audio=True, audio_args=["-c:a", "libmp3lame"])
    assert "-vn" in cmd and "-vf" not in cmd


def test_speed_on_silent_video_drops_the_audio_stream():
    cmd = d._build_speed_cmd("/i.mp4", "/o.mp4", 2.0, has_video=True,
                             has_audio=False, audio_args=["-c:a", "aac"])
    assert "-an" in cmd and "-af" not in cmd


def test_speed_reports_progress():
    cmd = d._build_speed_cmd("/i.mp3", "/o.mp3", 1.5, has_video=False,
                             has_audio=True, audio_args=["-c:a", "aac"])
    assert cmd[cmd.index("-progress") + 1] == "pipe:1"
    assert cmd[-1] == "/o.mp3"


@pytestmark_ff
@pytest.mark.parametrize("factor", [2.0, 0.5, 3.0])
def test_speed_produces_the_right_duration(tmp_path, factor):
    src = str(tmp_path / "in.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=8",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", src],
        check=True, capture_output=True, timeout=120)

    out = str(tmp_path / f"out{factor}.mp4")
    cmd = d._build_speed_cmd(src, out, factor, has_video=True, has_audio=True,
                             audio_args=["-c:a", "aac", "-b:a", "192k"])
    asyncio.run(d._run_ffmpeg(cmd, duration=8 / factor))

    probe = asyncio.run(d._probe_media(out))
    assert probe["duration"] == pytest.approx(8 / factor, abs=0.6)
    # Both streams must survive, or the "video stays in sync" claim is empty.
    assert probe["video"] and probe["audio"]


@pytestmark_ff
def test_speed_keeps_audio_and_video_the_same_length(tmp_path):
    """setpts and atempo have to agree, or the result drifts out of sync."""
    src = str(tmp_path / "in.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", src],
        check=True, capture_output=True, timeout=120)

    out = str(tmp_path / "fast.mp4")
    cmd = d._build_speed_cmd(src, out, 1.5, has_video=True, has_audio=True,
                             audio_args=["-c:a", "aac", "-b:a", "192k"])
    asyncio.run(d._run_ffmpeg(cmd, duration=4.0))

    def _stream_duration(kind):
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", kind,
             "-show_entries", "stream=duration", "-of", "csv=p=0", out],
            capture_output=True, text=True, timeout=60)
        return float((r.stdout or "0").strip().rstrip(",") or 0)

    v, a = _stream_duration("v:0"), _stream_duration("a:0")
    assert abs(v - a) < 0.3, f"video {v}s vs audio {a}s — out of sync"
