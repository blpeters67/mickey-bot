"""Tests for scripts/voice_eval.py — scoring rules and the evaluation loop,
with a stand-in transcriber (real whisper runs only on the bot host)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "voice_eval.py")
_spec = importlib.util.spec_from_file_location("voice_eval", _PATH)
ve = importlib.util.module_from_spec(_spec)
sys.modules["voice_eval"] = ve          # dataclasses look their module up here
_spec.loader.exec_module(ve)


@pytest.mark.parametrize("expected,got,verdict", [
    ("skip", "skip", "correct"),
    ("none", "skip", "false_action"),
    ("none", "wake_only", "correct"),
    ("skip", "none", "missed"),
    ("skip", "pause", "wrong"),
    ("play:mr brightside", "play:Mr. Brightside by The Killers", "correct"),
    ("play:mr brightside", "play:despacito", "wrong"),
    # The reviewer's case: same title, WRONG artist — must not count.
    ("play:the pretender by infected mushroom", "play:the pretender by foo fighters", "wrong"),
    ("play:the pretender by infected mushroom", "play:The Pretender by Infected Mushroom.", "correct"),
    ("play:the pretender by infected mushroom", "play:the pretender", "wrong"),       # artist lost
    ("play:the pretender by infected mushroom", "play:the pretender infected mushrooms", "correct"),
    ("play:stand by me | ben e king", "play:stand by me by ben e. king", "correct"),
    ("play:stand by me | ben e king", "play:stand by me by oasis", "wrong"),
    ("loop:on", "loop:on", "correct"),
    ("volume:50", "volume:50", "correct"),
])
def test_judge(expected, got, verdict):
    assert ve.judge(expected, got) == verdict


def _flite_ok():
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return " flite " in out.stdout


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def _tone_wav(path, seconds=1.0):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=440:duration={seconds}", path], check=True)


@needs_ffmpeg
def test_evaluate_scores_a_labelled_set(tmp_path):
    for name in ("a.wav", "b.wav", "c.wav", "d.wav"):
        _tone_wav(str(tmp_path / name))
    (tmp_path / "labels.csv").write_text(
        "file,expected\na.wav,skip\nb.wav,none\nc.wav,volume:50\nd.wav,play:mr brightside\n")
    # Stand-in "whisper": each file's audio → a scripted transcript.
    scripted = iter(["Hey DJ, skip.", "hey dj next week i'm away", "hey dj volume fifty",
                     "Hey DJ, play Mr. Brightside."])
    clips = ve.read_labels(str(tmp_path))
    res = ve.evaluate(clips, str(tmp_path), lambda audio: next(scripted),
                      wakes=["hey dj"], segment=False, label="fake")
    s = res.summary()
    assert [r[3] for r in res.rows] == ["correct", "correct", "correct", "correct"]
    assert s["command accuracy"] == 1.0 and s["false actions"] == 0
    assert s["clips"] == 4


@pytest.mark.skipif(not _flite_ok(), reason="needs ffmpeg with flite")
def test_segment_mode_finds_the_command_after_chatter(tmp_path):
    pytest.importorskip("onnxruntime")
    wav = str(tmp_path / "long.wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "flite=text='so anyway the game was good':voice=slt",
         "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=0.8",
         "-f", "lavfi", "-i", "flite=text='hey D J skip':voice=slt",
         "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", wav], check=True)
    (tmp_path / "labels.csv").write_text("file,expected\nlong.wav,skip\n")
    heard = iter(["so anyway the game was good", "hey dj skip"])
    res = ve.evaluate(ve.read_labels(str(tmp_path)), str(tmp_path), lambda a: next(heard),
                      wakes=["hey dj"], segment=True, label="seg")
    clip, text, got, verdict, _asr, _audio = res.rows[0]
    assert got == "skip" and verdict == "correct" and text.count("|") == 1


def test_song_match_scores_entities_separately():
    m = ve.song_match("the pretender by infected mushroom", "the pretender by foo fighters")
    assert m == {"title": 1.0, "artist": False, "exact": False}


@pytest.mark.parametrize("timeline,expected", [
    ([("hey dj", 0.0, 0.5), ("skip", 2.0, 2.4)], "skip"),           # inside the window
    ([("hey dj", 0.0, 0.5), ("skip", 6.0, 6.4)], "wake_only"),      # window closed (4 s)
    ([("hey dj", 0.0, 0.5), ("so what's up", 1.0, 2.0), ("skip", 4.2, 4.5)], "skip"),
    ([("I said hey dj skip", 0.0, 1.0)], "none"),                   # quoted: never acts
    ([("skip", 0.0, 0.4)], "none"),                                 # no wake, no window
])
def test_decide_replays_the_wake_window_with_timestamps(timeline, expected):
    assert ve.decide(timeline, ["hey dj"]) == expected


def test_summary_reports_exact_and_entity_matches_and_asr_only_timing():
    res = ve.ConfigResult("x")
    clip = ve.Clip("a.wav", "play:the pretender by infected mushroom",
                   transcript="hey dj play the pretender by infected mushroom")
    res.rows.append((clip, "hey dj play the pretender by foo fighters",
                     "play:the pretender by foo fighters", "wrong", 0.5, 2.0))
    res.rows.append((clip, "Hey DJ, play The Pretender by Infected Mushroom.",
                     "play:The Pretender by Infected Mushroom", "correct", 0.7, 2.0))
    s = res.summary()
    assert s["play exact query"] == 0.5 and s["play entity match"] == 0.5
    assert s["transcript exact"] == 0.5
    assert "asr-only median s" in s and not any(k.startswith("asr median") for k in s)
