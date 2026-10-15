"""Unit tests for core.formatters."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.formatters import (
    fmt_duration,
    fmt_size,
    interleave_queue,
    make_progress_bar,
    parse_indices,
    parse_seek,
    parse_size,
)


# ── fmt_duration ────────────────────────────────────────────────────────────

def test_fmt_duration_short():
    assert fmt_duration(0)    == "??:??"
    assert fmt_duration(None) == "??:??"
    assert fmt_duration(45)   == "0:45"
    assert fmt_duration(90)   == "1:30"
    assert fmt_duration(3599) == "59:59"

def test_fmt_duration_hours():
    assert fmt_duration(3600) == "1:00:00"
    assert fmt_duration(3661) == "1:01:01"


# ── parse_seek — the h:m:s fix ──────────────────────────────────────────────

def test_parse_seek_basic_mmss():
    assert parse_seek("1:30", 0) == 90.0
    assert parse_seek("0:05", 0) == 5.0

def test_parse_seek_hhmmss():
    """Original code couldn't seek past 60:00. This is the fix."""
    assert parse_seek("1:23:45", 0) == 3600 + 23*60 + 45
    assert parse_seek("2:00:00", 0) == 7200

def test_parse_seek_plain_seconds():
    assert parse_seek("90", 0)  == 90.0
    assert parse_seek("0",  0)  == 0.0

def test_parse_seek_relative():
    assert parse_seek("+30", 100) == 130.0
    assert parse_seek("-15", 100) == 85.0
    assert parse_seek("-200", 100) == 0.0
    assert parse_seek("+1:30", 60) == 150.0

def test_parse_seek_invalid():
    assert parse_seek("abc",     0) is None
    assert parse_seek("",        0) is None
    assert parse_seek("1:2:3:4", 0) is None


# ── parse_indices ───────────────────────────────────────────────────────────

def test_parse_indices_single():
    assert parse_indices("3") == {3}

def test_parse_indices_list():
    assert parse_indices("1,3,5") == {1, 3, 5}

def test_parse_indices_range():
    assert parse_indices("2-5") == {2, 3, 4, 5}

def test_parse_indices_mixed():
    assert parse_indices("1,3,5-8") == {1, 3, 5, 6, 7, 8}

def test_parse_indices_reversed_range():
    assert parse_indices("5-2") == {2, 3, 4, 5}

def test_parse_indices_with_spaces():
    assert parse_indices("1, 3 , 5-7") == {1, 3, 5, 6, 7}

def test_parse_indices_invalid():
    assert parse_indices("abc")    is None
    assert parse_indices("1,a,3")  is None


# ── parse_size — the new -resize helper ─────────────────────────────────────

def test_parse_size_gigabytes():
    assert parse_size("5g")     == 5 * 1024**3
    assert parse_size("5gb")    == 5 * 1024**3
    assert parse_size("5GB")    == 5 * 1024**3
    assert parse_size("5 GB")   == 5 * 1024**3
    assert parse_size("5 gb")   == 5 * 1024**3
    assert parse_size("0.5g")   == int(0.5 * 1024**3)
    assert parse_size("1.5gb")  == int(1.5 * 1024**3)

def test_parse_size_megabytes():
    assert parse_size("500m")   == 500 * 1024**2
    assert parse_size("500mb")  == 500 * 1024**2
    assert parse_size("500MB")  == 500 * 1024**2
    assert parse_size("500Mb")  == 500 * 1024**2

def test_parse_size_kilobytes():
    assert parse_size("5kb") == 5 * 1024
    assert parse_size("5k")  == 5 * 1024

def test_parse_size_terabytes():
    assert parse_size("1t")  == 1024**4
    assert parse_size("1tb") == 1024**4

def test_parse_size_default_is_megabytes():
    """A bare number is the most common case — 500 means 500 MB."""
    assert parse_size("500")  == 500 * 1024**2
    assert parse_size("1")    == 1024**2
    assert parse_size("0.5")  == 1024**2 // 2

def test_parse_size_whitespace_tolerant():
    assert parse_size("  500mb  ")  == 500 * 1024**2
    assert parse_size("\t5 gb\n")   == 5 * 1024**3

def test_parse_size_invalid():
    assert parse_size("")        is None
    assert parse_size("abc")     is None
    assert parse_size("gb")      is None        # no number
    assert parse_size("5xb")     is None        # bad magnitude
    assert parse_size("5 g b")   is None        # split unit
    assert parse_size("-5gb")    is None        # negative
    assert parse_size("5gb extra")  is None     # trailing junk


# ── fmt_size ────────────────────────────────────────────────────────────────

def test_fmt_size_bytes():
    assert fmt_size(0)    == "0 B"
    assert fmt_size(500)  == "500 B"
    assert fmt_size(1023) == "1023 B"

def test_fmt_size_units():
    assert fmt_size(1024)      == "1.00 KB"
    assert fmt_size(1024**2)   == "1.00 MB"
    assert fmt_size(1024**3)   == "1.00 GB"
    assert fmt_size(1024**4)   == "1.00 TB"

def test_fmt_size_fractional():
    assert fmt_size(1500)              == "1.46 KB"
    assert fmt_size(int(1.5 * 1024**3)) == "1.50 GB"


def test_parse_size_fmt_size_roundtrip():
    """Anything parse_size returns should be re-parsable from its fmt_size form,
    modulo rounding."""
    for orig in ("500mb", "1g", "0.5gb", "100kb"):
        b = parse_size(orig)
        s = fmt_size(b)
        # fmt_size produces forms like "1.00 GB" which parse_size also accepts
        assert parse_size(s) is not None


# ── interleave_queue ────────────────────────────────────────────────────────

def test_interleave_queue_balances():
    tracks = [
        {"title": "a1", "requester": "A"},
        {"title": "a2", "requester": "A"},
        {"title": "a3", "requester": "A"},
        {"title": "b1", "requester": "B"},
        {"title": "b2", "requester": "B"},
    ]
    out = list(interleave_queue(tracks))
    requesters_seq = [t["requester"] for t in out]
    for i in range(len(requesters_seq) - 2):
        assert not (
            requesters_seq[i] == "A"
            and requesters_seq[i+1] == "A"
            and requesters_seq[i+2] == "A"
        )
    assert len(out) == 5

def test_interleave_queue_single_requester():
    tracks = [{"title": str(i), "requester": "A"} for i in range(3)]
    out = list(interleave_queue(tracks))
    assert len(out) == 3
    assert [t["requester"] for t in out] == ["A", "A", "A"]


# ── make_progress_bar ───────────────────────────────────────────────────────

def test_progress_bar_zero():
    assert make_progress_bar(0, 100, width=10) == "░" * 10
    assert make_progress_bar(0, 0,   width=10) == "░" * 10
    assert make_progress_bar(50, None, width=10) == "░" * 10

def test_progress_bar_half():
    bar = make_progress_bar(50, 100, width=10)
    assert bar.count("█") == 5
    assert bar.count("░") == 5

def test_progress_bar_full():
    bar = make_progress_bar(100, 100, width=10)
    assert bar == "█" * 10

def test_progress_bar_over():
    bar = make_progress_bar(200, 100, width=10)
    assert bar == "█" * 10
