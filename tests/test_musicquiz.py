"""Unit tests for the music-quiz guess matcher — the fuzzy "did they name the
tune" logic that has to be lenient (typos, partial titles, noise words) without
accepting nonsense."""
from __future__ import annotations

from cogs.games.musicquiz import is_correct_guess, _candidates


_SONG = {
    "title": "Rick Astley - Never Gonna Give You Up (Official Video)",
    "artist": "Rick Astley",
    "song_title": None,
}


def test_exact_title_after_dash():
    assert is_correct_guess("never gonna give you up", _SONG)


def test_artist_counts():
    assert is_correct_guess("rick astley", _SONG)


def test_ignores_noise_words_and_case():
    assert is_correct_guess("Never Gonna Give You Up official video", _SONG)


def test_small_typo_accepted():
    assert is_correct_guess("never gona give you up", _SONG)


def test_partial_substring_accepted():
    # A solid chunk of the title should count.
    assert is_correct_guess("never gonna give", _SONG)


def test_wrong_answer_rejected():
    assert not is_correct_guess("bohemian rhapsody", _SONG)


def test_trivial_short_guess_rejected():
    assert not is_correct_guess("a", _SONG)


def test_candidates_include_both_sides_of_dash():
    cands = _candidates(_SONG)
    assert "rick astley" in cands
    assert "never gonna give you up" in cands
