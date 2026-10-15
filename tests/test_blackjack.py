"""Unit tests for blackjack hand value — the tricky bit is multi-ace handling."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogs.games.blackjack import bj_hand_value


def _c(rank):
    return {"rank": rank, "suit": "♠"}


def test_simple_hand():
    assert bj_hand_value([_c("5"), _c("6")]) == 11

def test_face_cards_count_as_ten():
    assert bj_hand_value([_c("J"), _c("Q")]) == 20
    assert bj_hand_value([_c("K"), _c("5")]) == 15

def test_ace_high_when_room():
    assert bj_hand_value([_c("A"), _c("9")]) == 20   # A=11
    assert bj_hand_value([_c("A"), _c("10")]) == 21  # blackjack

def test_ace_low_when_needed():
    assert bj_hand_value([_c("A"), _c("9"), _c("5")]) == 15   # 11+9+5=25 → A=1 → 15

def test_two_aces():
    """A,A → one ace high (11), one low (1) = 12."""
    assert bj_hand_value([_c("A"), _c("A")]) == 12

def test_two_aces_plus_nine():
    """The case the original handled correctly but is worth pinning:
    A=11, A=1, 9 = 21 (NOT 11+11+9=31, NOT 1+1+9=11)."""
    assert bj_hand_value([_c("A"), _c("A"), _c("9")]) == 21

def test_three_aces():
    """A,A,A — one high (11), two low (1) = 13."""
    assert bj_hand_value([_c("A"), _c("A"), _c("A")]) == 13

def test_four_aces():
    """All four aces — one high (11), three low (1) = 14."""
    assert bj_hand_value([_c("A"), _c("A"), _c("A"), _c("A")]) == 14

def test_bust():
    assert bj_hand_value([_c("K"), _c("Q"), _c("J")]) == 30

def test_soft_then_hard():
    """Start soft, then a high card forces ace to drop low."""
    # A + 6 = 17 (soft)
    assert bj_hand_value([_c("A"), _c("6")]) == 17
    # A + 6 + 10 = 17 (hard, ace dropped)
    assert bj_hand_value([_c("A"), _c("6"), _c("10")]) == 17
