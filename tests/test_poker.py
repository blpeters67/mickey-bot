"""Unit tests for poker hand evaluation and pot splitting."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We import the cog module purely for its pure functions. discord.py is
# imported by the cog module at top-level, so we have to install it for tests
# to run — but no Discord connection is made.
from cogs.games.poker import best_hand, compute_pots, eval_five


def _c(rank, suit):
    return {"rank": rank, "suit": suit}


# ── eval_five — hand ranks ──────────────────────────────────────────────────

def test_high_card():
    h = [_c("A", "♠"), _c("Q", "♥"), _c("9", "♦"), _c("5", "♣"), _c("2", "♠")]
    assert eval_five(h)[0] == 0

def test_one_pair():
    h = [_c("A", "♠"), _c("A", "♥"), _c("9", "♦"), _c("5", "♣"), _c("2", "♠")]
    assert eval_five(h)[0] == 1

def test_two_pair():
    h = [_c("A", "♠"), _c("A", "♥"), _c("9", "♦"), _c("9", "♣"), _c("2", "♠")]
    assert eval_five(h)[0] == 2

def test_three_of_a_kind():
    h = [_c("A", "♠"), _c("A", "♥"), _c("A", "♦"), _c("9", "♣"), _c("2", "♠")]
    assert eval_five(h)[0] == 3

def test_straight():
    h = [_c("9", "♠"), _c("10","♥"), _c("J", "♦"), _c("Q", "♣"), _c("K", "♠")]
    assert eval_five(h)[0] == 4

def test_wheel_straight_A2345():
    """The wheel: A counts as low. Subtle case — easy to get wrong."""
    h = [_c("A", "♠"), _c("2", "♥"), _c("3", "♦"), _c("4", "♣"), _c("5", "♠")]
    assert eval_five(h)[0] == 4

def test_flush():
    h = [_c("A", "♠"), _c("9", "♠"), _c("7", "♠"), _c("4", "♠"), _c("2", "♠")]
    assert eval_five(h)[0] == 5

def test_full_house():
    h = [_c("A", "♠"), _c("A", "♥"), _c("A", "♦"), _c("9", "♣"), _c("9", "♠")]
    assert eval_five(h)[0] == 6

def test_four_of_a_kind():
    h = [_c("A", "♠"), _c("A", "♥"), _c("A", "♦"), _c("A", "♣"), _c("9", "♠")]
    assert eval_five(h)[0] == 7

def test_straight_flush():
    h = [_c("9", "♠"), _c("10","♠"), _c("J", "♠"), _c("Q", "♠"), _c("K", "♠")]
    assert eval_five(h)[0] == 8


# ── eval_five — kickers ─────────────────────────────────────────────────────

def test_pair_kicker_breaks_tie():
    """Pair of 5s + A kicker > pair of 5s + K kicker."""
    a = [_c("5", "♠"), _c("5", "♥"), _c("A", "♦"), _c("9", "♣"), _c("2", "♠")]
    b = [_c("5", "♠"), _c("5", "♥"), _c("K", "♦"), _c("9", "♣"), _c("2", "♠")]
    assert eval_five(a) > eval_five(b)


# ── best_hand — picks the best 5 from 7 ─────────────────────────────────────

def test_best_hand_finds_flush_in_seven():
    """5 spades among 7 cards => flush."""
    seven = [
        _c("A", "♠"), _c("K", "♠"), _c("Q", "♠"), _c("J", "♠"), _c("9", "♠"),
        _c("2", "♥"), _c("3", "♦"),
    ]
    assert best_hand(seven)[0] == 5

def test_best_hand_finds_full_house_in_seven():
    seven = [
        _c("A", "♠"), _c("A", "♥"), _c("A", "♦"),
        _c("K", "♣"), _c("K", "♠"),
        _c("3", "♦"), _c("2", "♣"),
    ]
    assert best_hand(seven)[0] == 6

def test_best_hand_finds_quads_when_possible():
    """7 cards with quads — make sure we don't get distracted by lower hands."""
    seven = [
        _c("A", "♠"), _c("A", "♥"), _c("A", "♦"), _c("A", "♣"),
        _c("K", "♠"), _c("K", "♥"),
        _c("3", "♦"),
    ]
    assert best_hand(seven)[0] == 7

def test_best_hand_finds_straight_flush_over_quads():
    """Should pick straight flush (8) over four-of-a-kind (7)."""
    seven = [
        _c("9", "♠"), _c("10","♠"), _c("J", "♠"), _c("Q", "♠"), _c("K", "♠"),
        _c("K", "♥"), _c("K", "♦"),
    ]
    assert best_hand(seven)[0] == 8


# ── compute_pots — main/side pot construction ───────────────────────────────

def _p(pid, total_bet, folded=False):
    return {"id": pid, "total_bet": total_bet, "folded": folded}


def test_pots_single_tier_when_bets_equal():
    pots = compute_pots([_p(1, 100), _p(2, 100), _p(3, 100)])
    assert pots == [(300, {1, 2, 3})]


def test_pots_short_all_in_creates_side_pot():
    """Short stack all-in for 50 vs two players at 200: main pot 150 for
    everyone, side pot 300 only for the deep stacks."""
    pots = compute_pots([_p(1, 50), _p(2, 200), _p(3, 200)])
    assert pots == [(150, {1, 2, 3}), (300, {2, 3})]


def test_pots_uncalled_excess_returns_to_bettor():
    """A bets 500, B can only call 300 all-in — the uncalled 200 forms a
    'side pot' only A is eligible for, i.e. it comes straight back."""
    pots = compute_pots([_p(1, 500), _p(2, 300)])
    assert pots == [(600, {1, 2}), (200, {1})]


def test_pots_folded_money_stays_in_but_never_wins():
    """A folded player's chips are dead money in the tiers they reached."""
    pots = compute_pots([_p(1, 100), _p(2, 100), _p(3, 60, folded=True)])
    assert pots == [(260, {1, 2})]


def test_pots_folded_deep_stack_excess_merges_down():
    """Deep stack bet 300 then folded: their excess above the callers' 100
    has no eligible winner of its own, so it merges into the pot below."""
    pots = compute_pots([_p(1, 100), _p(2, 100), _p(3, 300, folded=True)])
    assert pots == [(500, {1, 2})]


def test_pots_three_way_staircase():
    """Three different all-in levels → three tiers with shrinking eligibility."""
    pots = compute_pots([_p(1, 50), _p(2, 150), _p(3, 400)])
    assert pots == [(150, {1, 2, 3}), (200, {2, 3}), (250, {3})]


def test_pots_conserve_total_chips():
    players = [_p(1, 37), _p(2, 512), _p(3, 512), _p(4, 90, folded=True)]
    pots = compute_pots(players)
    assert sum(a for a, _ in pots) == sum(p["total_bet"] for p in players)


def test_pots_empty_when_no_contributions():
    assert compute_pots([_p(1, 0), _p(2, 0)]) == []
