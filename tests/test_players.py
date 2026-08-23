import numpy as np
import pytest

from gomoku.board import BLACK, WHITE
from gomoku.game import GameState
from gomoku.players import HeuristicPlayer, RandomPlayer, candidate_moves


def rng():
    return np.random.default_rng(7)


def state_with(moves, size=9, win_length=5):
    """Apply `moves` in order starting from a fresh game (black first)."""
    s = GameState.new(size=size, win_length=win_length)
    for m in moves:
        s = s.play(m)
    return s


def test_random_player_returns_a_legal_move():
    s = GameState.new()
    move = RandomPlayer(rng()).select_move(s)
    assert move in set(s.legal_moves().tolist())


def test_random_player_is_reproducible_for_a_given_seed():
    s = GameState.new()
    a = RandomPlayer(np.random.default_rng(3)).select_move(s)
    b = RandomPlayer(np.random.default_rng(3)).select_move(s)
    assert a == b


def test_candidate_moves_restricts_to_the_neighbourhood_of_stones():
    s = state_with([40])
    cands = set(candidate_moves(s, radius=1).tolist())
    assert 30 in cands and 41 in cands and 50 in cands
    assert 0 not in cands
    assert 40 not in cands  # occupied


def test_candidate_moves_on_an_empty_board_offers_the_centre():
    s = GameState.new()
    assert candidate_moves(s).tolist() == [40]


def test_heuristic_completes_its_own_five():
    # Black has four in a row at 30..33, so both 29 and 34 complete five.
    s = state_with([30, 0, 31, 1, 32, 2, 33, 3])
    assert s.to_play == BLACK
    assert HeuristicPlayer(rng()).select_move(s) in (29, 34)


def test_heuristic_blocks_the_opponents_open_four():
    # White to move; black threatens 34 (and 29). Either block is acceptable.
    s = state_with([30, 0, 31, 1, 32, 2, 33])
    assert s.to_play == WHITE
    assert HeuristicPlayer(rng()).select_move(s) in (29, 34)


def test_heuristic_prefers_winning_over_blocking():
    # Black to move has four at 30..33 and wins at 29 or 34.
    # White simultaneously has four at 60..63 and would win at 59 or 64.
    # Taking the win must outrank blocking.
    s = state_with([30, 60, 31, 61, 32, 62, 33, 63])
    assert s.to_play == BLACK
    assert HeuristicPlayer(rng()).select_move(s) in (29, 34)


def test_heuristic_blocks_an_open_three():
    # White to move; black has an open three at 31,32,33.
    s = state_with([31, 0, 32, 1, 33])
    move = HeuristicPlayer(rng()).select_move(s)
    assert move in (30, 34)


def test_heuristic_always_returns_a_legal_move_across_random_games():
    r = rng()
    player = HeuristicPlayer(r)
    s = GameState.new()
    while not s.is_terminal():
        move = player.select_move(s)
        assert move in set(s.legal_moves().tolist())
        s = s.play(move)


def test_heuristic_beats_random_decisively():
    wins = 0
    for game in range(20):
        r = np.random.default_rng(game)
        strong, weak = HeuristicPlayer(r), RandomPlayer(r)
        # Alternate who moves first so the result is not a colour artefact.
        players = [strong, weak] if game % 2 == 0 else [weak, strong]
        s = GameState.new()
        while not s.is_terminal():
            s = s.play(players[s.ply % 2].select_move(s))
        if s.winner != 0 and players[(s.ply - 1) % 2] is strong:
            wins += 1
    assert wins >= 17
