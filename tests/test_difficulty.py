import json

import numpy as np
import pytest

from gomoku.difficulty import LEVELS, Level, load_levels, make_player
from gomoku.evaluator import UniformEvaluator
from gomoku.game import GameState
from gomoku.players import MCTSPlayer, sample_move


def test_sample_move_at_zero_temperature_is_argmax():
    counts = np.array([1.0, 7.0, 3.0, 0.0])
    rng = np.random.default_rng(0)
    assert all(sample_move(counts, 0.0, rng) == 1 for _ in range(10))


def test_sample_move_never_picks_an_unvisited_cell():
    counts = np.array([0.0, 5.0, 0.0, 2.0])
    rng = np.random.default_rng(0)
    assert {sample_move(counts, 1.0, rng) for _ in range(200)} <= {1, 3}


def test_higher_temperature_spreads_the_choice():
    counts = np.array([1.0, 9.0])
    rng = np.random.default_rng(0)
    hot = [sample_move(counts, 1.0, rng) for _ in range(400)]
    cold = [sample_move(counts, 0.1, rng) for _ in range(400)]
    assert hot.count(0) > cold.count(0)


def test_sample_move_raises_when_nothing_was_visited():
    with pytest.raises(ValueError):
        sample_move(np.zeros(9), 1.0, np.random.default_rng(0))


def test_mcts_player_returns_a_legal_move():
    s = GameState.new(size=5, win_length=5)
    player = MCTSPlayer(UniformEvaluator(), simulations=32,
                        rng=np.random.default_rng(0))
    assert player.select_move(s) in set(s.legal_moves().tolist())


def test_mcts_player_finds_an_immediate_win():
    s = GameState.new()
    for black_move, white_move in zip([30, 31, 32, 33], [0, 1, 2, 3]):
        s = s.play(black_move).play(white_move)
    player = MCTSPlayer(UniformEvaluator(), simulations=200,
                        rng=np.random.default_rng(0))
    assert player.select_move(s) in (29, 34)


def test_policy_only_player_uses_no_simulations():
    s = GameState.new(size=5, win_length=5)
    evaluator = UniformEvaluator()
    player = MCTSPlayer(evaluator, simulations=0, policy_only=True,
                        rng=np.random.default_rng(0))
    assert player.select_move(s) in set(s.legal_moves().tolist())


def test_there_are_exactly_five_levels_numbered_one_to_five():
    assert len(LEVELS) == 5
    assert [level.index for level in LEVELS] == [1, 2, 3, 4, 5]


def test_simulation_budget_increases_with_level():
    budgets = [level.simulations for level in LEVELS]
    assert budgets == sorted(budgets)
    assert budgets[0] < budgets[-1]


def test_temperature_falls_with_level():
    temperatures = [level.temperature for level in LEVELS]
    assert temperatures == sorted(temperatures, reverse=True)
    assert temperatures[-1] == 0.0


def test_level_one_is_policy_only():
    assert LEVELS[0].policy_only
    assert not any(level.policy_only for level in LEVELS[1:])


def test_levels_are_unrated_without_an_elo_file():
    assert all(level.elo is None for level in load_levels(None))


def test_load_levels_reads_measured_ratings(tmp_path):
    path = tmp_path / "elo.json"
    path.write_text(json.dumps({"ratings": {
        "level1": 900, "level2": 1100, "level3": 1350,
        "level4": 1600, "level5": 1850, "heuristic": 1200}}))
    levels = load_levels(path)
    assert [level.elo for level in levels] == [900, 1100, 1350, 1600, 1850]


def test_load_levels_ignores_a_corrupt_elo_file(tmp_path):
    path = tmp_path / "elo.json"
    path.write_text("{not json")
    assert all(level.elo is None for level in load_levels(path))


def test_make_player_builds_a_usable_player_for_every_level():
    s = GameState.new(size=5, win_length=5)
    for level in LEVELS:
        player = make_player(level, UniformEvaluator(), np.random.default_rng(0))
        assert player.select_move(s) in set(s.legal_moves().tolist())
        assert str(level.index) in player.name
