import json

import numpy as np
import pytest

from gomoku.arena import (
    MatchConfig,
    fit_ratings,
    play_match,
    play_pair,
    round_robin,
    write_elo,
)
from gomoku.game import GameState
from gomoku.players import HeuristicPlayer, Player, RandomPlayer


class FixedPlayer(Player):
    """Plays the lowest-numbered legal cell. Deterministic and weak."""

    name = "fixed"

    def select_move(self, state: GameState) -> int:
        return int(state.legal_moves()[0])


def small_config(**kwargs):
    defaults = dict(size=5, win_length=4, games_per_pair=4, opening_plies=(2, 2))
    defaults.update(kwargs)
    return MatchConfig(**defaults)


def test_play_pair_awards_exactly_two_points():
    rng = np.random.default_rng(0)
    a, b = play_pair(RandomPlayer(rng), RandomPlayer(rng), small_config(), rng)
    assert a + b == pytest.approx(2.0)


def test_play_match_totals_the_game_count():
    rng = np.random.default_rng(0)
    config = small_config(games_per_pair=6)
    a, b = play_match(RandomPlayer(rng), RandomPlayer(rng), config, rng)
    assert a + b == pytest.approx(6.0)


def test_play_match_rounds_odd_game_counts_up_to_a_whole_pair():
    rng = np.random.default_rng(0)
    a, b = play_match(RandomPlayer(rng), RandomPlayer(rng),
                      small_config(games_per_pair=5), rng)
    assert a + b == pytest.approx(6.0)


def test_the_stronger_player_scores_more():
    rng = np.random.default_rng(0)
    config = small_config(games_per_pair=10)
    strong, weak = HeuristicPlayer(rng), FixedPlayer()
    strong_points, weak_points = play_match(strong, weak, config, rng)
    assert strong_points > weak_points


def test_round_robin_matrix_is_square_and_hollow():
    rng = np.random.default_rng(0)
    players = {"a": RandomPlayer(rng), "b": RandomPlayer(rng), "c": FixedPlayer()}
    scores = round_robin(players, small_config(), rng)
    assert scores.shape == (3, 3)
    assert np.all(np.diag(scores) == 0)


def test_round_robin_pairs_account_for_every_game():
    rng = np.random.default_rng(0)
    config = small_config(games_per_pair=4)
    players = {"a": RandomPlayer(rng), "b": FixedPlayer()}
    scores = round_robin(players, config, rng)
    assert scores[0, 1] + scores[1, 0] == pytest.approx(4.0)


def synthetic_scores(true_ratings, games=400):
    """Expected results implied by a set of true ratings."""
    n = len(true_ratings)
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            expected = 1.0 / (1.0 + 10 ** ((true_ratings[j] - true_ratings[i]) / 400))
            scores[i, j] = expected * games
    return scores


def test_fit_recovers_known_ratings():
    names = ["w", "x", "y", "z"]
    truth = [1000.0, 1200.0, 1400.0, 1700.0]
    ratings = fit_ratings(synthetic_scores(truth), names, anchor="x",
                          anchor_rating=1200.0)
    for name, expected in zip(names, truth):
        assert ratings[name] == pytest.approx(expected, abs=40)


def test_the_anchor_keeps_its_assigned_rating():
    names = ["a", "b"]
    ratings = fit_ratings(synthetic_scores([1500.0, 1300.0]), names,
                          anchor="a", anchor_rating=1200.0)
    assert ratings["a"] == pytest.approx(1200.0, abs=1e-6)
    assert ratings["b"] < ratings["a"]


def test_ratings_stay_finite_for_an_undefeated_player():
    scores = np.array([[0.0, 30.0], [0.0, 0.0]])
    ratings = fit_ratings(scores, ["winner", "loser"], anchor="loser",
                          anchor_rating=1200.0)
    assert np.isfinite(ratings["winner"])
    assert ratings["winner"] > ratings["loser"]


def test_fit_rejects_an_unknown_anchor():
    with pytest.raises(KeyError):
        fit_ratings(np.zeros((2, 2)), ["a", "b"], anchor="missing")


def test_write_elo_produces_a_file_difficulty_can_read(tmp_path):
    from gomoku.difficulty import load_levels

    path = tmp_path / "elo.json"
    ratings = {f"level{i}": 1000.0 + 200 * i for i in range(1, 6)}
    ratings["heuristic"] = 1200.0
    write_elo(path, ratings, {"games_per_pair": 20})
    payload = json.loads(path.read_text())
    assert payload["ratings"]["level3"] == 1600.0
    assert payload["metadata"]["games_per_pair"] == 20
    assert [level.elo for level in load_levels(path)] == [1200, 1400, 1600, 1800, 2000]
