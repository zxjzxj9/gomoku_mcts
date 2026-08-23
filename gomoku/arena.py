"""Head-to-head play and rating estimation.

Every pairing is colour-paired: an opening is sampled once and played twice,
with the two players swapping colours. Gomoku's first-player advantage is
large, so without pairing a rating would mostly measure how often a player
drew black.

Ratings come from a logistic (Bradley-Terry) fit by gradient ascent, with a
weak Gaussian prior that keeps an undefeated player's rating finite, and the
heuristic bot pinned at a nominal 1200 so the numbers stay comparable across
runs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from gomoku.difficulty import LEVELS, Level, make_player
from gomoku.evaluator import Evaluator
from gomoku.game import GameState
from gomoku.players import HeuristicPlayer, Player
from gomoku.selfplay import SelfPlayConfig, random_opening

log = logging.getLogger(__name__)

ANCHOR_NAME = "heuristic"
ANCHOR_RATING = 1200.0
LOG10_OVER_400 = np.log(10.0) / 400.0


@dataclasses.dataclass(frozen=True)
class MatchConfig:
    size: int = 9
    win_length: int = 5
    games_per_pair: int = 20
    opening_plies: tuple[int, int] = (2, 4)
    opening_radius: int = 2

    def opening_config(self) -> SelfPlayConfig:
        return SelfPlayConfig(
            size=self.size,
            win_length=self.win_length,
            opening_plies=self.opening_plies,
            opening_radius=self.opening_radius,
        )


def _play_one(black: Player, white: Player, state: GameState) -> int:
    players = {1: black, 2: white}
    while not state.is_terminal():
        state = state.play(players[state.to_play].select_move(state))
    return state.winner


def play_pair(
    player_a: Player,
    player_b: Player,
    config: MatchConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Two games from one shared opening, with the colours swapped."""
    opening_state, _ = random_opening(rng, config.opening_config())
    points_a = points_b = 0.0
    for black, white in ((player_a, player_b), (player_b, player_a)):
        winner = _play_one(black, white, opening_state)
        if winner == 0:
            points_a += 0.5
            points_b += 0.5
        else:
            victor = black if winner == 1 else white
            if victor is player_a:
                points_a += 1.0
            else:
                points_b += 1.0
    return points_a, points_b


def play_match(
    player_a: Player,
    player_b: Player,
    config: MatchConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Play `games_per_pair` games, rounded up to a whole number of pairs."""
    n_pairs = (config.games_per_pair + 1) // 2
    totals = np.zeros(2)
    for _ in range(n_pairs):
        totals += play_pair(player_a, player_b, config, rng)
    return float(totals[0]), float(totals[1])


def round_robin(
    players: dict[str, Player],
    config: MatchConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Score matrix `S` where `S[i, j]` is the points `i` took from `j`."""
    names = list(players)
    scores = np.zeros((len(names), len(names)))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names[i + 1 :], start=i + 1):
            points_i, points_j = play_match(players[name_i], players[name_j],
                                            config, rng)
            scores[i, j] = points_i
            scores[j, i] = points_j
            log.info("%s %.1f - %.1f %s", name_i, points_i, points_j, name_j)
    return scores


def fit_ratings(
    scores: np.ndarray,
    names: list[str],
    anchor: str,
    anchor_rating: float = ANCHOR_RATING,
    prior_sigma: float = 400.0,
    iterations: int = 3000,
    step: float = 8.0,
) -> dict[str, float]:
    """Maximum-likelihood Bradley-Terry ratings, anchored and regularised."""
    if anchor not in names:
        raise KeyError(f"anchor {anchor!r} is not among {names}")
    anchor_index = names.index(anchor)
    ratings = np.full(len(names), 1500.0)
    for _ in range(iterations):
        difference = ratings[None, :] - ratings[:, None]      # r_j - r_i
        expected = 1.0 / (1.0 + np.power(10.0, difference / 400.0))  # p_ij
        gradient = LOG10_OVER_400 * (
            (scores * (1.0 - expected)).sum(axis=1)
            - (scores.T * expected).sum(axis=1)
        )
        gradient -= (ratings - 1500.0) / (prior_sigma**2)
        ratings += step * gradient
    ratings += anchor_rating - ratings[anchor_index]
    return {name: float(rating) for name, rating in zip(names, ratings)}


def write_elo(path: str | Path, ratings: dict[str, float], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"ratings": ratings, "metadata": metadata}, indent=2))
    tmp.replace(path)


def measure_levels(
    evaluator: Evaluator,
    config: MatchConfig,
    rng: np.random.Generator,
    elo_path: str | Path,
    levels: tuple[Level, ...] = LEVELS,
) -> dict[str, float]:
    """Run the ladder against the anchor and fit ratings for every level."""
    players: dict[str, Player] = {
        level.key: make_player(level, evaluator, rng) for level in levels
    }
    players[ANCHOR_NAME] = HeuristicPlayer(rng, name=ANCHOR_NAME)
    scores = round_robin(players, config, rng)
    ratings = fit_ratings(scores, list(players), anchor=ANCHOR_NAME)
    write_elo(
        elo_path,
        ratings,
        {
            "games_per_pair": config.games_per_pair,
            "size": config.size,
            "win_length": config.win_length,
            "anchor": ANCHOR_NAME,
            "anchor_rating": ANCHOR_RATING,
        },
    )
    return ratings
