"""Self-play game generation.

Three details here exist specifically to counter Gomoku's first-player
advantage, and they are load-bearing:

* Games start from a random multi-ply opening near the centre. Played from
  the empty board, black simply wins, and the value target becomes a function
  of side-to-move rather than of the position. A random opening frequently
  hands white the advantage instead, which is what forces the value head to
  read the board.
* The move temperature is 1.0 for the opening plies and 0 afterwards, so
  early play stays varied without throwing away endgame accuracy.
* Policy targets are recorded only from moves searched at the full simulation
  budget. Cheap moves still advance the game -- they just do not teach.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from gomoku.evaluator import Evaluator
from gomoku.game import GameState
from gomoku.mcts import MCTS, SearchConfig, run_search
from gomoku.players import sample_move
from gomoku.symmetry import N_SYMMETRIES, transform_grid, transform_policy


@dataclasses.dataclass(frozen=True)
class SelfPlayConfig:
    size: int = 9
    win_length: int = 5
    opening_plies: tuple[int, int] = (2, 4)
    opening_radius: int = 2
    full_simulations: int = 600
    fast_simulations: int = 100
    full_fraction: float = 0.25
    temperature: float = 1.0
    temperature_plies: int | None = None
    games_in_flight: int = 32
    leaf_batch: int = 8
    search: SearchConfig = dataclasses.field(default_factory=SearchConfig)

    def temperature_cutoff(self) -> int:
        return self.size if self.temperature_plies is None else self.temperature_plies


@dataclasses.dataclass
class Sample:
    encoded: np.ndarray
    policy: np.ndarray
    value: float


@dataclasses.dataclass
class GameStats:
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    lengths: list[int] = dataclasses.field(default_factory=list)
    openings: set[tuple[int, ...]] = dataclasses.field(default_factory=set)

    @property
    def n_games(self) -> int:
        return self.black_wins + self.white_wins + self.draws

    @property
    def black_win_rate(self) -> float:
        return self.black_wins / self.n_games if self.n_games else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0


def random_opening(
    rng: np.random.Generator,
    config: SelfPlayConfig,
) -> tuple[GameState, tuple[int, ...]]:
    """A random legal opening of 2-4 plies drawn from cells near the centre."""
    low, high = config.opening_plies
    centre = config.size // 2
    span = config.opening_radius + 1
    rows = np.arange(max(0, centre - span), min(config.size, centre + span + 1))
    pool = np.array([r * config.size + c for r in rows for c in rows])
    while True:
        n_plies = int(rng.integers(low, high + 1))
        moves = rng.choice(pool, size=n_plies, replace=False)
        state = GameState.new(config.size, config.win_length)
        for move in moves:
            state = state.play(int(move))
            if state.is_terminal():
                break
        if not state.is_terminal():
            return state, tuple(int(m) for m in moves)


def augment(sample: Sample, size: int) -> list[Sample]:
    """The sample under all eight dihedral symmetries."""
    return [
        Sample(
            transform_grid(sample.encoded, k),
            transform_policy(sample.policy, k, size),
            sample.value,
        )
        for k in range(N_SYMMETRIES)
    ]


@dataclasses.dataclass
class _Record:
    encoded: np.ndarray
    policy: np.ndarray
    to_play: int


class _Game:
    def __init__(self, state: GameState, opening: tuple[int, ...],
                 config: SelfPlayConfig, rng: np.random.Generator) -> None:
        self.state = state
        self.opening = opening
        self.tree = MCTS(state, config.search, rng)
        self.records: list[_Record] = []


def play_games(
    evaluator: Evaluator,
    n_games: int,
    config: SelfPlayConfig,
    rng: np.random.Generator,
) -> tuple[list[Sample], GameStats]:
    """Generate `n_games` self-play games, returning augmented training samples."""
    samples: list[Sample] = []
    stats = GameStats()
    remaining = n_games
    while remaining > 0:
        batch_size = min(config.games_in_flight, remaining)
        remaining -= batch_size
        games = []
        for _ in range(batch_size):
            state, opening = random_opening(rng, config)
            games.append(_Game(state, opening, config, rng))
            stats.openings.add(opening)
        _run_batch(games, evaluator, config, rng)
        for game in games:
            samples.extend(_finish(game, stats, config))
    return samples, stats


def _run_batch(games, evaluator, config, rng) -> None:
    active = list(games)
    while active:
        use_full = rng.random(len(active)) < config.full_fraction
        for flag, simulations in ((True, config.full_simulations),
                                  (False, config.fast_simulations)):
            group = [g for g, f in zip(active, use_full) if bool(f) is flag]
            if group:
                run_search([g.tree for g in group], evaluator, simulations,
                           config.leaf_batch)
        still_active = []
        for game, full in zip(active, use_full):
            counts = game.tree.visit_counts()
            if counts.sum() <= 0:      # terminal position, nothing to play
                continue
            # The schedule counts plies since the opening, not total plies.
            plies_played = game.state.ply - len(game.opening)
            temperature = (
                config.temperature
                if plies_played < config.temperature_cutoff()
                else 0.0
            )
            move = sample_move(counts, temperature, rng)
            if full:
                game.records.append(
                    _Record(game.state.encode(), counts / counts.sum(),
                            game.state.to_play)
                )
            game.tree.advance(move)
            game.state = game.state.play(move)
            if not game.state.is_terminal():
                still_active.append(game)
        active = still_active


def _finish(game: _Game, stats: GameStats, config: SelfPlayConfig) -> list[Sample]:
    winner = game.state.winner
    if winner == 0:
        stats.draws += 1
    elif winner == 1:
        stats.black_wins += 1
    else:
        stats.white_wins += 1
    stats.lengths.append(game.state.ply)
    samples: list[Sample] = []
    for record in game.records:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == record.to_play else -1.0
        samples.extend(
            augment(Sample(record.encoded, record.policy, value), config.size)
        )
    return samples
