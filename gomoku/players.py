"""Players: the move-selection interface plus two network-free implementations.

`HeuristicPlayer` scores a candidate cell by the threats it creates for the
mover and the threats it denies the opponent. It is deliberately simple and
deliberately fixed: the arena pins it at 1200 ELO as the ladder's anchor, so
changing its strength changes the meaning of every reported rating.
"""

from __future__ import annotations

import abc

import numpy as np

from gomoku.board import DIRECTIONS, EMPTY, Board, other
from gomoku.evaluator import Evaluator
from gomoku.game import GameState
from gomoku.mcts import MCTS, SearchConfig, run_search


class Player(abc.ABC):
    name: str

    @abc.abstractmethod
    def select_move(self, state: GameState) -> int:
        """Return a legal flat move index for `state`."""

    def reset(self) -> None:
        """Discard per-game state. Search players override this."""


class RandomPlayer(Player):
    def __init__(self, rng: np.random.Generator, name: str = "random") -> None:
        self.rng = rng
        self.name = name

    def select_move(self, state: GameState) -> int:
        return int(self.rng.choice(state.legal_moves()))


def candidate_moves(state: GameState, radius: int = 2) -> np.ndarray:
    """Empty cells within `radius` of an existing stone.

    Cells far from every stone are never useful in Gomoku, and pruning them
    keeps the heuristic fast enough to serve as an arena opponent.
    """
    grid = state.board.grid
    size = state.size
    if grid.any():
        occupied = grid != EMPTY
        near = _shifted_or(occupied, radius)
        cells = np.flatnonzero((near & ~occupied).reshape(-1))
        if cells.size:
            return cells
        return state.legal_moves()
    centre = (size // 2) * size + size // 2
    return np.array([centre], dtype=np.int64)


def _shifted_or(occupied: np.ndarray, radius: int) -> np.ndarray:
    """OR of `occupied` shifted by every offset within `radius`, without wrap."""
    size = occupied.shape[0]
    out = np.zeros_like(occupied)
    for d_row in range(-radius, radius + 1):
        for d_col in range(-radius, radius + 1):
            src_rows = slice(max(0, -d_row), size - max(0, d_row))
            dst_rows = slice(max(0, d_row), size - max(0, -d_row))
            src_cols = slice(max(0, -d_col), size - max(0, d_col))
            dst_cols = slice(max(0, d_col), size - max(0, -d_col))
            out[dst_rows, dst_cols] |= occupied[src_rows, src_cols]
    return out


def _run(grid: np.ndarray, row: int, col: int, d_row: int, d_col: int,
         player: int, size: int) -> tuple[int, int]:
    """Length of the run through (row, col) in one orientation, and how many
    of its two ends are empty."""
    count = 1
    open_ends = 0
    for step in (1, -1):
        r, c = row + d_row * step, col + d_col * step
        while 0 <= r < size and 0 <= c < size and grid[r, c] == player:
            count += 1
            r += d_row * step
            c += d_col * step
        if 0 <= r < size and 0 <= c < size and grid[r, c] == EMPTY:
            open_ends += 1
    return count, open_ends


def _pattern_score(count: int, open_ends: int, win_length: int) -> float:
    if count >= win_length:
        return 1_000_000.0
    if open_ends == 0:
        return 0.0
    if count == win_length - 1:
        return 100_000.0 if open_ends == 2 else 10_000.0
    if count == win_length - 2:
        return 5_000.0 if open_ends == 2 else 300.0
    if count == win_length - 3:
        return 200.0 if open_ends == 2 else 20.0
    return 10.0 if open_ends == 2 else 1.0


def move_score(board: Board, move: int, player: int) -> float:
    """Sum of the pattern scores this move creates for `player`."""
    row, col = divmod(move, board.size)
    grid = board.grid
    grid[row, col] = player
    try:
        total = 0.0
        for d_row, d_col in DIRECTIONS:
            count, open_ends = _run(grid, row, col, d_row, d_col, player, board.size)
            total += _pattern_score(count, open_ends, board.win_length)
        return total
    finally:
        grid[row, col] = EMPTY


class HeuristicPlayer(Player):
    """Threat-based rule bot. Fixed strength: it is the rating anchor."""

    # Denying the opponent is worth slightly less than building, so that a
    # winning move always outranks a block of equal nominal size.
    DEFENCE_WEIGHT = 0.9

    def __init__(self, rng: np.random.Generator, name: str = "heuristic",
                 radius: int = 2) -> None:
        self.rng = rng
        self.name = name
        self.radius = radius

    def select_move(self, state: GameState) -> int:
        board = state.board
        me = state.to_play
        opponent = other(me)
        moves = candidate_moves(state, self.radius)
        scores = np.array(
            [
                move_score(board, int(m), me)
                + self.DEFENCE_WEIGHT * move_score(board, int(m), opponent)
                for m in moves
            ]
        )
        best = np.flatnonzero(scores == scores.max())
        return int(moves[self.rng.choice(best)])


def sample_move(counts: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Choose a move from visit counts.

    Temperature 0 is a deterministic argmax; higher temperatures flatten the
    distribution. Cells with no visits are never chosen, so an illegal move
    cannot be produced.
    """
    if not np.any(counts > 0):
        raise ValueError("no visited moves to sample from")
    if temperature <= 0.0:
        return int(np.argmax(counts))
    weights = np.power(counts.astype(np.float64), 1.0 / temperature)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(counts))
    return int(rng.choice(len(weights), p=weights / total))


class MCTSPlayer(Player):
    """Plays by PUCT search, or by the raw policy when `policy_only` is set.

    Difficulty is entirely a matter of `simulations` and `temperature`; one
    checkpoint drives every level.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int,
        temperature: float = 0.0,
        policy_only: bool = False,
        config: SearchConfig | None = None,
        rng: np.random.Generator | None = None,
        name: str = "mcts",
        leaf_batch: int = 8,
    ) -> None:
        self.evaluator = evaluator
        self.simulations = simulations
        self.temperature = temperature
        self.policy_only = policy_only
        # Search noise belongs to training, not to play.
        self.config = config or SearchConfig(add_noise=False)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.name = name
        self.leaf_batch = leaf_batch

    def select_move(self, state: GameState) -> int:
        if self.policy_only or self.simulations <= 0:
            return self._policy_move(state)
        tree = MCTS(state, self.config, self.rng)
        run_search([tree], self.evaluator, self.simulations, self.leaf_batch)
        return sample_move(tree.visit_counts(), self.temperature, self.rng)

    def _policy_move(self, state: GameState) -> int:
        priors, _ = self.evaluator.evaluate(state.encode()[None])
        masked = np.zeros_like(priors[0])
        legal = state.legal_moves()
        masked[legal] = priors[0][legal]
        if masked.sum() <= 0:
            masked[legal] = 1.0
        return sample_move(masked, max(self.temperature, 1e-3), self.rng)
