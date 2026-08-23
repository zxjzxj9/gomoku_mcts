"""The Gomoku board: stone storage, legality, and win detection.

Win detection is incremental: only the four lines through the move just
played are scanned, so a check costs O(size) rather than O(size**2).
"""

from __future__ import annotations

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

# The four line orientations: horizontal, vertical, and both diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def other(player: int) -> int:
    """Return the opposing player."""
    return BLACK if player == WHITE else WHITE


class Board:
    """A square board of stones. Knows nothing about whose turn it is."""

    __slots__ = ("size", "win_length", "grid", "n_moves")

    def __init__(
        self,
        size: int = 9,
        win_length: int = 5,
        grid: np.ndarray | None = None,
    ) -> None:
        self.size = size
        self.win_length = win_length
        if grid is None:
            self.grid = np.zeros((size, size), dtype=np.int8)
        else:
            self.grid = np.ascontiguousarray(grid, dtype=np.int8)
        self.n_moves = int(np.count_nonzero(self.grid))

    def copy(self) -> "Board":
        new = Board.__new__(Board)
        new.size = self.size
        new.win_length = self.win_length
        new.grid = self.grid.copy()
        new.n_moves = self.n_moves
        return new

    def is_legal(self, move: int) -> bool:
        if move < 0 or move >= self.size * self.size:
            return False
        return bool(self.grid.reshape(-1)[move] == EMPTY)

    def legal_moves(self) -> np.ndarray:
        """Flat indices of every empty cell, ascending."""
        return np.flatnonzero(self.grid.reshape(-1) == EMPTY)

    def place(self, move: int, player: int) -> None:
        if not self.is_legal(move):
            raise ValueError(f"illegal move {move} on {self.size}x{self.size} board")
        self.grid.reshape(-1)[move] = player
        self.n_moves += 1

    def is_full(self) -> bool:
        return self.n_moves >= self.size * self.size

    def winning_line(self, move: int, player: int) -> list[int] | None:
        """Flat indices of a winning line through `move`, or None.

        Freestyle: a run of `win_length` or more counts, so an overline wins.
        """
        size = self.size
        row, col = divmod(move, size)
        grid = self.grid
        if grid[row, col] != player:
            return None
        for d_row, d_col in DIRECTIONS:
            cells = [(row, col)]
            for step in (1, -1):
                r, c = row + d_row * step, col + d_col * step
                while 0 <= r < size and 0 <= c < size and grid[r, c] == player:
                    cells.append((r, c))
                    r += d_row * step
                    c += d_col * step
            if len(cells) >= self.win_length:
                return sorted(r * size + c for r, c in cells)
        return None

    def is_win(self, move: int, player: int) -> bool:
        return self.winning_line(move, player) is not None
