"""Immutable game state: the board plus whose turn it is, and the encoding
handed to the network.

`play` returns a new state rather than mutating, because MCTS holds many
states alive at once and aliasing bugs there are near-impossible to find.
"""

from __future__ import annotations

import numpy as np

from gomoku.board import BLACK, WHITE, Board, other

# Planes: own stones, opponent stones, last move, side-to-move constant.
N_PLANES = 4

EMPTY_MOVES = np.empty(0, dtype=np.int64)


class GameState:
    __slots__ = ("board", "to_play", "last_move", "winner", "ply")

    def __init__(
        self,
        board: Board,
        to_play: int,
        last_move: int | None,
        winner: int | None,
        ply: int,
    ) -> None:
        self.board = board
        self.to_play = to_play
        self.last_move = last_move
        self.winner = winner
        self.ply = ply

    @classmethod
    def new(cls, size: int = 9, win_length: int = 5) -> "GameState":
        return cls(Board(size, win_length), BLACK, None, None, 0)

    @property
    def size(self) -> int:
        return self.board.size

    def is_terminal(self) -> bool:
        return self.winner is not None

    def legal_moves(self) -> np.ndarray:
        if self.is_terminal():
            return EMPTY_MOVES
        return self.board.legal_moves()

    def play(self, move: int) -> "GameState":
        if self.is_terminal():
            raise ValueError("cannot play in a finished game")
        board = self.board.copy()
        board.place(move, self.to_play)
        if board.is_win(move, self.to_play):
            winner = self.to_play
        elif board.is_full():
            winner = 0
        else:
            winner = None
        return GameState(board, other(self.to_play), move, winner, self.ply + 1)

    def encode(self) -> np.ndarray:
        size = self.size
        planes = np.zeros((N_PLANES, size, size), dtype=np.float32)
        grid = self.board.grid
        planes[0] = grid == self.to_play
        planes[1] = grid == other(self.to_play)
        if self.last_move is not None:
            planes[2].reshape(-1)[self.last_move] = 1.0
        if self.to_play == BLACK:
            planes[3] = 1.0
        return planes

    def result_for(self, player: int) -> float:
        """Final result from `player`'s view. Raises if the game is unfinished."""
        if self.winner is None:
            raise ValueError("game is not finished")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def value_for_player_to_move(self) -> float:
        return self.result_for(self.to_play)

    def __repr__(self) -> str:
        colour = "black" if self.to_play == BLACK else "white"
        return f"<GameState {self.size}x{self.size} ply={self.ply} {colour} to play>"
