"""Rendering a board to rich text.

Black is X, white is O. The cursor is bracketed, the last move is highlighted,
and a completed winning line is shown in reverse video.
"""

from __future__ import annotations

from rich.text import Text

from gomoku.board import BLACK, EMPTY, WHITE
from gomoku.game import GameState

_GLYPH = {EMPTY: ".", BLACK: "X", WHITE: "O"}
_COLUMN_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def board_text(
    state: GameState,
    cursor: int | None = None,
    winning_line: list[int] | None = None,
) -> Text:
    size = state.size
    win_cells = set(winning_line or ())
    text = Text()
    # Every cell renders as exactly three characters so that bracketing the
    # cursor cannot shift the grid out of alignment.
    text.append("   " + "".join(f" {c} " for c in _COLUMN_LABELS[:size]) + "\n",
                style="dim")
    for row in range(size):
        text.append(f"{row + 1:2d} ", style="dim")
        for col in range(size):
            move = row * size + col
            glyph = _GLYPH[int(state.board.grid[row, col])]
            if move in win_cells:
                style = "reverse bold green"
            elif move == state.last_move:
                style = "bold yellow"
            elif glyph == "X":
                style = "bold cyan"
            elif glyph == "O":
                style = "bold magenta"
            else:
                style = "dim"
            left, right = ("[", "]") if move == cursor else (" ", " ")
            text.append(left, style="bold")
            text.append(glyph, style=style)
            text.append(right, style="bold")
        text.append("\n")
    return text
