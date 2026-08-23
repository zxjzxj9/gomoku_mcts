"""The Textual application: board, status line, and key bindings.

Bot moves run on a worker thread so that a long search leaves the interface
responsive.
"""

from __future__ import annotations

import numpy as np
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static

from gomoku.board import BLACK, WHITE
from gomoku.difficulty import Level, load_levels
from gomoku.engine import build_opponent
from gomoku.evaluator import Evaluator
from gomoku.game import GameState
from gomoku.players import Player
from gomoku.tui.render import board_text

# Distinguishes "caller passed None, meaning a human" from "caller said nothing".
_UNSET = object()


class GomokuApp(App):
    CSS = """
    Screen { align: center middle; }
    #board { padding: 1 2; }
    #status { padding: 0 2; height: 3; }
    """

    BINDINGS = [
        Binding("up", "move_cursor(-1, 0)", "Up"),
        Binding("down", "move_cursor(1, 0)", "Down"),
        Binding("left", "move_cursor(0, -1)", "Left"),
        Binding("right", "move_cursor(0, 1)", "Right"),
        Binding("enter,space", "place", "Place"),
        Binding("n", "new_game", "New game"),
        Binding("1", "set_level(1)", "L1"),
        Binding("2", "set_level(2)", "L2"),
        Binding("3", "set_level(3)", "L3"),
        Binding("4", "set_level(4)", "L4"),
        Binding("5", "set_level(5)", "L5"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        black: Player | None = _UNSET,
        white: Player | None = _UNSET,
        size: int = 9,
        win_length: int = 5,
        levels: tuple[Level, ...] | None = None,
        level_index: int = 3,
        evaluator: Evaluator | None = None,
        rng: np.random.Generator | None = None,
        mode: str = "human-vs-pc",
    ) -> None:
        super().__init__()
        # Textual's App.size is a read-only property, so the board dimension
        # cannot be stored as self.size.
        self.board_size = size
        self.win_length = win_length
        self.levels = levels if levels is not None else load_levels()
        self.level = self.levels[level_index - 1]
        self.evaluator = evaluator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mode = mode
        # An explicitly supplied player wins over the mode, and passing None
        # explicitly means "a human plays this colour".
        self.explicit_players = black is not _UNSET or white is not _UNSET
        if self.explicit_players:
            self.players: dict[int, Player | None] = {
                BLACK: None if black is _UNSET else black,
                WHITE: None if white is _UNSET else white,
            }
        else:
            self.players = self._players_for_mode()
        self.state = GameState.new(size, win_length)
        self.cursor = (size // 2) * size + size // 2
        self.status = "Your move."
        self.winning_line: list[int] | None = None
        self._thinking = False
        self._generation = 0

    def _players_for_mode(self) -> dict[int, Player | None]:
        opponent = build_opponent(self.level, self.evaluator, self.rng)
        if self.mode == "pc-vs-pc":
            return {
                BLACK: opponent,
                WHITE: build_opponent(self.level, self.evaluator, self.rng),
            }
        return {BLACK: None, WHITE: opponent}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(id="level")
            yield Static(id="board")
            yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_view()
        self.maybe_bot_move()

    def refresh_view(self) -> None:
        self.query_one("#level", Static).update(
            f"Level {self.level.label()}   [1-5 to change]"
        )
        show_cursor = self.players[self.state.to_play] is None
        self.query_one("#board", Static).update(
            board_text(
                self.state,
                self.cursor if show_cursor and not self.state.is_terminal() else None,
                self.winning_line,
            )
        )
        self.query_one("#status", Static).update(self.status)

    def action_move_cursor(self, d_row: int, d_col: int) -> None:
        row, col = divmod(self.cursor, self.board_size)
        row = min(self.board_size - 1, max(0, row + d_row))
        col = min(self.board_size - 1, max(0, col + d_col))
        self.cursor = row * self.board_size + col
        self.refresh_view()

    def action_place(self) -> None:
        if self.state.is_terminal():
            self.status = "Game over. Press n for a new game."
        elif self.players[self.state.to_play] is not None:
            self.status = "Not your turn."
        elif not self.state.board.is_legal(self.cursor):
            self.status = "That cell is occupied."
        else:
            self.apply_move(self.cursor)
            self.refresh_view()
            self.maybe_bot_move()
            return
        self.refresh_view()

    def action_new_game(self) -> None:
        self._generation += 1
        self.state = GameState.new(self.board_size, self.win_length)
        self.winning_line = None
        self.status = "New game."
        self.refresh_view()
        self.maybe_bot_move()

    def action_set_level(self, index: int) -> None:
        """Switch difficulty. Changing opponent mid-game would be unfair, so
        this starts a fresh game."""
        self.level = self.levels[index - 1]
        if not self.explicit_players:
            self.players = self._players_for_mode()
        self.action_new_game()

    def apply_move(self, move: int) -> None:
        mover = self.state.to_play
        self.state = self.state.play(move)
        if self.state.winner not in (None, 0):
            self.winning_line = self.state.board.winning_line(move, mover)
            self.status = f"{'Black' if mover == BLACK else 'White'} wins."
        elif self.state.winner == 0:
            self.status = "Draw."
        else:
            self.status = "Your move." if self.players[self.state.to_play] is None \
                else "Thinking..."

    def maybe_bot_move(self) -> None:
        """Start the engine thinking, unless it already is.

        `exclusive=True` is deliberately not used: `finish_bot_turn` chains the
        next move from inside the worker that is still completing, and an
        exclusive dispatch would cancel its own successor. The guard flag does
        the same job without that race.
        """
        player = self.players[self.state.to_play]
        if self._thinking or self.state.is_terminal() or player is None:
            return
        self._thinking = True
        self.run_worker(self.bot_turn, thread=True)

    def bot_turn(self) -> None:
        generation = self._generation
        state = self.state
        player = self.players[state.to_play]
        try:
            move = player.select_move(state)
        except Exception as error:
            # If select_move raises, call_from_thread below would never run
            # and _thinking would stay True forever, permanently freezing the
            # engine. Clear it and surface the failure instead.
            self.call_from_thread(self.fail_bot_turn, str(error), generation)
            return
        self.call_from_thread(self.finish_bot_turn, move, generation)

    def fail_bot_turn(self, message: str, generation: int) -> None:
        self._thinking = False
        if generation != self._generation:
            self.maybe_bot_move()
            return
        self.status = f"Engine failure: {message}"
        self.refresh_view()

    def finish_bot_turn(self, move: int, generation: int) -> None:
        self._thinking = False
        if generation != self._generation:
            # The game was reset (or otherwise replaced) while this move was
            # being computed. Discard it rather than apply a move that was
            # never legal in the current game, and let the new game's own
            # maybe_bot_move take over so a bot-vs-bot game does not stall.
            self.maybe_bot_move()
            return
        self.apply_move(move)
        self.refresh_view()
        self.maybe_bot_move()


def run_tui(
    size: int = 9,
    win_length: int = 5,
    levels: tuple[Level, ...] | None = None,
    level_index: int = 3,
    evaluator: Evaluator | None = None,
    rng: np.random.Generator | None = None,
    mode: str = "human-vs-pc",
) -> None:
    GomokuApp(
        size=size, win_length=win_length, levels=levels, level_index=level_index,
        evaluator=evaluator, rng=rng, mode=mode,
    ).run()
