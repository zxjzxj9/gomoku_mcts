"""The Textual application: board, status line, and key bindings.

Bot moves run on a worker thread so that a long search leaves the interface
responsive.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static

from gomoku.board import BLACK, WHITE
from gomoku.game import GameState
from gomoku.players import Player
from gomoku.tui.render import board_text


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
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        black: Player | None = None,
        white: Player | None = None,
        size: int = 9,
        win_length: int = 5,
    ) -> None:
        super().__init__()
        self.players: dict[int, Player | None] = {BLACK: black, WHITE: white}
        self.board_size = size
        self.win_length = win_length
        self.state = GameState.new(size, win_length)
        self.cursor = (size // 2) * size + size // 2
        self.status = "Your move."
        self.winning_line: list[int] | None = None
        self._thinking = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(id="board")
            yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_view()
        self.maybe_bot_move()

    def refresh_view(self) -> None:
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
        self.state = GameState.new(self.board_size, self.win_length)
        self.winning_line = None
        self.status = "New game."
        self.refresh_view()
        self.maybe_bot_move()

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
        player = self.players[self.state.to_play]
        move = player.select_move(self.state)
        self.call_from_thread(self.finish_bot_turn, move)

    def finish_bot_turn(self, move: int) -> None:
        self._thinking = False
        self.apply_move(move)
        self.refresh_view()
        self.maybe_bot_move()


def run_tui(
    black: Player | None = None,
    white: Player | None = None,
    size: int = 9,
    win_length: int = 5,
) -> None:
    GomokuApp(black, white, size, win_length).run()
