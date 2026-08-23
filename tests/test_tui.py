import numpy as np
import pytest

from gomoku.board import BLACK, WHITE
from gomoku.game import GameState
from gomoku.players import HeuristicPlayer
from gomoku.tui.app import GomokuApp
from gomoku.tui.render import board_text


def test_board_text_shows_stones_and_coordinates():
    s = GameState.new(size=5, win_length=5).play(12)
    text = board_text(s, cursor=None, winning_line=None).plain
    assert "X" in text            # black stone
    assert text.count("\n") >= 5  # header plus one line per row


def test_board_text_marks_the_cursor():
    s = GameState.new(size=5, win_length=5)
    plain = board_text(s, cursor=0, winning_line=None).plain
    assert "[" in plain and "]" in plain


@pytest.mark.asyncio
async def test_human_can_place_a_stone_with_the_keyboard():
    app = GomokuApp(black=None, white=None, size=5, win_length=5)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        assert app.state.ply == 1
        assert app.state.board.grid.reshape(-1)[app.cursor] == BLACK


@pytest.mark.asyncio
async def test_cursor_moves_with_the_arrow_keys():
    app = GomokuApp(black=None, white=None, size=5, win_length=5)
    async with app.run_test() as pilot:
        start = app.cursor
        await pilot.press("right")
        assert app.cursor == start + 1
        await pilot.press("down")
        assert app.cursor == start + 1 + 5


@pytest.mark.asyncio
async def test_placing_on_an_occupied_cell_is_rejected_without_crashing():
    app = GomokuApp(black=None, white=None, size=5, win_length=5)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.press("enter")
        assert app.state.ply == 1
        assert "occupied" in app.status.lower()


@pytest.mark.asyncio
async def test_new_game_resets_the_board():
    app = GomokuApp(black=None, white=None, size=5, win_length=5)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.press("n")
        assert app.state.ply == 0


@pytest.mark.asyncio
async def test_bot_replies_after_a_human_move():
    rng = np.random.default_rng(0)
    app = GomokuApp(black=None, white=HeuristicPlayer(rng), size=5, win_length=5)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.state.ply == 2
        assert app.state.to_play == BLACK
