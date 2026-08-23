import numpy as np
import pytest

from gomoku.board import BLACK, EMPTY, WHITE, Board, other


def make(size=9, win_length=5):
    return Board(size=size, win_length=win_length)


def test_new_board_is_empty_and_all_moves_legal():
    b = make()
    assert b.grid.shape == (9, 9)
    assert b.grid.dtype == np.int8
    assert np.all(b.grid == EMPTY)
    assert len(b.legal_moves()) == 81
    assert b.n_moves == 0


def test_place_marks_cell_and_advances_count():
    b = make()
    b.place(40, BLACK)
    assert b.grid[4, 4] == BLACK
    assert b.n_moves == 1
    assert not b.is_legal(40)
    assert len(b.legal_moves()) == 80


def test_place_on_occupied_cell_raises():
    b = make()
    b.place(0, BLACK)
    with pytest.raises(ValueError):
        b.place(0, WHITE)


def test_place_off_board_raises():
    b = make()
    with pytest.raises(ValueError):
        b.place(81, BLACK)
    with pytest.raises(ValueError):
        b.place(-1, BLACK)


@pytest.mark.parametrize(
    "cells",
    [
        [30, 31, 32, 33, 34],           # horizontal
        [3, 12, 21, 30, 39],            # vertical
        [0, 10, 20, 30, 40],            # diagonal down-right
        [4, 12, 20, 28, 36],            # diagonal down-left
    ],
)
def test_five_in_a_row_wins_in_every_direction(cells):
    b = make()
    for c in cells[:-1]:
        b.place(c, BLACK)
    b.place(cells[-1], BLACK)
    assert b.is_win(cells[-1], BLACK)
    assert b.winning_line(cells[-1], BLACK) == sorted(cells)


def test_win_detected_when_last_move_is_in_the_middle_of_the_line():
    b = make()
    for c in [30, 31, 33, 34]:
        b.place(c, BLACK)
    b.place(32, BLACK)
    assert b.is_win(32, BLACK)


def test_overline_of_six_is_a_win_under_freestyle():
    b = make()
    for c in [30, 31, 32, 33, 34]:
        b.place(c, BLACK)
    b.place(35, BLACK)
    assert b.is_win(35, BLACK)


def test_line_does_not_wrap_across_rows():
    b = make()
    # Cells 7, 8 end row 0; cells 9, 10, 11 begin row 1. Not a real line.
    for c in [7, 8, 9, 10]:
        b.place(c, BLACK)
    b.place(11, BLACK)
    assert not b.is_win(11, BLACK)


def test_four_in_a_row_is_not_a_win():
    b = make()
    for c in [30, 31, 32]:
        b.place(c, BLACK)
    b.place(33, BLACK)
    assert not b.is_win(33, BLACK)
    assert b.winning_line(33, BLACK) is None


def test_opponent_stones_do_not_complete_a_line():
    b = make()
    for c in [30, 31, 33, 34]:
        b.place(c, BLACK)
    b.place(32, WHITE)
    assert not b.is_win(32, WHITE)
    assert not b.is_win(32, BLACK)


def test_is_full_and_draw_on_small_board():
    b = make(size=2, win_length=5)
    for i, m in enumerate(range(4)):
        b.place(m, BLACK if i % 2 == 0 else WHITE)
    assert b.is_full()
    assert len(b.legal_moves()) == 0


def test_win_length_is_configurable():
    b = make(size=3, win_length=3)
    b.place(0, BLACK)
    b.place(1, BLACK)
    b.place(2, BLACK)
    assert b.is_win(2, BLACK)


def test_copy_is_independent():
    b = make()
    b.place(0, BLACK)
    c = b.copy()
    c.place(1, WHITE)
    assert b.grid[0, 1] == EMPTY
    assert b.n_moves == 1
    assert c.n_moves == 2


def test_other_swaps_players():
    assert other(BLACK) == WHITE
    assert other(WHITE) == BLACK
