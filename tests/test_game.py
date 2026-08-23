import numpy as np
import pytest

from gomoku.board import BLACK, WHITE
from gomoku.game import N_PLANES, GameState


def test_new_game_black_to_play_and_not_terminal():
    s = GameState.new()
    assert s.to_play == BLACK
    assert s.last_move is None
    assert s.winner is None
    assert not s.is_terminal()
    assert s.ply == 0
    assert len(s.legal_moves()) == 81


def test_play_returns_new_state_and_leaves_original_untouched():
    s = GameState.new()
    t = s.play(40)
    assert t is not s
    assert s.board.grid[4, 4] == 0
    assert t.board.grid[4, 4] == BLACK
    assert t.to_play == WHITE
    assert t.last_move == 40
    assert t.ply == 1


def test_illegal_move_raises():
    s = GameState.new().play(40)
    with pytest.raises(ValueError):
        s.play(40)


def test_five_in_a_row_ends_the_game_with_that_player_as_winner():
    s = GameState.new()
    for black_move, white_move in zip([30, 31, 32, 33], [0, 1, 2, 3]):
        s = s.play(black_move).play(white_move)
    s = s.play(34)
    assert s.is_terminal()
    assert s.winner == BLACK
    assert s.legal_moves().size == 0


def test_full_board_without_a_line_is_a_draw():
    # 2x2 board cannot hold five in a row, so filling it must draw.
    s = GameState.new(size=2, win_length=5)
    for move in range(4):
        s = s.play(move)
    assert s.is_terminal()
    assert s.winner == 0


def test_result_for_reports_win_loss_and_draw():
    s = GameState.new()
    for black_move, white_move in zip([30, 31, 32, 33], [0, 1, 2, 3]):
        s = s.play(black_move).play(white_move)
    s = s.play(34)
    assert s.result_for(BLACK) == 1.0
    assert s.result_for(WHITE) == -1.0
    # After black wins it is white to move, so the side to move has lost.
    assert s.value_for_player_to_move() == -1.0


def test_encoding_shape_and_dtype():
    s = GameState.new()
    planes = s.encode()
    assert planes.shape == (N_PLANES, 9, 9)
    assert planes.dtype == np.float32


def test_encoding_is_relative_to_the_side_to_move():
    s = GameState.new().play(40)      # black at centre, now white to move
    planes = s.encode()
    assert planes[0].sum() == 0.0     # side to move (white) has no stones
    assert planes[1][4, 4] == 1.0     # opponent (black) stone is on plane 1
    t = s.play(0)                     # white at corner, now black to move
    planes = t.encode()
    assert planes[0][4, 4] == 1.0     # black's own stone is now on plane 0
    assert planes[1][0, 0] == 1.0


def test_encoding_marks_the_last_move_and_the_side_to_move():
    s = GameState.new().play(40)
    planes = s.encode()
    assert planes[2][4, 4] == 1.0
    assert planes[2].sum() == 1.0
    assert np.all(planes[3] == 0.0)   # white to move
    assert np.all(s.play(0).encode()[3] == 1.0)   # black to move


def test_encoding_of_a_fresh_board_has_no_last_move():
    planes = GameState.new().encode()
    assert planes[2].sum() == 0.0
    assert np.all(planes[3] == 1.0)


def test_board_size_and_win_length_propagate():
    s = GameState.new(size=3, win_length=3)
    assert s.size == 3
    assert s.encode().shape == (N_PLANES, 3, 3)
    s = s.play(0).play(3).play(1).play(4).play(2)
    assert s.is_terminal() and s.winner == BLACK
