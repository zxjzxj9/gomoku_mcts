# Gomoku MCTS/RL Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gomoku engine with a terminal UI, an AlphaZero-style self-play trainer running on Apple Silicon, and five difficulty levels labelled with measured ELO ratings.

**Architecture:** A numpy board and immutable `GameState` underpin everything. A PUCT MCTS searches using a policy/value ResNet, reached only through an `Evaluator` interface so the hardware backend can be swapped without touching search. Self-play feeds a replay buffer; training produces checkpoints; an arena measures ratings for the five difficulty presets, which are configuration over a single checkpoint.

**Tech Stack:** Python 3.12, PyTorch 2.9 (MPS backend), numpy, Textual, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-gomoku-mcts-design.md`

## Global Constraints

- Python 3.12; PyTorch 2.9 with the MPS backend. **No code path may require MPS** — every module falls back to CPU with a logged notice.
- Rules are **freestyle**: five *or more* in a row wins. An overline (six) is a win.
- Board size is a parameter everywhere, default `size=9`, `win_length=5`. Never hard-code 9, 15, or 5.
- Cells are addressed as **flat integer indices** `move = row * size + col` in every public API. Never pass `(row, col)` tuples across module boundaries.
- Player constants are `EMPTY = 0`, `BLACK = 1`, `WHITE = 2`, defined once in `gomoku/board.py` and imported everywhere else.
- All randomness goes through an explicitly passed `numpy.random.Generator`. No module-level `np.random` calls, no bare `random`. Every function that samples takes an `rng` parameter.
- Values are always from the perspective of the **side to move**: `+1` win, `-1` loss, `0` draw.
- Tests run with `pytest -q` from the repository root and must pass before every commit.

**Layout note.** Three modules split out from the spec's §4 sketch, each because
it has a distinct responsibility and a distinct set of consumers: `symmetry.py`
(the spec folded the dihedral transforms into `board.py`, but they are pure
index arithmetic used by self-play, not board state), `metrics.py` (diagnostics
consumed by training and by anyone inspecting a run), and `engine.py` (building
a playable opponent from a checkpoint, used by both the TUI and the CLI).
Everything else follows the spec's file list exactly.

---

### Task 1: Project scaffold and the board

**Files:**
- Create: `pyproject.toml`
- Create: `gomoku/__init__.py`
- Create: `gomoku/board.py`
- Test: `tests/test_board.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EMPTY: int`, `BLACK: int`, `WHITE: int`, `DIRECTIONS: tuple`, `other(player: int) -> int`, and `Board` with `__init__(self, size: int = 9, win_length: int = 5, grid: np.ndarray | None = None)`, attributes `size: int`, `win_length: int`, `grid: np.ndarray` (shape `(size, size)`, dtype `int8`), `n_moves: int`, and methods `copy() -> Board`, `is_legal(move: int) -> bool`, `legal_moves() -> np.ndarray`, `place(move: int, player: int) -> None`, `is_full() -> bool`, `winning_line(move: int, player: int) -> list[int] | None`, `is_win(move: int, player: int) -> bool`.

- [ ] **Step 1: Create the package scaffold**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gomoku"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26", "torch>=2.9", "textual>=0.80"]

[project.scripts]
gomoku = "gomoku.cli:main"

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
include = ["gomoku*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`gomoku/__init__.py`:

```python
"""Gomoku engine: board, search, training, and terminal UI."""
```

Then create an empty `tests/__init__.py` and install in editable mode:

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

- [ ] **Step 2: Write the failing tests**

`tests/test_board.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_board.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.board'`

- [ ] **Step 4: Implement the board**

`gomoku/board.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_board.py -q`
Expected: PASS, 17 passed (13 test functions, one of them parametrized four ways).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml gomoku/ tests/
git commit -m "feat: board with incremental freestyle win detection"
```

---

### Task 2: Dihedral symmetry

**Files:**
- Create: `gomoku/symmetry.py`
- Test: `tests/test_symmetry.py`

**Interfaces:**
- Consumes: nothing from Task 1 beyond numpy conventions.
- Produces: `N_SYMMETRIES: int` (= 8), `transform_grid(grid: np.ndarray, k: int) -> np.ndarray` (operates on the last two axes, so it accepts both `(size, size)` and `(C, size, size)`), `move_map(k: int, size: int) -> np.ndarray` mapping old flat index to new flat index, `transform_move(move: int, k: int, size: int) -> int`, `transform_policy(policy: np.ndarray, k: int, size: int) -> np.ndarray` for a flat `(size*size,)` vector, and `inverse(k: int) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_symmetry.py`:

```python
import numpy as np
import pytest

from gomoku.symmetry import (
    N_SYMMETRIES,
    inverse,
    move_map,
    transform_grid,
    transform_move,
    transform_policy,
)

SIZE = 4


@pytest.mark.parametrize("k", range(N_SYMMETRIES))
def test_transform_then_inverse_restores_grid(k):
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 3, size=(SIZE, SIZE)).astype(np.int8)
    restored = transform_grid(transform_grid(grid, k), inverse(k))
    assert np.array_equal(restored, grid)


@pytest.mark.parametrize("k", range(N_SYMMETRIES))
def test_move_follows_its_stone(k):
    """A stone placed at `move` must land on `transform_move(move, k)`."""
    for move in range(SIZE * SIZE):
        grid = np.zeros((SIZE, SIZE), dtype=np.int8)
        grid.reshape(-1)[move] = 1
        moved = transform_grid(grid, k)
        assert moved.reshape(-1)[transform_move(move, k, SIZE)] == 1
        assert moved.sum() == 1


@pytest.mark.parametrize("k", range(N_SYMMETRIES))
def test_move_map_is_a_permutation(k):
    mapping = move_map(k, SIZE)
    assert sorted(mapping.tolist()) == list(range(SIZE * SIZE))


@pytest.mark.parametrize("k", range(N_SYMMETRIES))
def test_policy_transform_preserves_mass_and_follows_moves(k):
    rng = np.random.default_rng(1)
    policy = rng.random(SIZE * SIZE)
    policy /= policy.sum()
    moved = transform_policy(policy, k, SIZE)
    assert moved.shape == policy.shape
    assert np.isclose(moved.sum(), 1.0)
    for move in range(SIZE * SIZE):
        assert np.isclose(moved[transform_move(move, k, SIZE)], policy[move])


def test_identity_is_a_no_op():
    grid = np.arange(SIZE * SIZE, dtype=np.int8).reshape(SIZE, SIZE)
    assert np.array_equal(transform_grid(grid, 0), grid)
    assert transform_move(7, 0, SIZE) == 7


def test_transform_handles_channel_dimension():
    planes = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    planes[1, 0, 1] = 1.0
    moved = transform_grid(planes, 3)
    assert moved.shape == planes.shape
    assert moved[1].sum() == 1.0
    assert moved[0].sum() == 0.0


def test_the_eight_symmetries_are_distinct():
    grid = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
    seen = {transform_grid(grid, k).tobytes() for k in range(N_SYMMETRIES)}
    assert len(seen) == N_SYMMETRIES
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_symmetry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.symmetry'`

- [ ] **Step 3: Implement the symmetry helpers**

`gomoku/symmetry.py`:

```python
"""The eight dihedral symmetries of a square board.

Index `k` encodes `k % 4` quarter-turns followed by a horizontal flip when
`k >= 4`. Gomoku is invariant under all eight, so every training sample can be
presented in eight equivalent forms.
"""

from __future__ import annotations

import functools

import numpy as np

N_SYMMETRIES = 8


def transform_grid(grid: np.ndarray, k: int) -> np.ndarray:
    """Apply symmetry `k` to the last two axes of `grid`."""
    out = np.rot90(grid, k % 4, axes=(-2, -1))
    if k >= 4:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


@functools.lru_cache(maxsize=None)
def move_map(k: int, size: int) -> np.ndarray:
    """Array `m` such that `m[old_index] == new_index` under symmetry `k`."""
    indices = np.arange(size * size).reshape(size, size)
    # transform_grid moves the value at `source[old]` to position `new`,
    # so transforming the index grid yields new -> old; invert it.
    new_to_old = transform_grid(indices, k).reshape(-1)
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(size * size)
    old_to_new.flags.writeable = False
    return old_to_new


def transform_move(move: int, k: int, size: int) -> int:
    return int(move_map(k, size)[move])


def transform_policy(policy: np.ndarray, k: int, size: int) -> np.ndarray:
    """Permute a flat `(size*size,)` policy vector under symmetry `k`."""
    out = np.empty_like(policy)
    out[move_map(k, size)] = policy
    return out


@functools.lru_cache(maxsize=None)
def inverse(k: int) -> int:
    """The symmetry undoing `k`. Reflections are self-inverse; rotations are not."""
    if k >= 4:
        return k
    return (-k) % 4
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_symmetry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gomoku/symmetry.py tests/test_symmetry.py
git commit -m "feat: dihedral symmetry transforms for boards, moves, and policies"
```

---

### Task 3: Game state and network encoding

**Files:**
- Create: `gomoku/game.py`
- Test: `tests/test_game.py`

**Interfaces:**
- Consumes: `gomoku.board.{Board, BLACK, WHITE, EMPTY, other}`.
- Produces: `N_PLANES: int` (= 4), and `GameState` with classmethod `new(size: int = 9, win_length: int = 5) -> GameState`, attributes `board: Board`, `to_play: int`, `last_move: int | None`, `winner: int | None` (`None` while ongoing, `0` for a draw, else `BLACK`/`WHITE`), `size: int`, `ply: int`, and methods `legal_moves() -> np.ndarray`, `is_terminal() -> bool`, `play(move: int) -> GameState` (returns a new state; never mutates), `encode() -> np.ndarray` of shape `(N_PLANES, size, size)` dtype `float32`, `result_for(player: int) -> float`, `value_for_player_to_move() -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/test_game.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_game.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.game'`

- [ ] **Step 3: Implement the game state**

`gomoku/game.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_game.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gomoku/game.py tests/test_game.py
git commit -m "feat: immutable game state with side-to-move relative encoding"
```

---

### Task 4: Players — the interface, random, and the heuristic anchor

**Files:**
- Create: `gomoku/players.py`
- Test: `tests/test_players.py`

**Interfaces:**
- Consumes: `gomoku.board.{BLACK, WHITE, EMPTY, DIRECTIONS, other}`, `gomoku.game.GameState`.
- Produces: abstract `Player` with `name: str` and `select_move(state: GameState) -> int`; `RandomPlayer(rng: np.random.Generator, name: str = "random")`; `HeuristicPlayer(rng: np.random.Generator, name: str = "heuristic", radius: int = 2)`; helper `candidate_moves(state: GameState, radius: int = 2) -> np.ndarray`; helper `move_score(board, move: int, player: int) -> float`.

`HeuristicPlayer` is the fixed rating anchor for the arena and the opponent the TUI uses before any network exists. It must therefore be deterministic given its `rng`.

- [ ] **Step 1: Write the failing tests**

`tests/test_players.py`:

```python
import numpy as np
import pytest

from gomoku.board import BLACK, WHITE
from gomoku.game import GameState
from gomoku.players import HeuristicPlayer, RandomPlayer, candidate_moves


def rng():
    return np.random.default_rng(7)


def state_with(moves, size=9, win_length=5):
    """Apply `moves` in order starting from a fresh game (black first)."""
    s = GameState.new(size=size, win_length=win_length)
    for m in moves:
        s = s.play(m)
    return s


def test_random_player_returns_a_legal_move():
    s = GameState.new()
    move = RandomPlayer(rng()).select_move(s)
    assert move in set(s.legal_moves().tolist())


def test_random_player_is_reproducible_for_a_given_seed():
    s = GameState.new()
    a = RandomPlayer(np.random.default_rng(3)).select_move(s)
    b = RandomPlayer(np.random.default_rng(3)).select_move(s)
    assert a == b


def test_candidate_moves_restricts_to_the_neighbourhood_of_stones():
    s = state_with([40])
    cands = set(candidate_moves(s, radius=1).tolist())
    assert 30 in cands and 41 in cands and 50 in cands
    assert 0 not in cands
    assert 40 not in cands  # occupied


def test_candidate_moves_on_an_empty_board_offers_the_centre():
    s = GameState.new()
    assert candidate_moves(s).tolist() == [40]


def test_heuristic_completes_its_own_five():
    # Black has four in a row at 30..33, so both 29 and 34 complete five.
    s = state_with([30, 0, 31, 1, 32, 2, 33, 3])
    assert s.to_play == BLACK
    assert HeuristicPlayer(rng()).select_move(s) in (29, 34)


def test_heuristic_blocks_the_opponents_open_four():
    # White to move; black threatens 34 (and 29). Either block is acceptable.
    s = state_with([30, 0, 31, 1, 32, 2, 33])
    assert s.to_play == WHITE
    assert HeuristicPlayer(rng()).select_move(s) in (29, 34)


def test_heuristic_prefers_winning_over_blocking():
    # Black to move has four at 30..33 and wins at 29 or 34.
    # White simultaneously has four at 60..63 and would win at 59 or 64.
    # Taking the win must outrank blocking.
    s = state_with([30, 60, 31, 61, 32, 62, 33, 63])
    assert s.to_play == BLACK
    assert HeuristicPlayer(rng()).select_move(s) in (29, 34)


def test_heuristic_blocks_an_open_three():
    # White to move; black has an open three at 31,32,33.
    s = state_with([31, 0, 32, 1, 33])
    move = HeuristicPlayer(rng()).select_move(s)
    assert move in (30, 34)


def test_heuristic_always_returns_a_legal_move_across_random_games():
    r = rng()
    player = HeuristicPlayer(r)
    s = GameState.new()
    while not s.is_terminal():
        move = player.select_move(s)
        assert move in set(s.legal_moves().tolist())
        s = s.play(move)


def test_heuristic_beats_random_decisively():
    wins = 0
    for game in range(20):
        r = np.random.default_rng(game)
        strong, weak = HeuristicPlayer(r), RandomPlayer(r)
        # Alternate who moves first so the result is not a colour artefact.
        players = [strong, weak] if game % 2 == 0 else [weak, strong]
        s = GameState.new()
        while not s.is_terminal():
            s = s.play(players[s.ply % 2].select_move(s))
        if s.winner != 0 and players[(s.ply - 1) % 2] is strong:
            wins += 1
    assert wins >= 17
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_players.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.players'`

- [ ] **Step 3: Implement the players**

`gomoku/players.py`:

```python
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
from gomoku.game import GameState


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_players.py -q`
Expected: PASS. If `test_heuristic_beats_random_decisively` fails, the pattern scores are miscalibrated — do not weaken the test; fix `_pattern_score`.

- [ ] **Step 5: Commit**

```bash
git add gomoku/players.py tests/test_players.py
git commit -m "feat: player interface with random and threat-heuristic bots"
```

---

### Task 5: Terminal UI

**Files:**
- Create: `gomoku/tui/__init__.py`
- Create: `gomoku/tui/render.py`
- Create: `gomoku/tui/app.py`
- Test: `tests/test_tui.py`

**Interfaces:**
- Consumes: `gomoku.game.GameState`, `gomoku.board.{BLACK, WHITE, EMPTY}`, `gomoku.players.{Player, HeuristicPlayer, RandomPlayer}`.
- Produces: `render.board_text(state: GameState, cursor: int | None, winning_line: list[int] | None) -> Text` (a `rich.text.Text`); `app.GomokuApp(black: Player | None, white: Player | None, size: int = 9, win_length: int = 5)` where `None` means "a human plays this colour", with attributes `state: GameState`, `cursor: int`, `status: str`, and methods `action_move_cursor(d_row, d_col)`, `action_place()`, `action_new_game()`; `app.run_tui(...) -> None`.

Bot moves run through Textual's `run_worker(..., thread=True)` so a deep search never freezes the interface.

- [ ] **Step 1: Write the failing tests**

`tests/test_tui.py`:

```python
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
```

Add `pytest-asyncio` and the async mode to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

Then `.venv/bin/pip install -e '.[dev]'` again.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tui.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.tui'`

- [ ] **Step 3: Implement rendering**

`gomoku/tui/__init__.py`:

```python
"""Terminal user interface."""
```

`gomoku/tui/render.py`:

```python
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
```

- [ ] **Step 4: Implement the application**

`gomoku/tui/app.py`:

```python
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
        self.size = size
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
        row, col = divmod(self.cursor, self.size)
        row = min(self.size - 1, max(0, row + d_row))
        col = min(self.size - 1, max(0, col + d_col))
        self.cursor = row * self.size + col
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
        self.state = GameState.new(self.size, self.win_length)
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tui.py -q`
Expected: PASS.

- [ ] **Step 6: Play it by hand**

Run: `.venv/bin/python -c "
import numpy as np
from gomoku.players import HeuristicPlayer
from gomoku.tui.app import run_tui
run_tui(black=None, white=HeuristicPlayer(np.random.default_rng(0)))
"`
Expected: a playable 9x9 board. Arrow keys move, Enter places, the bot replies, `q` quits.

- [ ] **Step 7: Commit**

```bash
git add gomoku/tui tests/test_tui.py pyproject.toml
git commit -m "feat: Textual TUI with human and bot players"
```

---

### Task 6: Policy/value network

**Files:**
- Create: `gomoku/net.py`
- Test: `tests/test_net.py`

**Interfaces:**
- Consumes: `gomoku.game.N_PLANES`.
- Produces: `select_device(prefer: str | None = None) -> torch.device`; `NetConfig` dataclass with fields `channels: int = 64`, `blocks: int = 6`, `in_planes: int = N_PLANES`; `PolicyValueNet(config: NetConfig)` with `forward(x: Tensor) -> tuple[Tensor, Tensor]` returning `(policy_logits (B, size*size), value (B,))`; `save_checkpoint(path, net, optimizer, generation, config, extra)` and `load_checkpoint(path, map_location) -> dict`.

The network is fully convolutional with a globally pooled value head, so one architecture serves any board size and a 9x9 checkpoint can be fine-tuned at 15x15.

- [ ] **Step 1: Write the failing tests**

`tests/test_net.py`:

```python
import torch

from gomoku.game import N_PLANES
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint, save_checkpoint, select_device


def small():
    return PolicyValueNet(NetConfig(channels=8, blocks=1))


def test_forward_shapes():
    net = small()
    logits, value = net(torch.zeros(3, N_PLANES, 9, 9))
    assert logits.shape == (3, 81)
    assert value.shape == (3,)


def test_value_is_bounded():
    net = small()
    _, value = net(torch.randn(16, N_PLANES, 9, 9) * 5)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_same_weights_accept_a_different_board_size():
    net = small()
    logits9, value9 = net(torch.zeros(1, N_PLANES, 9, 9))
    logits15, value15 = net(torch.zeros(1, N_PLANES, 15, 15))
    assert logits9.shape == (1, 81)
    assert logits15.shape == (1, 225)
    assert value15.shape == (1,)


def test_gradients_flow_to_both_heads():
    net = small()
    logits, value = net(torch.zeros(2, N_PLANES, 9, 9))
    (logits.sum() + value.sum()).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_select_device_falls_back_to_cpu_when_asked():
    assert select_device("cpu").type == "cpu"


def test_select_device_returns_something_usable():
    device = select_device()
    assert device.type in {"cpu", "mps", "cuda"}
    torch.zeros(1, device=device)


def test_checkpoint_round_trip(tmp_path):
    net = small()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    path = tmp_path / "ckpt.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, net, optimizer, generation=4, config=config,
                    extra={"samples": 100})
    payload = load_checkpoint(path, map_location="cpu")
    assert payload["generation"] == 4
    assert payload["config"]["channels"] == 8
    assert payload["extra"]["samples"] == 100

    restored = PolicyValueNet(NetConfig(**payload["config"]))
    restored.load_state_dict(payload["model"])
    net.eval()
    restored.eval()
    x = torch.randn(2, N_PLANES, 9, 9)
    with torch.no_grad():
        assert torch.allclose(net(x)[0], restored(x)[0], atol=1e-6)


def test_checkpoint_write_is_atomic(tmp_path):
    """No partial file is left behind under the final name."""
    net, path = small(), tmp_path / "ckpt.pt"
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    save_checkpoint(path, net, optimizer, 0, NetConfig(channels=8, blocks=1), {})
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_net.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.net'`

- [ ] **Step 3: Implement the network**

`gomoku/net.py`:

```python
"""The policy/value network and checkpoint I/O.

Fully convolutional, with the value head reduced by global average pooling
rather than a flatten, so a single set of weights accepts any board size.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import torch
from torch import Tensor, nn

from gomoku.game import N_PLANES

log = logging.getLogger(__name__)


def select_device(prefer: str | None = None) -> torch.device:
    """Pick a compute device. MPS when available, otherwise CPU.

    Nothing in this project requires MPS; the fallback is always usable.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    log.info("MPS unavailable; falling back to CPU")
    return torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class NetConfig:
    channels: int = 64
    blocks: int = 6
    in_planes: int = N_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        channels = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(self.config.blocks)]
        )
        # One logit per cell: a 1x1 convolution, so the output length follows
        # the input resolution automatically.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(self.stem(x))
        policy_logits = self.policy_head(features).flatten(1)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


def save_checkpoint(
    path: str | os.PathLike,
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    generation: int,
    config: NetConfig,
    extra: dict | None = None,
) -> None:
    """Write a checkpoint atomically, so an interrupted run never corrupts one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "generation": generation,
        "config": dataclasses.asdict(config),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_net.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gomoku/net.py tests/test_net.py
git commit -m "feat: size-agnostic policy/value network with atomic checkpoints"
```

---

### Task 7: The evaluator boundary

**Files:**
- Create: `gomoku/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `gomoku.net.{PolicyValueNet, select_device}`.
- Produces: abstract `Evaluator` with `evaluate(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, taking `(B, C, H, W)` float32 and returning `(B, H*W)` probabilities summing to 1 and `(B,)` values in `[-1, 1]`; `UniformEvaluator()` for tests; `NetEvaluator(net, device=None, max_batch=256)` with attribute `n_evaluated: int` and method `refresh(net)`.

Everything above this line is hardware-agnostic. `ServerEvaluator` in Task 16 implements the same three-line contract.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluator.py`:

```python
import numpy as np
import torch

from gomoku.evaluator import Evaluator, NetEvaluator, UniformEvaluator
from gomoku.game import N_PLANES, GameState
from gomoku.net import NetConfig, PolicyValueNet


def batch(n=3, size=9):
    return np.zeros((n, N_PLANES, size, size), dtype=np.float32)


def test_uniform_evaluator_returns_a_normalised_distribution():
    policies, values = UniformEvaluator().evaluate(batch())
    assert policies.shape == (3, 81)
    assert values.shape == (3,)
    assert np.allclose(policies.sum(axis=1), 1.0)
    assert np.allclose(values, 0.0)


def test_net_evaluator_shapes_and_normalisation():
    net = PolicyValueNet(NetConfig(channels=8, blocks=1))
    policies, values = NetEvaluator(net, device="cpu").evaluate(batch(5))
    assert policies.shape == (5, 81)
    assert values.shape == (5,)
    assert policies.dtype == np.float32
    assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(values >= -1) and np.all(values <= 1)


def test_net_evaluator_chunks_batches_larger_than_max_batch():
    net = PolicyValueNet(NetConfig(channels=8, blocks=1))
    evaluator = NetEvaluator(net, device="cpu", max_batch=4)
    policies, values = evaluator.evaluate(batch(10))
    assert policies.shape == (10, 81)
    assert evaluator.n_evaluated == 10


def test_net_evaluator_is_deterministic_and_does_not_track_gradients():
    net = PolicyValueNet(NetConfig(channels=8, blocks=1))
    evaluator = NetEvaluator(net, device="cpu")
    x = np.random.default_rng(0).random((4, N_PLANES, 9, 9)).astype(np.float32)
    first = evaluator.evaluate(x)
    second = evaluator.evaluate(x)
    assert np.allclose(first[0], second[0])
    assert all(p.grad is None for p in net.parameters())


def test_net_evaluator_handles_a_single_state_without_batchnorm_error():
    net = PolicyValueNet(NetConfig(channels=8, blocks=1))
    policies, values = NetEvaluator(net, device="cpu").evaluate(batch(1))
    assert policies.shape == (1, 81)


def test_net_evaluator_follows_the_board_size():
    net = PolicyValueNet(NetConfig(channels=8, blocks=1))
    policies, _ = NetEvaluator(net, device="cpu").evaluate(batch(2, size=5))
    assert policies.shape == (2, 25)


def test_evaluator_is_abstract():
    assert issubclass(UniformEvaluator, Evaluator)
    assert issubclass(NetEvaluator, Evaluator)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_evaluator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.evaluator'`

- [ ] **Step 3: Implement the evaluators**

`gomoku/evaluator.py`:

```python
"""The single boundary between tree search and compute hardware.

Search never imports torch. It hands an evaluator a batch of encoded states
and receives priors and values back. That is the whole contract, and it is
what lets the multiprocess evaluator in phase two be a drop-in replacement.
"""

from __future__ import annotations

import abc

import numpy as np
import torch

from gomoku.net import PolicyValueNet, select_device


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map `(B, C, H, W)` float32 states to `(B, H*W)` priors and `(B,)` values.

        Priors sum to one per row; values lie in [-1, 1] and are from the
        perspective of the side to move in each state.
        """


class UniformEvaluator(Evaluator):
    """Uniform priors and zero values. Used to test search in isolation."""

    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_states = encoded.shape[0]
        n_cells = encoded.shape[-1] * encoded.shape[-2]
        policies = np.full((n_states, n_cells), 1.0 / n_cells, dtype=np.float32)
        return policies, np.zeros(n_states, dtype=np.float32)


class NetEvaluator(Evaluator):
    """Runs the network in inference mode, chunked to a maximum batch size."""

    def __init__(
        self,
        net: PolicyValueNet,
        device: str | torch.device | None = None,
        max_batch: int = 256,
    ) -> None:
        self.device = select_device(device) if not isinstance(device, torch.device) \
            else device
        self.net = net.to(self.device).eval()
        self.max_batch = max_batch
        self.n_evaluated = 0

    def refresh(self, net: PolicyValueNet) -> None:
        """Swap in newer weights between generations."""
        self.net = net.to(self.device).eval()

    @torch.inference_mode()
    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policies: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for start in range(0, encoded.shape[0], self.max_batch):
            chunk = encoded[start : start + self.max_batch]
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
            logits, value = self.net(x)
            prior = torch.softmax(logits.float(), dim=1)
            policies.append(prior.cpu().numpy().astype(np.float32))
            values.append(value.float().cpu().numpy().astype(np.float32))
        self.n_evaluated += int(encoded.shape[0])
        return np.concatenate(policies, axis=0), np.concatenate(values, axis=0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_evaluator.py -q`
Expected: PASS. `eval()` mode is what makes the single-state batch-norm test pass; do not remove it.

- [ ] **Step 5: Commit**

```bash
git add gomoku/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluator interface with uniform and network implementations"
```

---

### Task 8: PUCT Monte Carlo tree search

**Files:**
- Create: `gomoku/mcts.py`
- Test: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `gomoku.game.GameState`, `gomoku.evaluator.Evaluator`.
- Produces: `SearchConfig` dataclass with `c_puct: float = 1.5`, `dirichlet_alpha: float = 0.15`, `dirichlet_epsilon: float = 0.25`, `virtual_loss: float = 1.0`, `add_noise: bool = True`; `MCTS(root_state, config=SearchConfig(), rng=None)` with attribute `simulations: int` and methods `collect(n_leaves: int) -> np.ndarray`, `apply(priors: np.ndarray, values: np.ndarray) -> None`, `visit_counts() -> np.ndarray` (length `size*size`, zero on illegal cells), `root_value() -> float`, `advance(move: int) -> None`; module functions `run_search(trees, evaluator, simulations, leaf_batch=8) -> None` and `search(state, evaluator, simulations, config=..., rng=None) -> np.ndarray`.

The split between `collect` and `apply` is what makes batching possible: a tree hands out several leaves at once under virtual loss, many trees pool their leaves into one network call, and the results are distributed back. All trees in one `run_search` must share a board size.

- [ ] **Step 1: Write the failing tests**

`tests/test_mcts.py`:

```python
import numpy as np
import pytest

from gomoku.board import BLACK, WHITE
from gomoku.evaluator import Evaluator, UniformEvaluator
from gomoku.game import GameState
from gomoku.mcts import MCTS, SearchConfig, run_search, search


def state_with(moves, size=9, win_length=5):
    s = GameState.new(size=size, win_length=win_length)
    for m in moves:
        s = s.play(m)
    return s


def quiet_config(**kwargs):
    defaults = dict(add_noise=False)
    defaults.update(kwargs)
    return SearchConfig(**defaults)


def test_search_finds_an_immediate_win():
    # Black has 30..33, so both 29 and 34 complete five.
    s = state_with([30, 0, 31, 1, 32, 2, 33, 3])
    counts = search(s, UniformEvaluator(), simulations=200, config=quiet_config())
    assert int(np.argmax(counts)) in (29, 34)


def test_search_blocks_an_immediate_loss():
    # White to move; black wins at 34 or 29 next. Any other move loses at once.
    s = state_with([30, 0, 31, 1, 32, 2, 33])
    counts = search(s, UniformEvaluator(), simulations=400, config=quiet_config())
    assert int(np.argmax(counts)) in (29, 34)


def test_visits_are_zero_on_occupied_cells():
    s = state_with([40, 30])
    counts = search(s, UniformEvaluator(), simulations=64, config=quiet_config())
    assert counts[40] == 0 and counts[30] == 0
    assert counts.sum() > 0


def test_visit_counts_total_the_simulation_budget():
    """The first simulation expands the root itself along an empty path, so it
    increments no root edge: N simulations leave N-1 root visits."""
    s = GameState.new(size=5, win_length=5)
    counts = search(s, UniformEvaluator(), simulations=100, config=quiet_config())
    assert counts.sum() == 99


def test_counts_length_matches_the_board():
    s = GameState.new(size=5, win_length=5)
    assert search(s, UniformEvaluator(), 20, quiet_config()).shape == (25,)


def test_terminal_root_produces_no_visits_and_still_terminates():
    s = state_with([30, 0, 31, 1, 32, 2, 33, 3, 34])
    assert s.is_terminal()
    tree = MCTS(s, quiet_config())
    run_search([tree], UniformEvaluator(), simulations=10)
    assert tree.visit_counts().sum() == 0
    assert tree.simulations == 10


def test_root_value_is_negative_when_the_side_to_move_is_lost():
    # White to move, black has an unstoppable double threat -- with a perfect
    # evaluator the root value must be clearly negative for white.
    class LossyEvaluator(Evaluator):
        def evaluate(self, encoded):
            n = encoded.shape[0]
            cells = encoded.shape[-1] * encoded.shape[-2]
            return (np.full((n, cells), 1.0 / cells, np.float32),
                    np.full(n, -1.0, np.float32))

    s = state_with([30, 0, 31, 1, 32])
    tree = MCTS(s, quiet_config())
    run_search([tree], LossyEvaluator(), simulations=50)
    assert tree.root_value() < 0


def test_statistics_stay_consistent_under_virtual_loss():
    s = GameState.new(size=5, win_length=5)
    tree = MCTS(s, quiet_config(virtual_loss=3.0))
    run_search([tree], UniformEvaluator(), simulations=200, leaf_batch=16)
    counts = tree.visit_counts()
    assert np.all(counts >= 0)
    assert counts.sum() == 199
    assert np.all(np.abs(tree.root.W) <= tree.root.N + 1e-6)


def test_leaf_batch_size_does_not_change_the_totals():
    s = GameState.new(size=5, win_length=5)
    totals = []
    for leaf_batch in (1, 4, 32):
        tree = MCTS(s, quiet_config())
        run_search([tree], UniformEvaluator(), 128, leaf_batch=leaf_batch)
        totals.append(tree.visit_counts().sum())
    assert totals == [127, 127, 127]


def test_dirichlet_noise_changes_root_priors_only():
    s = GameState.new(size=5, win_length=5)
    rng = np.random.default_rng(0)
    noisy = MCTS(s, SearchConfig(add_noise=True, dirichlet_epsilon=0.9), rng)
    quiet = MCTS(s, quiet_config())
    for tree in (noisy, quiet):
        run_search([tree], UniformEvaluator(), 40)
    assert not np.allclose(noisy.root.P, quiet.root.P)
    # A child one ply down sees the unmodified priors.
    child_noisy = next(c for c in noisy.root.children if c is not None and c.expanded)
    child_quiet = next(c for c in quiet.root.children if c is not None and c.expanded)
    assert np.allclose(child_noisy.P, child_quiet.P)


def test_noise_is_reproducible_for_a_seed():
    s = GameState.new(size=5, win_length=5)
    trees = [MCTS(s, SearchConfig(), np.random.default_rng(11)) for _ in range(2)]
    for tree in trees:
        run_search([tree], UniformEvaluator(), 30)
    assert np.allclose(trees[0].visit_counts(), trees[1].visit_counts())


def test_run_search_pools_leaves_from_several_trees():
    class CountingEvaluator(UniformEvaluator):
        def __init__(self):
            self.calls = 0
            self.max_batch_seen = 0

        def evaluate(self, encoded):
            self.calls += 1
            self.max_batch_seen = max(self.max_batch_seen, encoded.shape[0])
            return super().evaluate(encoded)

    s = GameState.new(size=5, win_length=5)
    trees = [MCTS(s, quiet_config()) for _ in range(4)]
    evaluator = CountingEvaluator()
    run_search(trees, evaluator, simulations=64, leaf_batch=8)
    assert evaluator.max_batch_seen > 8       # leaves from several trees at once
    assert all(t.visit_counts().sum() == 63 for t in trees)


def test_advance_reuses_the_subtree():
    s = GameState.new(size=5, win_length=5)
    tree = MCTS(s, quiet_config())
    run_search([tree], UniformEvaluator(), 100)
    move = int(np.argmax(tree.visit_counts()))
    visits_before = tree.visit_counts()[move]
    tree.advance(move)
    assert tree.root.state.last_move == move
    assert tree.root.N.sum() == pytest.approx(visits_before - 1)
    run_search([tree], UniformEvaluator(), 50)
    assert tree.root.state.ply == 1


def test_advance_rebases_the_simulation_counter_on_reused_visits():
    s = GameState.new(size=5, win_length=5)
    tree = MCTS(s, quiet_config())
    run_search([tree], UniformEvaluator(), 100)
    move = int(np.argmax(tree.visit_counts()))
    tree.advance(move)
    reused = int(tree.root.N.sum())
    assert tree.simulations == reused
    run_search([tree], UniformEvaluator(), 60)
    # The budget counts inherited visits, so the root totals 60 rather than
    # 60 on top of whatever was reused.
    assert tree.root.N.sum() == pytest.approx(max(60, reused), abs=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_mcts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.mcts'`

- [ ] **Step 3: Implement the search**

`gomoku/mcts.py`:

```python
"""PUCT Monte Carlo tree search with batched leaf evaluation.

A search round works in two halves. `collect` descends the tree several times,
applying a virtual loss on each edge it walks so that concurrent descents are
pushed apart, and returns the encoded leaves it found. The caller evaluates
them -- pooling leaves from many trees into one network call -- and hands the
results to `apply`, which removes the virtual losses and backs up the values.

Terminal leaves never reach the network: their value is known exactly and is
backed up inside `collect`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from gomoku.evaluator import Evaluator
from gomoku.game import GameState


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25
    virtual_loss: float = 1.0
    add_noise: bool = True


class Node:
    """One position, with per-edge statistics stored as parallel arrays."""

    __slots__ = ("state", "moves", "P", "N", "W", "children", "expanded", "pending")

    def __init__(self, state: GameState) -> None:
        self.state = state
        self.moves = state.legal_moves()
        n_edges = self.moves.shape[0]
        self.P = np.zeros(n_edges, dtype=np.float32)
        self.N = np.zeros(n_edges, dtype=np.float32)
        self.W = np.zeros(n_edges, dtype=np.float32)
        self.children: list[Node | None] = [None] * n_edges
        self.expanded = False
        self.pending = False


class MCTS:
    def __init__(
        self,
        root_state: GameState,
        config: SearchConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or SearchConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = Node(root_state)
        self.simulations = 0
        self._pending: list[tuple[Node, list[tuple[Node, int]]]] = []

    # -- collection -----------------------------------------------------

    def collect(self, n_leaves: int) -> np.ndarray:
        """Descend up to `n_leaves` times; return the leaves needing evaluation.

        Terminal leaves are backed up here and do not appear in the result, so
        the returned array may be shorter than `n_leaves` -- or empty.
        """
        encoded: list[np.ndarray] = []
        for _ in range(max(0, n_leaves)):
            found = self._descend()
            if found is None:
                break
            node, path, is_terminal = found
            if is_terminal:
                self._backup(path, node.state.value_for_player_to_move())
                self.simulations += 1
                continue
            node.pending = True
            self._pending.append((node, path))
            encoded.append(node.state.encode())
        if not encoded:
            return np.zeros((0,), dtype=np.float32)
        return np.stack(encoded)

    def _descend(self):
        node = self.root
        path: list[tuple[Node, int]] = []
        virtual_loss = self.config.virtual_loss
        while True:
            if node.state.is_terminal():
                return node, path, True
            if not node.expanded:
                if node.pending:
                    # Another descent this round already claimed this leaf.
                    self._undo_virtual_loss(path)
                    return None
                return node, path, False
            index = self._select(node)
            path.append((node, index))
            node.N[index] += virtual_loss
            node.W[index] -= virtual_loss
            child = node.children[index]
            if child is None:
                child = Node(node.state.play(int(node.moves[index])))
                node.children[index] = child
            node = child

    def _select(self, node: Node) -> int:
        total = node.N.sum()
        # Unvisited edges get Q = 0, the standard AlphaZero first-play urgency.
        q_values = np.where(node.N > 0, node.W / np.maximum(node.N, 1e-8), 0.0)
        exploration = self.config.c_puct * node.P * np.sqrt(max(total, 1e-8)) / (1.0 + node.N)
        return int(np.argmax(q_values + exploration))

    def _undo_virtual_loss(self, path: list[tuple[Node, int]]) -> None:
        virtual_loss = self.config.virtual_loss
        for node, index in path:
            node.N[index] -= virtual_loss
            node.W[index] += virtual_loss

    # -- application ----------------------------------------------------

    def apply(self, priors: np.ndarray, values: np.ndarray) -> None:
        """Expand the pending leaves with `priors` and back up `values`."""
        if len(self._pending) != len(priors):
            raise ValueError(
                f"expected {len(self._pending)} evaluations, got {len(priors)}"
            )
        pending, self._pending = self._pending, []
        for (node, path), prior, value in zip(pending, priors, values):
            node.pending = False
            if not node.expanded:
                legal_prior = prior[node.moves]
                total = legal_prior.sum()
                if total <= 0:
                    legal_prior = np.full_like(legal_prior, 1.0 / max(len(node.moves), 1))
                else:
                    legal_prior = legal_prior / total
                if node is self.root and self.config.add_noise:
                    legal_prior = self._with_noise(legal_prior)
                node.P = legal_prior.astype(np.float32)
                node.expanded = True
            self._backup(path, float(value))
            self.simulations += 1

    def _with_noise(self, prior: np.ndarray) -> np.ndarray:
        epsilon = self.config.dirichlet_epsilon
        noise = self.rng.dirichlet(np.full(len(prior), self.config.dirichlet_alpha))
        return (1.0 - epsilon) * prior + epsilon * noise

    def _backup(self, path: list[tuple[Node, int]], value: float) -> None:
        """Back up `value`, which is from the perspective of the leaf's mover.

        Each step up the tree swaps perspective. The virtual loss applied on
        the way down is removed here in the same arithmetic.
        """
        virtual_loss = self.config.virtual_loss
        current = value
        for node, index in reversed(path):
            current = -current
            node.N[index] += 1.0 - virtual_loss
            node.W[index] += current + virtual_loss

    # -- readout --------------------------------------------------------

    def visit_counts(self) -> np.ndarray:
        size = self.root.state.size
        counts = np.zeros(size * size, dtype=np.float32)
        counts[self.root.moves] = self.root.N
        return counts

    def root_value(self) -> float:
        total = self.root.N.sum()
        if total <= 0:
            return 0.0
        return float(self.root.W.sum() / total)

    def advance(self, move: int) -> None:
        """Re-root on `move`, keeping the subtree already searched.

        The simulation counter is rebased on the reused visits, so a caller
        asking for N simulations gets a root with N total visits rather than
        N fresh ones on top of the inherited subtree.
        """
        index = int(np.flatnonzero(self.root.moves == move)[0])
        child = self.root.children[index]
        self.root = child if child is not None else Node(self.root.state.play(move))
        self._pending.clear()
        self.root.pending = False
        self.simulations = int(self.root.N.sum())


def run_search(
    trees,
    evaluator: Evaluator,
    simulations: int,
    leaf_batch: int = 8,
) -> None:
    """Run every tree to `simulations`, pooling leaves into shared network calls.

    All trees must share a board size, since their leaves are concatenated.
    """
    trees = list(trees)
    while True:
        batches: list[np.ndarray] = []
        owners: list[tuple[MCTS, int]] = []
        for tree in trees:
            remaining = simulations - tree.simulations
            if remaining <= 0:
                continue
            encoded = tree.collect(min(leaf_batch, remaining))
            if encoded.shape[0]:
                batches.append(encoded)
                owners.append((tree, encoded.shape[0]))
        if not owners:
            if all(tree.simulations >= simulations for tree in trees):
                return
            # Every remaining tree is fully terminal; nothing more to search.
            return
        priors, values = evaluator.evaluate(np.concatenate(batches, axis=0))
        offset = 0
        for tree, count in owners:
            tree.apply(priors[offset : offset + count], values[offset : offset + count])
            offset += count


def search(
    state: GameState,
    evaluator: Evaluator,
    simulations: int,
    config: SearchConfig | None = None,
    rng: np.random.Generator | None = None,
    leaf_batch: int = 8,
) -> np.ndarray:
    """Convenience wrapper for searching a single position."""
    tree = MCTS(state, config, rng)
    run_search([tree], evaluator, simulations, leaf_batch)
    return tree.visit_counts()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_mcts.py -q`
Expected: PASS. If `test_visit_counts_total_the_simulation_budget` fails by a small margin, the culprit is the virtual-loss arithmetic in `_backup`, not the test.

- [ ] **Step 5: Commit**

```bash
git add gomoku/mcts.py tests/test_mcts.py
git commit -m "feat: PUCT search with virtual loss and batched leaf evaluation"
```

---

### Task 9: Search player and the five difficulty levels

**Files:**
- Modify: `gomoku/players.py` (append `sample_move` and `MCTSPlayer`)
- Create: `gomoku/difficulty.py`
- Test: `tests/test_difficulty.py`

**Interfaces:**
- Consumes: `gomoku.mcts.{MCTS, SearchConfig, run_search}`, `gomoku.evaluator.Evaluator`, `gomoku.players.Player`.
- Produces: `players.sample_move(counts: np.ndarray, temperature: float, rng) -> int`; `players.MCTSPlayer(evaluator, simulations, temperature=0.0, policy_only=False, config=None, rng=None, name="mcts", leaf_batch=8)`; `difficulty.Level` frozen dataclass with `index: int`, `name: str`, `simulations: int`, `temperature: float`, `policy_only: bool`, `elo: int | None`; `difficulty.LEVELS: tuple[Level, ...]` (five entries, indices 1-5); `difficulty.load_levels(elo_path=None) -> tuple[Level, ...]`; `difficulty.make_player(level, evaluator, rng) -> Player`.

`elo` is `None` until Task 14's arena measures it. The UI must render "unrated" rather than invent a number.

- [ ] **Step 1: Write the failing tests**

`tests/test_difficulty.py`:

```python
import json

import numpy as np
import pytest

from gomoku.difficulty import LEVELS, Level, load_levels, make_player
from gomoku.evaluator import UniformEvaluator
from gomoku.game import GameState
from gomoku.players import MCTSPlayer, sample_move


def test_sample_move_at_zero_temperature_is_argmax():
    counts = np.array([1.0, 7.0, 3.0, 0.0])
    rng = np.random.default_rng(0)
    assert all(sample_move(counts, 0.0, rng) == 1 for _ in range(10))


def test_sample_move_never_picks_an_unvisited_cell():
    counts = np.array([0.0, 5.0, 0.0, 2.0])
    rng = np.random.default_rng(0)
    assert {sample_move(counts, 1.0, rng) for _ in range(200)} <= {1, 3}


def test_higher_temperature_spreads_the_choice():
    counts = np.array([1.0, 9.0])
    rng = np.random.default_rng(0)
    hot = [sample_move(counts, 1.0, rng) for _ in range(400)]
    cold = [sample_move(counts, 0.1, rng) for _ in range(400)]
    assert hot.count(0) > cold.count(0)


def test_sample_move_raises_when_nothing_was_visited():
    with pytest.raises(ValueError):
        sample_move(np.zeros(9), 1.0, np.random.default_rng(0))


def test_mcts_player_returns_a_legal_move():
    s = GameState.new(size=5, win_length=5)
    player = MCTSPlayer(UniformEvaluator(), simulations=32,
                        rng=np.random.default_rng(0))
    assert player.select_move(s) in set(s.legal_moves().tolist())


def test_mcts_player_finds_an_immediate_win():
    s = GameState.new()
    for black_move, white_move in zip([30, 31, 32, 33], [0, 1, 2, 3]):
        s = s.play(black_move).play(white_move)
    player = MCTSPlayer(UniformEvaluator(), simulations=200,
                        rng=np.random.default_rng(0))
    assert player.select_move(s) in (29, 34)


def test_policy_only_player_uses_no_simulations():
    s = GameState.new(size=5, win_length=5)
    evaluator = UniformEvaluator()
    player = MCTSPlayer(evaluator, simulations=0, policy_only=True,
                        rng=np.random.default_rng(0))
    assert player.select_move(s) in set(s.legal_moves().tolist())


def test_there_are_exactly_five_levels_numbered_one_to_five():
    assert len(LEVELS) == 5
    assert [level.index for level in LEVELS] == [1, 2, 3, 4, 5]


def test_simulation_budget_increases_with_level():
    budgets = [level.simulations for level in LEVELS]
    assert budgets == sorted(budgets)
    assert budgets[0] < budgets[-1]


def test_temperature_falls_with_level():
    temperatures = [level.temperature for level in LEVELS]
    assert temperatures == sorted(temperatures, reverse=True)
    assert temperatures[-1] == 0.0


def test_level_one_is_policy_only():
    assert LEVELS[0].policy_only
    assert not any(level.policy_only for level in LEVELS[1:])


def test_levels_are_unrated_without_an_elo_file():
    assert all(level.elo is None for level in load_levels(None))


def test_load_levels_reads_measured_ratings(tmp_path):
    path = tmp_path / "elo.json"
    path.write_text(json.dumps({"ratings": {
        "level1": 900, "level2": 1100, "level3": 1350,
        "level4": 1600, "level5": 1850, "heuristic": 1200}}))
    levels = load_levels(path)
    assert [level.elo for level in levels] == [900, 1100, 1350, 1600, 1850]


def test_load_levels_ignores_a_corrupt_elo_file(tmp_path):
    path = tmp_path / "elo.json"
    path.write_text("{not json")
    assert all(level.elo is None for level in load_levels(path))


def test_make_player_builds_a_usable_player_for_every_level():
    s = GameState.new(size=5, win_length=5)
    for level in LEVELS:
        player = make_player(level, UniformEvaluator(), np.random.default_rng(0))
        assert player.select_move(s) in set(s.legal_moves().tolist())
        assert str(level.index) in player.name
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_difficulty.py -q`
Expected: FAIL — `ImportError: cannot import name 'sample_move'`

- [ ] **Step 3: Append the search player to `gomoku/players.py`**

Add these imports at the top of the file:

```python
from gomoku.evaluator import Evaluator
from gomoku.mcts import MCTS, SearchConfig, run_search
```

Then append:

```python
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
```

- [ ] **Step 4: Implement the difficulty levels**

`gomoku/difficulty.py`:

```python
"""The five difficulty levels.

Every level runs the same checkpoint. Strength comes from the simulation
budget, and the residual randomness that keeps the weaker levels beatable
comes from the sampling temperature. ELO is measured by the arena and read
from disk; a level whose rating has not been measured reports None rather
than a guess.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from gomoku.evaluator import Evaluator
from gomoku.players import MCTSPlayer, Player

log = logging.getLogger(__name__)

DEFAULT_ELO_PATH = Path("runs/elo.json")


@dataclasses.dataclass(frozen=True)
class Level:
    index: int
    name: str
    simulations: int
    temperature: float
    policy_only: bool
    elo: int | None = None

    @property
    def key(self) -> str:
        return f"level{self.index}"

    def label(self) -> str:
        rating = f"{self.elo} ELO" if self.elo is not None else "unrated"
        return f"{self.index}. {self.name} ({rating})"


LEVELS: tuple[Level, ...] = (
    Level(1, "Beginner", simulations=0, temperature=1.0, policy_only=True),
    Level(2, "Casual", simulations=25, temperature=0.6, policy_only=False),
    Level(3, "Club", simulations=100, temperature=0.3, policy_only=False),
    Level(4, "Strong", simulations=400, temperature=0.0, policy_only=False),
    Level(5, "Expert", simulations=1600, temperature=0.0, policy_only=False),
)


def load_levels(elo_path: str | Path | None = DEFAULT_ELO_PATH) -> tuple[Level, ...]:
    """Return the levels, annotated with measured ratings when available."""
    if elo_path is None:
        return LEVELS
    path = Path(elo_path)
    if not path.exists():
        return LEVELS
    try:
        ratings = json.loads(path.read_text())["ratings"]
    except (ValueError, KeyError, OSError) as error:
        log.warning("ignoring unreadable ELO file %s: %s", path, error)
        return LEVELS
    return tuple(
        dataclasses.replace(level, elo=_as_int(ratings.get(level.key)))
        for level in LEVELS
    )


def _as_int(value) -> int | None:
    return None if value is None else int(round(float(value)))


def make_player(
    level: Level,
    evaluator: Evaluator,
    rng: np.random.Generator,
) -> Player:
    return MCTSPlayer(
        evaluator,
        simulations=level.simulations,
        temperature=level.temperature,
        policy_only=level.policy_only,
        rng=rng,
        name=f"level{level.index}",
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_difficulty.py -q`
Expected: PASS.

- [ ] **Step 6: Run the whole suite for regressions**

Run: `.venv/bin/pytest -q`
Expected: PASS, everything from Tasks 1-9.

- [ ] **Step 7: Commit**

```bash
git add gomoku/players.py gomoku/difficulty.py tests/test_difficulty.py
git commit -m "feat: MCTS player and the five difficulty levels"
```

---

### Task 10: Self-play with the first-player-advantage mitigations

**Files:**
- Create: `gomoku/selfplay.py`
- Test: `tests/test_selfplay.py`

**Interfaces:**
- Consumes: `gomoku.game.GameState`, `gomoku.mcts.{MCTS, SearchConfig, run_search}`, `gomoku.players.sample_move`, `gomoku.symmetry.{N_SYMMETRIES, transform_grid, transform_policy}`, `gomoku.evaluator.Evaluator`.
- Produces: `SelfPlayConfig` frozen dataclass with `size=9`, `win_length=5`, `opening_plies=(2, 4)`, `opening_radius=2`, `full_simulations=600`, `fast_simulations=100`, `full_fraction=0.25`, `temperature=1.0`, `temperature_plies=None`, `games_in_flight=32`, `leaf_batch=8`, `search=SearchConfig()`; `Sample` dataclass with `encoded: np.ndarray`, `policy: np.ndarray`, `value: float`; `GameStats` dataclass with `black_wins`, `white_wins`, `draws`, `lengths: list[int]`, `openings: set[tuple[int, ...]]`, and properties `n_games`, `black_win_rate`, `mean_length`; `random_opening(rng, config) -> tuple[GameState, tuple[int, ...]]`; `augment(sample, size) -> list[Sample]`; `play_games(evaluator, n_games, config, rng) -> tuple[list[Sample], GameStats]`.

This task implements section 3 of the spec. The three requirements that must not be quietly dropped: openings are random and multi-ply (this is what decorrelates the outcome from side-to-move), the temperature schedule falls to zero after `temperature_plies`, and policy targets are recorded **only** from full-simulation moves.

- [ ] **Step 1: Write the failing tests**

`tests/test_selfplay.py`:

```python
import numpy as np
import pytest

from gomoku.board import BLACK, WHITE
from gomoku.evaluator import UniformEvaluator
from gomoku.game import N_PLANES, GameState
from gomoku.selfplay import (
    GameStats,
    Sample,
    SelfPlayConfig,
    augment,
    play_games,
    random_opening,
)


def small_config(**kwargs):
    defaults = dict(
        size=5, win_length=4, full_simulations=16, fast_simulations=4,
        games_in_flight=4, opening_plies=(2, 3),
    )
    defaults.update(kwargs)
    return SelfPlayConfig(**defaults)


def test_random_opening_places_the_configured_number_of_plies():
    rng = np.random.default_rng(0)
    config = small_config(opening_plies=(3, 3))
    state, opening = random_opening(rng, config)
    assert state.ply == 3
    assert len(opening) == 3
    assert not state.is_terminal()


def test_random_opening_stones_are_legal_and_distinct():
    rng = np.random.default_rng(1)
    config = small_config()
    for _ in range(30):
        state, opening = random_opening(rng, config)
        assert len(set(opening)) == len(opening)
        assert state.board.n_moves == len(opening)


def test_random_opening_is_diverse_across_seeds():
    config = SelfPlayConfig(size=9, opening_plies=(2, 4))
    openings = {random_opening(np.random.default_rng(seed), config)[1]
                for seed in range(60)}
    assert len(openings) > 30


def test_random_opening_stays_near_the_centre():
    config = SelfPlayConfig(size=9, opening_radius=2, opening_plies=(2, 2))
    rng = np.random.default_rng(4)
    for _ in range(20):
        _, opening = random_opening(rng, config)
        for move in opening:
            row, col = divmod(move, 9)
            assert abs(row - 4) <= 3 and abs(col - 4) <= 3


def test_augment_produces_eight_consistent_variants():
    encoded = np.zeros((N_PLANES, 5, 5), dtype=np.float32)
    encoded[0, 0, 1] = 1.0
    policy = np.zeros(25, dtype=np.float32)
    policy[6] = 1.0
    variants = augment(Sample(encoded, policy, 0.5), size=5)
    assert len(variants) == 8
    for variant in variants:
        assert variant.value == 0.5
        assert variant.encoded.shape == encoded.shape
        assert np.isclose(variant.policy.sum(), 1.0)
        assert variant.encoded[0].sum() == 1.0
        # The marked cell and its policy mass must move together.
        cell = int(np.argmax(variant.encoded[0]))
        assert int(np.argmax(variant.policy)) == cell


def test_play_games_returns_samples_with_matching_shapes():
    config = small_config()
    samples, stats = play_games(UniformEvaluator(), 4, config,
                                np.random.default_rng(0))
    assert stats.n_games == 4
    assert samples
    for sample in samples[:20]:
        assert sample.encoded.shape == (N_PLANES, 5, 5)
        assert sample.policy.shape == (25,)
        assert np.isclose(sample.policy.sum(), 1.0)
        assert -1.0 <= sample.value <= 1.0


def test_stats_account_for_every_game():
    config = small_config()
    _, stats = play_games(UniformEvaluator(), 6, config, np.random.default_rng(2))
    assert stats.black_wins + stats.white_wins + stats.draws == 6
    assert len(stats.lengths) == 6
    assert 0.0 <= stats.black_win_rate <= 1.0
    assert len(stats.openings) >= 1


def test_values_are_from_the_perspective_of_the_side_to_move():
    """Consecutive positions in one game must alternate in sign (or be draws)."""
    config = small_config(full_fraction=1.0, games_in_flight=1)
    samples, _ = play_games(UniformEvaluator(), 1, config, np.random.default_rng(5))
    values = [s.value for s in samples[::8]]  # one per position, pre-augmentation
    non_zero = [v for v in values if v != 0.0]
    for earlier, later in zip(non_zero, non_zero[1:]):
        assert earlier == -later


def test_only_full_simulation_moves_are_recorded():
    """With full_fraction 0, no policy targets are produced at all."""
    config = small_config(full_fraction=0.0)
    samples, stats = play_games(UniformEvaluator(), 3, config,
                                np.random.default_rng(0))
    assert stats.n_games == 3
    assert samples == []


def test_all_moves_recorded_when_full_fraction_is_one():
    config = small_config(full_fraction=1.0, games_in_flight=1)
    samples, stats = play_games(UniformEvaluator(), 1, config,
                                np.random.default_rng(3))
    assert len(samples) == 8 * stats.lengths[0] - 8 * len(next(iter(stats.openings)))


def test_temperature_schedule_switches_to_argmax_after_the_opening():
    """With temperature_plies 0 every move is greedy, so two runs of the same
    seed and evaluator agree exactly."""
    config = small_config(temperature_plies=0, full_fraction=1.0, games_in_flight=1)
    first, _ = play_games(UniformEvaluator(), 1, config, np.random.default_rng(9))
    second, _ = play_games(UniformEvaluator(), 1, config, np.random.default_rng(9))
    assert len(first) == len(second)
    assert all(np.allclose(a.policy, b.policy) for a, b in zip(first, second))


def test_play_games_is_reproducible_for_a_seed():
    config = small_config()
    a, stats_a = play_games(UniformEvaluator(), 3, config, np.random.default_rng(12))
    b, stats_b = play_games(UniformEvaluator(), 3, config, np.random.default_rng(12))
    assert stats_a.lengths == stats_b.lengths
    assert len(a) == len(b)


def test_games_in_flight_does_not_change_the_game_count():
    for in_flight in (1, 3, 16):
        config = small_config(games_in_flight=in_flight)
        _, stats = play_games(UniformEvaluator(), 5, config,
                              np.random.default_rng(1))
        assert stats.n_games == 5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_selfplay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.selfplay'`

- [ ] **Step 3: Implement self-play**

`gomoku/selfplay.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_selfplay.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gomoku/selfplay.py tests/test_selfplay.py
git commit -m "feat: self-play with random openings, temperature schedule, playout caps"
```

---

### Task 11: Replay buffer

**Files:**
- Create: `gomoku/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `gomoku.selfplay.Sample`.
- Produces: `ReplayBuffer(capacity: int, directory: str | Path | None = None)` with attributes `capacity`, `directory`, `n_added: int`, `__len__`, and methods `add(samples: Iterable[Sample]) -> None`, `sample_batch(batch_size: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(encoded (B,C,H,W), policy (B,cells), value (B,))`, `save_shard(generation: int) -> Path | None`, `load_shards() -> int`, `value_statistics() -> dict[str, float]`.

The buffer is a bounded deque in memory plus optional on-disk shards, so a multi-day run resumes without throwing away its history. A corrupt shard is skipped with a warning rather than aborting the run (spec §6).

- [ ] **Step 1: Write the failing tests**

`tests/test_replay.py`:

```python
import numpy as np
import pytest

from gomoku.game import N_PLANES
from gomoku.replay import ReplayBuffer
from gomoku.selfplay import Sample


def make_samples(n, size=5, value=1.0):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        policy = rng.random(size * size).astype(np.float32)
        out.append(Sample(
            rng.random((N_PLANES, size, size)).astype(np.float32),
            (policy / policy.sum()).astype(np.float32),
            value,
        ))
    return out


def test_buffer_starts_empty():
    buffer = ReplayBuffer(capacity=10)
    assert len(buffer) == 0
    assert buffer.n_added == 0


def test_add_grows_the_buffer():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(make_samples(4))
    assert len(buffer) == 4
    assert buffer.n_added == 4


def test_capacity_evicts_the_oldest_samples():
    buffer = ReplayBuffer(capacity=5)
    buffer.add(make_samples(4, value=-1.0))
    buffer.add(make_samples(4, value=1.0))
    assert len(buffer) == 5
    assert buffer.n_added == 8
    # Four of the five survivors are the newer, +1 samples: mean = (4 - 1) / 5.
    assert buffer.value_statistics()["mean"] == pytest.approx(0.6)


def test_sample_batch_shapes():
    buffer = ReplayBuffer(capacity=100)
    buffer.add(make_samples(20))
    encoded, policy, value = buffer.sample_batch(8, np.random.default_rng(0))
    assert encoded.shape == (8, N_PLANES, 5, 5)
    assert policy.shape == (8, 25)
    assert value.shape == (8,)
    assert encoded.dtype == np.float32


def test_sample_batch_larger_than_the_buffer_draws_with_replacement():
    buffer = ReplayBuffer(capacity=100)
    buffer.add(make_samples(3))
    encoded, _, _ = buffer.sample_batch(10, np.random.default_rng(0))
    assert encoded.shape[0] == 10


def test_sample_batch_from_an_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10).sample_batch(4, np.random.default_rng(0))


def test_shards_round_trip(tmp_path):
    buffer = ReplayBuffer(capacity=100, directory=tmp_path)
    buffer.add(make_samples(6))
    path = buffer.save_shard(generation=3)
    assert path is not None and path.exists()

    restored = ReplayBuffer(capacity=100, directory=tmp_path)
    assert restored.load_shards() == 6
    assert len(restored) == 6


def test_load_shards_skips_a_corrupt_file(tmp_path):
    buffer = ReplayBuffer(capacity=100, directory=tmp_path)
    buffer.add(make_samples(4))
    buffer.save_shard(generation=0)
    (tmp_path / "shard_000099.npz").write_bytes(b"not an npz file")

    restored = ReplayBuffer(capacity=100, directory=tmp_path)
    assert restored.load_shards() == 4
    assert len(restored) == 4


def test_load_shards_respects_capacity(tmp_path):
    writer = ReplayBuffer(capacity=100, directory=tmp_path)
    for generation in range(3):
        writer.add(make_samples(5))
        writer.save_shard(generation)

    reader = ReplayBuffer(capacity=6, directory=tmp_path)
    reader.load_shards()
    assert len(reader) == 6


def test_save_shard_without_a_directory_is_a_no_op():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(make_samples(2))
    assert buffer.save_shard(generation=0) is None


def test_value_statistics_report_mean_and_variance():
    buffer = ReplayBuffer(capacity=100)
    buffer.add(make_samples(5, value=1.0))
    buffer.add(make_samples(5, value=-1.0))
    stats = buffer.value_statistics()
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["variance"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_replay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.replay'`

- [ ] **Step 3: Implement the buffer**

`gomoku/replay.py`:

```python
"""The replay buffer: a bounded window over recent self-play samples.

Shards on disk make a multi-day run resumable. A shard that fails to load is
skipped rather than fatal -- losing part of the history costs some training
signal, but aborting a two-day run costs the whole thing.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from gomoku.selfplay import Sample

log = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity: int, directory: str | Path | None = None) -> None:
        self.capacity = capacity
        self.directory = Path(directory) if directory is not None else None
        self._samples: collections.deque[Sample] = collections.deque(maxlen=capacity)
        self.n_added = 0

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self._samples.append(sample)
            self.n_added += 1

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._samples:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = rng.integers(0, len(self._samples), size=batch_size)
        chosen = [self._samples[int(i)] for i in indices]
        encoded = np.stack([s.encoded for s in chosen]).astype(np.float32)
        policy = np.stack([s.policy for s in chosen]).astype(np.float32)
        value = np.array([s.value for s in chosen], dtype=np.float32)
        return encoded, policy, value

    def value_statistics(self) -> dict[str, float]:
        if not self._samples:
            return {"mean": 0.0, "variance": 0.0}
        values = np.array([s.value for s in self._samples], dtype=np.float64)
        return {"mean": float(values.mean()), "variance": float(values.var())}

    # -- persistence ----------------------------------------------------

    def save_shard(self, generation: int) -> Path | None:
        """Write the current window to `shard_%06d.npz`. No directory, no shard."""
        if self.directory is None or not self._samples:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"shard_{generation:06d}.npz"
        tmp = path.with_suffix(".npz.tmp")
        np.savez_compressed(
            tmp,
            encoded=np.stack([s.encoded for s in self._samples]),
            policy=np.stack([s.policy for s in self._samples]),
            value=np.array([s.value for s in self._samples], dtype=np.float32),
        )
        tmp.replace(path)
        return path

    def load_shards(self) -> int:
        """Load every shard in generation order. Returns the number loaded."""
        if self.directory is None or not self.directory.exists():
            return 0
        loaded = 0
        for path in sorted(self.directory.glob("shard_*.npz")):
            try:
                with np.load(path) as data:
                    encoded, policy, value = data["encoded"], data["policy"], data["value"]
            except (OSError, ValueError, KeyError) as error:
                log.warning("skipping unreadable replay shard %s: %s", path, error)
                continue
            self.add(
                Sample(encoded[i], policy[i], float(value[i]))
                for i in range(len(value))
            )
            loaded += len(value)
        return min(loaded, self.capacity) if loaded else 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_replay.py -q`
Expected: PASS. Note `load_shards` reports how many samples were read, capped by capacity, which is what `test_shards_round_trip` and `test_load_shards_skips_a_corrupt_file` assert.

- [ ] **Step 5: Commit**

```bash
git add gomoku/replay.py tests/test_replay.py
git commit -m "feat: replay buffer with resumable on-disk shards"
```

---

### Task 12: Training loop and diagnostics

**Files:**
- Create: `gomoku/metrics.py`
- Create: `gomoku/train.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: everything from Tasks 6, 7, 10, 11.
- Produces: `metrics.MetricsWriter(path)` with `write(record: dict) -> None` and `read_all() -> list[dict]`; `metrics.policy_entropy(policies: np.ndarray) -> float`; `metrics.baseline_value_losses(values, is_black_to_move) -> dict[str, float]` returning `{"constant": float, "parity": float}`; `train.TrainConfig` frozen dataclass with `generations=100`, `games_per_generation=64`, `batch_size=256`, `batches_per_generation=200`, `learning_rate=2e-3`, `weight_decay=1e-4`, `buffer_capacity=200_000`, `run_dir="runs/default"`, `device=None`, `net=NetConfig()`, `selfplay=SelfPlayConfig()`; `train.train(config, rng, resume=True) -> Path` returning the final checkpoint path; `train.loss_terms(policy_logits, value_pred, policy_target, value_target) -> tuple[Tensor, Tensor]`.

The diagnostics in `metrics` are the spec's §3 instrumentation. `baseline_value_losses` is the important one: a value head that cannot beat the **parity** baseline has learned only "black is winning", which is exactly the failure the design is guarding against.

- [ ] **Step 1: Write the failing metrics tests**

`tests/test_metrics.py`:

```python
import numpy as np
import pytest

from gomoku.metrics import MetricsWriter, baseline_value_losses, policy_entropy


def test_entropy_of_a_uniform_policy_is_maximal():
    uniform = np.full((2, 16), 1.0 / 16)
    assert policy_entropy(uniform) == pytest.approx(np.log(16))


def test_entropy_of_a_deterministic_policy_is_zero():
    sharp = np.zeros((2, 16))
    sharp[:, 3] = 1.0
    assert policy_entropy(sharp) == pytest.approx(0.0, abs=1e-9)


def test_constant_baseline_equals_the_variance_of_the_targets():
    values = np.array([1.0, 1.0, -1.0, 0.0])
    is_black = np.array([True, False, True, False])
    losses = baseline_value_losses(values, is_black)
    assert losses["constant"] == pytest.approx(values.var())


def test_parity_baseline_beats_the_constant_baseline_when_colour_predicts_the_result():
    values = np.array([1.0, -1.0, 1.0, -1.0])
    is_black = np.array([True, False, True, False])
    losses = baseline_value_losses(values, is_black)
    assert losses["parity"] == pytest.approx(0.0)
    assert losses["constant"] > losses["parity"]


def test_parity_baseline_matches_the_constant_one_when_colour_is_uninformative():
    values = np.array([1.0, -1.0, 1.0, -1.0])
    is_black = np.array([True, True, False, False])
    losses = baseline_value_losses(values, is_black)
    assert losses["parity"] == pytest.approx(losses["constant"])


def test_baselines_handle_a_missing_colour():
    values = np.array([1.0, 0.5])
    losses = baseline_value_losses(values, np.array([True, True]))
    assert losses["parity"] == pytest.approx(values.var())


def test_metrics_writer_appends_json_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    writer.write({"generation": 0, "loss": 1.5})
    writer.write({"generation": 1, "loss": 1.2})
    records = MetricsWriter(path).read_all()
    assert [r["generation"] for r in records] == [0, 1]
    assert records[1]["loss"] == 1.2


def test_metrics_writer_reads_an_absent_file_as_empty(tmp_path):
    assert MetricsWriter(tmp_path / "missing.jsonl").read_all() == []


def test_metrics_writer_skips_a_truncated_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    MetricsWriter(path).write({"generation": 0})
    with path.open("a") as handle:
        handle.write('{"generation": 1, "loss"\n')
    assert len(MetricsWriter(path).read_all()) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.metrics'`

- [ ] **Step 3: Implement the metrics**

`gomoku/metrics.py`:

```python
"""Training diagnostics and a JSON-lines metrics log.

`baseline_value_losses` is the instrument that detects the parity shortcut
described in section 3 of the design. The constant baseline is what a value
head achieves by predicting the mean outcome; the parity baseline is what it
achieves by predicting the mean outcome *for each colour*. A value head that
does not beat the parity baseline has learned nothing but "black is winning".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def policy_entropy(policies: np.ndarray) -> float:
    """Mean Shannon entropy in nats over a batch of policy distributions."""
    probabilities = np.clip(policies, 1e-12, 1.0)
    return float(-(probabilities * np.log(probabilities)).sum(axis=-1).mean())


def baseline_value_losses(
    values: np.ndarray,
    is_black_to_move: np.ndarray,
) -> dict[str, float]:
    """Mean-squared error of the constant and parity predictors."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(is_black_to_move, dtype=bool)
    constant = float(((values - values.mean()) ** 2).mean())
    predicted = np.empty_like(values)
    for group in (mask, ~mask):
        if group.any():
            predicted[group] = values[group].mean()
    parity = float(((values - predicted) ** 2).mean())
    return {"constant": constant, "parity": parity}


class MetricsWriter:
    """Append-only JSON-lines log. One record per training generation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                log.warning("skipping malformed metrics line in %s", self.path)
        return records
```

- [ ] **Step 4: Write the failing training tests**

`tests/test_train.py`:

```python
import numpy as np
import torch

from gomoku.mcts import SearchConfig
from gomoku.metrics import MetricsWriter
from gomoku.net import NetConfig, load_checkpoint
from gomoku.selfplay import SelfPlayConfig
from gomoku.train import TrainConfig, loss_terms, train


def tiny_config(run_dir, **kwargs):
    defaults = dict(
        generations=2,
        games_per_generation=2,
        batch_size=8,
        batches_per_generation=3,
        buffer_capacity=2000,
        run_dir=str(run_dir),
        device="cpu",
        net=NetConfig(channels=8, blocks=1),
        selfplay=SelfPlayConfig(
            size=5, win_length=4, full_simulations=8, fast_simulations=2,
            full_fraction=1.0, games_in_flight=2, opening_plies=(2, 2),
            search=SearchConfig(dirichlet_alpha=0.5),
        ),
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def test_loss_terms_are_non_negative_and_differentiable():
    logits = torch.zeros(4, 25, requires_grad=True)
    value_pred = torch.zeros(4, requires_grad=True)
    policy_target = torch.full((4, 25), 1.0 / 25)
    value_target = torch.zeros(4)
    policy_loss, value_loss = loss_terms(logits, value_pred, policy_target, value_target)
    assert policy_loss.item() >= 0 and value_loss.item() >= 0
    (policy_loss + value_loss).backward()
    assert logits.grad is not None


def test_policy_loss_is_lower_when_the_prediction_matches():
    target = torch.zeros(1, 4)
    target[0, 2] = 1.0
    good = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
    bad = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    zeros = torch.zeros(1)
    assert loss_terms(good, zeros, target, zeros)[0] < \
        loss_terms(bad, zeros, target, zeros)[0]


def test_train_produces_a_checkpoint_and_metrics(tmp_path):
    path = train(tiny_config(tmp_path), np.random.default_rng(0))
    assert path.exists()
    payload = load_checkpoint(path)
    assert payload["generation"] == 2
    assert payload["config"]["channels"] == 8

    records = MetricsWriter(tmp_path / "metrics.jsonl").read_all()
    assert len(records) == 2
    for record in records:
        for key in ("generation", "policy_loss", "value_loss", "policy_entropy",
                    "black_win_rate", "value_baseline_constant",
                    "value_baseline_parity", "mean_game_length",
                    "distinct_openings", "buffer_size", "seconds"):
            assert key in record


def test_train_resumes_from_the_last_checkpoint(tmp_path):
    train(tiny_config(tmp_path), np.random.default_rng(0))
    path = train(tiny_config(tmp_path, generations=4), np.random.default_rng(1))
    assert load_checkpoint(path)["generation"] == 4
    assert len(MetricsWriter(tmp_path / "metrics.jsonl").read_all()) == 4


def test_train_from_scratch_ignores_an_existing_checkpoint(tmp_path):
    train(tiny_config(tmp_path), np.random.default_rng(0))
    path = train(tiny_config(tmp_path), np.random.default_rng(0), resume=False)
    assert load_checkpoint(path)["generation"] == 2


def test_replay_shards_are_written(tmp_path):
    train(tiny_config(tmp_path), np.random.default_rng(0))
    assert list((tmp_path / "replay").glob("shard_*.npz"))
```

- [ ] **Step 5: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_train.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.train'`

- [ ] **Step 6: Implement the training loop**

`gomoku/train.py`:

```python
"""The AlphaZero training loop: self-play, fit, checkpoint, repeat.

Every generation logs the section 3 diagnostics alongside the losses, so the
failure modes of a first-player-advantaged game are visible in the metrics
file rather than inferred from a weak final bot.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from gomoku.evaluator import NetEvaluator
from gomoku.game import N_PLANES
from gomoku.metrics import MetricsWriter, baseline_value_losses, policy_entropy
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint, save_checkpoint, select_device
from gomoku.replay import ReplayBuffer
from gomoku.selfplay import SelfPlayConfig, play_games

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    generations: int = 100
    games_per_generation: int = 64
    batch_size: int = 256
    batches_per_generation: int = 200
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    buffer_capacity: int = 200_000
    run_dir: str = "runs/default"
    device: str | None = None
    net: NetConfig = dataclasses.field(default_factory=NetConfig)
    selfplay: SelfPlayConfig = dataclasses.field(default_factory=SelfPlayConfig)

    @property
    def directory(self) -> Path:
        return Path(self.run_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / "checkpoint.pt"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.jsonl"

    @property
    def replay_dir(self) -> Path:
        return self.directory / "replay"


def loss_terms(
    policy_logits: Tensor,
    value_pred: Tensor,
    policy_target: Tensor,
    value_target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Cross-entropy against the visit distribution, MSE against the outcome."""
    log_probabilities = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probabilities).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
    return policy_loss, value_loss


def train(
    config: TrainConfig,
    rng: np.random.Generator,
    resume: bool = True,
) -> Path:
    device = select_device(config.device)
    net = PolicyValueNet(config.net).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    buffer = ReplayBuffer(config.buffer_capacity, config.replay_dir)
    metrics = MetricsWriter(config.metrics_path)
    start_generation = 0

    if resume and config.checkpoint_path.exists():
        payload = load_checkpoint(config.checkpoint_path, map_location=device)
        net.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        start_generation = int(payload["generation"])
        buffer.load_shards()
        log.info("resumed from generation %d", start_generation)

    evaluator = NetEvaluator(net, device=device)

    for generation in range(start_generation, config.generations):
        started = time.monotonic()
        net.eval()
        evaluator.refresh(net)
        samples, stats = play_games(
            evaluator, config.games_per_generation, config.selfplay, rng
        )
        buffer.add(samples)

        net.train()
        policy_losses: list[float] = []
        value_losses: list[float] = []
        for _ in range(config.batches_per_generation):
            encoded, policy_target, value_target = buffer.sample_batch(
                config.batch_size, rng
            )
            x = torch.from_numpy(encoded).to(device)
            policy_logits, value_pred = net(x)
            policy_loss, value_loss = loss_terms(
                policy_logits,
                value_pred,
                torch.from_numpy(policy_target).to(device),
                torch.from_numpy(value_target).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_loss).backward()
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))

        record = _diagnostics(generation, samples, stats, buffer,
                              policy_losses, value_losses, started)
        metrics.write(record)
        log.info(
            "gen %d policy %.3f value %.3f (parity baseline %.3f) black %.2f",
            generation, record["policy_loss"], record["value_loss"],
            record["value_baseline_parity"], record["black_win_rate"],
        )
        buffer.save_shard(generation)
        save_checkpoint(
            config.checkpoint_path, net, optimizer, generation + 1,
            config.net, extra={"buffer_size": len(buffer)},
        )

    return config.checkpoint_path


def _diagnostics(generation, samples, stats, buffer, policy_losses,
                 value_losses, started) -> dict:
    """Assemble one generation's metrics record, including the §3 diagnostics."""
    if samples:
        policies = np.stack([s.policy for s in samples])
        values = np.array([s.value for s in samples], dtype=np.float64)
        # Plane 3 is the side-to-move constant: 1 when black is to move.
        is_black = np.array([bool(s.encoded[3].flat[0]) for s in samples])
        baselines = baseline_value_losses(values, is_black)
        entropy = policy_entropy(policies)
    else:
        baselines = {"constant": 0.0, "parity": 0.0}
        entropy = 0.0
    return {
        "generation": generation,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "policy_entropy": entropy,
        "black_win_rate": stats.black_win_rate,
        "value_baseline_constant": baselines["constant"],
        "value_baseline_parity": baselines["parity"],
        "mean_game_length": stats.mean_length,
        "distinct_openings": len(stats.openings),
        "buffer_size": len(buffer),
        "samples_added": len(samples),
        "seconds": time.monotonic() - started,
    }
```

- [ ] **Step 7: Run the training tests to verify they pass**

Run: `.venv/bin/pytest tests/test_train.py tests/test_metrics.py -q`
Expected: PASS. These run on CPU with a one-block network; if any single test takes more than ~30 seconds, reduce `batches_per_generation` in `tiny_config`, not the assertions.

- [ ] **Step 8: Commit**

```bash
git add gomoku/metrics.py gomoku/train.py tests/test_metrics.py tests/test_train.py
git commit -m "feat: training loop with parity-shortcut and diversity diagnostics"
```

---

### Task 13: End-to-end learning test on a 3x3 board

**Files:**
- Create: `tests/test_learning.py`
- Modify: `pyproject.toml` (register the `slow` marker)

**Interfaces:**
- Consumes: `gomoku.train.{TrainConfig, train}`, `gomoku.net`, `gomoku.evaluator.NetEvaluator`, `gomoku.players.{MCTSPlayer, RandomPlayer}`, `gomoku.difficulty`.
- Produces: no library code — this task's deliverable is proof that the pipeline learns.

Tic-tac-toe (3x3, three in a row) is small enough to solve in seconds and shares every code path with the real game. If this test fails, the training pipeline is broken; there is no point starting a 9x9 run.

- [ ] **Step 1: Register the marker**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["slow: end-to-end training runs (tens of seconds)"]
```

- [ ] **Step 2: Write the failing test**

`tests/test_learning.py`:

```python
"""Proof that the whole pipeline learns, on a board small enough to be fast.

3x3 with three in a row is tic-tac-toe: a draw under perfect play, trivially
solvable, and it exercises self-play, the replay buffer, the training loop,
checkpointing and the evaluator exactly as the 9x9 game does.
"""

import numpy as np
import pytest
import torch

from gomoku.evaluator import NetEvaluator
from gomoku.game import GameState
from gomoku.mcts import SearchConfig
from gomoku.metrics import MetricsWriter
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint
from gomoku.players import MCTSPlayer, RandomPlayer
from gomoku.selfplay import SelfPlayConfig
from gomoku.train import TrainConfig, train

pytestmark = pytest.mark.slow

SIZE, WIN = 3, 3


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("ttt")
    config = TrainConfig(
        generations=12,
        games_per_generation=32,
        batch_size=64,
        batches_per_generation=40,
        learning_rate=3e-3,
        buffer_capacity=20_000,
        run_dir=str(run_dir),
        device="cpu",
        net=NetConfig(channels=16, blocks=2),
        selfplay=SelfPlayConfig(
            size=SIZE,
            win_length=WIN,
            opening_plies=(1, 2),
            opening_radius=2,
            full_simulations=40,
            fast_simulations=10,
            full_fraction=0.5,
            temperature_plies=2,
            games_in_flight=16,
            search=SearchConfig(dirichlet_alpha=0.6),
        ),
    )
    path = train(config, np.random.default_rng(0))
    payload = load_checkpoint(path)
    net = PolicyValueNet(NetConfig(**payload["config"]))
    net.load_state_dict(payload["model"])
    records = MetricsWriter(run_dir / "metrics.jsonl").read_all()
    return NetEvaluator(net, device="cpu"), records


def play(black, white, rng):
    state = GameState.new(SIZE, WIN)
    players = [black, white]
    while not state.is_terminal():
        state = state.play(players[state.ply % 2].select_move(state))
    return state.winner


def test_searching_agent_never_loses_to_random(trained):
    evaluator, _ = trained
    losses = 0
    for game in range(20):
        rng = np.random.default_rng(1000 + game)
        agent = MCTSPlayer(evaluator, simulations=100, rng=rng)
        random_player = RandomPlayer(rng)
        if game % 2 == 0:
            winner = play(agent, random_player, rng)
            losses += winner == 2
        else:
            winner = play(random_player, agent, rng)
            losses += winner == 1
    assert losses == 0


def test_raw_network_policy_is_much_better_than_random(trained):
    """No search at all: this measures what the network itself learned."""
    evaluator, _ = trained
    wins = losses = 0
    for game in range(40):
        rng = np.random.default_rng(2000 + game)
        agent = MCTSPlayer(evaluator, simulations=0, policy_only=True,
                           temperature=0.05, rng=rng)
        random_player = RandomPlayer(rng)
        agent_is_black = game % 2 == 0
        winner = play(agent, random_player, rng) if agent_is_black \
            else play(random_player, agent, rng)
        agent_colour = 1 if agent_is_black else 2
        wins += winner == agent_colour
        losses += winner not in (0, agent_colour)
    assert wins >= 24
    assert losses <= 4


def test_agent_takes_an_immediate_win(trained):
    evaluator, _ = trained
    # X at 0 and 1; O at 3 and 4. X to move wins at 2.
    state = GameState.new(SIZE, WIN)
    for move in (0, 3, 1, 4):
        state = state.play(move)
    agent = MCTSPlayer(evaluator, simulations=100, rng=np.random.default_rng(0))
    assert agent.select_move(state) == 2


def test_agent_blocks_an_immediate_loss(trained):
    evaluator, _ = trained
    # X at 0 and 1 threatens 2; O to move must block.
    state = GameState.new(SIZE, WIN)
    for move in (0, 4, 1):
        state = state.play(move)
    agent = MCTSPlayer(evaluator, simulations=100, rng=np.random.default_rng(0))
    assert agent.select_move(state) == 2


def test_self_play_between_trained_agents_mostly_draws(trained):
    """Tic-tac-toe is a draw under perfect play."""
    evaluator, _ = trained
    draws = 0
    for game in range(10):
        rng = np.random.default_rng(3000 + game)
        draws += play(
            MCTSPlayer(evaluator, simulations=150, rng=rng),
            MCTSPlayer(evaluator, simulations=150, rng=rng),
            rng,
        ) == 0
    assert draws >= 7


def test_losses_fall_over_training(trained):
    _, records = trained
    early = np.mean([r["policy_loss"] for r in records[:3]])
    late = np.mean([r["policy_loss"] for r in records[-3:]])
    assert late < early


def test_value_head_beats_the_parity_baseline(trained):
    """The §3 diagnostic: the value head must learn more than 'black is winning'."""
    _, records = trained
    final = records[-1]
    assert final["value_loss"] < final["value_baseline_parity"]


def test_exploration_does_not_collapse(trained):
    _, records = trained
    assert all(r["policy_entropy"] > 0.05 for r in records)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_learning.py -q`
Expected: FAIL — the assertions fail, or the fixture errors, until the pipeline actually trains.

- [ ] **Step 4: Make it pass**

No new module is required. If the test fails, the defect is in existing code — debug it rather than weakening the assertions. The usual suspects, in order:

1. **Value sign errors.** `test_agent_blocks_an_immediate_loss` failing while the win test passes points at `_backup` in `mcts.py` or the `value` assignment in `selfplay._finish`.
2. **Policy target/board mismatch.** `test_losses_fall_over_training` failing points at `augment` — verify the encoded planes and the policy vector are transformed by the *same* symmetry index.
3. **Value head not beating the parity baseline.** Increase `generations`, and check that `opening_plies` is genuinely randomising.

Run the full suite before moving on:

Run: `.venv/bin/pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_learning.py pyproject.toml
git commit -m "test: end-to-end learning proof on a 3x3 board"
```

---

### Task 14: Arena and ELO measurement

**Files:**
- Create: `gomoku/arena.py`
- Test: `tests/test_arena.py`

**Interfaces:**
- Consumes: `gomoku.players.Player`, `gomoku.selfplay.random_opening`, `gomoku.difficulty.{LEVELS, make_player}`, `gomoku.evaluator.Evaluator`.
- Produces: `MatchConfig` frozen dataclass with `size=9`, `win_length=5`, `games_per_pair=20`, `opening_plies=(2, 4)`, `opening_radius=2`; `play_pair(player_a, player_b, config, rng) -> tuple[float, float]` returning points from one colour-paired opening (two games, so the pair totals 2.0); `play_match(player_a, player_b, config, rng) -> tuple[float, float]`; `round_robin(players: dict[str, Player], config, rng) -> np.ndarray` score matrix; `fit_ratings(scores, names, anchor, anchor_rating=1200.0, prior_sigma=400.0, iterations=3000, step=8.0) -> dict[str, float]`; `write_elo(path, ratings, metadata) -> None`; `measure_levels(evaluator, config, rng, anchor_rng, elo_path) -> dict[str, float]`.

Colour pairing lives here, and only here: each opening is played once from each side so that a level's rating reflects its strength rather than the advantage of moving first.

- [ ] **Step 1: Write the failing tests**

`tests/test_arena.py`:

```python
import json

import numpy as np
import pytest

from gomoku.arena import (
    MatchConfig,
    fit_ratings,
    play_match,
    play_pair,
    round_robin,
    write_elo,
)
from gomoku.game import GameState
from gomoku.players import HeuristicPlayer, Player, RandomPlayer


class FixedPlayer(Player):
    """Plays the lowest-numbered legal cell. Deterministic and weak."""

    name = "fixed"

    def select_move(self, state: GameState) -> int:
        return int(state.legal_moves()[0])


def small_config(**kwargs):
    defaults = dict(size=5, win_length=4, games_per_pair=4, opening_plies=(2, 2))
    defaults.update(kwargs)
    return MatchConfig(**defaults)


def test_play_pair_awards_exactly_two_points():
    rng = np.random.default_rng(0)
    a, b = play_pair(RandomPlayer(rng), RandomPlayer(rng), small_config(), rng)
    assert a + b == pytest.approx(2.0)


def test_play_match_totals_the_game_count():
    rng = np.random.default_rng(0)
    config = small_config(games_per_pair=6)
    a, b = play_match(RandomPlayer(rng), RandomPlayer(rng), config, rng)
    assert a + b == pytest.approx(6.0)


def test_play_match_rounds_odd_game_counts_up_to_a_whole_pair():
    rng = np.random.default_rng(0)
    a, b = play_match(RandomPlayer(rng), RandomPlayer(rng),
                      small_config(games_per_pair=5), rng)
    assert a + b == pytest.approx(6.0)


def test_the_stronger_player_scores_more():
    rng = np.random.default_rng(0)
    config = small_config(games_per_pair=10)
    strong, weak = HeuristicPlayer(rng), FixedPlayer()
    strong_points, weak_points = play_match(strong, weak, config, rng)
    assert strong_points > weak_points


def test_round_robin_matrix_is_square_and_hollow():
    rng = np.random.default_rng(0)
    players = {"a": RandomPlayer(rng), "b": RandomPlayer(rng), "c": FixedPlayer()}
    scores = round_robin(players, small_config(), rng)
    assert scores.shape == (3, 3)
    assert np.all(np.diag(scores) == 0)


def test_round_robin_pairs_account_for_every_game():
    rng = np.random.default_rng(0)
    config = small_config(games_per_pair=4)
    players = {"a": RandomPlayer(rng), "b": FixedPlayer()}
    scores = round_robin(players, config, rng)
    assert scores[0, 1] + scores[1, 0] == pytest.approx(4.0)


def synthetic_scores(true_ratings, games=400):
    """Expected results implied by a set of true ratings."""
    n = len(true_ratings)
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            expected = 1.0 / (1.0 + 10 ** ((true_ratings[j] - true_ratings[i]) / 400))
            scores[i, j] = expected * games
    return scores


def test_fit_recovers_known_ratings():
    names = ["w", "x", "y", "z"]
    truth = [1000.0, 1200.0, 1400.0, 1700.0]
    ratings = fit_ratings(synthetic_scores(truth), names, anchor="x",
                          anchor_rating=1200.0)
    for name, expected in zip(names, truth):
        assert ratings[name] == pytest.approx(expected, abs=40)


def test_the_anchor_keeps_its_assigned_rating():
    names = ["a", "b"]
    ratings = fit_ratings(synthetic_scores([1500.0, 1300.0]), names,
                          anchor="a", anchor_rating=1200.0)
    assert ratings["a"] == pytest.approx(1200.0, abs=1e-6)
    assert ratings["b"] < ratings["a"]


def test_ratings_stay_finite_for_an_undefeated_player():
    scores = np.array([[0.0, 30.0], [0.0, 0.0]])
    ratings = fit_ratings(scores, ["winner", "loser"], anchor="loser",
                          anchor_rating=1200.0)
    assert np.isfinite(ratings["winner"])
    assert ratings["winner"] > ratings["loser"]


def test_fit_rejects_an_unknown_anchor():
    with pytest.raises(KeyError):
        fit_ratings(np.zeros((2, 2)), ["a", "b"], anchor="missing")


def test_write_elo_produces_a_file_difficulty_can_read(tmp_path):
    from gomoku.difficulty import load_levels

    path = tmp_path / "elo.json"
    ratings = {f"level{i}": 1000.0 + 200 * i for i in range(1, 6)}
    ratings["heuristic"] = 1200.0
    write_elo(path, ratings, {"games_per_pair": 20})
    payload = json.loads(path.read_text())
    assert payload["ratings"]["level3"] == 1600.0
    assert payload["metadata"]["games_per_pair"] == 20
    assert [level.elo for level in load_levels(path)] == [1200, 1400, 1600, 1800, 2000]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_arena.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.arena'`

- [ ] **Step 3: Implement the arena**

`gomoku/arena.py`:

```python
"""Head-to-head play and rating estimation.

Every pairing is colour-paired: an opening is sampled once and played twice,
with the two players swapping colours. Gomoku's first-player advantage is
large, so without pairing a rating would mostly measure how often a player
drew black.

Ratings come from a logistic (Bradley-Terry) fit by gradient ascent, with a
weak Gaussian prior that keeps an undefeated player's rating finite, and the
heuristic bot pinned at a nominal 1200 so the numbers stay comparable across
runs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from gomoku.difficulty import LEVELS, Level, make_player
from gomoku.evaluator import Evaluator
from gomoku.game import GameState
from gomoku.players import HeuristicPlayer, Player
from gomoku.selfplay import SelfPlayConfig, random_opening

log = logging.getLogger(__name__)

ANCHOR_NAME = "heuristic"
ANCHOR_RATING = 1200.0
LOG10_OVER_400 = np.log(10.0) / 400.0


@dataclasses.dataclass(frozen=True)
class MatchConfig:
    size: int = 9
    win_length: int = 5
    games_per_pair: int = 20
    opening_plies: tuple[int, int] = (2, 4)
    opening_radius: int = 2

    def opening_config(self) -> SelfPlayConfig:
        return SelfPlayConfig(
            size=self.size,
            win_length=self.win_length,
            opening_plies=self.opening_plies,
            opening_radius=self.opening_radius,
        )


def _play_one(black: Player, white: Player, state: GameState) -> int:
    players = {1: black, 2: white}
    while not state.is_terminal():
        state = state.play(players[state.to_play].select_move(state))
    return state.winner


def play_pair(
    player_a: Player,
    player_b: Player,
    config: MatchConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Two games from one shared opening, with the colours swapped."""
    opening_state, _ = random_opening(rng, config.opening_config())
    points_a = points_b = 0.0
    for black, white in ((player_a, player_b), (player_b, player_a)):
        winner = _play_one(black, white, opening_state)
        if winner == 0:
            points_a += 0.5
            points_b += 0.5
        else:
            victor = black if winner == 1 else white
            if victor is player_a:
                points_a += 1.0
            else:
                points_b += 1.0
    return points_a, points_b


def play_match(
    player_a: Player,
    player_b: Player,
    config: MatchConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Play `games_per_pair` games, rounded up to a whole number of pairs."""
    n_pairs = (config.games_per_pair + 1) // 2
    totals = np.zeros(2)
    for _ in range(n_pairs):
        totals += play_pair(player_a, player_b, config, rng)
    return float(totals[0]), float(totals[1])


def round_robin(
    players: dict[str, Player],
    config: MatchConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Score matrix `S` where `S[i, j]` is the points `i` took from `j`."""
    names = list(players)
    scores = np.zeros((len(names), len(names)))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names[i + 1 :], start=i + 1):
            points_i, points_j = play_match(players[name_i], players[name_j],
                                            config, rng)
            scores[i, j] = points_i
            scores[j, i] = points_j
            log.info("%s %.1f - %.1f %s", name_i, points_i, points_j, name_j)
    return scores


def fit_ratings(
    scores: np.ndarray,
    names: list[str],
    anchor: str,
    anchor_rating: float = ANCHOR_RATING,
    prior_sigma: float = 400.0,
    iterations: int = 3000,
    step: float = 8.0,
) -> dict[str, float]:
    """Maximum-likelihood Bradley-Terry ratings, anchored and regularised."""
    if anchor not in names:
        raise KeyError(f"anchor {anchor!r} is not among {names}")
    anchor_index = names.index(anchor)
    ratings = np.full(len(names), 1500.0)
    for _ in range(iterations):
        difference = ratings[None, :] - ratings[:, None]      # r_j - r_i
        expected = 1.0 / (1.0 + np.power(10.0, difference / 400.0))  # p_ij
        gradient = LOG10_OVER_400 * (
            (scores * (1.0 - expected)).sum(axis=1)
            - (scores.T * expected).sum(axis=1)
        )
        gradient -= (ratings - 1500.0) / (prior_sigma**2)
        ratings += step * gradient
    ratings += anchor_rating - ratings[anchor_index]
    return {name: float(rating) for name, rating in zip(names, ratings)}


def write_elo(path: str | Path, ratings: dict[str, float], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"ratings": ratings, "metadata": metadata}, indent=2))
    tmp.replace(path)


def measure_levels(
    evaluator: Evaluator,
    config: MatchConfig,
    rng: np.random.Generator,
    elo_path: str | Path,
    levels: tuple[Level, ...] = LEVELS,
) -> dict[str, float]:
    """Run the ladder against the anchor and fit ratings for every level."""
    players: dict[str, Player] = {
        level.key: make_player(level, evaluator, rng) for level in levels
    }
    players[ANCHOR_NAME] = HeuristicPlayer(rng, name=ANCHOR_NAME)
    scores = round_robin(players, config, rng)
    ratings = fit_ratings(scores, list(players), anchor=ANCHOR_NAME)
    write_elo(
        elo_path,
        ratings,
        {
            "games_per_pair": config.games_per_pair,
            "size": config.size,
            "win_length": config.win_length,
            "anchor": ANCHOR_NAME,
            "anchor_rating": ANCHOR_RATING,
        },
    )
    return ratings
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_arena.py -q`
Expected: PASS. If `test_fit_recovers_known_ratings` misses by more than 40 points, raise `iterations`; do not widen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add gomoku/arena.py tests/test_arena.py
git commit -m "feat: colour-paired arena with Bradley-Terry rating fit"
```

---

### Task 15: Difficulty in the TUI, and the command line

**Files:**
- Modify: `gomoku/tui/app.py` (level display, level switching, engine construction)
- Create: `gomoku/engine.py`
- Create: `gomoku/cli.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_tui_levels.py`

**Interfaces:**
- Consumes: `gomoku.difficulty.{Level, load_levels, make_player}`, `gomoku.net.{PolicyValueNet, NetConfig, load_checkpoint}`, `gomoku.evaluator.{NetEvaluator, UniformEvaluator}`, `gomoku.players.HeuristicPlayer`, `gomoku.arena.{MatchConfig, measure_levels}`, `gomoku.train.{TrainConfig, train}`.
- Produces: `engine.load_evaluator(checkpoint: str | Path | None, device=None) -> tuple[Evaluator | None, int | None]` returning the evaluator and the generation it was trained to, or `(None, None)` when there is no checkpoint; `engine.build_opponent(level, evaluator, rng) -> Player` falling back to `HeuristicPlayer` when `evaluator` is `None`; `cli.main(argv: list[str] | None = None) -> int` with subcommands `play`, `train`, `selfplay`, `arena`.
- `GomokuApp` gains `__init__(..., levels=None, level_index=3, evaluator=None, rng=None, mode="human-vs-pc")`, the attribute `level: Level`, the binding `1`-`5` for level selection, and a header line showing the level name and its ELO.

- [ ] **Step 1: Write the failing engine tests**

`tests/test_engine.py`:

```python
import numpy as np
import torch

from gomoku.difficulty import LEVELS
from gomoku.engine import build_opponent, load_evaluator
from gomoku.evaluator import NetEvaluator
from gomoku.game import GameState
from gomoku.net import NetConfig, PolicyValueNet, save_checkpoint
from gomoku.players import HeuristicPlayer, MCTSPlayer


def test_load_evaluator_without_a_checkpoint_returns_nothing(tmp_path):
    evaluator, generation = load_evaluator(tmp_path / "absent.pt", device="cpu")
    assert evaluator is None and generation is None


def test_load_evaluator_reads_a_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, PolicyValueNet(config), None, 7, config, {})
    evaluator, generation = load_evaluator(path, device="cpu")
    assert isinstance(evaluator, NetEvaluator)
    assert generation == 7


def test_build_opponent_falls_back_to_the_heuristic_without_a_network():
    player = build_opponent(LEVELS[4], None, np.random.default_rng(0))
    assert isinstance(player, HeuristicPlayer)


def test_build_opponent_uses_the_network_when_present(tmp_path):
    path = tmp_path / "ckpt.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, PolicyValueNet(config), None, 1, config, {})
    evaluator, _ = load_evaluator(path, device="cpu")
    player = build_opponent(LEVELS[1], evaluator, np.random.default_rng(0))
    assert isinstance(player, MCTSPlayer)
    state = GameState.new(size=5, win_length=5)
    assert player.select_move(state) in set(state.legal_moves().tolist())
```

- [ ] **Step 2: Write the failing TUI and CLI tests**

`tests/test_tui_levels.py`:

```python
import json

import numpy as np
import pytest

from gomoku.difficulty import load_levels
from gomoku.tui.app import GomokuApp


def app_with_levels(tmp_path, **kwargs):
    path = tmp_path / "elo.json"
    path.write_text(json.dumps({"ratings": {
        "level1": 800, "level2": 1050, "level3": 1300,
        "level4": 1550, "level5": 1800}}))
    defaults = dict(size=5, win_length=5, levels=load_levels(path),
                    rng=np.random.default_rng(0))
    defaults.update(kwargs)
    return GomokuApp(**defaults)


async def test_header_shows_the_level_name_and_elo(tmp_path):
    app = app_with_levels(tmp_path, level_index=3)
    async with app.run_test():
        text = str(app.query_one("#level").renderable)
        assert "Club" in text and "1300" in text


async def test_number_keys_switch_level(tmp_path):
    app = app_with_levels(tmp_path, level_index=3)
    async with app.run_test() as pilot:
        await pilot.press("5")
        assert app.level.index == 5
        assert "Expert" in str(app.query_one("#level").renderable)


async def test_unrated_levels_are_labelled_rather_than_invented():
    app = GomokuApp(size=5, win_length=5, levels=load_levels(None),
                    rng=np.random.default_rng(0))
    async with app.run_test():
        assert "unrated" in str(app.query_one("#level").renderable)


async def test_pc_vs_pc_mode_plays_itself_without_human_input(tmp_path):
    app = app_with_levels(tmp_path, mode="pc-vs-pc", level_index=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(6):
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert app.state.ply >= 2


async def test_changing_level_starts_a_new_game(tmp_path):
    app = app_with_levels(tmp_path, level_index=2)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.press("4")
        assert app.state.ply == 0
        assert app.level.index == 4
```

`tests/test_cli.py`:

```python
import json

import numpy as np
import pytest

from gomoku.cli import main
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint, save_checkpoint


def test_no_arguments_prints_usage_and_fails():
    assert main([]) == 2


def test_unknown_subcommand_fails():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_train_subcommand_writes_a_checkpoint(tmp_path):
    code = main([
        "train", "--run-dir", str(tmp_path), "--generations", "1",
        "--games", "2", "--batches", "2", "--batch-size", "8",
        "--size", "5", "--win-length", "4", "--channels", "8", "--blocks", "1",
        "--simulations", "8", "--fast-simulations", "2", "--device", "cpu",
        "--seed", "0",
    ])
    assert code == 0
    assert load_checkpoint(tmp_path / "checkpoint.pt")["generation"] == 1


def test_selfplay_subcommand_reports_game_statistics(tmp_path, capsys):
    code = main([
        "selfplay", "--games", "2", "--size", "5", "--win-length", "4",
        "--simulations", "8", "--fast-simulations", "2", "--seed", "0",
        "--device", "cpu",
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "games" in output and "black win rate" in output


def test_arena_subcommand_writes_elo(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(checkpoint, PolicyValueNet(config), None, 1, config, {})
    code = main([
        "arena", "--checkpoint", str(checkpoint), "--out", str(tmp_path / "elo.json"),
        "--games-per-pair", "2", "--size", "5", "--win-length", "4",
        "--device", "cpu", "--seed", "0",
    ])
    assert code == 0
    payload = json.loads((tmp_path / "elo.json").read_text())
    assert payload["ratings"]["heuristic"] == pytest.approx(1200.0, abs=1e-6)
    assert set(payload["ratings"]) >= {f"level{i}" for i in range(1, 6)}


def test_play_subcommand_explains_itself_without_a_terminal(monkeypatch, capsys):
    """`play` needs a TTY; without one it must explain, not crash."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert main(["play"]) == 0
    assert "interactive terminal" in capsys.readouterr().out


def test_play_subcommand_builds_everything_without_launching(tmp_path):
    """--no-launch exercises checkpoint and level loading with no UI."""
    assert main(["play", "--no-launch", "--checkpoint", str(tmp_path / "none.pt"),
                 "--elo", str(tmp_path / "none.json")]) == 0
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_engine.py tests/test_cli.py tests/test_tui_levels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.engine'`

- [ ] **Step 4: Implement the engine helpers**

`gomoku/engine.py`:

```python
"""Assembling a playable opponent from a checkpoint, or from nothing.

The TUI must be usable before any network exists, so a missing checkpoint
degrades to the heuristic bot rather than failing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gomoku.difficulty import Level, make_player
from gomoku.evaluator import Evaluator, NetEvaluator
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint
from gomoku.players import HeuristicPlayer, Player

log = logging.getLogger(__name__)


def load_evaluator(
    checkpoint: str | Path | None,
    device: str | None = None,
) -> tuple[Evaluator | None, int | None]:
    if checkpoint is None:
        return None, None
    path = Path(checkpoint)
    if not path.exists():
        log.info("no checkpoint at %s; falling back to the heuristic bot", path)
        return None, None
    payload = load_checkpoint(path, map_location="cpu")
    net = PolicyValueNet(NetConfig(**payload["config"]))
    net.load_state_dict(payload["model"])
    return NetEvaluator(net, device=device), int(payload["generation"])


def build_opponent(
    level: Level,
    evaluator: Evaluator | None,
    rng: np.random.Generator,
) -> Player:
    if evaluator is None:
        return HeuristicPlayer(rng, name=f"heuristic-level{level.index}")
    return make_player(level, evaluator, rng)
```

- [ ] **Step 5: Extend the TUI**

In `gomoku/tui/app.py`, add these imports:

```python
import numpy as np

from gomoku.difficulty import LEVELS, Level, load_levels
from gomoku.engine import build_opponent
from gomoku.evaluator import Evaluator

# Distinguishes "caller passed None, meaning a human" from "caller said nothing".
_UNSET = object()
```

Replace `GomokuApp.__init__` and `compose`, and add the level binding:

```python
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
        self.size = size
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

    def action_set_level(self, index: int) -> None:
        """Switch difficulty. Changing opponent mid-game would be unfair, so
        this starts a fresh game."""
        self.level = self.levels[index - 1]
        if not self.explicit_players:
            self.players = self._players_for_mode()
        self.action_new_game()
```

Extend `refresh_view` to render the level line:

```python
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
```

And update `run_tui` to accept the same arguments:

```python
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
```

Note `Level.label()` already renders "3. Club (1300 ELO)" or "3. Club (unrated)", which is what the tests assert.

- [ ] **Step 6: Implement the command line**

`gomoku/cli.py`:

```python
"""Command line: play, train, selfplay, arena."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from gomoku.arena import MatchConfig, measure_levels
from gomoku.difficulty import DEFAULT_ELO_PATH, load_levels
from gomoku.engine import load_evaluator
from gomoku.evaluator import UniformEvaluator
from gomoku.mcts import SearchConfig
from gomoku.net import NetConfig
from gomoku.selfplay import SelfPlayConfig, play_games
from gomoku.train import TrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gomoku", description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    def board_arguments(sub):
        sub.add_argument("--size", type=int, default=9)
        sub.add_argument("--win-length", type=int, default=5)
        sub.add_argument("--device", default=None)
        sub.add_argument("--seed", type=int, default=None)

    play = subparsers.add_parser("play", help="play in the terminal")
    board_arguments(play)
    play.add_argument("--checkpoint", default="runs/default/checkpoint.pt")
    play.add_argument("--elo", default=str(DEFAULT_ELO_PATH))
    play.add_argument("--level", type=int, default=3, choices=range(1, 6))
    play.add_argument("--mode", default="human-vs-pc",
                      choices=["human-vs-pc", "pc-vs-pc"])
    play.add_argument("--no-launch", action="store_true",
                      help="build everything but do not start the UI")

    trainer = subparsers.add_parser("train", help="run self-play training")
    board_arguments(trainer)
    trainer.add_argument("--run-dir", default="runs/default")
    trainer.add_argument("--generations", type=int, default=100)
    trainer.add_argument("--games", type=int, default=64)
    trainer.add_argument("--batches", type=int, default=200)
    trainer.add_argument("--batch-size", type=int, default=256)
    trainer.add_argument("--channels", type=int, default=64)
    trainer.add_argument("--blocks", type=int, default=6)
    trainer.add_argument("--simulations", type=int, default=600)
    trainer.add_argument("--fast-simulations", type=int, default=100)
    trainer.add_argument("--full-fraction", type=float, default=0.25)
    trainer.add_argument("--games-in-flight", type=int, default=32)
    trainer.add_argument("--no-resume", action="store_true")

    selfplay = subparsers.add_parser("selfplay", help="generate games only")
    board_arguments(selfplay)
    selfplay.add_argument("--checkpoint", default=None)
    selfplay.add_argument("--games", type=int, default=8)
    selfplay.add_argument("--simulations", type=int, default=600)
    selfplay.add_argument("--fast-simulations", type=int, default=100)

    arena = subparsers.add_parser("arena", help="measure level ratings")
    board_arguments(arena)
    arena.add_argument("--checkpoint", default="runs/default/checkpoint.pt")
    arena.add_argument("--out", default=str(DEFAULT_ELO_PATH))
    arena.add_argument("--games-per-pair", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_usage()
        return 2
    rng = np.random.default_rng(args.seed)
    return {
        "play": _play,
        "train": _train,
        "selfplay": _selfplay,
        "arena": _arena,
    }[args.command](args, rng)


def _play(args, rng) -> int:
    from gomoku.tui.app import run_tui

    evaluator, generation = load_evaluator(args.checkpoint, args.device)
    if evaluator is None:
        print("No checkpoint found; playing against the heuristic bot.")
    else:
        print(f"Loaded checkpoint trained to generation {generation}.")
    levels = load_levels(args.elo)
    if args.no_launch or not sys.stdout.isatty():
        if args.no_launch:
            return 0
        print("gomoku play needs an interactive terminal.")
        return 0
    run_tui(size=args.size, win_length=args.win_length, levels=levels,
            level_index=args.level, evaluator=evaluator, rng=rng, mode=args.mode)
    return 0


def _selfplay_config(args) -> SelfPlayConfig:
    return SelfPlayConfig(
        size=args.size,
        win_length=args.win_length,
        full_simulations=args.simulations,
        fast_simulations=args.fast_simulations,
        full_fraction=getattr(args, "full_fraction", 0.25),
        games_in_flight=getattr(args, "games_in_flight", 32),
        search=SearchConfig(dirichlet_alpha=10.0 / (args.size * args.size)),
    )


def _train(args, rng) -> int:
    config = TrainConfig(
        generations=args.generations,
        games_per_generation=args.games,
        batch_size=args.batch_size,
        batches_per_generation=args.batches,
        run_dir=args.run_dir,
        device=args.device,
        net=NetConfig(channels=args.channels, blocks=args.blocks),
        selfplay=_selfplay_config(args),
    )
    path = train(config, rng, resume=not args.no_resume)
    print(f"checkpoint: {path}")
    return 0


def _selfplay(args, rng) -> int:
    evaluator, _ = load_evaluator(args.checkpoint, args.device)
    if evaluator is None:
        evaluator = UniformEvaluator()
    samples, stats = play_games(evaluator, args.games, _selfplay_config(args), rng)
    print(f"{stats.n_games} games, {len(samples)} samples")
    print(f"black win rate {stats.black_win_rate:.2f}, "
          f"mean length {stats.mean_length:.1f}, "
          f"distinct openings {len(stats.openings)}")
    return 0


def _arena(args, rng) -> int:
    evaluator, _ = load_evaluator(args.checkpoint, args.device)
    if evaluator is None:
        print("No checkpoint found; nothing to rate.")
        return 1
    config = MatchConfig(size=args.size, win_length=args.win_length,
                         games_per_pair=args.games_per_pair)
    ratings = measure_levels(evaluator, config, rng, args.out)
    for name, rating in sorted(ratings.items(), key=lambda item: item[1]):
        print(f"{name:>12}  {rating:7.0f}")
    print(f"written to {args.out}")
    return 0
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_engine.py tests/test_cli.py tests/test_tui_levels.py tests/test_tui.py -q`
Expected: PASS. `tests/test_tui.py` from Task 5 must still pass: the explicit `black=`/`white=` arguments continue to override the mode.

- [ ] **Step 8: Play it**

Run: `.venv/bin/gomoku play --size 9 --level 3`
Expected: a 9x9 board with a level line reading `Level 3. Club (unrated)`, an opponent that responds, and `1`-`5` switching difficulty.

- [ ] **Step 9: Commit**

```bash
git add gomoku/engine.py gomoku/cli.py gomoku/tui/app.py tests/test_engine.py tests/test_cli.py tests/test_tui_levels.py
git commit -m "feat: difficulty levels in the TUI and a gomoku command line"
```

---

### Task 16: Multiprocess evaluator for long runs

**Files:**
- Create: `gomoku/server_evaluator.py`
- Modify: `gomoku/train.py` (accept `workers` in `TrainConfig`)
- Modify: `gomoku/cli.py` (add `--workers`)
- Test: `tests/test_server_evaluator.py`

**Interfaces:**
- Consumes: `gomoku.evaluator.Evaluator`, `gomoku.net.{PolicyValueNet, NetConfig}`.
- Produces: `InferenceServer(net_config, state_dict, device=None, max_batch=256, wait_ms=2.0)` with `start()`, `stop()`, `client() -> ServerEvaluator`, and `refresh(state_dict)`; `ServerEvaluator(request_queue, response_queue, worker_id)` implementing `Evaluator`; `run_selfplay_workers(server, n_workers, games_per_worker, config, seed) -> tuple[list[Sample], GameStats]`.

Phase 2, and strictly an optimisation: everything above the `Evaluator` interface is unchanged. Do not start this until a real 9x9 run is producing sensible metrics with the in-process evaluator, because it makes debugging materially harder.

- [ ] **Step 1: Write the failing tests**

`tests/test_server_evaluator.py`:

```python
import numpy as np
import pytest

from gomoku.evaluator import Evaluator, NetEvaluator
from gomoku.game import N_PLANES
from gomoku.net import NetConfig, PolicyValueNet
from gomoku.selfplay import SelfPlayConfig
from gomoku.server_evaluator import InferenceServer, run_selfplay_workers

pytestmark = pytest.mark.slow


@pytest.fixture
def server():
    config = NetConfig(channels=8, blocks=1)
    net = PolicyValueNet(config)
    server = InferenceServer(config, net.state_dict(), device="cpu", max_batch=32)
    server.start()
    yield server, net
    server.stop()


def test_client_is_an_evaluator(server):
    instance, _ = server
    assert isinstance(instance.client(), Evaluator)


def test_client_results_match_the_in_process_evaluator(server):
    instance, net = server
    x = np.random.default_rng(0).random((4, N_PLANES, 9, 9)).astype(np.float32)
    expected_policy, expected_value = NetEvaluator(net, device="cpu").evaluate(x)
    policy, value = instance.client().evaluate(x)
    assert np.allclose(policy, expected_policy, atol=1e-4)
    assert np.allclose(value, expected_value, atol=1e-4)


def test_several_clients_are_served_concurrently(server):
    instance, _ = server
    clients = [instance.client() for _ in range(3)]
    x = np.zeros((2, N_PLANES, 5, 5), dtype=np.float32)
    results = [client.evaluate(x) for client in clients]
    assert all(policy.shape == (2, 25) for policy, _ in results)


def test_stopping_twice_is_safe(server):
    instance, _ = server
    instance.stop()
    instance.stop()


def test_workers_generate_the_requested_games(server):
    instance, _ = server
    config = SelfPlayConfig(size=5, win_length=4, full_simulations=8,
                            fast_simulations=2, games_in_flight=2,
                            opening_plies=(2, 2))
    samples, stats = run_selfplay_workers(instance, n_workers=2,
                                          games_per_worker=2, config=config,
                                          seed=0)
    assert stats.n_games == 4
    assert all(s.encoded.shape == (N_PLANES, 5, 5) for s in samples[:5])


def test_worker_output_matches_single_process_shapes(server):
    instance, _ = server
    config = SelfPlayConfig(size=5, win_length=4, full_simulations=8,
                            fast_simulations=2, full_fraction=1.0,
                            games_in_flight=1, opening_plies=(2, 2))
    samples, stats = run_selfplay_workers(instance, n_workers=1,
                                          games_per_worker=1, config=config,
                                          seed=1)
    assert stats.n_games == 1
    assert samples and np.isclose(samples[0].policy.sum(), 1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_server_evaluator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gomoku.server_evaluator'`

- [ ] **Step 3: Implement the inference server**

`gomoku/server_evaluator.py`:

```python
"""Phase two throughput: MCTS in worker processes, one batched model.

Tree search is pure Python and single-core; the model wants large batches on
one device. Splitting them lets both run flat out -- workers descend trees in
parallel while the server coalesces their leaves into batches large enough to
keep the GPU busy.

The interface is unchanged: a worker holds a `ServerEvaluator`, which is an
`Evaluator` like any other, so MCTS, self-play and training are untouched.
"""

from __future__ import annotations

import logging
import queue
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gomoku.evaluator import Evaluator
from gomoku.net import NetConfig, PolicyValueNet, select_device
from gomoku.selfplay import GameStats, SelfPlayConfig, play_games

log = logging.getLogger(__name__)

_STOP = "stop"
_REFRESH = "refresh"


class ServerEvaluator(Evaluator):
    """A client handle. Sends states to the server and waits for its results."""

    def __init__(self, request_queue, response_queue, worker_id: int) -> None:
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.request_queue.put((self.worker_id, np.ascontiguousarray(encoded)))
        return self.response_queue.get()


def _serve(net_config, state_dict, device, max_batch, wait_seconds,
           request_queue, response_queues, control_queue) -> None:
    net = PolicyValueNet(net_config)
    net.load_state_dict(state_dict)
    torch_device = select_device(device)
    net = net.to(torch_device).eval()
    while True:
        try:
            control = control_queue.get_nowait()
        except queue.Empty:
            control = None
        if control is not None:
            action, payload = control
            if action == _STOP:
                return
            if action == _REFRESH:
                net.load_state_dict(payload)
                net = net.to(torch_device).eval()

        pending: list[tuple[int, np.ndarray]] = []
        try:
            pending.append(request_queue.get(timeout=0.05))
        except queue.Empty:
            continue
        # Coalesce whatever else has arrived, up to the batch limit.
        deadline = time.monotonic() + wait_seconds
        total = pending[0][1].shape[0]
        while total < max_batch and time.monotonic() < deadline:
            try:
                item = request_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.0005)
                continue
            pending.append(item)
            total += item[1].shape[0]

        batch = np.concatenate([states for _, states in pending], axis=0)
        with torch.inference_mode():
            logits, values = net(torch.from_numpy(batch).to(torch_device))
            policies = torch.softmax(logits.float(), dim=1).cpu().numpy()
            values = values.float().cpu().numpy()
        offset = 0
        for worker_id, states in pending:
            count = states.shape[0]
            response_queues[worker_id].put(
                (policies[offset : offset + count].astype(np.float32),
                 values[offset : offset + count].astype(np.float32))
            )
            offset += count


class InferenceServer:
    def __init__(self, net_config: NetConfig, state_dict, device=None,
                 max_batch: int = 256, wait_ms: float = 2.0,
                 max_clients: int = 16) -> None:
        self.context = mp.get_context("spawn")
        self.net_config = net_config
        self.state_dict = {k: v.cpu() for k, v in state_dict.items()}
        self.device = device
        self.max_batch = max_batch
        self.wait_seconds = wait_ms / 1000.0
        self.request_queue = self.context.Queue()
        self.control_queue = self.context.Queue()
        self.response_queues = [self.context.Queue() for _ in range(max_clients)]
        self._next_client = 0
        self._process = None

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = self.context.Process(
            target=_serve,
            args=(self.net_config, self.state_dict, self.device, self.max_batch,
                  self.wait_seconds, self.request_queue, self.response_queues,
                  self.control_queue),
            daemon=True,
        )
        self._process.start()

    def client(self) -> ServerEvaluator:
        if self._next_client >= len(self.response_queues):
            raise RuntimeError("no response queues left; raise max_clients")
        evaluator = ServerEvaluator(
            self.request_queue, self.response_queues[self._next_client],
            self._next_client,
        )
        self._next_client += 1
        return evaluator

    def refresh(self, state_dict) -> None:
        self.control_queue.put((_REFRESH, {k: v.cpu() for k, v in state_dict.items()}))

    def stop(self) -> None:
        if self._process is None:
            return
        self.control_queue.put((_STOP, None))
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None


def _worker(server_args, worker_id, games, config, seed, output_queue) -> None:
    request_queue, response_queue = server_args
    evaluator = ServerEvaluator(request_queue, response_queue, worker_id)
    samples, stats = play_games(evaluator, games, config,
                                np.random.default_rng(seed))
    output_queue.put((samples, stats))


def run_selfplay_workers(
    server: InferenceServer,
    n_workers: int,
    games_per_worker: int,
    config: SelfPlayConfig,
    seed: int,
) -> tuple[list, GameStats]:
    """Generate games in `n_workers` processes, all served by one model."""
    output_queue = server.context.Queue()
    processes = []
    for index in range(n_workers):
        evaluator = server.client()
        process = server.context.Process(
            target=_worker,
            args=((evaluator.request_queue, evaluator.response_queue),
                  evaluator.worker_id, games_per_worker, config,
                  seed + index, output_queue),
            daemon=True,
        )
        process.start()
        processes.append(process)

    samples: list = []
    combined = GameStats()
    for _ in processes:
        worker_samples, worker_stats = output_queue.get()
        samples.extend(worker_samples)
        combined.black_wins += worker_stats.black_wins
        combined.white_wins += worker_stats.white_wins
        combined.draws += worker_stats.draws
        combined.lengths.extend(worker_stats.lengths)
        combined.openings |= worker_stats.openings
    for process in processes:
        process.join(timeout=10)
    return samples, combined
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_server_evaluator.py -q`
Expected: PASS. These spawn processes and are slow; they carry the `slow` marker for that reason.

- [ ] **Step 5: Wire it into training**

Add to `TrainConfig`:

```python
    workers: int = 0          # 0 keeps self-play in this process
    inference_max_batch: int = 256
```

In `train`, replace the self-play call with:

```python
        if config.workers > 0:
            from gomoku.server_evaluator import InferenceServer, run_selfplay_workers

            server = InferenceServer(config.net, net.state_dict(),
                                     device=config.device,
                                     max_batch=config.inference_max_batch)
            server.start()
            try:
                per_worker = max(1, config.games_per_generation // config.workers)
                samples, stats = run_selfplay_workers(
                    server, config.workers, per_worker, config.selfplay,
                    seed=int(rng.integers(0, 2**31)),
                )
            finally:
                server.stop()
        else:
            samples, stats = play_games(
                evaluator, config.games_per_generation, config.selfplay, rng
            )
```

Add `--workers` to the `train` subparser in `cli.py` (`type=int, default=0`) and pass it into `TrainConfig`.

- [ ] **Step 6: Add a training test covering the worker path**

Append to `tests/test_train.py`:

```python
@pytest.mark.slow
def test_train_with_workers_produces_the_same_artefacts(tmp_path):
    path = train(tiny_config(tmp_path, workers=2), np.random.default_rng(0))
    assert load_checkpoint(path)["generation"] == 2
    assert MetricsWriter(tmp_path / "metrics.jsonl").read_all()
```

Add `import pytest` at the top of that file.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add gomoku/server_evaluator.py gomoku/train.py gomoku/cli.py tests/
git commit -m "feat: multiprocess inference server for long training runs"
```

---

## Running it for real

After Task 15 the system is complete; Task 16 is throughput. The intended sequence on this machine:

```bash
# A short run first, to see the diagnostics move in the right direction.
.venv/bin/gomoku train --run-dir runs/9x9 --size 9 --generations 5 \
    --games 64 --batches 200

# Check the metrics before committing to an overnight run.
.venv/bin/python -c "
from gomoku.metrics import MetricsWriter
for r in MetricsWriter('runs/9x9/metrics.jsonl').read_all():
    print(r['generation'], round(r['policy_loss'], 3), round(r['value_loss'], 3),
          'parity baseline', round(r['value_baseline_parity'], 3),
          'entropy', round(r['policy_entropy'], 2),
          'black', round(r['black_win_rate'], 2))
"
```

What to look for, per spec §3:

- `value_loss` should fall **below** `value_baseline_parity`. If it plateaus at or above it, the value head has learned only "black is winning" — check that openings are genuinely random before touching anything else.
- `policy_entropy` should fall gradually, not collapse toward zero within a few generations.
- `black_win_rate` should be high but stable; `distinct_openings` should stay close to the number of games.

Then the long run:

```bash
.venv/bin/gomoku train --run-dir runs/9x9 --generations 200 --games 128 \
    --workers 6 --simulations 600 --fast-simulations 100
.venv/bin/gomoku arena --checkpoint runs/9x9/checkpoint.pt --out runs/elo.json \
    --games-per-pair 40
.venv/bin/gomoku play --checkpoint runs/9x9/checkpoint.pt --level 4
```

Training resumes from the checkpoint automatically, so the run can be interrupted and restarted.
