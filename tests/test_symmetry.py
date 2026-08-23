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
