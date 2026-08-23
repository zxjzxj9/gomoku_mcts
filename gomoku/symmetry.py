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
