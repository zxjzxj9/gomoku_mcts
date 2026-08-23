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
    while any(tree.simulations < simulations for tree in trees):
        batches: list[np.ndarray] = []
        owners: list[tuple[MCTS, int]] = []
        progressed = False
        for tree in trees:
            remaining = simulations - tree.simulations
            if remaining <= 0:
                continue
            before = tree.simulations
            encoded = tree.collect(min(leaf_batch, remaining))
            # A round can legitimately hand back no leaves and still make
            # progress: terminal leaves are backed up inside `collect`.
            if tree.simulations != before:
                progressed = True
            if encoded.shape[0]:
                batches.append(encoded)
                owners.append((tree, encoded.shape[0]))
        if owners:
            priors, values = evaluator.evaluate(np.concatenate(batches, axis=0))
            offset = 0
            for tree, count in owners:
                tree.apply(priors[offset : offset + count],
                           values[offset : offset + count])
                offset += count
            progressed = True
        if not progressed:
            # Nothing advanced anywhere this round; the remaining trees cannot
            # reach their budget. Stop rather than spin.
            return


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
