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
    """Finding the block needs depth-2 lookahead, and with uniform priors the
    search has no guidance at all -- it must visit each root child, then each
    of that child's replies. On a 9x9 board with ~74 legal moves that costs a
    few thousand simulations, so the position is posed on a small board where
    the refutation is genuinely reachable within the budget."""
    # Black has three at 6, 7, 8 and completes four at 5 or 9. White to move.
    s = state_with([6, 0, 7, 1, 8], size=5, win_length=4)
    counts = search(s, UniformEvaluator(), simulations=600, config=quiet_config())
    assert int(np.argmax(counts)) in (5, 9)


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


def test_backup_flips_perspective_between_plies():
    """Every leaf sits one ply below the root, so it is the opponent to move
    there. An evaluator that calls every position a win for whoever moves in
    it must therefore drive the root value to exactly -1. This pins the sign
    flip in `_backup`, which is the single easiest thing to get backwards.

    The budget must stay below the root's branching factor (25 legal moves
    here), because once the search reaches depth 2 the constant evaluator's
    value flips a second time and cancels the depth-1 signal. That is correct
    behaviour, but it would make this test measure nothing."""

    class OptimisticEvaluator(Evaluator):
        def evaluate(self, encoded):
            n = encoded.shape[0]
            cells = encoded.shape[-1] * encoded.shape[-2]
            return (np.full((n, cells), 1.0 / cells, np.float32),
                    np.full(n, 1.0, np.float32))

    s = GameState.new(size=5, win_length=4)
    tree = MCTS(s, quiet_config())
    run_search([tree], OptimisticEvaluator(), simulations=20)
    assert tree.root_value() == pytest.approx(-1.0)


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
