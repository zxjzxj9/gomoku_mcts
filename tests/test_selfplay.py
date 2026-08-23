import numpy as np
import pytest

from gomoku.evaluator import Evaluator, UniformEvaluator
from gomoku.game import N_PLANES
from gomoku.mcts import SearchConfig
from gomoku.selfplay import (
    PLAY_PREFIX_PLIES,
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
    # Row 1, column 1 is flat index 1 * 5 + 1 == 6, so the marked stone and
    # the marked policy cell are the same cell. They must move together.
    encoded[0, 1, 1] = 1.0
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


def test_temperature_cutoff_defaults_to_the_board_size():
    assert SelfPlayConfig(size=9).temperature_cutoff() == 9
    assert SelfPlayConfig(size=9, temperature_plies=3).temperature_cutoff() == 3


def test_the_temperature_schedule_actually_changes_play():
    """A schedule that turns greedy immediately must produce different games
    from one that stays hot throughout.

    Running the SAME config twice under one seed asserts nothing -- two
    identical runs agree whatever the schedule does, including a schedule
    that ignores `temperature_plies` entirely."""
    greedy = small_config(temperature_plies=0, full_fraction=1.0, games_in_flight=1)
    hot = small_config(temperature_plies=99, temperature=2.0, full_fraction=1.0,
                       games_in_flight=1)
    greedy_samples, _ = play_games(UniformEvaluator(), 4, greedy,
                                   np.random.default_rng(9))
    hot_samples, _ = play_games(UniformEvaluator(), 4, hot,
                                np.random.default_rng(9))
    assert [int(np.argmax(s.policy)) for s in greedy_samples] != \
        [int(np.argmax(s.policy)) for s in hot_samples]


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


class CollapsedEvaluator(Evaluator):
    """A policy that has collapsed onto one line: the lowest-numbered cell.

    This is what section 3's diversity failure looks like from the outside --
    the network plays the same moves whatever the position -- and it is the
    thing the diversity diagnostics exist to catch.
    """

    def evaluate(self, encoded):
        n_states = encoded.shape[0]
        n_cells = encoded.shape[-1] * encoded.shape[-2]
        weights = np.exp(-np.arange(n_cells, dtype=np.float64))
        policies = np.tile(weights / weights.sum(), (n_states, 1))
        return policies.astype(np.float32), np.zeros(n_states, dtype=np.float32)


def test_distinct_play_prefixes_collapse_while_openings_stay_diverse():
    """The contrast is the whole point.

    `distinct_openings` counts moves the RNG imposed, so it reads healthy
    however completely play collapses. Only the prefixes of moves the policy
    chose can tell the difference.
    """
    config = SelfPlayConfig(
        size=9, win_length=5, opening_plies=(2, 2), opening_radius=1,
        full_simulations=4, fast_simulations=2, full_fraction=0.0,
        temperature_plies=0, games_in_flight=8,
        search=SearchConfig(add_noise=False),
    )
    _, stats = play_games(CollapsedEvaluator(), 8, config,
                          np.random.default_rng(0))
    assert len(stats.openings) >= 7
    assert len(stats.play_prefixes) == 1
    assert len(next(iter(stats.play_prefixes))) == PLAY_PREFIX_PLIES


def test_distinct_play_prefixes_stay_high_when_play_is_varied():
    config = small_config(full_fraction=0.0, games_in_flight=8)
    _, stats = play_games(UniformEvaluator(), 8, config, np.random.default_rng(1))
    assert len(stats.play_prefixes) >= 6


def test_length_quantiles_describe_the_distribution():
    stats = GameStats(lengths=[4, 4, 10, 10, 10])
    assert stats.length_quantiles() == [4.0, 4.0, 10.0, 10.0, 10.0]
    # A mean of 7.6 says "medium games"; the quantiles show there are none.
    assert stats.mean_length == pytest.approx(7.6)


def test_length_quantiles_of_no_games_are_zero():
    assert GameStats().length_quantiles() == [0.0] * 5


def test_opening_radius_means_exactly_that_many_cells_from_the_centre():
    """The name is the contract: radius r, not r + 1."""
    config = SelfPlayConfig(size=9, opening_radius=1, opening_plies=(2, 2))
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(40):
        _, opening = random_opening(rng, config)
        for move in opening:
            row, col = divmod(move, 9)
            assert abs(row - 4) <= 1 and abs(col - 4) <= 1
            seen.add((row, col))
    # The whole 3x3 window is reachable, so the radius is not too small either.
    assert len(seen) == 9
