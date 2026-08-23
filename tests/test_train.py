from unittest import mock

import numpy as np
import pytest
import torch

from gomoku.game import N_PLANES
from gomoku.mcts import SearchConfig
from gomoku.metrics import MetricsWriter
from gomoku.net import NetConfig, load_checkpoint
from gomoku.replay import ReplayBuffer
from gomoku.selfplay import Sample, SelfPlayConfig
from gomoku.symmetry import N_SYMMETRIES
from gomoku.train import TrainConfig, _split_holdout, loss_terms, train


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
        for key in ("generation", "policy_loss", "value_loss",
                    "value_loss_on_fresh", "policy_entropy",
                    "black_win_rate", "value_baseline_constant",
                    "value_baseline_parity", "mean_game_length",
                    "distinct_openings", "buffer_size", "seconds"):
            assert key in record
        # Presence alone would pass against an all-zero or NaN record.
        assert record["policy_loss"] > 0.0
        assert record["value_baseline_constant"] > 0.0
        assert 0.0 <= record["black_win_rate"] <= 1.0
        assert record["policy_entropy"] > 0.0
        assert record["buffer_size"] > 0
        assert record["distinct_openings"] >= 1


def test_train_resumes_from_the_last_checkpoint(tmp_path):
    train(tiny_config(tmp_path), np.random.default_rng(0))
    path = train(tiny_config(tmp_path, generations=4), np.random.default_rng(1))
    assert load_checkpoint(path)["generation"] == 4
    assert len(MetricsWriter(tmp_path / "metrics.jsonl").read_all()) == 4


def test_train_from_scratch_ignores_an_existing_checkpoint(tmp_path):
    """The generation counter alone cannot detect this.

    With a checkpoint at generation 2 and `generations=2`, a run that wrongly
    resumed would execute `range(2, 2)` -- nothing at all -- and leave the
    checkpoint reading 2, exactly what a correct from-scratch run produces. So
    ask for three generations and check the metrics log records all of them."""
    train(tiny_config(tmp_path), np.random.default_rng(0))
    path = train(tiny_config(tmp_path, generations=3), np.random.default_rng(0),
                 resume=False)
    assert load_checkpoint(path)["generation"] == 3
    records = MetricsWriter(tmp_path / "metrics.jsonl").read_all()
    assert [r["generation"] for r in records] == [0, 1, 0, 1, 2]


def test_replay_shards_are_written(tmp_path):
    train(tiny_config(tmp_path), np.random.default_rng(0))
    assert list((tmp_path / "replay").glob("shard_*.npz"))


@pytest.mark.slow
def test_train_with_workers_produces_the_same_artefacts(tmp_path):
    path = train(tiny_config(tmp_path, workers=2), np.random.default_rng(0))
    assert load_checkpoint(path)["generation"] == 2
    assert MetricsWriter(tmp_path / "metrics.jsonl").read_all()


def test_split_holdout_takes_whole_positions():
    """Augmentation emits eight consecutive entries per position.

    Splitting by sample would leave seven symmetries of every "held out"
    position in the training set, and the holdout would measure nothing.
    """
    samples = list(range(80))
    holdout, trainable = _split_holdout(samples, every=10)
    assert holdout == list(range(8))          # exactly position 0, all eight
    assert trainable == list(range(8, 80))
    assert len(holdout) + len(trainable) == len(samples)
    assert len(holdout) == pytest.approx(len(samples) * 0.1)


def test_split_holdout_of_nothing_is_empty():
    assert _split_holdout([]) == ([], [])


def test_the_holdout_is_kept_out_of_training_and_added_afterwards(tmp_path):
    """The parity check is only honest if the value head has not been fit
    on the positions it is scored on."""
    sizes_during_training = []
    original = ReplayBuffer.sample_batch

    def spy(self, batch_size, rng):
        sizes_during_training.append(len(self))
        return original(self, batch_size, rng)

    with mock.patch.object(ReplayBuffer, "sample_batch", spy):
        train(tiny_config(tmp_path, generations=1), np.random.default_rng(0))

    records = MetricsWriter(tmp_path / "metrics.jsonl").read_all()
    added = records[0]["samples_added"]
    positions = added // N_SYMMETRIES
    held_out_positions = -(-positions // 10)      # every tenth, position 0 first
    assert sizes_during_training
    assert max(sizes_during_training) == added - held_out_positions * N_SYMMETRIES
    assert max(sizes_during_training) < added
    # Nothing is wasted: the holdout joins the window once it has been used.
    assert records[0]["buffer_size"] == added


def test_metrics_record_the_new_diagnostics(tmp_path):
    train(tiny_config(tmp_path, generations=1), np.random.default_rng(0))
    record = MetricsWriter(tmp_path / "metrics.jsonl").read_all()[0]
    quantiles = record["length_quantiles"]
    assert len(quantiles) == 5
    assert quantiles == sorted(quantiles)
    assert quantiles[0] > 0
    assert record["distinct_play_prefixes"] >= 1
    black, white = record["value_baseline_group_sizes"]
    assert black + white > 0
    assert isinstance(record["value_loss_on_fresh_is_held_out"], bool)


def test_a_generation_too_small_to_hold_out_falls_back_rather_than_crashing(tmp_path):
    """Two games produce a handful of positions; a parity baseline fitted to
    two of them is 0.0 and unpassable. Report the in-sample number and say so."""
    train(tiny_config(tmp_path, generations=1), np.random.default_rng(0))
    record = MetricsWriter(tmp_path / "metrics.jsonl").read_all()[0]
    assert record["value_loss_on_fresh_is_held_out"] is False
    assert record["value_baseline_constant"] > 0.0


def test_a_generation_large_enough_measures_out_of_sample(tmp_path):
    config = tiny_config(tmp_path, generations=1, games_per_generation=16,
                         batches_per_generation=1)
    train(config, np.random.default_rng(0))
    record = MetricsWriter(tmp_path / "metrics.jsonl").read_all()[0]
    assert record["value_loss_on_fresh_is_held_out"] is True
    assert sum(record["value_baseline_group_sizes"]) >= 8 * N_SYMMETRIES


def test_a_from_scratch_run_does_not_inherit_the_old_run_s_shards(tmp_path):
    """`--no-resume` left the old shards on disk, and `save_shard` names by
    generation, so a shorter re-run silently trained on the old population."""
    train(tiny_config(tmp_path), np.random.default_rng(0))
    stale = sorted((tmp_path / "replay").glob("shard_*.npz"))
    assert len(stale) == 2
    train(tiny_config(tmp_path, generations=1), np.random.default_rng(1),
          resume=False)
    remaining = sorted((tmp_path / "replay").glob("shard_*.npz"))
    assert [p.name for p in remaining] == ["shard_000000.npz"]


def test_shards_from_a_different_board_are_skipped_not_fatal(tmp_path):
    """A reused run directory is how two board sizes end up in one buffer."""
    train(tiny_config(tmp_path), np.random.default_rng(0))
    other = ReplayBuffer(capacity=100, directory=tmp_path / "replay")
    other.add([
        Sample(np.zeros((N_PLANES, 9, 9), dtype=np.float32),
               np.full(81, 1.0 / 81, dtype=np.float32), 0.0)
        # Enough that essentially every batch would draw one of them, so
        # the test does not depend on a lucky sample.
        for _ in range(400)
    ])
    other.save_shard(generation=99)

    # Resuming must load the 5x5 shards and skip the 9x9 one rather than
    # dying on whichever batch first draws from both.
    path = train(tiny_config(tmp_path, generations=3), np.random.default_rng(0))
    assert load_checkpoint(path)["generation"] == 3
