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
