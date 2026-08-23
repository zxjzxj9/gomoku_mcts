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


def test_save_shard_writes_only_the_samples_it_is_given(tmp_path):
    buffer = ReplayBuffer(capacity=100, directory=tmp_path)
    buffer.add(make_samples(6))
    fresh = make_samples(2)
    buffer.add(fresh)
    buffer.save_shard(generation=0, samples=fresh)

    reader = ReplayBuffer(capacity=100, directory=tmp_path)
    assert reader.load_shards() == 2


def test_prune_shards_keeps_just_enough_to_refill_the_buffer(tmp_path):
    writer = ReplayBuffer(capacity=100, directory=tmp_path)
    for generation in range(5):
        batch = make_samples(4)
        writer.add(batch)
        writer.save_shard(generation, samples=batch)

    # Newest-first, three shards (12 samples) are the fewest covering 10.
    assert writer.prune_shards(keep_samples=10) == 2
    assert len(list(tmp_path.glob("shard_*.npz"))) == 3


def test_prune_shards_keeps_everything_when_there_is_too_little(tmp_path):
    writer = ReplayBuffer(capacity=100, directory=tmp_path)
    writer.add(make_samples(4))
    writer.save_shard(0)
    assert writer.prune_shards(keep_samples=1000) == 0
    assert len(list(tmp_path.glob("shard_*.npz"))) == 1


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
