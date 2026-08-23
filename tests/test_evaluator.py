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
