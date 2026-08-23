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
