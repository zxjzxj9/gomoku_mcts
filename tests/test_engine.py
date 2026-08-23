import numpy as np
import torch

from gomoku.difficulty import LEVELS
from gomoku.engine import build_opponent, load_evaluator
from gomoku.evaluator import NetEvaluator
from gomoku.game import GameState
from gomoku.net import NetConfig, PolicyValueNet, save_checkpoint
from gomoku.players import HeuristicPlayer, MCTSPlayer


def test_load_evaluator_without_a_checkpoint_returns_nothing(tmp_path):
    evaluator, generation = load_evaluator(tmp_path / "absent.pt", device="cpu")
    assert evaluator is None and generation is None


def test_load_evaluator_reads_a_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, PolicyValueNet(config), None, 7, config, {})
    evaluator, generation = load_evaluator(path, device="cpu")
    assert isinstance(evaluator, NetEvaluator)
    assert generation == 7


def test_build_opponent_falls_back_to_the_heuristic_without_a_network():
    player = build_opponent(LEVELS[4], None, np.random.default_rng(0))
    assert isinstance(player, HeuristicPlayer)


def test_build_opponent_uses_the_network_when_present(tmp_path):
    path = tmp_path / "ckpt.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, PolicyValueNet(config), None, 1, config, {})
    evaluator, _ = load_evaluator(path, device="cpu")
    player = build_opponent(LEVELS[1], evaluator, np.random.default_rng(0))
    assert isinstance(player, MCTSPlayer)
    state = GameState.new(size=5, win_length=5)
    assert player.select_move(state) in set(state.legal_moves().tolist())
