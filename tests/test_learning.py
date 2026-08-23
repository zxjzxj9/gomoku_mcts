"""Proof that the whole pipeline learns, on a board small enough to be fast.

3x3 with three in a row is tic-tac-toe: a draw under perfect play, trivially
solvable, and it exercises self-play, the replay buffer, the training loop,
checkpointing and the evaluator exactly as the 9x9 game does.
"""

import numpy as np
import pytest
import torch

from gomoku.evaluator import NetEvaluator
from gomoku.game import GameState
from gomoku.mcts import SearchConfig
from gomoku.metrics import MetricsWriter
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint
from gomoku.players import MCTSPlayer, RandomPlayer
from gomoku.selfplay import SelfPlayConfig
from gomoku.train import TrainConfig, train

pytestmark = pytest.mark.slow

SIZE, WIN = 3, 3


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("ttt")
    config = TrainConfig(
        generations=12,
        games_per_generation=32,
        batch_size=64,
        batches_per_generation=40,
        learning_rate=3e-3,
        buffer_capacity=20_000,
        run_dir=str(run_dir),
        device="cpu",
        net=NetConfig(channels=16, blocks=2),
        selfplay=SelfPlayConfig(
            size=SIZE,
            win_length=WIN,
            opening_plies=(1, 2),
            opening_radius=2,
            full_simulations=40,
            fast_simulations=10,
            full_fraction=0.5,
            temperature_plies=2,
            games_in_flight=16,
            search=SearchConfig(dirichlet_alpha=0.6),
        ),
    )
    path = train(config, np.random.default_rng(0))
    payload = load_checkpoint(path)
    net = PolicyValueNet(NetConfig(**payload["config"]))
    net.load_state_dict(payload["model"])
    records = MetricsWriter(run_dir / "metrics.jsonl").read_all()
    return NetEvaluator(net, device="cpu"), records


def play(black, white, rng):
    state = GameState.new(SIZE, WIN)
    players = [black, white]
    while not state.is_terminal():
        state = state.play(players[state.ply % 2].select_move(state))
    return state.winner


def test_searching_agent_never_loses_to_random(trained):
    evaluator, _ = trained
    losses = 0
    for game in range(20):
        rng = np.random.default_rng(1000 + game)
        agent = MCTSPlayer(evaluator, simulations=100, rng=rng)
        random_player = RandomPlayer(rng)
        if game % 2 == 0:
            winner = play(agent, random_player, rng)
            losses += winner == 2
        else:
            winner = play(random_player, agent, rng)
            losses += winner == 1
    assert losses == 0


def test_raw_network_policy_is_much_better_than_random(trained):
    """No search at all: this measures what the network itself learned."""
    evaluator, _ = trained
    wins = losses = 0
    for game in range(40):
        rng = np.random.default_rng(2000 + game)
        agent = MCTSPlayer(evaluator, simulations=0, policy_only=True,
                           temperature=0.05, rng=rng)
        random_player = RandomPlayer(rng)
        agent_is_black = game % 2 == 0
        winner = play(agent, random_player, rng) if agent_is_black \
            else play(random_player, agent, rng)
        agent_colour = 1 if agent_is_black else 2
        wins += winner == agent_colour
        losses += winner not in (0, agent_colour)
    assert wins >= 24
    assert losses <= 4


def test_agent_takes_an_immediate_win(trained):
    evaluator, _ = trained
    # X at 0 and 1; O at 3 and 4. X to move wins at 2.
    state = GameState.new(SIZE, WIN)
    for move in (0, 3, 1, 4):
        state = state.play(move)
    agent = MCTSPlayer(evaluator, simulations=100, rng=np.random.default_rng(0))
    assert agent.select_move(state) == 2


def test_agent_blocks_an_immediate_loss(trained):
    evaluator, _ = trained
    # X at 0 and 1 threatens 2; O to move must block.
    state = GameState.new(SIZE, WIN)
    for move in (0, 4, 1):
        state = state.play(move)
    agent = MCTSPlayer(evaluator, simulations=100, rng=np.random.default_rng(0))
    assert agent.select_move(state) == 2


def test_self_play_between_trained_agents_mostly_draws(trained):
    """Tic-tac-toe is a draw under perfect play."""
    evaluator, _ = trained
    draws = 0
    for game in range(10):
        rng = np.random.default_rng(3000 + game)
        draws += play(
            MCTSPlayer(evaluator, simulations=150, rng=rng),
            MCTSPlayer(evaluator, simulations=150, rng=rng),
            rng,
        ) == 0
    assert draws >= 7


def test_losses_fall_over_training(trained):
    _, records = trained
    early = np.mean([r["policy_loss"] for r in records[:3]])
    late = np.mean([r["policy_loss"] for r in records[-3:]])
    assert late < early


def test_value_head_beats_the_parity_baseline(trained):
    """The §3 diagnostic: the value head must learn more than 'black is winning'.

    Compare against `value_loss_on_fresh`, not `value_loss`: the latter is
    averaged over the whole replay window during optimisation, while the
    baselines describe this generation's fresh samples."""
    _, records = trained
    final = records[-1]
    assert final["value_loss_on_fresh"] < final["value_baseline_parity"]


def test_exploration_does_not_collapse(trained):
    _, records = trained
    assert all(r["policy_entropy"] > 0.05 for r in records)
