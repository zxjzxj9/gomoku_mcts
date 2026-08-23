"""The AlphaZero training loop: self-play, fit, checkpoint, repeat.

Every generation logs the section 3 diagnostics alongside the losses, so the
failure modes of a first-player-advantaged game are visible in the metrics
file rather than inferred from a weak final bot.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from gomoku.evaluator import NetEvaluator
from gomoku.game import N_PLANES
from gomoku.metrics import MetricsWriter, baseline_value_losses, policy_entropy
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint, save_checkpoint, select_device
from gomoku.replay import ReplayBuffer
from gomoku.selfplay import SelfPlayConfig, play_games
from gomoku.symmetry import N_SYMMETRIES

log = logging.getLogger(__name__)

# One position in ten is held out of each generation's optimisation batches,
# purely so the parity honesty check is measured out of sample.
HOLDOUT_EVERY = 10
# Below this many held-out positions the baselines are noise -- a parity
# baseline fitted to two or three positions reaches 0.0 and makes the check
# unpassable -- so a tiny generation reports the in-sample number instead,
# flagged by `value_loss_on_fresh_is_held_out`.
MIN_HOLDOUT_POSITIONS = 8


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    generations: int = 100
    games_per_generation: int = 64
    batch_size: int = 256
    batches_per_generation: int = 200
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    buffer_capacity: int = 200_000
    run_dir: str = "runs/default"
    device: str | None = None
    net: NetConfig = dataclasses.field(default_factory=NetConfig)
    selfplay: SelfPlayConfig = dataclasses.field(default_factory=SelfPlayConfig)
    workers: int = 0          # 0 keeps self-play in this process
    inference_max_batch: int = 256

    @property
    def directory(self) -> Path:
        return Path(self.run_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / "checkpoint.pt"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.jsonl"

    @property
    def replay_dir(self) -> Path:
        return self.directory / "replay"


def loss_terms(
    policy_logits: Tensor,
    value_pred: Tensor,
    policy_target: Tensor,
    value_target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Cross-entropy against the visit distribution, MSE against the outcome."""
    log_probabilities = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probabilities).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
    return policy_loss, value_loss


def train(
    config: TrainConfig,
    rng: np.random.Generator,
    resume: bool = True,
) -> Path:
    device = select_device(config.device)
    # Network init and dropout draw from torch's global RNG, so seed it from
    # the caller's generator or two runs with the same seed still diverge.
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    net = PolicyValueNet(config.net).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # Shards carry no run identity, so a reused run directory can hold shards
    # from a different board. Recording the shape this run produces lets the
    # buffer skip them at load time instead of failing on a random batch hours
    # in.
    sample_shape = (N_PLANES, config.selfplay.size, config.selfplay.size)
    buffer = ReplayBuffer(config.buffer_capacity, config.replay_dir,
                          sample_shape=sample_shape)
    metrics = MetricsWriter(config.metrics_path)
    start_generation = 0

    if not resume:
        removed = buffer.clear_shards()
        if removed:
            log.info("from-scratch run: deleted %d replay shard(s) in %s",
                     removed, config.replay_dir)

    if resume and config.checkpoint_path.exists():
        payload = load_checkpoint(config.checkpoint_path, map_location=device)
        net.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        start_generation = int(payload["generation"])
        buffer.load_shards()
        log.info("resumed from generation %d", start_generation)

    evaluator = NetEvaluator(net, device=device)

    for generation in range(start_generation, config.generations):
        started = time.monotonic()
        net.eval()
        evaluator.refresh(net)
        if config.workers > 0:
            from gomoku.server_evaluator import InferenceServer, run_selfplay_workers

            server = InferenceServer(config.net, net.state_dict(),
                                     device=config.device,
                                     max_batch=config.inference_max_batch)
            server.start()
            try:
                samples, stats = run_selfplay_workers(
                    server, config.workers, config.games_per_generation,
                    config.selfplay, seed=int(rng.integers(0, 2**31)),
                )
            finally:
                server.stop()
        else:
            samples, stats = play_games(
                evaluator, config.games_per_generation, config.selfplay, rng
            )
        # Hold a slice of this generation's positions out of the optimisation
        # so the parity honesty check below is measured out of sample. It is
        # added to the buffer afterwards, so nothing is thrown away.
        holdout, trainable = _split_holdout(samples)
        buffer.add(trainable)

        net.train()
        policy_losses: list[float] = []
        value_losses: list[float] = []
        for _ in range(config.batches_per_generation):
            encoded, policy_target, value_target = buffer.sample_batch(
                config.batch_size, rng
            )
            x = torch.from_numpy(encoded).to(device)
            policy_logits, value_pred = net(x)
            policy_loss, value_loss = loss_terms(
                policy_logits,
                value_pred,
                torch.from_numpy(policy_target).to(device),
                torch.from_numpy(value_target).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_loss).backward()
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))

        # The batch losses above are averaged over the whole replay window,
        # while the baselines below describe this generation's fresh samples.
        # Comparing those two directly would drift as the window widens, so
        # measure the value head on the same fresh samples the baselines use --
        # and on the held-out ones, so neither side of the comparison has seen
        # the positions it is scored on.
        held_out = len(holdout) >= MIN_HOLDOUT_POSITIONS * N_SYMMETRIES
        evaluation = holdout if held_out else samples
        if not held_out and samples:
            log.warning(
                "generation %d produced only %d held-out positions; reporting "
                "an in-sample value_loss_on_fresh", generation,
                len(holdout) // N_SYMMETRIES,
            )
        fresh_value_loss = _value_loss_on_samples(net, evaluation, device,
                                                  config.batch_size)
        # Whatever was measured, the holdout still belongs in the window.
        buffer.add(holdout)
        record = _diagnostics(generation, samples, evaluation, held_out, stats,
                              buffer, policy_losses, value_losses,
                              fresh_value_loss, started)
        metrics.write(record)
        log.info(
            "gen %d policy %.3f value %.3f (%s %.3f vs parity baseline %.3f "
            "on %s positions) black %.2f prefixes %d/%d",
            generation, record["policy_loss"], record["value_loss"],
            "held-out" if held_out else "in-sample",
            record["value_loss_on_fresh"], record["value_baseline_parity"],
            record["value_baseline_group_sizes"], record["black_win_rate"],
            record["distinct_play_prefixes"], stats.n_games,
        )
        buffer.save_shard(generation, samples=samples)
        buffer.prune_shards(config.buffer_capacity)
        save_checkpoint(
            config.checkpoint_path, net, optimizer, generation + 1,
            config.net, extra={"buffer_size": len(buffer)},
        )

    return config.checkpoint_path


@torch.inference_mode()
def _value_loss_on_samples(net, samples, device, batch_size: int) -> float:
    """Mean squared error of the value head on `samples`.

    This is the number that belongs next to the baselines: both describe the
    same fresh positions, so `value_loss_on_fresh < value_baseline_parity` is
    an honest statement that the value head reads the board rather than the
    side to move. Pass the held-out positions where there are enough of them,
    or the statement is about data the network was just fit on.
    """
    if not samples:
        return 0.0
    net.eval()
    total = 0.0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        x = torch.from_numpy(np.stack([s.encoded for s in chunk])).to(device)
        target = torch.tensor([s.value for s in chunk], dtype=torch.float32,
                              device=device)
        _, value = net(x)
        total += float(((value - target) ** 2).sum())
    return total / len(samples)


def _split_holdout(
    samples: list, every: int = HOLDOUT_EVERY
) -> tuple[list, list]:
    """Split off every `every`-th POSITION, augmentations and all.

    Splitting by sample would put seven of a position's eight symmetries in
    the training set and call the eighth a holdout, which measures nothing:
    the network would have been fit on the same board under a rotation. So
    the block index -- `index // N_SYMMETRIES` -- decides, not the index.
    """
    holdout: list = []
    trainable: list = []
    for index, sample in enumerate(samples):
        target = holdout if (index // N_SYMMETRIES) % every == 0 else trainable
        target.append(sample)
    return holdout, trainable


def _diagnostics(generation, samples, evaluation, held_out, stats, buffer,
                 policy_losses, value_losses, fresh_value_loss, started) -> dict:
    """Assemble one generation's metrics record, including the §3 diagnostics.

    `samples` is everything the generation produced -- entropy and the added
    count describe all of it -- while `evaluation` is the subset the value
    head was scored on, which is what the baselines must describe too.
    """
    if samples:
        entropy = policy_entropy(np.stack([s.policy for s in samples]))
    else:
        entropy = 0.0
    if evaluation:
        values = np.array([s.value for s in evaluation], dtype=np.float64)
        # Plane 3 is the side-to-move constant: 1 when black is to move.
        is_black = np.array([bool(s.encoded[3].flat[0]) for s in evaluation])
        baselines = baseline_value_losses(values, is_black)
        group_sizes = [int(is_black.sum()), int((~is_black).sum())]
    else:
        baselines = {"constant": 0.0, "parity": 0.0}
        group_sizes = [0, 0]
    return {
        "generation": generation,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "value_loss_on_fresh": fresh_value_loss,
        "value_loss_on_fresh_is_held_out": bool(held_out),
        "policy_entropy": entropy,
        "black_win_rate": stats.black_win_rate,
        "value_baseline_constant": baselines["constant"],
        "value_baseline_parity": baselines["parity"],
        "value_baseline_group_sizes": group_sizes,
        "mean_game_length": stats.mean_length,
        "length_quantiles": stats.length_quantiles(),
        "distinct_openings": len(stats.openings),
        "distinct_play_prefixes": len(stats.play_prefixes),
        "buffer_size": len(buffer),
        "samples_added": len(samples),
        "seconds": time.monotonic() - started,
    }
