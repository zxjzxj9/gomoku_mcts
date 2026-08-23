"""Assembling a playable opponent from a checkpoint, or from nothing.

The TUI must be usable before any network exists, so a missing checkpoint
degrades to the heuristic bot rather than failing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gomoku.difficulty import Level, make_player
from gomoku.evaluator import Evaluator, NetEvaluator
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint
from gomoku.players import HeuristicPlayer, Player

log = logging.getLogger(__name__)


def load_evaluator(
    checkpoint: str | Path | None,
    device: str | None = None,
) -> tuple[Evaluator | None, int | None]:
    if checkpoint is None:
        return None, None
    path = Path(checkpoint)
    if not path.exists():
        log.info("no checkpoint at %s; falling back to the heuristic bot", path)
        return None, None
    payload = load_checkpoint(path, map_location="cpu")
    net = PolicyValueNet(NetConfig(**payload["config"]))
    net.load_state_dict(payload["model"])
    return NetEvaluator(net, device=device), int(payload["generation"])


def build_opponent(
    level: Level,
    evaluator: Evaluator | None,
    rng: np.random.Generator,
) -> Player:
    if evaluator is None:
        return HeuristicPlayer(rng, name=f"heuristic-level{level.index}")
    return make_player(level, evaluator, rng)
