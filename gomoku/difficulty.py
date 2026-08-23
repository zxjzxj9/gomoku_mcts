"""The five difficulty levels.

Every level runs the same checkpoint. Strength comes from the simulation
budget, and the residual randomness that keeps the weaker levels beatable
comes from the sampling temperature. ELO is measured by the arena and read
from disk; a level whose rating has not been measured reports None rather
than a guess.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from gomoku.evaluator import Evaluator
from gomoku.players import MCTSPlayer, Player

log = logging.getLogger(__name__)

DEFAULT_ELO_PATH = Path("runs/elo.json")


@dataclasses.dataclass(frozen=True)
class Level:
    index: int
    name: str
    simulations: int
    temperature: float
    policy_only: bool
    elo: int | None = None

    @property
    def key(self) -> str:
        return f"level{self.index}"

    def label(self) -> str:
        rating = f"{self.elo} ELO" if self.elo is not None else "unrated"
        return f"{self.index}. {self.name} ({rating})"


LEVELS: tuple[Level, ...] = (
    Level(1, "Beginner", simulations=0, temperature=1.0, policy_only=True),
    Level(2, "Casual", simulations=25, temperature=0.6, policy_only=False),
    Level(3, "Club", simulations=100, temperature=0.3, policy_only=False),
    Level(4, "Strong", simulations=400, temperature=0.0, policy_only=False),
    Level(5, "Expert", simulations=1600, temperature=0.0, policy_only=False),
)


def load_levels(elo_path: str | Path | None = DEFAULT_ELO_PATH) -> tuple[Level, ...]:
    """Return the levels, annotated with measured ratings when available."""
    if elo_path is None:
        return LEVELS
    path = Path(elo_path)
    if not path.exists():
        return LEVELS
    try:
        ratings = json.loads(path.read_text())["ratings"]
    except (ValueError, KeyError, OSError) as error:
        log.warning("ignoring unreadable ELO file %s: %s", path, error)
        return LEVELS
    return tuple(
        dataclasses.replace(level, elo=_as_int(ratings.get(level.key)))
        for level in LEVELS
    )


def _as_int(value) -> int | None:
    return None if value is None else int(round(float(value)))


def make_player(
    level: Level,
    evaluator: Evaluator,
    rng: np.random.Generator,
) -> Player:
    return MCTSPlayer(
        evaluator,
        simulations=level.simulations,
        temperature=level.temperature,
        policy_only=level.policy_only,
        rng=rng,
        name=f"level{level.index}",
    )
