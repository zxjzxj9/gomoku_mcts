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
    # 25 simulations left the visit counts nearly flat -- 11 moves visited with
    # a top share of 0.21 -- and sampling that at temperature 0.6 discarded what
    # little signal the search had, so this level played its raw policy and
    # measured within 14 ELO of level 1. Doubling the budget makes the counts
    # informative and the cooler temperature keeps them.
    Level(2, "Casual", simulations=50, temperature=0.5, policy_only=False),
    Level(3, "Club", simulations=100, temperature=0.3, policy_only=False),
    Level(4, "Strong", simulations=400, temperature=0.0, policy_only=False),
    Level(5, "Expert", simulations=1600, temperature=0.0, policy_only=False),
)


def load_levels(
    elo_path: str | Path | None = DEFAULT_ELO_PATH,
    size: int | None = None,
    win_length: int | None = None,
) -> tuple[Level, ...]:
    """Return the levels, annotated with measured ratings when available.

    Pass `size` and `win_length` to demand that the ratings were measured on
    the board about to be played. A rating is a property of a board as much as
    of a checkpoint, and showing a 9x9 number above a 15x15 game is the one
    route by which a *displayed* rating can be false. When the metadata
    disagrees -- or is missing, so nothing can be checked -- the levels come
    back unrated rather than wrong.
    """
    if elo_path is None:
        return LEVELS
    path = Path(elo_path)
    if not path.exists():
        return LEVELS
    try:
        payload = json.loads(path.read_text())
        ratings = payload["ratings"]
    except (ValueError, KeyError, OSError, TypeError) as error:
        log.warning("ignoring unreadable ELO file %s: %s", path, error)
        return LEVELS
    if not isinstance(ratings, dict):
        log.warning("ignoring ELO file %s: 'ratings' is not an object", path)
        return LEVELS
    if not _board_matches(path, payload, size, win_length):
        return LEVELS
    return tuple(
        dataclasses.replace(level, elo=_as_int(ratings.get(level.key)))
        for level in LEVELS
    )


def _board_matches(path, payload, size: int | None, win_length: int | None) -> bool:
    """Does this ELO file describe the board the caller is about to play?"""
    if size is None and win_length is None:
        return True
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        log.warning("ignoring ratings in %s: it records no board metadata, so "
                    "they cannot be confirmed to describe a %sx%s board",
                    path, size, size)
        return False
    for name, wanted in (("size", size), ("win_length", win_length)):
        if wanted is not None and metadata.get(name) != wanted:
            log.warning("ignoring ratings in %s: measured with %s=%r, playing "
                        "%s=%r", path, name, metadata.get(name), name, wanted)
            return False
    return True


def _as_int(value) -> int | None:
    """Coerce a stored rating, or report it as unmeasured.

    Anything unparseable becomes None -- "unrated" -- rather than raising.
    A half-written or hand-edited ELO file must never stop someone playing,
    and it must never be rendered as a number nobody measured. Booleans are
    rejected explicitly because `int(True)` would otherwise show as 1 ELO.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    # OverflowError belongs here even though it looks out of place:
    # json.loads accepts bare Infinity/-Infinity tokens, and
    # round(float("inf")) raises OverflowError rather than ValueError.
    # (NaN is already covered: round(float("nan")) raises ValueError.)
    except (TypeError, ValueError, OverflowError):
        log.warning("ignoring malformed ELO value %r", value)
        return None


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
