"""Training diagnostics and a JSON-lines metrics log.

`baseline_value_losses` is the instrument that detects the parity shortcut
described in section 3 of the design. The constant baseline is what a value
head achieves by predicting the mean outcome; the parity baseline is what it
achieves by predicting the mean outcome *for each colour*. A value head that
does not beat the parity baseline has learned nothing but "black is winning".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def policy_entropy(policies: np.ndarray) -> float:
    """Mean Shannon entropy in nats over a batch of policy distributions."""
    probabilities = np.clip(policies, 1e-12, 1.0)
    return float(-(probabilities * np.log(probabilities)).sum(axis=-1).mean())


def baseline_value_losses(
    values: np.ndarray,
    is_black_to_move: np.ndarray,
) -> dict[str, float]:
    """Mean-squared error of the constant and parity predictors."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"constant": 0.0, "parity": 0.0}
    mask = np.asarray(is_black_to_move, dtype=bool)
    constant = float(((values - values.mean()) ** 2).mean())
    predicted = np.empty_like(values)
    for group in (mask, ~mask):
        if group.any():
            predicted[group] = values[group].mean()
    parity = float(((values - predicted) ** 2).mean())
    return {"constant": constant, "parity": parity}


class MetricsWriter:
    """Append-only JSON-lines log. One record per training generation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                log.warning("skipping malformed metrics line in %s", self.path)
        return records
