"""The replay buffer: a bounded window over recent self-play samples.

Shards on disk make a multi-day run resumable. A shard that fails to load is
skipped rather than fatal -- losing part of the history costs some training
signal, but aborting a two-day run costs the whole thing.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from gomoku.selfplay import Sample

log = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity: int, directory: str | Path | None = None) -> None:
        self.capacity = capacity
        self.directory = Path(directory) if directory is not None else None
        self._samples: collections.deque[Sample] = collections.deque(maxlen=capacity)
        self.n_added = 0

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self._samples.append(sample)
            self.n_added += 1

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._samples:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = rng.integers(0, len(self._samples), size=batch_size)
        chosen = [self._samples[int(i)] for i in indices]
        encoded = np.stack([s.encoded for s in chosen]).astype(np.float32)
        policy = np.stack([s.policy for s in chosen]).astype(np.float32)
        value = np.array([s.value for s in chosen], dtype=np.float32)
        return encoded, policy, value

    def value_statistics(self) -> dict[str, float]:
        if not self._samples:
            return {"mean": 0.0, "variance": 0.0}
        values = np.array([s.value for s in self._samples], dtype=np.float64)
        return {"mean": float(values.mean()), "variance": float(values.var())}

    # -- persistence ----------------------------------------------------

    def save_shard(self, generation: int) -> Path | None:
        """Write the current window to `shard_%06d.npz`. No directory, no shard."""
        if self.directory is None or not self._samples:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"shard_{generation:06d}.npz"
        tmp = self.directory / f"shard_{generation:06d}.tmp"
        np.savez_compressed(
            str(tmp),
            encoded=np.stack([s.encoded for s in self._samples]),
            policy=np.stack([s.policy for s in self._samples]),
            value=np.array([s.value for s in self._samples], dtype=np.float32),
        )
        (tmp.parent / f"{tmp.name}.npz").replace(path)
        return path

    def load_shards(self) -> int:
        """Load every shard in generation order. Returns the number loaded."""
        if self.directory is None or not self.directory.exists():
            return 0
        loaded = 0
        for path in sorted(self.directory.glob("shard_*.npz")):
            try:
                with np.load(path) as data:
                    encoded, policy, value = data["encoded"], data["policy"], data["value"]
            except (OSError, ValueError, KeyError) as error:
                log.warning("skipping unreadable replay shard %s: %s", path, error)
                continue
            self.add(
                Sample(encoded[i], policy[i], float(value[i]))
                for i in range(len(value))
            )
            loaded += len(value)
        return min(loaded, self.capacity) if loaded else 0
