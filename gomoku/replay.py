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

    def save_shard(
        self,
        generation: int,
        samples: list[Sample] | None = None,
    ) -> Path | None:
        """Write `samples` to `shard_%06d.npz`. No directory, no shard.

        Pass the generation's new samples rather than letting this default to
        the whole window: a shard per generation each holding the entire
        buffer makes disk use quadratic in the number of generations, and
        fills a resumed buffer with duplicates of the same old positions.
        """
        if self.directory is None:
            return None
        samples = list(self._samples) if samples is None else list(samples)
        if not samples:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"shard_{generation:06d}.npz"
        # `savez_compressed` appends `.npz` to any name not already ending in
        # it, so the temp file must be named without that suffix and the
        # written file reconstructed before the rename. The `.tmp` name also
        # keeps the partial file out of `load_shards`'s `shard_*.npz` glob.
        tmp = self.directory / f"shard_{generation:06d}.tmp"
        np.savez_compressed(
            str(tmp),
            encoded=np.stack([s.encoded for s in samples]),
            policy=np.stack([s.policy for s in samples]),
            value=np.array([s.value for s in samples], dtype=np.float32),
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

    def prune_shards(self, keep_samples: int) -> int:
        """Delete the oldest shards the buffer can no longer hold.

        Shards accumulate one per generation, so without pruning a long run
        keeps far more history on disk than `capacity` can ever load. Walks
        newest-first and keeps just enough shards to refill the buffer.
        Returns the number of shards deleted.
        """
        if self.directory is None or not self.directory.exists():
            return 0
        paths = sorted(self.directory.glob("shard_*.npz"))
        counts = []
        for path in paths:
            try:
                with np.load(path) as data:
                    counts.append(len(data["value"]))
            except (OSError, ValueError, KeyError):
                counts.append(0)
        keep_from = 0
        running = 0
        for index in range(len(paths) - 1, -1, -1):
            running += counts[index]
            if running >= keep_samples:
                keep_from = index
                break
        for path in paths[:keep_from]:
            path.unlink()
        return keep_from
