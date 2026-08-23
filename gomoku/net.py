"""The policy/value network and checkpoint I/O.

Fully convolutional, with the value head reduced by global average pooling
rather than a flatten, so a single set of weights accepts any board size.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import torch
from torch import Tensor, nn

from gomoku.game import N_PLANES

log = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def select_device(prefer: str | None = None) -> torch.device:
    """Pick a compute device. MPS when available, otherwise CPU.

    Nothing in this project requires MPS; the fallback is always usable.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    log.info("MPS unavailable; falling back to CPU")
    return torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class NetConfig:
    channels: int = 64
    blocks: int = 6
    in_planes: int = N_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        channels = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(self.config.blocks)]
        )
        # One logit per cell: a 1x1 convolution, so the output length follows
        # the input resolution automatically.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.trunk(self.stem(x))
        policy_logits = self.policy_head(features).flatten(1)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


def save_checkpoint(
    path: str | os.PathLike,
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    generation: int,
    config: NetConfig,
    extra: dict | None = None,
) -> None:
    """Write a checkpoint atomically, so an interrupted run never corrupts one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Schema version, so a future format change can migrate rather than
        # guess. `load_checkpoint` tolerates its absence: every checkpoint
        # written before this field existed is version 1 by definition.
        "version": CHECKPOINT_VERSION,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "generation": generation,
        "config": dataclasses.asdict(config),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
