"""The single boundary between tree search and compute hardware.

Search never imports torch. It hands an evaluator a batch of encoded states
and receives priors and values back. That is the whole contract, and it is
what lets the multiprocess evaluator in phase two be a drop-in replacement.
"""

from __future__ import annotations

import abc

import numpy as np
import torch

from gomoku.net import PolicyValueNet, select_device


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map `(B, C, H, W)` float32 states to `(B, H*W)` priors and `(B,)` values.

        Priors sum to one per row; values lie in [-1, 1] and are from the
        perspective of the side to move in each state.
        """


class UniformEvaluator(Evaluator):
    """Uniform priors and zero values. Used to test search in isolation."""

    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_states = encoded.shape[0]
        n_cells = encoded.shape[-1] * encoded.shape[-2]
        policies = np.full((n_states, n_cells), 1.0 / n_cells, dtype=np.float32)
        return policies, np.zeros(n_states, dtype=np.float32)


class NetEvaluator(Evaluator):
    """Runs the network in inference mode, chunked to a maximum batch size."""

    def __init__(
        self,
        net: PolicyValueNet,
        device: str | torch.device | None = None,
        max_batch: int = 256,
    ) -> None:
        self.device = select_device(device) if not isinstance(device, torch.device) \
            else device
        self.net = net.to(self.device).eval()
        self.max_batch = max_batch
        self.n_evaluated = 0

    def refresh(self, net: PolicyValueNet) -> None:
        """Swap in newer weights between generations."""
        self.net = net.to(self.device).eval()

    @torch.inference_mode()
    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policies: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for start in range(0, encoded.shape[0], self.max_batch):
            chunk = encoded[start : start + self.max_batch]
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
            logits, value = self.net(x)
            prior = torch.softmax(logits.float(), dim=1)
            policies.append(prior.cpu().numpy().astype(np.float32))
            values.append(value.float().cpu().numpy().astype(np.float32))
        self.n_evaluated += int(encoded.shape[0])
        return np.concatenate(policies, axis=0), np.concatenate(values, axis=0)
