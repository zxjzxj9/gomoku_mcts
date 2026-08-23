"""Phase two throughput: MCTS in worker processes, one batched model.

Tree search is pure Python and single-core; the model wants large batches on
one device. Splitting them lets both run flat out -- workers descend trees in
parallel while the server coalesces their leaves into batches large enough to
keep the GPU busy.

The interface is unchanged: a worker holds a `ServerEvaluator`, which is an
`Evaluator` like any other, so MCTS, self-play and training are untouched.
"""

from __future__ import annotations

import logging
import queue
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gomoku.evaluator import Evaluator
from gomoku.net import NetConfig, PolicyValueNet, select_device
from gomoku.selfplay import GameStats, SelfPlayConfig, play_games

log = logging.getLogger(__name__)

_STOP = "stop"
_REFRESH = "refresh"


class ServerEvaluator(Evaluator):
    """A client handle. Sends states to the server and waits for its results."""

    def __init__(self, request_queue, response_queue, worker_id: int) -> None:
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.request_queue.put((self.worker_id, np.ascontiguousarray(encoded)))
        return self.response_queue.get()


def _serve(net_config, state_dict, device, max_batch, wait_seconds,
           request_queue, response_queues, control_queue) -> None:
    net = PolicyValueNet(net_config)
    net.load_state_dict(state_dict)
    torch_device = select_device(device)
    net = net.to(torch_device).eval()
    while True:
        try:
            control = control_queue.get_nowait()
        except queue.Empty:
            control = None
        if control is not None:
            action, payload = control
            if action == _STOP:
                return
            if action == _REFRESH:
                net.load_state_dict(payload)
                net = net.to(torch_device).eval()

        pending: list[tuple[int, np.ndarray]] = []
        try:
            pending.append(request_queue.get(timeout=0.05))
        except queue.Empty:
            continue
        # Coalesce whatever else has arrived, up to the batch limit.
        deadline = time.monotonic() + wait_seconds
        total = pending[0][1].shape[0]
        while total < max_batch and time.monotonic() < deadline:
            try:
                item = request_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.0005)
                continue
            pending.append(item)
            total += item[1].shape[0]

        batch = np.concatenate([states for _, states in pending], axis=0)
        with torch.inference_mode():
            logits, values = net(torch.from_numpy(batch).to(torch_device))
            policies = torch.softmax(logits.float(), dim=1).cpu().numpy()
            values = values.float().cpu().numpy()
        offset = 0
        for worker_id, states in pending:
            count = states.shape[0]
            response_queues[worker_id].put(
                (policies[offset : offset + count].astype(np.float32),
                 values[offset : offset + count].astype(np.float32))
            )
            offset += count


class InferenceServer:
    def __init__(self, net_config: NetConfig, state_dict, device=None,
                 max_batch: int = 256, wait_ms: float = 2.0,
                 max_clients: int = 16) -> None:
        self.context = mp.get_context("spawn")
        self.net_config = net_config
        self.state_dict = {k: v.cpu() for k, v in state_dict.items()}
        self.device = device
        self.max_batch = max_batch
        self.wait_seconds = wait_ms / 1000.0
        self.request_queue = self.context.Queue()
        self.control_queue = self.context.Queue()
        self.response_queues = [self.context.Queue() for _ in range(max_clients)]
        self._next_client = 0
        self._process = None

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = self.context.Process(
            target=_serve,
            args=(self.net_config, self.state_dict, self.device, self.max_batch,
                  self.wait_seconds, self.request_queue, self.response_queues,
                  self.control_queue),
            daemon=True,
        )
        self._process.start()

    def client(self) -> ServerEvaluator:
        if self._next_client >= len(self.response_queues):
            raise RuntimeError("no response queues left; raise max_clients")
        evaluator = ServerEvaluator(
            self.request_queue, self.response_queues[self._next_client],
            self._next_client,
        )
        self._next_client += 1
        return evaluator

    def refresh(self, state_dict) -> None:
        self.control_queue.put((_REFRESH, {k: v.cpu() for k, v in state_dict.items()}))

    def stop(self) -> None:
        if self._process is None:
            return
        self.control_queue.put((_STOP, None))
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None


def _worker(server_args, worker_id, games, config, seed, output_queue) -> None:
    request_queue, response_queue = server_args
    evaluator = ServerEvaluator(request_queue, response_queue, worker_id)
    samples, stats = play_games(evaluator, games, config,
                                np.random.default_rng(seed))
    output_queue.put((samples, stats))


def run_selfplay_workers(
    server: InferenceServer,
    n_workers: int,
    games_per_worker: int,
    config: SelfPlayConfig,
    seed: int,
) -> tuple[list, GameStats]:
    """Generate games in `n_workers` processes, all served by one model."""
    output_queue = server.context.Queue()
    processes = []
    for index in range(n_workers):
        evaluator = server.client()
        process = server.context.Process(
            target=_worker,
            args=((evaluator.request_queue, evaluator.response_queue),
                  evaluator.worker_id, games_per_worker, config,
                  seed + index, output_queue),
            daemon=True,
        )
        process.start()
        processes.append(process)

    samples: list = []
    combined = GameStats()
    for _ in processes:
        worker_samples, worker_stats = output_queue.get()
        samples.extend(worker_samples)
        combined.black_wins += worker_stats.black_wins
        combined.white_wins += worker_stats.white_wins
        combined.draws += worker_stats.draws
        combined.lengths.extend(worker_stats.lengths)
        combined.openings |= worker_stats.openings
    for process in processes:
        process.join(timeout=10)
    return samples, combined
