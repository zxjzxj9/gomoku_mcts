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
_FAILED = "failed"

# A worker blocked forever on a queue is the failure mode this module has to
# avoid: it does not fail a test, it hangs a multi-day run at 3am. Every wait
# on the request path is therefore bounded.
DEFAULT_REQUEST_TIMEOUT = 300.0


class ServerEvaluator(Evaluator):
    """A client handle. Sends states to the server and waits for its results."""

    def __init__(self, request_queue, response_queue, worker_id: int,
                 timeout: float = DEFAULT_REQUEST_TIMEOUT) -> None:
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self.timeout = timeout

    def evaluate(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.request_queue.put((self.worker_id, np.ascontiguousarray(encoded)))
        try:
            first, second = self.response_queue.get(timeout=self.timeout)
        except queue.Empty as error:
            raise RuntimeError(
                f"inference server did not respond within {self.timeout}s; "
                "it has probably died"
            ) from error
        if isinstance(first, str) and first == _FAILED:
            raise RuntimeError(f"inference server failed: {second}")
        return first, second


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

        # Any failure here -- an OOM, a dtype mismatch, two clients sending
        # different board sizes -- must be reported back. Dying silently would
        # leave every worker blocked on a queue that will never be served.
        try:
            batch = np.concatenate([states for _, states in pending], axis=0)
            with torch.inference_mode():
                logits, values = net(torch.from_numpy(batch).to(torch_device))
                policies = torch.softmax(logits.float(), dim=1).cpu().numpy()
                values = values.float().cpu().numpy()
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            log.exception("inference server failed on a batch of %d", len(pending))
            for worker_id, _ in pending:
                response_queues[worker_id].put((_FAILED, str(error)))
            continue
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
        self._next_client = 0
        self.control_queue.put((_STOP, None))
        self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None


def split_games(total: int, workers: int) -> list[int]:
    """Divide `total` games across `workers`, distributing the remainder.

    Integer division alone silently drops games: 128 games over 6 workers
    would run 126.
    """
    base, remainder = divmod(total, workers)
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _worker(server_args, worker_id, games, config, seed, output_queue) -> None:
    request_queue, response_queue = server_args
    evaluator = ServerEvaluator(request_queue, response_queue, worker_id)
    samples, stats = play_games(evaluator, games, config,
                                np.random.default_rng(seed))
    output_queue.put((samples, stats))


def run_selfplay_workers(
    server: InferenceServer,
    n_workers: int,
    total_games: int,
    config: SelfPlayConfig,
    seed: int,
) -> tuple[list, GameStats]:
    """Generate `total_games` games across `n_workers` processes, one model.

    The game count is the total, not a per-worker figure, so that an uneven
    split cannot quietly change how many games a generation produces.
    """
    output_queue = server.context.Queue()
    processes = []
    for index, games in enumerate(split_games(total_games, n_workers)):
        if games <= 0:
            continue
        evaluator = server.client()
        process = server.context.Process(
            target=_worker,
            args=((evaluator.request_queue, evaluator.response_queue),
                  evaluator.worker_id, games, config,
                  seed + index, output_queue),
            daemon=True,
        )
        process.start()
        processes.append(process)

    samples: list = []
    combined = GameStats()
    for received in range(len(processes)):
        worker_samples, worker_stats = _collect_result(output_queue, processes,
                                                       received)
        samples.extend(worker_samples)
        combined.black_wins += worker_stats.black_wins
        combined.white_wins += worker_stats.white_wins
        combined.draws += worker_stats.draws
        combined.lengths.extend(worker_stats.lengths)
        combined.openings |= worker_stats.openings
        combined.play_prefixes |= worker_stats.play_prefixes
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            log.warning("terminating a self-play worker that would not exit")
            process.terminate()
            process.join(timeout=5)
    return samples, combined


def _collect_result(output_queue, processes, collected: int):
    """Wait for one worker's result, giving up if a worker died first.

    A worker that has already reported is no longer alive, so a dead worker is
    one that is neither running nor accounted for by a collected result.
    """
    while True:
        try:
            return output_queue.get(timeout=5.0)
        except queue.Empty:
            alive = sum(process.is_alive() for process in processes)
            if alive + collected < len(processes):
                raise RuntimeError(
                    "a self-play worker exited without reporting results"
                ) from None
