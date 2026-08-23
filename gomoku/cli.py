"""Command line: play, train, selfplay, arena."""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from gomoku.arena import MatchConfig, measure_levels
from gomoku.difficulty import DEFAULT_ELO_PATH, load_levels
from gomoku.engine import load_evaluator
from gomoku.evaluator import UniformEvaluator
from gomoku.mcts import SearchConfig
from gomoku.net import NetConfig
from gomoku.selfplay import SelfPlayConfig, play_games
from gomoku.train import TrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gomoku", description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    def board_arguments(sub):
        sub.add_argument("--size", type=int, default=9)
        sub.add_argument("--win-length", type=int, default=5)
        sub.add_argument("--device", default=None)
        sub.add_argument("--seed", type=int, default=None)

    play = subparsers.add_parser("play", help="play in the terminal")
    board_arguments(play)
    play.add_argument("--checkpoint", default="runs/default/checkpoint.pt")
    play.add_argument("--elo", default=str(DEFAULT_ELO_PATH))
    play.add_argument("--level", type=int, default=3, choices=range(1, 6))
    play.add_argument("--mode", default="human-vs-pc",
                      choices=["human-vs-pc", "pc-vs-pc"])
    play.add_argument("--no-launch", action="store_true",
                      help="build everything but do not start the UI")

    trainer = subparsers.add_parser("train", help="run self-play training")
    board_arguments(trainer)
    trainer.add_argument("--run-dir", default="runs/default")
    trainer.add_argument("--generations", type=int, default=100)
    trainer.add_argument("--games", type=int, default=64)
    trainer.add_argument("--batches", type=int, default=200)
    trainer.add_argument("--batch-size", type=int, default=256)
    trainer.add_argument("--channels", type=int, default=64)
    trainer.add_argument("--blocks", type=int, default=6)
    trainer.add_argument("--simulations", type=int, default=600)
    trainer.add_argument("--fast-simulations", type=int, default=100)
    trainer.add_argument("--full-fraction", type=float, default=0.25)
    trainer.add_argument("--games-in-flight", type=int, default=32)
    trainer.add_argument("--no-resume", action="store_true")

    selfplay = subparsers.add_parser("selfplay", help="generate games only")
    board_arguments(selfplay)
    selfplay.add_argument("--checkpoint", default=None)
    selfplay.add_argument("--games", type=int, default=8)
    selfplay.add_argument("--simulations", type=int, default=600)
    selfplay.add_argument("--fast-simulations", type=int, default=100)

    arena = subparsers.add_parser("arena", help="measure level ratings")
    board_arguments(arena)
    arena.add_argument("--checkpoint", default="runs/default/checkpoint.pt")
    arena.add_argument("--out", default=str(DEFAULT_ELO_PATH))
    arena.add_argument("--games-per-pair", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_usage()
        return 2
    rng = np.random.default_rng(args.seed)
    return {
        "play": _play,
        "train": _train,
        "selfplay": _selfplay,
        "arena": _arena,
    }[args.command](args, rng)


def _play(args, rng) -> int:
    from gomoku.tui.app import run_tui

    evaluator, generation = load_evaluator(args.checkpoint, args.device)
    if evaluator is None:
        print("No checkpoint found; playing against the heuristic bot.")
    else:
        print(f"Loaded checkpoint trained to generation {generation}.")
    levels = load_levels(args.elo)
    if args.no_launch or not sys.stdout.isatty():
        if args.no_launch:
            return 0
        print("gomoku play needs an interactive terminal.")
        return 0
    run_tui(size=args.size, win_length=args.win_length, levels=levels,
            level_index=args.level, evaluator=evaluator, rng=rng, mode=args.mode)
    return 0


def _selfplay_config(args) -> SelfPlayConfig:
    return SelfPlayConfig(
        size=args.size,
        win_length=args.win_length,
        full_simulations=args.simulations,
        fast_simulations=args.fast_simulations,
        full_fraction=getattr(args, "full_fraction", 0.25),
        games_in_flight=getattr(args, "games_in_flight", 32),
        search=SearchConfig(dirichlet_alpha=10.0 / (args.size * args.size)),
    )


def _train(args, rng) -> int:
    config = TrainConfig(
        generations=args.generations,
        games_per_generation=args.games,
        batch_size=args.batch_size,
        batches_per_generation=args.batches,
        run_dir=args.run_dir,
        device=args.device,
        net=NetConfig(channels=args.channels, blocks=args.blocks),
        selfplay=_selfplay_config(args),
    )
    path = train(config, rng, resume=not args.no_resume)
    print(f"checkpoint: {path}")
    return 0


def _selfplay(args, rng) -> int:
    evaluator, _ = load_evaluator(args.checkpoint, args.device)
    if evaluator is None:
        evaluator = UniformEvaluator()
    samples, stats = play_games(evaluator, args.games, _selfplay_config(args), rng)
    print(f"{stats.n_games} games, {len(samples)} samples")
    print(f"black win rate {stats.black_win_rate:.2f}, "
          f"mean length {stats.mean_length:.1f}, "
          f"distinct openings {len(stats.openings)}")
    return 0


def _arena(args, rng) -> int:
    evaluator, _ = load_evaluator(args.checkpoint, args.device)
    if evaluator is None:
        print("No checkpoint found; nothing to rate.")
        return 1
    config = MatchConfig(size=args.size, win_length=args.win_length,
                         games_per_pair=args.games_per_pair)
    ratings = measure_levels(evaluator, config, rng, args.out)
    for name, rating in sorted(ratings.items(), key=lambda item: item[1]):
        print(f"{name:>12}  {rating:7.0f}")
    print(f"written to {args.out}")
    return 0
