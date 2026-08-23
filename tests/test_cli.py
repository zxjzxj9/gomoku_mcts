import json

import pytest

from gomoku.cli import main
from gomoku.net import NetConfig, PolicyValueNet, load_checkpoint, save_checkpoint


def test_no_arguments_prints_usage_and_fails():
    assert main([]) == 2


def test_unknown_subcommand_fails():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_train_subcommand_writes_a_checkpoint(tmp_path):
    code = main([
        "train", "--run-dir", str(tmp_path), "--generations", "1",
        "--games", "2", "--batches", "2", "--batch-size", "8",
        "--size", "5", "--win-length", "4", "--channels", "8", "--blocks", "1",
        "--simulations", "8", "--fast-simulations", "2", "--device", "cpu",
        "--seed", "0",
    ])
    assert code == 0
    assert load_checkpoint(tmp_path / "checkpoint.pt")["generation"] == 1


def test_selfplay_subcommand_reports_game_statistics(tmp_path, capsys):
    code = main([
        "selfplay", "--games", "2", "--size", "5", "--win-length", "4",
        "--simulations", "8", "--fast-simulations", "2", "--seed", "0",
        "--device", "cpu",
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "games" in output and "black win rate" in output


def test_arena_subcommand_writes_elo(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(checkpoint, PolicyValueNet(config), None, 1, config, {})
    code = main([
        "arena", "--checkpoint", str(checkpoint), "--out", str(tmp_path / "elo.json"),
        "--games-per-pair", "2", "--size", "5", "--win-length", "4",
        "--device", "cpu", "--seed", "0",
    ])
    assert code == 0
    payload = json.loads((tmp_path / "elo.json").read_text())
    assert payload["ratings"]["heuristic"] == pytest.approx(1200.0, abs=1e-6)
    assert set(payload["ratings"]) >= {f"level{i}" for i in range(1, 6)}


def test_play_subcommand_explains_itself_without_a_terminal(monkeypatch, capsys):
    """`play` needs a TTY; without one it must explain, not crash."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert main(["play"]) == 0
    assert "interactive terminal" in capsys.readouterr().out


def test_play_subcommand_builds_everything_without_launching(tmp_path, capsys):
    """--no-launch exercises checkpoint and level loading with no UI.

    Asserting only the return code would pass even if `_play` never loaded
    anything, so check it reported what it found."""
    assert main(["play", "--no-launch", "--checkpoint", str(tmp_path / "none.pt"),
                 "--elo", str(tmp_path / "none.json")]) == 0
    assert "No checkpoint found" in capsys.readouterr().out


def test_play_subcommand_reports_a_loaded_checkpoint(tmp_path, capsys):
    checkpoint = tmp_path / "checkpoint.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(checkpoint, PolicyValueNet(config), None, 5, config, {})
    assert main(["play", "--no-launch", "--checkpoint", str(checkpoint),
                 "--elo", str(tmp_path / "none.json"), "--device", "cpu"]) == 0
    assert "generation 5" in capsys.readouterr().out
