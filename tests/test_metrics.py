import numpy as np
import pytest

from gomoku.metrics import MetricsWriter, baseline_value_losses, policy_entropy


def test_entropy_of_a_uniform_policy_is_maximal():
    uniform = np.full((2, 16), 1.0 / 16)
    assert policy_entropy(uniform) == pytest.approx(np.log(16))


def test_entropy_of_a_deterministic_policy_is_zero():
    sharp = np.zeros((2, 16))
    sharp[:, 3] = 1.0
    assert policy_entropy(sharp) == pytest.approx(0.0, abs=1e-9)


def test_constant_baseline_equals_the_variance_of_the_targets():
    values = np.array([1.0, 1.0, -1.0, 0.0])
    is_black = np.array([True, False, True, False])
    losses = baseline_value_losses(values, is_black)
    assert losses["constant"] == pytest.approx(values.var())


def test_parity_baseline_beats_the_constant_baseline_when_colour_predicts_the_result():
    values = np.array([1.0, -1.0, 1.0, -1.0])
    is_black = np.array([True, False, True, False])
    losses = baseline_value_losses(values, is_black)
    assert losses["parity"] == pytest.approx(0.0)
    assert losses["constant"] > losses["parity"]


def test_parity_baseline_matches_the_constant_one_when_colour_is_uninformative():
    values = np.array([1.0, -1.0, 1.0, -1.0])
    is_black = np.array([True, True, False, False])
    losses = baseline_value_losses(values, is_black)
    assert losses["parity"] == pytest.approx(losses["constant"])


def test_baselines_handle_a_missing_colour():
    values = np.array([1.0, 0.5])
    losses = baseline_value_losses(values, np.array([True, True]))
    assert losses["parity"] == pytest.approx(values.var())


def test_metrics_writer_appends_json_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    writer.write({"generation": 0, "loss": 1.5})
    writer.write({"generation": 1, "loss": 1.2})
    records = MetricsWriter(path).read_all()
    assert [r["generation"] for r in records] == [0, 1]
    assert records[1]["loss"] == 1.2


def test_metrics_writer_reads_an_absent_file_as_empty(tmp_path):
    assert MetricsWriter(tmp_path / "missing.jsonl").read_all() == []


def test_metrics_writer_skips_a_truncated_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    MetricsWriter(path).write({"generation": 0})
    with path.open("a") as handle:
        handle.write('{"generation": 1, "loss"\n')
    assert len(MetricsWriter(path).read_all()) == 1
