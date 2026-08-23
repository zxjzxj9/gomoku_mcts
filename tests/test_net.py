import torch

from gomoku.game import N_PLANES
from gomoku.net import (
    CHECKPOINT_VERSION,
    NetConfig,
    PolicyValueNet,
    load_checkpoint,
    save_checkpoint,
    select_device,
)


def small():
    return PolicyValueNet(NetConfig(channels=8, blocks=1))


def test_forward_shapes():
    net = small()
    logits, value = net(torch.zeros(3, N_PLANES, 9, 9))
    assert logits.shape == (3, 81)
    assert value.shape == (3,)


def test_value_is_bounded():
    net = small()
    _, value = net(torch.randn(16, N_PLANES, 9, 9) * 5)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_same_weights_accept_a_different_board_size():
    net = small()
    logits9, value9 = net(torch.zeros(1, N_PLANES, 9, 9))
    logits15, value15 = net(torch.zeros(1, N_PLANES, 15, 15))
    assert logits9.shape == (1, 81)
    assert logits15.shape == (1, 225)
    assert value15.shape == (1,)


def test_gradients_flow_to_both_heads():
    net = small()
    logits, value = net(torch.zeros(2, N_PLANES, 9, 9))
    (logits.sum() + value.sum()).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_select_device_falls_back_to_cpu_when_asked():
    assert select_device("cpu").type == "cpu"


def test_select_device_returns_something_usable():
    device = select_device()
    assert device.type in {"cpu", "mps", "cuda"}
    torch.zeros(1, device=device)


def test_checkpoint_round_trip(tmp_path):
    net = small()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    path = tmp_path / "ckpt.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, net, optimizer, generation=4, config=config,
                    extra={"samples": 100})
    payload = load_checkpoint(path, map_location="cpu")
    assert payload["generation"] == 4
    assert payload["config"]["channels"] == 8
    assert payload["extra"]["samples"] == 100

    restored = PolicyValueNet(NetConfig(**payload["config"]))
    restored.load_state_dict(payload["model"])
    net.eval()
    restored.eval()
    x = torch.randn(2, N_PLANES, 9, 9)
    with torch.no_grad():
        assert torch.allclose(net(x)[0], restored(x)[0], atol=1e-6)


def test_checkpoint_write_is_atomic(tmp_path):
    """No partial file is left behind under the final name."""
    net, path = small(), tmp_path / "ckpt.pt"
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    save_checkpoint(path, net, optimizer, 0, NetConfig(channels=8, blocks=1), {})
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_gradients_reach_the_policy_head_and_the_value_head_separately():
    """A dead value head is the failure this project exists to detect.

    Asserting that *some* parameter has a gradient passes with a value head
    that is entirely disconnected, so check each head by name.
    """
    net = small()
    logits, value = net(torch.randn(2, N_PLANES, 9, 9))
    (logits.sum() + value.sum()).backward()

    def total_gradient(prefix):
        grads = [p.grad for name, p in net.named_parameters()
                 if name.startswith(prefix)]
        assert grads, f"no parameters under {prefix}"
        assert all(g is not None for g in grads), f"a {prefix} parameter got no grad"
        return sum(float(g.abs().sum()) for g in grads)

    assert total_gradient("policy_head") > 0.0
    assert total_gradient("value_head") > 0.0


def test_checkpoints_record_a_schema_version(tmp_path):
    path = tmp_path / "ckpt.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, small(), None, generation=1, config=config)
    assert load_checkpoint(path, map_location="cpu")["version"] == CHECKPOINT_VERSION


def test_loading_a_checkpoint_written_before_versioning_still_works(tmp_path):
    """Every checkpoint from the runs already on disk lacks the field."""
    path = tmp_path / "old.pt"
    config = NetConfig(channels=8, blocks=1)
    save_checkpoint(path, small(), None, generation=3, config=config)
    payload = load_checkpoint(path, map_location="cpu")
    del payload["version"]
    torch.save(payload, path)

    reloaded = load_checkpoint(path, map_location="cpu")
    assert reloaded.get("version") is None
    net = PolicyValueNet(NetConfig(**reloaded["config"]))
    net.load_state_dict(reloaded["model"])
    assert reloaded["generation"] == 3
