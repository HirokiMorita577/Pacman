import copy
import pytest
import torch

from experiments.config import ALGORITHMS, DEFAULT, HEURISTICS, ONLINE, merge, validate
from experiments.engine import run_trial, load_saved, torch_load
from experiments.runtime import device_name, execution_info


def tiny(algorithm):
    options = (dict(rollout_steps=6, epochs=1, batch_size=2) if algorithm == "ppo" else
               ({} if algorithm in ONLINE else dict(batch_size=2, warmup=2, buffer_capacity=16)))
    return validate(merge(DEFAULT, dict(
        algorithm=algorithm, algorithm_options=options, agent=dict(hidden_sizes=[8]),
        env=dict(classic_lookahead=False, max_steps=4),
        train=dict(steps=12, explore_steps=8, checkpoint_episodes=1, eval_every=0),
        evaluation=dict(episodes=1))))


def test_explicit_device_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("PACMAN_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        device_name("dqn")
    assert device_name("greedy_food") == "cpu"
    monkeypatch.setenv("PACMAN_DEVICE", "auto")
    assert device_name("dqn") == "cpu"
    monkeypatch.setenv("PACMAN_DEVICE", "typo")
    with pytest.raises(ValueError):
        device_name()


def test_execution_metadata_records_backend_and_thread_count(monkeypatch):
    monkeypatch.setenv("PACMAN_DEVICE", "cpu")
    monkeypatch.setenv("PACMAN_TORCH_THREADS", "2")
    assert execution_info()["torch_threads"] == 2
    monkeypatch.setenv("PACMAN_TORCH_THREADS", "0")
    with pytest.raises(ValueError):
        execution_info()


def test_legacy_cpu_checkpoint_never_initializes_cuda(tmp_path, monkeypatch):
    from dqn_agent import DQNAgent

    agent = DQNAgent(4, 2, device="cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU checkpoint must not access CUDA RNG, including in forked workers")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    path = tmp_path / "training.pt"
    agent.save_training(path, {"episode": 1}, {"test": True})
    saved = torch.load(path, weights_only=False)
    assert saved["cuda_rng"] is None
    # Older CPU snapshots could contain an unrelated CUDA RNG state.
    saved["cuda_rng"] = [torch.tensor([1], dtype=torch.uint8)]
    torch.save(saved, path)
    assert agent.load_training(path, {"test": True}) == {"episode": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA GPU")
@pytest.mark.parametrize("algorithm", [a for a in ALGORITHMS if a not in HEURISTICS])
def test_cuda_train_checkpoint_reload_and_resume(tmp_path, monkeypatch, algorithm):
    monkeypatch.setenv("PACMAN_DEVICE", "cuda")
    monkeypatch.setenv("PACMAN_TORCH_THREADS", "1")
    cfg = tiny(algorithm)
    complete = run_trial(cfg, 27, tmp_path / "full")
    paused = run_trial(cfg, 27, tmp_path / "resume", stop_after_episodes=1)
    assert paused["paused"]
    saved = torch_load(tmp_path / "resume" / "runs" / paused["id"] / "training.pt.gz", compressed=True)
    assert saved["rng"]["cuda"] is not None
    resumed = run_trial(cfg, 27, tmp_path / "resume")
    first, _ = load_saved(tmp_path / "full" / "runs" / complete["id"])
    second, _ = load_saved(tmp_path / "resume" / "runs" / resumed["id"])
    assert first.device.type == "cuda" and second.device.type == "cuda"
    assert first.updates > 0
    assert complete["evaluation"] == resumed["evaluation"]
    for left, right in zip(first.network.parameters(), second.network.parameters()):
        assert torch.equal(left, right)
