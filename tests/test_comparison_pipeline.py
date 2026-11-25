import copy
import json
from pathlib import Path

import pytest

from experiments.catalog import build_plan
from experiments.config import DEFAULT, merge, trial_id, validate
from experiments.engine import run_trial, torch_load
from experiments.runner import run_plan
from experiments.search import optimize, run_search
from experiments.storage import restore_trial, sha256, sync_trial


def config():
    return validate(merge(DEFAULT, {
        "agent": {"hidden_sizes": [8]},
        "algorithm_options": {"batch_size": 2, "warmup": 2, "buffer_capacity": 32},
        "env": {"classic_lookahead": False, "max_steps": 4},
        "train": {"steps": 12, "explore_steps": 8, "eval_every": 4,
                  "curve_episodes": 1, "checkpoint_episodes": 1},
        "evaluation": {"episodes": 2},
    }))


def test_complete_plan_emits_pngs_with_learning_episodes_not_environment_steps(tmp_path, monkeypatch):
    from experiments import report as reports
    from PIL import Image
    axes = []
    original = reports.finish_figure

    def finish(fig, ax, path, title, ylabel, x_label="Learning episodes"):
        axes.append((x_label, [list(line.get_xdata()) for line in ax.lines]))
        return original(fig, ax, path, title, ylabel, x_label)

    monkeypatch.setattr(reports, "finish_figure", finish)
    grid = dict(name="power", axes={"rewards.power": [0, 200]})
    plan = build_plan([], [1, 2], base=config(), grid=grid)
    summary = run_plan(plan, tmp_path, workers=1, final=True, report_every=0)
    assert summary["completed_trials"] == 4 and summary["failed_trials"] == 0
    assert summary["images"] > 15
    assert axes and all("Learning episodes" in label for label, _ in axes)
    assert all(0 < x <= 3 for _, lines in axes for line in lines for x in line)
    images = list((tmp_path / "figures").rglob("*.png"))
    with Image.open(images[0]) as image:
        assert image.width >= 1000 and image.height >= 600
    figure_manifest = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert {f["x_axis"] for f in figure_manifest} == {"learning_episode"}
    assert (tmp_path / "paired_differences.csv").exists()
    assert (tmp_path / "reward_parameters.csv").exists()


def test_spawn_workers_and_cached_runs_produce_a_complete_plan(tmp_path):
    plan = build_plan([], [3, 4], base=config(),
                      grid=dict(name="gamma", axes={"agent.gamma": [0.9]}))
    first = run_plan(plan, tmp_path, workers=2, final=False, report_every=0)
    second = run_plan(plan, tmp_path, workers=1, final=False, report_every=0)
    assert first["completed_trials"] == second["completed_trials"] == 2
    assert not first["failed_trials"]


def test_verified_backup_restores_weights_and_drops_only_own_replay_checkpoints(tmp_path):
    cfg = config()
    remote = tmp_path / "remote"
    a = run_trial(cfg, 5, tmp_path / "local", backup=remote)
    folder = remote / "runs" / a["id"]
    assert (folder / "saved_result.json").exists()
    assert not list(folder.glob("recovery_*.pt.gz"))
    assert not (tmp_path / "local" / "runs" / a["id"] / "training.pt.gz").exists()
    restored = run_trial(cfg, 5, tmp_path / "restored", backup=remote)
    assert a["evaluation"] == restored["evaluation"]
    assert sha256(folder / "model.pt") == sha256(tmp_path / "restored" / "runs" / a["id"] / "model.pt")


def test_corrupt_recovery_slot_does_not_hide_a_valid_one(tmp_path):
    cfg = config()
    cfg["train"]["steps"] = 100
    remote = tmp_path / "remote"
    paused = run_trial(cfg, 6, tmp_path / "local", stop_after_episodes=1, backup=remote)
    folder = remote / "runs" / paused["id"]
    assert (folder / "recovery_A.pt.gz").exists()
    (folder / "recovery_B.pt.gz").write_bytes(b"broken gzip")
    local = tmp_path / "restored" / "runs" / paused["id"]
    source = tmp_path / "local" / "runs" / paused["id"]
    manifest = json.loads((source / "manifest.json").read_text())
    restore_trial(local, folder, manifest)
    saved = torch_load(local / "training.pt.gz", compressed=True)
    assert saved["progress"]["total_steps"] == 4


def test_low_space_keeps_the_previous_verified_checkpoint(tmp_path, monkeypatch):
    from experiments import storage
    cfg = config()
    cfg["train"]["steps"] = 100
    paused = run_trial(cfg, 7, tmp_path / "local", stop_after_episodes=1, backup=tmp_path / "remote")
    source = tmp_path / "local" / "runs" / paused["id"]
    remote = tmp_path / "remote" / "runs" / paused["id"]
    before = sha256(remote / "recovery_A.pt.gz")
    state = json.loads((source / "progress.json").read_text())
    state["total_steps"] = 40000
    (source / "progress.json").write_text(json.dumps(state))
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: type("Usage", (), {"free": 1})())
    assert not sync_trial(source, remote)
    assert sha256(remote / "recovery_A.pt.gz") == before


def test_real_search_selects_before_heldout_and_prunes_reproducible_candidate_weights(tmp_path):
    cfg = config()
    summaries = run_search(tmp_path, cfg, ["double_dqn"], ["gamma"], ["baseline", "pso"],
                           [42], [11], [12], candidates=2, particles=1, search_steps=8)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["final"]
    root = tmp_path / "double_dqn_gamma_seed42"
    for method in ("baseline", "pso"):
        selection = json.loads((root / method / "selection.json").read_text())
        assert selection["final_train_seeds"] == [12] and selection["search_complete"]
        assert selection["selected_config"]["train"]["steps"] == 12
        assert not list((root / method / "candidates").rglob("model.pt"))
        assert list((root / method / "final").rglob("model.pt"))
        assert list((root / method / "candidates").rglob("MODEL_PRUNED.json"))
    assert (root / "figures" / "best_fitness.png").exists()
    assert (root / "figures" / "heldout_score_mean.png").exists()
    again = run_search(tmp_path, cfg, ["double_dqn"], ["gamma"], ["baseline", "pso"],
                       [42], [11], [12], candidates=2, particles=1, search_steps=8)
    assert summaries == again


def test_search_rejects_a_reference_outside_its_bounds_before_evaluation(tmp_path):
    called = []
    with pytest.raises(ValueError, match="inside"):
        optimize("pso", config(), {"agent.gamma": {"low": 0.1, "high": 0.2}}, 1, 2, 1,
                 tmp_path, lambda c: called.append(c))
    assert not called


def test_changing_reward_or_algorithm_cannot_reuse_model_identity():
    cfg = config()
    assert trial_id(cfg, 1) != trial_id(merge(cfg, {"rewards": {"power": 200}}), 1)
    other = copy.deepcopy(cfg)
    other.update(algorithm="sarsa", algorithm_options={})
    assert trial_id(cfg, 1) != trial_id(other, 1)
