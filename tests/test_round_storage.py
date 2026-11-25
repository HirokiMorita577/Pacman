"""colab_round2.py (第2・3回で使う Colab の保存・復旧) のテスト: A/B スロット、同期の間隔、失敗時の保持、
破損・設定違いのチェックポイントの除外、完了済み結果の復元"""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    storage = load("storage_for_round", "colab_ghost4_recovery.py")
    runner = load("round_runner", "colab_round2.py")
    local, drive = tmp_path / "local", tmp_path / "drive"
    (local / "runs").mkdir(parents=True)
    drive.mkdir()
    monkeypatch.setattr(storage, "LOCAL", local)
    monkeypatch.setattr(storage, "DRIVE", drive)
    monkeypatch.setattr(storage, "drive_free", lambda: 10 ** 12)
    runner.storage = storage
    return SimpleNamespace(storage=storage, runner=runner, local=local, drive=drive)


def write_checkpoint(path, manifest, steps):
    buf = io.BytesIO()
    torch.save({"manifest": manifest, "progress": {"total_steps": steps}}, buf)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(buf.getvalue())


def test_sync_uses_two_slots_and_respects_interval_and_budget(env):
    cp = env.local / "runs" / "job" / "training.pt.gz"
    cp.parent.mkdir(parents=True)
    synced = []
    for steps in [10000, 30000, 45000, 60000, 90000, 95000]:
        cp.write_bytes(str(steps).encode())
        synced.append(env.runner.sync_checkpoint(cp, steps, budget=95000))
    # 初回、前回の同期から3万ステップ以上 (10000→45000→90000)、予算到達 (95000) で同期する
    assert synced == [True, False, True, False, True, True]
    slots = sorted(p.name for p in (env.drive / "runs" / "job").glob("recovery_checkpoint_*.pt.gz"))
    assert slots == ["recovery_checkpoint_A.pt.gz", "recovery_checkpoint_B.pt.gz"]
    state = json.loads((cp.parent / "recovery_sync.json").read_text())
    assert state["steps"] == 95000
    assert hashlib.sha256((env.drive / "runs" / "job" / state["file"]).read_bytes()).hexdigest() == state["sha256"]


def test_failed_upload_keeps_previous_slot_and_state(env, monkeypatch):
    cp = env.local / "runs" / "job" / "training.pt.gz"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"first")
    assert env.runner.sync_checkpoint(cp, 30000, budget=200000)
    state = (cp.parent / "recovery_sync.json").read_bytes()
    cp.write_bytes(b"second")
    monkeypatch.setattr(env.storage, "upload", lambda src, dst: False)
    assert not env.runner.sync_checkpoint(cp, 60000, budget=200000)
    assert (cp.parent / "recovery_sync.json").read_bytes() == state
    assert (env.drive / "runs" / "job" / "recovery_checkpoint_A.pt.gz").read_bytes() == b"first"


def fake_study(manifest, run="job_s1"):
    return SimpleNamespace(experiment_manifest=lambda job: manifest, job_id=lambda job: run)


def test_restore_picks_largest_valid_checkpoint_and_skips_corrupt_or_mismatched(env, capsys):
    manifest = {"job": {"x": 1}}
    remote = env.drive / "runs" / "job_s1"
    write_checkpoint(remote / "recovery_checkpoint_A.pt.gz", manifest, 30000)
    write_checkpoint(remote / "recovery_checkpoint_B.pt.gz", {"job": {"x": 2}}, 90000)   # 設定が違う
    write_checkpoint(env.local / "runs" / "job_s1" / "training.pt.gz", manifest, 20000)
    env.runner.restore_job({}, fake_study(manifest))
    out = capsys.readouterr().out
    assert "RESUME_VERIFIED job_s1 30000" in out and "INVALID_CHECKPOINT" in out
    restored = env.runner.load_checkpoint(env.local / "runs" / "job_s1" / "training.pt.gz")
    assert restored["progress"]["total_steps"] == 30000
    (remote / "recovery_checkpoint_A.pt.gz").write_bytes(b"not a gzip file")             # 破損
    env.runner.restore_job({}, fake_study(manifest))
    assert "INVALID_CHECKPOINT" in capsys.readouterr().out
    assert env.runner.load_checkpoint(env.local / "runs" / "job_s1" / "training.pt.gz")["progress"]["total_steps"] == 30000


def test_restore_complete_result_checks_receipt_and_manifest(env, capsys):
    manifest = {"job": {"x": 1}}
    remote = env.drive / "runs" / "job_s1"
    remote.mkdir(parents=True)
    (remote / "result.json").write_text(json.dumps({"manifest": manifest}))
    (remote / "model.pt").write_bytes(b"weights")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (remote / "saved_result.json").write_text(json.dumps(
        {"sha256": {n: digest(remote / n) for n in ("result.json", "model.pt")}}))
    env.runner.restore_job({}, fake_study(manifest))
    assert "RESTORED_COMPLETE" in capsys.readouterr().out
    assert (env.local / "runs" / "job_s1" / "model.pt").read_bytes() == b"weights"
    with pytest.raises(ValueError, match="manifest mismatch"):
        env.runner.restore_job({}, fake_study({"job": {"x": 2}}))
    (remote / "model.pt").write_bytes(b"tampered")                 # 受領記録と合わない → 完了扱いしない
    capsys.readouterr()
    env.runner.restore_job({}, fake_study(manifest))
    assert "RESTORED_COMPLETE" not in capsys.readouterr().out
