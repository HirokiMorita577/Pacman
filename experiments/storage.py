"""Bounded, checksummed backups; never reuse a different experiment manifest."""
import hashlib
import json
import pickle
from pathlib import Path
import shutil

OUTPUTS = ("manifest.json", "result.json", "model.pt", "train.csv", "learning_curve.csv",
           "progress.json", "evaluation_dev.csv", "evaluation_final.json", "evaluation_final.csv")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(source) == sha256(destination):
            return True
        if shutil.disk_usage(destination.parent).free < source.stat().st_size + 64 * 1024 * 1024:
            print("BACKUP_DEFERRED_LOW_SPACE", source, flush=True)
            return False
        # Do not rename on mounted Drive; a second checkpoint slot protects the old one.
        with source.open("rb") as incoming, destination.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        if sha256(source) != sha256(destination):
            raise OSError("Backup checksum mismatch")
        return True
    except OSError as exc:
        print("BACKUP_DEFERRED", source, str(exc), flush=True)
        return False


def sync_trial(local, remote, final=False):
    local, remote = Path(local), Path(remote)
    ok = True
    for name in OUTPUTS:
        if (local / name).exists():
            ok = copy_verified(local / name, remote / name) and ok
    checkpoint = local / "training.pt.gz"
    if final:
        if not ok or not (remote / "result.json").exists() or not (remote / "model.pt").exists():
            return False
        receipt = {"files": {n: sha256(remote / n) for n in ("result.json", "model.pt")}}
        receipt_path = local / "saved_result.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        if not copy_verified(receipt_path, remote / receipt_path.name):
            return False
        # Only this runner's disposable replay checkpoints, after verified model/result persistence.
        for root in (local, remote):
            for name in ("training.pt.gz", "recovery_A.pt.gz", "recovery_B.pt.gz"):
                candidate = root / name
                if candidate.is_file():
                    try:
                        candidate.unlink()
                    except OSError:
                        print("CHECKPOINT_CLEANUP_DEFERRED", candidate, flush=True)
        return True
    if not checkpoint.exists():
        return ok
    state_path = local / "backup_state.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except (OSError, ValueError):
        print("INVALID_BACKUP_STATE_UPLOAD_DEFERRED", state_path, flush=True)
        return False
    progress = json.loads((local / "progress.json").read_text())
    if (state and progress["total_steps"] < progress["budget"]
            and progress["total_steps"] - state.get("total_steps", 0) < 30000):
        return ok
    slot = "B" if state.get("slot") == "A" else "A"
    target = remote / ("recovery_" + slot + ".pt.gz")
    if not copy_verified(checkpoint, target):
        return False
    record = dict(slot=slot, total_steps=progress["total_steps"], sha256=sha256(checkpoint))
    state_path.write_text(json.dumps(record), encoding="utf-8")
    return copy_verified(state_path, remote / "backup_state.json") and ok


def restore_trial(local, remote, manifest):
    from .engine import torch_load
    local, remote = Path(local), Path(remote)
    if not remote.exists():
        return
    receipt = remote / "saved_result.json"
    if receipt.exists():
        files = json.loads(receipt.read_text(encoding="utf-8")).get("files", {})
        if set(files) != {"result.json", "model.pt"}:
            raise ValueError("Invalid completion receipt")
        if all((remote / n).exists() and sha256(remote / n) == expected for n, expected in files.items()):
            result = json.loads((remote / "result.json").read_text(encoding="utf-8"))
            if result["manifest"] != manifest:
                raise ValueError("Backup manifest mismatch")
            for name in OUTPUTS:
                source = remote / name
                if source.exists():
                    local.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, local / name)
            return
    best_steps = -1
    best = None
    for path in (local / "training.pt.gz", remote / "recovery_A.pt.gz", remote / "recovery_B.pt.gz"):
        if not path.exists():
            continue
        try:
            state = torch_load(path, compressed=True)
            if state.get("schema") != 1 or state.get("manifest") != manifest:
                continue
            steps = state["progress"]["total_steps"]
            if steps > best_steps:
                best_steps, best = steps, path
        except (OSError, ValueError, EOFError, RuntimeError, KeyError, TypeError, AttributeError, pickle.UnpicklingError):
            print("INVALID_RECOVERY_CHECKPOINT", path, flush=True)
    if best is not None and best != local / "training.pt.gz":
        local.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, local / "training.pt.gz")
