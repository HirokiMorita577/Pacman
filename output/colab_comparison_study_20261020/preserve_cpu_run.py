"""One-time, verified preservation of the specific CPU run before a GPU switch."""
import gzip
import hashlib
import json
from pathlib import Path
import signal
import shutil

import psutil
import torch

parent = psutil.Process(17462)
assert "comparison_study.py" in " ".join(parent.cmdline())
local = Path("/content/pacman_comparison_study")
destination = Path("/content/drive/MyDrive/DQN_Pacman_comparisons_20261020_3c20ffba/pre_gpu_snapshot")
processes = [parent, *parent.children(recursive=True)]
paused = []
verified = False
try:
    for process in processes:
        process.send_signal(signal.SIGSTOP)
        paused.append(process)
    checkpoints = list(local.glob("factors/runs/*/training.pt.gz"))
    assert len(checkpoints) >= 2, "Wait for both active trials to checkpoint before switching"
    progress = []
    for path in checkpoints:
        with gzip.open(path, "rb") as stream:
            saved = torch.load(stream, map_location="cpu", weights_only=False)
        progress.append(dict(trial=path.parent.name, episodes=saved["progress"]["episode"],
                             steps=saved["progress"]["total_steps"]))
    files = []
    for source in local.rglob("*"):
        if not source.is_file() or source.name.endswith((".tmp", ".lock")):
            continue
        target = destination / source.relative_to(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, target
        files.append(dict(path=source.relative_to(local).as_posix(), sha256=expected))
    receipt = dict(files=files, checkpoints=progress, reason="CPU to GPU runtime switch")
    (destination / "preservation_receipt.json").write_text(json.dumps(receipt, indent=1))
    verified = True
    print("CPU_BACKUP_VERIFIED", json.dumps(progress), str(destination), flush=True)
finally:
    for process in paused:
        try:
            if verified:
                process.terminate()
            process.send_signal(signal.SIGCONT)
        except psutil.NoSuchProcess:
            pass
if verified:
    gone, alive = psutil.wait_procs(paused, timeout=15)
    print("CPU_RUN_STOPPED", len(gone), "remaining", [p.pid for p in alive], flush=True)
