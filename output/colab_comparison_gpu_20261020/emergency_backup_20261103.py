"""Snapshot completed trials whose verified Drive receipt is still absent.

This sidecar leaves the frozen training source and running workers untouched.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import zipfile
from pathlib import Path


LOCAL_RUNS = Path("/content/pacman_comparison_gpu_study/factors/runs")
DRIVE_RUNS = Path(
    "/content/drive/MyDrive/"
    "DQN_Pacman_comparisons_gpu_20261020_9e75afec/results/factors/runs"
)
METRIC_FILES = (
    "manifest.json",
    "result.json",
    "train.csv",
    "learning_curve.csv",
    "progress.json",
    "evaluation_dev.csv",
    "evaluation_final.json",
    "evaluation_final.csv",
)
RESERVE_BYTES = 20 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_archive(path: Path, trials: list[Path], names: tuple[str, ...]) -> int:
    added = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for trial in trials:
            for name in names:
                source = trial / name
                if source.is_file():
                    archive.write(source, arcname=f"runs/{trial.name}/{name}")
                    added += 1
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Archive corruption: {corrupt}")
    return added


def main() -> None:
    if not LOCAL_RUNS.is_dir() or not DRIVE_RUNS.is_dir():
        raise FileNotFoundError("Local study or mounted Drive runs directory is missing")
    trials = sorted(
        trial
        for trial in LOCAL_RUNS.iterdir()
        if trial.is_dir()
        and (trial / "result.json").is_file()
        and (trial / "evaluation_final.json").is_file()
        and not (DRIVE_RUNS / trial.name / "saved_result.json").is_file()
    )
    if not trials:
        print("SNAPSHOT 0 trials; no completed unsynced trials", flush=True)
        return

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    metrics = Path("/content") / f"pacman_emergency_metrics_{stamp}.zip"
    full = Path("/content") / f"pacman_emergency_full_{stamp}.zip"
    metric_files = write_archive(metrics, trials, METRIC_FILES)
    full_files = write_archive(full, trials, METRIC_FILES + ("model.pt",))
    models = sum((trial / "model.pt").is_file() for trial in trials)
    print(
        "SNAPSHOT",
        f"trials={len(trials)}",
        f"models={models}",
        f"metric_files={metric_files}",
        f"full_files={full_files}",
        f"metrics_MiB={metrics.stat().st_size / 2**20:.1f}",
        f"full_MiB={full.stat().st_size / 2**20:.1f}",
        flush=True,
    )

    destination = DRIVE_RUNS.parent / metrics.name
    free_bytes = shutil.disk_usage(DRIVE_RUNS).free
    if free_bytes >= metrics.stat().st_size + RESERVE_BYTES:
        shutil.copyfile(metrics, destination)
        if sha256(destination) != sha256(metrics):
            raise RuntimeError("Drive metrics copy SHA256 mismatch")
        print("DRIVE_METRICS", destination, "SHA256", sha256(destination), flush=True)
    else:
        print(
            "DRIVE_METRICS_DEFERRED",
            f"free_MiB={free_bytes / 2**20:.1f}",
            f"needed_MiB={(metrics.stat().st_size + RESERVE_BYTES) / 2**20:.1f}",
            flush=True,
        )
    print("PC_ARCHIVE_DOWNLOAD_REQUIRED", full, "SHA256", sha256(full), flush=True)


if __name__ == "__main__":
    main()
