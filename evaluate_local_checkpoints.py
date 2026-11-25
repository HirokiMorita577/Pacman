"""Evaluate read-only snapshots of migrated models on development episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from experiments.agents import Agent
from experiments.config import digest
from experiments.engine import atomic_json, evaluate, make_env, torch_load, write_csv
from experiments.runtime import configure
from experiments.storage import sha256


def evaluate_snapshot(trial, target):
    provenance = json.loads((trial / "migration.json").read_text(encoding="utf-8"))
    with TemporaryDirectory() as directory:
        snapshot = Path(directory) / "training.pt.gz"
        shutil.copyfile(trial / snapshot.name, snapshot)
        snapshot_hash = sha256(snapshot)
        state = torch_load(snapshot, compressed=True)
    frozen = state["manifest"]
    if frozen != provenance["local_manifest"]:
        raise ValueError(f"Manifest mismatch for {trial.name}")

    config = frozen["config"]
    execution = configure(config["algorithm"])
    env = make_env(config)
    try:
        agent = Agent(config, env.observation_space.shape[0], env.action_mask_indices)
        agent.load_state_dict(state["agent"], training=False)
    finally:
        env.close()
    steps = state["progress"]["total_steps"]
    episode = state["progress"]["episode"]
    del state
    output = target / f"{trial.name}_{steps}.json"
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))["summary"]
        if existing["checkpoint_sha256"] == snapshot_hash:
            print(json.dumps(existing), flush=True)
            return existing
    evaluation = evaluate(agent, config, phase="dev")
    summary = dict(trial=trial.name, source_trial=provenance["source_trial"],
                   condition=digest(config)[:20], train_seed=frozen["seed"],
                   source_steps=provenance["resumed_at_step"], checkpoint_steps=steps,
                   checkpoint_episode=episode, budget=config["train"]["steps"],
                   phase=evaluation["phase"], episodes=len(evaluation["episodes"]),
                   score_mean=evaluation["score_mean"], clear_rate=evaluation["clear_rate"],
                   checkpoint_sha256=snapshot_hash, evaluation_device=execution["device"])
    atomic_json(output, {
        "summary": summary, "config": config, "evaluation": evaluation,
    })
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ["PACMAN_DEVICE"] = "cpu"
    os.environ["PACMAN_TORCH_THREADS"] = "1"
    trials = sorted(args.runs.glob("*/migration.json"))
    if not trials:
        parser.error("No migrated trials found")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate_snapshot, (path.parent for path in trials), repeat(args.out)))
    write_csv(args.out / "checkpoint_evaluation.csv", rows)


if __name__ == "__main__":
    main()
