"""Warm-start a longer DQN run from a completed model without changing the 200k-step result.

The completed trial intentionally discards optimizer/replay checkpoints. This starts a
new 1M-step experiment from its learned network with fresh optimizer/replay state.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from experiments.agents import Agent
from experiments.config import manifest, trial_id, validate
from experiments.engine import make_env, rng_state, run_trial, seed_all, torch_load, torch_save


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(source, out, target_steps, backup=None):
    source = Path(source).resolve()
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    original = result["config"]
    seed = result["train_seed"]
    if original["algorithm"] != "dqn" or result["total_steps"] != 200000:
        raise ValueError("Expected a completed 200k-step DQN source")
    if result["model_sha256"] != hashlib.sha256((source / "model.pt").read_bytes()).hexdigest():
        raise ValueError("Source model checksum mismatch")
    if result["manifest"] != manifest(original, seed):
        raise ValueError("Source code/runtime differs from the frozen model manifest")
    if target_steps <= result["total_steps"]:
        raise ValueError("Target steps must exceed the completed source steps")
    config = copy.deepcopy(original)
    config["train"]["steps"] = target_steps
    config = validate(config)
    destination = Path(out).resolve() / "runs" / trial_id(config, seed)
    if destination.exists() and any(destination.iterdir()):
        receipt = destination / "warm_start.json"
        if not receipt.is_file():
            raise FileExistsError(f"Existing directory has no warm-start receipt: {destination}")
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if (previous.get("source_model_sha256") != result["model_sha256"]
                or previous.get("target_steps") != target_steps
                or previous.get("source_trial") != result["id"]):
            raise ValueError("Existing warm-start experiment differs from requested source/budget")
        remote = Path(backup).resolve() / "runs" / destination.name if backup else None
        has_local = any((destination / name).is_file() for name in ("training.pt.gz", "result.json"))
        has_remote = remote is not None and any((remote / name).is_file() for name in
                                                ("recovery_A.pt.gz", "recovery_B.pt.gz", "training.pt.gz", "result.json"))
        if not has_local and not has_remote:
            raise ValueError("Existing warm-start state is missing its checkpoint/result")
        return config, seed, destination
    destination.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    env = make_env(config)
    agent = Agent(config, env.observation_space.shape[0], env.action_mask_indices)
    saved = torch_load(source / "model.pt")
    if saved["manifest"] != result["manifest"]:
        raise ValueError("Source model manifest differs from result.json")
    agent.load_state_dict(saved["agent"], training=False)
    if agent.target is not None:
        agent.target.load_state_dict(agent.network.state_dict())
    rows = read_rows(source / "train.csv")
    curves = read_rows(source / "learning_curve.csv")
    if len(rows) != result["episodes_completed"] or len(curves) != 10:
        raise ValueError("Source history is incomplete")
    progress = dict(episode=result["episodes_completed"], total_steps=result["total_steps"],
                    rows=rows, curves=curves, wall_seconds=result["wall_seconds"])
    torch_save(destination / "training.pt.gz",
               dict(schema=1, manifest=manifest(config, seed), agent=agent.state_dict(),
                    progress=progress, rng=rng_state()), compressed=True)
    (destination / "warm_start.json").write_text(json.dumps(dict(
        source_trial=result["id"], source_model_sha256=result["model_sha256"],
        source_steps=result["total_steps"], target_steps=target_steps,
        optimizer="fresh", replay="fresh", target_network="copied_from_online",
        continuation="warm_start_from_weights"), indent=2), encoding="utf-8")
    env.close()
    return config, seed, destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, help="Completed DQN trial folder")
    parser.add_argument("--out", required=True, help="New local experiment output root")
    parser.add_argument("--backup", help="Separate Google Drive backup root")
    parser.add_argument("--target-steps", type=int, default=1000000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config, seed, destination = prepare(args.source_root, args.out, args.target_steps, args.backup)
    print("WARM_START_READY", destination, flush=True)
    if not args.prepare_only:
        print("WARM_START_RUNNING", seed, args.target_steps, flush=True)
        run_trial(config, seed, args.out, backup=args.backup)


if __name__ == "__main__":
    main()
