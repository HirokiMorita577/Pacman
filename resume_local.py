"""Continue a verified Colab checkpoint as a separate local CUDA experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.config import digest, manifest, trial_id
from experiments.storage import sha256


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_source(plan_path, source):
    if (source / "result.json").exists() or (source / "saved_result.json").exists():
        raise ValueError("This trial has already finished training in the source experiment")
    receipt = read_json(source / "backup_state.json")
    if receipt["slot"] not in ("A", "B"):
        raise ValueError("Invalid checkpoint slot")
    checkpoint = source / f"recovery_{receipt['slot']}.pt.gz"
    if sha256(checkpoint) != receipt["sha256"]:
        raise ValueError("Checkpoint checksum differs from the Drive receipt")
    original = read_json(source / "manifest.json")
    config, seed = original["config"], original["seed"]
    expected_id = f"{config['algorithm']}_{digest(original)[:20]}_s{seed}"
    if original["schema"] != 1 or source.name != expected_id:
        raise ValueError("Source manifest schema or trial ID differs from the source trial")

    document = read_json(plan_path)
    plan, runtime = document["plan"], document["runtime"]
    if seed not in plan["seeds"] or not any(config == row["config"] for row in plan["conditions"]):
        raise ValueError("Checkpoint condition or seed is absent from the source plan")
    if any(original[k] != runtime[k] for k in ("python", "packages", "sources", "execution")):
        raise ValueError("Checkpoint runtime differs from the source plan's CUDA runtime")
    if original["execution"]["device"] != "cuda":
        raise ValueError("This migration requires a CUDA source checkpoint")

    root = Path(__file__).resolve().parent
    names = ["pacman_env.py", "pso_optimizer.py", *(p.relative_to(root).as_posix()
             for p in (root / "experiments").glob("*.py"))]
    # Windows checkouts can use CRLF while the source bundle uses LF.
    local_lf = {
        name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in names
    }
    if original["sources"] != local_lf:
        raise ValueError("Local training sources differ from the checkpoint sources")
    if not 0 < receipt["total_steps"] < config["train"]["steps"]:
        raise ValueError("Receipt does not point to an unfinished training position")
    return original, receipt, checkpoint


def prepare(plan_path, source, out):
    from experiments.engine import atomic_json, torch_load, torch_save

    original, receipt, checkpoint = verify_source(plan_path, source)
    config, seed = original["config"], original["seed"]
    local = manifest(config, seed)
    if local["execution"]["device"] != "cuda":
        raise ValueError("Set PACMAN_DEVICE=cuda for a local CUDA continuation")
    saved = torch_load(checkpoint, compressed=True)
    if saved["schema"] != 1 or saved["manifest"] != original:
        raise ValueError("Checkpoint schema or manifest differs from the source trial manifest")
    progress = saved["progress"]
    steps, episode = progress["total_steps"], progress["episode"]
    if steps != receipt["total_steps"]:
        raise ValueError("Checkpoint does not represent a verified, unfinished training position")
    rows = progress["rows"]
    if not rows or len(rows) != episode or rows[-1]["total_steps"] != steps or rows[-1]["learning_episode"] != episode:
        raise ValueError("Checkpoint training history does not match its counters")
    if len(saved["rng"].get("cuda") or []) != 1:
        raise ValueError("Expected a checkpoint from a single CUDA device")

    destination = out / "runs" / trial_id(config, seed)
    provenance = dict(source_trial=source.name, source_plan_sha256=sha256(plan_path),
                      source_checkpoint_sha256=receipt["sha256"], source_manifest=original,
                      local_manifest=local, resumed_at_episode=episode, resumed_at_step=steps)
    record = destination / "migration.json"
    target = destination / "training.pt.gz"
    result = destination / "result.json"
    if record.exists():
        if read_json(record) != provenance:
            raise ValueError("Existing local trial has different migration provenance")
    elif target.exists() or result.exists():
        raise ValueError("Existing local trial has no migration provenance")
    else:
        atomic_json(record, provenance)
    if not (target.exists() or result.exists()):
        saved["manifest"] = local
        torch_save(target, saved, compressed=True)
    print(f"LOCAL_RESUME {destination.name} source={source.name} episode={episode} "
          f"steps={steps}/{config['train']['steps']}", flush=True)
    return config, seed, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--source-trial", required=True, type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true", help="check source files without importing PyTorch")
    args = parser.parse_args()
    if args.verify_only:
        for source in args.source_trial:
            _, receipt, _ = verify_source(args.source_plan, source)
            print(f"SOURCE_VERIFIED {source.name} steps={receipt['total_steps']}", flush=True)
        return
    if args.out is None:
        parser.error("--out is required for preparation and training")
    out = args.out.resolve()
    source_roots = [args.source_plan.resolve().parent, *(source.resolve() for source in args.source_trial)]
    if any(out.is_relative_to(root) or root.is_relative_to(out) for root in source_roots):
        parser.error("Use a separate output directory for the local experiment")

    from experiments.engine import evaluate_final, run_lock, run_trial

    with run_lock(args.out / "local_resume.lock"):
        prepared = [prepare(args.source_plan, source, args.out) for source in args.source_trial]
        if not args.prepare_only:
            for config, seed, destination in prepared:
                run_trial(config, seed, args.out)
                evaluate_final(destination)


if __name__ == "__main__":
    main()
