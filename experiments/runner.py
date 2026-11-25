"""Execute and resume a frozen catalog plan, preserving failures as failures."""
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import traceback

from .config import manifest, trial_id
from .engine import atomic_json, evaluate_final, run_trial, write_csv
from .report import report
from .storage import copy_verified


def sync_report(out, backup):
    if not backup:
        return
    out, backup = Path(out), Path(backup)
    for name in ("COMPLETE.json", "summary.csv", "summary.json", "trial_status.csv",
                 "evaluation_episodes.csv", "reward_parameters.csv", "paired_differences.csv",
                 "figure_manifest.json", "REPORT.md"):
        if (out / name).exists():
            copy_verified(out / name, backup / name)
    for image in (out / "figures").rglob("*.png"):
        copy_verified(image, backup / image.relative_to(out))


def execute_one(task):
    condition, seed, out, backup, final = task
    result = run_trial(condition["config"], seed, out, backup=backup)
    if final:
        evaluate_final(Path(out) / "runs" / result["id"], backup=backup)
    return dict(condition=condition["id"], train_seed=seed, id=result["id"],
                score=result["evaluation"]["score_mean"], status="complete")


def run_plan(plan, out, workers=2, backup=None, final=True, retry_failed=False, report_every=25):
    if workers < 1:
        raise ValueError("workers must be positive")
    out = Path(out)
    if out.resolve().is_relative_to(Path("/content/drive").resolve()):
        raise ValueError("Use local --out and mounted Drive --backup")
    out.mkdir(parents=True, exist_ok=True)
    first = manifest(plan["conditions"][0]["config"], plan["seeds"][0])
    runtime = {k: first[k] for k in ("python", "packages", "sources", "execution")}
    document = dict(plan=plan, runtime=runtime)
    plan_path = out / "plan.json"
    if backup and not plan_path.exists() and (Path(backup) / "plan.json").exists():
        import shutil
        shutil.copy2(Path(backup) / "plan.json", plan_path)
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != document:
        raise ValueError("Frozen plan/source/runtime differs; use a new output directory")
    atomic_json(plan_path, document)
    if backup:
        copy_verified(plan_path, Path(backup) / "plan.json")
    tasks, statuses = [], []
    for condition in plan["conditions"]:
        for seed in plan["seeds"]:
            trial = trial_id(condition["config"], seed)
            failure = out / "runs" / trial / "FAILED.json"
            if failure.exists() and not retry_failed:
                statuses.append(json.loads(failure.read_text(encoding="utf-8")))
                continue
            tasks.append((condition, seed, str(out), str(backup) if backup else None, final))
    print(f"PLAN_START conditions={plan['unique_conditions']} trials={plan['training_trials']} "
          f"environment_steps={plan['total_environment_steps']} workers={workers}", flush=True)

    def record(task, result=None, error=None):
        condition, seed = task[:2]
        if error is not None:
            result = dict(condition=condition["id"], train_seed=seed,
                          id=trial_id(condition["config"], seed), status="failed",
                          error=str(error), traceback="".join(traceback.format_exception(error)))
            atomic_json(out / "runs" / result["id"] / "FAILED.json", result)
        else:
            stale = out / "runs" / result["id"] / "FAILED.json"
            if stale.exists():
                stale.unlink()
        statuses.append(result)
        write_csv(out / "trial_status.csv", statuses)
        print(f"PLAN_PROGRESS {len(statuses)}/{plan['training_trials']} {result['status']} {result['id']}", flush=True)
        if backup:
            copy_verified(out / "trial_status.csv", Path(backup) / "trial_status.csv")
        if report_every and len(statuses) % report_every == 0:
            report(out, plan)
            sync_report(out, backup)

    if workers == 1:
        for task in tasks:
            try:
                result = execute_one(task)
            except Exception as exc:
                record(task, error=exc)
            else:
                record(task, result)
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(execute_one, task): task for task in tasks}
            for future in cf.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    record(futures[future], error=exc)
                else:
                    record(futures[future], result)
    figures = report(out, plan)
    completed = sum(s["status"] == "complete" for s in statuses)
    failed = sum(s["status"] == "failed" for s in statuses)
    summary = dict(expected_trials=plan["training_trials"], completed_trials=completed,
                   failed_trials=failed, all_trials_accounted_for=completed + failed == plan["training_trials"],
                   final_evaluation=final, **figures)
    atomic_json(out / "COMPLETE.json", summary)
    sync_report(out, backup)
    print("PLAN_COMPLETE", summary, flush=True)
    return summary
