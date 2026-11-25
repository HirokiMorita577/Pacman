"""
ゴースト4体・残機3のクリア率改善 (第2回比較, 2026-10-07)。

前回 (output/ghost4_retrain_20261006) の改善版 = tactical観測 + 壁方向マスク + γ0.95 + 死亡−100 + クリア0 を基準 (control) にし、
失敗解析 (output/ghost4_diagnosis_20261007) で根拠が得られた変更を1つずつ加えて比べる。ゲーム条件 (敵4体・残機3・
全エサでクリア・最大1,000ステップ) は変えない。

  control   : 前回の改善版と同じ設定を、同じ PC・同じソフトの版で学習し直した対照
  safe_obs  : 観測に行動ごとの先読み特徴量を追加 (observation_version="tactical_safe")   … 仮説F
  leave_fix : エサを食べた直後の leave_dot を付けない (fix_leave_dot=True)                … 仮説C
  gamma098  : 割引率 0.98                                                                  … 仮説B

学習: seed 101/202/303、各20万環境ステップ (前回と同じ)。
開発用の評価: seed 930000〜930029 (前回と同じ集合。既に結果を見ているので比較・選択だけに使う)。
最終評価: 未使用の seed 940000〜940099 を先に固定し、比較で選んだ条件と control だけを評価する (--final)。

  python ghost4_round2.py --out output/ghost4_round2_20261007 --workers 6
  python ghost4_round2.py --out output/ghost4_round2_20261007 --final            # 選択を保存してから最終評価
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import reward_study as study
from dqn_agent import load_agent
from pacman_env import PacManEnv

BASE = dict(exp="ghost4_round2", num_ghosts=4, lives=3, include_lives=True,
            observation_version="tactical", mask_illegal_actions=True,
            rewards=study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0), gamma=0.95)
VARIANTS = {
    "control": {},
    "safe_obs": {"observation_version": "tactical_safe"},
    "leave_fix": {"env_options": {"fix_leave_dot": True}},
    "gamma098": {"gamma": 0.98},
}
DEV_SEED0, DEV_EPISODES = 930000, 30
FINAL_SEED0, FINAL_EPISODES = 940000, 100


def jobs_for(args):
    return [dict(BASE, **VARIANTS[v], label=v, seed=seed, episodes=args.steps, step_budget=args.steps,
                 eval_episodes=DEV_EPISODES, eval_seed0=DEV_SEED0, checkpoint_every=50,
                 compress_checkpoint=True, save_model=True, show_progress=True, runs_dir=str(Path(args.out) / "runs"))
            for v in args.variants for seed in args.seeds]


def per_seed_rows(results, metric_keys):
    rows = []
    for r in results:
        ev = r["eval"]
        rows.append(dict(variant=r["label"], train_seed=r["seed"], clears=sum(ev["clears"]), episodes=len(ev["clears"]),
                         train_minutes=r["train_minutes"], episodes_trained=r["episodes_completed"],
                         **{k: ev[k] for k in metric_keys}))
    return rows


METRICS = ["clear_rate", "pellet_fraction_mean", "lives_lost_mean", "score_mean", "steps_mean", "blocked_action_rate"]


def compare(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(args)
    plan = dict(jobs=[{**j, "id": study.job_id(j)} for j in jobs], dev_seed0=DEV_SEED0, dev_episodes=DEV_EPISODES,
                final_seed0=FINAL_SEED0, final_episodes=FINAL_EPISODES, primary_metric="clear_rate",
                note="dev seeds 930000- were already inspected; use only for comparison/selection. "
                     "final seeds 940000-940099 are reserved for the selected variant and control.")
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing plan differs; use a separate output directory")
    study.atomic_json(plan_path, plan)
    results = study.run_jobs(jobs, args.workers)
    study.atomic_json(out / "results.json", results)
    seed_rows = per_seed_rows(results, METRICS)
    rows = []
    for v in args.variants:
        g = [r for r in seed_rows if r["variant"] == v]
        rows.append(dict(variant=v, seeds=len(g), clears=sum(r["clears"] for r in g), episodes=sum(r["episodes"] for r in g),
                         clears_by_seed="/".join(str(r["clears"]) for r in g),
                         **{k: float(np.mean([r[k] for r in g])) for k in METRICS},
                         train_minutes_mean=float(np.mean([r["train_minutes"] for r in g]))))
    study.atomic_json(out / "evaluation_dev.json", {"by_variant": rows, "by_seed": seed_rows})
    for name, data in (("evaluation_dev.csv", rows), ("evaluation_dev_by_seed.csv", seed_rows)):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    study.atomic_json(out / "COMPLETE_dev.json", dict(jobs=len(results), steps_per_seed=args.steps, dev_seed0=DEV_SEED0))
    print(json.dumps(rows, ensure_ascii=False, indent=1), flush=True)


def final(args):
    """開発用の結果で条件を1つ選び (selection.json を先に保存)、選んだ条件と control を未使用seedで評価する。学習はしない"""
    out = Path(args.out).resolve()
    dev = json.loads((out / "evaluation_dev.json").read_text(encoding="utf-8"))["by_variant"]
    sel_path = out / "selection.json"
    if sel_path.exists():
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
    else:
        candidates = [r for r in dev if r["variant"] != "control"]
        best = max(candidates, key=lambda r: (r["clear_rate"], r["pellet_fraction_mean"]))
        selection = dict(selected=best["variant"], rule="dev clear_rate, then pellet_fraction_mean",
                         final_seed0=FINAL_SEED0, final_episodes=FINAL_EPISODES)
        study.atomic_json(sel_path, selection)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    rows = []
    for r in results:
        if r["label"] not in ("control", selection["selected"]):
            continue
        env_options = r.get("env_options", {})
        env = PacManEnv(num_ghosts=r["num_ghosts"], lives=r["lives"], include_lives=r["include_lives"],
                        observation_version=r["observation_version"], **env_options)
        agent = load_agent(str(out / "runs" / r["id"] / "model.pt"), env.observation_space.shape[0], 4)
        ev = study.evaluate(agent, r["rewards"], FINAL_EPISODES, r["lives"], seed0=FINAL_SEED0,
                            include_lives=r["include_lives"], num_ghosts=r["num_ghosts"],
                            observation_version=r["observation_version"], env_options=env_options)
        rows.append(dict(variant=r["label"], train_seed=r["seed"], clears=sum(ev["clears"]), episodes=FINAL_EPISODES,
                         **{k: ev[k] for k in METRICS}))
        print(rows[-1], flush=True)
    study.atomic_json(out / "evaluation_final.json", dict(selection=selection, by_seed=rows))
    with (out / "evaluation_final.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--final", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve().is_relative_to(Path("/content/drive").resolve()):
        p.error("Train on local storage; copy results to Drive afterwards")
    final(a) if a.final else compare(a)


if __name__ == "__main__":
    main()
