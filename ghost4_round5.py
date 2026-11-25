"""
ゴースト4体・残機3：パワーエサの報酬の比較 (第5回比較, 2026-10-15 設計)。

第4回 (output/ghost4_round4_20261014) で最良だった safe_lookahead (敵の実際の動き方で5手先を予測) を基準にし、
パワーエサを取ったときの学習用の報酬だけを変えて比べる。ゲームの得点 (パワーエサ50点) とゲーム条件は変えない。

  power0    : 0     (報酬なし。クリアに必要なエサとしてだけ数える)
  power10   : 10    (普通のエサと同じ)
  power50   : 50    (今の設定 = ゲームの得点と同じ。基準)
  power200  : 200   (敵1体を食べたのと同じ)

ほかの報酬 (エサ10・敵の捕食200・死亡100・クリア0 など)、γ0.95、学習量20万ステップ、探索8万ステップは全条件で同じ。
第4回と同じ方法: 学習seed 5本 (101〜505)、開発用評価は学習seedごとに100回 (930000〜930099)、
選択はせず4条件とも未使用の評価seed 970000〜970099 で最終評価する (plan.json で先に固定)。
主指標はクリア率。あわせて平均得点・捕食数・パワーエサを取ったタイミング (最初の1個を取ったステップ)・
パワーエサ1個あたりの捕食数を比べる。

  python ghost4_round5.py --out output/ghost4_round5 --workers 10
  python ghost4_round5.py --out output/ghost4_round5 --final
"""
import argparse
import json
from pathlib import Path

import numpy as np

import reward_study as study
import ghost4_round3 as r3

BASE_REWARDS = study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0)
BASE = dict(r3.BASE, exp="ghost4_round5", env_options={"classic_lookahead": True})
POWER = {"power0": 0.0, "power10": 10.0, "power50": 50.0, "power200": 200.0}
VARIANTS = {v: dict(job={"rewards": study.with_reward(BASE_REWARDS, power=p)}, steps=1) for v, p in POWER.items()}
BASELINE = "power50"
SEEDS = [101, 202, 303, 404, 505]
CURVE = r3.CURVE
DEV_SEED0, DEV_EPISODES = 930000, 100
FINAL_SEED0, FINAL_EPISODES = 970000, 100
USED_EVAL_SETS = {**r3.USED_EVAL_SETS, "950000-950099": "第3回の最終評価 (結果を見た)",
                  "960000-960099": "第4回の最終評価 (結果を見た)"}
POWER_METRICS = ["power_first_step_mean", "ghosts_per_power"]


def make_job(variant, seed, steps, out):
    v = VARIANTS[variant]
    budget = steps * v["steps"]
    return dict(BASE, **v["job"], label=variant, seed=seed, episodes=budget, step_budget=budget,
                eval_episodes=DEV_EPISODES, eval_seed0=DEV_SEED0, curve_eval=CURVE, checkpoint_every=50,
                compress_checkpoint=True, save_model=True, show_progress=True, runs_dir=str(Path(out) / "runs"))


def jobs_for(args):
    return [make_job(v, seed, args.steps, args.out) for v in args.variants for seed in args.seeds]


def plan_for(args):
    jobs = [make_job(v, s, args.steps, args.out) for v in VARIANTS for s in args.seeds]
    strip = lambda j: {k: v for k, v in j.items() if k != "runs_dir"}
    return dict(jobs=[{**strip(j), "id": study.job_id(j)} for j in jobs], baseline=BASELINE, power_values=POWER,
                primary_metric="clear_rate", dev_seed0=DEV_SEED0, dev_episodes=DEV_EPISODES, curve=CURVE,
                final_seed0=FINAL_SEED0, final_episodes=FINAL_EPISODES, final_train_seeds=list(args.seeds),
                used_eval_sets=USED_EVAL_SETS,
                note="no selection step: all variants go to the final evaluation on unused seeds")


def check_plan(out, args):
    plan = plan_for(args)
    path = out / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing plan differs; use a separate output directory")
    study.atomic_json(path, plan)
    return plan


def power_row(ev):
    return {k: ev.get(k) for k in POWER_METRICS}


def compare(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    check_plan(out, args)
    results = study.run_jobs(jobs_for(args), args.workers)
    order = [study.job_id(make_job(v, s, args.steps, args.out)) for v in VARIANTS for s in args.seeds]
    done = {}
    if (out / "results.json").exists():
        done = {r["id"]: r for r in json.loads((out / "results.json").read_text(encoding="utf-8"))}
    done.update({r["id"]: r for r in results})
    all_results = [done[i] for i in order if i in done]
    study.atomic_json(out / "results.json", all_results)
    seed_rows = [dict(r3.seed_row(r), power_reward=r["rewards"]["power"], **power_row(r["eval"])) for r in all_results]
    rows = r3.summarize(seed_rows, VARIANTS)
    for row in rows:
        g = [r for r in seed_rows if r["variant"] == row["variant"]]
        row.update(power_reward=POWER[row["variant"]], **{k: r3.mean_or_none([r[k] for r in g]) for k in POWER_METRICS})
    study.atomic_json(out / "evaluation_dev.json", {"by_variant": rows, "by_seed": seed_rows})
    r3.write_table(out / "evaluation_dev.csv", rows)
    r3.write_table(out / "evaluation_dev_by_seed.csv", seed_rows)
    curve_rows = [dict(variant=r["label"], train_seed=r["seed"], **c) for r in all_results for c in r.get("curve", [])]
    if curve_rows:
        r3.write_table(out / "eval_curve.csv", curve_rows)
    if len(all_results) == len(order):
        study.atomic_json(out / "COMPLETE_dev.json", dict(jobs=len(all_results), dev_seed0=DEV_SEED0))
    print(json.dumps(rows, ensure_ascii=False, indent=1), flush=True)


def final(args):
    """4条件とも、学習seed 5本・未使用seedで評価する。1回ごとの結果をモデルのハッシュと保存する"""
    from dqn_agent import load_agent
    from pacman_env import PacManEnv
    out = Path(args.out).resolve()
    plan = check_plan(out, args)
    if not (out / "COMPLETE_dev.json").exists():
        raise ValueError("Run the dev comparison first")
    selection = dict(selected="all", baseline=BASELINE, rule="no selection (all variants evaluated)",
                     final_seed0=FINAL_SEED0, final_episodes=FINAL_EPISODES, final_train_seeds=plan["final_train_seeds"])
    study.atomic_json(out / "selection.json", selection)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    episodes, rows = [], []
    for r in results:
        run = out / "runs" / r["id"]
        env_options = r.get("env_options", {})
        env = PacManEnv(num_ghosts=r["num_ghosts"], lives=r["lives"], include_lives=r["include_lives"],
                        observation_version=r["observation_version"], **env_options)
        agent = load_agent(str(run / "model.pt"), env.observation_space.shape[0], 4)
        ev = study.evaluate(agent, r["rewards"], FINAL_EPISODES, r["lives"], seed0=FINAL_SEED0,
                            include_lives=r["include_lives"], num_ghosts=r["num_ghosts"],
                            observation_version=r["observation_version"], env_options=env_options)
        model_sha = r3.sha256(run / "model.pt")
        for i in range(FINAL_EPISODES):
            ps = ev["power_steps_each"][i]
            episodes.append(dict(variant=r["label"], train_seed=r["seed"], model_id=r["id"], model_sha256=model_sha,
                                 eval_seed=FINAL_SEED0 + i, cleared=int(ev["clears"][i]), score=ev["scores"][i],
                                 steps=ev["steps"][i], pellet_fraction=round(ev["pellet_fraction_each"][i], 6),
                                 dots_left=ev["dots_left_each"][i], lives_lost=ev["lives_lost_each"][i],
                                 ending=ev["ending_each"][i],
                                 **{k: ev[f"{k}_each"][i] for k in ("dot_points", "power_points", "ghost_points",
                                                                     "ghosts_eaten", "power_eaten")},
                                 power_steps=" ".join(map(str, ps)), power_first_step=ps[0] if ps else ""))
        row = dict(variant=r["label"], power_reward=r["rewards"]["power"], train_seed=r["seed"], clears=sum(ev["clears"]),
                   episodes=FINAL_EPISODES, **{k: ev[k] for k in r3.METRICS}, **{k: ev.get(k) for k in r3.SCORE_METRICS},
                   **power_row(ev),
                   caught_with_le5_dots=sum(1 for c, d, e in zip(ev["clears"], ev["dots_left_each"], ev["ending_each"])
                                            if not c and e == "caught" and d <= 5),
                   timeouts=ev["ending_each"].count("truncated"))
        rows.append(row)
        print(row, flush=True)
    by = {(r["variant"], r["train_seed"]): r for r in rows}
    diffs = [dict(variant=v, train_seed=s, clear_rate=by[(v, s)]["clear_rate"], baseline_clear_rate=by[(BASELINE, s)]["clear_rate"],
                  diff_points=round(100 * (by[(v, s)]["clear_rate"] - by[(BASELINE, s)]["clear_rate"]), 1),
                  score_diff=round(by[(v, s)]["score_mean"] - by[(BASELINE, s)]["score_mean"], 1))
             for v in VARIANTS if v != BASELINE for s in plan["final_train_seeds"]]
    summary = r3.summarize([dict(r, train_minutes=0) for r in rows], VARIANTS)
    for row in summary:
        g = [r for r in rows if r["variant"] == row["variant"]]
        row.update(power_reward=POWER[row["variant"]], **{k: r3.mean_or_none([r[k] for r in g]) for k in POWER_METRICS})
    study.atomic_json(out / "evaluation_final.json", dict(selection=selection, by_seed=rows, by_variant=summary,
                                                         diff_by_train_seed=diffs, sources=results[0]["manifest"]["sources"],
                                                         plan_sha256=r3.sha256(out / "plan.json")))
    r3.write_table(out / "evaluation_final.csv", rows)
    r3.write_table(out / "evaluation_final_by_variant.csv", summary)
    r3.write_table(out / "evaluation_final_episodes.csv", episodes)
    print(json.dumps(dict(by_variant=summary), ensure_ascii=False, indent=1), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--final", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve().is_relative_to(Path("/content/drive").resolve()):
        p.error("Train on local storage; copy results to Drive afterwards")
    final(a) if a.final else compare(a)


if __name__ == "__main__":
    main()
