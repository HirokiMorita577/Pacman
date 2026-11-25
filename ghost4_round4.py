"""
ゴースト4体・残機3のクリア率改善 (第4回比較, 2026-10-10 設計)。

第3回 (output/ghost4_round3_20261010) の結論: 終盤の失敗の構造はどの条件でも変わらず、終盤の約6割は
「最短距離で自由に動く敵」という見積もりでは安全に行けるエサが無いと判定されていた。そこで、敵の実際の動き方
(classic の狙い方・逆戻りしない規則・怖がった敵の逃げ方) で5手先まで予測する特徴量 (classic_lookahead) を
基準 (safe = 第2回の safe_obs と同じ設定) に1つだけ足して比べる。ゲーム条件 (敵4体・残機3・全エサでクリア・最大1,000ステップ) は変えない。

  safe            : 基準 (第3回の safe と同じ設定)
  safe_lookahead  : + classic_lookahead (行動ごとに、5手後に捕まらずにいられるマスの数・その間に安全に通れるエサ・
                    逃げ場がある最後の手。4行動 x 3)

第3回の反省 (開発用評価が1条件90回で、10ポイント程度の差を選び分けられなかった) から:
  ・学習seedを最初から5本 (101/202/303/404/505) にする
  ・開発用評価を学習seedごとに100回 (930000〜930099。930000〜930029 は既に結果を見た集合を含む) にする
  ・候補は1つなので選択はせず、2条件とも、未使用の評価seed 960000〜960099 で最終評価する (plan.json で先に固定)
学習: 各20万ステップ、探索の長さ 8万ステップ、2万ステップごとに貪欲評価 (第3回と同じ)。

  python ghost4_round4.py --out output/ghost4_round4 --workers 10
  python ghost4_round4.py --out output/ghost4_round4 --final
"""
import argparse
import json
from pathlib import Path

import numpy as np

import reward_study as study
import ghost4_round3 as r3

BASE = dict(r3.BASE, exp="ghost4_round4")
VARIANTS = {
    "safe": dict(job={}, steps=1),
    "safe_lookahead": dict(job={"env_options": {"classic_lookahead": True}}, steps=1),
}
BASELINE, CANDIDATE = "safe", "safe_lookahead"
SEEDS = [101, 202, 303, 404, 505]
CURVE = r3.CURVE
DEV_SEED0, DEV_EPISODES = 930000, 100
FINAL_SEED0, FINAL_EPISODES = 960000, 100
USED_EVAL_SETS = {**r3.USED_EVAL_SETS, "950000-950099": "第3回の最終評価 (結果を見た)"}


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
    return dict(jobs=[{**strip(j), "id": study.job_id(j)} for j in jobs], baseline=BASELINE, candidate=CANDIDATE,
                primary_metric="clear_rate", dev_seed0=DEV_SEED0, dev_episodes=DEV_EPISODES, curve=CURVE,
                final_seed0=FINAL_SEED0, final_episodes=FINAL_EPISODES, final_train_seeds=list(args.seeds),
                used_eval_sets=USED_EVAL_SETS,
                note="no selection step: both variants go to the final evaluation on unused seeds")


def check_plan(out, args):
    plan = plan_for(args)
    path = out / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing plan differs; use a separate output directory")
    study.atomic_json(path, plan)
    return plan


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
    seed_rows = [r3.seed_row(r) for r in all_results]
    rows = r3.summarize(seed_rows, VARIANTS)
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
    """2条件とも、学習seed 5本・未使用seedで評価する。1回ごとの結果をモデルのハッシュと保存する"""
    from dqn_agent import load_agent
    from pacman_env import PacManEnv
    out = Path(args.out).resolve()
    plan = check_plan(out, args)
    if not (out / "COMPLETE_dev.json").exists():
        raise ValueError("Run the dev comparison first")
    selection = dict(selected=CANDIDATE, baseline=BASELINE, rule="no selection (single candidate)",
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
            episodes.append(dict(variant=r["label"], train_seed=r["seed"], model_id=r["id"], model_sha256=model_sha,
                                 eval_seed=FINAL_SEED0 + i, cleared=int(ev["clears"][i]), score=ev["scores"][i],
                                 steps=ev["steps"][i], pellet_fraction=round(ev["pellet_fraction_each"][i], 6),
                                 dots_left=ev["dots_left_each"][i], lives_lost=ev["lives_lost_each"][i],
                                 ending=ev["ending_each"][i],
                                 **{k: ev[f"{k}_each"][i] for k in ("dot_points", "power_points", "ghost_points",
                                                                     "ghosts_eaten", "power_eaten")}))
        row = dict(variant=r["label"], train_seed=r["seed"], clears=sum(ev["clears"]), episodes=FINAL_EPISODES,
                   **{k: ev[k] for k in r3.METRICS}, **{k: ev.get(k) for k in r3.SCORE_METRICS},
                   caught_with_le5_dots=sum(1 for c, d, e in zip(ev["clears"], ev["dots_left_each"], ev["ending_each"])
                                            if not c and e == "caught" and d <= 5),
                   timeouts=ev["ending_each"].count("truncated"))
        rows.append(row)
        print(row, flush=True)
    by = {(r["variant"], r["train_seed"]): r for r in rows}
    diffs = [dict(train_seed=s, selected_clear_rate=by[(CANDIDATE, s)]["clear_rate"],
                  baseline_clear_rate=by[(BASELINE, s)]["clear_rate"],
                  diff_points=round(100 * (by[(CANDIDATE, s)]["clear_rate"] - by[(BASELINE, s)]["clear_rate"]), 1),
                  score_diff=round(by[(CANDIDATE, s)]["score_mean"] - by[(BASELINE, s)]["score_mean"], 1))
             for s in plan["final_train_seeds"]]
    summary = r3.summarize([dict(r, train_minutes=0) for r in rows], VARIANTS)
    study.atomic_json(out / "evaluation_final.json", dict(selection=selection, by_seed=rows, by_variant=summary,
                                                         diff_by_train_seed=diffs, sources=results[0]["manifest"]["sources"],
                                                         plan_sha256=r3.sha256(out / "plan.json")))
    r3.write_table(out / "evaluation_final.csv", rows)
    r3.write_table(out / "evaluation_final_episodes.csv", episodes)
    print(json.dumps(dict(by_variant=summary, diff_by_train_seed=diffs), ensure_ascii=False, indent=1), flush=True)


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
