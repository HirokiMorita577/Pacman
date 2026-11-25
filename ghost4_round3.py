"""
ゴースト4体・残機3のクリア率改善 (第3回比較, 2026-10-08 設計)。

第2回 (output/ghost4_round2_20261007) で選んだ safe_obs を基準 (safe) にし、レビュー (review/改善点レビュー.md) の
優先3・4 を、要因が混ざらないように比べる。ゲーム条件 (敵4体・残機3・全エサでクリア・最大1,000ステップ) は変えない。

  要因の比較 (leave_fix × 学習量。探索の長さは explore_steps=80,000 で全条件同じ = 第2回の20万ステップと同じ)
    safe             : 第2回の safe_obs と同じ設定 (基準)                                  … A
    safe_leave       : + fix_leave_dot                                                    … B
    safe_400k        : 学習 40万ステップ (探索の長さは同じ)                                … C
    safe_leave_400k  : + fix_leave_dot、40万ステップ                                       … D
  先読みの改良
    safe_timing      : 怖がっている敵も恐怖の残り時間を考えて危険として数える (safety_timing)
    safe_route       : 安全なマスだけで行ける最寄りエサの近さと、最寄りエサへの経路上の余裕を追加 (safe_route)
                       … 第2回の失敗分類で、最後の1機を失う場面の63%が残りエサ10個以下・3ステップ前には既に逃げ場が
                         狭かった (安全な領域<30マスが97%) ため。終盤に「どのエサなら安全に取れるか」を入力で渡す
  特徴群の寄与 (学習時から外す。入力の次元は同じで、外した群は 0)
    no_area / no_dots / no_margin

学習: seed 101/202/303。学習中に 2万ステップごとに開発用seed 930000〜930009 で貪欲評価する (eval_curve.csv)。
  C と A は最初の20万ステップが同じ計算になるので、学習曲線の前半が一致することも確かめる (curve_check)。
  ただし C は A の続きから再開するのではなく、最初から学習し直す (重みだけの追加学習と完全な再開を混同しないため)。
開発用の評価: seed 930000〜930029 (第1・2回でも使った集合。既に結果を見ているので比較・選択だけに使う)。
  なお 940000〜940099 は第2回の最終評価で結果を見たので、この回では使わない。
最終評価: 未使用の seed 950000〜950099 を plan.json で先に固定する。開発用評価で基準 (safe) との比較候補から1条件を選び
  (selection.json)、選んだ条件と safe を、学習seedを 101/202/303/404/505 の5本に増やして評価する (--final)。
  1回ごとの結果をモデルのハッシュと一緒に保存する (evaluation_final_episodes.csv)。

  python ghost4_round3.py --out output/ghost4_round3 --workers 6 --variants safe safe_leave safe_timing safe_route no_area no_dots no_margin
  python ghost4_round3.py --out output/ghost4_round3 --workers 6 --variants safe_400k safe_leave_400k
  python ghost4_round3.py --out output/ghost4_round3 --final
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import reward_study as study

BASE = dict(exp="ghost4_round3", num_ghosts=4, lives=3, include_lives=True,
            observation_version="tactical_safe", mask_illegal_actions=True,
            rewards=study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0), gamma=0.95,
            explore_steps=80000)
# 条件ごとの変更 (env_options は env にそのまま渡す) と学習量の倍率
VARIANTS = {
    "safe": dict(job={}, steps=1),
    "safe_leave": dict(job={"env_options": {"fix_leave_dot": True}}, steps=1),
    "safe_400k": dict(job={}, steps=2),
    "safe_leave_400k": dict(job={"env_options": {"fix_leave_dot": True}}, steps=2),
    "safe_timing": dict(job={"env_options": {"safety_timing": True}}, steps=1),
    "safe_route": dict(job={"env_options": {"safe_route": True}}, steps=1),
    "no_area": dict(job={"env_options": {"safety_groups": ["dots", "margin"]}}, steps=1),
    "no_dots": dict(job={"env_options": {"safety_groups": ["area", "margin"]}}, steps=1),
    "no_margin": dict(job={"env_options": {"safety_groups": ["area", "dots"]}}, steps=1),
}
BASELINE = "safe"
CANDIDATES = ["safe_leave", "safe_400k", "safe_leave_400k", "safe_timing", "safe_route"]     # 最終評価に進める候補 (no_* は分析用)
CURVE = {"every": 20000, "episodes": 10, "seed0": 930000}
DEV_SEED0, DEV_EPISODES = 930000, 30
FINAL_SEED0, FINAL_EPISODES = 950000, 100
FINAL_TRAIN_SEEDS = [101, 202, 303, 404, 505]
USED_EVAL_SETS = {"930000-930029": "第1〜3回の開発用 (結果を見た)", "940000-940099": "第2回の最終評価 (結果を見た)"}
METRICS = ["clear_rate", "pellet_fraction_mean", "lives_lost_mean", "score_mean", "steps_mean", "blocked_action_rate"]
# 得点の内訳 (主指標はクリア率。得点はクリア率を保てる候補どうしの比較と、理由の説明に使う。クリア試行だけの平均は
# 成功した試行の構成が変わるので単独で優劣を決めない)
SCORE_METRICS = ["dot_points_mean", "power_points_mean", "ghost_points_mean", "ghosts_eaten_mean", "power_eaten_mean",
                 "score_mean_cleared", "lives_lost_mean_cleared"]


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def make_job(variant, seed, steps, out):
    v = VARIANTS[variant]
    budget = steps * v["steps"]
    return dict(BASE, **v["job"], label=variant, seed=seed, episodes=budget, step_budget=budget,
                eval_episodes=DEV_EPISODES, eval_seed0=DEV_SEED0, curve_eval=CURVE, checkpoint_every=50,
                compress_checkpoint=True, save_model=True, show_progress=True, runs_dir=str(Path(out) / "runs"))


def jobs_for(args):
    return [make_job(v, seed, args.steps, args.out) for v in args.variants for seed in args.seeds]


def plan_for(args):
    """全条件・全seedの計画 (一部の条件だけ先に回しても同じ計画になる)"""
    all_jobs = [make_job(v, s, args.steps, args.out) for v in VARIANTS for s in args.seeds]
    extra = [make_job(v, s, args.steps, args.out) for v in VARIANTS for s in FINAL_TRAIN_SEEDS if s not in args.seeds]
    strip = lambda j: {k: v for k, v in j.items() if k != "runs_dir"}
    return dict(jobs=[{**strip(j), "id": study.job_id(j)} for j in all_jobs],
                final_extra_jobs=[{**strip(j), "id": study.job_id(j)} for j in extra],
                baseline=BASELINE, candidates=CANDIDATES, primary_metric="clear_rate",
                selection_rule="dev clear_rate, then pellet_fraction_mean; candidates only",
                dev_seed0=DEV_SEED0, dev_episodes=DEV_EPISODES, curve=CURVE,
                final_seed0=FINAL_SEED0, final_episodes=FINAL_EPISODES, final_train_seeds=FINAL_TRAIN_SEEDS,
                used_eval_sets=USED_EVAL_SETS)


def check_plan(out, args):
    plan = plan_for(args)
    path = out / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing plan differs; use a separate output directory")
    study.atomic_json(path, plan)
    return plan


def seed_row(r):
    ev = r["eval"]
    curve = r.get("curve", [])
    return dict(variant=r["label"], train_seed=r["seed"], clears=sum(ev["clears"]), episodes=len(ev["clears"]),
                train_minutes=r["train_minutes"], total_steps=r["total_steps"], episodes_trained=r["episodes_completed"],
                **{k: ev[k] for k in METRICS}, **{k: ev.get(k) for k in SCORE_METRICS},
                caught_with_le5_dots=sum(1 for c, d, e in zip(ev["clears"], ev["dots_left_each"], ev["ending_each"])
                                         if not c and e == "caught" and d <= 5),
                timeouts=ev["ending_each"].count("truncated"),
                curve_last_clear_rate=curve[-1]["clear_rate"] if curve else None)


def write_table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(seed_rows, variants):
    rows = []
    for v in variants:
        g = [r for r in seed_rows if r["variant"] == v]
        if not g:
            continue
        rows.append(dict(variant=v, seeds=len(g), clears=sum(r["clears"] for r in g),
                         episodes=sum(r["episodes"] for r in g), clears_by_seed="/".join(str(r["clears"]) for r in g),
                         **{k: float(np.mean([r[k] for r in g])) for k in METRICS},
                         **{k: mean_or_none([r[k] for r in g]) for k in SCORE_METRICS},
                         caught_with_le5_dots=sum(r["caught_with_le5_dots"] for r in g),
                         timeouts=sum(r["timeouts"] for r in g),
                         train_minutes_mean=float(np.mean([r["train_minutes"] for r in g]))))
    return rows


def curve_check(results):
    """safe_400k と safe の学習曲線が、最初の20万ステップで一致するか (同じ計算のはず)"""
    by = {(r["label"], r["seed"]): r.get("curve", []) for r in results}
    out = {}
    for long_, short in (("safe_400k", "safe"), ("safe_leave_400k", "safe_leave")):
        for s in sorted({k[1] for k in by}):
            a, b = by.get((long_, s)), by.get((short, s))
            if a and b:
                # 短い方の最後の行は予算で途中打ち切りした時点なので比べない
                n = len(b) - 1
                out[f"{long_}_vs_{short}_s{s}"] = a[:n] == b[:n]
    return out


def compare(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    check_plan(out, args)
    results = study.run_jobs(jobs_for(args), args.workers)
    # これまでに終わった条件をすべて集める (段階的に回した分も含む)
    done = {}
    if (out / "results.json").exists():
        done = {r["id"]: r for r in json.loads((out / "results.json").read_text(encoding="utf-8"))}
    done.update({r["id"]: r for r in results})
    order = [study.job_id(make_job(v, s, args.steps, args.out)) for v in VARIANTS for s in args.seeds]
    all_results = [done[i] for i in order if i in done]
    study.atomic_json(out / "results.json", all_results)
    seed_rows = [seed_row(r) for r in all_results]
    rows = summarize(seed_rows, VARIANTS)
    study.atomic_json(out / "evaluation_dev.json", {"by_variant": rows, "by_seed": seed_rows,
                                                    "curve_check": curve_check(all_results)})
    write_table(out / "evaluation_dev.csv", rows)
    write_table(out / "evaluation_dev_by_seed.csv", seed_rows)
    curve_rows = [dict(variant=r["label"], train_seed=r["seed"], **c) for r in all_results for c in r.get("curve", [])]
    if curve_rows:
        write_table(out / "eval_curve.csv", curve_rows)
    if len(all_results) == len(order):
        study.atomic_json(out / "COMPLETE_dev.json", dict(jobs=len(all_results), dev_seed0=DEV_SEED0))
    print(json.dumps(rows, ensure_ascii=False, indent=1), flush=True)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def final(args):
    """開発用の結果で候補から1条件を選び (selection.json を先に保存)、選んだ条件と基準を学習seed 5本・未使用seedで評価する"""
    out = Path(args.out).resolve()
    plan = check_plan(out, args)
    if not (out / "COMPLETE_dev.json").exists():
        raise ValueError("Run all dev comparisons first")
    dev = json.loads((out / "evaluation_dev.json").read_text(encoding="utf-8"))["by_variant"]
    sel_path = out / "selection.json"
    if sel_path.exists():
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
    else:
        base = next(r for r in dev if r["variant"] == BASELINE)
        best = max((r for r in dev if r["variant"] in CANDIDATES),
                   key=lambda r: (r["clear_rate"], r["pellet_fraction_mean"]))
        better = (best["clear_rate"], best["pellet_fraction_mean"]) > (base["clear_rate"], base["pellet_fraction_mean"])
        selection = dict(selected=best["variant"] if better else BASELINE, baseline=BASELINE,
                         best_candidate=best["variant"], rule=plan["selection_rule"],
                         final_seed0=FINAL_SEED0, final_episodes=FINAL_EPISODES, final_train_seeds=FINAL_TRAIN_SEEDS)
        study.atomic_json(sel_path, selection)
    variants = list(dict.fromkeys([selection["selected"], BASELINE]))
    jobs = [make_job(v, s, args.steps, args.out) for v in variants for s in FINAL_TRAIN_SEEDS]
    results = study.run_jobs(jobs, args.workers)       # 足りない学習seedだけ学習する (開発用評価も付く)
    study.atomic_json(out / "results_final.json", results)
    episodes, rows = [], []
    for r in results:
        run = out / "runs" / r["id"]
        env_options = r.get("env_options", {})
        from dqn_agent import load_agent
        from pacman_env import PacManEnv
        env = PacManEnv(num_ghosts=r["num_ghosts"], lives=r["lives"], include_lives=r["include_lives"],
                        observation_version=r["observation_version"], **env_options)
        agent = load_agent(str(run / "model.pt"), env.observation_space.shape[0], 4)
        ev = study.evaluate(agent, r["rewards"], FINAL_EPISODES, r["lives"], seed0=FINAL_SEED0,
                            include_lives=r["include_lives"], num_ghosts=r["num_ghosts"],
                            observation_version=r["observation_version"], env_options=env_options)
        model_sha = sha256(run / "model.pt")
        for i in range(FINAL_EPISODES):
            episodes.append(dict(variant=r["label"], train_seed=r["seed"], model_id=r["id"], model_sha256=model_sha,
                                 eval_seed=FINAL_SEED0 + i, cleared=int(ev["clears"][i]), score=ev["scores"][i],
                                 steps=ev["steps"][i], pellet_fraction=round(ev["pellet_fraction_each"][i], 6),
                                 dots_left=ev["dots_left_each"][i], lives_lost=ev["lives_lost_each"][i],
                                 ending=ev["ending_each"][i],
                                 **{k: ev[f"{k}_each"][i] for k in ("dot_points", "power_points", "ghost_points",
                                                                     "ghosts_eaten", "power_eaten")}))
        row = dict(variant=r["label"], train_seed=r["seed"], clears=sum(ev["clears"]), episodes=FINAL_EPISODES,
                   **{k: ev[k] for k in METRICS}, **{k: ev.get(k) for k in SCORE_METRICS},
                   caught_with_le5_dots=sum(1 for c, d, e in zip(ev["clears"], ev["dots_left_each"], ev["ending_each"])
                                            if not c and e == "caught" and d <= 5),
                   timeouts=ev["ending_each"].count("truncated"),
                   lives_left_on_clear_mean=float(np.mean([r["lives"] - l for c, l in zip(ev["clears"], ev["lives_lost_each"]) if c]))
                   if any(ev["clears"]) else None)
        rows.append(row)
        print(row, flush=True)
    # 学習seedごとの差 (選んだ条件 − 基準)。プールしたp値ではなく、まず学習seed別の改善幅を示す
    diffs = []
    if selection["selected"] != BASELINE:
        by = {(r["variant"], r["train_seed"]): r for r in rows}
        for s in FINAL_TRAIN_SEEDS:
            a, b = by[(selection["selected"], s)], by[(BASELINE, s)]
            diffs.append(dict(train_seed=s, selected_clear_rate=a["clear_rate"], baseline_clear_rate=b["clear_rate"],
                              diff_points=round(100 * (a["clear_rate"] - b["clear_rate"]), 1),
                              score_diff=round(a["score_mean"] - b["score_mean"], 1),
                              ghosts_eaten_diff=round(a["ghosts_eaten_mean"] - b["ghosts_eaten_mean"], 3)))
    sources = results[0]["manifest"]["sources"]
    study.atomic_json(out / "evaluation_final.json", dict(selection=selection, by_seed=rows, diff_by_train_seed=diffs,
                                                         sources=sources, plan_sha256=sha256(out / "plan.json")))
    write_table(out / "evaluation_final.csv", rows)
    write_table(out / "evaluation_final_episodes.csv", episodes)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200000, help="基準の学習量 (400k の条件はこの2倍)")
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
