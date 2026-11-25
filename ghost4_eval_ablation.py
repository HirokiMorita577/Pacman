"""
学習済みの tactical_safe モデルが、先読み特徴量のどの群に頼っているかを調べる (評価時だけ特徴群を0にする)。学習はしない。

  python ghost4_eval_ablation.py --study output/ghost4_round2_20261007/study --label safe_obs \\
      --seed0 940000 --episodes 100 --out output/ghost4_round2_20261007/eval_ablation

・学習時から外して学習し直す比較 (ghost4_round3 の no_area など) とは別物。こちらは「学習済み方策の依存度の検査」で、
  消した入力に方策が適応する機会はない。数値が下がっても「その特徴が無ければ学べない」ことの証明にはならない
・条件: full (そのまま) / no_area / no_dots / no_margin / none (3群すべて0)
・評価seedは既に結果を見た集合を使う (分析用。最終評価には使わない)
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pacman_env import SAFETY_GROUPS

CONDITIONS = {"full": list(SAFETY_GROUPS), "none": []}
CONDITIONS.update({f"no_{g}": [x for x in SAFETY_GROUPS if x != g] for g in SAFETY_GROUPS})


def run(args):
    study_dir, r, cond, seed0, n = args
    import torch
    import reward_study as study
    from dqn_agent import load_agent
    from pacman_env import PacManEnv
    torch.set_num_threads(1)
    options = {**r.get("env_options", {}), "safety_groups": CONDITIONS[cond]}
    env = PacManEnv(num_ghosts=r["num_ghosts"], lives=r["lives"], include_lives=r["include_lives"],
                    observation_version=r["observation_version"], **options)
    agent = load_agent(str(Path(study_dir) / "runs" / r["id"] / "model.pt"), env.observation_space.shape[0], 4)
    ev = study.evaluate(agent, r["rewards"], n, r["lives"], seed0=seed0, include_lives=r["include_lives"],
                        num_ghosts=r["num_ghosts"], observation_version=r["observation_version"], env_options=options)
    fails = [i for i, c in enumerate(ev["clears"]) if not c]
    return dict(condition=cond, train_seed=r["seed"], clears=sum(ev["clears"]), episodes=n,
                clear_rate=ev["clear_rate"], pellet_fraction_mean=round(ev["pellet_fraction_mean"], 4),
                lives_lost_mean=ev["lives_lost_mean"], score_mean=ev["score_mean"],
                failed_with_le5_dots=sum(ev["dots_left_each"][i] <= 5 for i in fails),
                timeouts=ev["ending_each"].count("truncated"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--label", default="safe_obs")
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [r for r in json.loads((Path(a.study) / "results.json").read_text(encoding="utf-8"))
               if r["label"] == a.label and r["observation_version"] == "tactical_safe"]
    jobs = [(a.study, r, c, a.seed0, a.episodes) for c in CONDITIONS for r in results]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(run, jobs))
    with (out / "eval_ablation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for c in CONDITIONS:
        g = [x for x in rows if x["condition"] == c]
        summary[c] = dict(clears=sum(x["clears"] for x in g), episodes=sum(x["episodes"] for x in g),
                          clears_by_seed="/".join(str(x["clears"]) for x in g),
                          pellet_fraction_mean=round(sum(x["pellet_fraction_mean"] for x in g) / len(g), 4),
                          failed_with_le5_dots=sum(x["failed_with_le5_dots"] for x in g),
                          timeouts=sum(x["timeouts"] for x in g))
    (out / "summary.json").write_text(json.dumps(dict(label=a.label, seed0=a.seed0, episodes=a.episodes,
                                                      note="evaluation-time ablation (dependence check), not retraining",
                                                      by_condition=summary), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
