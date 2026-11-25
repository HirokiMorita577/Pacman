"""
報酬設計の実験で「クリアできない原因」を調べる: エピソードがどう終わったかを条件ごとに集計する。
  python analyze_endings.py --out /content/drive/MyDrive/DQN_reward_study
終わり方: クリア / 捕獲 (ゴーストにやられた) / 打ち切り (1,000 ステップの上限)
学習中 (直近100エピソード) と評価 (学習後の貪欲方策) の両方を見る。結果は figures/endings.{md,csv,png} にも保存。
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

MAX_STEPS = 1000


def endings_from_log(path, n=100):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-n:]
    clear = np.array([int(r["cleared"]) for r in rows])
    death = np.array([float(r["death"]) < 0 for r in rows])
    steps = np.array([int(r["steps"]) for r in rows])
    trunc = (~death) & (clear == 0) & (steps >= MAX_STEPS)
    dots = np.array([float(r["score"]) for r in rows])
    return {"clear": clear.mean(), "death": death.mean(), "trunc": trunc.mean(), "steps": steps.mean(),
            "steps_death": steps[death].mean() if death.any() else float("nan"), "score": dots.mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    groups = {}
    for rp in glob.glob(os.path.join(a.out, "runs", "*", "result.json")):
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        if r["exp"] == "exp4_pso":
            continue
        tr = endings_from_log(os.path.join(os.path.dirname(rp), "log.csv"))
        ev = r["eval"]
        death_r = r["rewards"]["death"]
        ev_death = -ev["reward_items_mean"].get("death", 0.0) / death_r if death_r else float("nan")
        ev_trunc = max(0.0, 1 - ev["clear_rate"] - ev_death)   # 残りは上限で打ち切り
        g = groups.setdefault((r["exp"], r["label"]), [])
        g.append({"tr": tr, "ev_clear": ev["clear_rate"], "ev_death": ev_death, "ev_trunc": ev_trunc,
                  "ev_steps": ev["steps_mean"], "ev_score": ev["score_mean"]})
    header = ["実験", "条件", "学習: 捕獲で終了", "学習: 打ち切り", "学習: 平均ステップ", "学習: 捕獲までのステップ",
              "評価: 捕獲で終了", "評価: 打ち切り", "評価: 平均ステップ", "評価: クリア"]
    rows = []
    for (exp, label), g in sorted(groups.items()):
        m = lambda f: float(np.nanmean([f(x) for x in g]))
        rows.append([exp, label, f"{m(lambda x: x['tr']['death']):.0%}", f"{m(lambda x: x['tr']['trunc']):.0%}",
                     f"{m(lambda x: x['tr']['steps']):.0f}", f"{m(lambda x: x['tr']['steps_death']):.0f}",
                     f"{m(lambda x: x['ev_death']):.0%}", f"{m(lambda x: x['ev_trunc']):.0%}",
                     f"{m(lambda x: x['ev_steps']):.0f}", f"{m(lambda x: x['ev_clear']):.0%}"])
    d = os.path.join(a.out, "figures")
    with open(os.path.join(d, "endings.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([header] + rows)
    with open(os.path.join(d, "endings.md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + " --- |" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    try:
        from reward_study import _plt, table_png
        table_png(_plt(), os.path.join(d, "endings.png"), header, rows, "エピソードの終わり方 (クリア / 捕獲 / 1,000ステップで打ち切り)")
    except Exception as e:
        print("[PNG スキップ]", e)
    for r in rows:
        print(" | ".join(r))


if __name__ == "__main__":
    main()
