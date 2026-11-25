"""
捕獲が「避けられたか」を調べる (反実仮想)。学習はしない。

  python ghost4_counterfactual.py --study output/ghost4_retrain_20261006/study --out output/ghost4_diagnosis_20261007 --label improved

評価を保存時と同じ手順で再生し、各捕獲の k ステップ前 (k = 1, 3, 5) の環境を複製 (ゲームの乱数の状態ごと) して、
そこで選べた各行動を取ったあと同じ方策 (貪欲) で H ステップ進め、その間に残機を失うかを調べる。
・実際の行動でも捕まらない場合は「方策の続きが違えば避けられた」ではなく、再生の確認として数える。
・どの行動でも H ステップ以内に捕まる → その時点ではこの方策では避けられない (挟み撃ちなど)
・別の行動なら捕まらない → 判断の誤り。その行動が各頭 (エサ / ゴースト / 移動) のどれの最良行動だったかも記録する
評価そのものの乱数や結果は変えない (複製した環境だけを進める)。
"""
import argparse
import copy
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from pacman_env import PacManEnv, REWARD_COMPONENT_NAMES
from dqn_agent import load_agent
import reward_study as study

HEADS = list(REWARD_COMPONENT_NAMES)


def legal_actions(agent, obs):
    if agent.action_mask_indices is None:
        return list(range(4))
    allowed = [a for a in range(4) if obs[agent.action_mask_indices[a]] > 0.5]
    return allowed or list(range(4))


def survives(env, agent, obs, first_action, horizon):
    """複製した環境で first_action を取り、その後は貪欲方策で horizon ステップ。残機を失わなければ True"""
    e = copy.deepcopy(env)
    lives = e.lives
    a = first_action
    for _ in range(horizon):
        obs, _, te, tr, info = e.step(a)
        if info["lives"] < lives:
            return False
        if te or tr:
            return True
        a = agent.select_action(obs, training=False)
    return True


def head_best(agent, obs, legal):
    with torch.no_grad():
        hq = agent.policy_net.head_q(torch.FloatTensor(obs).unsqueeze(0))[0].numpy()
    return {h: legal[int(np.argmax(hq[i][legal]))] for i, h in enumerate(HEADS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="improved")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    a = ap.parse_args()
    results = [r for r in json.loads((Path(a.study) / "results.json").read_text(encoding="utf-8")) if r["label"] == a.label]
    rows = []
    for r in results:
        env = PacManEnv(render_mode=None, ghost_mode=study.FIXED["ghost_mode"], num_ghosts=r["num_ghosts"],
                        rewards=r["rewards"], lives=r["lives"], include_lives=r["include_lives"],
                        max_steps=study.FIXED["max_steps"], observation_version=r["observation_version"],
                        **r.get("env_options", {}))
        agent = load_agent(str(Path(a.study) / "runs" / r["id"] / "model.pt"), env.observation_space.shape[0], 4)
        for i in range(len(r["eval"]["scores"])):
            seed = r["eval"]["seed0"] + i
            obs, _ = env.reset(seed=seed)
            history = deque(maxlen=max(a.ks) + 1)      # (複製した環境, 観測, 実際の行動)
            while True:
                action = agent.select_action(obs, training=False)
                history.append((copy.deepcopy(env), obs, action))
                lives = env.lives
                obs, _, te, tr, info = env.step(action)
                if info["lives"] < lives:
                    row = {"label": a.label, "train_seed": r["seed"], "eval_seed": seed, "t": env.steps,
                           "lives_before": lives, "dots_left": env.total_dots - env.dots_eaten}
                    for k in a.ks:
                        if len(history) < k:
                            continue
                        snap, o, act = history[-k]
                        legal = legal_actions(agent, o)
                        ok = {b: survives(snap, agent, o, b, a.horizon) for b in legal}
                        best = head_best(agent, o, legal)
                        row[f"k{k}_chosen_survives"] = ok[act]
                        row[f"k{k}_alternatives_survive"] = [b for b in legal if b != act and ok[b]]
                        row[f"k{k}_n_legal"] = len(legal)
                        row[f"k{k}_head_best"] = best
                        row[f"k{k}_ghost_best_survives"] = ok[best["ghost"]]
                    rows.append(row)
                if te or tr:
                    break
        print(f"{a.label} s{r['seed']}: 捕獲 {sum(x['train_seed'] == r['seed'] for x in rows)} 件", flush=True)
    summary = {}
    for k in a.ks:
        R = [x for x in rows if f"k{k}_n_legal" in x]
        real = [x for x in R if not x[f"k{k}_chosen_survives"]]     # 実際の行動で捕まった (再生の確認)
        avoidable = [x for x in real if x[f"k{k}_alternatives_survive"]]
        summary[f"k{k}"] = {
            "deaths": len(R), "chosen_action_also_caught_in_rollout": len(real),
            "avoidable_by_other_first_action": len(avoidable),
            "unavoidable_for_this_policy": len(real) - len(avoidable),
            "ghost_head_best_would_survive": sum(x[f"k{k}_ghost_best_survives"] for x in real),
            "avoidable_by_life": dict(Counter(f"life{x['lives_before']}" for x in avoidable)),
        }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"counterfactual_{a.label}.json").write_text(
        json.dumps({"horizon": a.horizon, "summary": summary, "deaths": rows}, ensure_ascii=False, indent=1,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
