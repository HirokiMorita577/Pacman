"""
ゴースト4体の保存済み評価を再現し、1ステップごとの記録 (トレース) と失敗の分析を出力する。学習はしない。

  python ghost4_diagnose.py --study output/ghost4_retrain_20261006/study --out output/ghost4_diagnosis_20261007

・各モデルを、保存時と同じ設定・同じ評価seed (930000〜930029) で貪欲方策のままプレイし、
  得点・成否・ステップ数が results.json と一致するかを確かめる (reproduction.json)。
・記録はゲームの乱数を使わず状態も変えない (読み取りと、捕獲したゴーストの記録だけ)。
・出力: traces/*.json.gz (全ステップ), deaths.csv (捕獲ごとの特徴), summary.json / summary.md, heatmaps.png
"""
import argparse
import csv
import gzip
import json
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from pacman_env import PacManEnv, DIRS, CELL_DOT, CELL_POWER, CELL_WALL, POWER_DURATION, REWARD_COMPONENT_NAMES
from dqn_agent import load_agent
import reward_study as study


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def save_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=_jsonable)
    os.replace(tmp, path)

HEADS = list(REWARD_COMPONENT_NAMES)          # dot / ghost / move
LATE_FRACTION = 0.90                           # 「終盤」= エサ回収率90%以上


def bfs_field(env, sources):
    """迷路上の距離 (到達不能は -1)。環境の内部状態は変えない"""
    return env._distance_field(sources)


def first_step_dir(env, start, goal_field):
    """start から goal までの最短経路の最初の一歩 (0..3)。goal_field は goal からの距離"""
    best, best_d = None, None
    for a, (dr, dc) in enumerate(DIRS):
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols and env.grid[nr][nc] != CELL_WALL:
            d = goal_field[nr][nc]
            if d >= 0 and (best_d is None or d < best_d):
                best, best_d = a, d
    return best


def snapshot(env):
    return {"pac": list(env.pacman), "dir": env.pacman_dir, "lives": env.lives, "power": env.power_timer,
            "dots_left": env.total_dots - env.dots_eaten,
            "ghosts": [[g.pos[0], g.pos[1], int(g.scared), g.scared_timer,
                        None if g.last_pos is None else [g.last_pos[0], g.last_pos[1]]] for g in env.ghosts]}


def danger_context(env):
    """危険なゴーストまでの迷路距離と、近い敵が来る方向の数 (挟み撃ちの目安)、逃げ道の数"""
    pac = tuple(env.pacman)
    open_dirs = [a for a, (dr, dc) in enumerate(DIRS)
                 if 0 <= pac[0] + dr < env.rows and 0 <= pac[1] + dc < env.cols
                 and env.grid[pac[0] + dr][pac[1] + dc] != CELL_WALL]
    dists, dirs = [], set()
    for g in env.ghosts:
        if g.scared:
            continue
        f = bfs_field(env, [tuple(g.pos)])
        d = f[pac[0]][pac[1]]
        if d < 0:
            continue
        dists.append(d)
        if d <= 6:
            a = first_step_dir(env, pac, f)
            if a is not None:
                dirs.add(a)
    return {"danger_min": min(dists) if dists else 99, "danger_dirs_within6": len(dirs),
            "exits": len(open_dirs), "danger_within6": sum(d <= 6 for d in dists)}


def trace_episode(env, agent, seed, record_steps=True):
    killers = []
    original = env._handle_ghost_collision

    def spy(g, rc, terminated):
        was_scared = g.scared
        out = original(g, rc, terminated)
        if not was_scared and not env.no_ghosts and not terminated:
            killers.append(g.ghost_id)
        return out
    env._handle_ghost_collision = spy
    try:
        obs, _ = env.reset(seed=seed)
        steps, prev_items, prev_score = [], {}, 0
        last_power_end, last_respawn, last_eat_ghost = None, 0, None
        deaths = []
        recent_ctx = deque(maxlen=4)
        while True:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                head_q = agent.policy_net.head_q(state_t)[0].numpy()     # [heads, actions]
            legal = list(range(4))
            if agent.action_mask_indices is not None:
                allowed = [a for a in range(4) if obs[agent.action_mask_indices[a]] > 0.5]
                legal = allowed or legal
            total = head_q.sum(axis=0)
            action = agent.select_action(obs, training=False)
            before = snapshot(env)
            ctx = danger_context(env)
            recent_ctx.append(ctx)
            power_before = env.power_timer
            n_kill = len(killers)
            obs, _, te, tr, info = env.step(action)
            items = {k: round(v - prev_items.get(k, 0.0), 3) for k, v in info["reward_items"].items()
                     if abs(v - prev_items.get(k, 0.0)) > 1e-9}
            prev_items = dict(info["reward_items"])
            t = env.steps
            if power_before > 0 and env.power_timer == 0:
                last_power_end = t
            if "eat_ghost" in items:
                last_eat_ghost = t
            row = {"t": t, **before, **ctx, "legal": legal, "action": action,
                   "head_q": np.round(head_q, 4).tolist(), "items": items,
                   "score_delta": info["score"] - prev_score}
            prev_score = info["score"]
            if len(killers) > n_kill:
                deaths.append({"t": t, "killer": killers[-1], "lives_before": before["lives"],
                               "dots_left": before["dots_left"], "pos": before["pac"],
                               "since_power_end": None if last_power_end is None else t - last_power_end,
                               "power_active_before": power_before > 0,
                               "since_respawn": t - last_respawn,
                               "since_eat_ghost": None if last_eat_ghost is None else t - last_eat_ghost,
                               "danger_dirs_at": ctx["danger_dirs_within6"], "exits_at": ctx["exits"],
                               "danger_dirs_3before": recent_ctx[0]["danger_dirs_within6"],
                               "danger_min_3before": recent_ctx[0]["danger_min"],
                               "exits_3before": recent_ctx[0]["exits"]})
                last_respawn = t
            if record_steps:
                steps.append(row)
            if te or tr:
                break
        remaining = [[r, c] for r in range(env.rows) for c in range(env.cols)
                     if env.grid[r][c] in (CELL_DOT, CELL_POWER)]
        return {"seed": seed, "score": info["score"], "cleared": bool(info["cleared"]), "steps": env.steps,
                "ending": "clear" if info["cleared"] else ("truncated" if tr and not te else "caught"),
                "dots_eaten": info["dots_eaten"], "total_dots": env.total_dots, "lives_left": info["lives"],
                "deaths": deaths, "remaining": remaining, "trace": steps}
    finally:
        env._handle_ghost_collision = original


def head_attribution(trace, last_k=5):
    """捕獲直前の数ステップで、選んだ行動が各頭の最良行動と一致したか"""
    out = []
    for row in trace[-last_k:]:
        legal = row["legal"]
        hq = np.array(row["head_q"])
        best = {h: legal[int(np.argmax(hq[i][legal]))] for i, h in enumerate(HEADS)}
        spread = {h: float(np.ptp(hq[i][legal])) if len(legal) > 1 else 0.0 for i, h in enumerate(HEADS)}
        out.append({"t": row["t"], "action": row["action"], "best": best, "spread": spread,
                    "danger_min": row["danger_min"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=None, help="省略時は保存結果と同じ回数")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    results = json.loads((Path(a.study) / "results.json").read_text(encoding="utf-8"))
    repro, death_rows, episodes = [], [], []
    for r in results:
        run_dir = Path(a.study) / "runs" / r["id"]
        env = PacManEnv(render_mode=None, ghost_mode=study.FIXED["ghost_mode"], num_ghosts=r["num_ghosts"],
                        rewards=r["rewards"], lives=r["lives"], include_lives=r["include_lives"],
                        max_steps=study.FIXED["max_steps"], observation_version=r["observation_version"],
                        **r.get("env_options", {}))
        agent = load_agent(str(run_dir / "model.pt"), env.observation_space.shape[0], 4)
        ev = r["eval"]
        n = a.episodes or len(ev["scores"])
        match = 0
        for i in range(n):
            seed = ev["seed0"] + i
            ep = trace_episode(env, agent, seed)
            same = (ep["score"] == ev["scores"][i] and ep["cleared"] == ev["clears"][i] and ep["steps"] == ev["steps"][i])
            match += same
            ep.update(label=r["label"], train_seed=r["seed"], reproduced=same)
            with gzip.open(out / "traces" / f"{r['label']}_s{r['seed']}_e{seed}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(ep, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            attr = head_attribution(ep["trace"]) if ep["ending"] == "caught" else []
            for d in ep["deaths"]:
                death_rows.append({"label": r["label"], "train_seed": r["seed"], "eval_seed": seed, **d,
                                   "final": d["lives_before"] == 1,
                                   "pellet_fraction": 1 - d["dots_left"] / ep["total_dots"]})
            episodes.append({k: ep[k] for k in ("label", "train_seed", "seed", "score", "cleared", "steps",
                                                 "ending", "dots_eaten", "total_dots", "lives_left", "reproduced")}
                            | {"n_deaths": len(ep["deaths"]), "attribution": attr,
                               "remaining": ep["remaining"], "death_pos": [d["pos"] for d in ep["deaths"]],
                               "leave_after_eat": leave_after_eat(ep["trace"]),
                               "history_items": history_items(ep["trace"])})
            print(f"{r['label']} s{r['seed']} e{seed}: score {ep['score']} {ep['ending']} "
                  f"steps {ep['steps']} {'一致' if same else '不一致'}", flush=True)
        repro.append({"label": r["label"], "train_seed": r["seed"], "episodes": n, "matched": match})
    save_json(out / "reproduction.json", repro)
    with open(out / "deaths.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(death_rows[0]))
        w.writeheader()
        w.writerows(death_rows)
    save_json(out / "episodes.json", episodes)
    summarize(out, episodes, death_rows)


def leave_after_eat(trace):
    """leave_dot が付いたステップのうち、直前2ステップ以内にエサを食べていたもの"""
    total = after = 0
    for i, row in enumerate(trace):
        if "leave_dot" in row["items"]:
            total += 1
            if any("dot" in trace[j]["items"] or "power" in trace[j]["items"] for j in range(max(0, i - 2), i + 1)):
                after += 1
    return [after, total]


def history_items(trace):
    """履歴に依存する補助報酬の合計 (観測に履歴全体は入っていない)"""
    c = Counter()
    for row in trace:
        for k in ("new_cell", "loop", "oscillation", "heading", "approach_dot", "leave_dot"):
            if k in row["items"]:
                c[k] += row["items"][k]
    return dict(c)


def summarize(out, episodes, deaths):
    summary = {}
    for label in dict.fromkeys(e["label"] for e in episodes):
        E = [e for e in episodes if e["label"] == label]
        D = [d for d in deaths if d["label"] == label]
        fails = [e for e in E if e["ending"] == "caught"]
        final = [d for d in D if d["final"]]
        att = [x for e in fails for x in e["attribution"]]
        conflict = [x for x in att if x["best"]["ghost"] != x["action"]]
        dot_over_ghost = [x for x in conflict if x["best"]["dot"] == x["action"]]
        la = np.array([e["leave_after_eat"] for e in E])
        hist = Counter()
        for e in E:
            hist.update(e["history_items"])
        s = {
            "episodes": len(E), "clears": sum(e["cleared"] for e in E),
            "reproduced": sum(e["reproduced"] for e in E),
            "failures": len(fails), "failures_pellet90": sum(e["dots_eaten"] / e["total_dots"] >= 0.9 for e in fails),
            "remaining_mean_in_failures": float(np.mean([e["total_dots"] - e["dots_eaten"] for e in fails])) if fails else None,
            "deaths": len(D),
            "deaths_by_life": dict(Counter(f"life{d['lives_before']}" for d in D)),
            "deaths_late_pellet90": sum(d["pellet_fraction"] >= LATE_FRACTION for d in D),
            "deaths_within10_after_power_end": sum(d["since_power_end"] is not None and d["since_power_end"] <= 10 for d in D),
            "deaths_during_power": sum(d["power_active_before"] for d in D),
            "deaths_within10_after_eat_ghost": sum(d["since_eat_ghost"] is not None and d["since_eat_ghost"] <= 10 for d in D),
            "deaths_within30_after_respawn": sum(d["since_respawn"] <= 30 and d["lives_before"] < 3 for d in D),
            "killer_ghost": dict(Counter(["Blinky", "Pinky", "Inky", "Clyde"][d["killer"]] for d in D)),
            "final_death_pellet_fraction_mean": float(np.mean([d["pellet_fraction"] for d in final])) if final else None,
            "last5_steps_before_final_death": len(att),
            "action_differs_from_ghost_head_best": len(conflict),
            "of_which_equals_dot_head_best": len(dot_over_ghost),
            "leave_dot_right_after_eating": [int(la[:, 0].sum()), int(la[:, 1].sum())],
            "history_dependent_reward_totals_per_episode": {k: round(v / len(E), 1) for k, v in hist.items()},
        }
        s["deaths_two_or_more_danger_dirs_3before"] = sum(d["danger_dirs_3before"] >= 2 for d in D)
        s["deaths_dead_end_or_corridor_3before"] = sum(d["exits_3before"] <= 2 for d in D)
        summary[label] = s
    save_json(out / "summary.json", summary)
    heatmaps(out, episodes)


def heatmaps(out, episodes):
    plt = study._plt()
    env = PacManEnv()
    env.reset(seed=0)
    walls = np.array([[c == CELL_WALL for c in row] for row in env.grid], dtype=float)
    labels = list(dict.fromkeys(e["label"] for e in episodes))
    fig, axes = plt.subplots(len(labels), 2, figsize=(10, 5.5 * len(labels)))
    axes = np.atleast_2d(axes)
    for i, label in enumerate(labels):
        E = [e for e in episodes if e["label"] == label]
        death = np.zeros_like(walls)
        rem = np.zeros_like(walls)
        for e in E:
            for p in e["death_pos"]:
                death[p[0], p[1]] += 1
            if e["ending"] == "caught":
                for p in e["remaining"]:
                    rem[p[0], p[1]] += 1
        for ax, m, title in ((axes[i, 0], death, "死亡位置 (全捕獲)"), (axes[i, 1], rem, "失敗時の残りエサ")):
            ax.imshow(np.where(walls > 0, np.nan, m), cmap="hot_r")
            ax.imshow(np.where(walls > 0, 1, np.nan), cmap="Blues", vmin=0, vmax=1.5)
            ax.set_title(f"{label}: {title}  合計 {int(m.sum())}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
