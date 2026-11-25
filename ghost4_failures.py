"""
保存済みモデルの評価を再生し、残機を失った場面と時間切れを分類する。学習はしない。

  python ghost4_failures.py --study output/ghost4_round2_20261007/study --labels safe_obs control \\
      --seed0 940000 --episodes 100 --check output/ghost4_round2_20261007/report/final_by_episode.csv \\
      --out output/ghost4_round2_20261007/failures

・成功した試行の中で残機を失った場面も含め、全試行を対象にする (死因の頻度を偏らせないため)
・1回ごとの得点・ステップ・成否を --check の CSV と1行ずつ照合する (評価の再現)
・残機を失うごとに記録 (deaths.csv): 残りエサ、何機目か、復帰からの経過、パワー終了からの経過、3ステップ前の挟み撃ち、
  直前20ステップの往復・目標 (最寄りエサ) の切り替え・訪れたマスの数、選んだ行動の安全領域 (tactical_safe のみ) が
  選べた行動の中で最大だったか
・時間切れ (timeouts.csv): 最後の100ステップの訪問マス数・往復・目標切り替え・食べたエサ・危険な敵の近さ
・分類は重なりを許すフラグで数え、主な分類 (優先順: パワー終了直後 > 復帰直後 > 挟み撃ち > 終盤 > その他) も付ける
・残りエサの位置 (失敗した試行の終了時) をマスごとに数える (remaining_dots.json)
"""
import argparse
import csv
import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ENDGAME_DOTS = 10
FEW_DOTS = 5
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


def nearest_dot(env):
    """パックマンから迷路距離で最も近いエサ (同じ距離なら行・列の小さい方)"""
    from pacman_env import CELL_DOT, CELL_POWER
    field = env._distance_field([tuple(env.pacman)])
    best = None
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r][c] in (CELL_DOT, CELL_POWER) and field[r][c] >= 0:
                key = (field[r][c], r, c)
                if best is None or key < best:
                    best = key
    return None if best is None else (best[1], best[2])


def safe_dot_distances(env):
    """行動ごとに、敵より先に着けるマスだけを通って行ける最寄りエサまでの距離 (上限なし。行けなければ None)。
    pacman_env._route_features の [0] と同じ判定で、30マスで切らない版 (遠いのか行けないのかを区別するため)"""
    from pacman_env import DIRS, CELL_WALL, CELL_DOT, CELL_POWER
    threat = env._threat_field()
    dots = {(r, c) for r in range(env.rows) for c in range(env.cols) if env.grid[r][c] in (CELL_DOT, CELL_POWER)}
    ok = lambda r, c, d: threat is None or threat[r][c] < 0 or threat[r][c] - (d + 1) > 0
    pr, pc = env.pacman
    out = []
    for dr, dc in DIRS:
        nr, nc = pr + dr, pc + dc
        if not (0 <= nr < env.rows and 0 <= nc < env.cols and env.grid[nr][nc] != CELL_WALL):
            nr, nc = pr, pc
        found = None
        if ok(nr, nc, 0):
            seen, queue = {(nr, nc): 0}, deque([(nr, nc)])
            while queue:
                cell = queue.popleft()
                if cell in dots:
                    found = seen[cell]
                    break
                for ddr, ddc in DIRS:
                    q = (cell[0] + ddr, cell[1] + ddc)
                    if (0 <= q[0] < env.rows and 0 <= q[1] < env.cols and q not in seen
                            and env.grid[q[0]][q[1]] != CELL_WALL and ok(q[0], q[1], seen[cell] + 1)):
                        seen[q] = seen[cell] + 1
                        queue.append(q)
        out.append(found)
    return out


def route_class(d):
    """safe_route の [0] から見た分類: near (0〜29 で正の値) / far (30以上で0) / none (行けないので0)"""
    return "none" if d is None else ("far" if d >= 30 else "near")


def window_stats(steps):
    actions = [s["action"] for s in steps]
    reversals = sum(1 for a, b in zip(actions, actions[1:]) if b == OPPOSITE[a])
    targets = [s["target"] for s in steps]
    # 目標の切り替え: 前の目標がまだエサとして残っているのに、最寄りのエサが別のマスに変わった回数
    switches = sum(1 for i in range(1, len(steps))
                   if targets[i] != targets[i - 1] and targets[i - 1] is not None
                   and steps[i]["dots_left"] == steps[i - 1]["dots_left"])
    return dict(reversals=reversals, target_switches=switches, distinct_cells=len({tuple(s["pac"]) for s in steps}),
                dots_eaten=steps[0]["dots_left"] - steps[-1]["dots_left"] if steps else 0,
                danger_le3_steps=sum(s["danger_min"] <= 3 for s in steps))


def run_model(args):
    study_dir, r, seed0, n = args
    import torch
    import reward_study as study
    from dqn_agent import load_agent
    from pacman_env import PacManEnv
    from ghost4_diagnose import danger_context
    torch.set_num_threads(1)
    env = PacManEnv(render_mode=None, ghost_mode=study.FIXED["ghost_mode"], num_ghosts=r["num_ghosts"],
                    rewards=r["rewards"], lives=r["lives"], include_lives=r["include_lives"],
                    max_steps=study.FIXED["max_steps"], observation_version=r["observation_version"],
                    **r.get("env_options", {}))
    agent = load_agent(str(Path(study_dir) / "runs" / r["id"] / "model.pt"), env.observation_space.shape[0], 4)
    safe_obs = r["observation_version"] == "tactical_safe"
    episodes, deaths, timeouts, remaining = [], [], [], Counter()
    endgame_steps = []
    for i in range(n):
        seed = seed0 + i
        obs, _ = env.reset(seed=seed)
        history = deque(maxlen=100)
        last_respawn, last_power_end, life_no = 0, None, 0
        first_death_dots = None
        while True:
            action = agent.select_action(obs, training=False)
            ctx = danger_context(env)
            row = dict(t=env.steps + 1, pac=list(env.pacman), action=action, dots_left=env.total_dots - env.dots_eaten,
                       lives=env.lives, power=env.power_timer, target=nearest_dot(env), **ctx)
            legal = [a for a in range(4) if obs[agent.action_mask_indices[a]] > 0.5] or list(range(4))
            if row["dots_left"] <= ENDGAME_DOTS:          # 終盤だけ計算 (計算量を抑える)
                dist = safe_dot_distances(env)
                row["route_chosen"] = route_class(dist[action])
                row["route_best"] = min((route_class(dist[b]) for b in legal), key=["near", "far", "none"].index)
                endgame_steps.append(dict(label=r["label"], route_chosen=row["route_chosen"], route_best=row["route_best"]))
            if safe_obs:
                area = obs[-12:].reshape(4, 3)[:, 0]
                row["chosen_area_is_max"] = bool(area[action] >= max(area[a] for a in legal) - 1e-6)
                row["chosen_area"] = float(area[action])
            history.append(row)
            power_before, lives_before = env.power_timer, env.lives
            obs, _, te, tr, info = env.step(action)
            t = env.steps
            if power_before > 0 and env.power_timer == 0:
                last_power_end = t
            if info["lives"] < lives_before:
                life_no += 1
                h = list(history)
                last20 = h[-20:]
                before3 = h[-4] if len(h) >= 4 else h[0]
                d = dict(label=r["label"], train_seed=r["seed"], eval_seed=seed, t=t, life_no=life_no,
                         lives_before=lives_before, dots_left=row["dots_left"],
                         since_respawn=t - last_respawn,
                         since_power_end=None if last_power_end is None else t - last_power_end,
                         power_active=power_before > 0,
                         danger_dirs_3before=before3["danger_dirs_within6"], danger_min_3before=before3["danger_min"],
                         exits_3before=before3["exits"], **{f"last20_{k}": v for k, v in window_stats(last20).items()})
                for k in (10, 3):
                    back = h[-(k + 1)] if len(h) > k else h[0]
                    d[f"route_best_{k}before"] = back.get("route_best")
                    d[f"route_chosen_{k}before"] = back.get("route_chosen")
                if safe_obs:
                    d["chosen_area_max_3before"] = before3.get("chosen_area_is_max")
                    d["chosen_area_3before"] = round(before3.get("chosen_area", 0.0), 3)
                flags = dict(power_end=d["since_power_end"] is not None and d["since_power_end"] <= 10,
                             respawn=life_no > 1 and d["since_respawn"] <= 30,
                             pincer=d["danger_dirs_3before"] >= 2,
                             endgame=d["dots_left"] <= ENDGAME_DOTS,
                             few_dots=d["dots_left"] <= FEW_DOTS,
                             dithering=d["last20_target_switches"] >= 3 or d["last20_reversals"] >= 4)
                d.update({f"flag_{k}": v for k, v in flags.items()})
                d["primary"] = next((k for k in ("power_end", "respawn", "pincer", "endgame") if flags[k]), "other")
                deaths.append(d)
                if first_death_dots is None:
                    first_death_dots = row["dots_left"]
                last_respawn = t
            if te or tr:
                break
        ending = "clear" if info["cleared"] else ("truncated" if tr and not te else "caught")
        dots_left = env.total_dots - env.dots_eaten
        episodes.append(dict(label=r["label"], train_seed=r["seed"], eval_seed=seed, cleared=int(bool(info["cleared"])),
                             score=info["score"], steps=t, ending=ending, dots_left=dots_left,
                             lives_lost=r["lives"] - info["lives"], first_death_dots_left=first_death_dots))
        if ending == "truncated":
            timeouts.append(dict(label=r["label"], train_seed=r["seed"], eval_seed=seed, dots_left=dots_left,
                                 lives_left=info["lives"], **{f"last100_{k}": v for k, v in window_stats(list(history)).items()}))
        if not info["cleared"]:
            from pacman_env import CELL_DOT, CELL_POWER
            for rr in range(env.rows):
                for cc in range(env.cols):
                    if env.grid[rr][cc] in (CELL_DOT, CELL_POWER):
                        remaining[f"{rr},{cc}"] += 1
    return episodes, deaths, timeouts, dict(remaining), dict(Counter((x["route_best"], x["route_chosen"]) for x in endgame_steps))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))     # 条件によって列が違う (tactical_safe だけの列)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def summarize(label, episodes, deaths, timeouts):
    eps = [e for e in episodes if e["label"] == label]
    ds = [d for d in deaths if d["label"] == label]
    fails = [e for e in eps if not e["cleared"]]
    final_deaths = [d for d in ds if d["lives_before"] == 1]          # 最後の1機を失った場面
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    flag_names = [k[5:] for k in ds[0] if k.startswith("flag_")] if ds else []
    out = dict(
        episodes=len(eps), clears=sum(e["cleared"] for e in eps), failures=len(fails),
        failures_caught=sum(e["ending"] == "caught" for e in fails),
        failures_timeout=sum(e["ending"] == "truncated" for e in fails),
        failures_le5_dots=sum(e["dots_left"] <= FEW_DOTS for e in fails),
        lives_left_on_clear=dict(Counter(3 - e["lives_lost"] for e in eps if e["cleared"])),
        deaths_total=len(ds), deaths_in_cleared_games=sum(1 for d in ds if any(
            e["cleared"] and e["eval_seed"] == d["eval_seed"] and e["train_seed"] == d["train_seed"] for e in eps)),
        deaths_by_life=dict(Counter(d["life_no"] for d in ds)),
        primary_all=dict(Counter(d["primary"] for d in ds)),
        primary_final_life=dict(Counter(d["primary"] for d in final_deaths)),
        flags_all_pct={k: pct(sum(d[f"flag_{k}"] for d in ds), len(ds)) for k in flag_names},
        flags_final_life_pct={k: pct(sum(d[f"flag_{k}"] for d in final_deaths), len(final_deaths)) for k in flag_names},
        first_death_dots_left_median=float(np.median([e["first_death_dots_left"] for e in eps
                                                      if e["first_death_dots_left"] is not None])) if ds else None,
    )
    if ds and "chosen_area_max_3before" in ds[0]:
        out["chosen_area_max_3before_pct"] = pct(sum(bool(d["chosen_area_max_3before"]) for d in ds), len(ds))
        few = [d for d in ds if d["flag_few_dots"]]
        out["few_dots_deaths"] = len(few)
        out["few_dots_chosen_area_max_3before_pct"] = pct(sum(bool(d["chosen_area_max_3before"]) for d in few), len(few))
        out["few_dots_dithering_pct"] = pct(sum(d["flag_dithering"] for d in few), len(few))
        out["few_dots_pincer_pct"] = pct(sum(d["flag_pincer"] for d in few), len(few))
    out["timeouts"] = [{k: t[k] for k in t if k not in ("label",)} for t in timeouts if t["label"] == label]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--check", help="1回ごとの結果の CSV (variant, train_seed, eval_seed, cleared, score, steps)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((Path(a.study) / "results.json").read_text(encoding="utf-8"))
    if (Path(a.study) / "results_final.json").exists():     # 第3回: 最終評価で学習seedを増やしたモデルも含める
        seen = {r["id"] for r in results}
        results += [r for r in json.loads((Path(a.study) / "results_final.json").read_text(encoding="utf-8"))
                    if r["id"] not in seen]
    results = [r for r in results if r["label"] in a.labels]
    with ProcessPoolExecutor(a.workers) as pool:
        parts = list(pool.map(run_model, [(a.study, r, a.seed0, a.episodes) for r in results]))
    episodes = [e for p in parts for e in p[0]]
    deaths = [d for p in parts for d in p[1]]
    timeouts = [t for p in parts for t in p[2]]
    remaining = {}
    for r, p in zip(results, parts):
        bucket = remaining.setdefault(r["label"], Counter())
        bucket.update(p[3])
    route_counts = {}
    for r, p in zip(results, parts):
        bucket = route_counts.setdefault(r["label"], Counter())
        bucket.update({f"best={k[0]},chosen={k[1]}": v for k, v in p[4].items()})
    if a.check:
        saved = {(x["variant"], int(x["train_seed"]), int(x["eval_seed"])): x
                 for x in csv.DictReader(open(a.check, encoding="utf-8-sig"))}
        bad = [e for e in episodes if (lambda s: s is None or (int(s["cleared"]), int(s["score"]), int(s["steps"]))
                                       != (e["cleared"], e["score"], e["steps"]))(saved.get((e["label"], e["train_seed"], e["eval_seed"])))]
        assert not bad, f"{len(bad)} episodes differ from {a.check}: {bad[:3]}"
        print("EPISODES_REPRODUCED", len(episodes), flush=True)
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "deaths.csv", deaths)
    write_csv(out / "timeouts.csv", timeouts)
    (out / "remaining_dots.json").write_text(json.dumps({k: dict(v) for k, v in remaining.items()}, indent=1), encoding="utf-8")
    summary = {label: summarize(label, episodes, deaths, timeouts) for label in a.labels}
    for label in a.labels:
        rc = route_counts.get(label, Counter())
        total = sum(rc.values())
        best = Counter()
        for k, v in rc.items():
            best[k.split(",")[0][5:]] += v
        lab_deaths = [d for d in deaths if d["label"] == label and d["lives_before"] == 1 and d["dots_left"] <= ENDGAME_DOTS]
        summary[label]["route_endgame_steps"] = dict(
            steps=total, best_action_pct={k: round(100 * v / total, 1) for k, v in best.items()} if total else {},
            detail=dict(rc),
            final_life_endgame_deaths=len(lab_deaths),
            best_10before=dict(Counter(d["route_best_10before"] for d in lab_deaths)),
            best_3before=dict(Counter(d["route_best_3before"] for d in lab_deaths)),
            note="best = 選べる行動の中で一番良い分類 (near > far > none)。far と none は safe_route の [0] ではどちらも 0")
    summary["_settings"] = dict(seed0=a.seed0, episodes=a.episodes, endgame_dots=ENDGAME_DOTS, few_dots=FEW_DOTS,
                                reproduced=bool(a.check))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
