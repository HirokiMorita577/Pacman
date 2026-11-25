"""
報酬設計の実験 (自作パックマン, HRA 版 DQN)
================================================================
発表用に「1回に1つの変数だけ変える比較」と「PSO による報酬の探索」を自動で行い、
条件ごとのグラフ・重ね合わせグラフ・Q値の振れ幅・結果の表・設計の根拠の表を --out に保存する。
--out を Google Drive にすれば成果物がそのまま Drive に残る。中断しても、終わった学習は飛ばして続きから。

  python reward_study.py --out /content/drive/MyDrive/DQN_reward_study --workers 8
  python reward_study.py --out <dir> --report-only     # 図表だけ作り直す
  python reward_study.py --out <dir> --quick           # 動作確認 (数エピソード)

実験の流れ (前の実験の結果で次の実験の基準値を決める):
  実験0  報酬セット      前回 (old) / トータル設計 (designed)
  実験1  捕獲ペナルティ  100 / 500 / 1,560 (=10N) / 15,600 (=100N, 大きすぎる例)   他は designed で固定
  実験2  クリアボーナス  0 / 500 / 1,560 (=10N) / 5,000                          death は実験1の最良
  実験3  割引率 γ        0.95 / 0.99 / 0.995                                     報酬は実験2までの最良
  実験4  PSO             PSO_SPACE の6変数を探す (適応度 = 学習後の評価スコア)
  実験5  最終比較        前回 / 単変量比較の最良 / PSO の最良 を同じ条件・同じ seed で
================================================================
"""
import argparse
from collections import deque
import concurrent.futures as cf
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np

from pacman_env import (PacManEnv, REWARD_SPEC, REWARD_SIGN, OLD_REWARDS, DESIGNED_REWARDS,
                        SHAPING_ITEMS, N_PELLETS, N_REWARD_COMPONENTS, REWARD_COMPONENT_NAMES)

# ── 固定する学習設定 (train.py の既定値と同じ) ──
FIXED = {
    "lr": 5e-4,                  # 学習率 α (Adam)
    "gamma": 0.99,               # 割引率 γ (実験3で比較)
    "epsilon_start": 1.0,
    "epsilon_end": 0.0248,
    "explore_ratio": 0.4,        # 全エピソードの40%で ε が下限に届くよう減衰率を決める
    "batch_size": 128,
    "target_update": 1000,       # ターゲットネットワークの更新間隔 (学習ステップ)
    "buffer": 100000,            # 経験再生バッファ
    "max_steps": 1000,           # 1エピソードの上限 (env 側)
    "num_ghosts": 1,
    "ghost_mode": "classic",
}
ENDGAME_DOTS = 10              # 終盤 = 残りエサがこの数以下 (リプレイの診断用)
EVAL_SEED0 = 900000             # 評価に使う盤面の乱数 (全条件で同じ)

# ── PSO の設定 ──
PSO_SPACE = [  # (報酬の名前, 下限, 上限, 説明)
    ("death",     50.0, 3000.0, "捕獲ペナルティ"),
    ("clear",      0.0, 5000.0, "クリアボーナス"),
    ("power",      0.0,  200.0, "パワーエサ"),
    ("eat_ghost",  0.0, 1000.0, "弱体ゴースト"),
    ("time",       0.0,    1.0, "1ステップの罰"),
    ("shaping",    0.0,    2.0, "整形報酬の倍率 (近づく・往復・初めてのマス など9項目)"),
]
PSO_DEFAULT = {"particles": 8, "iterations": 8, "w_start": 0.9, "w_end": 0.4, "c1": 1.5, "c2": 1.5,
               "train_episodes": 400, "eval_episodes": 20, "seed": 101}

NAMES_JA = {k: v[2] for k, v in REWARD_SPEC.items()}
CONDITION_JA = {  # 報酬表の「与える条件」
    "dot": "エサのマスに入った", "dot_progress": "エサを食べたとき (食べた割合 × 値)", "power": "パワーエサのマスに入った",
    "approach_dot": "2ステップごと、最寄りのエサまでの距離が縮んだ", "leave_dot": "2ステップごと、距離が伸びた",
    "clear": "全156個を食べた (エピソード終了)", "eat_ghost": "パワー中にゴーストと同じマスになった",
    "death": "通常のゴーストと同じマス/すれ違い (エピソード終了)", "ghost_adjacent": "パワー中でなく、ゴーストが隣のマス",
    "ghost_near": "パワー中でなく、ゴーストが2マス先", "chase_blue": "パワー中、ゴーストが2マス以内",
    "time": "毎ステップ", "loop": "直近8ステップで同じマスに3回以上 (回数に応じて重く)",
    "oscillation": "直近6ステップが2マスの往復", "heading": "同じ向きに3回続けて進んだ", "new_cell": "そのエピソードで初めてのマスに入った",
}


# ════════════════════════════════════════════════════════════
#  1回の学習
# ════════════════════════════════════════════════════════════
def experiment_manifest(job):
    packages = {}
    for name in ("torch", "numpy", "gymnasium"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    root = Path(__file__).resolve().parent
    return {"schema": 2,
            "job": {k: v for k, v in job.items() if k not in ("runs_dir", "checkpoint_every", "stop_after")},
            "fixed": FIXED, "python": platform.python_version(), "packages": packages,
            "sources": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                        for p in ("pacman_env.py", "dqn_agent.py", "reward_study.py", "pso_optimizer.py",
                                  "improvement_study.py", "ghost_study.py", "ghost4_improvement.py", "ghost4_round2.py",
                                  "ghost4_round3.py", "ghost4_round4.py", "ghost4_round5.py")
                        if (root / p).exists()}}


def job_id(job):
    key = json.dumps(experiment_manifest(job), sort_keys=True, ensure_ascii=False)
    return f"{job['exp']}_v2_{hashlib.sha256(key.encode()).hexdigest()[:16]}_s{job['seed']}"


def atomic_json(path, data):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(temporary, path)


def reward_unit(rewards):
    """Q値のスケール: エサ1個の報酬が 1 になるように報酬を割ってから学習する"""
    return rewards["dot"] if rewards.get("dot", 0) > 0 else 10.0


def run_job(job):
    """OS-held lock prevents two processes from overwriting the same run."""
    run_dir = Path(job["runs_dir"]) / job_id(job)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_job(job)


def _run_job(job):
    import torch
    torch.set_num_threads(1)
    from dqn_agent import HRADQNAgent
    manifest = experiment_manifest(job)
    random.seed(job["seed"]); np.random.seed(job["seed"]); torch.manual_seed(job["seed"])
    run_dir = os.path.join(job["runs_dir"], job_id(job))
    os.makedirs(run_dir, exist_ok=True)
    atomic_json(os.path.join(run_dir, "manifest.json"), manifest)
    unit = reward_unit(job["rewards"])
    lives = job.get("lives", 1)
    num_ghosts = job.get("num_ghosts", FIXED["num_ghosts"])
    if isinstance(num_ghosts, bool) or num_ghosts not in (1, 2, 3, 4):
        raise ValueError("num_ghosts must be an integer from 1 to 4")
    env_options = job.get("env_options", {})          # 例: {"fix_leave_dot": True} (2026-10-07 追加。なければ従来どおり)
    env = PacManEnv(render_mode=None, ghost_mode=FIXED["ghost_mode"], num_ghosts=num_ghosts,
                    rewards=job["rewards"], lives=lives,
                    include_lives=job.get("include_lives", False), max_steps=FIXED["max_steps"],
                    observation_version=job.get("observation_version", "legacy"), **env_options)
    obs_size = env.observation_space.shape[0]
    # ε が下限に届くまでの長さ。explore_steps があればそれを使う (学習を延ばしても探索の期間を変えないため。
    # 2026-10-08 追加)。なければ従来どおり 学習予算 × explore_ratio
    explore = job.get("explore_steps") or max(1, int((job.get("step_budget") or job["episodes"]) * FIXED["explore_ratio"]))
    if job.get("explore_steps") and not job.get("step_budget"):
        raise ValueError("explore_steps needs step_budget")
    agent = HRADQNAgent(obs_size, 4, N_REWARD_COMPONENTS, lr=FIXED["lr"], gamma=job["gamma"],
                        epsilon_start=FIXED["epsilon_start"], epsilon_end=FIXED["epsilon_end"],
                        epsilon_decay=FIXED["epsilon_end"] ** (1.0 / explore), batch_size=FIXED["batch_size"],
                        target_update_freq=FIXED["target_update"], decay_per_episode=True,
                        buffer_capacity=FIXED["buffer"], device="cpu",
                        action_mask_indices=env.action_mask_indices if job.get("mask_illegal_actions", False) else None)
    # Q値を測る固定の状態 (ランダムに動いたときの256状態。全条件で同じ)
    probe_env = PacManEnv(render_mode=None, num_ghosts=num_ghosts, lives=lives,
                         include_lives=job.get("include_lives", False),
                         observation_version=job.get("observation_version", "legacy"), **env_options)
    rng = np.random.default_rng(0)
    o, _ = probe_env.reset(seed=0)
    probe = []
    while len(probe) < 256:
        probe.append(o)
        o, _, te, tr, _ = probe_env.step(int(rng.integers(4)))
        if te or tr:
            o, _ = probe_env.reset(seed=len(probe))
    probe_array = np.array(probe)
    probe_t = torch.FloatTensor(probe_array).to(agent.device)

    items = list(REWARD_SPEC)
    fields = ["episode", "score", "cleared", "steps", "lives_left", "return", "epsilon", "loss",
              "q_mean", "q_min", "q_max", "total_steps", "updates",
              "replay_endgame_frac", "endgame_pushed", "endgame_sampled_expected"] + items
    for h in range(N_REWARD_COMPONENTS):
        fields += [f"head{h}_loss", f"head{h}_q_mean"]
    log_path = os.path.join(run_dir, "log.csv")
    checkpoint_path = os.path.join(run_dir, "training.pt.gz" if job.get("compress_checkpoint") else "training.pt")
    progress = {"episode": 0, "rows": [], "total_steps": 0, "train_seconds": 0.0, "events": []}
    if os.path.exists(checkpoint_path):
        progress = agent.load_training(checkpoint_path, manifest)
    total_steps = progress["total_steps"]
    rows = progress["rows"]
    events = progress.get("events", [])
    # 学習途中の貪欲評価 (2026-10-08 追加)。curve_eval = {"every": 環境ステップ, "episodes": 回数, "seed0": 評価seed}
    # 学習中のクリア率は探索の行動を含むので、伸びているかは固定seedの貪欲評価で見る。評価は別の環境で行い、学習の乱数は使わない
    # 終盤 (残りエサ ENDGAME_DOTS 個以下) の経験がリプレイにどれだけあり、学習でどれだけ使われたか (2026-10-08 追加)。
    # 抽出は一様なので、使われた数の期待値 = Σ バッチサイズ × その時点のバッファ内の割合。乱数は使わない (学習は変わらない)
    endgame_flags = deque(progress.get("endgame_flags", []), maxlen=FIXED["buffer"])
    endgame_in_buffer = sum(endgame_flags)
    endgame_pushed = progress.get("endgame_pushed", 0)
    endgame_sampled = progress.get("endgame_sampled_expected", 0.0)
    curve_cfg = job.get("curve_eval")
    curve = progress.get("curve", [])
    curve_fields = ["total_steps", "episode", "epsilon", "clears", "episodes", "clear_rate", "score_mean",
                    "pellet_fraction_mean", "lives_lost_mean", "steps_mean"]
    t0 = time.time()
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        q_stat = progress.get("q_stat", (0.0, 0.0, 0.0))
        head_q_stat = progress.get("head_q_stat", [0.0] * N_REWARD_COMPONENTS)
        for ep in range(progress["episode"] + 1, job["episodes"] + 1):
            if job.get("step_budget") and total_steps >= job["step_budget"]:
                break
            obs, _ = env.reset(seed=job["seed"] * 100000 + ep)
            ret, losses, head_losses = 0.0, [], []
            while True:
                if job.get("step_budget"):
                    agent.epsilon = max(FIXED["epsilon_end"], FIXED["epsilon_end"] ** (total_steps / explore))
                lives_before = env.lives
                endgame = env.total_dots - env.dots_eaten <= ENDGAME_DOTS
                a = agent.select_action(obs, training=True)
                nxt, r, te, tr, info = env.step(a)
                if info["lives"] < lives_before or info["cleared"]:
                    with torch.no_grad():
                        heads = agent.policy_net.head_q(torch.FloatTensor(obs).unsqueeze(0).to(agent.device))[0]
                    events.append({"episode": ep, "step": env.steps, "action": a,
                                   "lives_before": lives_before, "lives_after": info["lives"],
                                   "cleared": bool(info["cleared"]), "score": info["score"],
                                   "epsilon": agent.epsilon, "head_q": (heads * unit).cpu().tolist()})
                # v2: the time limit is an external truncation, so bootstrap on it.
                agent.push(obs, a, info["reward_components"] / unit, nxt, float(te))
                if len(endgame_flags) == endgame_flags.maxlen:
                    endgame_in_buffer -= endgame_flags[0]
                endgame_flags.append(endgame)
                endgame_in_buffer += endgame
                endgame_pushed += endgame
                loss = agent.train_step()
                if loss is not None:
                    endgame_sampled += agent.batch_size * endgame_in_buffer / len(endgame_flags)
                    losses.append(loss)
                    head_losses.append(agent.last_head_losses)
                total_steps += 1
                ret += r
                obs = nxt
                if te or tr or (job.get("step_budget") and total_steps >= job["step_budget"]):
                    break
            agent.end_episode()
            if job.get("step_budget"):
                agent.epsilon = max(FIXED["epsilon_end"], FIXED["epsilon_end"] ** (total_steps / explore))
            if curve_cfg and total_steps >= (len(curve) + 1) * curve_cfg["every"]:
                cv = evaluate(agent, job["rewards"], curve_cfg["episodes"], lives, seed0=curve_cfg["seed0"],
                              include_lives=job.get("include_lives", False), num_ghosts=num_ghosts,
                              observation_version=job.get("observation_version", "legacy"), env_options=env_options)
                curve.append({"total_steps": total_steps, "episode": ep, "epsilon": round(agent.epsilon, 4),
                              "clears": sum(cv["clears"]), "episodes": curve_cfg["episodes"],
                              **{k: cv[k] for k in curve_fields[5:]}})
                with open(os.path.join(run_dir, "eval_curve.csv"), "w", newline="", encoding="utf-8") as cf_:
                    cw = csv.DictWriter(cf_, fieldnames=curve_fields)
                    cw.writeheader()
                    cw.writerows(curve)
            if ep % 10 == 0 or ep == 1:
                with torch.no_grad():
                    q = agent.policy_net(probe_t).max(dim=1).values.cpu().numpy() * unit
                    head_q_stat = (agent.policy_net.head_q(probe_t).mean(dim=(0, 2)) * unit).cpu().tolist()
                q_stat = (float(q.mean()), float(q.min()), float(q.max()))
            row = {"episode": ep, "score": info["score"], "cleared": int(info["cleared"]), "steps": env.steps,
                   "lives_left": info["lives"], "return": round(ret, 2), "epsilon": round(agent.epsilon, 4),
                   "loss": round(float(np.mean(losses)), 5) if losses else "",
                   "q_mean": round(q_stat[0], 2), "q_min": round(q_stat[1], 2), "q_max": round(q_stat[2], 2)}
            row.update(total_steps=total_steps, updates=agent.steps,
                       replay_endgame_frac=round(endgame_in_buffer / max(1, len(endgame_flags)), 5),
                       endgame_pushed=endgame_pushed, endgame_sampled_expected=round(endgame_sampled, 1))
            for h in range(N_REWARD_COMPONENTS):
                row[f"head{h}_loss"] = float(np.mean(head_losses, axis=0)[h]) if head_losses else ""
                row[f"head{h}_q_mean"] = head_q_stat[h]
            row.update({k: round(info["reward_items"].get(k, 0.0), 2) for k in items})
            w.writerow(row)
            rows.append(row)
            if ep % job.get("checkpoint_every", 50) == 0 or ep == job["episodes"] or (
                    job.get("step_budget") and total_steps >= job["step_budget"]) or ep == job.get("stop_after"):
                f.flush()
                snapshot = {"episode": ep, "rows": rows, "total_steps": total_steps,
                            "train_seconds": progress["train_seconds"] + time.time() - t0, "events": events}
                snapshot.update(q_stat=q_stat, head_q_stat=head_q_stat)
                if curve_cfg:
                    snapshot["curve"] = curve
                snapshot.update(endgame_flags=list(endgame_flags), endgame_pushed=endgame_pushed,
                                endgame_sampled_expected=endgame_sampled)
                agent.save_training(checkpoint_path, snapshot, manifest)
                atomic_json(os.path.join(run_dir, "events.json"), events)
                if job.get("show_progress"):
                    status = {"seed": job["seed"], "num_ghosts": num_ghosts,
                              "episode": ep, "total_steps": total_steps,
                              "step_budget": job.get("step_budget"),
                              "train_seconds": snapshot["train_seconds"]}
                    atomic_json(os.path.join(run_dir, "progress.json"), status)
                    print(f"PROGRESS seed={job['seed']} ghosts={num_ghosts} "
                          f"steps={total_steps}/{job.get('step_budget', '?')} episodes={ep}", flush=True)
            if ep == job.get("stop_after"):
                return {"paused": True, "episode": ep, "id": job_id(job)}
    train_sec = progress["train_seconds"] + time.time() - t0
    ev = evaluate(agent, job["rewards"], job["eval_episodes"], lives,
                  seed0=job.get("eval_seed0", EVAL_SEED0), include_lives=job.get("include_lives", False),
                  num_ghosts=num_ghosts, observation_version=job.get("observation_version", "legacy"),
                  env_options=env_options)
    if job.get("save_model"):
        agent.save(os.path.join(run_dir, "model.pt"))
    result = {"id": job_id(job), "exp": job["exp"], "label": job["label"], "seed": job["seed"],
              "rewards": job["rewards"], "gamma": job["gamma"], "episodes": job["episodes"], "lives": lives,
              "train_minutes": round(train_sec / 60, 1), "eval": ev, "manifest": manifest,
              "total_steps": total_steps, "updates": agent.steps, "episodes_completed": len(rows),
              "include_lives": job.get("include_lives", False), "num_ghosts": num_ghosts,
              "observation_size": obs_size, "observation_version": job.get("observation_version", "legacy"),
              "mask_illegal_actions": job.get("mask_illegal_actions", False), "env_options": env_options}
    if curve_cfg:
        result["curve"] = curve
    if job.get("explore_steps"):
        result["explore_steps"] = job["explore_steps"]
    atomic_json(os.path.join(run_dir, "result.json"), result)
    return result


def evaluate(agent, rewards, n, lives=1, seed0=EVAL_SEED0, include_lives=False, num_ghosts=None,
             observation_version="legacy", env_options=None):
    """貪欲方策 (ランダムな行動なし) で n 回プレイ。盤面の乱数は全条件で同じ"""
    num_ghosts = FIXED["num_ghosts"] if num_ghosts is None else num_ghosts
    if isinstance(num_ghosts, bool) or num_ghosts not in (1, 2, 3, 4):
        raise ValueError("num_ghosts must be an integer from 1 to 4")
    env = PacManEnv(render_mode=None, ghost_mode=FIXED["ghost_mode"], num_ghosts=num_ghosts, rewards=rewards,
                    lives=lives, include_lives=include_lives, max_steps=FIXED["max_steps"],
                    observation_version=observation_version, **(env_options or {}))
    scores, clears, steps, item_sum, ends = [], [], [], {}, []
    pellets, lives_lost, blocked, dots_left = [], [], [], []
    parts = {"dot_points": [], "power_points": [], "ghost_points": [], "ghosts_eaten": [], "power_eaten": []}
    power_steps = []      # 1回ごとの、パワーエサを取ったステップ (2026-10-15 追加)
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        blocked_count = 0
        while True:
            action = agent.select_action(obs, training=False)
            blocked_count += int(obs[env.action_mask_indices[action]] < 0.5)
            obs, _, te, tr, info = env.step(action)
            if te or tr:
                break
        scores.append(info["score"]); clears.append(info["cleared"]); steps.append(env.steps)
        pellets.append(info["dots_eaten"] / env.total_dots)
        dots_left.append(env.total_dots - info["dots_eaten"])
        for k, v in (("dot_points", info["score_parts"]["dot"]), ("power_points", info["score_parts"]["power"]),
                     ("ghost_points", info["score_parts"]["ghost"]), ("ghosts_eaten", info["ghosts_eaten"]),
                     ("power_eaten", info["power_eaten"])):
            parts[k].append(v)
        power_steps.append(info["power_steps"])
        lives_lost.append(lives - info["lives"])
        blocked.append(blocked_count)
        ends.append("clear" if info["cleared"] else ("truncated" if tr and not te else "caught"))
        for k, v in info["reward_items"].items():
            item_sum[k] = item_sum.get(k, 0.0) + v / n
    return {"seed0": seed0, "scores": scores, "clears": [bool(c) for c in clears], "steps": steps,
            "ending_each": ends, "score_median": float(np.median(scores)),
            "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
            "clear_rate": float(np.mean(clears)), "steps_mean": float(np.mean(steps)),
            "pellet_fraction_each": pellets, "pellet_fraction_mean": float(np.mean(pellets)), "dots_left_each": dots_left,
            "lives_lost_each": lives_lost, "lives_lost_mean": float(np.mean(lives_lost)),
            "blocked_actions_each": blocked, "blocked_action_rate": sum(blocked) / max(1, sum(steps)),
            "reward_items_mean": {k: round(v, 2) for k, v in item_sum.items()},
            # 得点の内訳 (2026-10-08 追加): 全試行の平均・クリアした試行だけの平均を分けて残す
            **{f"{k}_each": v for k, v in parts.items()}, **{f"{k}_mean": float(np.mean(v)) for k, v in parts.items()},
            "power_steps_each": power_steps,
            "power_first_step_mean": float(np.mean([p[0] for p in power_steps if p])) if any(power_steps) else None,
            "ghosts_per_power": (sum(parts["ghosts_eaten"]) / sum(parts["power_eaten"])) if sum(parts["power_eaten"]) else None,
            "score_mean_cleared": float(np.mean([s for s, c in zip(scores, clears) if c])) if any(clears) else None,
            "lives_lost_mean_cleared": float(np.mean([l for l, c in zip(lives_lost, clears) if c])) if any(clears) else None,
            "endings": {e: ends.count(e) / n for e in ("clear", "caught", "truncated")}}


def _run_job_with_fixed(job, fixed):
    """子プロセスで、親と同じ固定設定にしてから学習する"""
    FIXED.clear()
    FIXED.update(fixed)
    return run_job(job)


def run_jobs(jobs, workers):
    """終わっていない学習だけを並列で実行し、全部の結果を返す"""
    results, todo = {}, []
    for j in jobs:
        p = os.path.join(j["runs_dir"], job_id(j), "result.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("manifest") != experiment_manifest(j):
                raise ValueError(f"Cached result manifest mismatch: {p}")
            if j.get("save_model") and not os.path.exists(os.path.join(os.path.dirname(p), "model.pt")):
                raise ValueError(f"Completed result has no model: {p}")
            results[job_id(j)] = cached
        else:
            todo.append(j)
    if todo:
        print(f"  学習 {len(todo)} 本を実行 (済み {len(jobs) - len(todo)} 本, 並列 {workers})", flush=True)
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            # spawn (Windows) の子プロセスは FIXED を読み直すので、親の設定を明示的に渡す (fork と同じ結果にする)
            futs = {ex.submit(_run_job_with_fixed, j, copy.deepcopy(FIXED)): j for j in todo}
            for fu in cf.as_completed(futs):
                r = fu.result()
                results[r["id"]] = r
                print(f"  [{time.strftime('%m/%d %H:%M')}] 完了 {r['exp']} {r['label']} seed{r['seed']}: "
                      f"評価 {r['eval']['score_mean']:.0f} 点  クリア率 {r['eval']['clear_rate']:.0%}  "
                      f"({r['train_minutes']} 分)", flush=True)
    return [results[job_id(j)] for j in jobs]


# ════════════════════════════════════════════════════════════
#  実験の定義
# ════════════════════════════════════════════════════════════
def make_jobs(exp, conditions, args, save_model=True, episodes=None, seeds=None, lives=1):
    """conditions: [(ラベル, rewards, gamma, {変えた変数: 値})]。lives = 残機"""
    jobs = []
    for label, rewards, gamma, changed in conditions:
        for s in seeds or args.seeds:
            jobs.append({"exp": exp, "label": label, "rewards": rewards, "gamma": gamma, "changed": changed,
                         "episodes": episodes or args.episodes, "seed": s, "eval_episodes": args.eval_episodes,
                         "runs_dir": os.path.join(args.out, "runs"), "save_model": save_model, "lives": lives})
    return jobs


def summarize(results, conditions):
    """条件ごとに seed をまとめた数値"""
    out = []
    for label, rewards, gamma, changed in conditions:
        rs = [r for r in results if r["label"] == label]
        ev = [r["eval"]["score_mean"] for r in rs]
        out.append({"label": label, "changed": changed, "rewards": rewards, "gamma": gamma, "n": len(rs),
                    "eval_mean": float(np.mean(ev)), "eval_std": float(np.std(ev)),
                    "clear_rate": float(np.mean([r["eval"]["clear_rate"] for r in rs])),
                    "steps_mean": float(np.mean([r["eval"]["steps_mean"] for r in rs])), "ids": [r["id"] for r in rs]})
    return out


def best_of(summary):
    return max(summary, key=lambda s: (s["eval_mean"], s["clear_rate"]))


def with_reward(base, **kw):
    r = dict(base)
    r.update(kw)
    return r


def pso_rewards(base_rewards, pos):
    r = dict(base_rewards)
    for (name, lo, hi, _), v in zip(PSO_SPACE, pos):
        if name == "shaping":
            for k in SHAPING_ITEMS:
                r[k] = DESIGNED_REWARDS[k] * float(v)
        else:
            r[name] = float(v)
    return r


def pso_position(rewards):
    pos = []
    for name, lo, hi, _ in PSO_SPACE:
        if name == "shaping":
            k = SHAPING_ITEMS[0]
            v = rewards[k] / DESIGNED_REWARDS[k] if DESIGNED_REWARDS[k] else 1.0
        else:
            v = rewards[name]
        pos.append(min(max(v, lo), hi))
    return np.array(pos)


def run_pso(args, base_rewards, gamma):
    """PSO で報酬の値を探す。1世代ごとに状態を保存し、中断しても続きから"""
    from pso_optimizer import Particle
    P = dict(PSO_DEFAULT)
    if args.quick:
        P.update(particles=3, iterations=2, train_episodes=args.episodes, eval_episodes=args.eval_episodes)
    state_path = os.path.join(args.out, "pso_state.json")
    bounds = [(lo, hi) for _, lo, hi, _ in PSO_SPACE]
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            st = json.load(f)
        identity = experiment_manifest({"pso": P, "gamma": gamma, "base_rewards": base_rewards})
        if st.get("manifest") != identity:
            raise ValueError("PSO state code/config/runtime mismatch; use a new output directory")
    else:
        rng = np.random.default_rng(P["seed"])
        swarm = [Particle(bounds, rng) for _ in range(P["particles"])]
        swarm[0].pos = pso_position(base_rewards)   # 粒子0は単変量比較の最良から始める
        st = {"settings": P, "gamma": gamma, "base_rewards": base_rewards, "iteration": 0, "history": [],
              "manifest": experiment_manifest({"pso": P, "gamma": gamma, "base_rewards": base_rewards}),
              "particles": [{"pos": p.pos.tolist(), "vel": p.vel.tolist(), "pbest_pos": p.pos.tolist(),
                             "pbest": None} for p in swarm], "gbest_pos": None, "gbest": None}
    while st["iteration"] < P["iterations"]:
        t = st["iteration"]
        w = P["w_start"] - (P["w_start"] - P["w_end"]) * t / max(P["iterations"] - 1, 1)
        print(f"\n[PSO 世代 {t + 1}/{P['iterations']}]  w={w:.3f}", flush=True)
        conds = [(f"PSO 世代{t + 1} 粒子{i + 1}", pso_rewards(base_rewards, p["pos"]), gamma, {})
                 for i, p in enumerate(st["particles"])]
        jobs = make_jobs("exp4_pso", conds, args, save_model=False, episodes=P["train_episodes"], seeds=[P["seed"]])
        for j in jobs:
            j["eval_episodes"] = P["eval_episodes"]
        res = run_jobs(jobs, args.workers)
        rng = np.random.default_rng(P["seed"] * 1000 + t)   # 再開しても同じ乱数
        for p, r in zip(st["particles"], res):
            fit = r["eval"]["score_mean"]        # 適応度 F = 学習後の貪欲方策の平均スコア
            st["history"].append({"iteration": t + 1, "pos": p["pos"], "fitness": fit,
                                  "clear_rate": r["eval"]["clear_rate"], "id": r["id"]})
            if p["pbest"] is None or fit > p["pbest"]:
                p["pbest"], p["pbest_pos"] = fit, list(p["pos"])
            if st["gbest"] is None or fit > st["gbest"]:
                st["gbest"], st["gbest_pos"] = fit, list(p["pos"])
        for p in st["particles"]:
            part = Particle(bounds, rng)
            part.pos, part.vel = np.array(p["pos"]), np.array(p["vel"])
            part.pbest_pos = np.array(p["pbest_pos"])
            part.update_velocity(np.array(st["gbest_pos"]), w, P["c1"], P["c2"], rng)
            part.update_position()
            p["pos"], p["vel"] = part.pos.tolist(), part.vel.tolist()
        st["iteration"] = t + 1
        st.setdefault("gbest_by_iter", []).append(st["gbest"])
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
        print(f"  最良の適応度 {st['gbest']:.0f}", flush=True)
        report(args)
    return pso_rewards(base_rewards, st["gbest_pos"]), st


def main_pipeline(args):
    os.makedirs(args.out, exist_ok=True)
    chain = []   # 次の実験の基準をどう決めたか
    # 実験0 + 実験1 (互いに独立なので一緒に回す)
    c0 = [("前回 (old)", OLD_REWARDS, 0.99, {"報酬セット": "前回"}),
          ("トータル設計 (designed)", DESIGNED_REWARDS, 0.99, {"報酬セット": "トータル設計"})]
    c1 = [(f"捕獲ペナルティ −{v:,.0f}", with_reward(DESIGNED_REWARDS, death=v), 0.99, {"death": v})
          for v in (100.0, 500.0, 10.0 * N_PELLETS, 100.0 * N_PELLETS)]
    r01 = run_jobs(make_jobs("exp0", c0, args) + make_jobs("exp1", c1, args), args.workers)
    s1 = summarize(r01, c1)
    b1 = best_of(s1)
    chain.append({"from": "実験1", "chosen": {"death": b1["changed"]["death"]}, "reason":
                  f"評価スコアが最も高かった ({b1['eval_mean']:.0f} 点)"})
    report(args)
    base = with_reward(DESIGNED_REWARDS, death=b1["changed"]["death"])
    # 実験2
    c2 = [(f"クリアボーナス +{v:,.0f}", with_reward(base, clear=v), 0.99, {"clear": v})
          for v in (0.0, 500.0, 10.0 * N_PELLETS, 5000.0)]
    s2 = summarize(run_jobs(make_jobs("exp2", c2, args), args.workers), c2)
    b2 = best_of(s2)
    chain.append({"from": "実験2", "chosen": {"clear": b2["changed"]["clear"]}, "reason":
                  f"評価スコアが最も高かった ({b2['eval_mean']:.0f} 点)"})
    report(args)
    base = with_reward(base, clear=b2["changed"]["clear"])
    # 実験3
    c3 = [(f"γ = {g}", base, g, {"gamma": g}) for g in (0.95, 0.99, 0.995)]
    s3 = summarize(run_jobs(make_jobs("exp3", c3, args), args.workers), c3)
    b3 = best_of(s3)
    gamma = b3["changed"]["gamma"]
    chain.append({"from": "実験3", "chosen": {"gamma": gamma}, "reason": f"評価スコアが最も高かった ({b3['eval_mean']:.0f} 点)"})
    with open(os.path.join(args.out, "chain.json"), "w", encoding="utf-8") as f:
        json.dump({"chain": chain, "univariate_best": {"rewards": base, "gamma": gamma}}, f, ensure_ascii=False, indent=1)
    report(args)
    # 実験4 PSO
    pso_best, _ = run_pso(args, base, gamma)
    # 実験5 最終比較
    c5 = [("前回 (old)", OLD_REWARDS, 0.99, {"報酬": "前回"}),
          ("単変量比較の最良", base, gamma, {"報酬": "単変量比較の最良"}),
          ("PSO の最良", pso_best, gamma, {"報酬": "PSO の最良"})]
    run_jobs(make_jobs("exp5", c5, args), args.workers)
    report(args)
    # 実験6・7 残機 (実験0〜5 はどの条件もクリアできず、ほぼ全エピソードが1回の捕獲で終わったため)
    # 実験2 は1回もクリアしないので全条件が同じ結果になり、比較になっていなかった。
    # そこでクリアボーナスは設計どおり 10N に戻し、他は単変量比較の最良 (捕獲・γ) を使う
    base_l = with_reward(base, clear=10.0 * N_PELLETS)
    chain.append({"from": "実験6・7 の基準", "chosen": {"clear": 10.0 * N_PELLETS, "death": base_l["death"], "gamma": gamma},
                  "reason": "実験2 は全条件クリア 0 回で同じ結果だったため、クリアボーナスは設計値 10N に戻した"})
    with open(os.path.join(args.out, "chain.json"), "w", encoding="utf-8") as f:
        json.dump({"chain": chain, "univariate_best": {"rewards": base, "gamma": gamma},
                   "lives_base": {"rewards": base_l, "gamma": gamma}}, f, ensure_ascii=False, indent=1)
    c6 = [(f"残機 {n}", base_l, gamma, {"lives": n}) for n in (1, 3)]
    c7 = [(f"残機3 クリアボーナス +{v:,.0f}", with_reward(base_l, clear=v), gamma, {"clear": v})
          for v in (0.0, 500.0, 10.0 * N_PELLETS, 5000.0)]
    run_jobs(make_jobs("exp6", c6[:1], args, lives=1) + make_jobs("exp6", c6[1:], args, lives=3)
             + make_jobs("exp7", c7, args, lives=3), args.workers)
    report(args)
    print("\nすべての実験が終わりました:", args.out)


# ════════════════════════════════════════════════════════════
#  図表
# ════════════════════════════════════════════════════════════
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for c in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Yu Gothic", "Meiryo", "MS Gothic"):
        if c in names:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["font.size"] = 12
    return plt


def read_log(run_dir):
    p = os.path.join(run_dir, "log.csv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


def mavg(x, n=100):
    c = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - n)
    return (c[i] - c[lo]) / (i - lo)


def cond_line(rewards, gamma, episodes, seeds, lives=1):
    return (f"条件: 残機 {lives} / 捕獲 −{rewards['death']:,.0f} / クリア +{rewards['clear']:,.0f} / エサ +{rewards['dot']:g} / "
            f"γ={gamma} / 学習率 {FIXED['lr']} / ε 1.0→{FIXED['epsilon_end']} / {episodes:,}エピソード / seed {','.join(map(str, seeds))}")


def _text_width(x):
    """表示幅 (全角は2)"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(x))


def table_png(plt, path, header, rows, title, col_widths=None):
    widths = [max(_text_width(c) for c in col) + 2 for col in zip(header, *rows)]
    total = sum(widths)
    fig_w = max(8.0, total * 0.105)
    fig_h = 0.42 * (len(rows) + 1) + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1 - 0.55 / fig_h])
    ax.axis("off")
    t = ax.table(cellText=[list(map(str, r)) for r in rows], colLabels=header, loc="upper center",
                 cellLoc="center", colWidths=col_widths or [w / total for w in widths], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    for j in range(len(header)):
        t[0, j].set_facecolor("#dfe7f2")
    fig.suptitle(title, fontsize=13, y=1 - 0.12 / fig_h, va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt_num(v):
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def write_table(base, header, rows, title, plt):
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([header] + rows)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(f"## {title}\n\n| " + " | ".join(header) + " |\n|" + " --- |" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    table_png(plt, base + ".png", header, rows, title)


EXP_TITLE = {"exp0": "実験0 報酬セット (前回 vs トータル設計)", "exp1": "実験1 捕獲ペナルティ",
             "exp2": "実験2 クリアボーナス", "exp3": "実験3 割引率 γ", "exp6": "実験6 残機 (1 vs 3)",
             "exp7": "実験7 残機3でのクリアボーナス", "exp5": "実験5 最終比較 (前回 / 単変量の最良 / PSO の最良)"}
EXP_AIM = {"exp0": "報酬セット以外の条件をすべて固定し、報酬の設計を変えた効果だけを検証する",
           "exp1": "捕獲ペナルティ以外の条件をすべて固定し、罰の大きさが学習とQ値の振れ幅に与える影響だけを検証する",
           "exp2": "クリアボーナス以外の条件をすべて固定し、全エサを取る報酬の大きさの影響だけを検証する",
           "exp3": "割引率以外の条件をすべて固定し、γ がクリアボーナスの見え方 (γ^T) を通じて与える影響だけを検証する",
           "exp5": "学習の条件と seed をそろえ、報酬の決め方 (前回・単変量比較・PSO) の違いだけを比べる",
           "exp6": "残機以外の条件をすべて固定し、1回の捕獲で終わるかどうかがクリアまでの学習に与える影響だけを検証する",
           "exp7": "残機3で、クリアボーナス以外の条件をすべて固定し、全エサを取る報酬の大きさの影響だけを検証する"}


def report_experiment(plt, out, exp, results):
    rs = [r for r in results if r["exp"] == exp]
    if not rs:
        return None
    labels = list(dict.fromkeys(r["label"] for r in rs))
    d = os.path.join(out, "figures", exp)
    os.makedirs(d, exist_ok=True)
    runs = os.path.join(out, "runs")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    means, rows = {}, []
    for ci, lab in enumerate(labels):
        rr = [r for r in rs if r["label"] == lab]
        logs = [read_log(os.path.join(runs, r["id"])) for r in rr]
        logs = [lg for lg in logs if lg is not None]
        if not logs:
            continue
        L = min(len(lg["score"]) for lg in logs)
        sc = np.array([mavg(lg["score"])[:L] for lg in logs])
        means[lab] = (sc.mean(axis=0), np.array([lg["q_max"][:L] - lg["q_min"][:L] for lg in logs]).mean(axis=0), ci)
        # 1条件1枚
        fig, ax = plt.subplots(figsize=(11, 5.2))
        for s, lg in zip(rr, logs):
            ax.plot(lg["episode"][:L], mavg(lg["score"])[:L], color=colors[ci % 10], alpha=0.35, lw=1,
                    label=f"seed {s['seed']}")
        ax.plot(np.arange(1, L + 1), sc.mean(axis=0), color=colors[ci % 10], lw=2.6, label="seed の平均")
        ax.set_xlabel("エピソード")
        ax.set_ylabel("スコア (直近100エピソードの移動平均)")
        ax.set_title(f"{EXP_TITLE.get(exp, exp)}: {lab}\n{cond_line(rr[0]['rewards'], rr[0]['gamma'], rr[0]['episodes'], [r['seed'] for r in rr], rr[0].get('lives', 1))}",
                     fontsize=11)
        ax.axhline(1720, color="gray", ls=":", lw=1)
        ax.text(0.99, 1720, "全エサ 1,720 点 ", color="gray", fontsize=9, ha="right", va="bottom",
                transform=ax.get_yaxis_transform())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"cond{ci + 1}.png"), dpi=150)
        plt.close(fig)
        last = [np.nanmean(lg["score"][-100:]) for lg in logs]
        qr = [lg["q_max"][-1] - lg["q_min"][-1] for lg in logs]
        loss = [np.nanmean(lg["loss"][-100:]) for lg in logs]
        ev = [r["eval"]["score_mean"] for r in rr]
        rows.append([lab, f"{np.mean(ev):,.0f} ± {np.std(ev):,.0f}",
                     f"{np.mean([r['eval']['clear_rate'] for r in rr]):.0%}",
                     f"{np.mean(last):,.0f}", f"{np.mean([r['eval']['steps_mean'] for r in rr]):,.0f}",
                     f"{np.mean(qr):,.0f}", f"{np.mean(loss):.3f}", len(rr)])
    if not means:
        return None
    # 重ね合わせ
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for lab, (m, _, ci) in means.items():
        ax.plot(np.arange(1, len(m) + 1), m, color=colors[ci % 10], lw=2.2, label=lab)
    ax.set_xlabel("エピソード")
    ax.set_ylabel("スコア (直近100エピソードの移動平均, seed の平均)")
    ax.set_title(f"{EXP_TITLE.get(exp, exp)}: 全条件の重ね合わせ\n{EXP_AIM.get(exp, '')}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "overlay.png"), dpi=150)
    plt.close(fig)
    # Q値の振れ幅
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for lab, (_, q, ci) in means.items():
        ax.plot(np.arange(1, len(q) + 1), q, color=colors[ci % 10], lw=2, label=lab)
    ax.set_xlabel("エピソード")
    ax.set_ylabel("Q値の振れ幅 (最大 − 最小, 報酬の単位)")
    ax.set_title(f"{EXP_TITLE.get(exp, exp)}: Q値の振れ幅\n(固定した256状態での max_a Q の最大と最小の差。罰やボーナスを変えると Q のスケールがどう変わるか)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "q_range.png"), dpi=150)
    plt.close(fig)
    header = ["条件", "評価スコア (平均±標準偏差)", "クリア率", "直近100平均 (学習中)", "平均ステップ",
              "Q値の振れ幅", "損失 (直近100)", "試行数"]
    write_table(os.path.join(d, "results"), header, rows,
                f"{EXP_TITLE.get(exp, exp)}: 結果まとめ (評価 = 学習後の貪欲方策で{rs[0]['eval'].get('n', len(rs[0]['eval']['scores']))}回プレイ)", plt)
    return rows


def report_pso(plt, out):
    p = os.path.join(out, "pso_state.json")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    d = os.path.join(out, "figures", "exp4_pso")
    os.makedirs(d, exist_ok=True)
    H = st["history"]
    P = st["settings"]
    # 収束
    fig, ax = plt.subplots(figsize=(11, 5))
    it = [h["iteration"] for h in H]
    ax.scatter(it, [h["fitness"] for h in H], color="tab:gray", alpha=0.6, label="各粒子の適応度")
    g = st.get("gbest_by_iter", [])
    ax.plot(range(1, len(g) + 1), g, "o-", color="tab:red", lw=2.4, label="群れの最良 (gbest)")
    first0 = [h for h in H if h["iteration"] == 1][:1]
    if first0:
        ax.axhline(first0[0]["fitness"], color="tab:blue", ls="--", label="単変量比較の最良 (世代1 粒子1)")
    ax.set_xlabel("世代")
    ax.set_ylabel("適応度 F = 学習後の平均スコア")
    ax.set_title(f"実験4 PSO の収束\n条件: 粒子 {P['particles']} / 世代 {P['iterations']} / w {P['w_start']}→{P['w_end']} / "
                 f"c1={P['c1']} c2={P['c2']} / 1粒子 {P['train_episodes']}エピソード学習 + {P['eval_episodes']}回評価 / γ={st['gamma']}",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(d, "convergence.png"), dpi=150)
    plt.close(fig)
    # 変数ごとの粒子の位置
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for k, ((name, lo, hi, ja), ax) in enumerate(zip(PSO_SPACE, axes.flat)):
        ax.scatter(it, [h["pos"][k] for h in H], c=[h["fitness"] for h in H], cmap="viridis", s=30)
        if st.get("gbest_pos"):
            ax.axhline(st["gbest_pos"][k], color="tab:red", lw=1.5)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{ja} ({name})  範囲 {lo:g}〜{hi:g}", fontsize=10)
        ax.set_xlabel("世代")
        ax.grid(alpha=0.3)
    fig.suptitle("実験4 PSO: 各変数の粒子の位置 (色 = 適応度, 赤線 = 最良)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "positions.png"), dpi=150)
    plt.close(fig)
    # 設定の表
    rows = [[f"{ja} ({n})", f"{lo:g} 〜 {hi:g}", f"{st['base_rewards'].get(n, 1.0) if n != 'shaping' else 1.0:g}",
             (f"{st['gbest_pos'][k]:.3g}" if st.get("gbest_pos") else "-")] for k, (n, lo, hi, ja) in enumerate(PSO_SPACE)]
    write_table(os.path.join(d, "variables"), ["探索する変数", "範囲", "出発点 (単変量の最良)", "PSO の最良"], rows,
                "実験4 PSO: 探索する変数と結果", plt)
    rows = [["粒子数", P["particles"], "群れの大きさ。並列数 (8) に合わせて1世代を1回の並列実行で評価"],
            ["世代数 (反復回数)", P["iterations"], "粒子数 × 世代数 = 学習回数。計算時間の上限から決めた"],
            ["慣性重み w", f"{P['w_start']} → {P['w_end']} (線形に減らす)", "最初は広く探し、後半は最良の周りを細かく探す"],
            ["c1 (自分の最良へ)", P["c1"], "Kennedy & Eberhart の標準的な値"], ["c2 (群れの最良へ)", P["c2"], "同上。c1 と同じにして偏らせない"],
            ["適応度 F", "学習後の貪欲方策の平均スコア", f"{P['train_episodes']}エピソード学習 → 盤面を固定した{P['eval_episodes']}回のプレイ"],
            ["1粒子の学習 seed", P["seed"], "全粒子で同じにして、報酬の違いだけを比べる"],
            ["速度の上限", "範囲の 20%", "一度に範囲の端まで飛ばないように"]]
    write_table(os.path.join(d, "settings"), ["PSO の設定", "値", "理由"], [list(map(str, r)) for r in rows],
                "実験4 PSO の設定", plt)


def report_rewards(plt, out, results):
    """報酬表 (出来事 / 値 / 与える条件) と、前回・単変量の最良・PSO の最良の比較、得点の内訳"""
    d = os.path.join(out, "figures", "rewards")
    os.makedirs(d, exist_ok=True)
    sets = [("前回", OLD_REWARDS), ("トータル設計", DESIGNED_REWARDS)]
    cp = os.path.join(out, "chain.json")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            sets.append(("単変量比較の最良", json.load(f)["univariate_best"]["rewards"]))
    sp = os.path.join(out, "pso_state.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("gbest_pos"):
            sets.append(("PSO の最良", pso_rewards(st["base_rewards"], st["gbest_pos"])))
    rows = []
    for k in REWARD_SPEC:
        sign = "−" if REWARD_SIGN[k] < 0 else "+"
        rows.append([NAMES_JA[k].replace(" (罰)", ""), CONDITION_JA[k]] +
                    [f"{sign}{fmt_num(v[k])}" if v[k] else "0" for _, v in sets])
    write_table(os.path.join(d, "reward_table"), ["出来事", "与える条件"] + [n for n, _ in sets], rows,
                "報酬表 (罰は − で表示。PSO が選んだ値も同じ行に並べる)", plt)
    # 報酬の内訳 (実験5 の評価プレイで、1エピソードあたりどの報酬をいくら得たか)
    rs = [r for r in results if r["exp"] == "exp5"]
    if rs:
        labels = list(dict.fromkeys(r["label"] for r in rs))
        fig, ax = plt.subplots(figsize=(12, 5.5))
        x = np.arange(len(labels))
        pos_b, neg_b = np.zeros(len(labels)), np.zeros(len(labels))
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, k in enumerate(REWARD_SPEC):
            v = np.array([np.mean([r["eval"]["reward_items_mean"].get(k, 0.0) for r in rs if r["label"] == lab])
                          for lab in labels])
            if not np.any(v):
                continue
            b = np.where(v >= 0, pos_b, neg_b)
            ax.bar(x, v, bottom=b, color=cmap[i % 20], label=NAMES_JA[k])
            pos_b += np.where(v >= 0, v, 0)
            neg_b += np.where(v < 0, v, 0)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("1エピソードあたりの報酬")
        ax.set_title("実験5: 評価プレイで得た報酬の内訳 (各報酬セットで学習したモデル)", fontsize=12)
        ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(d, "reward_breakdown.png"), dpi=150)
        plt.close(fig)


def report_design(plt, out):
    """スライドの「状態・行動・報酬の設計」「学習設定」に使う数え上げと計算"""
    d = os.path.join(out, "figures", "design")
    os.makedirs(d, exist_ok=True)
    env = PacManEnv(render_mode=None, num_ghosts=FIXED["num_ghosts"])
    env.reset(seed=0)
    R, C = env.rows, env.cols
    write_table(os.path.join(d, "state"), ["状態の要素", "何を表すか", "数"], [
        ["盤面 4チャネル", "壁 / エサ / パワーエサ / ゴースト の有無 (22×19 マスごと)", f"{R}×{C}×4 = {R * C * 4:,}"],
        ["パックマンの位置", "行・列 (0〜1 に正規化)", "2"], ["ゴーストの位置", "行・列 × ゴースト1体", "2"],
        ["最寄りのエサまでの距離", "BFS 距離", "1"], ["ゴーストまでの距離", "1体分", "1"],
        ["最寄りのエサの方向", "上下左右", "4"], ["パワーの残り時間", "0〜1", "1"],
        ["行動別の特徴", "4方向 × (通路か / エサの近さ / 危険なゴーストの近さ / 弱いゴーストの近さ)", "16"],
        ["合計 (ネットワークの入力)", "", f"{env.observation_space.shape[0]:,}"]],
        "状態の構成 (DQN の入力ベクトル)", plt)
    T_min = 214
    rows = []
    for g in (0.95, 0.99, 0.995):
        rows.append([g] + [f"{g ** T:.3f}" for T in (100, T_min, 300, 500, 1000)] +
                    [f"{10 * N_PELLETS * g ** T_min:,.0f}"])
    write_table(os.path.join(d, "gamma_T"), ["γ", "γ^100", f"γ^{T_min} (最短クリア)", "γ^300", "γ^500", "γ^1000 (上限)",
                                             f"クリア +{10 * N_PELLETS:,} の開始時点の価値"], rows,
                "割引率とクリアまでのステップ数 T: γ^T (クリアボーナスが割引で見えなくならないか)", plt)
    rows = [["lr", "学習率 α (Adam)", FIXED["lr"]], ["gamma", "割引率 γ", "0.99 (実験3で比較)"],
            ["epsilon_start → end", "探索率 ε (最初 → 最後)", f"{FIXED['epsilon_start']} → {FIXED['epsilon_end']}"],
            ["epsilon_decay", "ε の減衰", f"エピソードごとに掛ける。全体の{int(FIXED['explore_ratio'] * 100)}%で下限に届く値"],
            ["batch_size", "1回の学習で使う経験の数", FIXED["batch_size"]], ["buffer", "経験再生バッファ", f"{FIXED['buffer']:,}"],
            ["target_update", "ターゲットネットワークの更新間隔", f"{FIXED['target_update']:,} 学習ステップ"],
            ["network", "ネットワーク", f"入力 {env.observation_space.shape[0]:,} → 512 → 256 (共有) → 頭ごとに 128 → 4 (ReLU)"],
            ["heads", "HRA の頭", f"{N_REWARD_COMPONENTS}つ ({' / '.join(REWARD_COMPONENT_NAMES)})。行動は3つの頭の Q の合計で選ぶ"],
            ["q_scale", "Q値のスケール", "報酬をエサ1個の報酬で割って学習 (エサ1個 = 1)。Q の初期値はネットワークの初期値 (≈0)"],
            ["loss", "損失 / 更新式", "Double DQN, Huber 損失, 勾配を 1.0 で切る"],
            ["max_steps", "1エピソードの上限", f"{FIXED['max_steps']:,} ステップ (または全エサ・捕獲で終了)"],
            ["actions", "行動", "4 (上下左右)。壁に向かうとその場に留まる"]]
    write_table(os.path.join(d, "hyperparameters"), ["変数名", "日本語名", "値"], [list(map(str, r)) for r in rows],
                "ハイパーパラメータ一覧 (全実験で固定。γ は実験3でのみ変える)", plt)
    facts = {"rows": R, "cols": C, "pellets": 152, "power_pellets": 4, "N": N_PELLETS,
             "score_all_pellets": 152 * 10 + 4 * 50, "score_all_pellets_plus_ghost_each_power": 152 * 10 + 4 * 50 + 4 * 200,
             "steps_shortest_clear_no_ghost": T_min, "reward_total_dots_designed": 10 * 152,
             "note": "最短クリアは BFS で最寄りのエサを順に取る動き (ゴーストなし) の実測値"}
    with open(os.path.join(d, "facts.json"), "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)


def report_endings(plt, out, results):
    """エピソードの終わり方 (クリア / 捕獲 / 1,000ステップで打ち切り) を条件ごとに集計する"""
    groups = {}
    for r in results:
        if r["exp"] == "exp4_pso":
            continue
        lg = read_log(os.path.join(out, "runs", r["id"]))
        if lg is None:
            continue
        st, cl = lg["steps"][-100:], lg["cleared"][-100:]
        tr_trunc = (cl == 0) & (st >= FIXED["max_steps"])
        tr = (cl.mean(), ((cl == 0) & ~tr_trunc).mean(), tr_trunc.mean(), st.mean())
        ev = r["eval"]
        if "endings" in ev:
            e = ev["endings"]
            evs = (e["clear"], e["caught"], e["truncated"])
        else:   # 残機を記録する前の結果: 捕獲の報酬から捕まった割合を求める
            caught = -ev["reward_items_mean"].get("death", 0.0) / r["rewards"]["death"] if r["rewards"]["death"] else 0.0
            evs = (ev["clear_rate"], caught, max(0.0, 1 - ev["clear_rate"] - caught))
        groups.setdefault((r["exp"], r["label"], r.get("lives", 1)), []).append(tr + evs + (ev["steps_mean"],))
    rows = []
    for (exp, label, lives), v in sorted(groups.items()):
        m = np.mean(np.array(v), axis=0)
        rows.append([EXP_TITLE.get(exp, exp).split(" ")[0], label, lives, f"{m[0]:.0%}", f"{m[1]:.0%}", f"{m[2]:.0%}",
                     f"{m[3]:.0f}", f"{m[4]:.0%}", f"{m[5]:.0%}", f"{m[6]:.0%}", f"{m[7]:.0f}"])
    header = ["実験", "条件", "残機", "学習: クリア", "学習: 捕獲で終了", "学習: 打ち切り", "学習: 平均ステップ",
              "評価: クリア", "評価: 捕獲で終了", "評価: 打ち切り", "評価: 平均ステップ"]
    write_table(os.path.join(out, "figures", "endings"), header, rows,
                "エピソードの終わり方 (学習 = 直近100エピソード / 評価 = 学習後の貪欲方策。打ち切り = 1,000ステップの上限)", plt)


def report(args):
    plt = _plt()
    out = args.out
    results = []
    runs = os.path.join(out, "runs")
    if os.path.isdir(runs):
        for r in os.listdir(runs):
            p = os.path.join(runs, r, "result.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    results.append(json.load(f))
    report_design(plt, out)
    for exp in ("exp0", "exp1", "exp2", "exp3", "exp5", "exp6", "exp7"):
        report_experiment(plt, out, exp, results)
    report_pso(plt, out)
    report_rewards(plt, out, results)
    report_endings(plt, out, results)
    write_readme(out, results)


def write_readme(out, results):
    lines = ["# 報酬設計の実験 (自動生成)", "",
             "評価指標: **学習後の貪欲方策 (ランダムな行動なし) で、盤面の乱数を固定して複数回プレイした平均スコア**。"
             "補助指標: クリア率、学習中の直近100エピソード平均、平均ステップ数、Q値の振れ幅、損失。", ""]
    cp = os.path.join(out, "chain.json")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            ch = json.load(f)
        lines += ["## 次の実験の基準値の決め方", ""]
        for c in ch["chain"]:
            lines.append(f"- {c['from']}: {c['chosen']} を採用 ({c['reason']})")
        lines.append("")
    lines += ["## ファイル", "",
              "- `figures/design/` 状態の構成・γ^T・ハイパーパラメータ一覧・数え上げ (facts.json)",
              "- `figures/rewards/` 報酬表 (前回 / トータル設計 / 単変量の最良 / PSO の最良)、報酬の内訳",
              "- `figures/exp0`〜`exp3`, `exp5` 条件ごとのグラフ (cond*.png)、重ね合わせ (overlay.png)、Q値の振れ幅 (q_range.png)、結果表 (results.*)",
              "- `figures/exp4_pso/` PSO の収束・各変数の位置・設定表・変数と結果",
              "- `runs/` 学習1本ごとのログ (log.csv) と結果 (result.json)、モデル (model.pt)", "",
              f"完了した学習: {len(results)} 本 ({time.strftime('%Y-%m-%d %H:%M')} 時点)"]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    p.add_argument("--episodes", type=int, default=1000, help="単変量比較・最終比較の1本あたりの学習エピソード数")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--quick", action="store_true", help="動作確認 (数エピソード)")
    a = p.parse_args()
    if a.quick:
        a.episodes, a.eval_episodes, a.seeds = 4, 2, [101, 202]
    if a.report_only:
        report(a)
    else:
        main_pipeline(a)


if __name__ == "__main__":
    main()
