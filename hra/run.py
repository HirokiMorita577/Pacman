"""
HRA Ms. Pac-Man の学習・評価 (FR-40〜FR-44, NFR-01, NFR-02, RM-20, RM-21)

使い方:
  python -m hra.run train --config hra/config.yaml --out runs --sync /content/drive/MyDrive/HRA_runs
  python -m hra.run eval  --config hra/config.yaml --out runs --mode fixed --video
  python -m hra.run train --set aggregator.type=linear --set exploration.count_based=false

  --out  : 学習中に書き込む場所 (Colab ではローカルディスク /content/... を推奨)
  --sync : 定期的に成果物をコピーする場所 (Google Drive)。再開時は --out になければここから読む
"""
import argparse
import copy
import csv
import os
import pickle
import shutil
import time
import hashlib
from pathlib import Path

import numpy as np
import yaml

from .agent import HRAAgent
from .env import MsPacmanEnv

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_FIELDS = ["episode", "score", "score_avg100", "frames", "steps", "level_reached", "levels_passed",
              "r_pellet", "r_power", "r_fruit", "r_blue_ghost", "r_ghost_contact",
              "gvfs", "states", "seconds"]


# ── 設定 (NFR-04) ──
def load_config(path=None, overrides=()):
    with open(path or os.path.join(HERE, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def run_name(cfg):
    """FR-43: 設定ごとに結果を分ける"""
    if cfg["experiment"]["name"] != "auto":
        return cfg["experiment"]["name"]
    ex = cfg["exploration"]
    parts = [cfg["aggregator"]["type"],
             "div" if ex["diversification"] else "nodiv",
             "cnt" if ex["count_based"] else "nocnt",
             "eq8" if cfg["gvf"]["target"] == "expected_sarsa" else "eq7",
             cfg["env"].get("position_source", "pixels")]
    if cfg["env"].get("step_mode", "frames") == "cell":
        parts.append("cell")
    if cfg["executive_memory"]["enabled"]:
        parts.append("memory")
    return "_".join(parts)


# ── 報酬分解のログ (FR-11) ──
def classify_reward(raw, prev_objs):
    """1ステップの得点を、どのオブジェクトから得たかに分ける (ログ用)"""
    out = {"r_pellet": 0.0, "r_power": 0.0, "r_fruit": 0.0, "r_blue_ghost": 0.0}
    rest = raw
    if rest <= 0:
        return out
    # 1ステップ(8フレーム)で食べられるエサは高々数個なので、端数 (10/20/30...) をエサ、
    # 50 を含むならパワーエサ、残りの大きな値を青ゴースト/フルーツに割り当てる
    small = rest % 100
    if small >= 50 and prev_objs.power_pellets:
        out["r_power"] += 50
        small -= 50
    out["r_pellet"] += small
    rest -= out["r_power"] + out["r_pellet"]
    if rest > 0:
        if any(b is not None for b in prev_objs.blue_ghosts):
            out["r_blue_ghost"] += rest
        else:
            out["r_fruit"] += rest
    return out


# ── 1エピソード ──
def play_episode(env, agent, learn=True, noops=0, start_actions=None, frames_out=None,
                 observer=None, max_frames=None):
    objs = env.reset(noops=noops)
    if start_actions is not None:           # FR-41 人間の軌跡の途中から開始
        for a in start_actions:
            objs, _, _, done, _ = env.step(int(a))
            if done:
                objs = env.reset()
    rec = {k: 0.0 for k in ("score", "r_pellet", "r_power", "r_fruit", "r_blue_ghost", "r_ghost_contact")}
    t = 0
    level, level_step, clean, level_actions = 1, 0, True, []
    levels_passed = 0
    started = False     # 開始直後の待ち時間 (Ms. Pac-Man が動けない) はGVFを更新しない
    s = agent.state(objs)
    done = False
    while not done:
        mem_a = agent.memory_action(level, level_step, clean)
        a = agent.act(objs, s, t, mem_a)
        if observer is not None:
            observer.decision(env, agent, objs, s, t)
        before_frames = env.frames
        nxt, raw, shaped, done, info = env.step(a)
        if observer is not None:
            observer.observe(objs, nxt, raw, info, env.frames - before_frames)
        s_next = agent.state(nxt)
        if objs.pacman is not None and nxt.pacman is not None and nxt.pacman != objs.pacman:
            started = True
        valid = (started and not info["life_lost"] and not info["level_cleared"]
                 and not info["in_transition"] and nxt.maze == objs.maze)
        if learn and valid:
            agent.learn(objs.maze, s, a, s_next)
        # 記録
        rec["score"] += raw
        for k, v in classify_reward(raw, objs).items():
            rec[k] += v
        if info["life_lost"]:
            rec["r_ghost_contact"] += env.ghost_penalty
            clean = False
            started = False
        level_actions.append(a)
        level_step += 1
        if info["level_cleared"]:
            levels_passed += 1
            if clean and learn:
                agent.remember(level, level_actions)       # FR-34
            level, level_step, clean, level_actions = info["level"], 0, True, []
        if frames_out is not None:
            frames_out.append(env.ale.getScreenRGB())
        objs, s = nxt, s_next
        t += 1
        if max_frames is not None and env.frames >= max_frames and not done:
            rec['truncated'] = True
            break
    rec.update(frames=env.frames, steps=t, level_reached=level, levels_passed=levels_passed)
    if max_frames is not None:
        rec.setdefault('truncated', False)
    if observer is not None:
        rec.update(observer.summary())
    return rec


# ── 保存と再開 (NFR-01, RM-21) ──
def _atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def sync_to(out_dir, sync_dir):
    """成果物 (チェックポイント・CSV・学習曲線・動画) を Google Drive などにコピー"""
    if not sync_dir:
        return
    os.makedirs(sync_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        src = os.path.join(out_dir, name)
        if os.path.isfile(src) and not name.endswith(".tmp"):
            dst = os.path.join(sync_dir, name)
            shutil.copy2(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)


def restore_from(out_dir, sync_dir):
    """--out にチェックポイントがなく --sync にあれば、そちらから戻す"""
    ck = os.path.join(out_dir, "checkpoint.pkl")
    if os.path.exists(ck) or not sync_dir or not os.path.exists(os.path.join(sync_dir, "checkpoint.pkl")):
        return
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(sync_dir):
        src = os.path.join(sync_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
    print(f"[再開] {sync_dir} から学習状態を復元しました")


def train(cfg, out_root, sync_root=None):
    name = run_name(cfg)
    out_dir = os.path.join(out_root, name)
    sync_dir = os.path.join(sync_root, name) if sync_root else None
    restore_from(out_dir, sync_dir)
    os.makedirs(out_dir, exist_ok=True)
    ex = cfg["experiment"]
    seed = ex["seed"]
    env = MsPacmanEnv(cfg, seed=seed)
    agent = HRAAgent(cfg, seed=seed)
    ck_path = os.path.join(out_dir, "checkpoint.pkl")
    csv_path = os.path.join(out_dir, "log.csv")
    start_ep, scores = 0, []
    if os.path.exists(ck_path):
        with open(ck_path, "rb") as f:
            ck = pickle.load(f)
        if 'config' in ck:
            for section in ('env', 'gvf', 'heads', 'aggregator', 'exploration', 'executive_memory'):
                if ck['config'][section] != cfg[section]:
                    raise ValueError(f'Resume configuration mismatch: {section}; use a separate experiment')
            if ck['config']['experiment']['seed'] != seed:
                raise ValueError('Resume seed mismatch')
        else:
            print('[注意] 旧モデルに設定記録がありません。過去の条件は未検証です。')
        agent.load_state_dict(ck["agent"])
        env.seed = ck["env_seed"]
        env.load_sites(ck["maze_colors"], ck["pellet_sites"])
        start_ep = ck["episode"]
        # チェックポイントより後のログ行 (やり直しになる分) を捨てる
        rows = []
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if int(r["episode"]) <= start_ep]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        scores = [float(r["score"]) for r in rows]
        print(f"[再開] {start_ep} エピソード目から ({name})")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"[新規] {name}")

    def checkpoint(ep):
        _atomic_pickle({"agent": agent.state_dict(), "episode": ep, "env_seed": env.seed,
                        "config": copy.deepcopy(cfg),
                        "provenance": {'legacy_prefix_unverified': start_ep > 0 and
                                       ('config' not in ck or ck.get('provenance', {}).get('legacy_prefix_unverified', False)),
                                       'code_sha256': {n: hashlib.sha256((Path(HERE) / n).read_bytes()).hexdigest()
                                                       for n in ('agent.py', 'env.py', 'run.py')}},
                        "maze_colors": env.maze_colors, "pellet_sites": env.pellet_sites},
                       ck_path)
        from .plot import save_learning_curve
        save_learning_curve(csv_path, os.path.join(out_dir, "learning_curve.png"), title=name)
        sync_to(out_dir, sync_dir)

    for ep in range(start_ep + 1, ex["episodes"] + 1):
        t0 = time.time()
        rec = play_episode(env, agent, learn=True)
        scores.append(rec["score"])
        st = agent.stats()
        row = {k: rec.get(k, 0) for k in CSV_FIELDS}
        row.update(episode=ep, score_avg100=np.mean(scores[-100:]), gvfs=st["gvfs"], states=st["states"],
                   seconds=round(time.time() - t0, 2))
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        if ep % ex["log_every"] == 0 or ep == 1:
            print(f"ep {ep:5d} | score {rec['score']:7.0f} | avg100 {row['score_avg100']:8.1f} | "
                  f"level {rec['level_reached']:3d} | frames {rec['frames']:6d} | GVF {st['gvfs']:4d} | "
                  f"{row['seconds']:.1f}s")
        if ep % ex["checkpoint_every"] == 0 or ep == ex["episodes"]:
            checkpoint(ep)
    env.close()
    return out_dir


def evaluate(cfg, out_root, mode="fixed", episodes=None, video=False, sync_root=None):
    """FR-40 固定開始 / FR-41 ランダム開始。GVFは更新しない"""
    name = run_name(cfg)
    out_dir = os.path.join(out_root, name)
    restore_from(out_dir, os.path.join(sync_root, name) if sync_root else None)
    with open(os.path.join(out_dir, "checkpoint.pkl"), "rb") as f:
        ck = pickle.load(f)
    ev = cfg["evaluation"]
    rng = np.random.default_rng(cfg["experiment"]["seed"] + 999)
    env = MsPacmanEnv(cfg, seed=cfg["experiment"]["seed"] + 10_000)
    env.load_sites(ck["maze_colors"], ck["pellet_sites"])
    agent = HRAAgent(cfg, seed=cfg["experiment"]["seed"] + 999)
    agent.load_state_dict(ck["agent"])
    human = None
    if mode == "random":
        if not ev["human_starts"]:
            raise SystemExit("ランダム開始評価には evaluation.human_starts (人間の行動列 .npy) が必要です (U-08)")
        human = np.load(ev["human_starts"])
    results = []
    for i in range(episodes or ev["episodes"]):
        frames = [] if (video and i == 0) else None
        if mode == "fixed":
            rec = play_episode(env, agent, learn=False, noops=int(rng.integers(0, ev["noop_max"] + 1)),
                               frames_out=frames)
        else:
            k = int(rng.integers(0, len(human)))
            rec = play_episode(env, agent, learn=False, start_actions=human[:k], frames_out=frames)
        results.append(rec)
        print(f"[{mode}] {i + 1}: score {rec['score']:.0f}  level {rec['level_reached']}")
        if frames:
            save_video(frames, os.path.join(out_dir, f"eval_{mode}_ep{ck['episode']}.mp4"))
    scores = [r["score"] for r in results]
    summary = {"mode": mode, "trained_episodes": ck["episode"], "mean": float(np.mean(scores)),
               "std": float(np.std(scores)), "max": float(np.max(scores)),
               "levels": [r["level_reached"] for r in results]}
    with open(os.path.join(out_dir, f"eval_{mode}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True)
    print(summary)
    sync_to(out_dir, os.path.join(sync_root, name) if sync_root else None)
    return summary


def save_video(frames, path, fps=15):
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps)
        print(f"動画保存: {path}")
    except Exception as e:  # imageio-ffmpeg がない環境など
        print(f"[動画スキップ] {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "eval"])
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[], help="設定の上書き 例: aggregator.type=linear")
    p.add_argument("--out", default="runs")
    p.add_argument("--sync", default=None, help="成果物を定期的にコピーする場所 (Google Drive)")
    p.add_argument("--mode", choices=["fixed", "random"], default="fixed")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    if args.command == "train":
        if args.episodes:
            cfg["experiment"]["episodes"] = args.episodes
        train(cfg, args.out, args.sync)
    else:
        evaluate(cfg, args.out, args.mode, args.episodes, args.video, args.sync)


if __name__ == "__main__":
    main()
