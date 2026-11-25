"""
訓練スクリプト（じっくり長期訓練用）
使い方:
  python train.py                      # 20000ep 新規訓練
  python train.py --episodes 50000     # エピソード数指定
  python train.py --resume             # 前回の続きから
  python train.py --resume --episodes 10000  # 続きから追加10000ep

--resume の動作:
  <model>_latest.pt がある場合 → 中断地点から再開し、合計 --episodes に達するまで学習
                                 (Colab のセッション切れ対策。--checkpoint-interval ごとに保存)
  ない場合                     → <model> を読み込み、さらに --episodes だけ追加学習
"""
import argparse
import csv
import json
import os
import subprocess
import time
import numpy as np
from pacman_env import PacManEnv, find_nearest_dot, DIRS, N_REWARD_COMPONENTS
from dqn_agent import DQNAgent, HRADQNAgent
from plot_curve import plot_learning_curve


def _push(agent, obs, action, reward, next_obs, done, info):
    """HRAなら要素別報酬ベクトル、通常DQNなら合計報酬をバッファに積む"""
    r = info["reward_components"] if isinstance(agent, HRADQNAgent) else reward
    agent.push(obs, action, r, next_obs, float(done))


def _expert_action(env):
    """
    貪欲BFS専門家ポリシー: 最近傍のエサへの次ステップを返す。
    Phase1の模倣学習で「クリア」を経験させるために使う。
    """
    pos = tuple(env.pacman)
    dot_pos, path = find_nearest_dot(env.grid, pos)
    if dot_pos is None or len(path) < 2:
        return int(env.rng.integers(4))
    nr, nc = path[1]
    dr, dc = nr - pos[0], nc - pos[1]
    for i, (dy, dx) in enumerate(DIRS):
        if dy == dr and dx == dc:
            return i
    return int(env.rng.integers(4))


def _run_expert_episode(env, agent):
    """
    専門家ポリシーで1エピソード実行し、遷移をreplay bufferに積む。
    戻り値: スコア
    """
    obs, _ = env.reset()
    while True:
        action = _expert_action(env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        _push(agent, obs, action, reward, next_obs, done, info)
        agent.train_step()
        obs = next_obs
        if done:
            break
    return info["score"]


def _record_best(model_path, score, ep, args):
    """ベストスコア更新時に0.5倍速でプレイを録画する"""
    out = f"best_{score}_ep{ep}.mp4"
    play_cmd = [
        "python", "play.py",
        "--model", model_path,
        "--record",
        "--speed", "0.5",
        "--episodes", "1",
        "--out", out,
        "--ghost-mode", args.ghost_mode,
    ]
    try:
        import sys
        env = os.environ.copy()
        if sys.platform == "linux" and "DISPLAY" not in env:
            # Colab/Linux: xvfb-run で仮想ディスプレイを使う
            import shutil
            if shutil.which("xvfb-run"):
                cmd = ["xvfb-run", "-a"] + play_cmd
            else:
                print("  [録画スキップ] xvfb-run が見つかりません。'apt-get install xvfb' を実行してください")
                return
        else:
            cmd = play_cmd
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        if result.returncode == 0:
            print(f"  [録画完了] {out}  (score={score}, ep={ep})")
        else:
            print(f"  [録画失敗] {result.stderr.strip()[:200]}")
    except Exception as e:
        print(f"  [録画スキップ] {e}")

# 旧PSO値 (lr=0.002056, gamma=0.9294, decay=0.99454/step) は短い試行で最適化されたもので、
# 長期訓練では「εが約700ステップ(1〜2エピソード)で下限に到達」「γが短期的すぎる(視野≈14歩)」
# ためスコアが頭打ちになっていた。
DEFAULT_LR            = 5e-4
DEFAULT_EPSILON_END   = 0.0248
DEFAULT_GAMMA         = 0.99
EPSILON_EXPLORE_RATIO = 0.4   # 全エピソードのこの割合で ε が下限に到達するよう減衰率を自動設定


def _checkpoint_paths(model_path):
    base = os.path.splitext(model_path)[0]
    return base + "_latest.pt", base + "_latest.json"


def _save_checkpoint(model_path, agent, episodes_done, best_score):
    """再開用の最新状態を保存 (モデル + 進捗)"""
    ckpt_path, meta_path = _checkpoint_paths(model_path)
    agent.save(ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"episodes_done": episodes_done, "best_score": best_score}, f)


def _run_phase(args, phase, total_eps, model_path, agent, no_ghosts,
               scores_all, losses_all, best_score, start_time, csv_writer, ep_offset,
               expert_ratio=0.0, start_ep=1, csv_file=None):
    """カリキュラム学習の1フェーズを実行。(best_score, ep_offset) を返す"""
    env = PacManEnv(render_mode=None, ghost_mode=args.ghost_mode, no_ghosts=no_ghosts,
                    num_ghosts=args.num_ghosts)
    save_interval = max(500, total_eps // 20)
    label = f"[Phase{phase}:{'NoGhost' if no_ghosts else 'Ghost'}]"
    if expert_ratio > 0:
        label += f"(Expert {int(expert_ratio*100)}%)"
    print(f"\n{'='*60}")
    print(f"  {label}  {total_eps} エピソード" + (f" ({start_ep} から再開)" if start_ep > 1 else ""))
    print(f"{'='*60}\n")

    expert_clears = 0

    for ep in range(start_ep, total_eps + 1):
        # 専門家エピソード（模倣学習）
        if no_ghosts and expert_ratio > 0 and np.random.random() < expert_ratio:
            score = _run_expert_episode(env, agent)
            scores_all.append(score)
            if score >= 2400:  # 全餌クリア
                expert_clears += 1
            csv_writer.writerow([ep + ep_offset, score, f"{agent.epsilon:.6f}",
                                 args.ghost_mode, args.lr, args.gamma, args.epsilon_decay,
                                 f"phase{phase}_expert"])
            if score > best_score:
                best_score = score
                agent.save(model_path)
        else:
            # 通常エピソード
            obs, _ = env.reset()
            ep_losses = []
            while True:
                action = agent.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                _push(agent, obs, action, reward, next_obs, done, info)
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)
                obs = next_obs
                if done:
                    break

            scores_all.append(info["score"])
            if ep_losses:
                losses_all.append(np.mean(ep_losses))
            csv_writer.writerow([ep + ep_offset, info["score"], f"{agent.epsilon:.6f}",
                                 args.ghost_mode, args.lr, args.gamma, args.epsilon_decay,
                                 f"phase{phase}"])
            if info["score"] > best_score:
                best_score = info["score"]
                agent.save(model_path)
                if args.record_best:
                    _record_best(model_path, best_score, ep + ep_offset, args)

        agent.end_episode()

        if ep % args.log_interval == 0:
            avg  = np.mean(scores_all[-100:])
            bar  = "#" * min(int(avg / 30), 25)
            elapsed = time.time() - start_time
            h, m = divmod(int(elapsed), 3600)
            m, s = divmod(m, 60)
            extra = f" ExpertClears={expert_clears}" if expert_clears > 0 else ""
            print(
                f"Ep {ep:6d}/{total_eps} | "
                f"Avg100={avg:6.0f} {bar:<25} | "
                f"Best={best_score:5d} | "
                f"e={agent.epsilon:.4f} | "
                f"{h:02d}:{m:02d}:{s:02d} {label}{extra}"
            )

        if ep % save_interval == 0:
            mid_path = f"pacman_model_ep{ep + ep_offset}.pt"
            agent.save(mid_path)
            plot_learning_curve(scores_all, save_path="learning_curve.png")
            print(f"  [中間保存] {mid_path}  learning_curve.png 更新")

        if ep % args.checkpoint_interval == 0 or ep == total_eps:
            if csv_file is not None:
                csv_file.flush()
            _save_checkpoint(model_path, agent, ep + ep_offset, best_score)

    env.close()
    return best_score, ep_offset + total_eps


def train(args):
    tmp_env = PacManEnv(render_mode=None, num_ghosts=args.num_ghosts)
    obs_size  = tmp_env.observation_space.shape[0]
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    if args.epsilon_decay is None:
        explore_eps = max(1, int(args.episodes * EPSILON_EXPLORE_RATIO))
        args.epsilon_decay = args.epsilon_end ** (1.0 / explore_eps)

    agent_kwargs = dict(
        lr            = args.lr,
        gamma         = args.gamma,
        epsilon_start = 1.0 if not args.resume else args.epsilon_end,
        epsilon_end   = args.epsilon_end,
        epsilon_decay = args.epsilon_decay,
        batch_size    = 128,
        target_update_freq = 1000,
        decay_per_episode  = True,
    )
    if args.no_hra:
        agent = DQNAgent(obs_size, n_actions, **agent_kwargs)
    else:
        agent = HRADQNAgent(obs_size, n_actions, N_REWARD_COMPONENTS, **agent_kwargs)

    model_path = args.model
    ckpt_path, meta_path = _checkpoint_paths(model_path)
    episodes_done = 0
    best_score    = 0
    if args.resume and os.path.exists(ckpt_path) and os.path.exists(meta_path):
        agent.load(ckpt_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        episodes_done = meta["episodes_done"]
        best_score    = meta["best_score"]
        print(f"チェックポイント読込: {ckpt_path}  ({episodes_done}/{args.episodes} ep 完了)")
    elif args.resume and os.path.exists(model_path):
        agent.load(model_path)
        print(f"前回モデル読込: {model_path}")
    elif args.resume:
        print(f"[警告] {model_path} が見つかりません。新規訓練を開始します。")

    scores, losses = [], []
    start_time  = time.time()

    # CSVログのパスとヘッダー
    csv_path = os.path.splitext(model_path)[0] + "_log.csv"
    header = ["episode", "score", "epsilon",
              "ghost_mode", "lr", "gamma", "epsilon_decay", "phase"]
    rows = []
    if episodes_done > 0 and os.path.exists(csv_path):
        # チェックポイント以降の(やり直しになる)行を捨て、学習曲線用にスコアを復元
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)][1:]
        rows = [r for r in rows if r and int(r[0]) <= episodes_done]
        scores = [int(r[1]) for r in rows]
    csv_is_new = not os.path.exists(csv_path) or not args.resume
    if episodes_done > 0:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)
    else:
        csv_file = open(csv_path, "w" if csv_is_new else "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if csv_is_new:
            csv_writer.writerow(header)

    print(f"\n{'='*60}")
    print(f"  Pac-Man DQN カリキュラム学習" if args.curriculum else "  Pac-Man DQN 長期訓練")
    print(f"  エピソード数  : {args.episodes}")
    print(f"  学習率        : {args.lr}")
    print(f"  gamma         : {args.gamma}")
    print(f"  epsilon_decay : {args.epsilon_decay}")
    print(f"  epsilon_end   : {args.epsilon_end}")
    print(f"  ゴーストAI    : {args.ghost_mode}  ({args.num_ghosts}体)")
    print(f"  エージェント  : {'Double DQN' if args.no_hra else 'HRA (Hybrid Reward Architecture) + Double DQN'}")
    print(f"  モデル保存先  : {model_path}")
    if args.curriculum:
        p1 = int(args.episodes * args.curriculum_ratio)
        print(f"  Phase1(NoGhost): {p1} ep  Phase2(Ghost): {args.episodes - p1} ep")
    print(f"{'='*60}\n")

    # (phase番号, エピソード数, ゴーストなし, 専門家割合)
    if args.curriculum:
        p1_eps = int(args.episodes * args.curriculum_ratio)
        phases = [(1, p1_eps, True, args.expert_ratio),
                  (2, args.episodes - p1_eps, False, 0.0)]
    else:
        phases = [(1, args.episodes, False, 0.0)]

    ep_offset = 0
    for phase, n_eps, no_ghosts, expert_ratio in phases:
        already = min(n_eps, max(0, episodes_done - ep_offset))
        if already >= n_eps:   # 再開時: 完了済みのフェーズは飛ばす
            ep_offset += n_eps
            continue
        if phase == 2 and already == 0:
            # Phase2開始時にepsilonをリセット（ゴーストへの対応を再探索）
            agent.epsilon = max(0.3, args.epsilon_end)
            print(f"\n  [Phase2開始] epsilon を {agent.epsilon:.2f} にリセット\n")
        best_score, ep_offset = _run_phase(
            args, phase, n_eps, model_path, agent, no_ghosts=no_ghosts,
            scores_all=scores, losses_all=losses, best_score=best_score,
            start_time=start_time, csv_writer=csv_writer, ep_offset=ep_offset,
            expert_ratio=expert_ratio, start_ep=already + 1, csv_file=csv_file)

    csv_file.close()
    print(f"ログ保存       : {csv_path}")

    elapsed = time.time() - start_time
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n訓練完了!  {h:02d}:{m:02d}:{s:02d}")
    print(f"ベストスコア : {best_score}")
    print(f"モデル保存   : {model_path}")

    plot_learning_curve(scores, save_path="learning_curve.png")
    print("学習曲線保存 : learning_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pac-Man DQN 長期訓練")
    parser.add_argument("--episodes",       type=int,   default=20000)
    parser.add_argument("--resume",         action="store_true", help="前回の続きから")
    parser.add_argument("--model",          type=str,   default="pacman_model.pt")
    parser.add_argument("--log-interval",   type=int,   default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=200,
                        help="再開用チェックポイント(<model>_latest.pt)の保存間隔(エピソード)")
    parser.add_argument("--lr",             type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma",          type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon-decay",  type=float, default=None,
                        help="1エピソードごとの ε 減衰率 (省略時: 全体の40%%で下限に達するよう自動計算)")
    parser.add_argument("--num-ghosts",     type=int,   default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--no-hra",         action="store_true",
                        help="HRAを使わず通常のDouble DQNで学習 (比較用)")
    parser.add_argument("--epsilon-end",    type=float, default=DEFAULT_EPSILON_END)
    parser.add_argument("--ghost-mode",     type=str,   default="classic",
                        choices=["classic", "bfs"],
                        help="ゴーストAI: classic=本家4体行動(デフォルト), bfs=全員直接追跡")
    parser.add_argument("--record-best",    action="store_true",
                        help="ベストスコア更新時に0.5倍速で自動録画")
    parser.add_argument("--curriculum",     action="store_true",
                        help="カリキュラム学習: Phase1=ゴーストなし, Phase2=ゴーストあり")
    parser.add_argument("--curriculum-ratio", type=float, default=0.1,
                        help="Phase1(ゴーストなし)の割合 (デフォルト: 0.1 = 10%%)")
    parser.add_argument("--expert-ratio",   type=float, default=0.5,
                        help="Phase1で専門家(A*貪欲)エピソードを混ぜる割合 (デフォルト: 0.5 = 50%%)")
    args = parser.parse_args()
    train(args)
