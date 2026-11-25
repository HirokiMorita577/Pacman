"""
Pac-Man PPO 訓練スクリプト
使い方:
  python train_ppo.py                        # 3,000,000ステップ
  python train_ppo.py --total-steps 5000000  # ステップ数指定
  python train_ppo.py --resume               # 前回の続きから
"""
import argparse
import csv
import os
import subprocess
import sys
import time
import numpy as np
from pacman_env import PacManEnv
from ppo_agent import PPOAgent
from plot_curve import plot_learning_curve

# PPOデフォルトハイパーパラメータ
DEFAULT_LR          = 3e-4
DEFAULT_GAMMA       = 0.99
DEFAULT_GAE_LAMBDA  = 0.95
DEFAULT_CLIP_EPS    = 0.2
DEFAULT_ENTROPY     = 0.02   # 探索促進のため少し高め
DEFAULT_VALUE_COEF  = 0.5
DEFAULT_N_STEPS     = 2048   # ロールアウト長
DEFAULT_N_EPOCHS    = 4
DEFAULT_BATCH_SIZE  = 256
DEFAULT_TOTAL_STEPS = 3_000_000


def _record_best(model_path, score, step, args):
    out = f"best_{score}_step{step}.mp4"
    play_cmd = [
        "python", "play_ppo.py",
        "--model", model_path,
        "--record",
        "--speed", "0.5",
        "--episodes", "1",
        "--out", out,
        "--ghost-mode", args.ghost_mode,
    ]
    try:
        env = os.environ.copy()
        if sys.platform == "linux" and "DISPLAY" not in env:
            import shutil
            if shutil.which("xvfb-run"):
                cmd = ["xvfb-run", "-a"] + play_cmd
            else:
                print("  [録画スキップ] xvfb-run が見つかりません")
                return
        else:
            cmd = play_cmd
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        if result.returncode == 0:
            print(f"  [録画完了] {out}  (score={score})")
        else:
            print(f"  [録画失敗] {result.stderr.strip()[:200]}")
    except Exception as e:
        print(f"  [録画スキップ] {e}")


def train(args):
    env = PacManEnv(render_mode=None, ghost_mode=args.ghost_mode)
    obs_size  = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(
        obs_size    = obs_size,
        n_actions   = n_actions,
        lr          = args.lr,
        gamma       = args.gamma,
        gae_lambda  = args.gae_lambda,
        clip_eps    = args.clip_eps,
        entropy_coef = args.entropy,
        value_coef  = args.value_coef,
        n_steps     = args.n_steps,
        n_epochs    = args.n_epochs,
        batch_size  = args.batch_size,
    )

    model_path = args.model
    if args.resume and os.path.exists(model_path):
        agent.load(model_path)
        print(f"前回モデル読込: {model_path}")
    elif args.resume:
        print(f"[警告] {model_path} が見つかりません。新規訓練を開始します。")

    csv_path = os.path.splitext(model_path)[0] + "_log.csv"
    csv_is_new = not os.path.exists(csv_path) or not args.resume
    csv_file = open(csv_path, "w" if csv_is_new else "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(["episode", "score", "total_steps",
                             "ghost_mode", "lr", "gamma", "entropy"])

    print(f"\n{'='*60}")
    print(f"  Pac-Man PPO 訓練")
    print(f"  総ステップ数  : {args.total_steps:,}")
    print(f"  ロールアウト  : {args.n_steps} steps")
    print(f"  エポック数    : {args.n_epochs}")
    print(f"  バッチサイズ  : {args.batch_size}")
    print(f"  学習率        : {args.lr}")
    print(f"  gamma         : {args.gamma}")
    print(f"  clip_eps      : {args.clip_eps}")
    print(f"  entropy_coef  : {args.entropy}")
    print(f"  ゴーストAI    : {args.ghost_mode}")
    print(f"  モデル保存先  : {model_path}")
    print(f"{'='*60}\n")

    scores = []
    best_score = 0
    ep_count = 0
    start_time = time.time()
    save_interval_steps = max(args.n_steps * 100, 200_000)
    last_save_step = 0

    obs, _ = env.reset()
    ep_score = 0

    while agent.total_steps < args.total_steps:
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.push(obs, action, reward, float(done), log_prob, value)
        ep_score += reward
        obs = next_obs

        if done:
            ep_count += 1
            scores.append(info["score"])
            csv_writer.writerow([ep_count, info["score"], agent.total_steps,
                                 args.ghost_mode, args.lr, args.gamma, args.entropy])
            if info["score"] > best_score:
                best_score = info["score"]
                agent.save(model_path)
                if args.record_best:
                    _record_best(model_path, best_score, agent.total_steps, args)
            ep_score = 0
            obs, _ = env.reset()

        # バッファが満杯になったらPPO更新
        if agent.buffer.full():
            update_stats = agent.update(obs)

            if ep_count > 0 and ep_count % args.log_interval == 0:
                avg = np.mean(scores[-100:]) if scores else 0
                bar = "#" * min(int(avg / 30), 25)
                elapsed = time.time() - start_time
                h, m = divmod(int(elapsed), 3600)
                m, s = divmod(m, 60)
                pct = agent.total_steps / args.total_steps * 100
                print(
                    f"Ep {ep_count:6d} | Step {agent.total_steps:8,d} ({pct:4.1f}%) | "
                    f"Avg100={avg:6.0f} {bar:<25} | "
                    f"Best={best_score:5d} | "
                    f"H={update_stats['entropy']:.3f} | "
                    f"{h:02d}:{m:02d}:{s:02d}"
                )

            if agent.total_steps - last_save_step >= save_interval_steps:
                mid_path = f"ppo_model_step{agent.total_steps}.pt"
                agent.save(mid_path)
                plot_learning_curve(scores, save_path="ppo_learning_curve.png")
                print(f"  [中間保存] {mid_path}")
                last_save_step = agent.total_steps

    csv_file.close()
    env.close()

    elapsed = time.time() - start_time
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n訓練完了!  {h:02d}:{m:02d}:{s:02d}")
    print(f"総エピソード : {ep_count}")
    print(f"ベストスコア : {best_score}")
    print(f"モデル保存   : {model_path}")

    plot_learning_curve(scores, save_path="ppo_learning_curve.png")
    print("学習曲線保存 : ppo_learning_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pac-Man PPO 訓練")
    parser.add_argument("--total-steps",  type=int,   default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--model",        type=str,   default="pacman_ppo.pt")
    parser.add_argument("--log-interval", type=int,   default=100,
                        help="何エピソードごとにログを出力するか")
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma",        type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gae-lambda",   type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument("--clip-eps",     type=float, default=DEFAULT_CLIP_EPS)
    parser.add_argument("--entropy",      type=float, default=DEFAULT_ENTROPY)
    parser.add_argument("--value-coef",   type=float, default=DEFAULT_VALUE_COEF)
    parser.add_argument("--n-steps",      type=int,   default=DEFAULT_N_STEPS)
    parser.add_argument("--n-epochs",     type=int,   default=DEFAULT_N_EPOCHS)
    parser.add_argument("--batch-size",   type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ghost-mode",   type=str,   default="classic",
                        choices=["classic", "bfs"])
    parser.add_argument("--record-best",  action="store_true")
    args = parser.parse_args()
    train(args)
