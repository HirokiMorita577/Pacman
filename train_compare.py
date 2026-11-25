"""
DQN vs SARSA 比較訓練スクリプト
使い方:
  python train_compare.py                    # 各5000ep で比較
  python train_compare.py --episodes 10000   # 各10000ep で比較
"""
import argparse
import time
import numpy as np
from pacman_env import PacManEnv
from dqn_agent import DQNAgent
from sarsa_agent import SARSAAgent
from plot_curve import plot_comparison

DEFAULT_LR            = 0.002056
DEFAULT_EPSILON_END   = 0.0248
DEFAULT_EPSILON_DECAY = 0.99454
DEFAULT_GAMMA         = 0.9294


def run_dqn(args):
    env = PacManEnv(render_mode=None, ghost_mode=args.ghost_mode)
    agent = DQNAgent(
        obs_size=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        lr=args.lr, gamma=args.gamma,
        epsilon_start=1.0, epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=128, target_update_freq=1000,
    )
    scores = []
    best = 0
    start = time.time()
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.push(obs, action, reward, next_obs, float(terminated or truncated))
            agent.train_step()
            obs = next_obs
            if terminated or truncated:
                break
        scores.append(info["score"])
        if info["score"] > best:
            best = info["score"]
            agent.save("compare_dqn.pt")
        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            elapsed = int(time.time() - start)
            print(f"  [DQN  ] Ep {ep:5d} | Avg={avg:6.0f} | Best={best:5d} | {elapsed//60:02d}:{elapsed%60:02d}")
    env.close()
    return scores


def run_sarsa(args):
    env = PacManEnv(render_mode=None, ghost_mode=args.ghost_mode)
    agent = SARSAAgent(
        obs_size=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        lr=args.lr, gamma=args.gamma,
        epsilon_start=1.0, epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=128,
    )
    scores = []
    best = 0
    start = time.time()
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        action = agent.select_action(obs, training=True)
        while True:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_obs, training=True)
            agent.push(obs, action, reward, next_obs, next_action, float(done))
            agent.train_step()
            obs, action = next_obs, next_action
            if done:
                break
        scores.append(info["score"])
        if info["score"] > best:
            best = info["score"]
            agent.save("compare_sarsa.pt")
        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            elapsed = int(time.time() - start)
            print(f"  [SARSA] Ep {ep:5d} | Avg={avg:6.0f} | Best={best:5d} | {elapsed//60:02d}:{elapsed%60:02d}")
    env.close()
    return scores


def main(args):
    print(f"\n{'='*60}")
    print(f"  DQN vs SARSA 比較訓練  ({args.episodes} ep each)")
    print(f"  ゴーストAI: {args.ghost_mode}")
    print(f"{'='*60}\n")

    print("--- DQN 訓練開始 ---")
    dqn_scores = run_dqn(args)

    print("\n--- SARSA 訓練開始 ---")
    sarsa_scores = run_sarsa(args)

    plot_comparison(
        {"DQN": dqn_scores, "SARSA": sarsa_scores},
        window=100,
        save_path="comparison.png",
    )

    print(f"\n{'='*60}")
    print(f"  DQN   最高: {max(dqn_scores):5d}  平均: {np.mean(dqn_scores):.0f}")
    print(f"  SARSA 最高: {max(sarsa_scores):5d}  平均: {np.mean(sarsa_scores):.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int,   default=5000)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma",         type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPSILON_DECAY)
    parser.add_argument("--epsilon-end",   type=float, default=DEFAULT_EPSILON_END)
    parser.add_argument("--ghost-mode",    type=str,   default="classic",
                        choices=["classic", "bfs"])
    args = parser.parse_args()
    main(args)
