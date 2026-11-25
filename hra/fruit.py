"""
RM-13 / M1: 論文 4.1節 フルーツ収集課題で HRA 本体を検証する (論文の図3を再現)

環境 (公式実装 github.com/Maluuba/hra の mini モードと同じ):
  10x10 格子、フルーツ候補地 10 か所のうち 5 か所にランダム配置、エージェントは候補地以外のランダムな位置から開始。
  フルーツを食べると +1。全部食べるか 300 ステップで終了。
  入力 = エージェント位置 one-hot(100) + 各候補地にフルーツがあるか(10) = 110

手法 (図3):
  dqn    : 頭10個のネットワークを合計Q値で学習 (式2)
  dqn+1  : 同上 + 各頭に無関係な特徴を除いた入力 (位置 + 自分のフルーツ)
  hra    : 頭ごとに自分の報酬で学習 (式6)
  hra+1  : hra + 無関係な特徴の除去
  hra+2  : hra+1 + 終端状態の明示 → 表現が one-hot になるので表形式 (論文4.1節)
  hra+3  : hra+2 + 擬似報酬の GVF (各候補地へ行く GVF を、フルーツがなくても学習)
  ネットワークは公式実装どおり 頭ごとに隠れ層 250/10=25 ユニット、RMSprop(lr=1e-3, rho=0.95)。

学習データはランダム方策で集め、定期的に貪欲方策を評価して「全部食べるまでのステップ数」を記録する。

  python -m hra.fruit --methods dqn hra hra+1 --episodes 5000
  python -m hra.fruit --methods hra+1 hra+2 hra+3 --episodes 200
"""
import argparse
import csv
import os

import numpy as np

W = H = 10
FRUITS = [(0, 0), (0, 9), (1, 2), (3, 6), (4, 4), (5, 7), (6, 2), (7, 7), (8, 8), (9, 0)]
N_FRUIT_LOC = len(FRUITS)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
MAX_STEPS = 300


class FruitCollection:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        on = self.rng.choice(N_FRUIT_LOC, 5, replace=False)
        self.fruit = np.zeros(N_FRUIT_LOC, dtype=np.float32)
        self.fruit[on] = 1
        while True:
            self.pos = (int(self.rng.integers(H)), int(self.rng.integers(W)))
            if self.pos not in FRUITS:
                break
        self.t = 0
        return self.obs()

    def obs(self):
        o = np.zeros(W * H + N_FRUIT_LOC, dtype=np.float32)
        o[self.pos[0] * W + self.pos[1]] = 1
        o[W * H:] = self.fruit
        return o

    def step(self, a):
        dr, dc = MOVES[a]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < H and 0 <= c < W:
            self.pos = (r, c)
        rewards = np.zeros(N_FRUIT_LOC, dtype=np.float32)   # 分解された報酬 (候補地ごと)
        if self.pos in FRUITS:
            k = FRUITS.index(self.pos)
            if self.fruit[k]:
                self.fruit[k] = 0
                rewards[k] = 1
        self.t += 1
        done = self.fruit.sum() == 0 or self.t >= MAX_STEPS
        return self.obs(), rewards, done


# ── 表形式 (hra+2, hra+3) ──
class TabularHRA:
    """
    hra+2: 頭k = 「フルーツkがある間」の Q_k(位置, 行動)。フルーツを食べたら終端 (価値0)
    hra+3: GVF_k(位置, 行動) = 候補地kへ行く価値。フルーツの有無と無関係に毎ステップ全GVFを更新し、
           頭k = フルーツkがあれば GVF_k、なければ 0
    """

    def __init__(self, gvf, alpha, gamma, mean, rng):
        self.gvf, self.alpha, self.gamma, self.mean, self.rng = gvf, alpha, gamma, mean, rng
        self.Q = np.zeros((N_FRUIT_LOC, W * H, 4))

    def q(self, obs):
        p = int(np.argmax(obs[:W * H]))
        return (self.Q[:, p, :] * obs[W * H:, None]).sum(axis=0)

    def learn(self, obs, a, rewards, nxt, done):
        p, p2 = int(np.argmax(obs[:W * H])), int(np.argmax(nxt[:W * H]))
        boot = self.Q[:, p2, :].mean(axis=1) if self.mean else self.Q[:, p2, :].max(axis=1)
        if self.gvf:
            reached = np.array([FRUITS[k] == divmod(p2, W) for k in range(N_FRUIT_LOC)], dtype=float)
            y = reached + (1 - reached) * self.gamma * boot
            self.Q[:, p, a] += self.alpha * (y - self.Q[:, p, a])
        else:
            active = obs[W * H:] > 0              # フルーツがない頭は終端 (学習しない)
            term = rewards > 0                    # 食べた → その頭は終端
            y = rewards + (1 - term) * self.gamma * boot
            self.Q[active, p, a] += self.alpha * (y[active] - self.Q[active, p, a])


# ── ネットワーク (dqn, dqn+1, hra, hra+1) ──
class DeepHRA:
    def __init__(self, hra, remove_features, mean, gamma, lr, rng, device="cpu",
                 batch=32, replay=10000, replay_min=1000, learn_every=4, target_every=100):
        import torch
        import torch.nn as nn
        self.torch, self.nn = torch, nn
        self.hra, self.remove, self.mean, self.gamma = hra, remove_features, mean, gamma
        self.rng = rng
        torch.manual_seed(int(rng.integers(1 << 31)))
        in_dim = W * H + 1 if remove_features else W * H + N_FRUIT_LOC

        def head():
            m = nn.Sequential(nn.Linear(in_dim, 250 // N_FRUIT_LOC), nn.ReLU(), nn.Linear(250 // N_FRUIT_LOC, 4))
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)
            return m
        self.net = nn.ModuleList([head() for _ in range(N_FRUIT_LOC)]).to(device)
        self.tgt = nn.ModuleList([head() for _ in range(N_FRUIT_LOC)]).to(device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.RMSprop(self.net.parameters(), lr=lr, alpha=0.95, eps=1e-7)
        self.device = device
        self.buf = []
        self.cap, self.bmin, self.batch = replay, replay_min, batch
        self.learn_every, self.target_every = learn_every, target_every
        self.steps = self.updates = 0

    def _heads(self, net, x):
        torch = self.torch
        if self.remove:
            pos = x[:, :W * H]
            outs = [h(torch.cat([pos, x[:, W * H + k:W * H + k + 1]], dim=1)) for k, h in enumerate(net)]
        else:
            outs = [h(x) for h in net]
        return torch.stack(outs, dim=1)          # [B, 10, 4]

    def q(self, obs):
        with self.torch.no_grad():
            x = self.torch.as_tensor(obs[None], device=self.device)
            return self._heads(self.net, x).sum(dim=1)[0].cpu().numpy()

    def learn(self, obs, a, rewards, nxt, done):
        self.buf.append((obs, a, rewards, nxt, float(done)))
        if len(self.buf) > self.cap:
            self.buf.pop(0)
        self.steps += 1
        if len(self.buf) < self.bmin or self.steps % self.learn_every:
            return
        torch = self.torch
        idx = self.rng.integers(len(self.buf), size=self.batch)
        o, a_, r, o2, d = zip(*(self.buf[i] for i in idx))
        o = torch.as_tensor(np.array(o), device=self.device)
        o2 = torch.as_tensor(np.array(o2), device=self.device)
        a_ = torch.as_tensor(a_, device=self.device)
        r = torch.as_tensor(np.array(r), device=self.device)
        d = torch.as_tensor(d, device=self.device, dtype=torch.float32)
        qs = self._heads(self.net, o)                                   # [B,10,4]
        with torch.no_grad():
            q2 = self._heads(self.tgt, o2)
        if self.hra:                     # 式(6): 頭ごとの損失
            pred = qs.gather(2, a_.view(-1, 1, 1).expand(-1, N_FRUIT_LOC, 1)).squeeze(2)
            boot = q2.mean(2) if self.mean else q2.max(2).values
            y = r + (1 - d[:, None]) * self.gamma * boot
        else:                            # 式(2): 合計Q値で1本の損失
            pred = qs.sum(1).gather(1, a_.view(-1, 1)).squeeze(1)
            q2s = q2.sum(1)
            boot = q2s.mean(1) if self.mean else q2s.max(1).values
            y = r.sum(1) + (1 - d) * self.gamma * boot
        loss = ((y - pred) ** 2).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.tgt.load_state_dict(self.net.state_dict())


# 手法ごとの設定。mean=True は式(8)。論文: DQN は式(7)、HRA は式(8) が最良
METHODS = {
    "dqn":   dict(kind="deep", hra=False, remove=False, mean=False),
    "dqn+1": dict(kind="deep", hra=False, remove=True, mean=False),
    "hra":   dict(kind="deep", hra=True, remove=False, mean=True),
    "hra+1": dict(kind="deep", hra=True, remove=True, mean=True),
    "hra+2": dict(kind="tab", gvf=False, mean=True),
    "hra+3": dict(kind="tab", gvf=True, mean=True),
}


def make_agent(method, rng, gamma=0.99, lr=1e-3, alpha=None, device="cpu"):
    m = METHODS[method]
    if m["kind"] == "tab":
        return TabularHRA(m["gvf"], alpha if alpha is not None else 1.0, gamma, m["mean"], rng)
    return DeepHRA(m["hra"], m["remove"], m["mean"], gamma, lr, rng, device=device)


def greedy_steps(agent, rng, n):
    env = FruitCollection(rng)
    total = 0
    for _ in range(n):
        obs, done = env.reset(), False
        while not done:
            q = agent.q(obs)
            a = int(rng.choice(np.flatnonzero(q >= q.max() - 1e-9)))
            obs, _, done = env.step(a)
        total += env.t
    return total / n


def run(method, episodes, seed=0, eval_every=10, eval_eps=20, out_csv=None, **kw):
    rng = np.random.default_rng(seed)
    env = FruitCollection(rng)
    agent = make_agent(method, rng, **kw)
    ev_rng = np.random.default_rng(seed + 1)
    curve = []
    for ep in range(1, episodes + 1):
        obs, done = env.reset(), False
        while not done:                    # 学習データはランダム方策で集める
            a = int(rng.integers(4))
            nxt, r, done = env.step(a)
            agent.learn(obs, a, r, nxt, done)
            obs = nxt
        if ep % eval_every == 0 or ep == 1:
            steps = greedy_steps(agent, ev_rng, eval_eps)
            curve.append((ep, steps))
            print(f"[{method}] ep {ep:5d}  greedy steps {steps:6.1f}")
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "steps"])
            w.writerows(curve)
    return curve


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in curves.items():
        ax.plot([e for e, _ in c], [s for _, s in c], label=name)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps")
    ax.set_ylim(0, 310)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Fruit collection (paper Fig. 3)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["hra+1", "hra+2", "hra+3"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=None, help="表形式の学習率 (省略時 1.0)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--out", default="runs/fruit")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    curves = {}
    for m in args.methods:
        curves[m] = run(m, args.episodes, args.seed, args.eval_every,
                        out_csv=os.path.join(args.out, f"{m}.csv"),
                        gamma=args.gamma, lr=args.lr, alpha=args.alpha, device=args.device)
    png = os.path.join(args.out, "fruit_" + "_".join(args.methods).replace("+", "p") + ".png")
    plot(curves, png)
    print("保存:", png)


if __name__ == "__main__":
    main()
