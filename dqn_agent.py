"""
DQN Agent for Pac-Man
Double DQN + Experience Replay + Target Network
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import gzip


class DQNNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class HRANetwork(nn.Module):
    """
    Hybrid Reward Architecture (van Seijen et al., 2017) 用ネットワーク。
    共有の特徴抽出部 + 報酬要素ごとのQ値ヘッド。forward は全ヘッドの合計Q値を返すので
    行動選択・Q値表示は通常のDQNと同じ扱いができる。
    """
    def __init__(self, obs_size, n_actions, n_heads):
        super().__init__()
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))
            for _ in range(n_heads)
        ])

    def head_q(self, x):
        """要素別Q値: shape [batch, n_heads, n_actions]"""
        h = self.trunk(x)
        return torch.stack([head(h) for head in self.heads], dim=1)

    def forward(self, x):
        return self.head_q(x).sum(dim=1)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_size, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=1000, decay_per_episode=False,
                 buffer_capacity=100000, device=None, action_mask_indices=None):
        """
        decay_per_episode=True : epsilon を end_episode() ごとに減衰
                                 (False は従来通り学習ステップごと。
                                  1エピソード数百ステップあるので数エピソードで探索が終わる点に注意)
        """
        self.n_actions = n_actions
        self.action_mask_indices = tuple(action_mask_indices) if action_mask_indices is not None else None
        if self.action_mask_indices is not None and (
                len(self.action_mask_indices) != n_actions or
                any(i < 0 or i >= obs_size for i in self.action_mask_indices)):
            raise ValueError("action_mask_indices must contain one observation index per action")
        self.decay_per_episode = decay_per_episode
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.device = torch.device(device) if device is not None else self._get_device()
        print(f"Using device: {self.device}")

        self.policy_net = self._build_net(obs_size, n_actions).to(self.device)
        self.target_net = self._build_net(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_history = []

    def _build_net(self, obs_size, n_actions):
        return DQNNetwork(obs_size, n_actions)

    @staticmethod
    def _get_device():
        """CUDA → DirectML (Intel/AMD GPU) → CPU の優先順で選択"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            dml = torch_directml.device()
            # 動作確認: 小さいテンソルを作れるか試す
            torch.tensor([1.0]).to(dml)
            return dml
        except Exception:
            pass
        return torch.device("cpu")

    def get_qvalues(self, state):
        """現在の状態のQ値をnumpy配列で返す"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t)
        return q.squeeze(0).cpu().numpy()

    def select_action(self, state, training=True):
        legal = np.arange(self.n_actions)
        if self.action_mask_indices is not None:
            allowed = np.flatnonzero(np.asarray(state)[list(self.action_mask_indices)] > 0.5)
            if len(allowed):
                legal = allowed
        if training and random.random() < self.epsilon:
            return int(legal[random.randrange(len(legal))])
        q_values = self.get_qvalues(state)
        return int(legal[np.argmax(q_values[legal])])

    def _next_actions(self, next_states_t):
        """Use the same legal actions for behaviour and Double DQN targets."""
        q = self.policy_net(next_states_t)
        if self.action_mask_indices is not None:
            allowed = next_states_t[:, list(self.action_mask_indices)] > 0.5
            # A malformed/isolated state must not produce -inf or NaN targets.
            allowed = allowed | ~allowed.any(dim=1, keepdim=True)
            q = q.masked_fill(~allowed, -torch.inf)
        return q.argmax(dim=1)

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t     = torch.FloatTensor(states).to(self.device)
        actions_t    = torch.LongTensor(actions).to(self.device)
        rewards_t    = torch.FloatTensor(rewards).to(self.device)
        next_states_t= torch.FloatTensor(next_states).to(self.device)
        dones_t      = torch.FloatTensor(dones).to(self.device)

        loss = self._compute_loss(states_t, actions_t, rewards_t, next_states_t, dones_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if not self.decay_per_episode:
            self._decay_epsilon()
        return loss.item()

    def _compute_loss(self, states_t, actions_t, rewards_t, next_states_t, dones_t):
        # Double DQN
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self._next_actions(next_states_t)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        return nn.SmoothL1Loss()(q_values, target_q)

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        """エピソード終了時に呼ぶ (decay_per_episode=True のときのみ epsilon を減衰)"""
        if self.decay_per_episode:
            self._decay_epsilon()

    def save(self, path):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "hra_heads": getattr(self, "n_heads", 0),
            "action_mask_indices": self.action_mask_indices,
        }, path)
        print(f"Model saved: {path}")

    def save_training(self, path, progress, manifest):
        """Episode-boundary snapshot, including replay and every RNG used by training."""
        state = {
            "schema": 1, "manifest": manifest, "progress": progress,
            "action_mask_indices": self.action_mask_indices,
            "policy": self.policy_net.state_dict(), "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon, "steps": self.steps,
            "replay": self.buffer.buffer, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }
        temporary = str(path) + ".tmp"
        if str(path).endswith(".gz"):
            with gzip.open(temporary, "wb", compresslevel=1) as stream:
                torch.save(state, stream)
        else:
            torch.save(state, temporary)
        os.replace(temporary, path)

    def load_training(self, path, manifest):
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as stream:
                state = torch.load(stream, map_location="cpu", weights_only=False)
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema") != 1 or state.get("manifest") != manifest:
            raise ValueError("Training checkpoint configuration/code/runtime mismatch")
        if state.get("action_mask_indices") != self.action_mask_indices:
            raise ValueError("Training checkpoint action mask mismatch")
        self.policy_net.load_state_dict(state["policy"])
        self.target_net.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon, self.steps = state["epsilon"], state["steps"]
        self.buffer.buffer = state["replay"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state["progress"]

    def load(self, path):
        # DirectML は map_location に直接渡せないので CPU 経由でロード
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("action_mask_indices") != self.action_mask_indices:
            raise ValueError("Model action mask mismatch; construct the agent with the saved mask")
        self.policy_net.load_state_dict(ckpt["policy"])
        self.policy_net.to(self.device)
        self.target_net.load_state_dict(ckpt["policy"])
        self.target_net.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps = ckpt["steps"]
        print(f"Model loaded: {path}  (epsilon={self.epsilon:.3f})")


class HRADQNAgent(DQNAgent):
    """
    Hybrid Reward Architecture 版 Double DQN。
    報酬を要素(エサ/ゴースト/移動)に分け、要素ごとのヘッドがそれぞれの報酬だけを予測する。
    各ヘッドは単純な関数を学べばよいので、全部入りの報酬1本を学ぶより安定しやすい。

    push() の reward には要素別報酬のベクトル(長さ n_heads)を渡す。
    次状態の行動は合計Q値 (policy_net) の argmax で全ヘッド共通に選び、
    各ヘッドは target_net のその行動の値でブートストラップする (Double DQN)。
    """
    def __init__(self, obs_size, n_actions, n_heads, **kwargs):
        self.n_heads = n_heads   # _build_net で使うため super().__init__ より前に設定
        super().__init__(obs_size, n_actions, **kwargs)

    def _build_net(self, obs_size, n_actions):
        return HRANetwork(obs_size, n_actions, self.n_heads)

    def _compute_loss(self, states_t, actions_t, rewards_t, next_states_t, dones_t):
        # rewards_t: [batch, n_heads]
        idx = actions_t.view(-1, 1, 1).expand(-1, self.n_heads, 1)
        q_heads = self.policy_net.head_q(states_t).gather(2, idx).squeeze(2)
        with torch.no_grad():
            next_actions = self._next_actions(next_states_t)
            next_idx = next_actions.view(-1, 1, 1).expand(-1, self.n_heads, 1)
            next_q = self.target_net.head_q(next_states_t).gather(2, next_idx).squeeze(2)
            target = rewards_t + self.gamma * next_q * (1 - dones_t).unsqueeze(1)
        per_head = nn.functional.smooth_l1_loss(q_heads, target, reduction="none").mean(dim=0)
        self.last_head_losses = per_head.detach().cpu().tolist()
        return per_head.mean()


def load_agent(path, obs_size, n_actions, **kwargs):
    """保存ファイルを見て DQN / HRA のどちらかを作り、重みを読み込んで返す"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n_heads = ckpt.get("hra_heads", 0)
    kwargs.setdefault("action_mask_indices", ckpt.get("action_mask_indices"))
    if n_heads:
        agent = HRADQNAgent(obs_size, n_actions, n_heads, **kwargs)
    else:
        agent = DQNAgent(obs_size, n_actions, **kwargs)
    agent.load(path)
    return agent
