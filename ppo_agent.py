"""
PPO Agent for Pac-Man
Proximal Policy Optimization with GAE, shared Actor-Critic network
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

        # 出力層を小さく初期化（学習安定化）
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class RolloutBuffer:
    def __init__(self, n_steps, obs_size, device):
        self.n_steps = n_steps
        self.device  = device
        self.obs     = torch.zeros(n_steps, obs_size, device=device)
        self.actions = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.dones   = torch.zeros(n_steps, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.values  = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def push(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]   = value
        self.ptr += 1

    def full(self):
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        advantages = torch.zeros(self.n_steps, device=self.device)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_done  = 0.0  if t == self.n_steps - 1 else self.dones[t]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return returns, advantages


class PPOAgent:
    def __init__(self, obs_size, n_actions,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
                 n_steps=2048, n_epochs=4, batch_size=256,
                 max_grad_norm=0.5):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.n_steps      = n_steps
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.max_grad_norm = max_grad_norm
        self.total_steps  = 0

        self.device = self._get_device()
        print(f"Using device: {self.device}")

        self.net = ActorCritic(obs_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(n_steps, obs_size, self.device)

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            dml = torch_directml.device()
            torch.tensor([1.0]).to(dml)
            return dml
        except Exception:
            pass
        return torch.device("cpu")

    def select_action(self, obs_np):
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action_and_value(obs_t)
        return (action.item(), log_prob.item(), value.item())

    def push(self, obs_np, action, reward, done, log_prob, value):
        obs_t = torch.FloatTensor(obs_np).to(self.device)
        self.buffer.push(obs_t, action, reward, done, log_prob, value)
        self.total_steps += 1

    def update(self, last_obs_np):
        """バッファが満杯のときに呼ぶ。PPOのミニバッチ更新を行う。"""
        last_obs_t = torch.FloatTensor(last_obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, last_value = self.net.get_action_and_value(last_obs_t)

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value.squeeze(), self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ミニバッチでn_epochs回更新
        n = self.n_steps
        indices = torch.arange(n, device=self.device)
        stats = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            perm = indices[torch.randperm(n, device=self.device)]
            for start in range(0, n, self.batch_size):
                mb = perm[start:start + self.batch_size]
                obs_mb     = self.buffer.obs[mb]
                actions_mb = self.buffer.actions[mb]
                old_lp_mb  = self.buffer.log_probs[mb]
                returns_mb = returns[mb]
                adv_mb     = advantages[mb]

                new_lp, entropy, new_values = self.net.evaluate(obs_mb, actions_mb)
                ratio = torch.exp(new_lp - old_lp_mb)

                # Clipped surrogate
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = nn.functional.mse_loss(new_values, returns_mb)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())

        self.buffer.reset()
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def save(self, path):
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.net.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        print(f"Model loaded: {path}  (steps={self.total_steps})")
