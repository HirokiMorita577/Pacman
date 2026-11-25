"""Common networks and explicitly separated TD, on-policy and replay algorithms."""
import copy
from collections import deque
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from pacman_env import CELL_DOT, CELL_POWER, CELL_WALL, DIRS
from .config import HEURISTICS, effective_options
from .runtime import device_name


def layers(widths, activation):
    cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[activation]
    result = []
    for a, b in zip(widths, widths[1:]):
        result.extend((nn.Linear(a, b), cls()))
    return nn.Sequential(*result)


class Network(nn.Module):
    def __init__(self, obs_size, hidden, activation, kind, n_actions=4):
        super().__init__()
        self.kind = kind
        if kind == "hra":
            trunk_hidden = hidden[:-1]
            self.trunk = layers([obs_size, *trunk_hidden], activation)
            size = trunk_hidden[-1] if trunk_hidden else obs_size
            self.heads = nn.ModuleList([
                nn.Sequential(layers([size, hidden[-1]], activation), nn.Linear(hidden[-1], n_actions))
                for _ in range(3)])
        else:
            self.trunk = layers([obs_size, *hidden], activation)
            size = hidden[-1]
            self.action = nn.Linear(size, n_actions)
            if kind in ("dueling", "ppo"):
                self.value = nn.Linear(size, 1)

    def head_q(self, obs):
        trunk = self.trunk(obs)
        return torch.stack([head(trunk) for head in self.heads], dim=1)

    def forward(self, obs):
        if self.kind == "hra":
            return self.head_q(obs).sum(dim=1)
        trunk = self.trunk(obs)
        action = self.action(trunk)
        if self.kind == "dueling":
            return self.value(trunk) + action - action.mean(dim=1, keepdim=True)
        if self.kind == "ppo":
            return action, self.value(trunk).squeeze(-1)
        return action


class Agent:
    def __init__(self, config, obs_size, action_mask_indices):
        self.config = copy.deepcopy(config)
        self.algorithm = config["algorithm"]
        self.device = torch.device(device_name(self.algorithm))
        self.options = effective_options(config)
        self.mask_indices = (tuple(action_mask_indices)
                             if config["train"]["mask_actions"] and action_mask_indices is not None else None)
        if self.mask_indices is not None and (len(self.mask_indices) != 4 or len(set(self.mask_indices)) != 4
                or any(i < 0 or i >= obs_size for i in self.mask_indices)):
            raise ValueError("Action mask needs four distinct, valid observation indices")
        self.steps = self.updates = self.samples = 0
        self.last_stats = {}
        self.pending = None
        self.rollout = []
        self.replay = deque(maxlen=self.options.get("buffer_capacity", 1))
        self.network = self.target = self.optimizer = None
        if self.algorithm in HEURISTICS:
            return
        a = config["agent"]
        kind = "hra" if self.algorithm == "hra" else (
            "dueling" if self.algorithm.startswith("dueling") else (
                "ppo" if self.algorithm == "ppo" else "scalar"))
        self.network = Network(obs_size, a["hidden_sizes"], a["activation"], kind).to(self.device)
        if self.algorithm != "ppo" and self.options["use_target"]:
            self.target = copy.deepcopy(self.network).eval()
        optimizer = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}[a["optimizer"]]
        self.optimizer = optimizer(self.network.parameters(), lr=a["lr"])

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.network.parameters()) if self.network is not None else 0

    def legal(self, obs):
        if self.mask_indices is None:
            return np.arange(4)
        allowed = np.flatnonzero(np.asarray(obs)[list(self.mask_indices)] > 0.5)
        return allowed if len(allowed) else np.arange(4)

    def allowed_tensor(self, obs):
        if self.mask_indices is None:
            return torch.ones((len(obs), 4), dtype=torch.bool, device=obs.device)
        allowed = obs[:, list(self.mask_indices)] > 0.5
        return allowed | ~allowed.any(dim=1, keepdim=True)

    def masked(self, values, obs):
        return values.masked_fill(~self.allowed_tensor(obs), -torch.inf)

    def action(self, obs, epsilon=0.0, training=False, env=None):
        legal = self.legal(obs)
        if self.algorithm == "random":
            return int(random.choice(legal))
        if self.algorithm in ("greedy_food", "safe_food"):
            if env is None:
                raise ValueError("Food baselines require the current environment")
            return self.food_action(env, legal)
        tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.algorithm == "ppo":
                logits, value = self.network(tensor)
                distribution = Categorical(logits=self.masked(logits, tensor))
                action = distribution.sample() if training else self.masked(logits, tensor).argmax(dim=1)
                if training:
                    self.pending = (float(distribution.log_prob(action)), float(value))
                return int(action.item())
            if training and random.random() < epsilon:
                return int(random.choice(legal))
            q = self.network(tensor)[0].cpu().numpy()
        return int(legal[np.argmax(q[legal])])

    def food_action(self, env, legal):
        foods = [(r, c) for r in range(env.rows) for c in range(env.cols)
                 if env.grid[r][c] in (CELL_DOT, CELL_POWER)]
        food_dist = env._distance_field(foods) if foods else None
        threats = env._threat_field() if self.algorithm == "safe_food" else None
        values = []
        for action in legal:
            dr, dc = DIRS[action]
            nr, nc = env.pacman[0] + dr, env.pacman[1] + dc
            if not (0 <= nr < env.rows and 0 <= nc < env.cols and env.grid[nr][nc] != CELL_WALL):
                nr, nc = env.pacman
            distance = food_dist[nr][nc] if food_dist is not None else -1
            distance = distance if distance >= 0 else env.rows * env.cols
            danger = threats[nr][nc] if threats is not None else -1
            margin = danger - 1 if danger >= 0 else env.rows * env.cols
            values.append((margin > 0, -distance, min(margin, 10)) if self.algorithm == "safe_food"
                          else (-distance,))
        return int(legal[max(range(len(legal)), key=lambda i: values[i])])

    @torch.no_grad()
    def td_targets(self, rewards, next_obs, dones, next_actions, epsilon):
        network = self.target if self.target is not None else self.network
        target_values = network(next_obs)
        if self.algorithm in ("sarsa", "replay_sarsa"):
            next_q = target_values.gather(1, next_actions[:, None]).squeeze(1)
        elif self.algorithm == "expected_sarsa":
            allowed = self.allowed_tensor(next_obs)
            probabilities = epsilon[:, None] * allowed / allowed.sum(dim=1, keepdim=True)
            greedy = self.masked(self.network(next_obs), next_obs).argmax(dim=1)
            probabilities.scatter_add_(1, greedy[:, None], (1 - epsilon)[:, None])
            next_q = (probabilities * target_values).sum(dim=1)
        elif self.algorithm in ("dqn", "dueling_dqn"):
            next_q = self.masked(target_values, next_obs).max(dim=1).values
        else:
            greedy = self.masked(self.network(next_obs), next_obs).argmax(dim=1)
            if self.algorithm == "hra":
                next_q = network.head_q(next_obs).gather(
                    2, greedy[:, None, None].expand(-1, 3, 1)).squeeze(2)
            else:
                next_q = target_values.gather(1, greedy[:, None]).squeeze(1)
        if self.algorithm == "hra":
            return rewards + self.config["agent"]["gamma"] * (1 - dones)[:, None] * next_q
        return rewards + self.config["agent"]["gamma"] * (1 - dones) * next_q

    def observe(self, obs, action, reward, next_obs, done, boundary, next_action=None, epsilon=0.0):
        self.steps += 1
        if self.algorithm in HEURISTICS:
            return None
        if self.algorithm == "ppo":
            if self.pending is None:
                raise ValueError("PPO action was not sampled before observe")
            with torch.no_grad():
                _, next_value = self.network(torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            log_prob, value = self.pending
            self.pending = None
            self.rollout.append((obs, action, float(reward), bool(done), bool(boundary),
                                 log_prob, value, float(next_value)))
            return self.update_ppo() if len(self.rollout) >= self.options["rollout_steps"] else None
        if self.algorithm in ("sarsa", "replay_sarsa") and not done and next_action is None:
            raise ValueError("SARSA requires the next behaviour action, chosen before updating")
        transition = (obs, action, reward, next_obs, float(done),
                      0 if next_action is None else int(next_action), float(epsilon))
        self.replay.append(transition)
        if len(self.replay) < max(self.options["warmup"], self.options["batch_size"]) or (
                self.steps % self.options["train_every"]):
            return None
        losses = []
        for _ in range(self.options["updates_per_step"]):
            batch = random.sample(self.replay, self.options["batch_size"]) if self.options["replay"] else [transition]
            states, actions, rewards, following, dones, next_actions, eps = zip(*batch)
            tensor = lambda x, dtype=torch.float32: torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)
            states, actions = tensor(states), tensor(actions, torch.long)
            rewards, following, dones = tensor(rewards), tensor(following), tensor(dones)
            targets = self.td_targets(rewards, following, dones, tensor(next_actions, torch.long), tensor(eps))
            predictions = (self.network.head_q(states).gather(
                2, actions[:, None, None].expand(-1, 3, 1)).squeeze(2) if self.algorithm == "hra"
                else self.network(states).gather(1, actions[:, None]).squeeze(1))
            loss_fn = nn.functional.smooth_l1_loss if self.config["agent"]["loss"] == "huber" else nn.functional.mse_loss
            loss = loss_fn(predictions, targets)
            self.optimize(loss, len(batch))
            self.last_stats.update(td_error=float((predictions.detach() - targets).abs().mean()))
            if self.algorithm == "hra":
                individual = loss_fn(predictions, targets, reduction="none").detach().mean(dim=0)
                self.last_stats.update({f"head{h}_loss": float(individual[h]) for h in range(3)})
                self.last_stats.update({f"head{h}_q": float(predictions.detach()[:, h].mean()) for h in range(3)})
            losses.append(float(loss.detach()))
            if self.target is not None:
                tau = self.options["target_tau"]
                if tau is not None:
                    with torch.no_grad():
                        for target, current in zip(self.target.parameters(), self.network.parameters()):
                            target.lerp_(current, tau)
                elif self.updates % self.options["target_update"] == 0:
                    self.target.load_state_dict(self.network.state_dict())
        return float(np.mean(losses))

    def optimize(self, loss, samples):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss; this trial must not be ranked as successful")
        self.optimizer.zero_grad()
        loss.backward()
        limit = self.config["agent"]["grad_clip"]
        norm = nn.utils.clip_grad_norm_(self.network.parameters(), float("inf") if limit is None else limit)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        self.optimizer.step()
        self.updates += 1
        self.samples += samples
        self.last_stats = dict(loss=float(loss.detach()), grad_norm=float(norm))

    def update_ppo(self):
        if not self.rollout:
            return None
        obs, actions, rewards, dones, boundaries, old_lp, values, following = zip(*self.rollout)
        gamma = self.config["agent"]["gamma"]
        advantages, carry = [], 0.0
        for r, d, cut, v, nv in reversed(list(zip(rewards, dones, boundaries, values, following))):
            delta = r + gamma * (1 - d) * nv - v
            carry = delta + gamma * self.options["gae_lambda"] * (not cut) * carry
            advantages.append(carry)
        advantage = torch.tensor(list(reversed(advantages)), dtype=torch.float32, device=self.device)
        returns = advantage + torch.tensor(values, dtype=torch.float32, device=self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(old_lp, dtype=torch.float32, device=self.device)
        losses = []
        for _ in range(self.options["epochs"]):
            indices = torch.randperm(len(obs), device=self.device)
            for offset in range(0, len(obs), self.options["batch_size"]):
                ix = indices[offset:offset + self.options["batch_size"]]
                logits, predicted = self.network(obs[ix])
                dist = Categorical(logits=self.masked(logits, obs[ix]))
                ratio = (dist.log_prob(actions[ix]) - old_lp[ix]).exp()
                unclipped = ratio * advantage[ix]
                clipped = ratio.clamp(1 - self.options["clip_eps"], 1 + self.options["clip_eps"]) * advantage[ix]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(predicted, returns[ix])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.options["value_coef"] * value_loss - self.options["entropy_coef"] * entropy
                self.optimize(loss, len(ix))
                self.last_stats.update(policy_loss=float(policy_loss.detach()),
                                       value_loss=float(value_loss.detach()), entropy=float(entropy.detach()))
                losses.append(float(loss.detach()))
        self.rollout.clear()
        return float(np.mean(losses))

    def finish(self):
        return self.update_ppo() if self.algorithm == "ppo" else None

    def state_dict(self, training=True):
        state = dict(algorithm=self.algorithm, steps=self.steps, updates=self.updates, samples=self.samples,
                     last_stats=self.last_stats,
                     network=self.network.state_dict() if self.network is not None else None)
        if training:
            state.update(target=self.target.state_dict() if self.target is not None else None,
                         optimizer=self.optimizer.state_dict() if self.optimizer is not None else None,
                         replay=self.replay, rollout=self.rollout, pending=self.pending)
        return state

    def load_state_dict(self, state, training=True):
        if state["algorithm"] != self.algorithm:
            raise ValueError("Agent algorithm mismatch")
        if self.network is not None:
            self.network.load_state_dict(state["network"])
        self.steps, self.updates, self.samples = state["steps"], state["updates"], state["samples"]
        self.last_stats = state["last_stats"]
        if training:
            if self.target is not None:
                self.target.load_state_dict(state["target"])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(state["optimizer"])
            self.replay, self.rollout, self.pending = state["replay"], state["rollout"], state["pending"]
