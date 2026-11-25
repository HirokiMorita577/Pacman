"""Configuration, manifests and reproducible experiment identities."""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
from importlib.metadata import version

from pacman_env import DESIGNED_REWARDS, PacManEnv, REWARD_SPEC

ALGORITHMS = (
    "q_learning", "sarsa", "expected_sarsa", "dqn", "double_dqn",
    "replay_sarsa", "hra", "dueling_dqn", "dueling_double_dqn",
    "ppo", "random", "greedy_food", "safe_food",
)
ONLINE = {"q_learning", "sarsa", "expected_sarsa"}
HEURISTICS = {"random", "greedy_food", "safe_food"}
REFERENCE_REWARDS = {**DESIGNED_REWARDS, "death": 100.0, "clear": 0.0}
DEFAULT = {
    "algorithm": "double_dqn",
    "env": {"num_ghosts": 4, "lives": 3, "include_lives": True, "max_steps": 1000,
            "ghost_mode": "classic", "observation_version": "tactical_safe",
            "classic_lookahead": True},
    "rewards": REFERENCE_REWARDS,
    "agent": {"lr": 0.0005, "gamma": 0.95, "hidden_sizes": [512, 256, 128],
              "activation": "relu", "optimizer": "adam", "loss": "huber", "grad_clip": 1.0},
    "algorithm_options": {},
    "train": {"steps": 200000, "explore_steps": 80000, "epsilon_start": 1.0,
              "epsilon_end": 0.0248, "schedule": "exponential",
              "reward_normalization": "pellet", "reward_clip": None,
              "eval_every": 20000, "curve_episodes": 10, "checkpoint_episodes": 50,
              "mask_actions": True, "bootstrap_truncation": True},
    "evaluation": {"episodes": 100, "dev_seed0": 990000, "final_seed0": 1000000},
}
METRICS = {
    "score_mean": 1, "reference_return_mean": 1, "return_mean": 1, "clear_rate": 1,
    "pellet_fraction_mean": 1, "ghosts_eaten_mean": 1, "lives_lost_mean": -1,
    "blocked_action_rate": -1, "steps_mean": 1,
}
TD_OPTIONS = {"batch_size", "buffer_capacity", "warmup", "train_every", "updates_per_step",
              "target_update", "target_tau", "replay", "use_target"}
PPO_OPTIONS = {"rollout_steps", "epochs", "batch_size", "gae_lambda", "clip_eps",
               "entropy_coef", "value_coef"}
PROFILES = {
    "online": dict(batch_size=1, buffer_capacity=1, warmup=1, train_every=1,
                   updates_per_step=1, target_update=1000, target_tau=None,
                   replay=False, use_target=False),
    "replay": dict(batch_size=128, buffer_capacity=100000, warmup=128, train_every=1,
                   updates_per_step=1, target_update=1000, target_tau=None,
                   replay=True, use_target=True),
    "ppo": dict(rollout_steps=512, epochs=4, batch_size=128, gae_lambda=0.95,
                clip_eps=0.2, entropy_coef=0.01, value_coef=0.5),
}


def merge(base, patch):
    result = copy.deepcopy(base)
    for k, v in patch.items():
        result[k] = merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else copy.deepcopy(v)
    return result


def set_path(config, path, value):
    parts = path.split(".")
    target = config
    for key in parts[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError("Not a configuration object: " + path)
    target[parts[-1]] = copy.deepcopy(value)


def effective_options(config):
    algorithm = config["algorithm"]
    if algorithm in HEURISTICS:
        return {}
    profile = "ppo" if algorithm == "ppo" else ("online" if algorithm in ONLINE else "replay")
    return merge(PROFILES[profile], config["algorithm_options"])


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")


def validate(config):
    unknown = set(config) - set(DEFAULT)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    cfg = merge(DEFAULT, config)
    if cfg["algorithm"] not in ALGORITHMS:
        raise ValueError("Unknown algorithm: " + str(cfg["algorithm"]))
    for group in ("agent", "train", "evaluation"):
        if set(cfg[group]) - set(DEFAULT[group]):
            raise ValueError("Unknown " + group + " option")
    if set(cfg["rewards"]) != set(REWARD_SPEC):
        raise ValueError("Reward names must match REWARD_SPEC")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0
           for v in cfg["rewards"].values()):
        raise ValueError("Rewards must be finite, nonnegative magnitudes")
    cfg["rewards"] = {k: float(v) for k, v in cfg["rewards"].items()}
    if "rewards" in cfg["env"] or "render_mode" in cfg["env"]:
        raise ValueError("Set rewards at top level; rendering is disabled during experiments")
    if cfg["env"].get("ghost_mode", "classic") not in ("classic", "bfs"):
        raise ValueError("Unknown ghost_mode")
    if cfg["env"].get("num_ghosts") not in (1, 2, 3, 4) or isinstance(cfg["env"].get("num_ghosts"), bool):
        raise ValueError("num_ghosts must be 1..4")
    positive_int(cfg["env"]["lives"], "lives")
    positive_int(cfg["env"]["max_steps"], "max_steps")
    # Let the environment validate all observation feature combinations.
    PacManEnv(**cfg["env"], rewards=cfg["rewards"]).close()
    agent, train, ev = cfg["agent"], cfg["train"], cfg["evaluation"]
    for k in ("lr", "gamma", "grad_clip"):
        if agent[k] is not None:
            if isinstance(agent[k], bool) or not isinstance(agent[k], (int, float)) or not math.isfinite(agent[k]):
                raise ValueError("agent." + k + " must be finite")
            agent[k] = float(agent[k])
    if agent["lr"] is None or agent["gamma"] is None:
        raise ValueError("lr and gamma are required")
    for k in ("epsilon_start", "epsilon_end", "reward_clip"):
        if train[k] is not None:
            if isinstance(train[k], bool) or not isinstance(train[k], (int, float)) or not math.isfinite(train[k]):
                raise ValueError("train." + k + " must be finite")
            train[k] = float(train[k])
    if not 0 <= agent["gamma"] <= 1 or not 0 < agent["lr"] <= 1:
        raise ValueError("gamma must be in [0,1]; lr must be in (0,1]")
    if not agent["hidden_sizes"]:
        raise ValueError("hidden_sizes must not be empty")
    for n in agent["hidden_sizes"]:
        positive_int(n, "hidden_sizes")
    if agent["activation"] not in ("relu", "tanh", "leaky_relu"):
        raise ValueError("Unknown activation")
    if agent["optimizer"] not in ("adam", "rmsprop", "sgd") or agent["loss"] not in ("huber", "mse"):
        raise ValueError("Unknown optimizer/loss")
    if agent["grad_clip"] is not None and agent["grad_clip"] <= 0:
        raise ValueError("grad_clip must be positive or null")
    for k in ("steps", "explore_steps", "checkpoint_episodes", "curve_episodes"):
        positive_int(train[k], "train." + k)
    if isinstance(train["eval_every"], bool) or not isinstance(train["eval_every"], int) or train["eval_every"] < 0:
        raise ValueError("eval_every must be a nonnegative integer (0 disables curves)")
    if not 0 <= train["epsilon_end"] <= train["epsilon_start"] <= 1:
        raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
    if train["schedule"] not in ("exponential", "linear", "cosine", "constant"):
        raise ValueError("Unknown exploration schedule")
    if train["reward_normalization"] not in ("none", "pellet", "fixed100"):
        raise ValueError("Unknown reward normalization")
    if train["reward_clip"] is not None and train["reward_clip"] <= 0:
        raise ValueError("reward_clip must be positive or null")
    for k in ("mask_actions", "bootstrap_truncation"):
        if not isinstance(train[k], bool):
            raise ValueError(k + " must be boolean")
    positive_int(ev["episodes"], "evaluation.episodes")
    for k in ("dev_seed0", "final_seed0"):
        if isinstance(ev[k], bool) or not isinstance(ev[k], int) or ev[k] < 0:
            raise ValueError("Evaluation seeds must be nonnegative integers")
        if ev[k] + ev["episodes"] >= 2**32:
            raise ValueError("Evaluation seeds must be below 2**32 (high seeds are reserved for training)")
    if set(range(ev["dev_seed0"], ev["dev_seed0"] + ev["episodes"])) & set(
            range(ev["final_seed0"], ev["final_seed0"] + ev["episodes"])):
        raise ValueError("Development and held-out evaluation seeds overlap")
    options = effective_options(cfg)
    allowed = set() if cfg["algorithm"] in HEURISTICS else (PPO_OPTIONS if cfg["algorithm"] == "ppo" else TD_OPTIONS)
    if set(cfg["algorithm_options"]) - allowed:
        raise ValueError("Unsupported algorithm_options for " + cfg["algorithm"])
    if cfg["algorithm"] == "ppo":
        for k in ("rollout_steps", "epochs", "batch_size"):
            positive_int(options[k], k)
        if not 0 <= options["gae_lambda"] <= 1 or not 0 < options["clip_eps"] < 1:
            raise ValueError("Invalid PPO lambda/clip")
        if options["entropy_coef"] < 0 or options["value_coef"] < 0:
            raise ValueError("PPO coefficients must be nonnegative")
    elif cfg["algorithm"] not in HEURISTICS:
        for k in ("replay", "use_target"):
            if not isinstance(options[k], bool):
                raise ValueError(k + " must be boolean")
        for k in ("batch_size", "buffer_capacity", "warmup", "train_every", "updates_per_step", "target_update"):
            positive_int(options[k], k)
        if options["batch_size"] > options["buffer_capacity"] or options["warmup"] > options["buffer_capacity"]:
            raise ValueError("Replay capacity is smaller than batch/warmup")
        if options["target_tau"] is not None and not 0 < options["target_tau"] <= 1:
            raise ValueError("target_tau must be in (0,1]")
        if cfg["algorithm"] in ONLINE and (options["replay"] or options["use_target"] or
                options["batch_size"] != 1 or options["warmup"] != 1 or
                options["train_every"] != 1 or options["updates_per_step"] != 1):
            raise ValueError("Online Q-learning/SARSA use the latest transition once, with no target/replay")
        if cfg["algorithm"] == "replay_sarsa" and not options["replay"]:
            raise ValueError("replay_sarsa must use replay (it is not on-policy SARSA)")
    cfg["algorithm_options"] = options
    # JSON also rejects NaN/inf in custom numerical fields.
    json.dumps(cfg, allow_nan=False)
    return cfg


def epsilon_at(config, steps):
    t = config["train"]
    fraction = min(1.0, max(0.0, steps / t["explore_steps"]))
    start, end = t["epsilon_start"], t["epsilon_end"]
    if t["schedule"] == "constant":
        return start
    if fraction == 1:
        return end
    if t["schedule"] == "exponential" and start > 0 and end > 0:
        return start * (end / start) ** fraction
    if t["schedule"] == "cosine":
        fraction = (1 - math.cos(math.pi * fraction)) / 2
    return start + (end - start) * fraction


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def manifest(config, seed):
    from .runtime import execution_info
    root = Path(__file__).resolve().parents[1]
    names = ["pacman_env.py", "pso_optimizer.py"] + [str(p.relative_to(root)).replace("\\", "/")
                                 for p in sorted((root / "experiments").glob("*.py"))]
    return dict(schema=1, config=validate(config), seed=seed,
                execution=execution_info(config["algorithm"]),
                python=platform.python_version(),
                packages={p: version(p) for p in ("numpy", "gymnasium", "torch")},
                sources={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names})


def trial_id(config, seed):
    return config["algorithm"] + "_" + digest(manifest(config, seed))[:20] + "_s" + str(seed)
