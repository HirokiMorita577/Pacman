"""Equal-step training, common-reward evaluation and episode-boundary recovery."""
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from pacman_env import PacManEnv, REWARD_SIGN
from .agents import Agent
from .config import REFERENCE_REWARDS, canonical, epsilon_at, effective_options, manifest, trial_id, validate


class MeasuredEnv(PacManEnv):
    """Count a common reference reward even when an experimental reward is zero."""
    def reset(self, **kwargs):
        self.reference_return = 0.0
        return super().reset(**kwargs)

    def _give(self, rc, name, scale=1.0):
        self.reference_return += REWARD_SIGN[name] * REFERENCE_REWARDS[name] * scale
        return super()._give(rc, name, scale)


def make_env(config):
    return MeasuredEnv(render_mode=None, rewards=config["rewards"], **config["env"])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def torch_save(path, data, compressed=False):
    temporary = Path(str(path) + ".tmp")
    if compressed:
        with gzip.open(temporary, "wb", compresslevel=1) as stream:
            torch.save(data, stream)
    else:
        torch.save(data, temporary)
    os.replace(temporary, path)


def torch_load(path, compressed=False):
    if compressed:
        with gzip.open(path, "rb") as stream:
            return torch.load(stream, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(agent, config, phase="dev", episodes=None):
    ev = config["evaluation"]
    count = ev["episodes"] if episodes is None else episodes
    if phase not in ("dev", "final") or count < 1 or count > ev["episodes"]:
        raise ValueError("Invalid evaluation phase/episode count")
    env = make_env(config)
    records = []
    with isolated_rng():
        for index in range(count):
            seed = ev[phase + "_seed0"] + index
            seed_all(seed)
            obs, _ = env.reset(seed=seed)
            raw_return, blocked, reversals = 0.0, 0, 0
            last_action = None
            actions, visited = Counter(), Counter({tuple(env.pacman): 1})
            stale = max_stale = 0
            while True:
                action = agent.action(obs, training=False, env=env)
                blocked += int(obs[env.action_mask_indices[action]] < 0.5)
                actions[action] += 1
                reversals += int(last_action is not None and action == (last_action ^ 1))
                last_action = action
                before = env.dots_eaten
                obs, reward, terminated, truncated, info = env.step(action)
                raw_return += reward
                visited[tuple(env.pacman)] += 1
                stale = 0 if env.dots_eaten > before else stale + 1
                max_stale = max(max_stale, stale)
                if terminated or truncated:
                    break
            probabilities = np.array(list(actions.values()), dtype=float) / env.steps
            records.append(dict(
                evaluation_episode=index + 1, evaluation_seed=seed, score=info["score"],
                **{"return": raw_return}, reference_return=env.reference_return,
                cleared=int(info["cleared"]), steps=env.steps,
                ending="clear" if info["cleared"] else ("caught" if terminated else "truncated"),
                lives_lost=config["env"]["lives"] - info["lives"],
                pellet_fraction=info["dots_eaten"] / env.total_dots,
                dots_left=env.total_dots - info["dots_eaten"], blocked_actions=blocked,
                dot_points=info["score_parts"]["dot"], power_points=info["score_parts"]["power"],
                ghost_points=info["score_parts"]["ghost"], ghosts_eaten=info["ghosts_eaten"],
                power_eaten=info["power_eaten"], power_first_step=info["power_steps"][0] if info["power_steps"] else None,
                ghosts_per_power=info["ghosts_eaten"] / info["power_eaten"] if info["power_eaten"] else None,
                action_entropy=float(-(probabilities * np.log(probabilities)).sum()),
                reversals=reversals, distinct_cells=len(visited), max_no_food_steps=max_stale,
                reward_items=copy.deepcopy(info["reward_items"])))
    env.close()
    return dict(phase=phase, episodes=records, **summarize_episodes(records))


def summarize_episodes(records):
    if not records:
        raise ValueError("Cannot summarize empty evaluation")
    means = {name + "_mean": float(np.mean([r[name] for r in records if r[name] is not None]))
             if any(r[name] is not None for r in records) else None
             for name in ("score", "return", "reference_return", "steps", "lives_lost", "pellet_fraction",
                          "dots_left", "dot_points", "power_points", "ghost_points", "ghosts_eaten",
                          "power_eaten", "power_first_step", "ghosts_per_power", "action_entropy",
                          "reversals", "distinct_cells", "max_no_food_steps")}
    scores = [r["score"] for r in records]
    means.update(clear_rate=float(np.mean([r["cleared"] for r in records])),
                 score_std=float(np.std(scores)), score_median=float(np.median(scores)),
                 score_min=min(scores), score_max=max(scores),
                 blocked_action_rate=sum(r["blocked_actions"] for r in records) / max(1, sum(r["steps"] for r in records)),
                 timeout_rate=float(np.mean([r["ending"] == "truncated" for r in records])),
                 caught_rate=float(np.mean([r["ending"] == "caught" for r in records])))
    return means


def learning_reward(config, components):
    unit = {"none": 1.0, "fixed100": 100.0,
            "pellet": config["rewards"]["dot"] or 10.0}[config["train"]["reward_normalization"]]
    components = np.asarray(components, dtype=np.float32) / unit
    clip = config["train"]["reward_clip"]
    total = float(components.sum())
    if clip is not None and abs(total) > clip:
        components = components * (clip / abs(total))
    return components if config["algorithm"] == "hra" else float(components.sum())


def run_trial(config, seed, out, stop_after_episodes=None, backup=None):
    config = validate(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Training seed must be an integer in [0,2**32)")
    if Path(out).resolve().is_relative_to(Path("/content/drive").resolve()):
        raise ValueError("Train on local storage and use --backup for Drive")
    if backup:
        local, remote = Path(out).resolve(), Path(backup).resolve()
        if local.is_relative_to(remote) or remote.is_relative_to(local):
            raise ValueError("Training and backup directories must be separate")
    root = Path(out) / "runs" / trial_id(config, seed)
    with run_lock(root / "trial.lock"):
        return _run_trial(config, seed, root, stop_after_episodes, backup)


def _run_trial(config, seed, root, stop_after_episodes, backup):
    from .runtime import configure
    execution = configure(config["algorithm"])
    print("TRIAL_DEVICE", root.name, execution, flush=True)
    frozen = manifest(config, seed)
    from .storage import restore_trial, sync_trial
    if backup:
        restore_trial(root, Path(backup) / "runs" / root.name, frozen)
    complete, model, checkpoint = root / "result.json", root / "model.pt", root / "training.pt.gz"
    if complete.exists():
        result = json.loads(complete.read_text(encoding="utf-8"))
        if result["manifest"] != frozen or not model.exists():
            raise ValueError("Completed trial manifest/model mismatch: " + str(root))
        if result["model_sha256"] != hashlib.sha256(model.read_bytes()).hexdigest():
            raise ValueError("Completed model checksum mismatch")
        return result
    seed_all(seed)
    env = make_env(config)
    agent = Agent(config, env.observation_space.shape[0], env.action_mask_indices)
    progress = dict(episode=0, total_steps=0, rows=[], curves=[], wall_seconds=0.0)
    if checkpoint.exists():
        saved = torch_load(checkpoint, compressed=True)
        if saved["manifest"] != frozen:
            raise ValueError("Checkpoint manifest mismatch")
        agent.load_state_dict(saved["agent"])
        progress = saved["progress"]
        restore_rng(saved["rng"])
    atomic_json(root / "manifest.json", frozen)
    started = time.monotonic()
    budget = config["train"]["steps"]

    def save():
        state_progress = {**progress, "wall_seconds": progress["wall_seconds"] + time.monotonic() - started}
        torch_save(checkpoint, dict(schema=1, manifest=frozen, agent=agent.state_dict(),
                                    progress=state_progress, rng=rng_state()), compressed=True)
        write_csv(root / "train.csv", progress["rows"])
        write_csv(root / "learning_curve.csv", progress["curves"])
        atomic_json(root / "progress.json", dict(episode=progress["episode"], total_steps=progress["total_steps"],
                                                budget=budget, updates=agent.updates, samples=agent.samples))
        if backup:
            sync_trial(root, Path(backup) / "runs" / root.name, final=False)

    while progress["total_steps"] < budget:
        episode = progress["episode"] + 1
        # Reserve high seeds for training; evaluation seeds live below 2**32.
        obs, _ = env.reset(seed=2**63 + seed * 1000000000 + episode)
        action = agent.action(obs, epsilon_at(config, progress["total_steps"]), training=True, env=env)
        total_return, losses = 0.0, []
        while True:
            following, reward, terminated, truncated, info = env.step(action)
            progress["total_steps"] += 1
            budget_cut = progress["total_steps"] >= budget
            done = terminated or ((truncated or budget_cut) and not config["train"]["bootstrap_truncation"])
            eps = epsilon_at(config, progress["total_steps"])
            next_action = (agent.action(following, eps, training=True, env=env)
                           if config["algorithm"] in ("sarsa", "replay_sarsa") and not done else None)
            loss = agent.observe(obs, action, learning_reward(config, info["reward_components"]),
                                 following, done, terminated or truncated or budget_cut, next_action, eps)
            if loss is not None:
                losses.append(loss)
            total_return += reward
            obs = following
            if terminated or truncated or budget_cut:
                break
            action = next_action if next_action is not None else agent.action(obs, eps, training=True, env=env)
        progress["episode"] = episode
        if budget_cut:
            last_loss = agent.finish()
            if last_loss is not None:
                losses.append(last_loss)
        row = dict(learning_episode=episode, total_steps=progress["total_steps"], score=info["score"],
                   **{"return": total_return}, reference_return=env.reference_return,
                   cleared=int(info["cleared"]), steps=env.steps, epsilon=eps,
                   updates=agent.updates, samples=agent.samples,
                   loss=float(np.mean(losses)) if losses else None,
                   ending="clear" if info["cleared"] else ("caught" if terminated else
                           ("budget" if budget_cut and not truncated else "truncated")),
                   **{"last_" + k: v for k, v in agent.last_stats.items()},
                   **{f"reward_{k}": v for k, v in info["reward_items"].items()})
        progress["rows"].append(row)
        every = config["train"]["eval_every"]
        if every and progress["total_steps"] >= (len(progress["curves"]) + 1) * every:
            ev = evaluate(agent, config, episodes=min(config["train"]["curve_episodes"], config["evaluation"]["episodes"]))
            progress["curves"].append(dict(learning_episode=episode, total_steps=progress["total_steps"],
                                            **{k: v for k, v in ev.items() if k not in ("episodes", "phase")}))
        should_pause = stop_after_episodes is not None and episode >= stop_after_episodes and not budget_cut
        if episode % config["train"]["checkpoint_episodes"] == 0 or budget_cut or should_pause:
            save()
            print(f"TRIAL_PROGRESS {root.name} episode={episode} steps={progress['total_steps']}/{budget}", flush=True)
        if should_pause:
            env.close()
            return dict(paused=True, id=root.name, episode=episode, total_steps=progress["total_steps"])
    evaluation = evaluate(agent, config)
    torch_save(model, dict(schema=1, manifest=frozen, config=config, agent=agent.state_dict(training=False)))
    result = dict(id=root.name, config=config, manifest=frozen, train_seed=seed,
                  execution=execution,
                  total_steps=progress["total_steps"], episodes_completed=progress["episode"],
                  updates=agent.updates, samples=agent.samples, parameter_count=agent.parameter_count,
                  effective_algorithm_options=effective_options(config),
                  wall_seconds=progress["wall_seconds"] + time.monotonic() - started,
                  evaluation=evaluation, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    atomic_json(complete, result)
    write_csv(root / "evaluation_dev.csv", evaluation["episodes"])
    if backup:
        # Complete verified outputs replace replay-heavy recovery slots.
        synced = sync_trial(root, Path(backup) / "runs" / root.name, final=True)
        print("BACKUP_SYNCED" if synced else "BACKUP_PENDING", root.name, flush=True)
    env.close()
    return result


def load_saved(root):
    root = Path(root)
    saved = torch_load(root / "model.pt")
    config = saved["config"]
    if saved["manifest"] != manifest(config, saved["manifest"]["seed"]):
        raise ValueError("Saved model source/config/runtime differs; use its frozen source/runtime")
    env = make_env(config)
    with isolated_rng():
        agent = Agent(config, env.observation_space.shape[0], env.action_mask_indices)
        agent.load_state_dict(saved["agent"], training=False)
    env.close()
    return agent, config


def evaluate_final(root, backup=None):
    root = Path(root)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if hashlib.sha256((root / "model.pt").read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("Final evaluation model checksum mismatch")
    path = root / "evaluation_final.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["model_sha256"] != result["model_sha256"]:
            raise ValueError("Final evaluation cache mismatch")
        return saved
    agent, config = load_saved(root)
    evaluated = dict(model_sha256=result["model_sha256"], train_seed=result["train_seed"],
                     **evaluate(agent, config, phase="final"))
    atomic_json(path, evaluated)
    write_csv(root / "evaluation_final.csv", evaluated["episodes"])
    if backup:
        from .storage import sync_trial
        sync_trial(root, Path(backup) / "runs" / root.name, final=True)
    return evaluated
