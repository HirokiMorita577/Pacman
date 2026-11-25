"""Budget-matched PSO/random/grid search with held-out retraining after selection."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from pso_optimizer import Particle
from .config import METRICS, ONLINE, HEURISTICS, canonical, digest, manifest, merge, set_path, trial_id, validate
from .engine import atomic_json, evaluate_final, run_trial, run_lock, write_csv
from .storage import copy_verified, sha256

METHODS = {
    "baseline": {},
    "random": {},
    "grid": {},
    "pso": dict(w_start=0.9, w_end=0.4, c1=1.5, c2=1.5),
    "pso_constant_w": dict(w_start=0.7, w_end=0.7, c1=1.5, c2=1.5),
    "pso_cognitive": dict(w_start=0.9, w_end=0.4, c1=2.0, c2=0.5),
    "pso_social": dict(w_start=0.9, w_end=0.4, c1=0.5, c2=2.0),
    "pso_no_inertia": dict(w_start=0.0, w_end=0.0, c1=1.5, c2=1.5),
    "pso_large_swarm": dict(w_start=0.9, w_end=0.4, c1=1.5, c2=1.5),
}
SPACES = {
    "gamma": {"agent.gamma": dict(low=0.8, high=0.999)},
    "power_gamma": {"rewards.power": dict(low=0, high=400), "agent.gamma": dict(low=0.8, high=0.999)},
    "rewards": {
        "rewards.power": dict(low=0, high=400), "rewards.death": dict(low=0, high=1560),
        "rewards.clear": dict(low=0, high=5000), "rewards.eat_ghost": dict(low=0, high=800),
        "rewards.time": dict(low=0, high=1),
    },
    "hyperparameters": {
        "agent.gamma": dict(low=0.8, high=0.999),
        "agent.lr": dict(low=0.00001, high=0.003, scale="log"),
        "train.epsilon_end": dict(low=0.001, high=0.2),
    },
}
SPACES["joint"] = {**SPACES["rewards"], **SPACES["hyperparameters"]}


def prepare_search(base, algorithm, space_name):
    family = lambda a: "ppo" if a == "ppo" else ("online" if a in ONLINE else
                        ("heuristic" if a in HEURISTICS else "replay"))
    reference = copy.deepcopy(base)
    reference["algorithm"] = algorithm
    if family(algorithm) != family(base["algorithm"]):
        reference["algorithm_options"] = {}
    reference = validate(reference)
    space = copy.deepcopy(SPACES[space_name])
    if algorithm == "ppo" and space_name in ("hyperparameters", "joint"):
        space.pop("train.epsilon_end", None)
        space.update({"algorithm_options.clip_eps": dict(low=0.05, high=0.4),
                      "algorithm_options.entropy_coef": dict(low=0, high=0.05),
                      "algorithm_options.gae_lambda": dict(low=0.8, high=1.0)})
    validate_space(space)
    encode(reference, space)
    return reference, space


def get_path(config, path):
    value = config
    for part in path.split("."):
        value = value[part]
    return value


def validate_space(space):
    if not space:
        raise ValueError("Empty search space")
    for path, spec in space.items():
        if set(spec) - {"low", "high", "scale", "type", "choices"}:
            raise ValueError("Unknown space option: " + path)
        if "choices" in spec:
            if not isinstance(spec["choices"], list) or len(spec["choices"]) < 2:
                raise ValueError("choices must contain at least two values")
        elif not (math.isfinite(spec["low"]) and math.isfinite(spec["high"]) and spec["low"] < spec["high"]):
            raise ValueError("Invalid search bounds: " + path)
        if spec.get("scale", "linear") not in ("linear", "log") or spec.get("type", "float") not in ("float", "int"):
            raise ValueError("Invalid scale/type")
        if spec.get("scale") == "log" and spec["low"] <= 0:
            raise ValueError("Log-space bounds must be positive")


def encode(config, space):
    position = []
    for path, spec in space.items():
        value = get_path(config, path)
        if "choices" in spec:
            position.append(spec["choices"].index(value) / (len(spec["choices"]) - 1))
            continue
        low, high = spec["low"], spec["high"]
        if not low <= value <= high:
            raise ValueError("The reference value must be inside the search bounds: " + path)
        if spec.get("scale") == "log":
            value, low, high = map(math.log, (value, low, high))
        position.append(float(np.clip((value - low) / (high - low), 0, 1)))
    return np.array(position)


def decode(position, space):
    values = {}
    for u, (path, spec) in zip(position, space.items()):
        u = float(np.clip(u, 0, 1))
        if "choices" in spec:
            value = copy.deepcopy(spec["choices"][round(u * (len(spec["choices"]) - 1))])
        else:
            low, high = spec["low"], spec["high"]
            if spec.get("scale") == "log":
                value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
            else:
                value = low + u * (high - low)
            if spec.get("type") == "int":
                value = round(value)
        values[path] = value
    return values


def pack_particle(particle):
    return dict(pos=particle.pos.tolist(), vel=particle.vel.tolist(), pbest_pos=particle.pbest_pos.tolist(),
                pbest_score=float(particle.pbest_score) if math.isfinite(particle.pbest_score) else None)


def grid_position(index, count, dimensions):
    # An evenly spaced subset of a Cartesian lattice when a complete grid exceeds the shared budget.
    levels = max(2, math.ceil(count ** (1 / dimensions)))
    cell = int(round(index * (levels**dimensions - 1) / max(1, count - 1)))
    coordinates = []
    for _ in range(dimensions):
        coordinates.append((cell % levels) / (levels - 1))
        cell //= levels
    return np.array(coordinates)


def optimize(method, base, space, seed, candidates, particles, folder, fitness, context=None, stop_after=None,
             checkpoint_hook=None):
    with run_lock(Path(folder) / "search.lock"):
        return _optimize(method, base, space, seed, candidates, particles, folder, fitness, context,
                         stop_after, checkpoint_hook)


def _optimize(method, base, space, seed, candidates, particles, folder, fitness, context, stop_after,
              checkpoint_hook):
    """fitness(config) returns fitness, learning_episodes and optional diagnostics."""
    validate_space(space)
    if method not in METHODS or candidates < 1 or particles < 1:
        raise ValueError("Invalid search method/budget")
    if method == "pso_large_swarm":
        particles *= 2
    if method.startswith("pso") and (candidates < particles or candidates % particles):
        raise ValueError("PSO candidates must be a positive multiple of swarm size")
    count = 1 if method == "baseline" else candidates
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = dict(method=method, base=base, space=space, seed=seed, candidates=count,
                   particles=particles, settings=METHODS[method], context=context or {})
    state_path = folder / "search_state.json"
    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * len(space)
    swarm = [Particle(bounds, rng) for _ in range(particles)]
    baseline_position = encode(base, space)
    swarm[0].pos = baseline_position.copy()
    best_score, best_position, best_config = -float("inf"), baseline_position.copy(), None
    history, cumulative_episodes = [], 0
    if state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved["request"] != request:
            raise ValueError("Search plan/source/runtime differs; use a separate output directory")
        history = saved["history"]
        cumulative_episodes = saved["cumulative_learning_episodes"]
        best_score = saved["best_score"] if saved["best_score"] is not None else -float("inf")
        best_position = np.array(saved["best_position"])
        best_config = saved["best_config"]
        for particle, state in zip(swarm, saved["swarm"]):
            particle.pos, particle.vel, particle.pbest_pos = (np.array(state[k]) for k in ("pos", "vel", "pbest_pos"))
            particle.pbest_score = state["pbest_score"] if state["pbest_score"] is not None else -float("inf")
        rng.bit_generator.state = saved["rng"]
    for index in range(len(history), count):
        if index == 0:
            position, candidate = baseline_position.copy(), copy.deepcopy(base)
        else:
            if method.startswith("pso"):
                position = swarm[index % particles].pos.copy()
            elif method == "grid":
                position = grid_position(index - 1, max(1, count - 1), len(space))
            else:
                position = rng.random(len(space))
            candidate = copy.deepcopy(base)
            for path, value in decode(position, space).items():
                set_path(candidate, path, value)
        diagnostics = fitness(candidate)
        value = diagnostics.get("fitness")
        if value is not None and not math.isfinite(value):
            value = None
        diagnostics["fitness"] = value
        cumulative_episodes += diagnostics.get("learning_episodes", 0)
        score = value if value is not None else -float("inf")
        if score > best_score:
            best_score, best_position, best_config = score, position.copy(), copy.deepcopy(candidate)
        record = dict(candidate=index + 1, parameters={p: get_path(candidate, p) for p in space},
                      fitness=value, best_fitness=best_score if math.isfinite(best_score) else None,
                      cumulative_learning_episodes=cumulative_episodes, diagnostics=diagnostics)
        history.append(record)
        if method.startswith("pso"):
            particle = swarm[index % particles]
            particle.pos = position.copy()
            particle.update_pbest(score)
            if (index + 1) % particles == 0 and best_config is not None:
                generations = count // particles
                generation = index // particles
                settings = METHODS[method]
                w = settings["w_start"] - (settings["w_start"] - settings["w_end"]) * generation / max(1, generations - 1)
                for p in swarm:
                    p.update_velocity(best_position, w, settings["c1"], settings["c2"], rng)
                    p.update_position()
        state = dict(request=request, history=history, cumulative_learning_episodes=cumulative_episodes,
                     best_score=best_score if math.isfinite(best_score) else None,
                     best_position=best_position.tolist(), best_config=best_config,
                     swarm=[pack_particle(p) for p in swarm], rng=rng.bit_generator.state, complete=len(history) == count)
        atomic_json(state_path, state)
        write_csv(folder / "search_history.csv", history)
        if checkpoint_hook is not None:
            checkpoint_hook(folder)
        print(f"SEARCH_PROGRESS {folder.name} {index+1}/{count} fitness={value} best={state['best_score']}", flush=True)
        if stop_after is not None and len(history) >= stop_after and len(history) < count:
            return state
    return json.loads(state_path.read_text(encoding="utf-8"))


def run_search(out, base, algorithms, spaces, methods, search_seeds, train_seeds, final_train_seeds,
               candidates=16, particles=4, search_steps=20000, objective="score_mean", backup=None,
               final=True, retain_candidate_models=False):
    if objective not in METRICS:
        raise ValueError("Unknown search objective")
    if set(train_seeds) & set(final_train_seeds):
        raise ValueError("Search and final retraining seeds must be disjoint")
    if not train_seeds or not final_train_seeds:
        raise ValueError("Search/final training seeds must not be empty")
    for name, values in (("search_seeds", search_seeds), ("train_seeds", train_seeds),
                         ("final_train_seeds", final_train_seeds)):
        if not values or len(values) != len(set(values)) or any(
                isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in values):
            raise ValueError(name + " must contain unique seeds below 2**32")
    if candidates < 1 or particles < 1 or search_steps < 1:
        raise ValueError("Search budgets must be positive")
    methods = list(METHODS) if "all" in methods else methods
    spaces = list(SPACES) if "all" in spaces else spaces
    if set(spaces) - set(SPACES) or set(methods) - set(METHODS):
        raise ValueError("Unknown search space/method")
    for method in methods:
        size = particles * (2 if method == "pso_large_swarm" else 1)
        if method.startswith("pso") and (candidates < size or candidates % size):
            raise ValueError("All PSO swarm sizes must divide the shared candidate budget")
    for name in spaces:
        validate_space(SPACES[name])
    out = Path(out)
    if out.resolve().is_relative_to(Path("/content/drive").resolve()):
        raise ValueError("Use local search output and mounted Drive backup")
    summaries = []
    for algorithm in algorithms:
        for space_name in spaces:
            reference, space = prepare_search(base, algorithm, space_name)
            if objective == "return_mean" and any(p.startswith("rewards.") for p in space):
                raise ValueError("Reward search must use score_mean or reference_return_mean, not its own changing reward scale")
            for search_seed in search_seeds:
                group = out / f"{algorithm}_{space_name}_seed{search_seed}"
                for method in methods:
                    folder = group / method
                    candidate_out = folder / "candidates"
                    remote_folder = Path(backup) / group.name / method if backup else None
                    if remote_folder:
                        folder.mkdir(parents=True, exist_ok=True)
                        for name in ("search_state.json", "search_history.csv", "selection.json", "search_result.json"):
                            if not (folder / name).exists() and (remote_folder / name).exists():
                                import shutil
                                shutil.copy2(remote_folder / name, folder / name)
                    direction = METRICS[objective]

                    def checkpoint_hook(local_folder):
                        if remote_folder:
                            for name in ("search_state.json", "search_history.csv"):
                                copy_verified(local_folder / name, remote_folder / name)

                    def fitness(candidate):
                        results = []
                        current_state = folder / "search_state.json"
                        previous = json.loads(current_state.read_text(encoding="utf-8"))["history"] if current_state.exists() else []
                        accounted = {trial for row in previous for trial in row["diagnostics"].get("trial_ids", [])}
                        try:
                            candidate = validate(candidate)
                            config = copy.deepcopy(candidate)
                            ratio = config["train"]["explore_steps"] / config["train"]["steps"]
                            config["train"].update(steps=search_steps, explore_steps=max(1, int(search_steps * ratio)))
                            config["train"]["eval_every"] = 0
                            for seed in train_seeds:
                                run = candidate_out / "runs" / trial_id(config, seed)
                                pruned = run / "MODEL_PRUNED.json"
                                cached = run / "result.json"
                                if pruned.exists() and cached.exists():
                                    receipt = json.loads(pruned.read_text(encoding="utf-8"))
                                    result = json.loads(cached.read_text(encoding="utf-8"))
                                    if receipt["result_sha256"] != sha256(cached) or result["manifest"] != manifest(config, seed):
                                        raise ValueError("Pruned candidate receipt mismatch")
                                else:
                                    result = run_trial(config, seed, candidate_out,
                                                       backup=remote_folder / "candidates" if remote_folder else None)
                                results.append(result)
                            means = {k: float(np.mean([r["evaluation"][k] for r in results])) for k in METRICS}
                            return dict(fitness=direction * means[objective], metrics=means,
                                        learning_episodes=sum(r["episodes_completed"] for r in results if r["id"] not in accounted),
                                        environment_steps=sum(r["total_steps"] for r in results if r["id"] not in accounted),
                                        assigned_learning_episodes=sum(r["episodes_completed"] for r in results),
                                        assigned_environment_steps=sum(r["total_steps"] for r in results),
                                        trial_ids=[r["id"] for r in results])
                        except (ValueError, FloatingPointError, RuntimeError) as exc:
                            return dict(fitness=None,
                                        learning_episodes=sum(r["episodes_completed"] for r in results if r["id"] not in accounted),
                                        environment_steps=sum(r["total_steps"] for r in results if r["id"] not in accounted),
                                        trial_ids=[r["id"] for r in results], cost_is_lower_bound=True, error=str(exc))

                    identity = manifest(reference, train_seeds[0])
                    context = dict(identity=identity, train_seeds=train_seeds, search_steps=search_steps,
                                   objective=objective, final_train_seeds=final_train_seeds)
                    state = optimize(method, reference, space, search_seed, candidates, particles,
                                     folder, fitness, context=context, checkpoint_hook=checkpoint_hook)
                    selection = dict(search_complete=state["complete"], objective=objective,
                                     direction=direction, selected_config=state["best_config"],
                                     dev_fitness=state["best_score"], search_seed=search_seed,
                                     final_train_seeds=final_train_seeds,
                                     final_seed0=reference["evaluation"]["final_seed0"])
                    # Freeze the selection before accessing any held-out results.
                    selection_path = folder / "selection.json"
                    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
                        raise ValueError("Frozen selection differs")
                    atomic_json(selection_path, selection)
                    if remote_folder and not copy_verified(selection_path, remote_folder / "selection.json"):
                        raise OSError("Selection could not be backed up; held-out evaluation deferred")
                    final_rows = []
                    if final and selection["selected_config"] is not None:
                        final_out = folder / "final"
                        final_backup = (Path(backup) / group.name / method / "final") if backup else None
                        for seed in final_train_seeds:
                            result = run_trial(selection["selected_config"], seed, final_out, backup=final_backup)
                            evaluated = evaluate_final(final_out / "runs" / result["id"], backup=final_backup)
                            final_rows.append(dict(train_seed=seed, learning_episodes=result["episodes_completed"],
                                                   **{k: evaluated[k] for k in METRICS}))
                    summary = dict(algorithm=algorithm, space=space_name, method=method,
                                   search_seed=search_seed, proposals=len(state["history"]),
                                   search_learning_episodes=state["cumulative_learning_episodes"],
                                   search_environment_steps=sum(r["diagnostics"].get("environment_steps", 0) for r in state["history"]),
                                   dev_fitness=state["best_score"], final=final_rows)
                    summaries.append(summary)
                    atomic_json(folder / "search_result.json", summary)
                    if remote_folder:
                        copy_verified(folder / "search_result.json", remote_folder / "search_result.json")
                    if not retain_candidate_models:
                        prune_candidates(candidate_out, remote_folder / "candidates" if remote_folder else None)
                    from .report import search_report
                    search_report(group)
                    if backup:
                        for image in (group / "figures").glob("*.png"):
                            copy_verified(image, Path(backup) / group.name / "figures" / image.name)
    atomic_json(out / "search_results.json", summaries)
    write_csv(out / "search_summary.csv", [
        {**{k: v for k, v in row.items() if k != "final"},
         **{f"final_{k}": float(np.mean([r[k] for r in row["final"]])) if row["final"] else None for k in METRICS}}
        for row in summaries])
    if backup:
        for name in ("search_results.json", "search_summary.csv"):
            copy_verified(out / name, Path(backup) / name)
    return summaries


def prune_candidates(local, remote=None):
    """Keep candidate metrics/logs; discard only this completed search's reproducible weight files."""
    for result_path in Path(local).glob("runs/*/result.json"):
        run = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf-8"))
        model = run / "model.pt"
        if model.exists() and sha256(model) != result["model_sha256"]:
            raise ValueError("Refusing to prune a mismatched candidate model")
        receipt = dict(result_sha256=sha256(result_path), model_sha256=result["model_sha256"])
        atomic_json(run / "MODEL_PRUNED.json", receipt)
        for name in ("model.pt", "training.pt.gz", "recovery_A.pt.gz", "recovery_B.pt.gz"):
            target = run / name
            if target.is_file():
                target.unlink()
        if remote:
            remote_run = Path(remote) / "runs" / run.name
            if remote_run.exists() and copy_verified(run / "MODEL_PRUNED.json", remote_run / "MODEL_PRUNED.json"):
                for name in ("model.pt", "training.pt.gz", "recovery_A.pt.gz", "recovery_B.pt.gz"):
                    target = remote_run / name
                    if target.is_file():
                        target.unlink()
