"""Unified reward/algorithm/PSO comparisons. All PNG horizontal axes count learning episodes."""
import argparse
import json
from pathlib import Path

from experiments.catalog import build_plan, catalog
from experiments.config import ALGORITHMS, DEFAULT, METRICS, merge


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def add_plan_arguments(parser):
    parser.add_argument("--suites", nargs="+", default=["all"], help="catalog names or all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--base", help="JSON overrides of common configuration")
    parser.add_argument("--grid", help="JSON factorial grid: name, base, axes")
    parser.add_argument("--steps", type=int, help="common baseline training steps (training_budget suite still varies it)")
    parser.add_argument("--out", required=True)


def make_plan(args):
    base = read_json(args.base)
    if args.steps is not None:
        base = merge(base, {"train": {"steps": args.steps}})
    suites = [] if args.grid and args.suites == ["all"] else args.suites
    return build_plan(suites, args.seeds, base, read_json(args.grid) if args.grid else None)


def base_config(args):
    config = merge(DEFAULT, read_json(args.base))
    if getattr(args, "steps", None) is not None:
        config["train"]["steps"] = args.steps
    return config


def smoke_plan():
    """Short functionality check; these results must not be used as research findings."""
    from experiments.config import digest, validate, effective_options, ONLINE, HEURISTICS
    base = merge(DEFAULT, {
        "agent": {"hidden_sizes": [16, 8]},
        "env": {"classic_lookahead": False, "max_steps": 8},
        "train": {"steps": 32, "explore_steps": 16, "eval_every": 16,
                  "curve_episodes": 1, "checkpoint_episodes": 2},
        "evaluation": {"episodes": 2},
    })
    conditions, groups = {}, {"algorithms_smoke": [], "power_smoke": [], "gamma_smoke": []}

    def add(group, label, changes):
        config = merge(base, changes)
        if config["algorithm"] == "ppo":
            config["algorithm_options"] = dict(rollout_steps=8, epochs=1, batch_size=4)
        elif config["algorithm"] not in ONLINE | HEURISTICS:
            config["algorithm_options"] = dict(batch_size=4, warmup=4, buffer_capacity=64)
        config = validate(config)
        key = digest(config)[:20]
        conditions[key] = dict(id=key, config=config)
        groups[group].append(dict(condition=key, label=label, changes=changes))
    for algorithm in ALGORITHMS:
        add("algorithms_smoke", algorithm, {"algorithm": algorithm})
    for power in (0, 10, 50, 200):
        add("power_smoke", f"power={power}", {"rewards": {"power": power}})
    for gamma in (0.9, 0.95, 0.99):
        add("gamma_smoke", f"gamma={gamma}", {"agent": {"gamma": gamma}})
    return dict(schema=1, seeds=[101], conditions=list(conditions.values()), groups=groups,
                unique_conditions=len(conditions), training_trials=len(conditions),
                total_environment_steps=sum(c["config"]["train"]["steps"] for c in conditions.values()),
                objective="score_mean", smoke_only=True,
                notes=["FUNCTIONALITY CHECK ONLY: 32 steps, small networks; no performance conclusions."])


def prepare_search_registry(args):
    from experiments.search import METHODS, SPACES, validate_space
    if args.search_space_file:
        space = read_json(args.search_space_file)
        validate_space(space)
        SPACES["custom"] = space
        args.spaces = ["custom"]
    if args.search_methods_file:
        import math
        methods = read_json(args.search_methods_file)
        for name, settings in methods.items():
            if not name.startswith("pso") or set(settings) != {"w_start", "w_end", "c1", "c2"}:
                raise ValueError("Custom PSO methods need pso-prefixed names and w_start/w_end/c1/c2")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in settings.values()):
                raise ValueError("PSO coefficients must be finite and nonnegative")
        METHODS.update(methods)
        args.methods = ["baseline", "random", "grid", *methods]
    methods = list(METHODS) if "all" in args.methods else args.methods
    spaces = list(SPACES) if "all" in args.spaces else args.spaces
    if set(methods) - set(METHODS) or set(spaces) - set(SPACES):
        raise ValueError("Unknown search methods/spaces")
    if args.candidates < 1 or args.particles < 1 or args.search_steps < 1:
        raise ValueError("Search budgets must be positive")
    for method in methods:
        size = args.particles * (2 if method == "pso_large_swarm" else 1)
        if method.startswith("pso") and (args.candidates < size or args.candidates % size):
            raise ValueError("Every PSO swarm size must divide the shared candidate budget")
    from experiments.search import prepare_search
    base = base_config(args)
    for algorithm in args.algorithms:
        for space in spaces:
            prepare_search(base, algorithm, space)
    if set(args.train_seeds) & set(args.final_train_seeds):
        raise ValueError("Search/final training seeds must be disjoint")
    for seeds in (args.search_seeds, args.train_seeds, args.final_train_seeds):
        if not seeds or len(seeds) != len(set(seeds)) or any(not 0 <= s < 2**32 for s in seeds):
            raise ValueError("Use unique nonnegative seeds below 2**32")
    objectives = args.objectives or [args.objective]
    if set(objectives) - (set(METRICS) | {"all"}):
        raise ValueError("Unknown objective")
    if "return_mean" in objectives and any(any(p.startswith("rewards.") for p in SPACES[s]) for s in spaces):
        raise ValueError("Use common-reference return for comparisons that vary reward magnitudes")


def search_objectives(args, out, base):
    from experiments.search import run_search
    prepare_search_registry(args)
    objectives = args.objectives or [args.objective]
    if "all" in objectives:
        objectives = ["score_mean", "reference_return_mean", "clear_rate", "pellet_fraction_mean",
                      "ghosts_eaten_mean", "lives_lost_mean", "blocked_action_rate", "steps_mean"]
    for objective in objectives:
        if objective not in METRICS:
            raise ValueError("Unknown objective: " + objective)
        destination = Path(out) / objective if len(objectives) > 1 else Path(out)
        remote = Path(args.backup) / objective if args.backup and len(objectives) > 1 else args.backup
        run_search(destination, base, args.algorithms, args.spaces, args.methods,
                   args.search_seeds, args.train_seeds, args.final_train_seeds,
                   candidates=args.candidates, particles=args.particles, search_steps=args.search_steps,
                   objective=objective, backup=remote, final=not args.no_final,
                   retain_candidate_models=args.retain_candidate_models)


def add_search_arguments(parser, extensive=False):
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS,
                        default=["dqn", "double_dqn", "sarsa", "expected_sarsa", "hra", "ppo"]
                        if extensive else ["dqn", "sarsa", "hra"])
    parser.add_argument("--spaces", nargs="+", default=["all"])
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--search-seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--final-train-seeds", nargs="+", type=int, default=[606, 707, 808])
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--search-steps", type=int, default=20000)
    parser.add_argument("--objective", choices=METRICS, default="score_mean")
    parser.add_argument("--objectives", nargs="+",
                        default=["score_mean", "reference_return_mean", "clear_rate"] if extensive else None,
                        help="multiple fitness objectives, or all")
    parser.add_argument("--retain-candidate-models", action="store_true",
                        help="keep all search weights; default keeps metrics/logs and final retrained models")
    parser.add_argument("--search-space-file", help="JSON map of configuration paths to bounds/choices")
    parser.add_argument("--search-methods-file", help="JSON map of pso method names to w_start/w_end/c1/c2")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="list all comparison axes and search spaces")
    plan_parser = sub.add_parser("plan", help="write an inspectable plan without training")
    add_plan_arguments(plan_parser)
    run_parser = sub.add_parser("run", help="train, evaluate, save PNGs; rerun same command to resume")
    add_plan_arguments(run_parser)
    run_parser.add_argument("--workers", type=int, default=2)
    run_parser.add_argument("--backup", help="mounted Drive directory (training remains local)")
    run_parser.add_argument("--no-final", action="store_true")
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.add_argument("--report-every", type=int, default=25)
    report_parser = sub.add_parser("report", help="regenerate PNGs and tables")
    report_parser.add_argument("--out", required=True)
    report_parser.add_argument("--window", type=int, default=50)
    smoke = sub.add_parser("smoke", help="test every algorithm, power rewards and gamma, using tiny budgets")
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--workers", type=int, default=1)
    search = sub.add_parser("search", help="compare PSO settings, random/grid search and untuned baseline")
    search.add_argument("--out", required=True)
    search.add_argument("--base")
    add_search_arguments(search)
    search.add_argument("--backup")
    search.add_argument("--no-final", action="store_true")
    all_parser = sub.add_parser("all", help="run the full factor catalog and PSO comparisons sequentially; resumable")
    add_plan_arguments(all_parser)
    add_search_arguments(all_parser, extensive=True)
    all_parser.add_argument("--workers", type=int, default=2)
    all_parser.add_argument("--backup")
    all_parser.add_argument("--no-final", action="store_true")
    all_parser.add_argument("--stages", nargs="+", choices=("factors", "search"), default=["factors", "search"])
    all_parser.add_argument("--plan-only", action="store_true", help="show/write full budgets without training")
    all_parser.add_argument("--retry-failed", action="store_true")
    all_parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        from experiments.search import METHODS, SPACES
        print(json.dumps({**{k: dict(description=v["description"], variants=len(v["variants"])) for k,v in catalog().items()},
                          "search_spaces": SPACES, "search_methods": METHODS}, ensure_ascii=False, indent=1))
    elif args.command == "plan":
        from experiments.engine import atomic_json
        plan = make_plan(args)
        atomic_json(Path(args.out) / "proposed_plan.json", plan)
        print(json.dumps({k: plan[k] for k in ("unique_conditions", "training_trials", "total_environment_steps")}, indent=1))
    elif args.command in ("run", "smoke"):
        from experiments.runner import run_plan
        if args.command == "smoke":
            run_plan(smoke_plan(), args.out, workers=args.workers, final=True, report_every=0)
        else:
            run_plan(make_plan(args), args.out, workers=args.workers, backup=args.backup,
                     final=not args.no_final, retry_failed=args.retry_failed, report_every=args.report_every)
    elif args.command == "report":
        from experiments.report import report
        print(report(args.out, window=args.window))
    elif args.command == "search":
        search_objectives(args, args.out, base_config(args))
    elif args.command == "all":
        from experiments.engine import atomic_json
        from experiments.search import METHODS, SPACES
        from experiments.runner import run_plan
        import copy
        prepare_search_registry(args)
        plan = make_plan(args)
        methods = list(METHODS) if "all" in args.methods else args.methods
        spaces = list(SPACES) if "all" in args.spaces else args.spaces
        objectives = args.objectives or [args.objective]
        if "all" in objectives:
            objectives = [k for k in METRICS if k != "return_mean"]
        group_count = len(args.algorithms) * len(spaces) * len(objectives) * len(args.search_seeds)
        proposals = group_count * sum(1 if m == "baseline" else args.candidates for m in methods)
        search_steps = proposals * len(args.train_seeds) * args.search_steps
        refit_steps = group_count * len(methods) * len(args.final_train_seeds) * (
            base_config(args)["train"]["steps"]) if not args.no_final else 0
        if "search" not in args.stages:
            search_steps = refit_steps = 0
        factor_steps = plan["total_environment_steps"] if "factors" in args.stages else 0
        overview = dict(factor_plan=plan, search_groups=group_count, search_proposals=proposals,
                        search_environment_steps=search_steps, refit_environment_steps=refit_steps,
                        total_environment_steps=factor_steps + search_steps + refit_steps,
                        stages=args.stages, search_settings=vars(args),
                        x_axis="learning episodes", finite_catalog=True)
        atomic_json(Path(args.out) / "all_plan.json", overview)
        print(json.dumps({k:v for k,v in overview.items() if k not in ("factor_plan", "search_settings")}, indent=1), flush=True)
        if args.plan_only:
            return
        completed = []
        if "factors" in args.stages:
            run_plan(plan, Path(args.out) / "factors", workers=args.workers,
                     backup=Path(args.backup) / "factors" if args.backup else None,
                     final=not args.no_final, retry_failed=args.retry_failed, report_every=args.report_every)
            completed.append("factors")
            atomic_json(Path(args.out) / "all_progress.json", dict(completed_stages=completed))
        if "search" in args.stages:
            options = copy.copy(args)
            options.backup = str(Path(args.backup) / "search") if args.backup else None
            search_objectives(options, Path(args.out) / "search", base_config(args))
            completed.append("search")
        atomic_json(Path(args.out) / "ALL_COMPLETE.json", dict(completed_stages=completed, x_axis="learning episodes"))


if __name__ == "__main__":
    main()
