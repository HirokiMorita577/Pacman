"""Create a separate, inspectable 1M-step shortlist; never mutate the 200k study."""
import argparse
import json
from pathlib import Path

from experiments.config import DEFAULT, digest, merge, validate

SEEDS = (101, 202, 303, 404, 505)
SETTINGS = (
    ("dqn", 50, 0.95, "common_control"),
    ("double_dqn", 50, 0.95, "common_control"),
    ("sarsa", 50, 0.95, "common_control"),
    ("q_learning", 50, 0.95, "common_control_unoptimized"),
    ("dqn", 50, 0.99, "prior_exploratory_best"),
    ("double_dqn", 10, 0.995, "prior_exploratory_best"),
    ("sarsa", 0, 0.95, "prior_exploratory_best"),
)


def make_plan():
    conditions = []
    for algorithm, power, gamma, role in SETTINGS:
        config = validate(merge(DEFAULT, {
            "algorithm": algorithm,
            "rewards": {"power": power},
            "agent": {"gamma": gamma},
            "train": {"steps": 1_000_000, "eval_every": 20_000,
                      "curve_episodes": 100, "checkpoint_episodes": 50},
            "evaluation": {"episodes": 100, "dev_seed0": 2_000_000,
                           "final_seed0": 3_000_000},
        }))
        conditions.append(dict(id=digest(config)[:20], role=role, config=config))
    assert len({x["id"] for x in conditions}) == len(conditions)
    return dict(schema=1, status="plan_only_not_trained",
                source="C:/PRO/Pacman/plan_long_study.py",
                seeds=list(SEEDS), conditions=conditions,
                trials=len(conditions) * len(SEEDS),
                minimum_environment_steps=len(conditions) * len(SEEDS) * 1_000_000,
                evaluation_interval_steps=20_000,
                validation_seed_range=[2_000_000, 2_000_099],
                confirmation_seed_start=2_100_000,
                final_test_seed_range=[3_000_000, 3_000_099],
                note="Prior best settings were selected on old evaluations; new final test seeds are separate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = make_plan()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PLAN_ONLY", plan["trials"], "trials", plan["minimum_environment_steps"],
          "minimum environment steps", target)
