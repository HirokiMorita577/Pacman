"""Measure the observation layout, network dimensions, and optimizer at runtime."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.agents import Agent
from experiments.config import validate
from experiments.engine import make_env


def describe(config):
    config = validate(config)
    env = make_env(config)
    obs, _ = env.reset(seed=12345)
    size = env.observation_space.shape[0]
    sections = []
    cursor = 0

    def add(name, count):
        nonlocal cursor
        sections.append(dict(name=name, count=count, start=cursor, end=cursor + count - 1))
        cursor += count

    n = env.num_ghosts
    add("wall_dot_power_ghost_channels", env.rows * env.cols * 4)
    add("pacman_and_ghost_positions", 2 + 2 * n)
    add("distances_directions_power_timer", 1 + n + 4 + 1)
    add("action_features", 4 * 4)
    if env.include_lives:
        add("lives_ratio", 1)
    if env.observation_version != "legacy":
        add("headings_ghost_state", 4 + 7 * n)
    if env.observation_version == "tactical_safe":
        add("safety_lookahead", 4 * 3)
        if env.safe_route:
            add("safe_route", 4 * 3)
        if env.classic_lookahead:
            add("classic_ghost_lookahead", 4 * 3)
    assert cursor == size == len(obs)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all() and (obs >= 0).all() and (obs <= 1).all()
    agent = Agent(config, size, env.action_mask_indices)
    batch = torch.as_tensor(np.stack([obs, obs]), dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        output = agent.network(batch)
    assert output.shape == (2, 4)
    linears = [layer for layer in agent.network.modules() if isinstance(layer, nn.Linear)]
    layers = [dict(input=layer.in_features, output=layer.out_features,
                   parameters=(layer.in_features + 1) * layer.out_features)
              for layer in linears]
    assert sum(layer["parameters"] for layer in layers) == agent.parameter_count
    groups = agent.optimizer.param_groups
    assert len(groups) == 1
    optimizer = {key: value for key, value in groups[0].items()
                 if key != "params" and isinstance(value, (str, int, float, bool, tuple, list, type(None)))}
    specification = dict(algorithm=config["algorithm"], observation_size=size,
                         batch_output_shape=list(output.shape), parameter_count=agent.parameter_count,
                         linear_layers=layers, activation=config["agent"]["activation"],
                         optimizer_name=type(agent.optimizer).__name__, optimizer_parameters=optimizer,
                         action_order=["up", "down", "left", "right"],
                         mask_indices=list(env.action_mask_indices),
                         learning_rate=config["agent"]["lr"])
    observation = dict(rows=env.rows, columns=env.cols, num_ghosts=n,
                       dtype=str(obs.dtype), shape=list(obs.shape),
                       sections=sections, action_mask_indices=list(env.action_mask_indices))
    env.close()
    return specification, observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    spec, observation = describe(plan["conditions"][0]["config"])
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "network_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (target / "observation_schema.json").write_text(json.dumps(observation, indent=2), encoding="utf-8")
    lines = ["# Network specification", "", f"Input: {spec['observation_size']} float32 values",
             f"Output: 4 linear Q values", f"Parameters: {spec['parameter_count']:,}", "",
             "| Linear | Input | Output | Parameters |", "|---:|---:|---:|---:|"]
    lines += [f"| {i} | {x['input']} | {x['output']} | {x['parameters']:,} |"
              for i, x in enumerate(spec["linear_layers"], 1)]
    lines += ["", f"Optimizer: {spec['optimizer_name']}", "",
              "Optimizer parameters are read from the instantiated param_groups;",
              "the JSON file records their actual values."]
    (target / "network_spec.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("NETWORK_VERIFIED", spec["observation_size"], spec["parameter_count"],
          spec["batch_output_shape"], target)


if __name__ == "__main__":
    main()
