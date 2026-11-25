import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from comparison_study import smoke_plan
from experiments.agents import Agent
from experiments.catalog import build_plan, catalog
from experiments.config import DEFAULT, ONLINE, HEURISTICS, digest, epsilon_at, merge, trial_id, validate
from experiments.engine import (MeasuredEnv, evaluate, evaluate_final, learning_reward, load_saved,
                                make_env, rng_state, run_trial, torch_load)
from experiments.search import METHODS, SPACES, decode, encode, optimize


def tiny(algorithm="double_dqn"):
    config = merge(DEFAULT, {
        "algorithm": algorithm, "agent": {"hidden_sizes": [8], "gamma": 0.5},
        "env": {"classic_lookahead": False, "max_steps": 4},
        "train": {"steps": 20, "explore_steps": 12, "checkpoint_episodes": 2,
                  "eval_every": 5, "curve_episodes": 1},
        "evaluation": {"episodes": 2, "dev_seed0": 1000, "final_seed0": 2000},
    })
    if algorithm == "ppo":
        config["algorithm_options"] = dict(rollout_steps=10, epochs=1, batch_size=4)
    elif algorithm not in ONLINE | HEURISTICS:
        config["algorithm_options"] = dict(batch_size=2, warmup=2, buffer_capacity=32, target_update=3)
    return validate(config)


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, observations):
        return self.values.expand(len(observations), -1)


class FixedHeads(FixedQ):
    def head_q(self, observations):
        return self.values.expand(len(observations), -1, -1)

    def forward(self, observations):
        return self.head_q(observations).sum(dim=1)


@pytest.mark.parametrize("algorithm,expected", [
    ("dqn", 5.5), ("double_dqn", 3.0), ("dueling_double_dqn", 3.0),
    ("q_learning", 4.0), ("sarsa", 1.5), ("expected_sarsa", 3.75), ("replay_sarsa", 5.5),
])
def test_td_equations_use_the_correct_policy_and_legal_actions(algorithm, expected):
    agent = Agent(tiny(algorithm), 4, [0, 1, 2, 3])
    agent.network = FixedQ([1, 100, 6, 90])
    if agent.target is not None:
        agent.target = FixedQ([9, 100, 4, 99])
    result = agent.td_targets(torch.tensor([1.]), torch.tensor([[1., 0., 1., 0.]]),
                              torch.tensor([0.]), torch.tensor([0]), torch.tensor([0.2]))
    assert result.item() == pytest.approx(expected)
    terminal = agent.td_targets(torch.tensor([1.]), torch.zeros(1, 4),
                                torch.tensor([1.]), torch.tensor([0]), torch.tensor([0.2]))
    assert terminal.item() == 1


def test_hra_uses_one_shared_next_action_for_all_heads():
    agent = Agent(tiny("hra"), 4, [0, 1, 2, 3])
    agent.network = FixedHeads([[5, 0, 1, 0], [0, 0, 9, 0], [0, 0, 3, 0]])
    agent.target = FixedHeads([[8, 4, 1, 0], [0, 9, 2, 0], [0, 0, 3, 7]])
    target = agent.td_targets(torch.tensor([[1., 2., 3.]]), torch.tensor([[1., 0., 1., 0.]]),
                              torch.tensor([0.]), torch.tensor([0]), torch.tensor([0.2]))
    assert torch.allclose(target, torch.tensor([[1.5, 3., 4.5]]))


@pytest.mark.parametrize("algorithm", ["double_dqn", "sarsa", "expected_sarsa", "ppo", "random"])
def test_action_masks_apply_to_exploration_and_evaluation(algorithm):
    agent = Agent(tiny(algorithm), 4, [0, 1, 2, 3])
    for _ in range(30):
        assert agent.action(np.array([1, 0, 1, 0], dtype=np.float32), 1, training=True) in (0, 2)
        assert agent.action(np.array([1, 0, 1, 0], dtype=np.float32), training=False) in (0, 2)
    assert agent.action(np.zeros(4, dtype=np.float32), training=False) in range(4)


def test_online_sarsa_does_not_accidentally_use_experience_replay():
    config = tiny("sarsa")
    assert not Agent(config, 4, None).options["replay"]
    with pytest.raises(ValueError, match="Online"):
        validate(merge(config, {"algorithm_options": {"replay": True}}))


def test_hra_reward_clipping_preserves_the_common_scalar_total():
    base = tiny("double_dqn")
    base["train"].update(reward_clip=1, reward_normalization="none")
    hra = merge(base, {"algorithm": "hra"})
    components = np.array([100., -30., 5.])
    scalar, heads = learning_reward(base, components), learning_reward(hra, components)
    assert scalar == pytest.approx(1)
    assert heads.sum() == pytest.approx(scalar)


def test_reference_return_does_not_disappear_when_experimental_reward_is_zero():
    config = tiny()
    config["rewards"] = {k: 0.0 for k in config["rewards"]}
    env = make_env(config)
    env.reset(seed=1)
    obs, reward, _, _, _ = env.step(2)
    assert reward == 0
    assert env.reference_return != 0
    env.reset(seed=1)
    assert env.reference_return == 0


def test_evaluation_preserves_training_rng_and_uses_distinct_fixed_seeds():
    config = tiny()
    env = make_env(config)
    agent = Agent(config, env.observation_space.shape[0], env.action_mask_indices)
    before = rng_state()
    dev, final = evaluate(agent, config), evaluate(agent, config, phase="final")
    after = rng_state()
    assert before["python"] == after["python"]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert torch.equal(before["torch"], after["torch"])
    assert {r["evaluation_seed"] for r in dev["episodes"]} == {1000, 1001}
    assert {r["evaluation_seed"] for r in final["episodes"]} == {2000, 2001}


@pytest.mark.parametrize("algorithm", ["double_dqn", "sarsa", "expected_sarsa", "hra", "ppo"])
def test_episode_boundary_resume_matches_uninterrupted_weights_and_evaluation(tmp_path, algorithm):
    config = tiny(algorithm)
    full = run_trial(config, 7, tmp_path / "full")
    paused = run_trial(config, 7, tmp_path / "resume", stop_after_episodes=2)
    assert paused["paused"] and paused["total_steps"] == 8
    resumed = run_trial(config, 7, tmp_path / "resume")
    a = torch_load(tmp_path / "full" / "runs" / full["id"] / "model.pt")
    b = torch_load(tmp_path / "resume" / "runs" / resumed["id"] / "model.pt")
    assert all(torch.equal(a["agent"]["network"][k], b["agent"]["network"][k]) for k in a["agent"]["network"])
    assert full["updates"] == resumed["updates"]
    assert full["evaluation"] == resumed["evaluation"]
    assert run_trial(config, 7, tmp_path / "resume")["evaluation"] == resumed["evaluation"]
    root = tmp_path / "resume" / "runs" / resumed["id"]
    evaluation = evaluate_final(root)
    assert evaluation["phase"] == "final"
    assert evaluate_final(root) == evaluation


def test_sarsa_executes_the_next_behaviour_action_it_stored(tmp_path, monkeypatch):
    transitions = []
    original = Agent.observe

    def observe(self, obs, action, reward, nxt, done, boundary, next_action=None, epsilon=0):
        if self.algorithm == "sarsa":
            transitions.append((action, next_action, boundary))
        return original(self, obs, action, reward, nxt, done, boundary, next_action, epsilon)

    monkeypatch.setattr(Agent, "observe", observe)
    run_trial(tiny("sarsa"), 8, tmp_path)
    pairs = [(a, b) for a, b in zip(transitions, transitions[1:]) if not a[2]]
    assert len(pairs) >= 5
    assert all(before[1] == after[0] for before, after in pairs)


def test_full_catalog_is_finite_valid_and_reuses_identical_trials():
    plan = build_plan(["all"], [101, 202])
    assert len(catalog()) >= 60
    assert len(plan["conditions"]) == len({c["id"] for c in plan["conditions"]})
    assert plan["training_trials"] == len(plan["conditions"]) * 2
    assert plan["total_environment_steps"] == sum(c["config"]["train"]["steps"] for c in plan["conditions"]) * 2
    for condition in plan["conditions"]:
        validate(condition["config"])
    assert "reward_power" in plan["groups"] and "power_gamma" in plan["groups"]
    conditions = {c["id"]: c["config"] for c in plan["conditions"]}
    assert {conditions[m["condition"]]["rewards"]["power"] for m in plan["groups"]["reward_power"]} >= {0, 10, 50, 200}


def test_custom_grid_crosses_reward_gamma_and_algorithm():
    grid = dict(name="my_grid", axes={"algorithm": ["dqn", "sarsa"], "rewards.power": [0, 200],
                                      "agent.gamma": [0.9, 0.99]})
    plan = build_plan([], [1], grid=grid)
    assert plan["unique_conditions"] == 8
    assert len(plan["groups"]["my_grid"]) == 8
    with pytest.raises(ValueError):
        build_plan(["not_a_suite"], [1])
    with pytest.raises(ValueError):
        build_plan(["gamma"], [1, 1])


@pytest.mark.parametrize("schedule", ["linear", "exponential", "cosine"])
def test_exploration_uses_learning_steps_independent_of_episode_length(schedule):
    config = tiny()
    config["train"].update(schedule=schedule, epsilon_start=1.0, epsilon_end=0.1, explore_steps=100)
    assert epsilon_at(config, 0) == 1
    assert epsilon_at(config, 100) == 0.1
    assert 1 > epsilon_at(config, 50) > 0.1


@pytest.mark.parametrize("patch", [
    {"rewards": {"bogus": 1}},
    {"agent": {"gamma": 1.5}},
    {"train": {"epsilon_end": -1}},
    {"evaluation": {"final_seed0": 1000}},
    {"algorithm_options": {"unknown": 1}},
    {"env": {"observation_version": "legacy", "classic_lookahead": True}},
])
def test_invalid_comparisons_fail_before_training(patch):
    with pytest.raises((ValueError, TypeError)):
        validate(merge(tiny(), patch))


@pytest.mark.parametrize("method", ["random", "grid", "pso", "pso_constant_w", "pso_large_swarm"])
def test_search_resume_preserves_actual_fitness_history(tmp_path, method):
    config = tiny()
    space = {"agent.gamma": dict(low=0.1, high=0.99)}

    def fitness(candidate):
        return dict(fitness=-(candidate["agent"]["gamma"] - 0.7)**2, learning_episodes=3)

    full = optimize(method, config, space, 42, 8, 2, tmp_path / "full", fitness)
    paused = optimize(method, config, space, 42, 8, 2, tmp_path / "resume", fitness, stop_after=3)
    assert not paused["complete"]
    resumed = optimize(method, config, space, 42, 8, 2, tmp_path / "resume", fitness)
    assert full == resumed
    assert len(resumed["history"]) == 8
    assert all(0.1 <= r["parameters"]["agent.gamma"] <= 0.99 for r in resumed["history"])
    assert [r["cumulative_learning_episodes"] for r in resumed["history"]] == list(range(3, 25, 3))
    for row in resumed["history"]:
        assert row["fitness"] == pytest.approx(-(row["parameters"]["agent.gamma"] - 0.7)**2)


def test_log_integer_and_categorical_search_parameters_round_trip():
    config = tiny()
    space = {"agent.lr": dict(low=1e-5, high=1e-2, scale="log"),
             "agent.gamma": dict(choices=[0.5, 0.9, 0.99]),
             "env.lives": dict(low=1, high=5, type="int")}
    values = decode(encode(config, space), space)
    assert values["agent.lr"] == pytest.approx(config["agent"]["lr"])
    assert values["agent.gamma"] == config["agent"]["gamma"]
    assert values["env.lives"] == config["env"]["lives"]


def test_legacy_pso_cur_scores_are_not_personal_best_scores():
    from pso_optimizer import PSO
    values = iter([10.0, 9.0, 1.0, 2.0])
    _, history = PSO(n_particles=2, n_iterations=2).optimize(lambda _: next(values), [(0, 1)], verbose=False)
    assert history[1]["cur_scores"] == [1.0, 2.0]
    assert history[1]["pbest_scores"] == [10.0, 9.0]


def test_seed_bootstrap_counts_training_seeds_not_play_episodes():
    from experiments.report import seed_interval
    mean, low, high = seed_interval([10, 20, 30])
    assert mean == 20 and low <= mean <= high
    mean, low, high = seed_interval([10])
    assert mean == 10 and low is None and high is None


def test_equivalent_numeric_reward_definitions_reuse_one_trial():
    a = tiny()
    b = merge(a, {"rewards": {"power": 50}})
    assert trial_id(a, 1) == trial_id(b, 1)
