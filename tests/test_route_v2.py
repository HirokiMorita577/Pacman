"""Route v2: endgame distances, equal-length paths and training integration."""
import copy

import numpy as np
import pytest

from pacman_env import ACTION_RIGHT, CELL_WALL, DIRS, PacManEnv


def make_env(**kwargs):
    return PacManEnv(num_ghosts=4, lives=3, include_lives=True,
                     observation_version="tactical_safe", safe_route=True,
                     route_version="v2", **kwargs)


def graph_env(cells, start, arrivals=None):
    """Small graphs isolate route selection from ghost movement heuristics."""
    env = make_env()
    env.rows = max(r for r, c in cells) + 2
    env.cols = max(c for r, c in cells) + 2
    env.grid = [[CELL_WALL] * env.cols for _ in range(env.rows)]
    for r, c in cells:
        env.grid[r][c] = 0
    env.pacman = list(start)
    field = [[-1] * env.cols for _ in range(env.rows)]
    for cell, arrival in (arrivals or {}).items():
        field[cell[0]][cell[1]] = arrival
    env._threat_field = lambda: field
    return env


def features(env, foods):
    return env._route_features(set(foods)).reshape(4, 3)


def test_far_food_stays_positive_and_distinguishable():
    env = graph_env({(1, c) for c in range(1, 63)}, (1, 1))
    near = features(env, {(1, 32)})[ACTION_RIGHT]
    far = features(env, {(1, 62)})[ACTION_RIGHT]
    assert 0 < far[0] < near[0] < 1
    assert np.allclose(near, [0.25, 0.25, 1])
    assert far[0] == far[1]


def test_unsafe_food_still_has_a_distance():
    env = graph_env({(1, c) for c in range(1, 8)}, (1, 1), {(1, 4): 1})
    f = features(env, {(1, 7)})[ACTION_RIGHT]
    assert f[0] > 0 and f[1] == 0 and f[2] == 0


def test_unreachable_or_missing_food_has_zero_features():
    env = graph_env({(1, 1), (1, 2), (1, 4)}, (1, 1))
    assert not features(env, {(1, 4)}).any()
    assert not features(env, set()).any()


def diamond():
    # Initial action enters (2,2), then two equal paths meet at (2,4).
    return {(2, 1), (2, 2), (2, 4),
            (1, 2), (1, 3), (1, 4), (3, 2), (3, 3), (3, 4)}


def test_safer_equal_length_path_survives_merge():
    env = graph_env(diamond(), (2, 1), {(1, 3): 3, (2, 4): 9})
    # The first BFS path (upper branch) is unsafe, lower path has margin 4.
    f = features(env, {(2, 4)})[ACTION_RIGHT]
    assert np.allclose(f, [10 / 14, 10 / 14, 0.4])


def test_equal_distance_food_chooses_the_safer_goal():
    cells = {(2, 1), (2, 2), (1, 2), (3, 2)}
    env = graph_env(cells, (2, 1), {(1, 2): 2, (3, 2): 8})
    f = features(env, {(1, 2), (3, 2)})[ACTION_RIGHT]
    assert np.allclose(f, [10 / 11, 10 / 11, 0.6])


def test_safe_detour_is_measured_separately_from_shortest_path():
    cells = {(2, c) for c in range(1, 6)} | {(3, c) for c in range(2, 6)}
    env = graph_env(cells, (2, 1), {(2, 3): 2})
    f = features(env, {(2, 5)})[ACTION_RIGHT]
    assert np.allclose(f, [10 / 13, 10 / 15, 0])


def test_collision_on_first_step_cannot_be_avoided_by_later_path():
    env = graph_env({(1, c) for c in range(1, 5)}, (1, 1), {(1, 2): 1})
    f = features(env, {(1, 4)})[ACTION_RIGHT]
    assert f[0] > 0 and f[1] == 0 and f[2] == 0


def test_immediate_food_and_wall_action():
    env = graph_env({(1, 1), (1, 2)}, (1, 1))
    assert np.array_equal(features(env, {(1, 2)})[ACTION_RIGHT], [1, 1, 1])
    # A wall action stays in place, as in PacManEnv.step; masking is unchanged.
    assert np.allclose(features(env, {(1, 2)})[0], [10 / 11, 10 / 11, 1])


@pytest.mark.parametrize("kwargs", [
    {"route_version": "unknown", "safe_route": True},
    {"route_version": "v2"},
    {"route_version": "v2", "safe_route": True, "observation_version": "tactical"},
])
def test_invalid_combinations_fail_early(kwargs):
    with pytest.raises(ValueError):
        PacManEnv(**kwargs)


def test_existing_inputs_and_action_mask_are_unchanged():
    base = PacManEnv(num_ghosts=4, lives=3, include_lives=True,
                     observation_version="tactical_safe", classic_lookahead=True)
    old = PacManEnv(num_ghosts=4, lives=3, include_lives=True,
                    observation_version="tactical_safe", safe_route=True, classic_lookahead=True)
    new = make_env(classic_lookahead=True)
    a, _ = base.reset(seed=9)
    b, _ = old.reset(seed=9)
    c, _ = new.reset(seed=9)
    assert len(c) == len(a) + 12 == len(b) + 4
    assert np.array_equal(c[:-24], a[:-12])
    assert np.array_equal(c[-12:], a[-12:])  # classic lookahead remains last
    assert new.action_mask_indices == base.action_mask_indices == old.action_mask_indices
    assert new.observation_space.contains(c)


def test_legacy_is_still_the_default_over_a_rollout():
    options = dict(num_ghosts=4, lives=3, include_lives=True,
                   observation_version="tactical_safe", safe_route=True,
                   classic_lookahead=True)
    implicit = PacManEnv(**options)
    explicit = PacManEnv(**options, route_version="legacy")
    a, _ = implicit.reset(seed=23)
    b, _ = explicit.reset(seed=23)
    assert np.array_equal(a, b)
    for action in np.random.default_rng(23).integers(4, size=100):
        a, ar, at, ax, ai = implicit.step(int(action))
        b, br, bt, bx, bi = explicit.step(int(action))
        assert np.array_equal(a, b) and (ar, at, ax) == (br, bt, bx)
        assert all(np.array_equal(ai[k], bi[k]) for k in ai)
        if at or ax:
            break


def test_features_do_not_change_state_or_randomness():
    env = make_env(classic_lookahead=True, safety_timing=True)
    env.reset(seed=7)
    for g in env.ghosts:
        g.scared, g.scared_timer = True, 3
    before = copy.deepcopy((env.grid, env.pacman, [vars(g) for g in env.ghosts],
                            env.rng.bit_generator.state))
    assert np.array_equal(env._get_obs(), env._get_obs())
    assert (env.grid, env.pacman, [vars(g) for g in env.ghosts],
            env.rng.bit_generator.state) == before
    for i in range(80):
        obs, _, te, tr, _ = env.step(i % len(DIRS))
        assert env.observation_space.contains(obs)
        if te or tr:
            env.reset(seed=i)


def test_training_saves_and_reuses_route_configuration(tmp_path, monkeypatch):
    import reward_study as study
    from dqn_agent import load_agent
    from ghost4_round5 import make_job

    monkeypatch.setitem(study.FIXED, "buffer", 256)
    job = make_job("power50", 101, 160, tmp_path)
    job.update(exp="route_v2_smoke", eval_episodes=1, curve_eval=None,
               show_progress=False, checkpoint_every=1)
    job["env_options"] = {**job["env_options"], "safe_route": True, "route_version": "v2"}
    old_job = {**job, "env_options": {**job["env_options"], "route_version": "legacy"}}
    assert study.job_id(job) != study.job_id(old_job)
    result = study.run_job(job)
    assert result["env_options"] == job["env_options"]
    assert study.run_job(job)["eval"] == result["eval"]  # completed run cache
    env = make_env(classic_lookahead=True)
    agent = load_agent(str(study.Path(job["runs_dir"]) / result["id"] / "model.pt"),
                       env.observation_space.shape[0], 4)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_agent(str(study.Path(job["runs_dir"]) / result["id"] / "model.pt"),
                   env.observation_space.shape[0] - 4, 4)  # legacy route input
    ev = study.evaluate(agent, job["rewards"], 1, job["lives"], seed0=job["eval_seed0"],
                        include_lives=True, num_ghosts=4, observation_version="tactical_safe",
                        env_options=job["env_options"])
    assert ev["scores"] == result["eval"]["scores"]
    assert ev["clears"] == result["eval"]["clears"]
