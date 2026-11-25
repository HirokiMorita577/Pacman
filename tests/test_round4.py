"""2026-10-10 の追加 (classic_lookahead: 敵の実際の動き方による先読み、第4回の計画) のテスト"""
import argparse
import copy

import numpy as np
import pytest

import reward_study as study
from pacman_env import PacManEnv, DIRS, CELL_WALL, LOOKAHEAD_STEPS


def make_env(**kw):
    return PacManEnv(num_ghosts=4, lives=3, include_lives=True, observation_version="tactical_safe",
                     classic_lookahead=True, **kw)


def walls_of(env):
    pr, pc = env.pacman
    return [a for a, (dr, dc) in enumerate(DIRS)
            if not (0 <= pr + dr < env.rows and 0 <= pc + dc < env.cols and env.grid[pr + dr][pc + dc] != CELL_WALL)]


def test_observation_adds_twelve_bounded_features_and_keeps_layout():
    base = PacManEnv(num_ghosts=4, lives=3, include_lives=True, observation_version="tactical_safe")
    env = make_env()
    assert env.observation_space.shape[0] == base.observation_space.shape[0] + 12
    a, _ = base.reset(seed=4)
    b, _ = env.reset(seed=4)
    assert np.array_equal(b[:-12], a)
    rng = np.random.default_rng(4)
    for _ in range(150):
        o, _, te, tr, _ = env.step(int(rng.integers(4)))
        assert o.min() >= 0 and o.max() <= 1
        if te or tr:
            break


def test_prediction_matches_real_ghosts_while_pacman_stays():
    """パックマンが壁に向かって留まる間、予測した敵の位置は実際の位置と一致する (怖がった敵は分岐のどれか)"""
    env = make_env()
    rng = np.random.default_rng(2)
    checked = 0
    for seed in range(6):
        env.reset(seed=seed)
        for step in range(120):
            walls = walls_of(env)
            if walls and step % 6 == 0:
                for scared in (False, True):
                    e = copy.deepcopy(env)
                    if scared:
                        for g in e.ghosts:
                            g.scared, g.scared_timer = True, int(rng.integers(1, 8))
                    timeline = e.predict_ghosts(tuple(e.pacman), e.pacman_dir, LOOKAHEAD_STEPS)
                    lives = e.lives
                    for t in range(1, LOOKAHEAD_STEPS + 1):
                        eaten = e.ghosts_eaten
                        _, _, te, tr, info = e.step(walls[0])
                        if info["lives"] < lives or e.ghosts_eaten != eaten or te or tr:
                            break
                        for gi, g in enumerate(e.ghosts):
                            pred = {(s[0], s[2]) for s in timeline[t][gi]}
                            assert (tuple(g.pos), g.scared) in pred
                            if not scared:
                                assert len(pred) == 1
                            checked += 1
            _, _, te, tr, _ = env.step(int(rng.integers(4)))
            if te or tr:
                break
    assert checked > 200


def test_features_do_not_change_game_state_or_rng():
    env = make_env()
    env.reset(seed=1)
    for _ in range(20):
        env.step(0)
    before = (copy.deepcopy(env.rng.bit_generator.state), copy.deepcopy([vars(g) for g in env.ghosts]), list(env.pacman))
    env._get_obs()
    assert (env.rng.bit_generator.state, [vars(g) for g in env.ghosts], list(env.pacman)) == before


def test_moving_into_a_chasing_ghost_leaves_no_escape():
    """隣のマスの先から危険な敵が向かってくる通路では、その向きの行動は1手目で捕まる (逃げ場0)"""
    env = make_env()
    for seed in range(30):
        env.reset(seed=seed)
        pr, pc = env.pacman
        for a, (dr, dc) in enumerate(DIRS):
            n1 = (pr + dr, pc + dc)
            if env.grid[n1[0]][n1[1]] == CELL_WALL:
                continue
            g = env.ghosts[0]
            for g2 in env.ghosts[1:]:
                g2.scared, g2.scared_timer = True, 40
            g.pos, g.last_pos, g.scared, g.scared_timer = list(n1), None, False, 0   # 敵が移動先のマスにいる
            f = env._lookahead_features(set()).reshape(4, 3)
            assert f[a, 0] == 0.0 and f[a, 2] == 0.0
            # 敵と反対側に開いた通路があれば、そちらは少なくとも1手は捕まらない
            back = (pr - dr, pc - dc)
            if env.grid[back[0]][back[1]] != CELL_WALL:
                assert f[DIRS.index((-dr, -dc)), 2] > 0
            return
    pytest.fail("no open neighbour found")


def test_scared_ghosts_are_not_dangerous_until_their_timer_ends():
    env = make_env()
    env.reset(seed=0)
    for g in env.ghosts:
        g.scared, g.scared_timer = True, 40
    long_fear = env._lookahead_features(set()).reshape(4, 3)
    for g in env.ghosts:
        g.scared_timer = 1
    short_fear = env._lookahead_features(set()).reshape(4, 3)
    assert np.all(long_fear[:, 2] == 1.0)
    assert np.all(short_fear <= long_fear + 1e-6)


def test_round4_plan_uses_five_seeds_and_unused_final_seeds(tmp_path):
    import ghost4_round4 as r4
    args = argparse.Namespace(out=str(tmp_path), steps=1000, seeds=r4.SEEDS, variants=list(r4.VARIANTS))
    jobs = r4.jobs_for(args)
    assert len(jobs) == 10 and len({study.job_id(j) for j in jobs}) == 10
    by = {(j["label"], j["seed"]): j for j in jobs}
    assert by[("safe_lookahead", 101)]["env_options"] == {"classic_lookahead": True}
    assert by[("safe", 101)].get("env_options", {}) == {}
    assert {j["explore_steps"] for j in jobs} == {80000} and {j["eval_episodes"] for j in jobs} == {100}
    final = set(range(r4.FINAL_SEED0, r4.FINAL_SEED0 + r4.FINAL_EPISODES))
    used = set(range(930000, 930100)) | set(range(940000, 940100)) | set(range(950000, 950100))
    assert not final & used
    assert r4.plan_for(args) == r4.plan_for(argparse.Namespace(**{**vars(args), "variants": ["safe"]}))
