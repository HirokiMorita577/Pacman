"""2026-10-07 の追加 (tactical_safe 観測・fix_leave_dot・env_options・第2回比較の入口) の回帰テスト"""
import argparse

import numpy as np

import reward_study as study
from pacman_env import PacManEnv


def test_tactical_safe_adds_twelve_bounded_features():
    kw = dict(num_ghosts=4, lives=3, include_lives=True)
    base = PacManEnv(observation_version="tactical", **kw)
    safe = PacManEnv(observation_version="tactical_safe", **kw)
    assert safe.observation_space.shape[0] == base.observation_space.shape[0] + 12
    o_base, _ = base.reset(seed=5)
    o_safe, _ = safe.reset(seed=5)
    assert np.array_equal(o_safe[:-12], o_base)          # 既存の入力の並びは変えない
    rng = np.random.default_rng(5)
    for _ in range(200):
        o, _, te, tr, _ = safe.step(int(rng.integers(4)))
        assert o.shape == safe.observation_space.shape and o.min() >= 0 and o.max() <= 1
        if te or tr:
            break


def test_safety_features_use_whole_reachable_area_when_no_dangerous_ghost():
    env = PacManEnv(num_ghosts=4, observation_version="tactical_safe")
    env.reset(seed=0)
    for g in env.ghosts:
        g.scared, g.scared_timer = True, 10
    dots = {(r, c) for r in range(env.rows) for c in range(env.cols) if env.grid[r][c] in (2, 3)}
    feats = env._safety_features(dots).reshape(4, 3)
    assert np.allclose(feats[:, 0], 1.0) and np.allclose(feats[:, 2], 1.0)


def test_fix_leave_dot_only_removes_penalty_right_after_eating():
    removed = 0
    for seed in range(10):
        a = PacManEnv(num_ghosts=4, lives=3, fix_leave_dot=False)
        b = PacManEnv(num_ghosts=4, lives=3, fix_leave_dot=True)
        a.reset(seed=seed)
        b.reset(seed=seed)
        rng = np.random.default_rng(seed)
        ate = False
        for _ in range(1000):
            # エサの方へ寄りやすい行動 (行動別特徴の「エサの近さ」が最大の方向) を半分の確率で選ぶ
            obs = a._get_obs()
            feats = obs[a.action_mask_indices[0]:a.action_mask_indices[0] + 16].reshape(4, 4)
            act = int(np.argmax(feats[:, 1])) if rng.random() < 0.5 else int(rng.integers(4))
            dots_before = a.dots_eaten
            oa, _, ta, tra, ia = a.step(act)
            ob, _, tb, trb, ib = b.step(act)
            assert np.array_equal(oa, ob) and ta == tb and ia["score"] == ib["score"]   # ゲームの進み方は同じ
            ate = ate or a.dots_eaten > dots_before
            da, db = ia["reward_components"], ib["reward_components"]
            if not np.allclose(da, db):
                assert ate and a.steps % 2 == 0            # 違いは「前回の測定以降に食べた」測定の時だけ
                assert np.isclose(da[0] - db[0], -a.rw["leave_dot"])
                removed += 1
            if a.steps % 2 == 0:
                ate = False
            if ta or tra:
                break
    assert removed > 0


def test_env_options_reach_training_and_evaluation(tmp_path):
    job = dict(exp="t_round2", label="safe", num_ghosts=4, lives=3, include_lives=True,
               observation_version="tactical_safe", mask_illegal_actions=True,
               env_options={"fix_leave_dot": True},
               rewards=study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0), gamma=0.95,
               seed=1, episodes=3, step_budget=64, eval_episodes=1, eval_seed0=930000,
               save_model=True, runs_dir=str(tmp_path))
    r = study.run_job(job)
    assert r["env_options"] == {"fix_leave_dot": True}
    assert r["observation_size"] == 1753 and r["observation_version"] == "tactical_safe"


def test_round2_variants_are_distinct_runs(tmp_path):
    import ghost4_round2 as round2
    args = argparse.Namespace(out=str(tmp_path), steps=1000, seeds=[101, 202], variants=list(round2.VARIANTS))
    jobs = round2.jobs_for(args)
    assert len(jobs) == 8 and len({study.job_id(j) for j in jobs}) == 8
    control = [j for j in jobs if j["label"] == "control"][0]
    assert control["observation_version"] == "tactical" and control["mask_illegal_actions"]
    assert control["rewards"]["death"] == 100 and control["rewards"]["clear"] == 0 and control["gamma"] == 0.95
