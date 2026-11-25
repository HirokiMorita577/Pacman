"""2026-10-15 の追加 (パワーエサを取ったステップの記録、第5回: パワーエサの報酬の比較) のテスト"""
import argparse

import numpy as np

import reward_study as study
from pacman_env import PacManEnv


def test_power_steps_are_recorded_when_power_pellets_are_eaten():
    """パワーエサの隣にパックマンを置き、そのマスへ動かすと、取ったステップが記録される"""
    from pacman_env import CELL_POWER, CELL_WALL, DIRS
    env = PacManEnv(num_ghosts=4, lives=3, include_lives=True, observation_version="tactical")
    env.reset(seed=0)
    assert env.power_steps == []
    powers = [(r, c) for r in range(env.rows) for c in range(env.cols) if env.grid[r][c] == CELL_POWER]
    for g in env.ghosts:                       # 敵は遠くで怖がらせておく (捕まって位置が戻らないように)
        g.scared, g.scared_timer = True, 40
    taken = 0
    for pr, pc in powers:
        for a, (dr, dc) in enumerate(DIRS):
            r, c = pr - dr, pc - dc            # パワーエサの手前のマス
            if 0 <= r < env.rows and 0 <= c < env.cols and env.grid[r][c] != CELL_WALL:
                env.pacman = [r, c]
                _, _, te, tr, info = env.step(a)
                taken += 1
                assert info["power_eaten"] == taken and len(info["power_steps"]) == taken
                assert info["power_steps"][-1] == env.steps
                break
    assert taken == len(powers) > 0
    env.reset(seed=1)
    assert env.power_steps == []


def test_round5_varies_only_the_power_reward(tmp_path):
    import ghost4_round5 as r5
    args = argparse.Namespace(out=str(tmp_path), steps=1000, seeds=r5.SEEDS, variants=list(r5.VARIANTS))
    jobs = r5.jobs_for(args)
    assert len(jobs) == 20 and len({study.job_id(j) for j in jobs}) == 20
    by = {(j["label"], j["seed"]): j for j in jobs}
    assert {v: by[(v, 101)]["rewards"]["power"] for v in r5.VARIANTS} == r5.POWER
    base = dict(by[("power50", 101)]["rewards"])
    for v in r5.VARIANTS:
        other = dict(by[(v, 101)]["rewards"])
        other["power"] = base["power"]
        assert other == base                                     # パワーエサ以外の報酬は同じ
        assert by[(v, 101)]["env_options"] == {"classic_lookahead": True}
    # 基準 (power50) は第4回の safe_lookahead と同じ報酬・観測
    import ghost4_round4 as r4
    r4_job = r4.make_job("safe_lookahead", 101, 1000, str(tmp_path))
    assert r4_job["rewards"] == base and r4_job["env_options"] == by[("power50", 101)]["env_options"]
    final = set(range(r5.FINAL_SEED0, r5.FINAL_SEED0 + r5.FINAL_EPISODES))
    used = set(range(930000, 930100)) | set(range(940000, 940100)) | set(range(950000, 950100)) | set(range(960000, 960100))
    assert not final & used


def test_evaluate_reports_power_timing(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, "buffer", 10)
    job = dict(exp="t_round5", label="p", num_ghosts=4, lives=3, include_lives=True, observation_version="tactical",
               mask_illegal_actions=True, rewards=study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0),
               gamma=0.95, seed=1, episodes=100, step_budget=100, eval_episodes=3, eval_seed0=930000,
               save_model=True, runs_dir=str(tmp_path))
    ev = study.run_job(job)["eval"]
    assert len(ev["power_steps_each"]) == 3
    assert [len(p) for p in ev["power_steps_each"]] == ev["power_eaten_each"]
