"""2026-10-08 の追加 (先読み特徴量の意味・safety_timing・特徴群の無効化・探索期間の固定・学習途中の評価・第3回の計画) のテスト"""
import argparse
import csv

import numpy as np
import pytest

import reward_study as study
from pacman_env import PacManEnv, DIRS, CELL_WALL, CELL_DOT, CELL_POWER, SAFETY_GROUPS


def open_cell(env, r, c):
    return 0 <= r < env.rows and 0 <= c < env.cols and env.grid[r][c] != CELL_WALL


def corridor(env):
    """パックマンの隣 n1 と、その先 n2 (n2 != パックマン) を1組と、別の向きの行動を返す"""
    pr, pc = env.pacman
    acts = [a for a, (dr, dc) in enumerate(DIRS) if open_cell(env, pr + dr, pc + dc)]
    for a in acts:
        n1 = (pr + DIRS[a][0], pc + DIRS[a][1])
        for dr, dc in DIRS:
            n2 = (n1[0] + dr, n1[1] + dc)
            if open_cell(env, *n2) and n2 != (pr, pc) and env._distance_field([(pr, pc)])[n2[0]][n2[1]] == 2:
                others = [b for b in acts if b != a]
                if others:
                    return a, n1, n2, others
    raise AssertionError("no corridor found")


def dots(env):
    return {(r, c) for r in range(env.rows) for c in range(env.cols) if env.grid[r][c] in (CELL_DOT, CELL_POWER)}


def setup(timing=False, groups=SAFETY_GROUPS):
    env = PacManEnv(num_ghosts=4, observation_version="tactical_safe", safety_timing=timing, safety_groups=groups)
    env.reset(seed=3)
    a, n1, n2, others = corridor(env)
    far = env.ghosts[1:]
    for g in far:                       # ほかの敵は長く怖がっている (どちらの設定でも危険でない)
        g.scared, g.scared_timer = True, 40
    return env, a, n2, others


def feats(env):
    return env._safety_features(dots(env)).reshape(4, 3)


def test_dangerous_ghost_on_one_side_lowers_that_action():
    env, a, n2, others = setup()
    g = env.ghosts[0]
    g.pos, g.scared, g.scared_timer = list(n2), False, 0
    f = feats(env)
    assert f[a, 2] == 0.0                                   # 移動先の隣に危険な敵 → 余裕 0
    assert all(f[a, 2] < f[b, 2] for b in others)
    assert all(f[a, 0] <= f[b, 0] for b in others)          # 敵の方へ進むほど安全な領域は狭い


def test_timing_counts_scared_ghost_whose_fear_is_about_to_end():
    env, a, n2, others = setup(timing=True)
    g = env.ghosts[0]
    g.pos, g.scared, g.scared_timer = list(n2), True, 1     # 1ステップ後に元に戻る
    with_timing = feats(env)
    env.safety_timing = False
    without = feats(env)
    assert without[a, 2] == 1.0 and with_timing[a, 2] == 0.0   # 従来は全員怖がっていて危険なしと扱っていた
    assert np.all(with_timing[:, 0] <= without[:, 0])
    assert with_timing[a, 0] < without[a, 0]


def test_timing_keeps_long_fear_safe_and_is_not_saturated_by_default_rules():
    env, a, n2, others = setup(timing=True)
    g = env.ghosts[0]
    g.pos, g.scared, g.scared_timer = list(n2), True, 40     # まだ長く怖がっている → 近くても安全
    f = feats(env)
    assert f[a, 2] == 1.0
    g.scared_timer = 3                                     # 残り3: 移動先 (距離1) では max(1,3)-1 = 2 の余裕
    assert np.isclose(feats(env)[a, 2], 0.2)


def test_eaten_ghost_is_dangerous_again_after_reset():
    env, a, n2, others = setup()
    g = env.ghosts[0]
    g.scared, g.scared_timer = True, 10
    before = env._threat_field()
    g.reset()                                               # 食べられた敵は開始位置に戻り、すぐ危険になる
    assert before is None and env._threat_field() is not None


@pytest.mark.parametrize("removed", SAFETY_GROUPS)
def test_safety_groups_zero_only_the_removed_group(removed):
    keep = [g for g in SAFETY_GROUPS if g != removed]
    full, a, n2, _ = setup()
    part, _, _, _ = setup(groups=keep)
    assert full.observation_space.shape == part.observation_space.shape
    for e in (full, part):
        e.ghosts[0].pos = list(n2)
    k = SAFETY_GROUPS.index(removed)
    f, p = feats(full), feats(part)
    assert np.all(p[:, k] == 0.0)
    others = [i for i in range(3) if i != k]
    assert np.array_equal(f[:, others], p[:, others])


def test_route_features_find_safe_dot_and_path_margin():
    env = PacManEnv(num_ghosts=4, observation_version="tactical_safe", safe_route=True)
    env.reset(seed=3)
    base = PacManEnv(num_ghosts=4, observation_version="tactical_safe")
    assert env.observation_space.shape[0] == base.observation_space.shape[0] + 8
    a, n1, n2, others = corridor(env)
    for g in env.ghosts:
        g.scared, g.scared_timer = True, 40
    n1 = tuple(n1)
    only = {n1}                                             # エサは移動先の1個だけ
    f = env._route_features(only).reshape(4, 2)
    assert np.isclose(f[a, 0], 1.0) and f[a, 1] == 1.0      # 危険な敵なし: すぐ食べられ、余裕は最大
    g = env.ghosts[0]
    g.pos, g.scared, g.scared_timer = list(n2), False, 0    # エサの隣に危険な敵
    f = env._route_features(only).reshape(4, 2)
    assert f[a, 0] == 0.0 and f[a, 1] == 0.0                # 安全には行けず、経路上の余裕もない
    assert np.all(f[:, 0] == 0.0)
    assert np.all((0.0 <= env._get_obs()) & (env._get_obs() <= 1.0))


def test_safety_options_require_tactical_safe():
    with pytest.raises(ValueError):
        PacManEnv(num_ghosts=4, observation_version="tactical", safe_route=True)
    with pytest.raises(ValueError):
        PacManEnv(num_ghosts=4, observation_version="tactical", safety_timing=True)
    with pytest.raises(ValueError):
        PacManEnv(num_ghosts=4, observation_version="tactical_safe", safety_groups=["area", "bogus"])


def small_job(tmp_path, name, **kw):
    job = dict(exp="t_round3", label=name, num_ghosts=4, lives=3, include_lives=True, observation_version="tactical",
               mask_illegal_actions=True, rewards=study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0),
               gamma=0.95, seed=1, eval_episodes=1, eval_seed0=930000, save_model=True, runs_dir=str(tmp_path / name))
    job.update(kw)
    job.setdefault("episodes", job.get("step_budget", 3))
    return job


def log_rows(result, job):
    path = study.Path(job["runs_dir"]) / result["id"] / "log.csv"
    return list(csv.DictReader(path.open(encoding="utf-8")))


def test_explore_steps_fixes_exploration_when_budget_grows(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, "buffer", 10)
    short = small_job(tmp_path, "short", step_budget=150, explore_steps=60)
    long_ = small_job(tmp_path, "long", step_budget=300, explore_steps=60)
    legacy = small_job(tmp_path, "legacy", step_budget=300)
    rs, rl, rg = (study.run_job(j) for j in (short, long_, legacy))
    a, b, c = log_rows(rs, short), log_rows(rl, long_), log_rows(rg, legacy)
    n = len(a) - 1                                          # 短い方の最後は予算で打ち切った回
    assert n > 0 and a[:n] == b[:n]                         # 学習を延ばしても、同じ所までは同じ計算
    assert a[0]["epsilon"] != c[0]["epsilon"]               # explore_steps なしは予算の40%で減衰 (従来どおり)
    assert rs["explore_steps"] == 60 and "explore_steps" not in rg
    assert "explore_steps" not in study.experiment_manifest(legacy)["job"]   # 既存の学習IDは変わらない


def test_curve_eval_is_recorded_and_survives_resume(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, "buffer", 10)
    curve = {"every": 50, "episodes": 1, "seed0": 930000}
    job = small_job(tmp_path, "curve", step_budget=200, explore_steps=80, curve_eval=curve, checkpoint_every=1)
    paused = study.run_job({**job, "stop_after": 2})
    assert paused.get("paused")
    before = list(csv.DictReader((study.Path(job["runs_dir"]) / paused["id"] / "eval_curve.csv").open(encoding="utf-8")))
    result = study.run_job(job)
    steps = [row["total_steps"] for row in result["curve"]]
    assert steps == sorted(steps) and len(steps) == 4 and steps[-1] == 200
    assert [str(row["total_steps"]) for row in result["curve"][:len(before)]] == [row["total_steps"] for row in before]
    fresh = study.run_job(small_job(tmp_path, "curve2", step_budget=200, explore_steps=80, curve_eval=curve))
    assert fresh["curve"] == result["curve"]                # 途中で止めて再開しても同じ学習曲線


def test_round3_plan_separates_factors_and_reserves_unused_final_seeds(tmp_path):
    import ghost4_round3 as r3
    args = argparse.Namespace(out=str(tmp_path), steps=1000, seeds=[101, 202, 303], variants=["safe"])
    jobs = r3.jobs_for(argparse.Namespace(**{**vars(args), "variants": list(r3.VARIANTS)}))
    assert len({study.job_id(j) for j in jobs}) == len(jobs) == 3 * len(r3.VARIANTS)
    by = {(j["label"], j["seed"]): j for j in jobs}
    assert by[("safe_400k", 101)]["step_budget"] == 2 * by[("safe", 101)]["step_budget"]
    assert {j["explore_steps"] for j in jobs} == {80000}    # 学習量を変えても探索の長さは同じ
    assert all(j["observation_version"] == "tactical_safe" for j in jobs)
    assert not any(v.startswith("no_") for v in r3.CANDIDATES) and r3.BASELINE not in r3.CANDIDATES
    assert r3.plan_for(args) == r3.plan_for(argparse.Namespace(**{**vars(args), "variants": ["no_area"]}))
    final = set(range(r3.FINAL_SEED0, r3.FINAL_SEED0 + r3.FINAL_EPISODES))
    used = set(range(930000, 930030)) | set(range(940000, 940100))
    assert not final & used
    env = PacManEnv(num_ghosts=4, lives=3, include_lives=True, observation_version="tactical_safe",
                    **by[("no_dots", 101)]["env_options"])
    assert env.safety_groups == ("area", "margin")


def test_score_parts_add_up_to_game_score():
    env = PacManEnv(num_ghosts=4, lives=3, include_lives=True, observation_version="tactical")
    rng = np.random.default_rng(0)
    for seed in range(5):
        env.reset(seed=seed)
        while True:
            _, _, te, tr, info = env.step(int(rng.integers(4)))
            p = info["score_parts"]
            assert p["dot"] + p["power"] + p["ghost"] == info["score"]
            assert p["ghost"] == 200 * info["ghosts_eaten"] and p["power"] == 50 * info["power_eaten"]
            if te or tr:
                break


def test_evaluate_reports_score_breakdown(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, "buffer", 10)
    r = study.run_job(small_job(tmp_path, "score", step_budget=100, eval_episodes=3))
    ev = r["eval"]
    for i in range(3):
        assert ev["dot_points_each"][i] + ev["power_points_each"][i] + ev["ghost_points_each"][i] == ev["scores"][i]
    assert np.isclose(ev["ghosts_eaten_mean"], np.mean(ev["ghosts_eaten_each"]))
    assert ev["score_mean_cleared"] is None or ev["score_mean_cleared"] > 0


def test_replay_endgame_diagnostics_are_logged_and_resume_exactly(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, "buffer", 50)
    monkeypatch.setitem(study.FIXED, "batch_size", 8)
    monkeypatch.setattr(study, "ENDGAME_DOTS", 200)              # 小さな学習でも終盤扱いの経験が入るように
    job = small_job(tmp_path, "replay", step_budget=150, explore_steps=60, checkpoint_every=1)
    paused = study.run_job({**job, "stop_after": 1})
    resumed = study.run_job(job)
    fresh_job = small_job(tmp_path, "replay2", step_budget=150, explore_steps=60)
    fresh = study.run_job(fresh_job)
    a, b = log_rows(resumed, job), log_rows(fresh, fresh_job)
    assert a == b                                               # 途中で止めて再開しても、診断の列まで同じ
    last = a[-1]
    assert int(last["endgame_pushed"]) == int(last["total_steps"])  # 全部が終盤扱い
    assert float(last["replay_endgame_frac"]) == 1.0
    assert float(last["endgame_sampled_expected"]) == 8 * (int(last["updates"]))
