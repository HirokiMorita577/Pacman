import copy
import csv
import random

import numpy as np
import pytest
import torch

from pacman_env import PacManEnv, CELL_DOT, CELL_EMPTY, ACTION_RIGHT, N_REWARD_COMPONENTS
from dqn_agent import HRADQNAgent
import reward_study as study


def test_lives_observation_and_legacy_shape():
    old = PacManEnv(lives=3)
    new = PacManEnv(lives=3, include_lives=True)
    a, _ = old.reset(seed=1)
    b, _ = new.reset(seed=1)
    assert len(b) == len(a) + 1
    np.testing.assert_array_equal(a, b[:-1])
    assert b[-1] == 1
    new.lives = 1
    assert new._get_obs()[-1] == pytest.approx(1 / 3)


def test_capture_respawn_and_final_life():
    env = PacManEnv(lives=3)
    env.reset(seed=1)
    grid = copy.deepcopy(env.grid)
    env.score = 123
    rc = np.zeros(N_REWARD_COMPONENTS, dtype=np.float32)
    assert not env._handle_ghost_collision(env.ghosts[0], rc, False)
    assert env.lives == 2 and env.score == 123 and env.grid == grid
    assert env.pacman == [16, 9]
    assert rc.sum() == -env.rw['death']
    env.lives = 1
    assert env._handle_ghost_collision(env.ghosts[0], rc, False)
    assert env.lives == 0


def test_scared_capture_is_score_not_life_loss():
    env = PacManEnv(lives=3)
    env.reset(seed=1)
    env.ghosts[0].scared = True
    rc = np.zeros(N_REWARD_COMPONENTS, dtype=np.float32)
    assert not env._handle_ghost_collision(env.ghosts[0], rc, False)
    assert env.score == 200 and env.lives == 3
    assert rc.sum() == env.rw['eat_ghost']


def test_clear_and_reward_components():
    env = PacManEnv(no_ghosts=True)
    env.reset(seed=1)
    env.pacman = [1, 1]
    env.grid[1][2] = CELL_DOT
    env.dots_eaten = env.total_dots - 1
    _, reward, terminated, _, info = env.step(ACTION_RIGHT)
    assert terminated and info['cleared']
    assert info['reward_items']['clear'] == env.rw['clear']
    assert reward == pytest.approx(info['reward_components'].sum())


def test_time_limit_is_truncation():
    env = PacManEnv(no_ghosts=True, max_steps=1)
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(ACTION_RIGHT)
    assert truncated and not terminated


def test_direct_collision_only_costs_one_life():
    env = PacManEnv(lives=3)
    env.reset(seed=1)
    env.pacman = [1, 1]
    env.ghosts[0].pos = [1, 2]
    env.ghosts[0].move = lambda *args, **kwargs: None
    _, _, terminated, _, info = env.step(ACTION_RIGHT)
    assert not terminated and info['lives'] == 2
    assert info['reward_items']['death'] == -env.rw['death']


def test_ghost_moving_into_player_causes_capture():
    env = PacManEnv(lives=1)
    env.reset(seed=1)
    env.pacman = [1, 1]
    env.ghosts[0].pos = [1, 3]
    def move_into_player(*args, **kwargs):
        env.ghosts[0].pos = [1, 2]
    env.ghosts[0].move = move_into_player
    _, _, terminated, _, info = env.step(ACTION_RIGHT)
    assert terminated and info['lives'] == 0
    assert info['reward_items']['death'] == -env.rw['death']


def test_head_rewards_add_up_over_episode():
    env = PacManEnv(lives=3)
    env.reset(seed=101)
    rng = np.random.default_rng(42)
    total = 0
    for _ in range(100):
        _, reward, te, tr, info = env.step(int(rng.integers(4)))
        assert reward == pytest.approx(info['reward_components'].sum())
        total += reward
        if te or tr:
            break
    assert total == pytest.approx(sum(info['reward_items'].values()), abs=1e-4)


def make_job(tmp_path):
    return dict(exp='test', label='resume', rewards=study.DESIGNED_REWARDS,
                gamma=0.95, episodes=3, seed=101, eval_episodes=2,
                runs_dir=str(tmp_path), lives=3, include_lives=True,
                save_model=True, checkpoint_every=1, eval_seed0=910000)


def test_experiment_identity_tracks_code_and_settings(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    initial = study.job_id(job)
    assert initial == study.job_id({**job, 'runs_dir': 'other', 'checkpoint_every': 100})
    assert initial != study.job_id({**job, 'include_lives': False})
    monkeypatch.setitem(study.FIXED, 'lr', 0.002)
    assert initial != study.job_id(job)


def test_checkpoint_restores_target_replay_rng(tmp_path):
    torch.set_num_threads(1)
    agent = HRADQNAgent(5, 4, 3, batch_size=2, device='cpu', buffer_capacity=10)
    for i in range(3):
        agent.push(np.ones(5) * i, i, np.ones(3), np.zeros(5), 0)
    agent.train_step()
    path = tmp_path / 'training.pt'
    agent.save_training(path, {'episode': 7}, {'id': 'test'})
    expected_rng = (random.random(), np.random.random(), torch.rand(1))
    clone = HRADQNAgent(5, 4, 3, batch_size=2, device='cpu')
    assert clone.load_training(path, {'id': 'test'}) == {'episode': 7}
    actual_rng = (random.random(), np.random.random(), torch.rand(1))
    assert actual_rng[:2] == expected_rng[:2]
    assert torch.equal(actual_rng[2], expected_rng[2])
    assert clone.steps == agent.steps and len(clone.buffer) == 3
    for a, b in zip(agent.target_net.parameters(), clone.target_net.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match='mismatch'):
        clone.load_training(path, {'id': 'different'})


def test_interrupted_run_matches_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, 'max_steps', 8)
    monkeypatch.setitem(study.FIXED, 'batch_size', 2)
    monkeypatch.setitem(study.FIXED, 'buffer', 50)
    job = make_job(tmp_path / 'continuous')
    full = study.run_job(job)
    resumed_job = {**job, 'runs_dir': str(tmp_path / 'resumed')}
    study.run_job({**resumed_job, 'stop_after': 1})
    resumed = study.run_job(resumed_job)
    assert full['eval'] == resumed['eval']
    assert full['total_steps'] == resumed['total_steps']
    checkpoints = []
    for j in (job, resumed_job):
        p = tmp_path / ('continuous' if j is job else 'resumed') / study.job_id(j) / 'training.pt'
        checkpoints.append(torch.load(p, weights_only=False))
    for key in ('policy', 'target'):
        for name, value in checkpoints[0][key].items():
            assert torch.equal(value, checkpoints[1][key][name])
    assert len(checkpoints[1]['progress']['rows']) == 3
    assert checkpoints[0]['progress']['rows'] == checkpoints[1]['progress']['rows']


def test_step_budget_is_exact(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, 'max_steps', 8)
    monkeypatch.setitem(study.FIXED, 'batch_size', 2)
    monkeypatch.setitem(study.FIXED, 'buffer', 50)
    job = {**make_job(tmp_path), 'episodes': 20, 'step_budget': 11}
    result = study.run_job(job)
    assert result['total_steps'] == 11
    checkpoint = torch.load(tmp_path / study.job_id(job) / 'training.pt', weights_only=False)
    assert checkpoint['epsilon'] == study.FIXED['epsilon_end']


@pytest.mark.parametrize('num_ghosts,compressed', [(1, False), (4, True)])
def test_budget_resume_matches_continuous(tmp_path, monkeypatch, num_ghosts, compressed):
    monkeypatch.setitem(study.FIXED, 'max_steps', 8)
    monkeypatch.setitem(study.FIXED, 'batch_size', 2)
    monkeypatch.setitem(study.FIXED, 'buffer', 50)
    job = {**make_job(tmp_path / 'full'), 'episodes': 30, 'step_budget': 25,
           'num_ghosts': num_ghosts, 'compress_checkpoint': compressed}
    full = study.run_job(job)
    resumed_job = {**job, 'runs_dir': str(tmp_path / 'resumed')}
    study.run_job({**resumed_job, 'stop_after': 1})
    resumed = study.run_job(resumed_job)
    assert full['eval'] == resumed['eval']
    assert full['total_steps'] == resumed['total_steps'] == 25
    import gzip
    snapshots = []
    for name in ('full', 'resumed'):
        path = tmp_path / name / study.job_id(job) / ('training.pt.gz' if compressed else 'training.pt')
        with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
            snapshots.append(torch.load(stream, weights_only=False))
    assert full['num_ghosts'] == num_ghosts
    assert full['observation_size'] == 1697 + 3 * num_ghosts
    for key in ('policy', 'target'):
        for name, value in snapshots[0][key].items():
            assert torch.equal(value, snapshots[1][key][name])


def test_cached_manifest_mismatch_is_rejected(tmp_path):
    job = make_job(tmp_path)
    path = tmp_path / study.job_id(job)
    path.mkdir()
    study.atomic_json(path / 'result.json', {'manifest': {}})
    with pytest.raises(ValueError, match='manifest mismatch'):
        study.run_jobs([job], workers=1)


def test_four_ghost_positions_and_distances_are_observed():
    env = PacManEnv(num_ghosts=4, lives=3, include_lives=True)
    obs, _ = env.reset(seed=101)
    assert obs.shape == env.observation_space.shape == (1709,)
    assert [g.ghost_id for g in env.ghosts] == [0, 1, 2, 3]
    offset = env.rows * env.cols * 4 + 2
    expected = [v for g in env.ghosts for v in (g.pos[0] / env.rows, g.pos[1] / env.cols)]
    np.testing.assert_allclose(obs[offset:offset + 8], expected)
    distances = [env._bfs_distance(tuple(env.pacman), {tuple(g.pos)}) / (env.rows + env.cols)
                 for g in env.ghosts]
    np.testing.assert_allclose(obs[offset + 9:offset + 13], distances)


@pytest.mark.parametrize('lives', [1, 3])
@pytest.mark.parametrize('move_into_player', [False, True])
def test_four_simultaneous_captures_cost_one_life(lives, move_into_player):
    env = PacManEnv(num_ghosts=4, lives=lives, include_lives=True)
    env.reset(seed=101)
    env.pacman = [1, 1]
    env.grid[1][2] = CELL_EMPTY
    for ghost in env.ghosts:
        ghost.pos = [1, 3] if move_into_player else [1, 2]
        def move(*args, g=ghost, **kwargs):
            if move_into_player:
                g.pos = [1, 2]
        ghost.move = move
    _, _, terminated, _, info = env.step(ACTION_RIGHT)
    assert info['lives'] == lives - 1
    assert terminated == (lives == 1)
    assert info['reward_items']['death'] == -env.rw['death']


def test_four_scared_ghosts_can_all_be_eaten():
    env = PacManEnv(num_ghosts=4, lives=3)
    env.reset(seed=101)
    env.pacman = [1, 1]
    env.grid[1][2] = CELL_EMPTY
    for ghost in env.ghosts:
        ghost.pos = [1, 2]
        ghost.scared = True
        ghost.move = lambda *args, **kwargs: None
    _, _, terminated, _, info = env.step(ACTION_RIGHT)
    assert not terminated and info['lives'] == 3
    assert info['score'] == 800
    assert info['reward_items']['eat_ghost'] == 4 * env.rw['eat_ghost']


def test_four_ghost_configuration_reaches_train_probe_and_evaluation(tmp_path, monkeypatch):
    created = []
    def tracked_env(*args, **kwargs):
        env = PacManEnv(*args, **kwargs)
        created.append(env.num_ghosts)
        return env
    monkeypatch.setattr(study, 'PacManEnv', tracked_env)
    monkeypatch.setitem(study.FIXED, 'max_steps', 3)
    job = {**make_job(tmp_path), 'episodes': 1, 'num_ghosts': 4}
    result = study.run_job(job)
    assert created == [4, 4, 4]
    assert result['num_ghosts'] == 4 and result['observation_size'] == 1709
    assert study.job_id(job) != study.job_id({**job, 'num_ghosts': 1})
