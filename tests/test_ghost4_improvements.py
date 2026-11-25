import copy
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

from dqn_agent import DQNAgent, HRADQNAgent, load_agent
from pacman_env import PacManEnv, DIRS, CELL_EMPTY, CELL_DOT, CELL_WALL
import reward_study as study
import ghost4_improvement as comparison


def test_tactical_observation_distinguishes_ghost_state_and_heading():
    env = PacManEnv(num_ghosts=4, lives=3, include_lives=True, observation_version='tactical')
    obs, _ = env.reset(seed=1)
    assert obs.shape == (1741,)
    before = obs.copy()
    ghost = env.ghosts[1]
    ghost.scared, ghost.scared_timer, ghost.last_pos = True, 20, [10, 7]
    env.pacman_dir = 0
    after = env._get_obs()
    # Appended features: player direction, then 7 independent values per ghost.
    np.testing.assert_array_equal(after[-32:-28], [1, 0, 0, 0])
    np.testing.assert_array_equal(after[-21:-14], [1, .5, 0, 0, 0, 1, 0])
    assert not np.array_equal(before, after)
    assert env.observation_space.contains(after)


def test_tactical_power_observation_and_maze_direction_are_valid():
    env = PacManEnv(observation_version='tactical')
    env.reset(seed=1)
    env.grid = [[CELL_WALL] * env.cols for _ in range(env.rows)]
    for r, c in [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3)]:
        env.grid[r][c] = CELL_EMPTY
    env.grid[1][3] = CELL_DOT
    env.pacman = [1, 1]
    env.power_timer = 40
    obs = env._get_obs()
    extra = env.rows * env.cols * 4 + 2 + 2 * env.num_ghosts
    np.testing.assert_array_equal(obs[extra + 2:extra + 6], [0, 1, 0, 0])
    assert env.observation_space.contains(obs)
    assert obs[extra + 6] == 1.0


def test_respawned_dangerous_ghost_penalized_during_global_power():
    env = PacManEnv(num_ghosts=4, lives=3)
    env.reset(seed=1)
    env.pacman = [1, 1]
    env.power_mode, env.power_timer = True, 20
    for g in env.ghosts:
        g.pos = [10, 9]
        g.scared = True
        g.scared_timer = 20
        g.move = lambda *args, **kwargs: None
    env.ghosts[0].pos = [1, 3]
    env.ghosts[0].scared = False
    _, _, _, _, info = env.step(3)  # player moves to 1,2; dangerous enemy at 1,3
    assert info['reward_items']['ghost_adjacent'] == -env.rw['ghost_adjacent']
    assert 'chase_blue' not in info['reward_items']
    assert info['lives'] == 3


def test_wall_separating_ghost_does_not_trigger_proximity_penalty():
    env = PacManEnv()
    env.reset(seed=1)
    env.pacman = [4, 2]
    env.ghosts[0].pos = [6, 3]
    env.ghosts[0].move = lambda *args, **kwargs: None
    _, _, _, _, info = env.step(3)  # 4,3 to 6,3 is blocked by wall 5,3
    assert 'ghost_near' not in info['reward_items']
    assert 'ghost_adjacent' not in info['reward_items']


class ConstantQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values', torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.expand(len(states), -1)


class ConstantHeads(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values', torch.tensor(values, dtype=torch.float32))

    def head_q(self, states):
        return self.values.expand(len(states), -1, -1)

    def forward(self, states):
        return self.head_q(states).sum(dim=1)


@pytest.mark.parametrize('hra', [False, True])
def test_mask_applies_to_exploration_greedy_and_double_dqn_targets(hra):
    cls = HRADQNAgent if hra else DQNAgent
    kwargs = {'n_heads': 3} if hra else {}
    agent = cls(4, 4, device='cpu', gamma=1, action_mask_indices=(0, 1, 2, 3), **kwargs)
    policy = [100., 0., 90., 5.]
    target = [100., 10., 90., 2.]
    agent.policy_net = ConstantHeads([policy] * 3) if hra else ConstantQ(policy)
    agent.target_net = ConstantHeads([target, [x*2 for x in target], [x*3 for x in target]]) if hra else ConstantQ(target)
    state = np.array([0, 1, 0, 1], dtype=np.float32)
    agent.epsilon = 1
    assert {agent.select_action(state) for _ in range(100)} == {1, 3}
    assert agent.select_action(state, training=False) == 3
    states = torch.tensor([state.tolist()])
    # Action 1 has predicted Q=0. Bootstrap from legal action 3, never illegal 0.
    loss = agent._compute_loss(states, torch.tensor([1]), torch.zeros((1, 3) if hra else 1),
                               states, torch.zeros(1))
    assert loss.item() == pytest.approx(3.5 if hra else 1.5)
    assert torch.isfinite(agent._compute_loss(states, torch.tensor([1]),
                         torch.zeros((1, 3) if hra else 1), torch.zeros_like(states), torch.ones(1)))


def test_masked_model_round_trip_and_configuration_rejection(tmp_path):
    agent = HRADQNAgent(4, 4, 3, device='cpu', action_mask_indices=(0, 1, 2, 3))
    p = tmp_path / 'model.pt'
    agent.save(p)
    clone = load_agent(p, 4, 4, device='cpu')
    assert clone.action_mask_indices == agent.action_mask_indices
    np.testing.assert_array_equal(clone.get_qvalues(np.ones(4)), agent.get_qvalues(np.ones(4)))
    cp = tmp_path / 'training.pt'
    agent.save_training(cp, {}, {})
    unmasked = HRADQNAgent(4, 4, 3, device='cpu')
    with pytest.raises(ValueError, match='action mask mismatch'):
        unmasked.load_training(cp, {})


def test_tactical_training_evaluation_and_resume_match(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(study.FIXED, 'max_steps', 6)
    monkeypatch.setitem(study.FIXED, 'batch_size', 2)
    monkeypatch.setitem(study.FIXED, 'buffer', 40)
    args = Namespace(out=str(tmp_path/'full'), steps=17, eval_episodes=2, seeds=[101], variants=['improved'])
    job = comparison.jobs_for(args)[0]
    full = study.run_job(job)
    resumed_job = {**job, 'runs_dir': str(tmp_path/'resumed'/'runs')}
    study.run_job({**resumed_job, 'stop_after': 1})
    resumed = study.run_job(resumed_job)
    assert full['observation_size'] == 1741
    assert full['total_steps'] == resumed['total_steps'] == 17
    assert full['eval'] == resumed['eval']
    assert full['eval']['blocked_action_rate'] == 0
    assert full['eval']['seed0'] == 930000
    import gzip
    snapshots = []
    for j in [job, resumed_job]:
        with gzip.open(__import__('pathlib').Path(j['runs_dir'])/study.job_id(j)/'training.pt.gz', 'rb') as f:
            snapshots.append(torch.load(f, weights_only=False))
    for key in ['policy', 'target']:
        for name, value in snapshots[0][key].items():
            assert torch.equal(value, snapshots[1][key][name])
    assert snapshots[0]['progress']['rows'] == snapshots[1]['progress']['rows']
    assert study.job_id(job) != study.job_id({**job, 'mask_illegal_actions': False})
    assert study.job_id(job) != study.job_id({**job, 'observation_version': 'legacy'})


def test_comparison_writes_results_and_rejects_changed_plan(tmp_path, monkeypatch):
    monkeypatch.setitem(study.FIXED, 'max_steps', 3)
    monkeypatch.setitem(study.FIXED, 'buffer', 10)
    args = ['--out', str(tmp_path), '--steps', '5', '--seeds', '101', '--eval-episodes', '1', '--workers', '1']
    comparison.main(args)
    assert (tmp_path/'COMPLETE.json').exists()
    assert (tmp_path/'evaluation.csv').exists()
    with pytest.raises(ValueError, match='plan differs'):
        comparison.main(args + ['--steps', '6'])
