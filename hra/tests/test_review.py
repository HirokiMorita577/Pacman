import json
import pickle

import pytest

from hra import review as module


def test_review_isolates_episodes_and_preserves_checkpoint(tmp_path, monkeypatch):
    ck = tmp_path / 'checkpoint.pkl'
    ck.write_bytes(pickle.dumps({'episode': 11, 'agent': {'N': 7}, 'maze_colors': [], 'pellet_sites': {}}))
    original = ck.read_bytes()
    seen = []

    class Env:
        def __init__(self, cfg, seed):
            self.seed = seed
        def load_sites(self, *args):
            pass
        def close(self):
            pass

    class Agent:
        def __init__(self, cfg):
            pass
        def load_state_dict(self, state):
            self.N = state['N']

    def play(env, agent, learn, noops):
        assert learn is False
        assert agent.N == 7
        seen.append(env.seed)
        agent.N += 100
        return {'score': 100.0, 'level_reached': 2}

    monkeypatch.setattr(module, 'MsPacmanEnv', Env)
    monkeypatch.setattr(module, 'HRAAgent', Agent)
    monkeypatch.setattr(module, 'play_episode', play)
    out = module.review(ck, None, tmp_path / 'results', episodes=3, expected_episode=11, video=False)
    from pathlib import Path
    result = json.loads((Path(out) / 'summary.json').read_text())
    assert result['mean'] == 100 and result['mean_bootstrap_95_ci'] == [100, 100]
    assert len(set(seen)) == 3
    assert ck.read_bytes() == original
    assert (Path(out) / 'COMPLETE.txt').exists()
    with pytest.raises(ValueError, match='Expected episode'):
        module.review(ck, None, tmp_path / 'bad', expected_episode=12, video=False)
