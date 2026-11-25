from types import SimpleNamespace
import numpy as np
from hra.behavior import BehaviorMetrics
from hra.agent import HRAAgent
from hra.env import Objects
from hra.run import load_config


def test_reversal_ignores_stationary_and_resets_on_death():
    m = BehaviorMetrics()
    def obj(p):
        return SimpleNamespace(pacman=p, maze=0)
    info = dict(life_lost=False, level_cleared=False, in_transition=False)
    for a, b in [((1, 1), (1, 2)), ((1, 2), (1, 2)), ((1, 2), (1, 1))]:
        m.observe(obj(a), obj(b), 0, info, 8)
    assert m.moves == 2 and m.reversals == 1 and m.stall_frames == 8
    m.observe(obj((1, 1)), obj((1, 2)), 0, dict(info, life_lost=True), 8)
    assert m.deaths == 1 and not m.path


def test_frozen_counts_preserve_values_and_rng_decision():
    cfg = load_config()
    a, b = HRAAgent(cfg, 55), HRAAgent(cfg, 55)
    o = Objects()
    o.pacman = (2, 2)
    sa, sb = a.state(o), b.state(o)
    b.update_counts = False
    assert a.act(o, sa, 0) == b.act(o, sb, 0)
    assert a.N == 1 and b.N == 0
    assert b.tables(0).n.sum() == 0
    np.testing.assert_array_equal(a.last_decision['q'], b.last_decision['q'])
