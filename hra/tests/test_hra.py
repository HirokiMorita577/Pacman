"""
RM-14 部品ごとの単体テスト
  python -m pytest hra/tests -q
"""
import numpy as np
import pytest

from hra.agent import HRAAgent, MazeTables
from hra.env import N_ACTIONS, Objects
from hra.run import load_config


# ── GVF: 1つのGVFが「その位置へ行く」を学べるか ──
def _corridor_tables(length=6, gamma=0.9):
    """一本道 0..length-1。行動0=左, 1=右, それ以外は動かない。向きは使わない(0固定)"""
    tab = MazeTables([gamma])
    states = [tab.state((0, i), 0) for i in range(length)]
    return tab, states


def _move(i, a, length):
    if a == 0:
        return max(0, i - 1)
    if a == 1:
        return min(length - 1, i + 1)
    return i


@pytest.mark.parametrize("target", ["expected_sarsa", "q_learning"])
def test_gvf_learns_to_reach_position(target):
    L, g = 6, 0.9
    tab, states = _corridor_tables(L, g)
    goal = tab.pos_index[(0, L - 1)]
    rng = np.random.default_rng(0)
    for _ in range(3000):                       # ランダム方策の経験 (オフポリシー)
        i = int(rng.integers(L - 1))
        a = int(rng.integers(N_ACTIONS))
        tab.update(states[i], a, states[_move(i, a, L)], 1.0, target)
    G = tab.G[g][goal]
    for i in range(L - 1):                      # ゴールへ近づく「右」が最も価値が高い
        assert np.argmax(G[states[i]]) == 1
        assert 0.0 < G[states[i], 1] <= 1.0
    # 近いほど価値が高い
    assert G[states[L - 2], 1] > G[states[0], 1]
    if target == "q_learning":                  # 式(7): 最適方策の価値 = γ^(距離-1)
        assert G[states[0], 1] == pytest.approx(g ** (L - 2), rel=1e-6)


def test_terminal_state_value_is_one_step():
    """終端の扱い (FR-23): 目標に着く遷移の目標値はちょうど1 (その先を足さない)"""
    tab, states = _corridor_tables(3, 0.99)
    goal = tab.pos_index[(0, 2)]
    tab.G[0.99][goal, states[2], :] = 123.0    # 目標位置での値が混ざらないこと
    tab.update(states[1], 1, states[2], 1.0, "expected_sarsa")
    assert tab.G[0.99][goal, states[1], 1] == pytest.approx(1.0)


def test_tables_grow_and_roundtrip():
    tab = MazeTables([0.99])
    for r in range(100):
        for d in range(4):
            tab.state((r // 40, r % 40), d)
    assert tab.n_gvfs == 100 and len(tab.state_index) == 400
    tab.G[0.99][3, 7, 2] = 0.5
    tab.n[399, 1] = 5
    # 保存 → 新しい表に読み込み (初期容量より大きい表でも戻せること)
    tab2 = MazeTables.from_trimmed(tab.trimmed(), [0.99])
    assert tab2.G[0.99][3, 7, 2] == 0.5 and tab2.n_gvfs == 100 and tab2.n[399, 1] == 5


# ── アグリゲーター ──
def _agent(**over):
    cfg = load_config()
    cfg["exploration"]["diversification"] = False
    cfg["exploration"]["count_based"] = False
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    return HRAAgent(cfg, seed=0)


def test_linear_aggregator_is_sum():
    ag = _agent(aggregator__type="linear")
    score = np.arange(9, dtype=float)
    ghost = np.ones(9)
    np.testing.assert_allclose(ag.aggregate(score, ghost), score - 1000)


def test_normalized_aggregator():
    ag = _agent(aggregator__type="normalized")
    score = np.array([0, 2, 4, 1, 0, 0, 0, 0, 0], dtype=float)
    ghost = np.zeros(9)
    ghost[2] = 0.5
    q = ag.aggregate(score, ghost)
    assert q.max() <= 1.0
    assert q[2] == pytest.approx(1.0 - 10 * 0.5)
    assert np.argmax(q) == 1


def test_heads_multiply_gvf_by_object_weight_and_missing_objects_are_zero():
    ag = _agent(aggregator__type="linear")
    tab = ag.tables(0)
    s = tab.state((5, 5), 1)
    tab.state((5, 6), 1)
    t = tab.pos_index[(5, 6)]
    tab.G[0.99][t, s, :] = np.linspace(0, 0.8, 9)
    o = Objects()
    o.pacman, o.direction, o.maze = (5, 5), 1, 0
    o.pellets = [(5, 6)]
    score, ghost = ag.head_values(o, s)
    np.testing.assert_allclose(score, 10 * np.linspace(0, 0.8, 9))   # エサの倍率10 (FR-15)
    assert not ghost.any()                                            # 画面にないゴーストは0 (FR-16)
    o.ghosts = [(5, 6), None, None, None]
    _, ghost = ag.head_values(o, s)
    np.testing.assert_allclose(ghost, np.linspace(0, 0.8, 9))


def test_exploration_heads():
    cfg = load_config()
    ag = HRAAgent(cfg, seed=0)
    o = Objects()
    o.pacman, o.maze = (1, 1), 0
    s = ag.state(o)
    early = ag.q_values(o, s, t=0)
    late = ag.q_values(o, s, t=50)
    assert early.max() > 1.0                  # 多様化頭 U(0,20) は最初の50ステップだけ (FR-30/31)
    assert np.allclose(late, late[0])         # 50ステップ以降はカウント頭のみ (全行動 n=0 で同値)
    ag.N = 10_000
    ag.tables(0).n[s, 3] = 50
    q = ag.count_head(ag.tables(0), s)
    assert q[3] < q[0]                        # 試した回数が多い行動ほどボーナスが小さい (FR-32)


def test_executive_memory_forces_action():
    cfg = load_config()
    cfg["executive_memory"]["enabled"] = True
    ag = HRAAgent(cfg, seed=0)
    ag.remember(1, [4, 4, 2])
    assert ag.memory_action(1, 2, clean=True) == 2
    assert ag.memory_action(1, 2, clean=False) is None     # 死んだ後は使わない
    o = Objects()
    o.pacman, o.maze = (1, 1), 0
    s = ag.state(o)
    assert ag.act(o, s, t=0, memory_action=4) == 4          # 多様化頭より強い (FR-35)


# ── 位置抽出 (ALE が入っている環境のみ) ──
def test_extraction_matches_ram():
    pytest.importorskip("ale_py")
    from hra.env import MsPacmanEnv
    cfg = load_config(overrides=["env.step_mode=frames", "env.position_source=pixels"])
    env = MsPacmanEnv(cfg, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    diffs, n_pellets = [], []
    a = 2
    for t in range(400):
        if t % 8 == 0:
            a = int(rng.integers(1, 5))
        objs, *_ , done, info = env.step(a)
        (pp, rp), gh = env.compare_sources()
        if pp is not None and rp is not None and not info["life_lost"]:
            diffs.append(max(abs(pp[0] - rp[0]), abs(pp[1] - rp[1])))
        n_pellets.append(len(objs.pellets) + len(objs.power_pellets))
        if done:
            break
    diffs = np.array(diffs)
    assert (diffs <= 1).mean() > 0.95          # 画面抽出と RAM の位置が1マス以内で一致 (RM-12)
    assert n_pellets[0] > 100                  # 面の開始時にエサを検出できている (FR-07)
    assert n_pellets[-1] < n_pellets[0]        # 食べたエサが減っている
    env.close()


def test_pellet_sites_are_real_pellets_only():
    """1面の候補地 = エサ150個 + パワーエサ4個。壁の切れ端を含めない (エピソードをまたいでも増えない)"""
    pytest.importorskip("ale_py")
    from hra.env import MsPacmanEnv
    env = MsPacmanEnv(load_config(overrides=["env.step_mode=frames", "env.position_source=pixels"]), seed=0)
    rng = np.random.default_rng(1)
    for _ in range(3):                          # ゴーストの位置が違う開始画面を何度も見る
        env.reset(noops=int(rng.integers(0, 30)))
        for _ in range(110):                    # 開始前の待ち時間とパワーエサの点滅を跨ぐ
            _, _, _, done, _ = env.step(0)
            if done:
                break
    sites = env.pellet_sites[0]
    powers = [c for c, v in sites.items() if v[2]]
    assert len(powers) == 4                                    # 四隅のパワーエサ (右側は壁に接している)
    # 通常のエサはすべて縦12px・横8px間隔の格子の上 (切り出し後の判定画素: 行 ≡ 1 mod 12)
    assert all(v[0] % 12 == 1 and v[1] % 8 in (2, 6) for v in sites.values() if not v[2])
    assert (14, 19) not in sites and (14, 20) not in sites     # ゴーストの家の扉の切れ端を含まない
    n = len(sites)
    # 古いチェックポイントの壁の切れ端は読み込み時に捨てる
    bad = dict(sites)
    bad[(14, 19)] = (56, 78, True)
    env.load_sites([env.wall], {0: bad})
    assert (14, 19) not in env.pellet_sites[0] and len(env.pellet_sites[0]) == n
    env.close()


def test_cell_step_moves_one_cell():
    """step_mode=cell: 1ステップで Ms. Pac-Man が隣のマスへ移る (動けるときは上限フレーム内に必ず移動)"""
    pytest.importorskip("ale_py")
    from hra.env import MsPacmanEnv
    env = MsPacmanEnv(load_config(overrides=["env.step_mode=cell", "env.position_source=ram"]), seed=0)
    o = env.reset()
    moved, jumps = 0, 0
    for t in range(300):
        prev = o.pacman
        o, _, _, done, info = env.step(2 if (t // 20) % 2 else 3)   # 左右に動かす
        if prev and o.pacman and prev != o.pacman and not info["life_lost"]:
            moved += 1
            if max(abs(prev[0] - o.pacman[0]), abs(prev[1] - o.pacman[1])) > 1 and abs(prev[1] - o.pacman[1]) < 30:
                jumps += 1                                          # 2マス以上の移動 (トンネルのワープは除く)
        if done:
            break
    assert moved > 100 and jumps == 0
    env.close()
