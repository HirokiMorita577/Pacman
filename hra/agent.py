"""
HRA 本体 (FR-12〜FR-37)

GVF (一般価値関数):
  迷路ごと・目標位置ごとに「その位置へ行く」ための擬似Q値 (0〜1) を表で持つ。
  状態 = Ms. Pac-Man の (位置, 向き)  [付録A.2]
  擬似報酬 = 目標位置に着いたら 1 (そこで終端 → 以後の価値は 0。FR-23)
  更新 = 全GVFを同じ遷移からオフポリシーで並行更新 (FR-20)
         式(8) 期待Sarsa (ランダム方策の価値) または 式(7) Q学習 (FR-22)

頭:
  各オブジェクトの頭 = オブジェクトのいる位置のGVF × 倍率 (FR-15)。画面にないオブジェクトは 0 (FR-16)

表の持ち方 (NFR-03):
  見つかった位置・状態だけに番号を振り (辞書)、その番号で引く numpy 配列に値を入れる。
  配列は見つかった分だけ伸ばすので、未到達の位置の分のメモリは使わない。
  付録A.2 の見積もり (4迷路 × 約400位置 × 約950状態 × 9行動 ≈ 14M要素) で float32 約55MB/γ。
"""
import numpy as np

from .env import N_ACTIONS, NORTH, EAST, SOUTH, WEST

# ALE の Ms. Pac-Man の行動: 0 NOOP 1 UP 2 RIGHT 3 LEFT 4 DOWN 5 UPRIGHT 6 UPLEFT 7 DOWNRIGHT 8 DOWNLEFT
# 向きごとに、その向きの成分を含む行動
HEADING_ACTIONS = {NORTH: [1, 5, 6], EAST: [2, 5, 7], SOUTH: [4, 7, 8], WEST: [3, 6, 8]}


class MazeTables:
    """1つの迷路の GVF 表とカウント"""

    def __init__(self, gammas):
        self.pos_index = {}      # 位置 → GVF番号 (FR-13: 新しい位置を見つけたら追加)
        self.state_index = {}    # (位置, 向き) → 状態番号
        self.state_pos = []      # 状態番号 → GVF番号 (その状態の位置)
        self.gammas = list(gammas)
        self.G = {g: np.zeros((16, 64, N_ACTIONS), dtype=np.float32) for g in self.gammas}
        self.n = np.zeros((64, N_ACTIONS), dtype=np.int64)   # カウントベース頭用 n(s,a)
        self._snap = {}

    @property
    def n_gvfs(self):
        return len(self.pos_index)

    def _grow(self):
        T, S = len(self.pos_index), len(self.state_index)

        def cap(old, need):
            return old if need <= old else max(old * 2, need)
        for g, arr in self.G.items():
            t_cap, s_cap, _ = arr.shape
            if T > t_cap or S > s_cap:
                new = np.zeros((cap(t_cap, T), cap(s_cap, S), N_ACTIONS), dtype=np.float32)
                new[:t_cap, :s_cap] = arr
                self.G[g] = new
        if S > self.n.shape[0]:
            new = np.zeros((cap(self.n.shape[0], S), N_ACTIONS), dtype=np.int64)
            new[:self.n.shape[0]] = self.n
            self.n = new

    def state(self, pos, direction):
        """状態番号を返す。初めての位置/状態なら GVF/状態を追加する"""
        if pos not in self.pos_index:
            self.pos_index[pos] = len(self.pos_index)
            self._snap.clear()
        key = (pos, direction)
        s = self.state_index.get(key)
        if s is None:
            s = len(self.state_index)
            self.state_index[key] = s
            self.state_pos.append(self.pos_index[pos])
        self._grow()
        return s

    def gvf_of(self, pos, radius):
        """オブジェクト位置に対応する GVF 番号。なければ radius マス以内で最も近い既知位置 (なければ -1)"""
        t = self.pos_index.get(pos)
        if t is not None:
            return t
        hit = self._snap.get(pos)
        if hit is None:
            hit, best = -1, None
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    q = self.pos_index.get((pos[0] + dr, pos[1] + dc))
                    d = abs(dr) + abs(dc)
                    if q is not None and (best is None or d < best):
                        hit, best = q, d
            self._snap[pos] = hit
        return hit

    def update(self, s, a, s_next, alpha, target):
        """1遷移で全GVFを更新 (FR-20)。s_next の位置を目標とするGVFは擬似報酬1で終端"""
        T = self.n_gvfs
        reached = self.state_pos[s_next]
        for g, G in self.G.items():
            nxt = G[:T, s_next, :]
            v = nxt.mean(axis=1) if target == "expected_sarsa" else nxt.max(axis=1)   # 式(8) / 式(7)
            y = g * v
            y[reached] = 1.0      # 擬似報酬1、終端なので以後の価値は0 (FR-23)
            if alpha == 1.0:
                G[:T, s, a] = y
            else:
                G[:T, s, a] += alpha * (y - G[:T, s, a])

    def trimmed(self):
        T, S = len(self.pos_index), len(self.state_index)
        return {
            "pos_index": self.pos_index, "state_index": self.state_index, "state_pos": self.state_pos,
            "G": {g: arr[:T, :S].copy() for g, arr in self.G.items()}, "n": self.n[:S].copy(),
        }

    @classmethod
    def from_trimmed(cls, d, gammas):
        m = cls(gammas)
        # 読み込み元 (チェックポイントの dict) を書き換えないようにコピーする
        m.pos_index, m.state_index, m.state_pos = dict(d["pos_index"]), dict(d["state_index"]), list(d["state_pos"])
        m._grow()
        T, S = len(m.pos_index), len(m.state_index)
        for g in m.gammas:
            # 保存時と γ が変わった場合は、最も近い γ の表を初期値として写す (途中で γ を変える実験用)
            src = g if g in d["G"] else min(d["G"], key=lambda k: abs(k - g))
            m.G[g][:T, :S] = d["G"][src]
        m.n[:S] = d["n"]
        return m


class HRAAgent:
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        gv, hd = cfg["gvf"], cfg["heads"]
        self.alpha = gv["alpha"]
        self.target = gv["target"]
        self.g_score = gv["gamma_score"]
        self.g_ghost = gv["gamma_ghost"]
        self.gammas = sorted({self.g_score, self.g_ghost})
        self.heads = hd
        self.snap = hd["snap_radius"]
        self.aggregator = cfg["aggregator"]["type"]
        self.normalize = cfg["aggregator"]["normalize"]
        ex = cfg["exploration"]
        self.use_div, self.div_max, self.div_steps = ex["diversification"], ex["div_max"], ex["div_steps"]
        self.use_count, self.kappa = ex["count_based"], ex["kappa"]
        self.heading_bonus = hd.get("heading_bonus", 0.0)
        em = cfg["executive_memory"]
        self.use_memory, self.memory_value = em["enabled"], em["value"]
        self.rng = np.random.default_rng(seed)
        self.mazes = {}
        self.N = 0                 # これまでに取った行動の総数 (式11)
        self.memory = {}           # 面番号 → 行動列 (FR-34)
        self.update_counts = True
        self.last_decision = None

    # ── 状態 ──
    def tables(self, maze):
        if maze not in self.mazes:
            self.mazes[maze] = MazeTables(self.gammas)
        return self.mazes[maze]

    def state(self, objs):
        if objs.pacman is None:
            return None
        return self.tables(objs.maze).state(objs.pacman, objs.direction)

    # ── 頭とアグリゲーター ──
    def _sum_heads(self, tab, s, positions, weights, gamma):
        idx, w = [], []
        for p, wt in zip(positions, weights):
            if p is None:
                continue                       # 画面にないオブジェクトは 0 (FR-16)
            t = tab.gvf_of(p, self.snap)
            if t >= 0:
                idx.append(t)
                w.append(wt)
        if not idx:
            return np.zeros(N_ACTIONS, dtype=np.float64)
        q = tab.G[gamma][np.array(idx), s, :]
        return (np.array(w, dtype=np.float64)[:, None] * q).sum(axis=0)

    def head_values(self, objs, s):
        """得点を生む頭の合計と、通常ゴーストの頭の合計 (倍率なし) を返す"""
        tab = self.tables(objs.maze)
        h = self.heads
        pos = objs.pellets + objs.power_pellets + [objs.fruit] + list(objs.blue_ghosts)
        wts = ([h["pellet"]] * len(objs.pellets) + [h["power_pellet"]] * len(objs.power_pellets)
               + [h["fruit"]] + [h["blue_ghost"]] * len(objs.blue_ghosts))
        score = self._sum_heads(tab, s, pos, wts, self.g_score)
        ghost = self._sum_heads(tab, s, objs.ghosts, [1.0] * len(objs.ghosts), self.g_ghost)
        return score, ghost

    def aggregate(self, score, ghost):
        if self.aggregator == "linear":                     # 式(5) FR-24
            return score + self.heads["ghost_linear"] * ghost
        # 正規化アグリゲーター FR-25
        if self.normalize == "minmax":
            lo, hi = score.min(), score.max()
            sn = (score - lo) / (hi - lo) if hi > lo else np.zeros_like(score)
        else:
            hi = score.max()
            sn = score / hi if hi > 0 else np.zeros_like(score)
        return sn + self.heads["ghost_normalized"] * ghost

    def count_head(self, tab, s):
        """式(11) Q_teh = κ sqrt(N^(1/4) / n(s,a))。n=0 は 1 として扱う"""
        n = np.maximum(tab.n[s], 1)
        return self.kappa * np.sqrt(self.N ** 0.25 / n)

    def q_values(self, objs, s, t, memory_action=None):
        q = np.zeros(N_ACTIONS, dtype=np.float64)
        if s is not None:
            q += self.aggregate(*self.head_values(objs, s))
            if self.use_count:
                q += self.count_head(self.tables(objs.maze), s)
            if self.heading_bonus:
                # 論文にない工夫: 進行方向を含む行動に小さなボーナス。
                # 左右に同じくらいの価値があるときの往復 (その場で止まる) を防ぐ
                q[HEADING_ACTIONS[objs.direction]] += self.heading_bonus
        if self.use_div and t < self.div_steps:            # 式(9)(10) FR-30/31
            q += self.rng.uniform(0.0, self.div_max, N_ACTIONS)
        if memory_action is not None:                       # FR-35
            q[memory_action] += self.memory_value
        return q

    def act(self, objs, s, t, memory_action=None):
        q = self.q_values(objs, s, t, memory_action)
        best = np.flatnonzero(q >= q.max() - 1e-12)
        a = int(self.rng.choice(best))                      # FR-27 (同点はランダム)
        self.last_decision = {'q': q.copy(), 'action': a, 'ties': len(best)}
        if s is not None and self.update_counts:
            self.tables(objs.maze).n[s, a] += 1
        if self.update_counts:
            self.N += 1
        return a

    # ── 学習 ──
    def learn(self, maze, s, a, s_next):
        if s is None or s_next is None:
            return
        self.tables(maze).update(s, a, s_next, self.alpha, self.target)

    # ── 実行記憶 (FR-34〜37) ──
    def memory_action(self, level, level_step, clean):
        if not self.use_memory or not clean:
            return None
        seq = self.memory.get(level)
        if seq is None or level_step >= len(seq):
            return None
        return seq[level_step]

    def remember(self, level, actions):
        if level not in self.memory or len(actions) < len(self.memory[level]):
            self.memory[level] = list(actions)

    # ── 集計・保存 ──
    def stats(self):
        return {"gvfs": sum(m.n_gvfs for m in self.mazes.values()),
                "states": sum(len(m.state_index) for m in self.mazes.values()),
                "mazes": len(self.mazes), "memory_levels": len(self.memory)}

    def state_dict(self):
        return {"mazes": {k: m.trimmed() for k, m in self.mazes.items()}, "N": self.N,
                "memory": self.memory, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, d):
        self.mazes = {k: MazeTables.from_trimmed(v, self.gammas) for k, v in d["mazes"].items()}
        self.N = d["N"]
        self.memory = {k: list(v) for k, v in d["memory"].items()}
        self.rng.bit_generator.state = d["rng"]
