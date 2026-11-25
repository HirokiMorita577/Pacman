"""
Pac-Man Custom Gymnasium Environment
カーナビ風経路表示付きパックマン強化学習環境
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import heapq


# マップ定義 (0=通路, 1=壁, 2=エサ, 3=パワーエサ)
DEFAULT_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1],
    [1,3,1,1,2,1,1,1,2,1,2,1,1,1,2,1,1,3,1],
    [1,2,1,1,2,1,1,1,2,1,2,1,1,1,2,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,2,1,2,1,1,1,1,1,2,1,2,1,1,2,1],
    [1,2,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,2,1],
    [1,1,1,1,2,1,1,1,0,1,0,1,1,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,0,0,0,0,0,0,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,1,1,0,1,1,0,1,2,1,1,1,1],
    [0,0,0,0,2,0,0,1,0,0,0,1,0,0,2,0,0,0,0],
    [1,1,1,1,2,1,0,1,1,1,1,1,0,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,0,0,0,0,0,0,1,2,1,1,1,1],
    [1,1,1,1,2,1,0,1,1,1,1,1,0,1,2,1,1,1,1],
    [1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,1],
    [1,2,1,1,2,1,1,1,2,1,2,1,1,1,2,1,1,2,1],
    [1,3,2,1,2,2,2,2,2,0,2,2,2,2,2,1,2,3,1],
    [1,1,2,1,2,1,2,1,1,1,1,1,2,1,2,1,2,1,1],
    [1,2,2,2,2,1,2,2,2,1,2,2,2,1,2,2,2,2,1],
    [1,2,1,1,1,1,1,1,2,1,2,1,1,1,1,1,1,2,1],
    [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

GHOST_COLORS = [(255,0,0), (255,182,255), (0,255,255), (255,165,0)]

CELL_EMPTY = 0
CELL_WALL = 1
CELL_DOT = 2
CELL_POWER = 3

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
POWER_DURATION = 40
SAFETY_GROUPS = ("area", "dots", "margin")   # tactical_safe の先読み特徴量 (行動ごとに3つ) の並び
LOOKAHEAD_STEPS = 5      # classic_lookahead で敵の動きを予測するステップ数
LOOKAHEAD_BRANCHES = 16  # 怖がった敵の行き先の分岐を、1体あたりこの数までに抑える

# 報酬の要素 (HRA: 要素ごとに別々のQ値ヘッドで学習する)
REWARD_DOT   = 0   # エサ・パワーエサ・エサへの接近・クリア
REWARD_GHOST = 1   # ゴースト衝突/捕食・接近ペナルティ
REWARD_MOVE  = 2   # 時間・ループ・方向維持・未訪問セル
N_REWARD_COMPONENTS = 3
REWARD_COMPONENT_NAMES = ["dot", "ghost", "move"]

# 学習用の報酬の値 (名前: (既定値, 要素, 説明))。PacManEnv(rewards={...}) で上書きできる (PSO で調整: pso_reward_tuning.py)
# ペナルティも正の値で持ち、使う側で引く
REWARD_SPEC = {
    "dot":            (2.0,   REWARD_DOT,   "エサを食べた (基本)"),
    "dot_progress":   (3.0,   REWARD_DOT,   "エサの終盤ボーナス (食べた割合 × この値)"),
    "power":          (5.0,   REWARD_DOT,   "パワーエサを食べた"),
    "approach_dot":   (0.3,   REWARD_DOT,   "最寄りのエサに近づいた"),
    "leave_dot":      (0.1,   REWARD_DOT,   "最寄りのエサから離れた (罰)"),
    "clear":          (100.0, REWARD_DOT,   "全部食べてクリア"),
    "eat_ghost":      (5.0,   REWARD_GHOST, "青ゴーストを食べた"),
    "death":          (25.0,  REWARD_GHOST, "ゴーストにやられた (罰)"),
    "ghost_adjacent": (1.5,   REWARD_GHOST, "ゴーストが隣のマス (罰)"),
    "ghost_near":     (0.5,   REWARD_GHOST, "ゴーストが2マス以内 (罰)"),
    "chase_blue":     (0.3,   REWARD_GHOST, "パワー中にゴーストへ近づいた"),
    "time":           (0.02,  REWARD_MOVE,  "1ステップごと (罰)"),
    "loop":           (2.0,   REWARD_MOVE,  "同じマスに留まり続けた (罰、回数に応じて増える)"),
    "oscillation":    (3.0,   REWARD_MOVE,  "2マスを往復した (罰)"),
    "heading":        (0.15,  REWARD_MOVE,  "同じ向きに進み続けた"),
    "new_cell":       (0.2,   REWARD_MOVE,  "初めてのマスに入った"),
}
OLD_REWARDS = {k: v[0] for k, v in REWARD_SPEC.items()}   # 前回 (2026-09 まで) の値

# 迷路の数え上げ: エサ152個 + パワーエサ4個 = N 156。全部食べたときのゲーム得点 = 152×10 + 4×50 = 1,720
# 報酬はトータルで設計する (先生の指摘): エサ1個 = +10 を単位にし、エサの報酬合計 10N に桁をそろえる
N_PELLETS = 156
# 報酬の整形 (近づく・往復・初めてのマスなど) は前回の比率を保ったまま、エサの単位 (2 → 10) に合わせて5倍する
SHAPING_ITEMS = ["approach_dot", "leave_dot", "ghost_adjacent", "ghost_near", "chase_blue",
                 "loop", "oscillation", "heading", "new_cell"]
DESIGNED_REWARDS = {
    **{k: OLD_REWARDS[k] * 5 for k in SHAPING_ITEMS},
    "dot":          10.0,              # ゲームの得点と同じ。エサの報酬合計 = 10 × 152 = 1,520
    "dot_progress": 0.0,               # 終盤ボーナスはクリアボーナスに置き換える
    "power":        50.0,              # ゲームの得点と同じ
    "eat_ghost":    200.0,             # ゲームの得点と同じ
    "clear":        10.0 * N_PELLETS,  # 1,560 = 10N。0.99^214 ≈ 0.12 を掛けても開始時点でエサ十数個分に見える
    "death":        500.0,             # 10N の約1/3。前回の 25 (エサ12個分) より重いが、10N を超えない
    "time":         0.1,               # 214 ステップ (最短クリア) で −21 (エサ2個分)
}
REWARD_PRESETS = {"old": OLD_REWARDS, "designed": DESIGNED_REWARDS}
DEFAULT_REWARDS = DESIGNED_REWARDS
REWARD_SIGN = {k: -1.0 if "(罰" in v[2] else 1.0 for k, v in REWARD_SPEC.items()}


class Ghost:
    """
    本家パックマンの4体ゴースト行動を再現

    Blinky（赤）  : パックマンの現在位置を直接追跡
    Pinky（ピンク）: パックマンの進行方向4マス先を狙う（先回り）
    Inky（水色）  : Blinkyとパックマンを使った挟み撃ち
    Clyde（橙）   : 8マス以上遠い→追跡、近い→散開コーナーへ逃げる
    """
    # Clyde の散開コーナー（左下）
    CLYDE_SCATTER = (20, 1)
    # 散開モード時の各ゴーストのコーナー
    SCATTER_TARGETS = [
        (1,  17),   # Blinky → 右上
        (1,   1),   # Pinky  → 左上
        (20, 17),   # Inky   → 右下
        (20,  1),   # Clyde  → 左下
    ]

    def __init__(self, row, col, color, ghost_id=0):
        self.start    = (row, col)
        self.pos      = [row, col]
        self.color    = color
        self.ghost_id = ghost_id   # 0:Blinky 1:Pinky 2:Inky 3:Clyde
        self.scared       = False
        self.scared_timer = 0
        self.last_pos     = None

    def reset(self):
        self.pos      = list(self.start)
        self.scared       = False
        self.scared_timer = 0
        self.last_pos     = None

    def _valid_moves(self, grid):
        rows, cols = len(grid), len(grid[0])
        r, c = self.pos
        moves = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL:
                moves.append((nr, nc))
        return moves

    def _get_target(self, pacman_pos, pacman_dir, blinky_pos):
        """本家ルールに基づいてターゲット座標を返す"""
        pr, pc = pacman_pos
        # 進行方向オフセット
        dir_dr = {ACTION_UP:-1, ACTION_DOWN:1, ACTION_LEFT:0,  ACTION_RIGHT:0}
        dir_dc = {ACTION_UP:0,  ACTION_DOWN:0, ACTION_LEFT:-1, ACTION_RIGHT:1}
        dr = dir_dr.get(pacman_dir, 0)
        dc = dir_dc.get(pacman_dir, 0)

        if self.ghost_id == 0:   # Blinky: 現在位置を直接追跡
            return (pr, pc)

        elif self.ghost_id == 1: # Pinky: 進行方向4マス先
            return (pr + dr*4, pc + dc*4)

        elif self.ghost_id == 2: # Inky: 挟み撃ち
            # パックマンの2マス先 → Blinkyからその2倍の延長線上
            mid_r = pr + dr*2
            mid_c = pc + dc*2
            br, bc = blinky_pos
            return (mid_r*2 - br, mid_c*2 - bc)

        else:                    # Clyde: 距離8以上→追跡、近い→左下コーナー
            dist = abs(self.pos[0]-pr) + abs(self.pos[1]-pc)
            if dist >= 8:
                return (pr, pc)
            else:
                return self.CLYDE_SCATTER

    def move(self, grid, pacman_pos, pacman_dir, blinky_pos, rng, mode="classic"):
        """
        mode="bfs"     : 全ゴーストがパックマンを直接BFS追跡（シンプル版）
        mode="classic" : 本家Pac-Manの4体個別ターゲット行動
        """
        valid = self._valid_moves(grid)
        if not valid:
            return

        # 逆戻り禁止（分岐がある場合のみ）
        no_reverse = [p for p in valid if list(p) != self.last_pos]
        candidates = no_reverse if no_reverse else valid

        if self.scared:
            # パワーエサ中: パックマンから最も遠いマスへ逃げる
            farthest_dist = -1
            for p in candidates:
                d = abs(p[0]-pacman_pos[0]) + abs(p[1]-pacman_pos[1])
                if d > farthest_dist:
                    farthest_dist = d
            farthest = [p for p in candidates
                        if abs(p[0]-pacman_pos[0])+abs(p[1]-pacman_pos[1]) == farthest_dist]
            chosen = rng.choice(farthest)
        else:
            # ターゲット決定: bfsモードは常にパックマン直接追跡
            if mode == "bfs":
                target = pacman_pos
            else:
                target = self._get_target(pacman_pos, pacman_dir, blinky_pos)

            rows, cols = len(grid), len(grid[0])
            target = (max(0, min(rows-1, target[0])),
                      max(0, min(cols-1, target[1])))

            path = _bfs_next(grid, tuple(self.pos), target)
            if path and len(path) > 1 and path[1] in candidates:
                chosen = path[1]
            else:
                # BFS失敗 → マンハッタン距離でターゲットに最も近い方向
                candidates.sort(key=lambda p: abs(p[0]-target[0])+abs(p[1]-target[1]))
                chosen = candidates[0]

        self.last_pos = self.pos[:]
        self.pos = list(chosen)

        if self.scared_timer > 0:
            self.scared_timer -= 1
            if self.scared_timer == 0:
                self.scared = False


def _bfs_next(grid, start, goal):
    """BFSで start→goal の最短経路を返す"""
    rows, cols = len(grid), len(grid[0])
    goal = tuple(goal)
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        r, c = pos
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL and (nr,nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path+[(nr, nc)]))
    return []


def astar(grid, start, goal):
    """A*アルゴリズムで経路探索"""
    rows, cols = len(grid), len(grid[0])
    if grid[goal[0]][goal[1]] == CELL_WALL:
        return []

    def h(p):
        return abs(p[0]-goal[0]) + abs(p[1]-goal[1])

    open_heap = [(h(start), 0, start, [start])]
    visited = set()

    while open_heap:
        f, g, pos, path = heapq.heappop(open_heap)
        if pos in visited:
            continue
        visited.add(pos)
        if pos == goal:
            return path
        r, c = pos
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL and (nr,nc) not in visited:
                new_g = g + 1
                heapq.heappush(open_heap, (new_g + h((nr,nc)), new_g, (nr,nc), path+[(nr,nc)]))
    return []


def find_nearest_dot(grid, pos):
    """最近傍のエサをBFSで探す"""
    rows, cols = len(grid), len(grid[0])
    queue = deque([(pos, [pos])])
    visited = {pos}
    while queue:
        (r, c), path = queue.popleft()
        if grid[r][c] in (CELL_DOT, CELL_POWER):
            return (r, c), path
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != CELL_WALL and (nr,nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path+[(nr, nc)]))
    return None, []


class PacManEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, ghost_mode="classic", no_ghosts=False, num_ghosts=1,
                 rewards=None, lives=1, include_lives=False, max_steps=None,
                 observation_version="legacy", fix_leave_dot=False, safety_timing=False,
                 safety_groups=SAFETY_GROUPS, safe_route=False, classic_lookahead=False,
                 route_version="legacy"):
        """
        ghost_mode="classic" : 本家Pac-Manの4体個別行動（Blinky/Pinky/Inky/Clyde）
        ghost_mode="bfs"     : 全ゴーストがパックマンをBFSで直接追跡（シンプル版）
        no_ghosts=True       : ゴーストなし（カリキュラム学習フェーズ1用）
        num_ghosts           : ゴーストの数 (1〜4, Blinky→Pinky→Inky→Clydeの順に追加)
        rewards              : 学習用の報酬の値の上書き {名前: 値} (REWARD_SPEC 参照)。ゲームの得点は変わらない
        lives                : 残機。捕まるたびに1減り、0でエピソード終了。残っていればパックマンとゴーストを
                               開始位置に戻して続ける (エサはそのまま。本家と同じ)。既定 1 = 1回捕まったら終了
        """
        super().__init__()
        unknown = set(rewards or {}) - set(DEFAULT_REWARDS)
        if unknown:
            raise ValueError(f"unknown reward names: {sorted(unknown)}")
        self.rw = {**DEFAULT_REWARDS, **(rewards or {})}
        self.max_lives = max(1, int(lives))
        # Opt in for new experiments; legacy checkpoints keep their input size.
        self.include_lives = bool(include_lives)
        if observation_version not in ("legacy", "tactical", "tactical_safe"):
            raise ValueError("observation_version must be legacy, tactical or tactical_safe")
        self.observation_version = observation_version
        # True: エサを食べた直後に最寄りエサが遠くなっただけでは leave_dot を付けない (2026-10-07 の失敗解析で、
        # leave_dot の約半数が「食べた直後」に付いていたため)。既定 False は従来どおり
        self.fix_leave_dot = bool(fix_leave_dot)
        # tactical_safe の先読み特徴量の設定 (2026-10-08 追加。既定は第2回比較の safe_obs と同じ)
        #   safety_timing=True : 怖がっている敵も、恐怖の残り時間が切れた後に着けるマスでは危険として数える
        #   safety_groups      : 使う特徴群 (area / dots / margin)。外した群は 0 にする (入力の次元は変えない)
        self.safety_timing = bool(safety_timing)
        self.safety_groups = tuple(safety_groups)
        if set(self.safety_groups) - set(SAFETY_GROUPS) or len(set(self.safety_groups)) != len(self.safety_groups):
            raise ValueError(f"safety_groups must be a subset of {SAFETY_GROUPS}")
        # safe_route=True : 行動ごとに「安全なマスだけを通って行ける最寄りのエサの近さ」と「最寄りのエサまでの最短経路上で、
        #                   敵より先に着ける余裕の最小値」を追加する (4行動 x 2)。終盤、安全な領域にエサが無いと
        #                   既存の特徴量が全行動で 0 になり、どのエサを狙えば安全かが分からないため (2026-10-08 の失敗分類)
        self.safe_route = bool(safe_route)
        # v2 adds an unconstrained food distance, preserves far-away distances,
        # and chooses the safest of equally short paths. Keep old models opt-out.
        if route_version not in ("legacy", "v2"):
            raise ValueError("route_version must be legacy or v2")
        if route_version != "legacy" and not self.safe_route:
            raise ValueError("route_version='v2' needs safe_route=True")
        self.route_version = route_version
        # classic_lookahead=True : 行動ごとに、敵の実際の動き方 (classic の狙い方・逆戻りしない規則・怖がった敵の逃げ方) で
        #                          LOOKAHEAD_STEPS 手先まで敵の位置を予測し、捕まらずにいられるマスを数える (4行動 x 3)。
        #                          最短距離で自由に動けるとみなす safety 特徴量より保守的でない見積もり (2026-10-10 追加)
        self.classic_lookahead = bool(classic_lookahead)
        if (self.safety_timing or self.safety_groups != SAFETY_GROUPS or self.safe_route or self.classic_lookahead) \
                and observation_version != "tactical_safe":
            raise ValueError("safety_timing / safety_groups / safe_route / classic_lookahead need observation_version='tactical_safe'")
        self.max_steps = int(max_steps if max_steps is not None else (600 if no_ghosts else 1000))
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.base_map = [row[:] for row in DEFAULT_MAP]
        self.rows = len(self.base_map)
        self.cols = len(self.base_map[0])
        self.render_mode = render_mode
        self.ghost_mode = ghost_mode
        self.no_ghosts = no_ghosts
        self.num_ghosts = max(1, min(4, num_ghosts))
        self.cell_size = 28
        self.screen = None
        self.clock = None
        self.rng = np.random.default_rng()

        # 観測空間: グリッド状態(4チャネル) + パックマン位置 + ゴースト位置
        #   + 拡張特徴量: 最近エサ距離(1) + ゴースト距離(n) + 最近エサ方向(4) + パワー残り(1)
        #   + 行動別特徴量: 4方向 × (通路/エサの近さ/危険ゴーストの近さ/弱体ゴーストの近さ)
        n = self.num_ghosts
        obs_size = self.rows * self.cols * 4 + 2 + n*2 + (1 + n + 4 + 1) + 4*4
        # The open-neighbour flags are also available to the agent's action mask.
        self.action_mask_indices = tuple(obs_size - 16 + 4 * a for a in range(4))
        obs_size += int(self.include_lives)
        if self.observation_version != "legacy":
            # Player heading, then per ghost: frightened, timer, heading (4 + unknown).
            obs_size += 4 + 7 * n
        if self.observation_version == "tactical_safe":
            # 行動ごとの先読み: 先に着けるマスの数 / その中のエサ / 移動先での危険な敵との余裕 (4行動 x 3)
            obs_size += 4 * 3
            if self.safe_route:
                obs_size += 4 * (3 if self.route_version == "v2" else 2)
            if self.classic_lookahead:
                obs_size += 4 * 3
        self.observation_space = spaces.Box(0, 1, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # 経路表示用
        self.planned_path = []
        self.target_pos = None

    def _init_game(self):
        self.grid = [row[:] for row in self.base_map]
        self.pacman = [16, 9]  # スタート位置
        self.pacman_dir = ACTION_RIGHT  # 初期向き
        self.mouth_tick = 0             # 口アニメ用カウンタ
        self.total_dots = sum(1 for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] in (CELL_DOT, CELL_POWER))
        self.dots_eaten = 0
        self.score = 0
        # ゲーム得点の内訳 (2026-10-08 追加。得点ルールからの逆算を不要にする診断用。ゲームの進み方は変えない)
        self.score_parts = {"dot": 0, "power": 0, "ghost": 0}
        self.ghosts_eaten = 0
        self.power_eaten = 0
        self.power_steps = []          # パワーエサを取ったステップ (診断用。2026-10-15 追加)
        self.power_mode = False
        self.power_timer = 0
        # ゴースト初期位置: ゴーストハウス内の通路セルに配置
        self.ghosts = [
            Ghost(10, 9,  GHOST_COLORS[0], ghost_id=0),  # Blinky（赤）   直接追跡
            Ghost(10, 8,  GHOST_COLORS[1], ghost_id=1),  # Pinky（ピンク）先回り
            Ghost(10, 10, GHOST_COLORS[2], ghost_id=2),  # Inky（水色）   挟み撃ち
            Ghost(12, 9,  GHOST_COLORS[3], ghost_id=3),  # Clyde（橙）    距離で切替
        ][:self.num_ghosts]
        self.steps = 0
        self.planned_path = []
        self.target_pos = None
        self.last_qvalues = None      # Q値可視化用
        self.pos_history = []         # ループ検出①: 位置履歴
        self.dir_history = []         # ループ検出: 方向履歴
        self.visited_cells = set()    # 未訪問セルボーナス用
        self.ghost_paths = [[] for _ in range(self.num_ghosts)]  # ゴースト経路表示用
        self.prev_dot_dist = None     # 報酬シェーピング用
        self.reward_items = {}        # このエピソードの報酬の項目別合計 (info["reward_items"])
        self._ate_since_dot_measure = False
        self.lives = self.max_lives
        self._respawned = False
        self.cleared = False

    def set_qvalues(self, qvalues):
        """エージェントのQ値を受け取り描画に使う (numpy配列 shape=[4])"""
        self.last_qvalues = np.array(qvalues, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_game()
        self._update_planned_path()
        return self._get_obs(), {}

    def _bfs_distance(self, start, target_cells):
        """BFSで start から target_cells のいずれかまでの最短距離を返す"""
        from collections import deque
        sr, sc = start
        visited = {(sr, sc)}
        queue = deque([(sr, sc, 0)])
        max_dist = self.rows + self.cols
        while queue:
            r, c, d = queue.popleft()
            if (r, c) in target_cells:
                return d
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.rows and 0<=nc<self.cols and (nr,nc) not in visited and self.grid[nr][nc] != CELL_WALL:
                    visited.add((nr,nc))
                    queue.append((nr, nc, d+1))
        return max_dist

    def _get_obs(self):
        # 4チャネル: 壁/エサ/パワーエサ/ゴースト
        ch_wall = np.array([[1 if c == CELL_WALL else 0 for c in row] for row in self.grid], dtype=np.float32).flatten()
        ch_dot  = np.array([[1 if c == CELL_DOT else 0 for c in row] for row in self.grid], dtype=np.float32).flatten()
        ch_pow  = np.array([[1 if c == CELL_POWER else 0 for c in row] for row in self.grid], dtype=np.float32).flatten()
        gh_map  = np.zeros((self.rows, self.cols), dtype=np.float32)
        for g in self.ghosts:
            gh_map[g.pos[0], g.pos[1]] = 1.0
        ch_ghost = gh_map.flatten()

        pacman_norm = np.array([self.pacman[0]/self.rows, self.pacman[1]/self.cols], dtype=np.float32)
        ghost_norm  = np.array([c for g in self.ghosts for c in [g.pos[0]/self.rows, g.pos[1]/self.cols]], dtype=np.float32)

        # 拡張特徴量
        max_dist = float(self.rows + self.cols)
        pr, pc = self.pacman

        # 最近エサまでのBFS距離（正規化）
        dot_cells = {(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] in (CELL_DOT, CELL_POWER)}
        nearest_dot_dist = self._bfs_distance((pr, pc), dot_cells) / max_dist if dot_cells else 0.0

        # 各ゴーストまでのBFS距離（正規化）
        ghost_dists = np.array([
            self._bfs_distance((pr, pc), {(g.pos[0], g.pos[1])}) / max_dist
            for g in self.ghosts
        ], dtype=np.float32)

        # 最近エサへの方向（4方向one-hot: 上下左右）
        dot_dir = np.zeros(4, dtype=np.float32)
        if dot_cells:
            nearest, path = find_nearest_dot(self.grid, (pr, pc))
            if nearest:
                target = path[1] if self.observation_version != "legacy" and len(path) > 1 else nearest
                dr = target[0] - pr
                dc = target[1] - pc
                if abs(dr) >= abs(dc):
                    dot_dir[0 if dr < 0 else 1] = 1.0  # 上/下
                else:
                    dot_dir[2 if dc < 0 else 3] = 1.0  # 左/右

        # パワーモード残り時間（正規化）
        duration = POWER_DURATION if self.observation_version != "legacy" else 30.0
        power_remain = np.array([self.power_timer / duration], dtype=np.float32)

        extra = np.concatenate([[nearest_dot_dist], ghost_dists, dot_dir, power_remain])
        observation = np.concatenate([ch_wall, ch_dot, ch_pow, ch_ghost, pacman_norm, ghost_norm, extra,
                                      self._action_features(dot_cells)])
        if self.include_lives:
            observation = np.concatenate([observation, [self.lives / self.max_lives]])
        if self.observation_version != "legacy":
            heading = np.eye(4, dtype=np.float32)[self.pacman_dir]
            ghosts = []
            for g in self.ghosts:
                direction = np.zeros(5, dtype=np.float32)
                delta = tuple(g.pos[i] - g.last_pos[i] for i in range(2)) if g.last_pos is not None else None
                direction[DIRS.index(delta) if delta in DIRS else 4] = 1.0
                ghosts.extend([float(g.scared), g.scared_timer / POWER_DURATION, *direction])
            observation = np.concatenate([observation, heading, ghosts])
            if self.observation_version == "tactical_safe":
                observation = np.concatenate([observation, self._safety_features(dot_cells)])
                if self.safe_route:
                    observation = np.concatenate([observation, self._route_features(dot_cells)])
                if self.classic_lookahead:
                    observation = np.concatenate([observation, self._lookahead_features(dot_cells)])
            # Maze paths can exceed rows + cols; keep the declared Box bounds.
            observation = np.clip(observation, 0.0, 1.0)
        return observation.astype(np.float32)

    def _safety_features(self, dot_cells):
        """行動ごとの先読み特徴量 (4行動 x 3)。今見えているゴーストの位置だけを使う (未来の乱数は使わない)。
        移動先のマスから、パックマンが危険なゴーストのどれよりも先に着けるマス (d_pac + 1 < d_ghost) を「安全な領域」とし、
          [0] 安全な領域のマス数 / 60   [1] その中のエサの数 / 20   [2] 移動先での危険な敵との距離の余裕 / 10
        いずれも 0〜1 に切る。挟み撃ちで逃げ場がなくなる方向ほど [0] が小さくなる。
        safety_timing=True のときは、怖がっている敵 (恐怖の残り t ステップ) も「max(迷路距離, t) ステップ後に着く危険な敵」
        として数える (怖がった敵も通常と同じ速さで動き、t 回動くと元に戻るため)。既定 False は怖がった敵を除外する。
        safety_groups で外した特徴群は 0 にする"""
        ghost_field = self._threat_field()
        pr, pc = self.pacman
        feats = np.zeros((4, 3), dtype=np.float32)
        for a, (dr, dc) in enumerate(DIRS):
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != CELL_WALL):
                nr, nc = pr, pc
            pac_field = self._distance_field([(nr, nc)])
            safe = dots = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    d = pac_field[r][c]
                    if d < 0:
                        continue
                    g = ghost_field[r][c] if ghost_field is not None else -1
                    if g < 0 or d + 1 < g:
                        safe += 1
                        dots += (r, c) in dot_cells
            margin = ghost_field[nr][nc] - 1 if ghost_field is not None and ghost_field[nr][nc] >= 0 else 10
            feats[a] = (min(safe / 60.0, 1.0), min(dots / 20.0, 1.0), min(max(margin, 0) / 10.0, 1.0))
        for k, name in enumerate(SAFETY_GROUPS):
            if name not in self.safety_groups:
                feats[:, k] = 0.0
        return feats.flatten()

    def _route_features(self, dot_cells):
        """行動ごとのエサへの経路の安全さ (4行動 x 2)。今見えている敵の位置だけを使う。
          [0] 移動先から、敵より先に着けるマス (d + 1 < 敵の到着) だけを通って行ける最寄りのエサの近さ: 1 - min(d, 30) / 30
              (安全に行けるエサが無ければ 0)
          [1] 移動先から最寄りのエサまでの最短経路上で、敵の到着 − (d + 1) の最小値 / 10 (0〜1。敵がいなければ 1)"""
        if self.route_version == "v2":
            return self._route_features_v2(dot_cells)
        threat = self._threat_field()
        pr, pc = self.pacman
        feats = np.zeros((4, 2), dtype=np.float32)
        if not dot_cells:
            return feats.flatten()

        def danger(r, c, d):
            g = threat[r][c] if threat is not None else -1
            return None if g < 0 else g - (d + 1)

        for a, (dr, dc) in enumerate(DIRS):
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != CELL_WALL):
                nr, nc = pr, pc
            # [0] 安全なマスだけを通る BFS
            m = danger(nr, nc, 0)
            if m is None or m > 0:
                seen = {(nr, nc): 0}
                queue = deque([(nr, nc)])
                while queue:
                    r, c = queue.popleft()
                    d = seen[(r, c)]
                    if (r, c) in dot_cells:
                        feats[a, 0] = 1.0 - min(d, 30) / 30.0
                        break
                    for ddr, ddc in DIRS:
                        q = (r + ddr, c + ddc)
                        if (0 <= q[0] < self.rows and 0 <= q[1] < self.cols and q not in seen
                                and self.grid[q[0]][q[1]] != CELL_WALL):
                            mq = danger(q[0], q[1], d + 1)
                            if mq is None or mq > 0:
                                seen[q] = d + 1
                                queue.append(q)
            # [1] 最寄りのエサまでの最短経路 (制限なし) 上の余裕の最小値
            parent = {(nr, nc): None}
            dist = {(nr, nc): 0}
            queue = deque([(nr, nc)])
            goal = None
            while queue:
                cell = queue.popleft()
                if cell in dot_cells:
                    goal = cell
                    break
                for ddr, ddc in DIRS:
                    q = (cell[0] + ddr, cell[1] + ddc)
                    if (0 <= q[0] < self.rows and 0 <= q[1] < self.cols and q not in parent
                            and self.grid[q[0]][q[1]] != CELL_WALL):
                        parent[q], dist[q] = cell, dist[cell] + 1
                        queue.append(q)
            worst = 10
            cell = goal
            while cell is not None:
                mq = danger(cell[0], cell[1], dist[cell])
                if mq is not None:
                    worst = min(worst, mq)
                cell = parent[cell]
            feats[a, 1] = min(max(worst, 0), 10) / 10.0 if goal is not None else 0.0
        return feats.flatten()

    def _route_features_v2(self, dot_cells):
        """行動ごとの経路特徴 (4 x 3)。旧版の入力の意味・次元は変えない。

        [0] 最寄りエサへの近さ (危険を無視した迷路距離)。
        [1] 敵より先に着くマスだけを通って行ける最寄りエサへの近さ。
        どちらも 10 / (10 + d)。遠いエサでも正、到達不能・エサなしだけ 0。
        [2] 最寄りエサへの全最短経路のうち、経路上の最小到着余裕が最大の
        経路の余裕 / 10 (0〜1)。同距離のエサも全て比較する。

        安全性は _threat_field の静的な近似。パワー取得・捕食による敵の
        状態変化やエサ取得後の退路は予測しない。[0] を別に残すことで、
        この近似が全経路を危険と判定しても目的地までの距離を失わない。
        """
        feats = np.zeros((4, 3), dtype=np.float32)
        if not dot_cells:
            return feats.flatten()
        threat = self._threat_field()

        def neighbours(cell):
            for dr, dc in DIRS:
                r, c = cell[0] + dr, cell[1] + dc
                if (0 <= r < self.rows and 0 <= c < self.cols
                        and self.grid[r][c] != CELL_WALL):
                    yield (r, c)

        def margin(cell, distance):
            arrival = threat[cell[0]][cell[1]] if threat is not None else -1
            return 10 if arrival < 0 else min(10, arrival - distance - 1)

        start = tuple(self.pacman)
        open_steps = set(neighbours(start))
        for a, (dr, dc) in enumerate(DIRS):
            target = (start[0] + dr, start[1] + dc)
            first = target if target in open_steps else start
            # FIFO BFS finishes all parents at depth d-1 before visiting depth d.
            # Carry the best bottleneck over *all* shortest-path predecessors.
            dist = {first: 0}
            best_margin = {first: margin(first, 0)}
            queue = deque([first])
            goal_distance = None
            goal_margin = -float("inf")
            while queue:
                cell = queue.popleft()
                d = dist[cell]
                if goal_distance is not None and d > goal_distance:
                    break
                if cell in dot_cells:
                    goal_distance = d
                    goal_margin = max(goal_margin, best_margin[cell])
                    continue
                if goal_distance is not None:
                    continue
                for q in neighbours(cell):
                    candidate = min(best_margin[cell], margin(q, d + 1))
                    if q not in dist:
                        dist[q] = d + 1
                        best_margin[q] = candidate
                        queue.append(q)
                    elif dist[q] == d + 1:
                        best_margin[q] = max(best_margin[q], candidate)
            if goal_distance is not None:
                feats[a, 0] = 10.0 / (10.0 + goal_distance)
                feats[a, 2] = max(goal_margin, 0) / 10.0

            # A safe detour may be longer than the unconstrained shortest path.
            # Earlier arrival dominates later arrival under this static threat field.
            if margin(first, 0) <= 0:
                continue
            safe_dist = {first: 0}
            queue = deque([first])
            while queue:
                cell = queue.popleft()
                d = safe_dist[cell]
                if cell in dot_cells:
                    feats[a, 1] = 10.0 / (10.0 + d)
                    break
                for q in neighbours(cell):
                    if q not in safe_dist and margin(q, d + 1) > 0:
                        safe_dist[q] = d + 1
                        queue.append(q)
        return feats.flatten()

    def predict_ghosts(self, pacman_pos, pacman_dir, steps, power_now=False):
        """敵の位置を steps 手先まで予測する (環境の状態・乱数は変えない)。
        パックマンは pacman_pos に留まり pacman_dir を向いているとみなす (敵の狙う位置の計算に使う)。
        返り値: [t=0..steps] ごとに、敵ごとの状態のリスト [(pos, last_pos, scared, timer), ...]。
        通常の敵は Ghost.move と同じ規則で1通りに決まる。怖がった敵は「パックマンから最も遠い候補」のどれかへ
        乱数で動くので、候補をすべて分岐として残す (1体あたり LOOKAHEAD_BRANCHES まで)。
        power_now=True は、この1手目でパワーエサを取る場合 (敵が動く前に全員が怖がる。環境と同じ順序)"""
        states = [[(tuple(g.pos), None if g.last_pos is None else tuple(g.last_pos), g.scared, g.scared_timer)]
                  for g in self.ghosts]
        timeline = [states]
        probe = Ghost(0, 0, (0, 0, 0))
        for t in range(1, steps + 1):
            nxt = []
            # Inky の狙いには、動く前の Blinky の位置を使う (環境と同じ)。Blinky が分岐していれば、その全部について計算する
            blinkies = list(dict.fromkeys(st[0] for st in states[0]))
            for gi, g in enumerate(self.ghosts):
                out = []
                for pos, last, scared, timer in states[gi]:
                    if t == 1 and power_now:
                        scared, timer = True, POWER_DURATION
                    probe.ghost_id = g.ghost_id
                    probe.pos, probe.last_pos = list(pos), None if last is None else list(last)
                    probe.scared, probe.scared_timer = scared, timer
                    valid = probe._valid_moves(self.grid)
                    if not valid:
                        out.append((pos, last, scared, timer))
                        continue
                    if scared:
                        cands = [q for q in valid if list(q) != probe.last_pos] or valid
                        far = max(abs(q[0] - pacman_pos[0]) + abs(q[1] - pacman_pos[1]) for q in cands)
                        moves = [q for q in cands if abs(q[0] - pacman_pos[0]) + abs(q[1] - pacman_pos[1]) == far]
                    else:
                        moves = []
                        for blinky in (blinkies if g.ghost_id == 2 else blinkies[:1]):
                            probe.pos, probe.last_pos = list(pos), None if last is None else list(last)
                            probe.move(self.grid, tuple(pacman_pos), pacman_dir, blinky, None, mode=self.ghost_mode)
                            moves.append(tuple(probe.pos))
                        moves = list(dict.fromkeys(moves))
                    nt = timer - 1 if timer > 0 else 0
                    for q in moves:
                        out.append((tuple(q), pos, scared and nt > 0, nt))
                nxt.append(list(dict.fromkeys(out))[:LOOKAHEAD_BRANCHES])
            states = nxt
            timeline.append(states)
        return timeline

    def _lookahead_features(self, dot_cells):
        """行動ごとの、敵の動きの予測にもとづく先読み (4行動 x 3)。
          [0] LOOKAHEAD_STEPS 手後に、捕まらずにいられるマスの数 / 10 (0 なら予測の上では逃げ場がない)
          [1] その間に安全に通れるマスにあるエサの数 / 5
          [2] 捕まらずにいられる最後の手 / LOOKAHEAD_STEPS (最後まで逃げ場があれば 1)
        マス c に時刻 t にいてよい条件: 危険な敵が時刻 t-1 にも t にも c にいない (環境の衝突判定①②・すり抜けに対応)。
        パックマンは2手目以降、隣のマスへ動くか、壁のある向きを選んでその場に留まれる。
        敵の狙いの計算ではパックマンを1手目の位置に置いたままにする (近似)。
        2手目以降に取るパワーエサの効果と、捕食で敵が戻る効果は入れない (その分は保守的)"""
        H = LOOKAHEAD_STEPS
        feats = np.zeros((4, 3), dtype=np.float32)
        if self.no_ghosts:
            feats[:] = 1.0
            return feats.flatten()
        pr, pc = self.pacman
        opens = lambda r, c: 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != CELL_WALL
        for a, (dr, dc) in enumerate(DIRS):
            nr, nc = pr + dr, pc + dc
            moved = opens(nr, nc)
            n1 = (nr, nc) if moved else (pr, pc)
            heading = a if moved else self.pacman_dir
            power_now = self.grid[n1[0]][n1[1]] == CELL_POWER
            timeline = self.predict_ghosts(n1, heading, H, power_now=power_now)
            danger = [{s[0] for g in states for s in g if not s[2]} for states in timeline]
            if power_now:
                danger[0] = set()      # パワーエサを取った直後の衝突①では、敵はもう怖がっている (食べられる)
            reach = {n1} if n1 not in danger[0] and n1 not in danger[1] else set()
            seen, last_t = set(reach), (1 if reach else 0)
            for t in range(2, H + 1):
                nxt = set()
                for r, c in reach:
                    stay = any(not opens(r + x, c + y) for x, y in DIRS)
                    cand = [(r + x, c + y) for x, y in DIRS if opens(r + x, c + y)] + ([(r, c)] if stay else [])
                    nxt.update(q for q in cand if q not in danger[t - 1] and q not in danger[t])
                reach = nxt
                if not reach:
                    break
                seen |= reach
                last_t = t
            feats[a] = (min(len(reach) / 10.0, 1.0), min(len(seen & dot_cells) / 5.0, 1.0), last_t / H)
        return feats.flatten()

    def _threat_field(self):
        """各マスに危険な敵が着くまでの最短ステップ数 (着けない・危険な敵がいないときは None / -1)"""
        if self.no_ghosts:
            return None
        if not self.safety_timing:
            dangerous = [tuple(g.pos) for g in self.ghosts if not g.scared]
            return self._distance_field(dangerous) if dangerous else None
        field = None
        for g in self.ghosts:
            own = self._distance_field([tuple(g.pos)])
            delay = g.scared_timer if g.scared else 0
            if field is None:
                field = [[-1] * self.cols for _ in range(self.rows)]
            for r in range(self.rows):
                for c in range(self.cols):
                    d = own[r][c]
                    if d < 0:
                        continue
                    d = max(d, delay)
                    if field[r][c] < 0 or d < field[r][c]:
                        field[r][c] = d
        return field

    def _distance_field(self, sources):
        """複数の始点からのBFS距離マップ (到達不能は -1)"""
        dist = [[-1] * self.cols for _ in range(self.rows)]
        queue = deque()
        for r, c in sources:
            if dist[r][c] < 0:
                dist[r][c] = 0
                queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            d = dist[r][c] + 1
            for dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and dist[nr][nc] < 0 and self.grid[nr][nc] != CELL_WALL):
                    dist[nr][nc] = d
                    queue.append((nr, nc))
        return dist

    def _action_features(self, dot_cells):
        """
        各行動(上下左右)の移動先から見た特徴量。
        全体マップの平坦化ベクトルだけではMLPが「どちらへ動くべきか」を読み取りにくいため、
        行動ごとに エサ・ゴーストまでの迷路上の距離 を直接与える。
        壁方向はその場に留まるので現在位置の値を使う。
        """
        dangerous = [tuple(g.pos) for g in self.ghosts if not g.scared and not self.no_ghosts]
        scared    = [tuple(g.pos) for g in self.ghosts if g.scared and not self.no_ghosts]
        fields = [
            (self._distance_field(dot_cells) if dot_cells else None, 20.0),
            (self._distance_field(dangerous) if dangerous else None, 10.0),
            (self._distance_field(scared) if scared else None, 15.0),
        ]

        pr, pc = self.pacman
        feats = np.zeros((4, 4), dtype=np.float32)
        for a, (dr, dc) in enumerate(DIRS):
            nr, nc = pr+dr, pc+dc
            is_open = 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != CELL_WALL
            if not is_open:
                nr, nc = pr, pc
            feats[a, 0] = 1.0 if is_open else 0.0
            for k, (field, horizon) in enumerate(fields, start=1):
                if field is None or field[nr][nc] < 0:
                    continue
                # 近いほど1に近い (horizon以上離れていれば0)
                feats[a, k] = 1.0 - min(field[nr][nc], horizon) / horizon
        return feats.flatten()

    def _update_planned_path(self):
        """カーナビ風: 最近傍のエサへのA*経路を計算"""
        pos = tuple(self.pacman)
        target, _ = find_nearest_dot(self.grid, pos)
        if target:
            self.target_pos = target
            self.planned_path = astar(self.grid, pos, target)
        else:
            self.planned_path = []
            self.target_pos = None

    def _update_ghost_paths(self):
        """各ゴーストのパックマンへのBFS経路を計算"""
        target = tuple(self.pacman)
        for i, g in enumerate(self.ghosts):
            if g.scared:
                self.ghost_paths[i] = []
            else:
                self.ghost_paths[i] = _bfs_next(self.grid, tuple(g.pos), target)

    def _give(self, rc, name, scale=1.0):
        """報酬 name を (罰なら負で) 該当要素に加え、エピソード内の項目別合計 (info["reward_items"]) に記録する"""
        v = REWARD_SIGN[name] * self.rw[name] * scale
        rc[REWARD_SPEC[name][1]] += v
        self.reward_items[name] = self.reward_items.get(name, 0.0) + v

    def _handle_ghost_collision(self, g, rc, terminated):
        """ゴーストとの衝突処理。報酬は rc[REWARD_GHOST] に加算し、terminated を返す"""
        if self.no_ghosts or terminated:
            return terminated
        if g.scared:
            g.reset()
            self.score += 200
            self.score_parts["ghost"] += 200
            self.ghosts_eaten += 1
            self._give(rc, "eat_ghost")
        else:
            # ループ罰(-2〜-3/step)より死亡が「安く」ならないよう大きめにする
            self._give(rc, "death")
            self.lives -= 1
            if self.lives <= 0:
                terminated = True
            else:
                self._respawn()
        return terminated

    def _respawn(self):
        """捕まったが残機がある: パックマンとゴーストを開始位置に戻す (エサ・得点はそのまま)"""
        self.pacman = [16, 9]
        self.pacman_dir = ACTION_RIGHT
        for g in self.ghosts:
            g.reset()
        self.power_mode = False
        self.power_timer = 0
        self.pos_history = []
        self.dir_history = []
        self.prev_dot_dist = None
        self._respawned = True

    def _nearest_dot_dist(self):
        """パックマンから最近傍エサまでのBFS距離"""
        pos = tuple(self.pacman)
        _, path = find_nearest_dot(self.grid, pos)
        return len(path) - 1 if path else 999

    def _min_ghost_dist(self):
        """パックマンから最近傍ゴーストまでのマンハッタン距離"""
        return min(abs(g.pos[0]-self.pacman[0]) + abs(g.pos[1]-self.pacman[1]) for g in self.ghosts)

    def step(self, action):
        self.steps += 1
        self.mouth_tick += 1
        dr, dc = DIRS[action]
        nr, nc = self.pacman[0]+dr, self.pacman[1]+dc

        # 報酬は要素ごとに集計し、合計を通常の報酬として返す (HRA用に info にも要素別で入れる)
        rc = np.zeros(N_REWARD_COMPONENTS, dtype=np.float32)
        self._give(rc, "time")  # 時間ペナルティ

        # ── パックマン移動 ──
        prev_pacman = self.pacman[:]
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != CELL_WALL:
            self.pacman = [nr, nc]
            self.pacman_dir = action
        else:
            pass  # 壁衝突: 報酬変化なし(0)

        r, c = self.pacman

        # ── ループ検出①: 同一セル滞留 ──
        HISTORY_LEN = 8    # 直近8ステップを監視
        LOOP_THRESH = 3    # 同じセルに3回以上いたらループ
        self.pos_history.append((r, c))
        if len(self.pos_history) > HISTORY_LEN:
            self.pos_history.pop(0)
        if len(self.pos_history) == HISTORY_LEN:
            visit_count = self.pos_history.count((r, c))
            if visit_count >= LOOP_THRESH:
                self._give(rc, "loop", 1 + (visit_count - LOOP_THRESH) * 0.75)  # 回数が多いほど重く

        # ── ループ検出②: 往復パターン（A→B→A→B）を直接検出 ──
        if len(self.pos_history) >= 6:
            h = self.pos_history
            # 直近6ステップが 2セル間の往復かチェック
            if (h[-1] == h[-3] == h[-5] and h[-2] == h[-4] == h[-6]
                    and h[-1] != h[-2]):
                self._give(rc, "oscillation")  # 往復ペナルティ

        # ── 方向維持ボーナス（同じ向きに3連続で進んだら小ボーナス）──
        self.dir_history.append(action)
        if len(self.dir_history) > 4:
            self.dir_history.pop(0)
        if (len(self.dir_history) == 4
                and self.dir_history[-1] == self.dir_history[-2] == self.dir_history[-3]
                and self.pacman != prev_pacman):
            self._give(rc, "heading")

        # ── 未訪問セルへの探索ボーナス ──
        if (r, c) not in self.visited_cells:
            self.visited_cells.add((r, c))
            self._give(rc, "new_cell")  # 新しいセルを訪問したら小さなボーナス

        # ── エサ取得 ──
        if self.grid[r][c] == CELL_DOT:
            self.grid[r][c] = CELL_EMPTY
            self.dots_eaten += 1
            self.score += 10
            self.score_parts["dot"] += 10
            # 終盤ほどエサの価値を上げて、最後の数個を取り切る動機を与える
            self._give(rc, "dot")
            self._give(rc, "dot_progress", self.dots_eaten / self.total_dots)
            self._ate_since_dot_measure = True
        elif self.grid[r][c] == CELL_POWER:
            self.grid[r][c] = CELL_EMPTY
            self.dots_eaten += 1
            self.score += 50
            self.score_parts["power"] += 50
            self.power_eaten += 1
            self.power_steps.append(self.steps)
            self._give(rc, "power")
            self._ate_since_dot_measure = True
            self.power_mode = True
            self.power_timer = POWER_DURATION
            for g in self.ghosts:
                g.scared = True
                g.scared_timer = POWER_DURATION

        # ── エサへの接近ボーナス (報酬シェーピング、2ステップ毎に計算) ──
        if self.steps % 2 == 0:
            curr_dist = self._nearest_dot_dist()
            if self.prev_dot_dist is not None and curr_dist < self.prev_dot_dist:
                self._give(rc, "approach_dot")   # エサに近づいた
            elif (self.prev_dot_dist is not None and curr_dist > self.prev_dot_dist
                  and not (self.fix_leave_dot and self._ate_since_dot_measure)):
                self._give(rc, "leave_dot")   # エサから離れた (食べた直後に次のエサが遠いだけなら fix_leave_dot で付けない)
            self.prev_dot_dist = curr_dist
            self._ate_since_dot_measure = False

        # ── ゴースト接近ペナルティ ──
        # A captured ghost respawns dangerous even while other ghosts stay blue.
        # Measure through corridors so an enemy behind a wall is not "adjacent".
        if not self.no_ghosts:
            dangerous = {tuple(g.pos) for g in self.ghosts if not g.scared}
            scared = {tuple(g.pos) for g in self.ghosts if g.scared}
            danger_dist = self._bfs_distance(tuple(self.pacman), dangerous) if dangerous else 999
            scared_dist = self._bfs_distance(tuple(self.pacman), scared) if scared else 999
            if danger_dist == 1:
                self._give(rc, "ghost_adjacent")   # 隣接
            elif danger_dist == 2:
                self._give(rc, "ghost_near")   # 2マス以内
            if scared_dist <= 2:
                self._give(rc, "chase_blue")   # パワー中は近づく方が良い

        # ── パワーモード更新 ──
        if self.power_timer > 0:
            self.power_timer -= 1
            if self.power_timer == 0:
                self.power_mode = False

        # ── 衝突判定① パックマンがゴーストのマスへ踏み込んだ ──
        terminated = False
        collided = set()
        self._respawned = False
        for i, g in enumerate(self.ghosts):
            if g.pos == self.pacman:
                terminated = self._handle_ghost_collision(g, rc, terminated)
                collided.add(i)
                if self._respawned:
                    break

        # ── ゴースト移動 ──
        prev_ghost_pos = [g.pos[:] for g in self.ghosts]
        blinky_pos = tuple(self.ghosts[0].pos)  # InkyのターゲットにBlinkyの位置が必要
        for g in self.ghosts:
            g.move(self.grid, tuple(self.pacman), self.pacman_dir, blinky_pos, self.rng,
                   mode=self.ghost_mode)

        # ── 衝突判定② ゴーストがパックマンのマスへ来た / すり抜け ──
        for i, g in enumerate(self.ghosts):
            if i in collided or self._respawned:   # 開始位置に戻った直後は判定しない
                continue
            # ゴーストがパックマスマスへ移動
            direct = (g.pos == self.pacman)
            # すり抜け: パックマンとゴーストが互いの前のマスへ移動
            passthrough = (prev_ghost_pos[i] == self.pacman and list(g.pos) == prev_pacman)
            if direct or passthrough:
                terminated = self._handle_ghost_collision(g, rc, terminated)

        # ── クリア判定 ──
        # 他の報酬(±数点)と桁を揃える。20000のような巨大値はQ値の学習を不安定にする
        if self.dots_eaten >= self.total_dots:
            self._give(rc, "clear")
            self.cleared = True
            terminated = True
        reward = float(rc.sum())

        truncated = self.steps >= self.max_steps

        # ── 経路更新: 描画時のみ実行（訓練時はスキップして高速化）──
        if self.render_mode == "human" and self.steps % 5 == 0:
            self._update_planned_path()
            self._update_ghost_paths()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score,
                                                                "reward_components": rc,
                                                                "reward_items": self.reward_items,
                                                                "cleared": self.cleared,
                                                                "lives": self.lives,
                                                                "dots_eaten": self.dots_eaten,
                                                                "score_parts": dict(self.score_parts),
                                                                "ghosts_eaten": self.ghosts_eaten,
                                                                "power_eaten": self.power_eaten,
                                                                "power_steps": list(self.power_steps)}

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame
        cs = self.cell_size
        W = self.cols * cs
        H = self.rows * cs + 60  # 情報バー分

        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Pac-Man RL - カーナビ風経路表示")
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((W, H))
            else:
                self.screen = pygame.Surface((W, H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14, bold=True)
            self.font_big = pygame.font.SysFont("Arial", 18, bold=True)

        self.screen.fill((0, 0, 0))

        # ── グリッド描画 ──
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c*cs, r*cs
                cell = self.grid[r][c]
                if cell == CELL_WALL:
                    pygame.draw.rect(self.screen, (30, 80, 200), (x, y, cs, cs))
                    pygame.draw.rect(self.screen, (50, 100, 220), (x+1, y+1, cs-2, cs-2), 1)
                elif cell == CELL_DOT:
                    pygame.draw.circle(self.screen, (255, 220, 150), (x+cs//2, y+cs//2), 3)
                elif cell == CELL_POWER:
                    t = pygame.time.get_ticks() // 300 % 2
                    col = (255, 200, 50) if t == 0 else (255, 100, 0)
                    pygame.draw.circle(self.screen, col, (x+cs//2, y+cs//2), 7)

        # ── カーナビ経路表示 ──
        path = self.planned_path
        if len(path) > 1:
            # 経路を半透明の帯で描画
            surf = pygame.Surface((W, self.rows*cs), pygame.SRCALPHA)
            for i in range(len(path)-1):
                r0,c0 = path[i]
                r1,c1 = path[i+1]
                x0,y0 = c0*cs+cs//2, r0*cs+cs//2
                x1,y1 = c1*cs+cs//2, r1*cs+cs//2
                # 進捗に応じて色変化 (青→緑)
                t = i / max(len(path)-1, 1)
                color = (int(0+100*t), int(180-80*t), int(255-100*t), 160)
                pygame.draw.line(surf, color, (x0,y0), (x1,y1), 6)
            self.screen.blit(surf, (0,0))

            # ウェイポイント矢印
            for i in range(0, len(path)-1, 2):
                r0,c0 = path[i]
                r1,c1 = path[i+1]
                cx,cy = c0*cs+cs//2, r0*cs+cs//2
                dr,dc = r1-r0, c1-c0
                arrow_pts = self._arrow_points(cx, cy, dc*cs//2, dr*cs//2, 5)
                pygame.draw.polygon(self.screen, (0, 255, 180), arrow_pts)

        # 目標地点マーカー
        if self.target_pos:
            tr, tc = self.target_pos
            tx, ty = tc*cs+cs//2, tr*cs+cs//2
            pygame.draw.circle(self.screen, (255, 50, 50), (tx, ty), 8, 3)
            pygame.draw.line(self.screen, (255, 50, 50), (tx, ty-12), (tx, ty-4), 2)

        # ── ゴースト予測経路表示 ──
        ghost_surf = pygame.Surface((W, self.rows*cs), pygame.SRCALPHA)
        for i, (g, path) in enumerate(zip(self.ghosts, self.ghost_paths)):
            if len(path) < 2 or g.scared:
                continue
            base_color = g.color
            for j in range(len(path) - 1):
                r0, c0 = path[j]
                r1, c1 = path[j+1]
                x0, y0 = c0*cs+cs//2, r0*cs+cs//2
                x1, y1 = c1*cs+cs//2, r1*cs+cs//2
                alpha = max(30, 140 - j * 15)
                col = (*base_color, alpha)
                pygame.draw.line(ghost_surf, col, (x0, y0), (x1, y1), 3)
            # 経路先端に小三角
            if len(path) >= 2:
                r0, c0 = path[-2]; r1, c1 = path[-1]
                ex, ey = c1*cs+cs//2, r1*cs+cs//2
                arrow = self._arrow_points(ex, ey,
                    (c1-c0)*cs//3, (r1-r0)*cs//3, 4)
                pygame.draw.polygon(ghost_surf, (*base_color, 200), arrow)
        self.screen.blit(ghost_surf, (0, 0))

        # ── ゴースト描画 ──
        for g in self.ghosts:
            gr, gc = g.pos
            gx, gy = gc*cs+cs//2, gr*cs+cs//2
            color = (100, 100, 255) if g.scared else g.color
            pygame.draw.circle(self.screen, color, (gx, gy), cs//2-3)
            pygame.draw.rect(self.screen, color, (gc*cs+3, gr*cs+cs//2, cs-6, cs//2-3))
            # 目
            ew = 4 if not g.scared else 3
            ec = (255,255,255) if not g.scared else (255,255,100)
            pygame.draw.circle(self.screen, ec, (gx-5, gy-3), ew)
            pygame.draw.circle(self.screen, ec, (gx+5, gy-3), ew)
            if not g.scared:
                pygame.draw.circle(self.screen, (0,0,100), (gx-4, gy-3), 2)
                pygame.draw.circle(self.screen, (0,0,100), (gx+6, gy-3), 2)

        # ── パックマン描画 (滑らか・向き対応) ──
        pr, pc = self.pacman
        px, py = pc*cs+cs//2, pr*cs+cs//2
        self._draw_pacman(px, py, cs)

        # ── Q値 方向確率矢印 (カーナビ意思決定表示) ──
        if self.last_qvalues is not None:
            self._draw_qvalue_arrows(px, py, cs)

        # ── 情報バー (カーナビ風) ──
        info_y = self.rows * cs
        pygame.draw.rect(self.screen, (20, 20, 40), (0, info_y, W, 60))
        pygame.draw.line(self.screen, (0, 200, 255), (0, info_y), (W, info_y), 2)

        # スコア
        s = self.font_big.render(f"SCORE: {self.score}", True, (0, 255, 200))
        self.screen.blit(s, (10, info_y+8))

        # 経路情報
        path_len = len(self.planned_path)
        nav_text = f"経路: {path_len}歩  目標: {self.target_pos}" if self.target_pos else "経路探索中..."
        nav_col = (0, 220, 255) if path_len > 0 else (255, 150, 0)
        ns = self.font.render(nav_text, True, nav_col)
        self.screen.blit(ns, (10, info_y+32))

        # パワーモード表示
        if self.power_mode:
            pt = self.font.render(f"** POWER MODE! ({self.power_timer})", True, (255, 220, 0))
            self.screen.blit(pt, (W//2-60, info_y+8))

        # エサ残数
        remain = self.total_dots - self.dots_eaten
        ds = self.font.render(f"残エサ: {remain}/{self.total_dots}", True, (200, 200, 200))
        self.screen.blit(ds, (W-130, info_y+8))

        # ステップ
        ss = self.font.render(f"Step: {self.steps}", True, (150, 150, 150))
        self.screen.blit(ss, (W-130, info_y+30))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))

    def _draw_qvalue_arrows(self, px, py, cs):
        """Q値をsoftmax確率に変換して4方向の矢印で表示"""
        import math, pygame
        qv = self.last_qvalues.astype(np.float64)
        # softmax
        qv -= qv.max()
        exp_q = np.exp(qv)
        probs = exp_q / exp_q.sum()

        # 方向: UP=0,DOWN=1,LEFT=2,RIGHT=3
        # (dy, dx) オフセット, 矢印色
        DIR_INFO = [
            (-1,  0, (255,  80,  80)),   # UP    赤
            ( 1,  0, (255, 200,  50)),   # DOWN  黄
            ( 0, -1, (255, 165,   0)),   # LEFT  オレンジ
            ( 0,  1, ( 80, 220,  80)),   # RIGHT 緑
        ]
        max_len = cs * 1.6   # 最大矢印長

        for i, (dy, dx, color) in enumerate(DIR_INFO):
            p = float(probs[i])
            length = max_len * p
            if length < 3:
                continue

            ex = px + dx * length
            ey = py + dy * length

            # 壁方向なら薄く
            nr = self.pacman[0] + dy
            nc = self.pacman[1] + dx
            is_wall = not (0 <= nr < self.rows and 0 <= nc < self.cols
                           and self.grid[nr][nc] != CELL_WALL)
            alpha = 80 if is_wall else 220

            surf = pygame.Surface((self.cols*cs, self.rows*cs), pygame.SRCALPHA)
            # 矢印の軸
            pygame.draw.line(surf, (*color, alpha),
                             (int(px), int(py)), (int(ex), int(ey)),
                             max(2, int(4 * p + 1)))
            # 矢印の頭
            angle = math.atan2(dy, dx)
            head_size = max(4, int(8 * p + 3))
            tip = (ex, ey)
            left  = (ex - head_size*math.cos(angle-0.5),
                     ey - head_size*math.sin(angle-0.5))
            right = (ex - head_size*math.cos(angle+0.5),
                     ey - head_size*math.sin(angle+0.5))
            pygame.draw.polygon(surf, (*color, alpha), [tip, left, right])
            self.screen.blit(surf, (0, 0))

            # パーセント表示
            pct = int(p * 100)
            if pct >= 5:
                label = self.font.render(f"{pct}%", True, color)
                lx = int(px + dx * (length + 6)) - label.get_width()//2
                ly = int(py + dy * (length + 6)) - label.get_height()//2
                self.screen.blit(label, (lx, ly))

    def _draw_pacman(self, px, py, cs):
        """滑らか・向き対応のパックマンを描画する"""
        import math, pygame
        color_pac = (255, 255, 0) if not self.power_mode else (255, 150, 0)

        # 口の開閉アニメ: 0→25→0度を往復
        mouth_open = abs((self.mouth_tick % 16) - 8) * 3.2  # 0〜25.6度

        # 進行方向 → 口の向き (右=0°, 下=90°, 左=180°, 上=270°)
        dir_angle = {
            ACTION_RIGHT: 0.0,
            ACTION_DOWN:  90.0,
            ACTION_LEFT:  180.0,
            ACTION_UP:    270.0,
        }.get(self.pacman_dir, 0.0)

        # 4倍サイズで描いてスムーズスケールダウン → アンチエイリアス
        SCALE = 4
        radius = (cs // 2 - 2) * SCALE
        surf_size = radius * 2 + SCALE
        pac_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        cx = cy = surf_size // 2

        # ① 黄色の塗り潰し円
        pygame.draw.circle(pac_surf, color_pac, (cx, cy), radius)

        # ② 口の開口部: dir_angle ± mouth_open の扇形を黒で塗り潰す
        if mouth_open > 0.5:
            STEPS = 60
            gap = [(cx, cy)]
            a_start = math.radians(dir_angle - mouth_open)
            a_end   = math.radians(dir_angle + mouth_open)
            for s in range(STEPS + 1):
                a = a_start + (a_end - a_start) * s / STEPS
                gap.append((cx + radius * math.cos(a),
                             cy + radius * math.sin(a)))
            pygame.draw.polygon(pac_surf, (0, 0, 0, 255), gap)

        # ③ スムーズスケールダウン
        disp_r = cs // 2 - 2
        disp_d = disp_r * 2
        small = pygame.transform.smoothscale(pac_surf, (disp_d, disp_d))
        self.screen.blit(small, (px - disp_r, py - disp_r))

    def _arrow_points(self, x, y, dx, dy, size):
        import math
        angle = math.atan2(dy, dx)
        pts = [
            (x + dx, y + dy),
            (x + dx - size*math.cos(angle-0.4), y + dy - size*math.sin(angle-0.4)),
            (x + dx - size*math.cos(angle+0.4), y + dy - size*math.sin(angle+0.4)),
        ]
        return pts

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
