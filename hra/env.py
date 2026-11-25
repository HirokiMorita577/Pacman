"""
Atari 2600 Ms. Pac-Man (ALE) 環境とオブジェクト抽出 (FR-01〜FR-07, FR-10, RM-10〜RM-12)

画面 → 160x160 に切り出し → オブジェクトの位置を 4ピクセル精度 (40x40 マス) に変換する。
位置の取り方は設定 env.position_source で切り替える (RM-12):
  pixels : 画面の色から抽出 (論文と同じ方針。具体的な方法は論文に記載なし = U-01)
  ram    : ALE の RAM から読む (RM-10。AtariARI の Ms. Pac-Man 注釈のアドレスを使用)

画面抽出の方法 (実機画面の観察にもとづく。決定記録 DECISIONS.md を参照):
  - Ms. Pac-Man と 4体のゴーストはそれぞれ固有の色を持つ
  - 赤以外のゴーストは3フレーム周期で1体ずつ交互に描画されるので、直近3フレームを合成する
  - エサは迷路の壁と同じ色の小さな塊。面の開始時に見えた塊の位置を「エサの候補地」として登録し、
    以後はその画素だけを見て残っているかを判定する (RM-11)
  - 青ゴースト(点滅時は白)は色で判定し、塊ごとに1体とみなす
  - 既知の色以外の塊をフルーツとみなす
"""
from collections import deque

import numpy as np
import gymnasium as gym
from scipy import ndimage


def _rgb(r, g, b):
    return (r << 16) | (g << 8) | b


C_PACMAN = _rgb(210, 164, 74)
C_GHOSTS = [_rgb(200, 72, 72), _rgb(198, 89, 179), _rgb(84, 184, 153), _rgb(180, 122, 48)]  # 赤,ピンク,水色,橙
C_BLUE   = [_rgb(66, 114, 194), _rgb(214, 214, 214)]   # 青ゴースト (通常 / 点滅時)

# AtariARI (Anand et al., 2019) の Ms. Pac-Man RAM 注釈
RAM_PLAYER = (10, 16)                                # (x, y)
RAM_GHOSTS = [(9, 15), (8, 14), (7, 13), (6, 12)]    # blinky(赤), pinky, inky(水色), sue(橙)
# RAM座標 → 画面座標 (切り出し前) の変換。実測で |誤差| ≤ 2px (95%点)
RAM_TO_PIX = (-8, 6)                                 # pix_x = ram_x - 8, pix_y = ram_y + 6

N_ACTIONS = 9            # [付録A.2] nine actions
NORTH, EAST, SOUTH, WEST = range(4)   # FR-05 向き
GRID = 40
SCREEN_H = 210
PELLET_ROW_CENTER = 7.5  # エサの行の中心 (切り出し前の画面で 7.5, 19.5, 31.5, ... 行目。実測)
PELLET_ROW_STEP = 12
SITE_SCAN_FRAMES = 800   # 面の開始からこのフレーム数の間、候補地を追加し続ける
                         # (パワーエサは開始前の待ち時間 約400フレームは描かれず、その後も点滅するため)


def _pack(frame):
    f = frame.astype(np.int32)
    return (f[..., 0] << 16) | (f[..., 1] << 8) | f[..., 2]


def _center(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    # 外接矩形の中心 (口のアニメーションで重心が揺れにくい)
    return (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0


def _is_small(sl):
    return sl[0].stop - sl[0].start <= 4 and sl[1].stop - sl[1].start <= 4


class Objects:
    """1ステップ分の抽出結果。位置はすべて 40x40 のマス座標 (row, col)"""
    __slots__ = ("pacman", "direction", "ghosts", "blue_ghosts", "fruit",
                 "pellets", "power_pellets", "maze")

    def __init__(self):
        self.pacman = None
        self.direction = EAST
        self.ghosts = [None] * 4        # 通常ゴースト (1体ずつ別)
        self.blue_ghosts = [None] * 4   # 青ゴースト (通常ゴーストとは別オブジェクト)
        self.fruit = None
        self.pellets = []
        self.power_pellets = []
        self.maze = 0

    def to_channels(self):
        """FR-03〜05: 11枚の 40x40 2値チャンネルと、向きの one-hot(4)"""
        ch = np.zeros((11, GRID, GRID), dtype=np.uint8)

        def put(i, p):
            if p is not None:
                ch[i, p[0], p[1]] = 1
        put(0, self.pacman)
        for i, g in enumerate(self.ghosts):
            put(1 + i, g)
        for i, g in enumerate(self.blue_ghosts):
            put(5 + i, g)
        put(9, self.fruit)
        for p in self.pellets + self.power_pellets:
            put(10, p)
        direction = np.zeros(4, dtype=np.uint8)
        direction[self.direction] = 1
        return ch, direction


class MsPacmanEnv:
    """
    ALE の Ms. Pac-Man を frameskip フレームずつ進める。
    step() → (objects, raw_reward, shaped_reward, done, info)
      shaped_reward = 得点 + ゴースト接触時の -1000 (FR-10)
      info: frames, lives, life_lost, level_cleared, level, in_transition
    """

    def __init__(self, cfg, seed=0):
        import ale_py
        gym.register_envs(ale_py)
        e = cfg["env"]
        self.frameskip = e["frameskip"]
        self.crop_top = e["crop_top"]
        self.cell = e["cell"]
        self.ghost_frames = e["ghost_frames"]
        self.source = e.get("position_source", "pixels")
        self.step_mode = e.get("step_mode", "frames")
        self.cell_substep = e.get("cell_substep", 2)
        self.cell_max_frames = e.get("cell_max_frames", 16)
        self.objs = None
        self._recent = deque(maxlen=self.ghost_frames)
        self.ghost_penalty = cfg["reward"]["ghost_contact"]
        self._gym = gym.make(e["game"], frameskip=1,
                             repeat_action_probability=e["repeat_action_probability"],
                             full_action_space=False)
        self.ale = self._gym.unwrapped.ale
        self.action_set = self._gym.unwrapped._action_set
        assert len(self.action_set) == N_ACTIONS
        self.seed = seed
        self.maze_colors = []   # 壁の色 → 迷路ID (FR-06)
        self.pellet_sites = {}  # 迷路ID → {マス: (y, x, is_power)}

    # ── 画面・座標 ──
    def _screen(self):
        return _pack(self.ale.getScreenRGB()[self.crop_top:self.crop_top + 160])

    def _cell(self, yx):
        r = int(yx[0] // self.cell)
        c = int(yx[1] // self.cell)
        if not (0 <= r < GRID and 0 <= c < GRID):
            return None
        return (r, c)

    def _ram_cell(self, ram, xy_addr):
        x = int(ram[xy_addr[0]]) + RAM_TO_PIX[0]
        y = int(ram[xy_addr[1]]) + RAM_TO_PIX[1] - self.crop_top
        return self._cell((y, x))

    @staticmethod
    def _colors(s):
        vals, counts = np.unique(s[::2, ::2], return_counts=True)
        order = np.argsort(-counts)
        bg = int(vals[order[0]])
        wall = int(vals[order[1]]) if len(order) > 1 else None
        return bg, wall   # 最多色=背景、2番目=壁 (エサも同じ色)

    def _on_pellet_row(self, y_cropped):
        d = (y_cropped + self.crop_top - PELLET_ROW_CENTER) % PELLET_ROW_STEP
        return min(d, PELLET_ROW_STEP - d) <= 3

    def _is_pellet(self, sl):
        """エサ(縦2×横4)/パワーエサ(縦7×横4、点滅中は欠けることがある)の形で、中心がエサの行にある塊だけを
        候補地にする。ゴーストが壁に重なって分断された壁の切れ端を、エサと取り違えないため"""
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        return w == 4 and 2 <= h <= 7 and self._on_pellet_row((sl[0].start + sl[0].stop - 1) // 2)

    def _add_sites(self, screens):
        """画面に見えているエサの塊のうち、未登録のものを候補地に追加して返す。
        パワーエサは点滅するので、面の開始から SITE_SCAN_FRAMES フレームの間くり返し呼ぶ"""
        sites = self.pellet_sites[self.maze]
        mask = np.zeros(screens[-1].shape, dtype=bool)
        for s in screens:
            m = s == self.wall
            mask |= m
            if self._wall_always is None:
                self._wall_always, self._wall_ever = m.copy(), m.copy()
            else:
                self._wall_always &= m
                self._wall_ever |= m
        # 点滅している画素 = 一度は壁の色になったが、ずっとではない画素。
        # パワーエサは壁に接していることがある (右端) ので、壁と切り離すためにこちらからも探す
        blinking = self._wall_ever & ~self._wall_always
        slices = []
        for mk in (mask, blinking):
            lab, _ = ndimage.label(mk)
            slices += [sl for sl in ndimage.find_objects(lab) if sl is not None]
        new = set()
        for sl in slices:
            if not self._is_pellet(sl):
                continue
            cell = self._cell(((sl[0].start + sl[0].stop - 1) / 2, (sl[1].start + sl[1].stop - 1) / 2))
            if cell is not None and cell not in sites:
                h = sl[0].stop - sl[0].start
                # 判定用の画素 = 塊の中心 (縦2のエサは上の行)
                sites[cell] = ((sl[0].start + sl[0].stop - 1) // 2, (sl[1].start + sl[1].stop) // 2, h >= 3)
                new.add(cell)
        if new or not hasattr(self, "_site_cells"):
            self._site_cells = list(sites.keys())
            self._site_y = np.array([sites[c][0] for c in self._site_cells], dtype=np.int64)
            self._site_x = np.array([sites[c][1] for c in self._site_cells], dtype=np.int64)
        return new

    def _register_maze(self, screens):
        """面の開始時: 壁の色で迷路を判定し (FR-06)、エサの候補地を登録する (FR-07, RM-11)"""
        self.background, wall = self._colors(screens[-1])
        if wall not in self.maze_colors:
            self.maze_colors.append(wall)
        self.maze, self.wall = self.maze_colors.index(wall), wall
        self.pellet_sites.setdefault(self.maze, {})
        if hasattr(self, "_site_cells"):
            del self._site_cells
        self._wall_always = self._wall_ever = None
        self._add_sites(screens)
        self.level_frames = 0

    def load_sites(self, maze_colors, pellet_sites):
        """保存したエサの候補地を読み込む。形の条件を満たさない古い候補地 (壁の切れ端) は捨てる"""
        self.maze_colors = list(maze_colors)
        self.pellet_sites = {m: {c: v for c, v in sites.items() if self._on_pellet_row(v[0])}
                             for m, sites in pellet_sites.items()}
        dropped = sum(len(s) for s in pellet_sites.values()) - sum(len(s) for s in self.pellet_sites.values())
        if dropped:
            print(f"[修正] エサの候補地から壁の切れ端 {dropped} 個を除外しました")

    def _visible_pellets(self, screens):
        if not self._site_cells:
            return set()
        seen = np.zeros(len(self._site_cells), dtype=bool)
        for s in screens:
            seen |= s[self._site_y, self._site_x] == self.wall
        return {c for c, v in zip(self._site_cells, seen) if v}

    # ── オブジェクト抽出 ──
    def _extract_pixels(self, screens):
        s = screens[-1]
        pac = _center(s == C_PACMAN)
        pac = self._cell(pac) if pac is not None else None
        ghosts = [None] * 4
        for i, col in enumerate(C_GHOSTS):
            for sc in reversed(screens[-self.ghost_frames:]):
                c = _center(sc == col)
                if c is not None:
                    ghosts[i] = self._cell(c)
                    break
        return pac, ghosts

    def _extract_ram(self):
        ram = self.ale.getRAM()
        pac = self._ram_cell(ram, RAM_PLAYER)
        ghosts = [self._ram_cell(ram, a) for a in RAM_GHOSTS]
        return pac, ghosts

    def _blue_ghosts(self, screens):
        blue = np.zeros(screens[-1].shape, dtype=bool)
        for sc in screens[-self.ghost_frames:]:
            for col in C_BLUE:
                blue |= sc == col
        if not blue.any():
            return []
        lab, n = ndimage.label(blue)
        cs = ndimage.center_of_mass(blue, lab, range(1, n + 1))
        cells = [self._cell(p) for p in sorted(cs, key=lambda p: (p[1], p[0]))]
        return [c for c in cells if c is not None][:4]

    def _fruit(self, s):
        known = [self.background, self.wall, C_PACMAN, 0, *C_GHOSTS, *C_BLUE]
        other = ~np.isin(s, known)
        if other.sum() < 6:
            return None
        return self._cell(_center(other))

    def _make_objects(self, screens):
        o = Objects()
        o.maze = self.maze
        if self.source == "ram":
            pac, ghosts = self._extract_ram()
        else:
            pac, ghosts = self._extract_pixels(screens)
        blues = self._blue_ghosts(screens)
        if self.source == "ram" and blues:
            # RAM には青状態がないので、青い塊に近いゴーストを青ゴーストとして扱う
            for i, g in enumerate(ghosts):
                if g is not None and any(max(abs(g[0] - b[0]), abs(g[1] - b[1])) <= 2 for b in blues):
                    ghosts[i] = None
        o.pacman = pac
        o.ghosts = ghosts
        for i, b in enumerate(blues):
            o.blue_ghosts[i] = b
        o.fruit = self._fruit(screens[-1])

        # 向き (FR-05): 位置の変化から求める。止まっているときは直前の向きを保つ
        if pac is not None and self.prev_pac is not None and pac != self.prev_pac:
            dr, dc = pac[0] - self.prev_pac[0], pac[1] - self.prev_pac[1]
            if abs(dc) > GRID // 2:     # トンネルの左右ワープ
                dc = -dc
            if abs(dr) >= abs(dc):
                self.direction = SOUTH if dr > 0 else NORTH
            else:
                self.direction = EAST if dc > 0 else WEST
        if pac is not None:
            self.prev_pac = pac
        o.direction = self.direction

        sites = self.pellet_sites[self.maze]
        o.pellets = [c for c in self.remaining if not sites[c][2]]
        o.power_pellets = [c for c in self.remaining if sites[c][2]]
        return o

    # ── エピソード ──
    def reset(self, noops=0):
        self._gym.reset(seed=self.seed)
        self.seed += 1
        for _ in range(noops):
            self.ale.act(self.action_set[0])
        s = self._screen()
        self._recent.clear()
        self._recent.append(s)
        self._register_maze([s])
        self.level = 1
        self.lives = self.ale.lives()
        self.remaining = self._visible_pellets([s])
        self.frames = 0
        self.prev_pac = None
        self.direction = EAST
        self.in_transition = False
        self.objs = self._make_objects([s])
        return self.objs

    def step(self, action):
        """1ステップ進める。
        step_mode=frames : frameskip フレーム固定 (HRAの厳密なフレーム設定は未確定)
        step_mode=cell   : Ms. Pac-Man が次のマスに入るまで同じ行動を続ける (最大 cell_max_frames)。
                           固定フレームだと同じ(状態,行動)から毎回同じ次状態になるのは2割程度だが、
                           マス単位では過去の標本で85〜95%。全状態の決定性を保証する値ではない"""
        if self.step_mode != "cell":
            return self._substep(action, self.frameskip)
        start = self.objs.pacman if self.objs is not None else None
        raw = shaped = 0.0
        life_lost = level_cleared = False
        frames = 0
        while True:
            objs, r, sh, done, info = self._substep(action, self.cell_substep)
            raw += r
            shaped += sh
            frames += self.cell_substep
            life_lost |= info["life_lost"]
            level_cleared |= info["level_cleared"]
            moved = start is not None and objs.pacman is not None and objs.pacman != start
            if (done or life_lost or level_cleared or info["in_transition"] or moved
                    or start is None or frames >= self.cell_max_frames):
                break
        info.update(life_lost=life_lost, level_cleared=level_cleared)
        return objs, raw, shaped, done, info

    def _substep(self, action, nframes):
        a = self.action_set[action]
        raw = 0.0
        for k in range(nframes):
            raw += self.ale.act(a)
            self.frames += 1
            self.level_frames += 1
            if k >= nframes - self.ghost_frames:
                self._recent.append(self._screen())
            if self.ale.game_over():
                break
        if not self._recent:
            self._recent.append(self._screen())
        screens = list(self._recent)   # 直近 ghost_frames 枚 (ステップをまたいでも保持)
        lives = self.ale.lives()
        life_lost = lives < self.lives
        self.lives = lives
        done = self.ale.game_over()
        s = screens[-1]

        level_cleared = False
        if not self.in_transition and self.level_frames < SITE_SCAN_FRAMES:
            self.remaining |= self._add_sites(screens)
        if not self.in_transition:
            # 見えなくなったエサは、Ms. Pac-Man がそのマスの近くにいたときだけ「食べた」とみなす
            # (ゴーストや点滅で一時的に隠れたエサを消さないため)
            visible = self._visible_pellets(screens)
            pac = self._extract_ram()[0] if self.source == "ram" else self._extract_pixels(screens)[0]
            if pac is not None:
                self.remaining = {c for c in self.remaining
                                  if c in visible or max(abs(c[0] - pac[0]), abs(c[1] - pac[1])) > 1}
            if not self.remaining:
                level_cleared = True
                self.in_transition = True
                self.level += 1
        else:
            # 面クリア後: 壁の点滅が終わり、新しい迷路にエサが並んだら次の面を開始
            _, wall = self._colors(s)
            if wall is not None:
                lab, _ = ndimage.label(s == wall)
                n_small = sum(1 for sl in ndimage.find_objects(lab) if sl is not None and _is_small(sl))
                if n_small > 50:
                    self._register_maze(screens)
                    self.remaining = self._visible_pellets(screens)
                    self.in_transition = False
                    self.prev_pac = None
        if life_lost:
            self.prev_pac = None

        self.objs = self._make_objects(screens)
        shaped = raw + (self.ghost_penalty if life_lost else 0.0)
        info = {"frames": self.frames, "lives": lives, "life_lost": life_lost,
                "level_cleared": level_cleared, "level": self.level,
                "in_transition": self.in_transition}
        return self.objs, raw, shaped, done, info

    def compare_sources(self, screens=None):
        """RM-12: 画面抽出と RAM 読み取りの位置の差 (マス) を返す"""
        screens = screens or [self._screen()]
        pix_pac, pix_g = self._extract_pixels(screens)
        ram_pac, ram_g = self._extract_ram()
        return (pix_pac, ram_pac), list(zip(pix_g, ram_g))

    def close(self):
        self._gym.close()
