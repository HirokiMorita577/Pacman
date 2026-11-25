"""
アーケード版風のプレイ動画 (見た目だけ。学習・評価には使わない)

Atari 2600 版の画面から取り出したオブジェクトの位置をもとに、アーケード版 Pac-Man 風に描き直す:
黒背景・青い縁取りの迷路・点のエサ・点滅するパワーエサ・口の開閉するパックマン・目のあるゴースト。

  python -m hra.arcade --checkpoint runs/<設定名>/checkpoint.pkl --out play.mp4 --tries 8
    tries 回プレイして一番点数の高い回を録画する (実時間の速さ)
"""
import argparse
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .env import (C_BLUE, C_GHOSTS, C_PACMAN, EAST, NORTH, RAM_GHOSTS, RAM_TO_PIX, SOUTH, WEST,
                  MsPacmanEnv, _center, _pack)

S = 4                    # 拡大率 (Atari の 1 画素 → S x S)
MAZE_ROWS = 172          # 迷路が描かれている行 (切り出し前の画面)
HEADER = 64
FOOTER = 44
BG = (0, 0, 0)
WALL = (33, 33, 222)
DOT = (255, 184, 151)
PAC = (255, 255, 0)
GHOST_RGB = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]   # 赤, ピンク, 水色, 橙
BLUE_GHOST = (33, 33, 255)
FLASH_GHOST = (222, 222, 255)
WHITE = (255, 255, 255)
MOUTH = {EAST: 0, SOUTH: 90, WEST: 180, NORTH: 270}


def _font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class ArcadeRenderer:
    def __init__(self, env, actor_mode='arcade'):
        self.env = env
        if actor_mode not in ('arcade', 'source'):
            raise ValueError('actor_mode must be arcade or source')
        self.actor_mode = actor_mode
        self.font = _font(22)
        self.small = _font(16)
        self.seen_color = [-99] * 4      # そのゴーストを通常色で最後に見たフレーム
        self.seen_blue = [-99] * 4       # 青/点滅で最後に見たフレーム
        self.blue_kind = [BLUE_GHOST] * 4
        self.tick = 0
        self._wall_key = None
        self._wall_img = None

    # ── 迷路 (壁の色の大きな塊 = 壁。小さな塊はエサなので除く) ──
    def _walls(self, s):
        key = (self.env.maze, self.env.wall)
        mask = s[:MAZE_ROWS] == self.env.wall
        lab, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        big = np.isin(lab, 1 + np.flatnonzero(sizes > 16))
        if self._wall_key == key and self._wall_img is not None and big.sum() > 0.8 * self._wall_n:
            return self._wall_img          # 面の間は壁は変わらないので使い回す (点滅中は描き直さない)
        up = np.repeat(np.repeat(big, S, axis=0), S, axis=1)
        inner = ndimage.binary_erosion(up, iterations=2)
        outline = up & ~ndimage.binary_erosion(up)             # 外側の線
        outline2 = inner & ~ndimage.binary_erosion(inner)      # 内側の線 (二重線)
        img = np.zeros(up.shape + (3,), dtype=np.uint8)
        img[outline | outline2] = WALL
        self._wall_key, self._wall_img, self._wall_n = key, img, big.sum()
        return img

    def _ghost(self, d, cx, cy, color, eyes=True, pupil_dir=None):
        r = 3.6 * S
        d.pieslice([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=color)
        d.rectangle([cx - r, cy, cx + r, cy + r * 0.8], fill=color)
        w = 2 * r / 3                                                  # 裾のギザギザ
        for k in range(3):
            x0 = cx - r + k * w
            d.polygon([(x0, cy + r * 0.8), (x0 + w / 2, cy + r * 1.05), (x0 + w, cy + r * 0.8)], fill=color)
        if eyes:
            dx, dy = {EAST: (1, 0), WEST: (-1, 0), NORTH: (0, -1), SOUTH: (0, 1)}.get(pupil_dir, (0, 0))
            for ex in (cx - r * 0.4, cx + r * 0.4):
                d.ellipse([ex - r * 0.3, cy - r * 0.45, ex + r * 0.3, cy + r * 0.15], fill=WHITE)
                px, py = ex + dx * r * 0.12, cy - r * 0.15 + dy * r * 0.12
                d.ellipse([px - r * 0.15, py - r * 0.15, px + r * 0.15, py + r * 0.15], fill=(33, 33, 255))
        else:                                                           # 青ゴーストの顔
            for ex in (cx - r * 0.35, cx + r * 0.35):
                d.rectangle([ex - r * 0.1, cy - r * 0.3, ex + r * 0.1, cy - r * 0.1], fill=DOT)
            pts = [(cx - r * 0.6 + i * r * 0.2, cy + r * (0.35 if i % 2 else 0.2)) for i in range(7)]
            d.line(pts, fill=DOT, width=max(1, S // 2))

    def _eyes(self, d, cx, cy, direction):
        r = 3.6 * S
        dx, dy = {EAST: (1, 0), WEST: (-1, 0), NORTH: (0, -1), SOUTH: (0, 1)}.get(direction, (0, 0))
        for ex in (cx - r * 0.4, cx + r * 0.4):
            d.ellipse([ex - r * 0.3, cy - r * 0.45, ex + r * 0.3, cy + r * 0.15], fill=WHITE)
            px, py = ex + dx * r * 0.12, cy - r * 0.15 + dy * r * 0.12
            d.ellipse([px - r * 0.15, py - r * 0.15, px + r * 0.15, py + r * 0.15], fill=(33, 33, 255))

    def frame(self, rgb, score, high, lives):
        env = self.env
        self.tick += 1
        s = _pack(rgb)
        W, H = 160 * S, MAZE_ROWS * S
        canvas = Image.new("RGB", (W, HEADER + H + FOOTER), BG)
        canvas.paste(Image.fromarray(self._walls(s)), (0, HEADER))
        d = ImageDraw.Draw(canvas)
        oy = HEADER
        top = env.crop_top

        # エサ・パワーエサ (追跡中の「残っているエサ」)
        sites = env.pellet_sites.get(env.maze, {})
        for c in env.remaining:
            y, x, power = sites[c]
            cy, cx = oy + (y + top + 0.5) * S, (x - 0.5) * S
            if power:
                if (self.tick // 12) % 2 == 0:
                    rr = 2.2 * S
                    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=DOT)
            else:
                rr = 0.55 * S
                d.rectangle([cx - rr, cy - rr, cx + rr, cy + rr], fill=DOT)

        actor_background = canvas.copy() if self.actor_mode == 'source' else None

        # フルーツ (既知の色以外の塊) → さくらんぼ
        known = [env.background, env.wall, C_PACMAN, 0, *C_GHOSTS, *C_BLUE]
        other = ~np.isin(s[:MAZE_ROWS], known)
        if other.sum() >= 6:
            fy, fx = _center(other)
            cx, cy = fx * S, oy + fy * S
            rr = 1.6 * S
            for ox, oyy in ((-1.3, 1.0), (1.3, 1.6)):
                d.ellipse([cx + ox * S - rr, cy + oyy * S - rr, cx + ox * S + rr, cy + oyy * S + rr], fill=(222, 0, 0))
            d.line([(cx - 1.3 * S, cy), (cx + 2 * S, cy - 3 * S), (cx + 1.3 * S, cy + 0.5 * S)], fill=(222, 151, 81), width=S // 2)

        # ゴースト: 位置は RAM から (画面では3フレームに1回しか描かれないため、毎フレーム描くには RAM が確実)。
        # 画面の色は「通常 / 青 / 点滅 / 食べられて目だけ」の判定にだけ使う
        ram = env.ale.getRAM()
        pos = [(int(ram[ay]) + RAM_TO_PIX[1], int(ram[ax]) + RAM_TO_PIX[0]) for ax, ay in RAM_GHOSTS]
        for i, col in enumerate(C_GHOSTS):
            c = _center(s[:MAZE_ROWS] == col)
            if c is not None and abs(c[0] - pos[i][0]) + abs(c[1] - pos[i][1]) < 12:
                self.seen_color[i] = self.tick
        for color_code, kind in ((C_BLUE[0], BLUE_GHOST), (C_BLUE[1], FLASH_GHOST)):
            mask = s[:MAZE_ROWS] == color_code
            if mask.any():
                lab, n = ndimage.label(mask)
                for by, bx in ndimage.center_of_mass(mask, lab, range(1, n + 1)):
                    dist = [abs(by - py) + abs(bx - px) for py, px in pos]
                    i = int(np.argmin(dist))
                    if dist[i] < 12:
                        self.seen_blue[i], self.blue_kind[i] = self.tick, kind
        any_ghost_visible = max(self.seen_color + self.seen_blue) >= self.tick - 6
        for i in range(4 if any_ghost_visible else 0):     # 死亡演出・面の切り替え中は描かない
            gy, gx = pos[i]
            if not (0 <= gy < MAZE_ROWS and 0 <= gx < 160):
                continue
            blue_recent = self.tick - self.seen_blue[i] <= 4
            color_recent = self.tick - self.seen_color[i] <= 6
            if blue_recent and self.seen_blue[i] >= self.seen_color[i]:
                self._ghost(d, gx * S, oy + gy * S, self.blue_kind[i], eyes=False)
            elif color_recent:
                self._ghost(d, gx * S, oy + gy * S, GHOST_RGB[i], pupil_dir=env.direction)
            elif self.tick > 12:
                self._eyes(d, gx * S, oy + gy * S, env.direction)     # 食べられて巣に戻る途中

        # パックマン (進行方向に口を開閉)
        c = _center(s[:MAZE_ROWS] == C_PACMAN)
        if c is not None:
            py, px = c
            cx, cy, r = px * S, oy + py * S, 3.8 * S
            mouth = abs((self.tick % 8) - 4) * 11                   # 0〜44 度
            base = MOUTH.get(env.direction, 0)
            d.pieslice([cx - r, cy - r, cx + r, cy + r], base + mouth, base + 360 - mouth, fill=PAC)

        # Diagnostic mode keeps exact source sprite pixels, including death animation.
        # Source multiplexing is intentional here: no invented positions or hitboxes.
        if actor_background is not None:
            canvas = actor_background
            pixels = np.asarray(canvas).copy()
            actor_mask = ~np.isin(s[:MAZE_ROWS], [env.background, env.wall, 0])
            up_mask = np.repeat(np.repeat(actor_mask, S, axis=0), S, axis=1)
            up_rgb = np.repeat(np.repeat(rgb[:MAZE_ROWS], S, axis=0), S, axis=1)
            area = pixels[HEADER:HEADER + H]
            area[up_mask] = up_rgb[up_mask]
            canvas = Image.fromarray(pixels)
            d = ImageDraw.Draw(canvas)

        # 見出しと残機
        d.text((W * 0.12, 6), "1UP", fill=WHITE, font=self.font)
        d.text((W * 0.12, 32), f"{int(score)}", fill=WHITE, font=self.font)
        d.text((W * 0.45, 6), "HIGH SCORE", fill=WHITE, font=self.font)
        d.text((W * 0.52, 32), f"{int(high)}", fill=WHITE, font=self.font)
        for k in range(max(0, lives - 1)):
            cx, cy, r = 24 + k * 34, HEADER + H + FOOTER / 2, 12
            d.pieslice([cx - r, cy - r, cx + r, cy + r], 180 + 35, 180 - 35 + 360, fill=PAC)
        d.text((W - 150, HEADER + H + 10), f"LEVEL {env.level}", fill=WHITE, font=self.small)
        return np.asarray(canvas)


class _Recorder:
    """ALE を包み、2 フレームごとにアーケード風の画面を動画に書き出す"""

    def __init__(self, env, writer, high):
        self._ale, self._env, self._w = env.ale, env, writer
        self.renderer = ArcadeRenderer(env)
        self.score, self.high, self._n = 0.0, high, 0

    def act(self, a):
        r = self._ale.act(a)
        self.score += r
        self.high = max(self.high, self.score)
        self._n += 1
        if self._n % 2 == 0 and hasattr(self._env, "remaining"):
            self._w.append_data(self.renderer.frame(self._ale.getScreenRGB(), self.score, self.high,
                                                    self._ale.lives()))
        return r

    def __getattr__(self, k):
        return getattr(self._ale, k)


def record(checkpoint, out, tries=8, config=None, fps=30):
    import imageio.v2 as imageio
    from .agent import HRAAgent
    from .run import load_config, play_episode
    cfg = load_config(config)
    with open(checkpoint, "rb") as f:
        ck = pickle.load(f)

    def run(i, writer=None, high=0):
        env = MsPacmanEnv(cfg, seed=i)
        env.load_sites(ck["maze_colors"], ck["pellet_sites"])
        ag = HRAAgent(cfg, seed=i)
        ag.load_state_dict(ck["agent"])
        ag.rng = np.random.default_rng(1000 + i)                 # 回ごとに違うプレイにする
        if writer is not None:
            env.ale = _Recorder(env, writer, high)
        return play_episode(env, ag, learn=False, noops=(7 * i) % 31)

    results = []
    for i in range(tries):
        r = run(i)
        results.append((r["score"], i))
        print(f"try {i}: score {r['score']:.0f}  level {r['level_reached']}", flush=True)
    best_score, best_i = max(results)
    writer = imageio.get_writer(out, fps=fps, macro_block_size=1, quality=7)
    r = run(best_i, writer, high=best_score)
    writer.close()
    print(f"保存: {out}  (score {r['score']:.0f}, level {r['level_reached']}, 学習 {ck['episode']} エピソード)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="arcade_play.mp4")
    p.add_argument("--tries", type=int, default=8)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    record(a.checkpoint, a.out, a.tries, a.config)


if __name__ == "__main__":
    main()
