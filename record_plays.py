"""
報酬設計の実験で学習したモデルのプレイ動画を作る (発表用)。
  python record_plays.py --out /content/drive/MyDrive/DQN_reward_study            # 実験5 の3条件
  python record_plays.py --out <dir> --exps exp5 exp6 exp7
出力 (<out>/videos/):
  compare_<exp>_board<k>.mp4  同じ盤面・同じ学習 seed で各条件を横に並べた動画 (先に終わった条件は最後の画面で止める)
  best_<exp>_<条件>.mp4       条件ごとに、3つの学習 seed × 評価の盤面20個 の中で一番点数の高かったプレイ
  plays_<exp>.csv             録画に使った各プレイの得点・終わり方・ステップ数 (実験ごとに別ファイル)
評価と同じく貪欲方策 (ランダムな行動なし)。盤面の乱数は reward_study.EVAL_SEED0 + k。
"""
import argparse
import csv
import glob
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # 画面のない環境 (Colab) で pygame を使う

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pacman_env import PacManEnv
from reward_study import EVAL_SEED0, FIXED


def _font(size):
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "C:/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/msgothic.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT, FONT_S = _font(22), _font(16)


def load(run_dir):
    from dqn_agent import load_agent
    with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
        r = json.load(f)
    env = PacManEnv(render_mode="rgb_array", ghost_mode=FIXED["ghost_mode"], num_ghosts=r.get("num_ghosts", FIXED["num_ghosts"]),
                    rewards=r["rewards"], lives=r.get("lives", 1),
                    include_lives=r.get("include_lives", False),
                    observation_version=r.get("observation_version", "legacy"),
                    max_steps=r.get("manifest", {}).get("fixed", FIXED)["max_steps"])
    agent = load_agent(os.path.join(run_dir, "model.pt"), env.observation_space.shape[0], 4)
    return r, env, agent


def play(env, agent, board, frames=True):
    """1回プレイして (画像のリスト, 得点, 終わり方, ステップ数) を返す"""
    obs, _ = env.reset(seed=EVAL_SEED0 + board)
    imgs = []
    while True:
        q = agent.get_qvalues(obs)
        env.set_qvalues(q)                      # 盤面に行動ごとのQ値の矢印を描く
        if frames:
            imgs.append(env.render())
        obs, _, te, tr, info = env.step(agent.select_action(obs, training=False))
        if te or tr:
            if frames:
                imgs.append(env.render())
            end = "クリア" if info["cleared"] else ("上限で打ち切り" if tr and not te else "捕獲")
            return imgs, info["score"], end, env.steps


def caption(img, title, sub):
    """上に条件名と結果の帯を付ける"""
    im = Image.fromarray(img)
    w, h = im.size
    out = Image.new("RGB", (w, h + 62), (20, 20, 40))
    out.paste(im, (0, 62))
    d = ImageDraw.Draw(out)
    d.text((8, 4), title, font=FONT, fill=(255, 230, 120))
    d.text((8, 36), sub, font=FONT_S, fill=(220, 220, 220))
    return np.array(out)


def save_mp4(frames, path, fps):
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print("保存:", path, f"({len(frames)} フレーム)")


def runs_of(out, exp):
    rs = []
    for p in sorted(glob.glob(os.path.join(out, "runs", f"{exp}_*", "result.json"))):
        if os.path.exists(os.path.join(os.path.dirname(p), "model.pt")):
            with open(p, encoding="utf-8") as f:
                rs.append((json.load(f), os.path.dirname(p)))
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="DQN_reward_study フォルダ")
    ap.add_argument("--exps", nargs="+", default=["exp5"])
    ap.add_argument("--boards", type=int, default=20, help="一番良いプレイを探す評価の盤面の数")
    ap.add_argument("--compare-seed", type=int, default=None,
                    help="並べ比べに使う学習 seed (省略時: クリアした条件の一番良いプレイの seed と盤面を使う)")
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    vd = os.path.join(a.out, "videos")
    os.makedirs(vd, exist_ok=True)
    log = []
    for exp in a.exps:
        rs = runs_of(a.out, exp)
        if not rs:
            print("モデルがありません:", exp)
            continue
        labels = list(dict.fromkeys(r["label"] for r, _ in rs))
        # 条件ごとに一番良いプレイを探す (画像なしで速く回し、見つけた回だけ録画し直す)
        best_board = {}
        for lab in labels:
            best = None
            for r, d in rs:
                if r["label"] != lab:
                    continue
                _, env, agent = load(d)
                for k in range(a.boards):
                    _, sc, end, st = play(env, agent, k, frames=False)
                    log.append([exp, lab, r["seed"], k, sc, end, st])
                    key = (end == "クリア", sc)
                    if best is None or key > best[0]:
                        best = (key, d, r, k, end, st)
            (_, d, r, k, end, st) = best
            _, env, agent = load(d)
            imgs, sc, end, st = play(env, agent, k)
            sub = f"学習 seed {r['seed']} / 盤面 {k} / 得点 {sc} / {end} ({st} ステップ)"
            save_mp4([caption(f, f"{exp} {lab}（一番良いプレイ）", sub) for f in imgs],
                     os.path.join(vd, f"best_{exp}_{lab.replace(' ', '').replace('/', '_')}.mp4"), a.fps)
            best_board[lab] = (k, r["seed"], end)
        # 同じ盤面・同じ seed で横に並べる。クリアした条件があればその盤面と seed を使う (クリアの場面を見せるため)
        ref = next((lab for lab in labels if best_board[lab][2] == "クリア"), labels[-1])
        k, seed, _ = best_board[ref]
        seed = a.compare_seed or seed
        panels = []
        for lab in labels:
            hit = [(r, d) for r, d in rs if r["label"] == lab and r["seed"] == seed]
            if not hit:
                continue
            r, d = hit[0]
            _, env, agent = load(d)
            imgs, sc, end, st = play(env, agent, k)
            panels.append([caption(f, lab, f"得点 {sc} / {end} ({st} ステップ)") for f in imgs])
        n = max(len(p) for p in panels)
        frames = []
        for t in range(n + 2 * a.fps):          # 最後に2秒止める
            row = [p[min(t, len(p) - 1)] for p in panels]
            frames.append(np.concatenate([np.pad(x, ((0, 0), (0, 8), (0, 0))) for x in row], axis=1))
        save_mp4(frames, os.path.join(vd, f"compare_{exp}_seed{seed}_board{k}.mp4"), a.fps)
    with open(os.path.join(vd, f"plays_{'_'.join(a.exps)}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["実験", "条件", "学習seed", "盤面", "得点", "終わり方", "ステップ"]] + log)


if __name__ == "__main__":
    main()
