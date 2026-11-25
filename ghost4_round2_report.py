"""
第2回比較 (ghost4_round2) の結果を PC で検証し、報告用の表と動画を作る。学習はしない。

  python ghost4_round2_report.py --study output/ghost4_round2_20261007/study --out output/ghost4_round2_20261007/report

1. 保存されたモデルと result.json のハッシュを saved_result.json と照合する
2. 最終評価 (未使用seed 940000〜940099) を、Colab と同じ関数 (reward_study.evaluate) で再実行し、
   evaluation_final.json の集計と一致するかを確かめる。Colab の1回ごとの結果 (evaluation_final_episodes.csv) があれば
   1行ずつ照合する (match_level = episode / aggregate)。1回ごとの結果を CSV に残す
3. 学習seedごとの改善幅と、同じ学習seed・同じ評価seed (初期の盤面) で control と選んだ条件を対にした比較
   (両方クリア / 片方だけ / 両方失敗)。行動が違えば乱数の使われ方も変わるので、敵の動きまで同じ比較ではない
4. 動画: 選んだ条件の成功 (最高得点) ・選んだ条件の代表的な失敗 (エサ回収率が失敗の中央値に近い回) ・
   成功と同じ盤面での control。どれも評価の再現で、得点・ステップ数が一致することを確かめる
"""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_env(r, render_mode=None):
    import reward_study as study
    from pacman_env import PacManEnv
    return PacManEnv(render_mode=render_mode, ghost_mode=study.FIXED["ghost_mode"], num_ghosts=r["num_ghosts"],
                     rewards=r["rewards"], lives=r["lives"], include_lives=r["include_lives"],
                     max_steps=study.FIXED["max_steps"], observation_version=r["observation_version"],
                     **r.get("env_options", {}))


def evaluate_run(args):
    """1モデルの最終評価を再実行する (別プロセス)"""
    run_dir, r, seed0, n = args
    import torch
    import reward_study as study
    from dqn_agent import load_agent
    torch.set_num_threads(1)
    env = make_env(r)
    agent = load_agent(str(Path(run_dir) / "model.pt"), env.observation_space.shape[0], 4)
    return study.evaluate(agent, r["rewards"], n, r["lives"], seed0=seed0, include_lives=r["include_lives"],
                          num_ghosts=r["num_ghosts"], observation_version=r["observation_version"],
                          env_options=r.get("env_options", {}))


def record_video(study_dir, r, eval_seed, expect, path, title, fps=10):
    import numpy as np
    import pygame
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import subprocess
    from PIL import Image, ImageDraw
    from dqn_agent import load_agent
    from record_plays import _font

    font, small = _font(24), _font(16)
    font_path = next((p for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                                  "C:/Windows/Fonts/meiryob.ttc") if Path(p).exists()), None)
    env = make_env(r, render_mode="rgb_array")
    agent = load_agent(str(Path(study_dir) / "runs" / r["id"] / "model.pt"), env.observation_space.shape[0], 4)
    obs, _ = env.reset(seed=eval_seed)
    env.render()
    if font_path:
        env.font = pygame.font.Font(font_path, 14)
        env.font_big = pygame.font.Font(font_path, 18)

    def frame(final=None):
        env._update_planned_path()
        im = Image.fromarray(env.render())
        canvas = Image.new("RGB", (im.width, im.height + 86), (18, 21, 40))
        canvas.paste(im, (0, 86))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 5), final or title, font=font, fill=(255, 225, 100))
        draw.text((12, 41), f"{r['label']} / 学習seed {r['seed']} / 評価seed {eval_seed}（未使用seed）",
                  font=small, fill=(226, 232, 242))
        draw.text((12, 62), f"Step {env.steps} / 残機 {env.lives} / エサ {env.dots_eaten}/{env.total_dots} / 貪欲方策",
                  font=small, fill=(170, 202, 226))
        return np.asarray(canvas)

    frames = 0
    with imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2,
                            ffmpeg_params=["-crf", "20", "-movflags", "+faststart"]) as writer:
        while True:
            env.set_qvalues(agent.get_qvalues(obs))
            writer.append_data(frame())
            frames += 1
            obs, _, te, tr, info = env.step(agent.select_action(obs, training=False))
            if te or tr:
                break
        assert (info["score"], env.steps, bool(info["cleared"])) == expect, (info["score"], env.steps, expect)
        end = "CLEAR! エサをすべて回収" if info["cleared"] else f"失敗：残りエサ {env.total_dots - env.dots_eaten}個"
        last = frame(end)
        for _ in range(3 * fps):
            writer.append_data(last)
            frames += 1
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"],
                   check=True)
    env.close()
    return dict(file=Path(path).name, variant=r["label"], model_id=r["id"], train_seed=r["seed"], eval_seed=eval_seed,
                score=info["score"], steps=env.steps, cleared=bool(info["cleared"]), lives_left=env.lives,
                dots_left=env.total_dots - env.dots_eaten, frames=frames, fps=fps, sha256=digest(path),
                reproduced_evaluation=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    study_dir, out = Path(a.study), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((study_dir / "results.json").read_text(encoding="utf-8"))
    final = json.loads((study_dir / "evaluation_final.json").read_text(encoding="utf-8"))
    sel = final["selection"]

    # 1. モデルの照合: Colab の受領記録 (saved_result.json) か、最終評価の1回ごとの記録にあるモデルのハッシュと比べる
    if (study_dir / "results_final.json").exists():
        results = results + [r for r in json.loads((study_dir / "results_final.json").read_text(encoding="utf-8"))
                             if r["id"] not in {x["id"] for x in results}]
    raw_path = study_dir / "evaluation_final_episodes.csv"
    model_sha = {x["model_id"]: x["model_sha256"] for x in csv.DictReader(raw_path.open(encoding="utf-8-sig"))}         if raw_path.exists() else {}
    verified = 0
    for r in results:
        run = study_dir / "runs" / r["id"]
        if (run / "saved_result.json").exists():
            receipt = json.loads((run / "saved_result.json").read_text())
            for name, h in receipt["sha256"].items():
                assert digest(run / name) == h, f"hash mismatch {r['id']} {name}"
            verified += 1
        elif r["id"] in model_sha:
            assert digest(run / "model.pt") == model_sha[r["id"]], f"hash mismatch {r['id']} model.pt"
            verified += 1
    print("MODELS_VERIFIED", verified, "of", len(results), flush=True)

    # 2. 最終評価の再実行
    base = sel.get("baseline", "control")          # 第2回は control、第3回は safe
    if (study_dir / "results_final.json").exists():   # 第3回: 最終評価で学習seedを増やしたモデルも含む
        results = json.loads((study_dir / "results_final.json").read_text(encoding="utf-8"))
    # 第5回のように全条件を最終評価した場合 (selected = "all") は、全条件を照合する
    all_mode = sel["selected"] == "all"
    targets = results if all_mode else [r for r in results if r["label"] in (base, sel["selected"])]
    jobs = [(str(study_dir / "runs" / r["id"]), r, sel["final_seed0"], sel["final_episodes"]) for r in targets]
    with ProcessPoolExecutor(a.workers) as pool:
        evals = list(pool.map(evaluate_run, jobs))
    saved = {(row["variant"], row["train_seed"]): row for row in final["by_seed"]}
    # Colab が1回ごとの結果を保存していれば (第3回以降の evaluation_final_episodes.csv)、1行ずつ照合する。
    # 第2回は集計値だけなので、集計値の一致しか確かめられない (どちらで照合したかを記録する)
    raw = {(x["variant"], int(x["train_seed"]), int(x["eval_seed"])): x
           for x in csv.DictReader(raw_path.open(encoding="utf-8-sig"))} if raw_path.exists() else None
    rows, checks = [], []
    for r, ev in zip(targets, evals):
        s = saved[(r["label"], r["seed"])]
        same = (sum(ev["clears"]) == s["clears"] and abs(ev["score_mean"] - s["score_mean"]) < 1e-9
                and abs(ev["steps_mean"] - s["steps_mean"]) < 1e-9
                and abs(ev["pellet_fraction_mean"] - s["pellet_fraction_mean"]) < 1e-9)
        check = dict(variant=r["label"], train_seed=r["seed"], clears=sum(ev["clears"]),
                     matches_colab=same, match_level="aggregate")
        if raw is not None:
            mism = [i for i in range(len(ev["scores"]))
                    if (lambda x: x is None or (int(x["cleared"]), int(x["score"]), int(x["steps"]))
                        != (int(ev["clears"][i]), ev["scores"][i], ev["steps"][i]))(raw.get((r["label"], r["seed"], ev["seed0"] + i)))]
            check.update(matches_colab=same and not mism, match_level="episode", mismatched_episodes=len(mism))
        checks.append(check)
        for i in range(len(ev["scores"])):
            rows.append(dict(variant=r["label"], train_seed=r["seed"], eval_seed=ev["seed0"] + i,
                             cleared=int(ev["clears"][i]), score=ev["scores"][i], steps=ev["steps"][i],
                             pellet_fraction=round(ev["pellet_fraction_each"][i], 4),
                             lives_lost=ev["lives_lost_each"][i], ending=ev["ending_each"][i]))
    assert all(c["matches_colab"] for c in checks), checks
    print("FINAL_REPRODUCED", checks, flush=True)
    with (out / "final_by_episode.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # 対比較・動画に使う条件: 1つを選んだ回はその条件、全条件の回は基準以外でクリア率が最も高い条件
    clear_by_variant = {v: sum(c["clears"] for c in checks if c["variant"] == v) for v in dict.fromkeys(c["variant"] for c in checks)}
    chosen = max((v for v in clear_by_variant if v != base), key=clear_by_variant.get) if all_mode else sel["selected"]

    # 3. 同じ盤面での対比較
    key = {(x["variant"], x["train_seed"], x["eval_seed"]): x for x in rows}
    paired = {}
    for (v, ts, es), x in key.items():
        if v != base:
            continue
        y = key[(chosen, ts, es)]
        k = {(1, 1): "both_clear", (0, 1): "only_selected", (1, 0): "only_control", (0, 0): "neither"}[(x["cleared"], y["cleared"])]
        paired.setdefault(ts, {}).setdefault(k, 0)
        paired[ts][k] += 1
    total = {k: sum(p.get(k, 0) for p in paired.values()) for k in ("both_clear", "only_selected", "only_control", "neither")}
    # 学習seedごとの改善幅 (選んだ条件 − control)。固定した学習済みモデルの比較であり、学習し直したモデル一般の保証ではない
    rate = {(c["variant"], c["train_seed"]): c["clears"] / sel["final_episodes"] for c in checks}
    seeds_ = sorted({c["train_seed"] for c in checks})
    diff = {s: round(100 * (rate[(chosen, s)] - rate[(base, s)]), 1) for s in seeds_}
    diff_by_variant = {v: {s: round(100 * (rate[(v, s)] - rate[(base, s)]), 1) for s in seeds_}
                       for v in clear_by_variant if v != base}
    # 終わり方の内訳 (失敗のうち残りエサ5個以下・時間切れ、成功時の残機)
    detail = {}
    for v in (clear_by_variant if all_mode else dict.fromkeys((base, chosen))):
        g = [x for x in rows if x["variant"] == v]
        fails = [x for x in g if not x["cleared"]]
        detail[v] = dict(failures=len(fails), caught=sum(x["ending"] == "caught" for x in fails),
                         timeouts=sum(x["ending"] == "truncated" for x in fails),
                         failed_with_le5_dots=sum(round((1 - x["pellet_fraction"]) * 156) <= 5 for x in fails),
                         lives_left_on_clear=dict(Counter(3 - x["lives_lost"] for x in g if x["cleared"])))
    (out / "final_reproduction.json").write_text(json.dumps(
        dict(selection=sel, compared=chosen, checks=checks, diff_points_by_train_seed=diff,
             diff_points_by_variant=diff_by_variant, paired_by_train_seed=paired,
             paired_total=total, endings=detail,
             note="paired = same train seed and same initial eval seed; ghost trajectories can still differ "
                  "because different actions change the game RNG usage"),
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("PAIRED", total, flush=True)

    # 4. 動画
    if a.no_video:
        return
    by_id = {(r["label"], r["seed"]): r for r in results}
    mine = [x for x in rows if x["variant"] == chosen]
    wins = [x for x in mine if x["cleared"]]
    fails = sorted((x for x in mine if not x["cleared"]), key=lambda x: x["pellet_fraction"])
    picks = []                                          # 成功が無い・失敗が無い場合は、その動画を作らない
    if wins:
        win = max(wins, key=lambda x: (x["score"], -x["steps"]))
        picks.append((win, f"{chosen}｜成功（最高得点）"))
    if fails:
        median = fails[len(fails) // 2]["pellet_fraction"]
        fail = min(fails, key=lambda x: (abs(x["pellet_fraction"] - median), x["train_seed"], x["eval_seed"]))
        picks.append((fail, f"{chosen}｜代表的な失敗"))
    if wins and base != chosen:
        picks.append((key[(base, win["train_seed"], win["eval_seed"])], f"{base}｜成功例と同じ盤面"))
    vids = out / "videos"
    vids.mkdir(exist_ok=True)
    records = []
    for x, title in picks:
        name = f"{x['variant']}_{'clear' if x['cleared'] else 'fail'}_train{x['train_seed']}_eval{x['eval_seed']}.mp4"
        rec = record_video(study_dir, by_id[(x["variant"], x["train_seed"])], x["eval_seed"],
                           (x["score"], x["steps"], bool(x["cleared"])), vids / name, title)
        records.append(rec)
        print("VIDEO_VERIFIED", rec["file"], rec["score"], rec["steps"], rec["cleared"], flush=True)
    (vids / "videos.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
