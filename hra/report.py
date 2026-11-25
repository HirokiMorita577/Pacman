"""
学習状況のレポート (班員向け)。Drive の各実験フォルダ (log.csv, config.yaml) から PNG を作る。
  python -m hra.report --runs /content/drive/MyDrive/HRA_runs --out /content/drive/MyDrive/HRA_runs/report
出力: 1_learning_curve.png  2_score_vs_frames.png  3_level_reached.png  4_score_breakdown.png
      5_settings.png  6_past_experiments.png  summary.json
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PAPER = [("HRA (paper, fixed start)", 25304), ("Human", 15693)]
METHOD = {"ram_cell": ("RAM position + 1 step = next cell", "tab:blue"),
          "pixels_frames": ("Pixel position + 1 step = 8 frames", "tab:orange")}
SEED_STYLE = {"101": "-", "202": "--", "303": ":"}


def read_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def rolling(x, n=100):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - n)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def load_runs(root):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, "clean_*", "log.csv"))):
        name = os.path.basename(os.path.dirname(p))
        log = read_log(p)
        if log is None:
            continue
        method = "ram_cell" if "ram_cell" in name else "pixels_frames"
        seed = name.rsplit("seed", 1)[-1]
        cfg_path = os.path.join(os.path.dirname(p), "config.yaml")
        cfg = None
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        runs.append(dict(name=name, method=method, seed=seed, log=log, cfg=cfg))
    return runs


def _style(run):
    label, color = METHOD[run["method"]]
    return dict(color=color, ls=SEED_STYLE.get(run["seed"], "-"), lw=1.6), label


def _paper_lines(ax):
    for (label, v), c in zip(PAPER, ("tab:red", "tab:green")):
        ax.axhline(v, color=c, lw=1, ls="-.", alpha=0.7)
        ax.text(ax.get_xlim()[1], v, f" {label} {v:,}", color=c, va="center", fontsize=8)


def fig_curve(runs, out, x_key, xlabel, fname, title):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for r in runs:
        st, label = _style(r)
        x = r["log"]["episode"] if x_key == "episode" else np.cumsum(r["log"]["frames"]) / 1e6
        ax.plot(x, rolling(r["log"]["score"]), label=f"{label} / seed {r['seed']}", **st)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score (moving average of 100 episodes)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 27000)
    _paper_lines(ax)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=140)
    plt.close(fig)


def fig_levels(runs, out):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for r in runs:
        st, label = _style(r)
        ax.plot(r["log"]["episode"], rolling(r["log"]["level_reached"] >= 2) * 100,
                label=f"{label} / seed {r['seed']}", **st)
    ax.set_xlabel("Episode")
    ax.set_ylabel("% of episodes that cleared level 1")
    ax.set_title("Level clear rate (moving average of 100 episodes)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "3_level_reached.png"), dpi=140)
    plt.close(fig)


def fig_breakdown(runs, out):
    parts = [("r_pellet", "Pellets", "#f2c14e"), ("r_power", "Power pellets", "#f78154"),
             ("r_fruit", "Fruit", "#d1495b"), ("r_blue_ghost", "Blue ghosts", "#2e86ab")]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    names = [f"{'RAM+cell' if r['method'] == 'ram_cell' else 'Pixel+8f'}\nseed {r['seed']}" for r in runs]
    bottom = np.zeros(len(runs))
    for key, label, color in parts:
        v = np.array([r["log"][key][-100:].mean() for r in runs])
        ax.bar(names, v, bottom=bottom, label=label, color=color)
        bottom += v
    ax.set_ylabel("Points per episode (last 100 episodes)")
    ax.set_title("Where the points come from")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "4_score_breakdown.png"), dpi=140)
    plt.close(fig)


def fig_settings(runs, out):
    pick = {}
    for r in runs:
        if r["cfg"] and r["method"] not in pick:
            pick[r["method"]] = r["cfg"]
    rows = [("Object position", lambda c: c["env"]["position_source"]),
            ("1 step", lambda c: "until next cell (max 16 frames)" if c["env"]["step_mode"] == "cell"
             else f"{c['env']['frameskip']} frames"),
            ("Learning rate alpha", lambda c: c["gvf"]["alpha"]),
            ("Discount gamma (score / ghost)", lambda c: f"{c['gvf']['gamma_score']} / {c['gvf']['gamma_ghost']}"),
            ("GVF target", lambda c: c["gvf"]["target"]),
            ("Aggregator", lambda c: f"{c['aggregator']['type']} ({c['aggregator']['normalize']})"),
            ("Weights pellet/power/fruit/blue/ghost",
             lambda c: f"{c['heads']['pellet']}/{c['heads']['power_pellet']}/{c['heads']['fruit']}/"
                       f"{c['heads']['blue_ghost']}/{c['heads']['ghost_normalized']}"),
            ("Diversification head", lambda c: f"U(0,{c['exploration']['div_max']:g}) first "
                                               f"{c['exploration']['div_steps']} steps"),
            ("Count head kappa", lambda c: c["exploration"]["kappa"]),
            ("Episodes (target)", lambda c: f"{c['experiment']['episodes']:,}"),
            ("Seeds", lambda c: "101 / 202 / 303")]
    cols = [m for m in ("ram_cell", "pixels_frames") if m in pick]
    cell = [[k] + [str(f(pick[m])) for m in cols] for k, f in rows]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.2))
    ax.axis("off")
    t = ax.table(cellText=cell, colLabels=["Setting"] + [METHOD[m][0] for m in cols],
                 loc="center", cellLoc="left", colWidths=[0.32] + [0.34] * len(cols))
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    for j in range(len(cols) + 1):
        t[0, j].set_facecolor("#dfe7f2")
    ax.set_title("Experiment settings (both methods start from zero)", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "5_settings.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_past(out):
    """これまでの比較実験 (DECISIONS.md の数値)。各条件とも同じ学習済みモデルから追加学習し、評価20回の平均"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    a = [("Pixel + 8 frames\nalpha 0.2", 8404), ("Pixel + cell\nalpha 1.0", 6176),
         ("Pixel + cell\nalpha 0.5", 9126), ("RAM + cell\nalpha 1.0", 9822)]
    ax = axes[0]
    bars = ax.bar([x for x, _ in a], [v for _, v in a],
                  color=["#bbbbbb", "#bbbbbb", "#bbbbbb", "tab:blue"])
    ax.bar_label(bars, fmt="{:,.0f}", fontsize=9)
    ax.set_title("Step mode / position / alpha\n(model at 5,200 ep + 300 ep, mean of 20 evals)", fontsize=10)
    ax.set_ylabel("Evaluation score")
    b = [("alpha 0.2\ngamma_g 0.99", 9970), ("alpha 0.1\ngamma_g 0.99", 8879),
         ("alpha 0.2\ngamma_g 0.97", 8381), ("alpha 0.2\ngamma_g 0.95", 7900)]
    ax = axes[1]
    bars = ax.bar([x for x, _ in b], [v for _, v in b], color=["tab:blue", "#bbbbbb", "#bbbbbb", "#bbbbbb"])
    ax.bar_label(bars, fmt="{:,.0f}", fontsize=9)
    ax.set_title("Learning rate alpha / ghost discount gamma (8-frame steps)\n"
                 "(model at 5,000 ep + 200 ep, mean of 20 evals)", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 11500)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "6_past_experiments.png"), dpi=140)
    plt.close(fig)


def summary(runs, out):
    s = {}
    for r in runs:
        lg = r["log"]
        s[r["name"]] = {"episodes": int(lg["episode"][-1]),
                        "avg100": float(lg["score"][-100:].mean()),
                        "best_score": float(lg["score"].max()),
                        "max_level": int(lg["level_reached"].max()),
                        "clear_rate_last100": float((lg["level_reached"][-100:] >= 2).mean()),
                        "frames_million": float(lg["frames"].sum() / 1e6),
                        "alpha": r["cfg"]["gvf"]["alpha"] if r["cfg"] else None}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="HRA_runs フォルダ")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    fig_curve(runs, a.out, "episode", "Episode", "1_learning_curve.png",
              "HRA on Ms. Pac-Man: training from zero (100-episode moving average)")
    fig_curve(runs, a.out, "frames", "Game frames played (millions)", "2_score_vs_frames.png",
              "Same curves against the amount of play (fair comparison of experience)")
    fig_levels(runs, a.out)
    fig_breakdown(runs, a.out)
    fig_settings(runs, a.out)
    fig_past(a.out)
    for k, v in summary(runs, a.out).items():
        print(f"{k:36s} ep {v['episodes']:5d}  avg100 {v['avg100']:8.0f}  best {v['best_score']:7.0f}  "
              f"maxlv {v['max_level']}  clear {v['clear_rate_last100']:.0%}")


if __name__ == "__main__":
    main()
