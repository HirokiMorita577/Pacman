"""学習曲線 (FR-42, NFR-05): エピソードごとのスコアと100エピソード移動平均、フレーム数"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 論文 Table 1 の基準線
BASELINES = {"HRA (paper, fixed start)": 25304, "Human": 15693}


def _read(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["episode"]) for r in rows]
    score = [float(r["score"]) for r in rows]
    avg = [float(r["score_avg100"]) for r in rows]
    frames = [int(r["frames"]) for r in rows]
    return ep, score, avg, frames


def learning_curve_figure(csv_paths, labels=None, baselines=True):
    """複数設定の比較もできる学習曲線 (FR-43)"""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    labels = labels or [None] * len(csv_paths)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for path, label in zip(csv_paths, labels):
        ep, score, avg, frames = _read(path)
        if not ep:
            continue
        if len(csv_paths) == 1:
            ax1.plot(ep, score, color="#9aa7b8", lw=0.6, alpha=0.6, label="score")
        ax1.plot(ep, avg, lw=2, label=(label or "") + " avg100")
        ax2.plot(ep, frames, lw=0.8, label=label)
    if baselines:
        for name, v in BASELINES.items():
            ax1.axhline(v, ls="--", lw=1, color="#888")
            ax1.text(ax1.get_xlim()[0], v, f" {name} {v:,}", va="bottom", fontsize=8, color="#555")
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Frames / episode")
    ax2.set_xlabel("Episode")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_learning_curve(csv_path, png_path, title=None):
    fig = learning_curve_figure(csv_path)
    if title:
        fig.axes[0].set_title(title)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
