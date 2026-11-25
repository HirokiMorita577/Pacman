"""Plot the local Double DQN development evaluations by saved checkpoint."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.metrics.open(encoding="utf-8-sig", newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: (row["condition"], int(row["checkpoint_steps"])))
    conditions = {"4b8ce14ba8d44e75f5d8": "ソフト更新率 τ=0.05",
                  "ba285031a1b85a13da7d": "4ステップごとに更新"}
    font = Path("C:/Windows/Fonts/meiryo.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    labels = [f"{conditions.get(r['condition'], r['condition'])} ／ seed {r['train_seed']} ／ "
              f"{int(r['checkpoint_steps']):,} ステップ" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    colors = ["#114b82" if r["condition"] == "4b8ce14ba8d44e75f5d8" else "#007c70" for r in rows]
    for ax, key, label, scale in zip(axes, ("clear_rate", "score_mean"), ("クリア率（%）", "平均ゲームスコア（点）"), (100, 1)):
        values = [float(row[key]) * scale for row in rows]
        positions = range(len(rows))
        ax.barh(positions, values, height=0.55, color=colors, alpha=0.78)
        for i, value in enumerate(values):
            text = f"{value:.0f}%" if scale == 100 else f"{value:,.1f}"
            ax.annotate(text, (value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, 105 if scale == 100 else max(values) * 1.21)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.24)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("ローカル実験のチェックポイント｜Double DQN の開発評価", weight="bold")
    fig.text(0.5, 0.017, "同じ開発用100盤面・探索なし。保存地点が異なる途中モデルの評価。",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
