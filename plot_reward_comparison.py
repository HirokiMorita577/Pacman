"""Plot verified final evaluations from the Drive condition ranking."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


def reward_series(rows, factor, reference):
    series = []
    for row in rows:
        for name, label, _ in row["factors"]:
            if name == factor:
                value = reference if label.startswith("reference") else float(label)
                series.append((value, row))
    return sorted(series, key=lambda pair: pair[0])


def plot_metric(panels, metric, output):
    clear_rate = metric == "clear"
    scale, seed_column = (100, 2) if clear_rate else (1, 1)
    label = "クリア率（%）" if clear_rate else "ゲームスコア（点）"
    number = "{:.1f}%" if clear_rate else "{:,.0f}"
    color = "#007f76" if clear_rate else "#3158a5"
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.1), sharey=True)
    fig.patch.set_facecolor("#f5f7fa")
    fig.text(0.066, 0.92, f"報酬条件と上位方式の比較｜{label}", fontsize=22, weight="bold", color="#152742")
    fig.text(0.067, 0.855, "ゴースト4体・残機3・20万環境ステップの Double DQN ／ 学習後の探索なし最終評価",
             fontsize=11, color="#455770")
    fig.subplots_adjust(left=0.074, right=0.982, bottom=0.22, top=0.76, wspace=0.13)
    offsets = [-0.13, -0.065, 0, 0.065, 0.13]
    for index, (ax, (title, ticks, rows)) in enumerate(zip(axes, panels)):
        x = list(range(len(rows)))
        averages = [row[metric] * scale for row in rows]
        ax.set_title(title, fontsize=13, weight="bold", pad=13)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dce4ef", linewidth=0.8)
        ax.set_ylim(0, 110 if clear_rate else 4100)
        ax.yaxis.set_major_locator(MultipleLocator(20 if clear_rate else 1000))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xticks(x, ticks, fontsize=9)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#a7b5c8")
        ax.set_xlim(-0.52, len(rows) - 0.48)
        if index == 0:
            ax.set_ylabel(label, fontsize=11)
        if index < 2:
            ax.plot(x, averages, color=color, marker="o", markersize=6, linewidth=2, zorder=3)
        else:
            bars = ["#63768f" if row["algorithm"] == "safe_food" else color for row in rows]
            ax.bar(x, averages, width=0.58, color=bars, alpha=0.28, zorder=2)
            ax.scatter(x, averages, c=bars, s=35, zorder=4)
        for position, row, average in zip(x, rows, averages):
            seeds = [] if row["algorithm"] == "safe_food" else [v[seed_column] * scale for v in row["per_seed"]]
            ax.scatter([position + offset for offset in offsets[:len(seeds)]], seeds,
                       color=color, alpha=0.45, s=17, edgecolors="white", linewidths=0.35, zorder=3)
            ax.annotate(number.format(average), (position, average), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9, weight="bold", color="#152742",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.3), zorder=5)
    fig.text(0.067, 0.088, "小さい点は学習seed別の値。Double DQN は各条件5学習seed × 各100プレイ。安全経路は学習を使わない基準方式。",
             fontsize=10, color="#455770")
    fig.text(0.067, 0.048, "基準の報酬はパワーえさ50・えさへの接近1.5、割引率 γ = 0.95。右端の最高得点条件のみ γ = 0.995。",
             fontsize=10, color="#455770")
    fig.savefig(output, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.ranking.read_text(encoding="utf-8"))
    index = {row["id"]: row for row in rows}
    power = reward_series(rows, "reward_power", 50)
    approach = reward_series(rows, "reward_approach_dot", 1.5)
    benchmark = [
        index["3b898ced50c16dab4474"], index["5df6465246ea3a1261d3"],
        index["335ca7fecf1a6af6a395"], index["3203abe33170c9667ce5"],
    ]
    panels = [
        ("パワーえさの学習用報酬", [f"{v:g}" for v, _ in power], [row for _, row in power]),
        ("えさへの接近報酬", [f"{v:g}" for v, _ in approach], [row for _, row in approach]),
        ("基準と最高成績の条件", ["Double DQN\n基準", "安全経路\n最高クリア率",
                            "接近報酬7.5\n学習方式で\n最高クリア率", "パワー10\nγ = 0.995\n最高得点"], benchmark),
    ]
    font = Path("C:/Windows/Fonts/meiryo.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    args.out.mkdir(parents=True, exist_ok=True)
    for metric, name in (("clear", "reward_clear_rate.png"), ("score", "reward_game_score.png")):
        plot_metric(panels, metric, args.out / name)


if __name__ == "__main__":
    main()
