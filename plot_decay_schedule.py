"""Draw proposed linear decay alongside the current comparison baseline."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np

from experiments.config import DEFAULT, epsilon_at


def main():
    font = Path("C:/Windows/Fonts/meiryo.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    output = Path(__file__).resolve().parent / "output" / "all_comparisons_20261125" / "alpha_epsilon_decay.png"
    budget = DEFAULT["train"]["steps"]
    steps = np.arange(0, budget + 1, 500)
    remaining = 1 - steps / budget
    baseline_alpha = DEFAULT["agent"]["lr"]
    baseline_epsilon = [epsilon_at(DEFAULT, step) for step in steps]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    panels = [
        ("学習率 α", 0.003 * remaining, np.full_like(remaining, baseline_alpha), "0.003", "#3158a5"),
        ("探索率 ε", remaining, baseline_epsilon, "100%", "#007f76"),
    ]
    for ax, (label, proposed, baseline, start, color) in zip(axes, panels):
        ax.plot(steps, proposed, color=color, linewidth=2.5, label="提案：線形減衰")
        ax.plot(steps, baseline, color="#63768f", linestyle="--", linewidth=1.8, label="現在の基準設定")
        ax.scatter([0, budget], [proposed[0], proposed[-1]], color=color, zorder=4)
        ax.annotate(start, (0, proposed[0]), xytext=(8, -4), textcoords="offset points", color=color)
        ax.annotate("0", (budget, 0), xytext=(-12, 8), textcoords="offset points", color=color)
        ax.set(title=label, xlabel="累計環境ステップ数", ylabel=label,
               xlim=(-0.025 * budget, 1.025 * budget), ylim=(-0.035 * proposed[0], 1.13 * proposed[0]))
        ax.set_xticks(np.linspace(0, budget, 5))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.23)
        ax.legend(loc="upper right", fontsize=9)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.suptitle("学習率と探索率を0まで下げる設定案（未学習）", weight="bold", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
