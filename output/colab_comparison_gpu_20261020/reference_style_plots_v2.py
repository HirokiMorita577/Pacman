"""Presentation plots from completed comparison runs; training code is untouched.

Example: python reference_style_plots.py --group replay_td
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(config):
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def choose_font():
    from matplotlib import font_manager
    bundled = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if bundled.exists():
        font_manager.fontManager.addfont(str(bundled))
    names = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "Yu Gothic"):
        if name in names:
            plt.rcParams["font.family"] = name
            return name
    raise RuntimeError("Japanese font unavailable; install a Japanese font before plotting")


def rolling(values, window):
    array = np.asarray(values, dtype=float)
    sums = np.cumsum(np.insert(array, 0, 0.0))
    indexes = np.arange(len(array))
    left = np.maximum(0, indexes + 1 - window)
    return (sums[indexes + 1] - sums[left]) / (indexes + 1 - left)


def load_curve(folder, metric, window):
    with (folder / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    episodes = np.asarray([int(float(row["learning_episode"])) for row in rows])
    values = np.asarray([float(row[metric]) for row in rows])
    if len(episodes) == 0 or len(np.unique(episodes)) != len(episodes):
        raise ValueError(f"Invalid training episodes: {folder}")
    order = np.argsort(episodes)
    episodes, values = episodes[order], values[order]
    return episodes, rolling(values, window)


def member_label(member, config, group):
    if group == "replay_td" and config["algorithm"] in ("dqn", "double_dqn"):
        return "DQN" if config["algorithm"] == "dqn" else "Double DQN"
    label = member["label"]
    if label.startswith("reference"):
        if group.startswith("reward_"):
            name = group.removeprefix("reward_")
            return f"{name}報酬 {config['rewards'][name]:g}"
        return f"基準 ({config['algorithm']})"
    if group.startswith("reward_"):
        name = group.removeprefix("reward_")
        return f"{name}報酬 {config['rewards'][name]:g}"
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/content/pacman_comparison_gpu_study/factors"))
    parser.add_argument("--output", type=Path, default=Path(
        "/content/drive/MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec/priority_dqn_double_20261029/reference_style"))
    parser.add_argument("--group", default="replay_td")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError("window must be positive")
    font = choose_font()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
    plan = json.loads((args.root / "plan.json").read_text(encoding="utf-8"))["plan"]
    conditions = {c["id"]: c["config"] for c in plan["conditions"]}
    members = plan["groups"][args.group]
    if args.group == "replay_td":
        members = [m for m in members if conditions[m["condition"]]["algorithm"] in ("dqn", "double_dqn")]
    results = {}
    for path in (args.root / "runs").glob("*/result.json"):
        if (path.parent / "FAILED.json").exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        results.setdefault(digest(result["config"]), []).append((path.parent, result))
    selected = [(m, conditions[m["condition"]], results.get(m["condition"], [])) for m in members]
    if len(selected) < 2 or any(len(runs) != len(plan["seeds"]) for _, _, runs in selected):
        raise RuntimeError("Every compared condition needs all planned training seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for metric, display, scale, filename in (
        ("score", "スコア", 1, "DQN_DoubleDQN_学習スコア.png"),
        ("cleared", "クリア率（%）", 100, "DQN_DoubleDQN_学習クリア率.png"),
    ):
        fig, ax = plt.subplots(figsize=(11.5, 5.5), dpi=150)
        for member, config, runs in selected:
            curves = [load_curve(folder, metric, args.window) for folder, _ in sorted(runs, key=lambda item: item[1]["train_seed"])]
            start = max(int(x[0]) for x, _ in curves)
            end = min(int(x[-1]) for x, _ in curves)
            if start > end:
                raise RuntimeError("Training curves have no common episode interval")
            x = np.arange(start, end + 1)
            y = np.mean([np.interp(x, episodes, values) for episodes, values in curves], axis=0) * scale
            ax.plot(x, y, linewidth=2, label=f"{member_label(member, config, args.group)}（{len(runs)}回平均）")
        ax.set_xlabel("学習エピソード数")
        ax.set_ylabel(f"{display}（直近{args.window}エピソードの移動平均、学習seed平均）")
        ax.set_title(f"DQNとDouble DQNの比較\nアルゴリズム以外の学習条件を固定", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_xlim(left=0)
        if metric == "cleared":
            ax.set_ylim(0, 100)
        elif metric == "score":
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        target = args.output / filename
        fig.savefig(target)
        plt.close(fig)
        artifacts.append(str(target))
    (args.output / "作図条件.txt").write_text(
        f"フォント: {font}\n横軸: 学習エピソード数\n縦軸: 各seedで直近{args.window}エピソードの移動平均を算出し、{len(plan['seeds'])}seed平均\n"
        "学習中スコアと学習中クリア率（探索あり）。最終評価の値ではありません。\n"
        "比較対象以外の学習条件は同一。学習エピソード数の共通区間のみを表示。\n", encoding="utf-8")
    print("REFERENCE_STYLE_READY", *artifacts, sep="\n", flush=True)


if __name__ == "__main__":
    main()
