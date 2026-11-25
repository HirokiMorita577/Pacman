"""Plot training loss for each factor comparison in the saved experiment."""
import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

from experiments.catalog import catalog
from experiments.config import HEURISTICS, digest
from experiments.report import aligned
from plot_all_comparisons import collect, read_batches


def loss_points(rows):
    window = deque()
    total = 0.0
    points = []
    last = None
    for row in rows:
        episode = int(row["learning_episode"])
        if row.get("loss"):
            value = float(row["loss"])
            window.append((episode, value))
            total += value
        while window and window[0][0] <= episode - 50:
            total -= window.popleft()[1]
        if window:
            last = [episode, total / len(window)]
            if not points or episode % 10 == 0:
                points.append(last)
    if last is not None and last != points[-1]:
        points.append(last)
    return points


def objective(config):
    algorithm = config["algorithm"]
    if algorithm in HEURISTICS:
        return None
    if algorithm == "ppo":
        return "PPO 複合損失"
    loss = {"huber": "Smooth L1", "mse": "MSE"}[config["agent"]["loss"]]
    return f"HRA 3ヘッド平均・{loss}" if algorithm == "hra" else f"TD 損失・{loss}"


def condition_losses(plan, runtime, records):
    summaries, matched = {}, set()
    for condition in plan["conditions"]:
        config = condition["config"]
        algorithm = config["algorithm"]
        execution = runtime["execution"]
        if algorithm in HEURISTICS:
            execution = {key: execution[key] for key in ("torch_threads", "deterministic")}
            execution["device"] = "cpu"
        series, episodes = [], []
        for seed in plan["seeds"]:
            frozen = {**runtime, "execution": execution, "schema": 1, "config": config, "seed": seed}
            trial = f"{algorithm}_{digest(frozen)[:20]}_s{seed}"
            if trial in records:
                matched.add(trial)
                if records[trial]["curve"]:
                    series.append(records[trial]["curve"])
                if records[trial].get("episodes"):
                    episodes.append(records[trial]["episodes"])
        summaries[condition["id"]] = {"name": algorithm, "objective": objective(config), "series": series,
                                      "episode_mean": average_episodes(episodes), "n_train": len(episodes)}
    if matched != records.keys():
        raise ValueError("Some training logs did not match the saved plan")
    return summaries


def average_episodes(trials):
    if not trials:
        return np.empty((0, 5))
    by_trial = [{row[0]: row[1:] for row in trial} for trial in trials]
    common = sorted(set.intersection(*(set(trial) for trial in by_trial)))
    if not common:
        return np.empty((0, 5))
    values = np.array([[trial[episode] for episode in common] for trial in by_trial], dtype=float)
    means = values[:, :, :2].mean(axis=0)
    loss = values[:, :, 2]
    n_loss = np.isfinite(loss).sum(axis=0)
    loss_mean = np.divide(np.nansum(loss, axis=0), n_loss, out=np.full(len(common), np.nan), where=n_loss > 0)
    return np.column_stack((common, means, loss_mean, n_loss))


def plot_episode_means(path, title, members, planned):
    objectives = list(dict.fromkeys(result["objective"] for _, result in members if len(result["series"])))
    fig, axes = plt.subplots(2 + len(objectives), 1, sharex=True, squeeze=False,
                             figsize=(13, 2.45 * (2 + len(objectives)) + 1.0))
    for i, (label, result) in enumerate(members):
        data = result["episode_mean"]
        if not len(data):
            continue
        color = matplotlib.colormaps["tab20"](i % 20)
        n = result["n_train"]
        style = "-" if n == planned and i < 20 else "--"
        for panel, column, factor in ((0, 1, 1), (1, 2, 100)):
            axes[panel, 0].plot(data[:, 0], data[:, column] * factor, color=color,
                               alpha=0.7, linewidth=0.75, linestyle=style,
                               label=f"{label} [{n}/{planned}]" if panel == 0 else None)
        if result["objective"] in objectives:
            panel = 2 + objectives.index(result["objective"])
            axes[panel, 0].plot(data[:, 0], data[:, 3], color=color, alpha=0.7,
                               linewidth=0.75, linestyle=style)
    for ax, label in zip(axes[:, 0], ["学習中のゲームスコア（点）", "学習中のクリア率（%）", *objectives]):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("学習エピソード数")
    axes[1, 0].set_ylim(-3, 103)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1),
                          fontsize=8, ncols=2 if len(handles) > 12 else 1)
    else:
        axes[0, 0].text(0.5, 0.5, "エピソード別の学習ログなし", transform=axes[0, 0].transAxes, ha="center")
    fig.suptitle(title + "｜各エピソードのseed平均", weight="bold")
    fig.text(0.012, 0.012, "探索を含む学習時の記録。seed間で同じエピソード番号を平均。損失は値が記録されたseedのみ。",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def write_averages(path, results, finals):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "algorithm", "loss_type", "train_seeds", "first_episode", "last_episode",
                         "mean_training_score", "mean_training_clear_rate", "mean_training_loss",
                         "episodes_with_loss", "final_seeds", "mean_final_score", "mean_final_clear_rate"])
        for key, result in results.items():
            values = result["episode_mean"]
            train = values[:, 1:3].mean(axis=0).tolist() if len(values) else ["", ""]
            observed_loss = values[np.isfinite(values[:, 3]), 3]
            final = finals[key]["finals"]
            writer.writerow([key, result["name"], result["objective"], result["n_train"],
                             int(values[0, 0]) if len(values) else "", int(values[-1, 0]) if len(values) else "",
                             *train, observed_loss.mean() if len(observed_loss) else "", len(observed_loss),
                             len(final), np.mean([r["score"] for r in final]) if final else "",
                             np.mean([r["clear"] for r in final]) if final else ""])


def plot_group(path, title, members, planned):
    panels = defaultdict(list)
    absent = []
    for label, result in members:
        if result["series"]:
            panels[result["objective"]].append((label, result))
        else:
            reason = "学習なし" if result["objective"] is None else "損失ログなし"
            absent.append(f"{label}（{reason}）")
    fig, axes = plt.subplots(max(1, len(panels)), 1, squeeze=False,
                             figsize=(13, max(3.4, 2.8 * len(panels) + 1.1)))
    for ax, (name, entries) in zip(axes[:, 0], panels.items()):
        for i, (label, result) in enumerate(entries):
            series = [[{"learning_episode": episode, "loss": loss} for episode, loss in trial]
                      for trial in result["series"]]
            count = len(series)
            color = matplotlib.colormaps["tab20"](i % 20)
            label = f"{label} [{count}/{planned}]"
            x, values = aligned(series, "loss")
            if x is None:
                for j, trial in enumerate(series):
                    ax.plot([r["learning_episode"] for r in trial], [r["loss"] for r in trial],
                            alpha=0.55, color=color, label=label + "［個別seed］" if j == 0 else None)
            else:
                ax.plot(x, values.mean(axis=0), color=color, linestyle="-" if count == planned else "--",
                        marker="." if len(x) == 1 else None, label=label)
        ax.set(ylabel=name, xlabel="学習エピソード数")
        ax.grid(alpha=0.24)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, ncols=2 if len(entries) > 12 else 1)
    if not panels:
        ax = axes[0, 0]
        ax.set_axis_off()
        ax.text(0.5, 0.57, "プロットできる学習損失がありません", transform=ax.transAxes, ha="center")
    fig.suptitle(title + "｜学習損失", weight="bold")
    fig.text(0.012, 0.012, "直近50エピソードの記録済み損失の移動平均。共通範囲でseed平均。破線は予定の一部のseed。",
             fontsize=9)
    if absent:
        for i in range(0, len(absent), 5):
            fig.text(0.012, 0.14 + 0.045 * (i // 5), "  /  ".join(absent[i:i + 5]), fontsize=8)
    fig.tight_layout(rect=(0, 0.22 + 0.045 * ((len(absent) - 1) // 5) if absent else 0.055, 1, 0.94))
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def plot_local(root, out):
    members = []
    for trial in sorted(root.glob("*/migration.json")):
        data = json.loads(trial.read_text(encoding="utf-8"))
        frozen = data["local_manifest"]
        path = trial.with_name("train.csv")
        series = []
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                points = loss_points(csv.DictReader(stream))
            if points:
                series.append(points)
        opts = frozen["config"]["algorithm_options"]
        setting = f"τ={opts['target_tau']}" if opts.get("target_tau") else f"更新間隔={opts.get('train_every', 1)}"
        label = f"{setting}・seed {frozen['seed']}"
        members.append((label, {"objective": objective(frozen["config"]), "series": series}))
    if members:
        plot_group(out, "ローカル Double DQN（各線は1試行）", members, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--local-runs", type=Path)
    args = parser.parse_args()
    document = json.loads((args.source / "plan.json").read_text(encoding="utf-8"))
    plan = document["plan"]
    records = read_batches(args.source, "loss_values_*.json")
    errors = [key for key, row in records.items() if "error" in row or "episodes" not in row]
    if errors:
        raise ValueError(f"{len(errors)} train.csv files could not be read: {errors[:4]}")
    results = condition_losses(plan, document["runtime"], records)
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    font = Path("C:/Windows/Fonts/meiryo.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    headings = catalog()
    for name, group in plan["groups"].items():
        title = {"reward_power": "パワーえさの学習用報酬", "reward_approach_dot": "えさへの接近報酬"}.get(
            name, headings.get(name, {}).get("description", name))
        members = [(entry["label"], results[entry["condition"]]) for entry in group]
        plot_group(figures / f"{name}_loss.png", title, members, len(plan["seeds"]))
        plot_episode_means(figures / f"{name}_episode_mean.png", title, members, len(plan["seeds"]))
    with (args.out / "loss_coverage.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "label", "condition", "algorithm", "loss", "seeds_with_loss", "planned_seeds"])
        for name, group in plan["groups"].items():
            for entry in group:
                result = results[entry["condition"]]
                writer.writerow([name, entry["label"], entry["condition"], result["name"], result["objective"],
                                 len(result["series"]), len(plan["seeds"])])
    finals = collect(plan, document["runtime"], read_batches(args.source, "rank_values_*.json"), {})
    write_averages(args.out / "condition_averages.csv", results, finals)
    if args.local_runs is not None:
        plot_local(args.local_runs, args.out / "local_training_loss.png")
    print(f"{2 * len(plan['groups'])} comparison figures; {len(records)} logs; "
          f"{sum(bool(result['series']) for result in results.values())} conditions with loss")


if __name__ == "__main__":
    main()
