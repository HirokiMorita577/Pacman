"""PNG comparisons. Every figure's horizontal axis counts learning episodes."""
import csv
import json
from pathlib import Path

import numpy as np

from .config import METRICS, digest
from .engine import atomic_json, write_csv

TRAIN_PLOTS = {
    "score": "Game score during training",
    "return": "Own-reward return during training",
    "reference_return": "Common-reference return during training",
    "cleared": "Clear rate during training",
    "loss": "Learning loss",
    "epsilon": "Exploration probability",
    "last_grad_norm": "Gradient norm before clipping",
    "last_td_error": "Absolute TD error",
    "last_entropy": "PPO action entropy",
    "last_policy_loss": "PPO policy loss",
    "last_value_loss": "PPO value loss",
    **{f"last_head{h}_loss": f"HRA head {h} loss" for h in range(3)},
    **{f"last_head{h}_q": f"HRA head {h} normalized Q" for h in range(3)},
}
EVAL_PLOTS = {
    "score_mean": "Game score (greedy evaluation)",
    "reference_return_mean": "Common-reference reward (greedy evaluation)",
    "return_mean": "Own-reward return (greedy evaluation)",
    "clear_rate": "Clear rate (greedy evaluation)",
    "lives_lost_mean": "Lives lost (greedy evaluation)",
    "pellet_fraction_mean": "Pellet fraction (greedy evaluation)",
    "ghosts_eaten_mean": "Ghosts eaten (greedy evaluation)",
    "power_points_mean": "Power-pellet points (greedy evaluation)",
    "blocked_action_rate": "Blocked action rate (greedy evaluation)",
}


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.unicode_minus": False, "figure.max_open_warning": 0})
    return plt


def seed_interval(values, resamples=500):
    """Resample independent training seeds, never treat evaluation episodes as seeds."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if len(values) < 2:
        return mean, None, None
    rng = np.random.default_rng(7301)
    samples = values[rng.integers(len(values), size=(resamples, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    return mean, low, high


def numeric_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{k: float(v) if v not in ("", None) and k not in ("ending",) else None
             for k, v in row.items() if k != "ending"} for row in rows]


def aligned(series, key):
    usable = [[r for r in rows if r.get(key) is not None] for rows in series]
    usable = [r for r in usable if r]
    if not usable:
        return None, None
    start = max(rows[0]["learning_episode"] for rows in usable)
    end = min(rows[-1]["learning_episode"] for rows in usable)
    if start > end:
        # Do not extrapolate sparse curves with no shared learning interval.
        return None, None
    points = sorted({r["learning_episode"] for rows in usable for r in rows if start <= r["learning_episode"] <= end})
    points = np.asarray(points)
    values = np.asarray([np.interp(points, [r["learning_episode"] for r in rows], [r[key] for r in rows])
                         for rows in usable])
    return points, values


def smooth(values, window):
    width = min(window, len(values))
    if width <= 1:
        return np.asarray(values)
    total = np.cumsum(np.insert(values, 0, 0))
    return np.asarray([(total[i + 1] - total[max(0, i + 1 - width)]) / min(i + 1, width)
                       for i in range(len(values))])


def finish_figure(fig, ax, path, title, ylabel, x_label="Learning episodes"):
    ax.set(xlabel=x_label, ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def report(out, plan=None, window=50):
    out = Path(out)
    plan = plan or json.loads((out / "plan.json").read_text(encoding="utf-8"))["plan"]
    conditions = {c["id"]: c for c in plan["conditions"]}
    result_index = {}
    for path in sorted((out / "runs").glob("*/result.json")):
        if (path.parent / "FAILED.json").exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        result_index.setdefault(digest(result["config"])[:20], []).append((path.parent, result))
    by_condition = {}
    summary_rows, episode_rows, reward_rows = [], [], []
    for condition_id, condition in conditions.items():
        matching = result_index.get(condition_id, [])
        if not matching:
            continue
        by_condition[condition_id] = matching
        evaluations = [r["evaluation"] for _, r in matching]
        row = dict(condition=condition_id, algorithm=condition["config"]["algorithm"], n_train_seeds=len(matching),
                   train_steps=condition["config"]["train"]["steps"],
                   learning_episodes_mean=float(np.mean([r["episodes_completed"] for _, r in matching])),
                   updates_mean=float(np.mean([r["updates"] for _, r in matching])),
                   parameter_count=matching[0][1]["parameter_count"],
                   wall_seconds_mean=float(np.mean([r["wall_seconds"] for _, r in matching])))
        for key in EVAL_PLOTS:
            values = [e[key] for e in evaluations if e.get(key) is not None]
            if values:
                mean, lo, hi = seed_interval(values)
                row.update({key: float(mean), key + "_seed_ci_low": float(lo) if lo is not None else None,
                            key + "_seed_ci_high": float(hi) if hi is not None else None})
        final_evals = []
        for folder, result in matching:
            episode_rows.extend(dict(condition=condition_id, train_seed=result["train_seed"], **r)
                                for r in result["evaluation"]["episodes"])
            final_path = folder / "evaluation_final.json"
            if final_path.exists():
                final_evals.append(json.loads(final_path.read_text(encoding="utf-8")))
        for key in EVAL_PLOTS:
            values = [e[key] for e in final_evals if e.get(key) is not None]
            if values:
                row["final_" + key] = float(np.mean(values))
        summary_rows.append(row)
        reward_rows.append(dict(condition=condition_id, algorithm=condition["config"]["algorithm"],
                                **condition["config"]["rewards"], gamma=condition["config"]["agent"]["gamma"],
                                normalization=condition["config"]["train"]["reward_normalization"],
                                reward_clip=condition["config"]["train"]["reward_clip"]))
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "evaluation_episodes.csv", episode_rows)
    write_csv(out / "reward_parameters.csv", reward_rows)
    atomic_json(out / "summary.json", summary_rows)
    plt = pyplot()
    images, plotted = [], []
    pairs = []
    for group, members in plan["groups"].items():
        completed = [m for m in members if m["condition"] in by_condition]
        if len(completed) < min(2, len(members)):
            continue
        # Paired differences use independent training seeds as the unit of replication.
        if completed and members[0]["condition"] in by_condition:
            reference = by_condition[members[0]["condition"]]
            ref_by_seed = {r["train_seed"]: r for _, r in reference}
            for member in (m for m in completed if m["condition"] != members[0]["condition"]):
                candidate = {r["train_seed"]: r for _, r in by_condition[member["condition"]]}
                shared = sorted(set(ref_by_seed) & set(candidate))
                for metric in ("score_mean", "reference_return_mean", "clear_rate"):
                    diffs = [candidate[s]["evaluation"][metric] - ref_by_seed[s]["evaluation"][metric] for s in shared]
                    if diffs:
                        mean, lo, hi = seed_interval(diffs)
                        pairs.append(dict(group=group, condition=member["condition"],
                                          baseline=members[0]["condition"], metric=metric,
                                          n_train_seeds=len(shared), diff=float(mean),
                                          seed_ci_low=float(lo) if lo is not None else None,
                                          seed_ci_high=float(hi) if hi is not None else None))
        for page_start in range(0, len(completed), 8):
            page = completed[page_start:page_start + 8]
            suffix = f"_part{page_start // 8 + 1}"
            for source, metrics in (("train", TRAIN_PLOTS), ("evaluation", EVAL_PLOTS), ("heldout", EVAL_PLOTS)):
                series_by_member = []
                for member in page:
                    series = []
                    for folder, result in by_condition[member["condition"]]:
                        if source == "train":
                            rows = numeric_rows(folder / "train.csv")
                        elif source == "evaluation":
                            rows = numeric_rows(folder / "learning_curve.csv")
                            final = dict(learning_episode=result["episodes_completed"],
                                         **{k: v for k, v in result["evaluation"].items() if k not in ("episodes", "phase")})
                            if not rows or rows[-1]["learning_episode"] < final["learning_episode"]:
                                rows.append(final)
                        else:
                            path = folder / "evaluation_final.json"
                            final = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                            rows = [dict(learning_episode=result["episodes_completed"],
                                         **{k:v for k,v in final.items() if k not in ("episodes", "phase")})] if final else []
                        series.append(rows)
                    series_by_member.append((member, series))
                for key, ylabel in metrics.items():
                    fig, ax = plt.subplots(figsize=(11, 5))
                    any_points = False
                    for member, series in series_by_member:
                        if source == "heldout":
                            points = [rows[0] for rows in series if rows and rows[0].get(key) is not None]
                            if points:
                                mean, lo, hi = seed_interval([r[key] for r in points])
                                xpoint = float(np.mean([r["learning_episode"] for r in points]))
                                line, = ax.plot([xpoint], [mean], marker="o", linestyle="none",
                                                label=f"{member['label']} (n={len(points)})")
                                if lo is not None:
                                    ax.vlines(xpoint, lo, hi, color=line.get_color())
                                any_points = True
                            continue
                        x, values = aligned(series, key)
                        label = member["label"]
                        if x is None:
                            # Sparse evaluation samples still have their actual learning-episode x coordinate.
                            for seed_i, rows in enumerate(series):
                                points = [r for r in rows if r.get(key) is not None]
                                if points:
                                    ax.plot([r["learning_episode"] for r in points], [r[key] for r in points],
                                            marker=".", alpha=0.65, label=label + f" seed#{seed_i+1}")
                                    any_points = True
                            continue
                        if source == "train" and key != "epsilon":
                            values = np.asarray([smooth(v, window) for v in values])
                        mean, lo, hi = seed_interval(values)
                        line, = ax.plot(x, mean, marker="." if source == "evaluation" else None,
                                        label=f"{label} (n={len(values)})")
                        if lo is not None:
                            ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.12)
                        any_points = True
                    if not any_points:
                        plt.close(fig)
                        continue
                    path = out / "figures" / group / (source + "_" + key + suffix + ".png")
                    finish_figure(fig, ax, path, f"{group}: {source}", ylabel)
                    images.append(str(path.relative_to(out)))
                    plotted.append(dict(path=str(path.relative_to(out)), x_axis="learning_episode",
                                        source=source, metric=key))
    write_csv(out / "paired_differences.csv", pairs)
    atomic_json(out / "figure_manifest.json", plotted)
    links = "\n".join(f"- [{p}]({p})" for p in images)
    (out / "REPORT.md").write_text(
        "# Pac-Man comparison report\n\n"
        "All PNG horizontal axes are learning episodes. Training curves include exploration; "
        "evaluation curves use greedy policies. Environment steps and gradient updates are separate CSV columns.\n\n"
        "Bands are exploratory 95% bootstrap intervals over training seeds. Single-seed curves have no band. "
        "Sparse curves are interpolated only inside their shared learning-episode range; no extrapolation. "
        "Many comparisons are exploratory and do not establish a confirmatory winner.\n\n"
        "Use game score and common-reference return across reward settings. Own-reward returns use different scales "
        "and are not a fair standalone ranking across reward definitions.\n\n"
        "The reference is the predeclared first member of each group. Failed/incomplete conditions are not ranked.\n\n"
        f"Completed conditions: {len(summary_rows)} / {len(conditions)}\n\n{links}\n", encoding="utf-8")
    return dict(completed_conditions=len(summary_rows), images=len(images))


def search_report(folder):
    folder = Path(folder)
    plt = pyplot()
    states = []
    for path in sorted(folder.glob("*/search_state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("history"):
            states.append((path.parent.name, state["history"]))
    if not states:
        return 0
    image_count = 0
    for metric in ("fitness", "best_fitness"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, rows in states:
            good = [r for r in rows if r.get(metric) is not None]
            ax.plot([r["cumulative_learning_episodes"] for r in good], [r[metric] for r in good],
                    marker=".", label=method)
        finish_figure(fig, ax, folder / "figures" / (metric + ".png"), "Search comparison", metric,
                      "Learning episodes (candidate training, cumulative)")
        image_count += 1
    parameter_names = sorted({k for _, rows in states for row in rows for k in row.get("parameters", {})})
    for parameter in parameter_names:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, rows in states:
            good = [r for r in rows if isinstance(r.get("parameters", {}).get(parameter), (int, float))]
            if good:
                ax.plot([r["cumulative_learning_episodes"] for r in good],
                        [r["parameters"][parameter] for r in good], marker=".", label=method)
        if ax.lines:
            finish_figure(fig, ax, folder / "figures" / (parameter.replace(".", "_") + ".png"),
                          "Search parameter: " + parameter, parameter,
                          "Learning episodes (candidate training, cumulative)")
            image_count += 1
        else:
            plt.close(fig)
    final_series = []
    for method_folder in sorted(folder.iterdir()):
        if not method_folder.is_dir():
            continue
        series = []
        for result_path in sorted((method_folder / "final" / "runs").glob("*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows = numeric_rows(result_path.parent / "learning_curve.csv")
            endpoint = dict(learning_episode=result["episodes_completed"],
                            **{k: v for k, v in result["evaluation"].items() if k not in ("episodes", "phase")})
            if not rows or rows[-1]["learning_episode"] < endpoint["learning_episode"]:
                rows.append(endpoint)
            series.append(rows)
        if series:
            final_series.append((method_folder.name, series))
    for metric, label in EVAL_PLOTS.items():
        if not final_series:
            break
        fig, ax = plt.subplots(figsize=(11, 5))
        for method, series in final_series:
            x, values = aligned(series, metric)
            if x is None:
                for index, rows in enumerate(series):
                    good = [r for r in rows if r.get(metric) is not None]
                    if good:
                        ax.plot([r["learning_episode"] for r in good], [r[metric] for r in good],
                                marker=".", label=f"{method} seed#{index+1}")
                continue
            mean, low, high = seed_interval(values)
            line, = ax.plot(x, mean, marker=".", label=f"{method} (n={len(values)})")
            if low is not None:
                ax.fill_between(x, low, high, color=line.get_color(), alpha=0.12)
        if ax.lines:
            finish_figure(fig, ax, folder / "figures" / ("refit_" + metric + ".png"),
                          "Selected configurations: independent retraining", label)
            image_count += 1
        else:
            plt.close(fig)
    for metric, label in EVAL_PLOTS.items():
        fig, ax = plt.subplots(figsize=(11, 5))
        for method_folder in sorted(folder.iterdir()):
            if not method_folder.is_dir():
                continue
            xs, ys = [], []
            for path in sorted((method_folder / "final" / "runs").glob("*/evaluation_final.json")):
                final = json.loads(path.read_text(encoding="utf-8"))
                result_path = path.parent / "result.json"
                if result_path.exists() and final.get(metric) is not None:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    xs.append(result["episodes_completed"])
                    ys.append(final[metric])
            if ys:
                mean, low, high = seed_interval(ys)
                err = [[float(mean - low)], [float(high - mean)]] if low is not None else None
                ax.errorbar([float(np.mean(xs))], [float(mean)], yerr=err, fmt="o", capsize=3,
                            label=f"{method_folder.name} (n={len(ys)})")
        if ax.lines:
            finish_figure(fig, ax, folder / "figures" / ("heldout_" + metric + ".png"),
                          "Held-out evaluation after independent retraining", label)
            image_count += 1
        else:
            plt.close(fig)
    return image_count
