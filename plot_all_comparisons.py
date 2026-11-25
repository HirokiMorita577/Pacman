"""Build an indexed set of final-evaluation and learning-curve comparisons."""
import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

from experiments.catalog import catalog
from experiments.config import HEURISTICS, digest
from experiments.report import aligned


def read_batches(source, pattern):
    return {row["trial"]: row for path in source.glob(pattern)
            for row in json.loads(path.read_text(encoding="utf-8"))}


def collect(plan, runtime, finals, curves):
    results = {}
    for condition in plan["conditions"]:
        config = condition["config"]
        algorithm = config["algorithm"]
        execution = runtime["execution"]
        if algorithm in HEURISTICS:
            execution = {key: execution[key] for key in ("torch_threads", "deterministic")}
            execution["device"] = "cpu"
        result = {"algorithm": algorithm, "finals": [], "curves": []}
        for seed in plan["seeds"]:
            manifest = {**runtime, "execution": execution, "schema": 1, "config": config, "seed": seed}
            name = f"{algorithm}_{digest(manifest)[:20]}_s{seed}"
            final = finals.get(name, {})
            if (final.get("phase") == "final" and final.get("seed") == seed
                    and final.get("episodes") == config["evaluation"]["episodes"]):
                result["finals"].append(final)
            curve = curves.get(name, {}).get("curve")
            if curve:
                result["curves"].append([
                    {"learning_episode": episode, "score": score, "clear": clear}
                    for episode, score, clear in curve
                ])
        results[condition["id"]] = result
    return results


METRICS = (("clear", "クリア率（%）", 100), ("score", "平均ゲームスコア（点）", 1))


def draw_final(path, title, members, count):
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.3, 1.65 + len(members) * 0.34)), sharey=True)
    for ax, (metric, label, scale) in zip(axes, METRICS):
        for i, (name, result) in enumerate(members):
            values = [row[metric] * scale for row in result["finals"]]
            if values:
                jitters = [(j - (len(values) - 1) / 2) * 0.065 for j in range(len(values))]
                ax.scatter(values, [i + delta for delta in jitters], color="#7399bb", alpha=0.68, s=17)
                if len(values) == count:
                    ax.scatter(mean(values), i, color="#123e6a", marker="D", s=27, zorder=4)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0, right=102 if metric == "clear" else None)
    axes[0].set_yticks(range(len(members)), [f"{label}   [{len(r['finals'])}/{count}]" for label, r in members])
    axes[0].set_ylim(len(members) - 0.5, -0.5)
    fig.suptitle(title + "｜最終評価", weight="bold")
    fig.text(0.5, 0.012, "点は試行別の値。ひし形は全予定試行が揃った条件の平均。[完了試行数 / 予定試行数]",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def draw_curves(path, title, members, count):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    handles, labels = [], []
    for i, (label, result) in enumerate(members):
        curves = result["curves"]
        if len(curves) != count:
            continue
        color = matplotlib.colormaps["tab20"](i % 20)
        for ax, (metric, metric_label, scale) in zip(axes, METRICS):
            x, values = aligned(curves, metric)
            if x is None:
                continue
            line, = ax.plot(x, values.mean(axis=0) * scale, color=color, marker=".", markersize=3,
                            linestyle="--" if i >= 20 else "-")
            if metric == "clear":
                handles.append(line)
                labels.append(label)
    for ax, (metric, metric_label, _) in zip(axes, METRICS):
        ax.set(xlabel="学習エピソード数", ylabel=metric_label)
        ax.grid(alpha=0.25)
        if metric == "clear":
            ax.set_ylim(-2, 102)
        if not handles:
            ax.text(0.5, 0.5, "全seedが揃う共通の曲線範囲はありません", transform=ax.transAxes, ha="center")
    fig.suptitle(title + "｜学習途中の探索なし評価", weight="bold")
    if handles:
        fig.legend(handles, labels, ncol=min(4, len(handles)), loc="lower center", bbox_to_anchor=(0.5, 0.055), fontsize=8)
    fig.text(0.5, 0.015, "全予定seedの曲線が揃う条件のみ。各seedに共通するエピソード範囲で補間し、平均。",
             ha="center", fontsize=9)
    legend_space = 0.13 + 0.04 * ((len(handles) + 3) // 4)
    fig.tight_layout(rect=(0, legend_space, 1, 0.93))
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return len(handles)


def draw_tradeoff(path, results, count):
    fig, ax = plt.subplots(figsize=(8.5, 5.4), layout="constrained")
    complete = {key: row["finals"] for key, row in results.items() if len(row["finals"]) == count}
    positions = {key: (mean(r["score"] for r in rows), 100 * mean(r["clear"] for r in rows))
                 for key, rows in complete.items()}
    ax.scatter(*zip(*positions.values()), color="#4878a7", s=24, alpha=0.44)
    highlights = (
        ("5df6465246ea3a1261d3", "安全経路", (6, 8)),
        ("335ca7fecf1a6af6a395", "接近報酬 7.5", (8, 9)),
        ("3203abe33170c9667ce5", "パワー報酬10・γ=0.995", (-135, -18)),
        ("3b898ced50c16dab4474", "Double DQN 基準", (8, -16)),
    )
    for key, text, offset in highlights:
        if key in positions:
            x, y = positions[key]
            ax.scatter(x, y, color="#b6422d", s=47, zorder=4)
            ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set(title=f"最終評価のスコアとクリア率（{len(positions)}完了条件）",
           xlabel="平均ゲームスコア（点）", ylabel="平均クリア率（%）", ylim=(-3, 105))
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_power_gamma(path, plan, results, count):
    configs = {condition["id"]: condition["config"] for condition in plan["conditions"]}
    cells = {}
    for entry in plan["groups"]["power_gamma"]:
        key = entry["condition"]
        config = configs[key]
        cells[config["rewards"]["power"], config["agent"]["gamma"]] = results[key]["finals"]
    powers = sorted({power for power, _ in cells})
    gammas = sorted({gamma for _, gamma in cells})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), layout="constrained")
    for ax, (metric, label, scale) in zip(axes, METRICS):
        matrix = np.full((len(powers), len(gammas)), np.nan)
        for i, power in enumerate(powers):
            for j, gamma in enumerate(gammas):
                rows = cells.get((power, gamma), [])
                if len(rows) == count:
                    matrix[i, j] = mean(row[metric] for row in rows) * scale
        color = ax.imshow(matrix, cmap="Blues", vmin=0 if metric == "clear" else None,
                          vmax=100 if metric == "clear" else None)
        low, high = color.get_clim()
        for (i, j), value in np.ndenumerate(matrix):
            text = "—" if np.isnan(value) else f"{value:.1f}%" if metric == "clear" else f"{value:,.0f}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if value > (low + high) / 2 else "#172741")
        ax.set(title=label, xlabel="割引率 γ", ylabel="パワーえさ報酬")
        ax.set_xticks(range(len(gammas)), [f"{value:g}" for value in gammas])
        ax.set_yticks(range(len(powers)), [f"{value:g}" for value in powers])
    fig.suptitle("パワーえさ報酬 × 割引率｜Double DQN の最終評価", weight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_index(out, rows, results, planned):
    complete = sum(len(result["finals"]) == planned for result in results.values())
    total = sum(len(result["finals"]) for result in results.values())
    trials = len(results) * planned
    local = ""
    if (out / "local_checkpoint_evaluation.png").is_file() and (out / "local_checkpoint_evaluation.csv").is_file():
        local = ('<p>ローカル実験の途中モデル8件の開発評価 · '
                 '<a href="local_checkpoint_evaluation.png">グラフ</a> · '
                 '<a href="local_checkpoint_evaluation.csv">評価 CSV</a></p>')
    has_loss = (out / "loss_coverage.csv").is_file()
    decay_link = ('<p><a href="alpha_epsilon_decay.png">学習率と探索率を0まで下げる設定案</a></p>'
                  if (out / "alpha_epsilon_decay.png").is_file() else "")
    has_episode_means = (out / "condition_averages.csv").is_file()
    loss_intro = ""
    if has_loss:
        loss_intro = ('<p>学習損失 · <a href="loss_coverage.csv">条件ごとの損失ログ件数</a>')
        if (out / "local_training_loss.png").is_file():
            loss_intro += ' · <a href="local_training_loss.png">ローカル試行の損失曲線</a>'
        if has_episode_means:
            loss_intro += ' · <a href="condition_averages.csv">条件ごとの平均値</a>'
        loss_intro += "</p>"
    loss_header = "<th>学習損失</th>" if has_loss else ""
    episode_header = "<th>各エピソードの平均</th>" if has_episode_means else ""
    links = "\n".join(
        f'<tr><th scope="row">{html.escape(title)} <small>{html.escape(name)}</small></th>'
        f'<td>{n_final}/{n_members}</td><td>{n_curve}/{n_members}</td>'
        f'<td><a href="figures/{name}_final.png">最終評価</a></td>'
        f'<td><a href="figures/{name}_curve.png">学習曲線</a></td>'
        + (f'<td><a href="figures/{name}_loss.png">損失</a></td>' if has_loss else "")
        + (f'<td><a href="figures/{name}_episode_mean.png">学習時の平均</a></td>' if has_episode_means else "")
        + '</tr>'
        for name, title, n_final, n_curve, n_members in rows
    )
    (out / "index.html").write_text(f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>パックマン比較実験の全グラフ</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:72rem;color:#172741}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #dce4ee}}
small{{display:block;color:#4b617b;font-weight:normal}}thead{{background:#eaf1f8}} img{{max-width:100%}}
</style>
<h1>パックマン比較実験の全グラフ</h1>
<p>2026年11月25日 JST の Colab 保存データ。{len(rows)}比較項目、{trials:,}予定試行のうち
最終評価 {total:,}件、学習seedが揃った条件 {complete}/{len(results)}件。</p>
<p><a href="score_clear_tradeoff.png">スコア対クリア率</a> · <a href="power_gamma_heatmap.png">パワーえさ報酬×割引率</a> · <a href="final_trial_metrics.csv">試行別の最終評価 CSV</a></p>
{local}
{loss_intro}
{decay_link}
<p>各比較項目の分母は予定条件数。完了数は全予定seedに最終評価がある条件、
曲線数は全予定seedに共通する学習エピソード範囲がある条件。条件は複数項目に登場する場合があります。</p>
<table><thead><tr><th>比較項目</th><th>完了条件</th><th>曲線条件</th><th>スコア・クリア率</th><th>途中評価</th>{loss_header}{episode_header}</tr></thead>
<tbody>{links}</tbody></table></html>""", encoding="utf-8")
    with (out / "final_trial_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["condition", "algorithm", "trial", "seed", "phase", "score", "clear", "episodes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, result in results.items():
            writer.writerows(dict(condition=key, algorithm=result["algorithm"], **r) for r in result["finals"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source, out = args.source, args.out
    document = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    plan = document["plan"]
    finals = read_batches(source, "rank_values_*.json")
    curves = read_batches(source, "curve_values_*.json")
    results = collect(plan, document["runtime"], finals, curves)
    if sum(len(result["finals"]) for result in results.values()) != len(finals):
        raise ValueError("Some final evaluations could not be matched to the frozen plan")
    if any("error" in row for row in curves.values()):
        raise ValueError("Some learning curves could not be downloaded")
    font = Path("C:/Windows/Fonts/meiryo.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    planned = len(plan["seeds"])
    headings = catalog()
    titles = {"reward_power": "パワーえさの学習用報酬", "reward_approach_dot": "えさへの接近報酬"}
    index = []
    for name, group in plan["groups"].items():
        title = titles.get(name, headings.get(name, {}).get("description", name))
        members = [(entry["label"], results[entry["condition"]]) for entry in group]
        draw_final(figures / f"{name}_final.png", title, members, planned)
        n_curve = draw_curves(figures / f"{name}_curve.png", title, members, planned)
        n_final = sum(len(result["finals"]) == planned for _, result in members)
        index.append((name, title, n_final, n_curve, len(members)))
    draw_tradeoff(out / "score_clear_tradeoff.png", results, planned)
    if "power_gamma" in plan["groups"]:
        draw_power_gamma(out / "power_gamma_heatmap.png", plan, results, planned)
    write_index(out, index, results, planned)
    print(f"{len(index)} comparisons, {len(finals)} final evaluations, {len(curves)} curve files")


if __name__ == "__main__":
    main()
