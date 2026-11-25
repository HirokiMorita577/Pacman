"""Combine verified DQN and SARSA runs into slide-ready comparison tables."""
from pathlib import Path
import csv
import itertools
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = Path("C:/PRO/Pacman")
PREVIOUS = PROJECT / "output/dqn_vs_double_dqn_20261111"
SARSA = PROJECT / "slide_work/sarsa_update_20261111/source"
SEEDS = (101, 202, 303, 404, 505)
FOLDERS = {
    101: "1XxDMhYm_9VgSIC0PcPntkW1OHnhbvXsX",
    202: "1cF26Gub2AGjT8sM9_GDUDy1Fka4cCuOr",
    303: "14r9pmB2malSa_yG_x4xFWduk8gzUUF5n",
    404: "1-Uv31rb9IXfH4TrjRcDgqaG6zLHaysgr",
    505: "1246TdzCVTa_qDAu1V4UDqby7tGLu3EDJ",
}
SOURCE_FILES = {
    101: ("1v8VKtr5wboSJ9iiwxRIQmOREwQTDBv2t", "1FOR5xrjFWAhXn_fBpd63gzit5mjEbpFf",
          "1i3z9QfA9cdXBvIQ6wMQg7F8WmgYHGhGP", "1Ssg2QjV9YTSz0YHV4c0st2OnDqFDuP6U"),
    202: ("1dY7__coFtBfBw-FAUplP6hhf4OC2k0lt", "1ZuCUnwKm9Xs7p_h0Zv36l61rmbHsZLSG",
          "1f_6ogwss45pszhMfynFui1UcFTNPppvI", "1O1fQJxJg8X9oQsp64WFsLIv1gktHqShA"),
    303: ("1aZSqmbs0ZEgthDFu9nLziUTozwzMVFwz", "14ZmPEfHJ7ISv9gWw2hUw_dJrQqTwUY1A",
          "15M_5WAaLITtINN3VJMHHzD3BGVNGuaKY", "1FavtQHHjdbQm3xnHN7vRX2HNob-dGKPh"),
    404: ("1N-uKWXzVJrspp5VklQscoBF3ZsCy2vX5", "1Oa2d5kEyNipO514PnKqhpogpZ1JiOZ7k",
          "1LRqhsMt907TR61NdoUTKiNS_iuNPd3K3", "1uJj6vN7XimSn1KHLMixrxqfJw3FEF28N"),
    505: ("1ZrPWdnBZ56TiaJZPA_lNBnZZUXtJ-8FG", "1uDBXGJMIFn_Qi1TjA2VRZL1pal8bFwLp",
          "1Glb61RVzV933xdypBiC2qZc_mSN2ULB5", "1ktDvrvK6dgR1vcTwo_7mDUzi4Se6S6De"),
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, keys):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stat(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (5,)
    samples = np.array([values[list(indices)].mean()
                        for indices in itertools.product(range(5), repeat=5)])
    return float(values.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


metric_keys = list(read_csv(PREVIOUS / "heldout_seed_metrics.csv")[0])
dqn_metrics = [r for r in read_csv(PREVIOUS / "heldout_seed_metrics.csv")
               if r["algorithm"] == "dqn"]
assert len(dqn_metrics) == 5
sarsa_metrics = []
sarsa_curves = []
for seed in SEEDS:
    folder = SARSA / str(seed)
    run = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    final = json.loads((folder / "evaluation_final.json").read_text(encoding="utf-8"))
    assert run["config"]["algorithm"] == "sarsa"
    assert run["config"]["agent"]["lr"] == 0.0005
    assert run["train_seed"] == seed and run["total_steps"] == 200000
    assert run["model_sha256"] == final["model_sha256"]
    assert final["train_seed"] == seed and len(final["episodes"]) == 100
    assert sorted(e["evaluation_seed"] for e in final["episodes"]) == list(range(1000000, 1000100))
    sarsa_metrics.append(dict(
        algorithm="sarsa", train_seed=seed, train_steps=run["total_steps"],
        learning_episodes=run["episodes_completed"], heldout_episodes=100,
        score_mean=final["score_mean"], clear_rate=final["clear_rate"],
        reference_return_mean=final["reference_return_mean"],
        lives_lost_mean=final["lives_lost_mean"],
        pellet_fraction_mean=final["pellet_fraction_mean"],
        ghosts_eaten_mean=final["ghosts_eaten_mean"],
        power_points_mean=final["power_points_mean"],
        wall_seconds=run["wall_seconds"], trial_id=run["id"]))
    curve = read_csv(folder / "learning_curve.csv")
    assert len(curve) == 10
    sarsa_curves.append(curve)
write_csv(ROOT / "heldout_seed_metrics.csv", dqn_metrics + sarsa_metrics, metric_keys)

summary = []
for algorithm, rows in (("dqn", dqn_metrics), ("sarsa", sarsa_metrics)):
    score = stat([float(r["score_mean"]) for r in rows])
    clear = stat([float(r["clear_rate"]) for r in rows])
    summary.append(dict(
        algorithm=algorithm, n_train_seeds=5, heldout_episodes_per_seed=100,
        heldout_score_mean=score[0], score_ci_low=score[1], score_ci_high=score[2],
        heldout_clear_rate=clear[0], clear_ci_low=clear[1], clear_ci_high=clear[2]))
write_csv(ROOT / "heldout_summary.csv", summary, list(summary[0]))

paired = []
for seed in SEEDS:
    dqn = next(r for r in dqn_metrics if int(r["train_seed"]) == seed)
    sarsa = next(r for r in sarsa_metrics if int(r["train_seed"]) == seed)
    paired.append((float(dqn["score_mean"]) - float(sarsa["score_mean"]),
                   float(dqn["clear_rate"]) - float(sarsa["clear_rate"])))
score_difference = stat([v[0] for v in paired])
clear_difference = stat([v[1] for v in paired])
write_csv(ROOT / "paired_difference.csv", [dict(
    comparison="DQN minus SARSA",
    score_difference=score_difference[0],
    score_ci_low=score_difference[1], score_ci_high=score_difference[2],
    clear_rate_difference=clear_difference[0],
    clear_ci_low=clear_difference[1], clear_ci_high=clear_difference[2])],
    ["comparison", "score_difference", "score_ci_low", "score_ci_high",
     "clear_rate_difference", "clear_ci_low", "clear_ci_high"])

train = [r for r in read_csv(PREVIOUS / "training_curve_100ep.csv")
         if r["algorithm"] == "dqn"]
train += read_csv(ROOT / "sarsa_training_curve_100ep.csv")
write_csv(ROOT / "training_curve_100ep.csv", train, list(train[0]))

evaluation = [r for r in read_csv(PREVIOUS / "evaluation_curve.csv")
              if r["algorithm"] == "dqn"]
for episode in np.arange(370, 1761, 20):
    vals = {}
    for metric in ("score_mean", "clear_rate"):
        vals[metric] = stat([
            np.interp(episode,
                      [float(row["learning_episode"]) for row in curve],
                      [float(row[metric]) for row in curve])
            for curve in sarsa_curves])
    evaluation.append(dict(
        algorithm="sarsa", learning_episode=int(episode),
        score_mean=vals["score_mean"][0],
        score_ci_low=vals["score_mean"][1],
        score_ci_high=vals["score_mean"][2],
        clear_rate_mean=vals["clear_rate"][0],
        clear_ci_low=vals["clear_rate"][1],
        clear_ci_high=vals["clear_rate"][2]))
write_csv(ROOT / "evaluation_curve.csv", evaluation, list(evaluation[0]))

sources = [r for r in read_csv(PREVIOUS / "source_links.csv")
           if r["algorithm"] == "dqn"]
for metric in sarsa_metrics:
    seed = int(metric["train_seed"])
    files = ["https://drive.google.com/file/d/" + id + "/view" for id in SOURCE_FILES[seed]]
    sources.append(dict(algorithm="sarsa", train_seed=seed,
                        trial_folder="https://drive.google.com/drive/folders/" + FOLDERS[seed],
                        train_csv=files[0], learning_curve_csv=files[1],
                        evaluation_final_json=files[2], result_json=files[3]))
write_csv(ROOT / "source_links.csv", sources, list(sources[0]))
print("VERIFIED", len(dqn_metrics), "DQN +", len(sarsa_metrics), "SARSA runs")
print("SUMMARY", summary)
