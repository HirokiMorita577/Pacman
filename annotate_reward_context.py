"""Attach the reward context to comparison figures and export exact settings."""
import argparse
import csv
import html
import json
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from experiments.catalog import catalog
from experiments.config import DEFAULT, HEURISTICS
from pacman_env import REWARD_SIGN, REWARD_SPEC


LABELS = {
    "dot": "通常えさ", "power": "パワーえさ", "eat_ghost": "ゴースト捕食", "death": "捕獲",
    "clear": "クリア", "approach_dot": "えさに接近", "leave_dot": "えさから離れる",
    "ghost_adjacent": "ゴースト隣接", "ghost_near": "ゴースト接近", "chase_blue": "青ゴースト追跡",
    "loop": "同じマスに滞在", "oscillation": "往復", "heading": "同方向", "new_cell": "新しいマス",
    "dot_progress": "えさ進捗", "time": "1ステップ",
}


def signed(key, weight):
    return f"{weight * REWARD_SIGN[key]:+g}" if weight else "0"


def reward_context(configs):
    base = DEFAULT["rewards"]
    values = {k: sorted({cfg["rewards"][k] for cfg in configs}) for k in REWARD_SPEC}
    changed = {k: v for k, v in values.items() if v != [base[k]]}
    notes = []
    if not changed:
        notes.append("学習報酬は全条件共通。スコアはゲーム内得点。")
    elif len(changed) <= 3:
        items = ["{} {}".format(LABELS[k], " / ".join(signed(k, weight) for weight in weights))
                 for k, weights in changed.items()]
        notes.append("比較で変えた設定  " + " ｜ ".join(items))
    else:
        notes.append("複数の報酬を変更。条件ラベルごとの全16項目は reward_settings.csv を参照。")
    norms = sorted({cfg["train"]["reward_normalization"] for cfg in configs})
    if norms != ["pellet"]:
        notes.append("正規化設定  " + " / ".join(norms) + "（pellet＝通常えさの報酬値で正規化）")
    clips = {cfg["train"]["reward_clip"] for cfg in configs}
    if clips != {None}:
        notes.append("報酬クリップ  " + " / ".join(sorted("なし" if v is None else f"{v:g}" for v in clips)))
    if any(cfg["algorithm"] in HEURISTICS for cfg in configs):
        notes.append("安全経路・最寄りえさ・ランダムなどの基準方式は学習なし。")
    return notes


def annotate(path, notes):
    with Image.open(path) as original:
        height = int(original.info.get("pacman_reward_context_base_height", original.height))
        image = original.crop((0, 0, original.width, height)).convert("RGB")
    fontsize = max(13, round(image.width / 85))
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", fontsize)
    width = max(35, int((image.width - 48) / fontsize))
    base = DEFAULT["rewards"]
    major = " ｜ ".join(f"{LABELS[k]} {signed(k, base[k])}" for k in ("dot", "power", "eat_ghost", "death", "clear"))
    text = [f"基準の学習報酬の重み  {major}", *notes, "全16項目の設定値は報酬設定表を参照。"]
    lines = [part for line in text for part in wrap(line, width=width, break_on_hyphens=False)]
    leading = int(fontsize * 1.55)
    result = Image.new("RGB", (image.width, height + len(lines) * leading + 26), "#f0f4f9")
    result.paste(image, (0, 0))
    draw = ImageDraw.Draw(result)
    draw.line((0, height, image.width, height), fill="#a8b9cd", width=2)
    for i, line in enumerate(lines):
        draw.text((20, height + 11 + leading * i), line, fill="#183455", font=font)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("pacman_reward_context_base_height", str(height))
    temporary = path.with_suffix(".tmp")
    result.save(temporary, format="PNG", pnginfo=meta)
    temporary.replace(path)


def learning_settings(cfg):
    agent, train = cfg["agent"], cfg["train"]
    schedule = {"exponential": "指数減衰", "linear": "線形減衰", "cosine": "コサイン減衰", "constant": "一定"}
    normalization = {"pellet": "通常えさの報酬で割る", "none": "なし", "fixed100": "100で割る"}
    return {
        "学習率 α": f"{agent['lr']:.12f}".rstrip("0").rstrip("."),
        "割引率 γ": f"{agent['gamma']:g}",
        "初期探索率 ε": f"{train['epsilon_start'] * 100:g}%",
        "最終探索率 ε": f"{train['epsilon_end'] * 100:g}%",
        "探索率の減らし方": schedule[train["schedule"]],
        "探索期間": f"{train['explore_steps']:,} 環境ステップ",
        "学習予算": f"{train['steps']:,} 環境ステップ",
        "報酬の正規化": normalization[train["reward_normalization"]],
        "報酬クリップ": "なし" if train["reward_clip"] is None else f"{train['reward_clip']:g}",
    }


def definition_list(rows):
    return "<dl>" + "".join(
        f"<div><dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd></div>" for key, value in rows.items()
    ) + "</dl>"


def write_reward_table(path, plan):
    configs = {row["id"]: row["config"] for row in plan["conditions"]}
    descriptions = catalog()
    algorithm_names = {"dqn": "DQN", "double_dqn": "Double DQN", "sarsa": "SARSA",
                       "replay_sarsa": "経験再生ありSARSA", "safe_food": "安全経路", "ppo": "PPO", "hra": "HRA"}
    sections = []
    for name, group in plan["groups"].items():
        trials = [configs[entry["condition"]] for entry in group]
        settings = [learning_settings(cfg) for cfg in trials]
        rewards = [{label: signed(key, cfg["rewards"][key]) for key, label in LABELS.items()} for cfg in trials]
        common_settings = {key: value for key, value in settings[0].items()
                           if all(row[key] == value for row in settings)}
        common_rewards = {key: value for key, value in rewards[0].items()
                          if all(row[key] == value for row in rewards)}
        main_rewards = {LABELS[key]: common_rewards[LABELS[key]]
                        for key in ("dot", "power", "eat_ghost", "death", "clear") if LABELS[key] in common_rewards}
        other_rewards = {key: value for key, value in common_rewards.items() if key not in main_rewards}
        cards = []
        for entry, cfg, setup, reward in zip(group, trials, settings, rewards):
            label = entry["label"].replace("reference", "基準", 1) if entry["label"].startswith("reference") else entry["label"]
            algorithm = algorithm_names.get(cfg["algorithm"], cfg["algorithm"])
            changes = {k: v for k, v in {**setup, **reward}.items()
                       if k not in common_settings and k not in common_rewards}
            cards.append(f"<article><h4>{html.escape(label)}</h4><p>{html.escape(algorithm)}</p>"
                         + (definition_list(changes) if changes else "")
                         + f"<details><summary>学習設定と報酬の全項目</summary>"
                         f"{definition_list(setup)}{definition_list(reward)}</details></article>")
        title = {"replay_td": "DQN・Double DQN・経験再生ありSARSAの比較",
                 "reward_power": "パワーえさ報酬の比較"}.get(name, descriptions.get(name, {}).get("description", name))
        sections.append(f"<section id='{html.escape(name, quote=True)}'><h2>{html.escape(title)}</h2>"
                        f"<h3>共通の学習設定</h3>{definition_list(common_settings)}"
                        f"<h3>共通の報酬設定</h3>{definition_list(main_rewards)}"
                        + (f"<details><summary>その他の共通報酬</summary>{definition_list(other_rewards)}</details>"
                           if other_rewards else "")
                        +
                        f"<h3>比較条件</h3><div class='cards'>{''.join(cards)}</div></section>")
    path.with_suffix(".html").write_text(
        """<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>パックマン比較実験の報酬設定</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:64rem;color:#172741}
section{margin:2.5rem 0;padding-top:1rem;border-top:1px solid #dce4ee;scroll-margin-top:1rem}
section:target h2{color:#00695c}
h2{line-height:1.35}h3{margin:1rem 0 .4rem}h4{margin:0;overflow-wrap:anywhere}
dl,.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,13rem),1fr));gap:.5rem;margin:.5rem 0 1rem}
dl div{background:#eef4fa;border-radius:.4rem;padding:.5rem .7rem}
dt{font-size:.88rem;color:#405978}dd{margin:0;font-weight:650;overflow-wrap:anywhere}
article{border:1px solid #dce4ee;border-radius:.6rem;padding:.75rem;min-width:0}
article p{margin:.15rem 0 .55rem;color:#405978}article dl{grid-template-columns:1fr}
summary{cursor:pointer;color:#15567e}details[open]{margin-bottom:.75rem}
</style><h1>パックマン比較実験の報酬設定</h1>
<p><a href="index.html">全グラフ一覧へ戻る</a> · <a href="reward_settings.csv">設定表のCSV</a></p>
<p>各比較の共通設定を最初に表示し、条件ごとに異なる学習設定と報酬を各カードに表示します。</p>
""" + "\n".join(sections) + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))["plan"]
    configs = {row["id"]: row["config"] for row in plan["conditions"]}
    with (args.out / "reward_settings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "label", "condition", "algorithm", *(f"{LABELS[k]} ({k})" for k in REWARD_SPEC),
                         "reward_normalization", "reward_clip"])
        for group, entries in plan["groups"].items():
            group_configs = [configs[entry["condition"]] for entry in entries]
            notes = reward_context(group_configs)
            for figure in (args.out / "figures").glob(group + "_*.png"):
                if figure.stem.removeprefix(group + "_") in ("final", "curve", "loss", "episode_mean"):
                    annotate(figure, notes)
            for entry, cfg in zip(entries, group_configs):
                writer.writerow([group, entry["label"], entry["condition"], cfg["algorithm"],
                                 *(signed(key, cfg["rewards"][key]) for key in REWARD_SPEC),
                                 cfg["train"]["reward_normalization"], cfg["train"]["reward_clip"]])
    write_reward_table(args.out / "reward_settings.csv", plan)
    for filename, note in {
        "score_clear_tradeoff.png": "条件ごとに報酬が異なる場合があります。各条件の設定は報酬設定表を参照。",
        "power_gamma_heatmap.png": "パワーえさ報酬を 0 / +10 / +50 / +200 に変更。横軸は割引率 γ。",
        "local_checkpoint_evaluation.png": "ローカル Double DQN の全試行は基準の報酬を使用。",
        "local_training_loss.png": "ローカル Double DQN の全試行は基準の報酬を使用。",
        "alpha_epsilon_decay.png": "学習率と探索率のみの設定案。報酬は基準の設定を想定。",
    }.items():
        figure = args.out / filename
        if figure.is_file():
            annotate(figure, [note])
    for directory in ("dqn_vs_double_dqn_20261111", "dqn_vs_sarsa_20261112", "dqn_double_reference_style_20261029"):
        for figure in (args.out.parent / directory).glob("*.png"):
            annotate(figure, reward_context([DEFAULT]))
    for figure in (args.out.parent / "reward_comparison_20261125").glob("*.png"):
        annotate(figure, ["左はパワーえさ報酬、中央はえさへの接近報酬を変更。右は各条件ラベルの設定を使用。"])


if __name__ == "__main__":
    main()
