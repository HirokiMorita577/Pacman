"""Attach the reward context to comparison figures and export exact settings."""
import argparse
import csv
import json
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

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
