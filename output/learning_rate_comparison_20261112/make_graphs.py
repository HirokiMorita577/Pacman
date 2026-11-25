"""Render presentation-sized learning-rate comparison figures from the CSV tables."""
from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
W, H = 1600 * S, 900 * S
ORDER = ("0.00001", "0.0001", "0.0005", "0.001", "0.003")
COLORS = {
    "0.00001": (41, 102, 173),
    "0.0001": (28, 145, 130),
    "0.0005": (226, 120, 28),
    "0.001": (126, 78, 170),
    "0.003": (192, 68, 77),
}
LABELS = {
    "0.00001": "1×10⁻⁵",
    "0.0001": "1×10⁻⁴",
    "0.0005": "5×10⁻⁴（基準）",
    "0.001": "1×10⁻³",
    "0.003": "3×10⁻³",
}


def font(size, bold=False):
    path = "C:/Windows/Fonts/meiryob.ttc" if bold else "C:/Windows/Fonts/meiryo.ttc"
    return ImageFont.truetype(path, size * S)


def centered(draw, xy, value, face, fill):
    box = draw.textbbox((0, 0), str(value), font=face)
    x, y = xy
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]),
              str(value), font=face, fill=fill)


def read_rows(filename):
    with (ROOT / filename).open(encoding="utf-8-sig", newline="") as f:
        return [{k: (v if k == "lr" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(f)]


def line_plot(filename, metric, title, ylabel, max_y, step_y, x_max, footer, output, percent=False):
    rows = read_rows(filename)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    left, right = 170 * S, 1530 * S
    top, bottom = 190 * S, 745 * S
    x_map = lambda x: left + x / x_max * (right - left)
    y_map = lambda y: bottom - y / max_y * (bottom - top)
    d.text((80 * S, 40 * S), title, font=font(34, True), fill=(31, 40, 48))
    for x in range(0, x_max + 1, 200):
        px = x_map(x)
        d.line([(px, top), (px, bottom)], fill=(226, 231, 235), width=2 * S)
        centered(d, (px, bottom + 30 * S), str(x), font(18), (69, 77, 84))
    for i in range(round(max_y / step_y) + 1):
        y = i * step_y
        py = y_map(y)
        d.line([(left, py), (right, py)], fill=(226, 231, 235), width=2 * S)
        label = f"{round(y * 100)}%" if percent else f"{round(y):,}"
        box = d.textbbox((0, 0), label, font=font(18))
        d.text((left - 20 * S - (box[2] - box[0]), py - (box[3] - box[1]) / 2 - box[1]),
               label, font=font(18), fill=(69, 77, 84))
    d.line([(left, top), (left, bottom), (right, bottom)], fill=(50, 59, 66), width=3 * S)
    centered(d, ((left + right) / 2, bottom + 74 * S), "学習エピソード数", font(21), (46, 54, 61))
    d.text((170 * S, 150 * S), ylabel, font=font(19), fill=(46, 54, 61))

    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    ci_prefix = "clear" if metric == "clear_rate" else metric
    for lr in ORDER:
        series = sorted((r for r in rows if r["lr"] == lr), key=lambda r: r["learning_episode"])
        low = [(x_map(r["learning_episode"]), y_map(r[ci_prefix + "_ci_low"])) for r in series]
        high = [(x_map(r["learning_episode"]), y_map(r[ci_prefix + "_ci_high"])) for r in series]
        bd.polygon(low + high[::-1], fill=(*COLORS[lr], 28))
    im = Image.alpha_composite(im.convert("RGBA"), band).convert("RGB")
    d = ImageDraw.Draw(im)
    for lr in ORDER:
        series = sorted((r for r in rows if r["lr"] == lr), key=lambda r: r["learning_episode"])
        pts = [(x_map(r["learning_episode"]), y_map(r[metric + "_mean"])) for r in series]
        d.line(pts, fill=COLORS[lr], width=5 * S, joint="curve")
        for p in pts[::max(1, len(pts) // 10)]:
            radius = 3 * S
            d.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius),
                      fill=COLORS[lr])
    lx, ly = 190 * S, 110 * S
    for i, lr in enumerate(ORDER):
        x = lx + i * 260 * S
        d.line([(x, ly), (x + 44 * S, ly)], fill=COLORS[lr], width=6 * S)
        d.text((x + 51 * S, ly - 16 * S), "α=" + LABELS[lr], font=font(19), fill=(38, 46, 53))
    d.text((170 * S, 835 * S), footer, font=font(16), fill=(85, 93, 99))
    im.resize((1600, 900), Image.Resampling.LANCZOS).save(ROOT / output, optimize=True)


line_plot("training_curve_100ep.csv", "score",
          "Double DQN｜学習率 α と学習中のスコア",
          "直近100エピソードの平均ゲームスコア", 3500, 500, 1000,
          "各条件5学習seedの平均。学習中は探索あり。淡色はseed単位の探索的95% bootstrap区間。",
          "01_training_score.png")
line_plot("training_curve_100ep.csv", "clear_rate",
          "Double DQN｜学習率 α と学習中のクリア率",
          "直近100エピソードのクリア率", 1.0, 0.2, 1000,
          "各条件5学習seedの平均。学習中は探索あり。淡色はseed単位の探索的95% bootstrap区間。",
          "02_training_clear_rate.png", True)
line_plot("evaluation_curve.csv", "score",
          "Double DQN｜学習率 α と探索なし評価のスコア",
          "各時点の評価10回の平均ゲームスコア", 3500, 500, 1200,
          "各条件5学習seedの平均。線は評価10点を結ぶ。横軸はその時点の学習エピソード数のseed平均。",
          "03_evaluation_score.png")
line_plot("evaluation_curve.csv", "clear_rate",
          "Double DQN｜学習率 α と探索なし評価のクリア率",
          "各時点の評価10回のクリア率", 1.0, 0.2, 1200,
          "各条件5学習seedの平均。線は評価10点を結ぶ。横軸はその時点の学習エピソード数のseed平均。",
          "04_evaluation_clear_rate.png", True)


def heldout():
    data = {r["lr"]: r for r in read_rows("heldout_summary.csv")}
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((80 * S, 38 * S), "Double DQN｜学習率 α と最終モデルの未使用評価",
           font=font(33, True), fill=(31, 40, 48))
    panels = [
        (75 * S, 160 * S, 765 * S, 730 * S, "平均ゲームスコア",
         "score_mean", "score_ci_low", "score_ci_high", 3000, False),
        (840 * S, 160 * S, 1530 * S, 730 * S, "クリア率",
         "clear_rate_mean", "clear_ci_low", "clear_ci_high", 1.0, True),
    ]
    for left, top, right, bottom, title, key, low, high, ymax, pct in panels:
        d.text((left, top - 43 * S), title, font=font(23, True), fill=(43, 52, 59))
        pl, pr = left + 67 * S, right - 25 * S
        pt, pb = top + 20 * S, bottom - 92 * S
        for i in range(6):
            val = i * ymax / 5
            yy = pb - val / ymax * (pb - pt)
            d.line([(pl, yy), (pr, yy)], fill=(226, 231, 235), width=2 * S)
            label = f"{val * 100:.0f}%" if pct else f"{val:,.0f}"
            d.text((left, yy - 10 * S), label, font=font(15), fill=(75, 82, 89))
        for i, lr in enumerate(ORDER):
            row = data[lr]
            cx = pl + (i + 0.5) * (pr - pl) / 5
            bw = 70 * S
            yv = pb - row[key] / ymax * (pb - pt)
            d.rectangle([(cx - bw / 2, yv), (cx + bw / 2, pb)], fill=COLORS[lr])
            ylo = pb - row[low] / ymax * (pb - pt)
            yhi = pb - row[high] / ymax * (pb - pt)
            d.line([(cx, ylo), (cx, yhi)], fill=(35, 39, 43), width=3 * S)
            d.line([(cx - 9 * S, ylo), (cx + 9 * S, ylo)], fill=(35, 39, 43), width=3 * S)
            d.line([(cx - 9 * S, yhi), (cx + 9 * S, yhi)], fill=(35, 39, 43), width=3 * S)
            value = f"{row[key] * 100:.1f}%" if pct else f"{row[key]:,.0f}"
            centered(d, (cx, yhi - 25 * S), value, font(17, True), (35, 42, 48))
            centered(d, (cx, pb + 34 * S), LABELS[lr].replace("（基準）", ""),
                     font(16), (49, 56, 63))
    d.text((80 * S, 815 * S),
           "各条件5学習seed × 未使用の最終評価100回。黒線はseed単位の探索的95% bootstrap区間。",
           font=font(17), fill=(85, 93, 99))
    im.resize((1600, 900), Image.Resampling.LANCZOS).save(
        ROOT / "05_heldout_summary.png", optimize=True)


heldout()
print("PLOTS_READY", *(p.name for p in sorted(ROOT.glob("*.png"))))
