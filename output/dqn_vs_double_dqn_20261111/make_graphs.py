from pathlib import Path
import csv
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/meiryo.ttc"
S = 2
W, H = 1600 * S, 900 * S
COLORS = {"dqn": (28, 112, 183), "double_dqn": (223, 91, 42)}
LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}

def font(size, bold=False):
    path = "C:/Windows/Fonts/meiryob.ttc" if bold else FONT
    return ImageFont.truetype(path, size * S)

def centered(draw, xy, value, f, fill):
    box = draw.textbbox((0, 0), str(value), font=f)
    x, y = xy
    draw.text((x - (box[2]-box[0])/2, y - (box[3]-box[1])/2 - box[1]),
              str(value), font=f, fill=fill)

def read_rows(filename):
    with (ROOT / filename).open(encoding="utf-8-sig", newline="") as f:
        return [{k: (v if k == "algorithm" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(f)]

def curve(filename, metric, title, ylabel, max_y, step_y, footer, output, percent=False):
    rows = read_rows(filename)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    left, right = 165*S, 1535*S
    top, bottom = 150*S, 745*S
    x_max = 1100
    X = lambda x: left + x / x_max * (right-left)
    Y = lambda y: bottom - y / max_y * (bottom-top)
    d.text((90*S, 42*S), title, font=font(35, True), fill=(32, 40, 48))
    for x in range(0, 1101, 200):
        px = X(x)
        d.line([(px, top), (px, bottom)], fill=(227, 231, 235), width=2*S)
        centered(d, (px, bottom+32*S), str(x), font(19), (70, 76, 84))
    n = round(max_y / step_y)
    for i in range(n+1):
        y = i * step_y
        py = Y(y)
        d.line([(left, py), (right, py)], fill=(227, 231, 235), width=2*S)
        label = f"{round(y*100)}%" if percent else f"{round(y):,}"
        box = d.textbbox((0,0), label, font=font(19))
        d.text((left-22*S-(box[2]-box[0]), py-(box[3]-box[1])/2-box[1]),
               label, font=font(19), fill=(70, 76, 84))
    d.line([(left,top),(left,bottom),(right,bottom)], fill=(55,61,68), width=3*S)
    centered(d, ((left+right)/2, bottom+78*S), "学習エピソード数", font(22), (48, 55, 62))
    d.text((165*S, 105*S), ylabel, font=font(20), fill=(48, 55, 62))
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    band_draw = ImageDraw.Draw(band)
    for alg in ("dqn", "double_dqn"):
        rr = sorted((r for r in rows if r["algorithm"] == alg), key=lambda r:r["learning_episode"])
        color = COLORS[alg]
        ci_prefix = "clear" if metric == "clear_rate" else metric
        lo = [(X(r["learning_episode"]), Y(r[ci_prefix+"_ci_low"])) for r in rr]
        hi = [(X(r["learning_episode"]), Y(r[ci_prefix+"_ci_high"])) for r in rr]
        band_draw.polygon(lo + hi[::-1], fill=(*color, 45))
    im = Image.alpha_composite(im.convert("RGBA"), band).convert("RGB")
    d = ImageDraw.Draw(im)
    for alg in ("dqn", "double_dqn"):
        rr = sorted((r for r in rows if r["algorithm"] == alg), key=lambda r:r["learning_episode"])
        color = COLORS[alg]
        pts = [(X(r["learning_episode"]), Y(r[metric+"_mean"])) for r in rr]
        d.line(pts, fill=color, width=5*S, joint="curve")
    lx, ly = 1120*S, 101*S
    for j, alg in enumerate(("dqn", "double_dqn")):
        y=ly+j*36*S
        d.line([(lx,y),(lx+45*S,y)], fill=COLORS[alg], width=5*S)
        d.text((lx+58*S,y-15*S), LABELS[alg], font=font(21), fill=(40,47,54))
    d.text((165*S, 835*S), footer, font=font(17), fill=(92, 98, 105))
    im.resize((1600,900), Image.Resampling.LANCZOS).save(ROOT/output, optimize=True)

curve("training_curve_100ep.csv", "score",
      "DQN と Double DQN｜学習中のスコア", "直近100エピソードの平均ゲームスコア",
      3000, 500, "5学習seedの平均。学習中は探索あり。淡色はseed単位の探索的95% bootstrap区間。",
      "01_training_score.png")
curve("training_curve_100ep.csv", "clear_rate",
      "DQN と Double DQN｜学習中のクリア率", "直近100エピソードのクリア率",
      1.0, 0.2, "5学習seedの平均。学習中は探索あり。淡色はseed単位の探索的95% bootstrap区間。",
      "02_training_clear_rate.png", True)
curve("evaluation_curve.csv", "score",
      "DQN と Double DQN｜探索なし評価のスコア", "評価10回の平均ゲームスコア",
      3000, 500, "5学習seedの平均。各学習時点で探索なし評価10回。共有範囲内で線形補間。",
      "03_evaluation_score.png")
curve("evaluation_curve.csv", "clear_rate",
      "DQN と Double DQN｜探索なし評価のクリア率", "評価10回のクリア率",
      1.0, 0.2, "5学習seedの平均。各学習時点で探索なし評価10回。共有範囲内で線形補間。",
      "04_evaluation_clear_rate.png", True)

def heldout():
    rows = read_rows("heldout_summary.csv")
    data = {r["algorithm"]:r for r in rows}
    im = Image.new("RGB", (W,H), "white")
    d=ImageDraw.Draw(im)
    d.text((80*S, 40*S), "DQN と Double DQN｜最終モデルの未使用評価", font=font(34,True), fill=(32,40,48))
    panels=[(80*S,155*S,760*S,730*S,"平均ゲームスコア","heldout_score_mean","score_ci_low","score_ci_high",3000,False),
            (850*S,155*S,1530*S,730*S,"クリア率","heldout_clear_rate","clear_ci_low","clear_ci_high",1.0,True)]
    for l,t,r,b,title,key,low,high,ymax,pct in panels:
        d.text((l,t-40*S),title,font=font(24,True),fill=(45,52,59))
        plot_l,plot_r=l+70*S,r-30*S
        plot_t,plot_b=t+20*S,b-80*S
        for j in range(6):
            val=j*ymax/5
            yy=plot_b-(val/ymax)*(plot_b-plot_t)
            d.line([(plot_l,yy),(plot_r,yy)],fill=(226,230,235),width=2*S)
            label=f"{val*100:.0f}%" if pct else f"{val:,.0f}"
            d.text((l,yy-11*S),label,font=font(16),fill=(75,82,89))
        for j,alg in enumerate(("dqn","double_dqn")):
            rr=data[alg]
            cx=plot_l+(j+0.5)*(plot_r-plot_l)/2
            bw=130*S
            yv=plot_b-rr[key]/ymax*(plot_b-plot_t)
            d.rectangle([(cx-bw/2,yv),(cx+bw/2,plot_b)],fill=COLORS[alg])
            ylo=plot_b-rr[low]/ymax*(plot_b-plot_t)
            yhi=plot_b-rr[high]/ymax*(plot_b-plot_t)
            d.line([(cx,ylo),(cx,yhi)],fill=(30,35,40),width=3*S)
            d.line([(cx-12*S,ylo),(cx+12*S,ylo)],fill=(30,35,40),width=3*S)
            d.line([(cx-12*S,yhi),(cx+12*S,yhi)],fill=(30,35,40),width=3*S)
            value=f"{rr[key]*100:.1f}%" if pct else f"{rr[key]:,.1f}"
            centered(d,(cx,yhi-25*S),value,font(22,True),(35,42,48))
            centered(d,(cx,plot_b+28*S),LABELS[alg],font(19),(49,56,63))
    d.text((80*S, 815*S),"各方式5学習seed × 未使用の最終評価100回。黒線はseed単位の探索的95% bootstrap区間。",font=font(18),fill=(91,98,105))
    im.resize((1600,900),Image.Resampling.LANCZOS).save(ROOT/"05_heldout_summary.png",optimize=True)
heldout()
print("PLOTS_READY", *(p.name for p in ROOT.glob("*.png")))
