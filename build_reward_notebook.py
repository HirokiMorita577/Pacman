"""
Colab 用ノートブック colab_reward_study.ipynb を生成する (報酬設計の実験 reward_study.py を Colab で回す)。
ソースを %%writefile セルとして埋め込むので、Colab 側で GitHub の認証は不要。
ソースを変えたら再生成する:  python build_reward_notebook.py
"""
import json
import os

from hra.build_notebook import code, md

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["pacman_env.py", "dqn_agent.py", "pso_optimizer.py", "reward_study.py", "record_plays.py"]


def build():
    cells = [md("""
# 報酬設計の実験（自作パックマン・HRA 版 DQN）

発表用に、報酬を「1回に1つの変数だけ変える」比較と、PSO による報酬の探索を自動で行う。
条件ごとのグラフ・重ね合わせ・Q値の振れ幅・結果表・報酬表・PSO の設定表を **Google Drive の `DQN_reward_study/`** に保存する。

- ランタイムは **CPU（ハイメモリ）** を選ぶ（8並列。1本の学習でバッファに約1.4GB使う）。
- 上から順に実行する。学習のセルは終わるまで実行中のままになる（タブを閉じても Pro+ のバックグラウンド実行で続く）。
- 止まったら、もう一度上から実行すれば終わった学習を飛ばして続きから。
- 全体で 1.5〜2 日かかる見込み。途中でも図表は Drive に更新される。
"""),
             code("""
from google.colab import drive
drive.mount('/content/drive', timeout_ms=600000)
import os
assert os.path.isdir('/content/drive/MyDrive')
OUT = '/content/drive/MyDrive/DQN_reward_study'
os.makedirs(OUT, exist_ok=True)
print('保存先:', OUT, ' CPU:', os.cpu_count())
"""),
             code("""
!pip install -q gymnasium
# 図表の日本語フォント
!apt-get -qq install -y fonts-noto-cjk > /dev/null && rm -rf ~/.cache/matplotlib
"""),
             md("## コード\nソースを書き出すだけのセル。変更は手元のリポジトリで行い、`python build_reward_notebook.py` で再生成する。")]
    for rel in FILES:
        with open(os.path.join(HERE, rel), encoding="utf-8") as f:
            cells.append(code(f"%%writefile {rel}\n{f.read()}", hidden=True))
    cells += [
        md("## 動作確認（1分ほど）\n数エピソードだけで全実験を通し、図表まで作れるか確かめる。結果は `/content/quick` に作る（Drive には保存しない）。"),
        code("!rm -rf /content/quick && python -u reward_study.py --out /content/quick --quick 2>&1 | grep -v -E 'Using device|Model saved' | tail -5"),
        md("""
## 本番（実験0〜5）
実験0 報酬セット → 実験1 捕獲ペナルティ → 実験2 クリアボーナス → 実験3 γ → 実験4 PSO → 実験5 最終比較 → 実験6 残機 (1 vs 3) → 実験7 残機3でのクリアボーナス。終わった学習は飛ばすので、実験5 まで済んでいれば実験6・7 だけを実行する。
前の実験で最も評価スコアが高かった値を、次の実験の基準にする（理由は `README.md` と `chain.json` に残る）。
"""),
        code("""
WORKERS = os.cpu_count()
EPISODES = 1000          # 1本あたりの学習エピソード数 (単変量比較・最終比較)
!python -u reward_study.py --out {OUT} --workers {WORKERS} --episodes {EPISODES} 2>&1 | grep --line-buffered -v -E 'Using device|Model saved'
"""),
        md("## 図表の確認（途中でも実行してよい）"),
        code("""
!python reward_study.py --out {OUT} --report-only > /dev/null 2>&1
import glob
from IPython.display import Image, Markdown, display
display(Markdown(open(f'{OUT}/README.md', encoding='utf-8').read()))
for p in sorted(glob.glob(f'{OUT}/figures/*/overlay.png')) + sorted(glob.glob(f'{OUT}/figures/*/results.png')) \\
         + glob.glob(f'{OUT}/figures/exp4_pso/convergence.png') + glob.glob(f'{OUT}/figures/rewards/reward_table.png'):
    print(p.replace(OUT + '/', ''))
    display(Image(p, width=900))
"""),
    ]
    cells += [md("""
## プレイ動画（発表用）
学習済みモデルで、条件ごとの一番良いプレイと、同じ盤面・同じ seed で各条件を横に並べた動画を
`DQN_reward_study/videos/` に作る（数分）。クリアした条件があれば、その盤面で並べる。
"""), code("""
!pip install -q pygame imageio-ffmpeg
!python record_plays.py --out {OUT} --exps exp5 exp6 exp7 2>&1 | grep -E 'mp4|ありません'
""")]
    nb = {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},  # hm = ハイメモリ (上書きしても設定が戻らないように)
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(HERE, "colab_reward_study.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("生成:", path)


if __name__ == "__main__":
    build()
