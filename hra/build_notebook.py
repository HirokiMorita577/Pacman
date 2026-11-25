"""
Colab 用ノートブック colab_hra.ipynb を生成する。
hra/ のソースを %%writefile セルとして埋め込むので、Colab 側で GitHub の認証は不要。
ソースを変えたら再生成する:  python hra/build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = ["__init__.py", "config.yaml", "env.py", "agent.py", "plot.py", "report.py", "run.py", "fruit.py", "arcade.py", "diagnose_collision.py", "review.py",
         "behavior.py", "audit_suite.py", "tests/test_hra.py", "tests/test_collision_recorder.py",
         "tests/test_behavior.py", "tests/test_review.py", "tests/test_source_render.py", "DECISIONS.md", "PAPER_GAPS.md"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def build():
    cells = [md("""
# HRA（Hybrid Reward Architecture）Ms. Pac-Man 再現

van Seijen et al., NeurIPS 2017 の HRA を Atari 2600 版 Ms. Pac-Man で再現する。

- 上から順に実行する。学習の成果物（チェックポイント・ログ・学習曲線・動画）は **Google Drive の `HRA_runs/` に定期的に保存**される。
- セッションが切れたら、もう一度上から実行すれば続きから再開する（NFR-01 / RM-21）。
- 論文の価値関数はすべて表形式なので GPU は不要（CPU ランタイムで可）。
- 未確定事項の決め方は `hra/DECISIONS.md`（下で生成）を参照。
"""),
             md("## 準備"),
             code("""
from google.colab import drive
drive.mount('/content/drive', timeout_ms=600000)
DRIVE_DIR = '/content/drive/MyDrive/HRA_runs'   # 成果物の保存先
LOCAL_DIR = '/content/runs'                     # 学習中の作業場所 (Drive へは定期的にコピー)
import os
assert os.path.isdir('/content/drive/MyDrive')  # マウント失敗のまま /content/drive に書き込まない
print('DRIVE_CONNECTED')
os.makedirs(DRIVE_DIR, exist_ok=True)
"""),
             code("!pip install -q gymnasium ale-py pyyaml imageio imageio-ffmpeg"),
             md("## コード（hra/ パッケージ）\nソースを書き出すだけのセル。変更は手元のリポジトリで行い、`python hra/build_notebook.py` で再生成する。"),
             code("import os; os.makedirs('hra/tests', exist_ok=True)")]
    for rel in FILES:
        with open(os.path.join(HERE, rel), encoding="utf-8") as f:
            src = f.read()
        cells.append(code(f"%%writefile hra/{rel}\n{src}", hidden=True))
    cells += [
        md("## 単体テスト（RM-14）\n位置抽出（画面と RAM の照合）、GVF、アグリゲーター、終端状態、探索頭、実行記憶。"),
        code("!python -m pytest hra/tests -q"),
        md("## M1：フルーツ収集課題（RM-13、論文 図3）\n表形式（hra+2, hra+3）は数十秒。ネットワーク版（dqn, hra など）は 5,000 エピソードで時間がかかる。"),
        code("""
# 結果がすでに Drive にあれば飛ばす (再開のたびにやり直さない)
FRUIT_PNG = f'{DRIVE_DIR}/fruit/fruit_hrap2_hrap3.png'
if not os.path.exists(FRUIT_PNG):
    !python -m hra.fruit --methods hra+2 hra+3 --episodes 200 --eval-every 10 --out {DRIVE_DIR}/fruit
from IPython.display import Image, display
display(Image(FRUIT_PNG))
"""),
        code("""
# 任意: 図3 左 (ネットワーク版)。1手法あたり数十分
# !python -m hra.fruit --methods dqn dqn+1 hra hra+1 --episodes 5000 --eval-every 50 --out {DRIVE_DIR}/fruit
"""),
        md("""
## M3〜M4：Ms. Pac-Man の学習

設定は `hra/config.yaml`（NFR-04）。`--set` で上書きでき、設定ごとに別フォルダに保存される（FR-43）。

| 段階 | 設定 |
| --- | --- |
| M3 最小構成 | `--set aggregator.type=linear --set exploration.count_based=false` |
| M4 論文の最良設定（既定） | 正規化アグリゲーター + 多様化頭 + カウントベース頭 |
| M5 実行記憶 | `--set executive_memory.enabled=true`（通常の評価とは分けて報告: FR-36） |

`checkpoint_every`（既定 10 エピソード）ごとに Drive へ保存される。
**タブを閉じたら学習は止まる**が、Drive に保存済みの地点から「すべてのセルを実行」で再開できる。
"""),
        code("""
EXTRA = ''   # 例: '--set aggregator.type=linear --set exploration.count_based=false'
!python -u -m hra.run train --out {LOCAL_DIR} --sync {DRIVE_DIR} {EXTRA} 2>&1 | grep --line-buffered -v -E 'A.L.E|Stella'
"""),
        md("## 学習曲線（NFR-05）\n表示して、ボタンで PNG を保存する（Drive に保存し、ブラウザにもダウンロード）。"),
        code("""
import glob, os, ipywidgets as w
from IPython.display import display
import matplotlib.pyplot as plt
from hra.plot import learning_curve_figure
runs = sorted(glob.glob(f'{DRIVE_DIR}/*/log.csv'))
fig = learning_curve_figure(runs, labels=[os.path.basename(os.path.dirname(p)) for p in runs])
plt.show()
btn = w.Button(description='PNG を保存', icon='download')
out = w.Output()
def _save(_):
    path = f'{DRIVE_DIR}/learning_curve_compare.png'
    fig.savefig(path, dpi=150)
    with out:
        print('保存:', path)
        from google.colab import files; files.download(path)
btn.on_click(_save)
display(btn, out)
"""),
        md("## 評価（FR-40 固定開始）と動画（RM-20）"),
        code("!python -m hra.run eval --out {LOCAL_DIR} --sync {DRIVE_DIR} --mode fixed --episodes 10 --video {EXTRA} 2>&1 | grep --line-buffered -v -E 'A.L.E|Stella'"),
        md("""
## アーケード版風のプレイ動画（見た目だけ）

Atari 版の画面から取り出した位置をもとに、アーケード版 Pac-Man 風（黒背景・青い縁取りの迷路・目のあるゴースト）に描き直す。
`TRIES` 回プレイして一番点数の高い回を、実時間の速さで Drive に保存する。学習・評価には影響しない。
"""),
        code("""
from hra.run import load_config, run_name
RUN = run_name(load_config())            # 設定ごとのフォルダ名 (例: normalized_div_cnt_eq8_ram_cell)
TRIES = 8
!python -m hra.arcade --checkpoint {LOCAL_DIR}/{RUN}/checkpoint.pkl --out {DRIVE_DIR}/{RUN}/arcade_play.mp4 --tries {TRIES} 2>&1 | grep --line-buffered -v -E 'A.L.E|Stella|IMAGEIO'
from IPython.display import Video
Video(f'{DRIVE_DIR}/{RUN}/arcade_play.mp4', embed=True, width=480)
"""),
    ]
    cells += [md("## 接触場面の比較録画（任意）\n原画と再描画を同期して保存する。モデルと設定の一致を確認してから有効にする。学習は行わない。"), code('''
RUN_COLLISION_DIAGNOSTIC = False   # 使うときだけ True (約30,000フレーム録画する)
if RUN_COLLISION_DIAGNOSTIC:
    from datetime import datetime
    from hra.diagnose_collision import diagnose
    from hra.run import load_config, run_name
    diagnostic_run = run_name(load_config())
    diagnostic_out = f'{DRIVE_DIR}/{diagnostic_run}/collision_{datetime.now():%Y%m%d_%H%M%S}'
    diagnose(f'{LOCAL_DIR}/{diagnostic_run}/checkpoint.pkl', 'hra/config.yaml', diagnostic_out,
             max_frames=30000, focus_first_loss=True)
    from IPython.display import Video, display
    display(Video(f'{diagnostic_out}/comparison.mp4', embed=True, width=960))
    print('保存:', diagnostic_out)
''')]
    cells += [md("""
## ゼロからの再学習（6本並列、任意）

RAM+次マス方式 / 画面+固定フレーム方式 × seed 101・202・303、各 5,000 エピソード、α=1。
既存の 11,000 回モデルとは別フォルダ（`HRA_runs/clean_*`）に保存する。
起動後もこのセルは実験が終わるまで実行中のままになり、10分ごとに進み具合を表示する（セルが実行中でないと Colab にアイドルとみなされ、タブを閉じたあと切断されるため）。
もう一度実行しても、動いている実験は二重に起動しない（止まったものだけ Drive から再開する）。
"""), code('''
RUN_CLEAN = False   # 始めるときだけ True
import subprocess, os
from hra.audit_suite import clean_configs
manifest = clean_configs('hra/config.yaml', '/content/clean_configs')
os.makedirs(LOCAL_DIR, exist_ok=True)
for m in manifest:
    if subprocess.run(['pgrep', '-f', m['config']], capture_output=True).returncode == 0:
        print('実行中:', m['name'])
    elif RUN_CLEAN:
        subprocess.run(f"nohup python -u -m hra.run train --config {m['config']} --out {LOCAL_DIR} "
                       f"--sync {DRIVE_DIR} >> {LOCAL_DIR}/{m['name']}.out 2>&1 &", shell=True)
        print('起動:', m['name'])
    else:
        print('未起動:', m['name'])

# 実験が動いている間はこのセルを実行中のままにする。ターミナルや nohup だけで動かすと
# Colab からはアイドルに見え、タブを閉じたあとアイドル切断される (Pro+ の24時間実行はセル実行中が条件)
import time, glob
from IPython.display import clear_output
while subprocess.run(['pgrep', '-f', 'hra.run train --config /content/clean_configs'],
                     capture_output=True).returncode == 0:
    clear_output(wait=True)
    print(time.strftime('%m/%d %H:%M'), '実行中。タブを閉じても続く (止めるときはセルを中断)')
    for p in sorted(glob.glob(f'{DRIVE_DIR}/clean_*/log.csv')):
        rows = open(p).read().strip().splitlines()
        last = rows[-1].split(',') if len(rows) > 1 else ['-', '-', '-']
        print(f"  {os.path.basename(os.path.dirname(p)):36s} ep {last[0]:>5s}  avg100 {last[2]}")
    time.sleep(600)
print('すべての実験が終了しました')
'''), md('''
## 学習状況のレポート（班員向け）
学習曲線・クリア率・得点の内訳・設定表・過去の比較実験を PNG で `HRA_runs/report_日付/` に保存し、
各方式のプレイ動画（アーケード風）も同じ場所に作る。学習中に実行してよい（モデルは読むだけ）。
'''), code('''
import datetime, subprocess
REPORT = f"{DRIVE_DIR}/report_{datetime.date.today():%Y%m%d}"
!python -m hra.report --runs {DRIVE_DIR} --out {REPORT}
MAKE_VIDEO = False   # True でプレイ動画も作る (1本数分)
if MAKE_VIDEO:
    for m in ('ram_cell', 'pixels_frames'):
        n = f'clean_{m}_alpha1_seed101'
        !cp {DRIVE_DIR}/{n}/checkpoint.pkl /content/ck_{m}.pkl
        !python -m hra.arcade --checkpoint /content/ck_{m}.pkl --config /content/clean_configs/{n}.yaml --out {REPORT}/play_{m}.mp4 --tries 3 2>&1 | grep -v -E 'A.L.E|Stella'
from IPython.display import Image, display
for f in ('1_learning_curve', '3_level_reached', '5_settings'):
    display(Image(f'{REPORT}/{f}.png'))
'''), code('''
# 進み具合 (各実験の最新行: エピソード, 得点, 100回平均)
import glob, os
for p in sorted(glob.glob(f'{DRIVE_DIR}/clean_*/log.csv')):
    rows = open(p).read().strip().splitlines()
    last = rows[-1].split(',') if len(rows) > 1 else ['-', '-', '-']
    print(os.path.basename(os.path.dirname(p)), 'ep', last[0], 'score', last[1], 'avg100', last[2])
''')]
    nb = {"cells": cells, "metadata": {"colab": {"provenance": [], "machine_shape": "hm"},  # hm = ハイメモリ (上書きしても設定が戻らないように)
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(ROOT, "colab_hra.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("生成:", path)


if __name__ == "__main__":
    build()
