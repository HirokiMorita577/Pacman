"""Build a standalone, training-free Colab evaluation and recording notebook."""
import json
from pathlib import Path

from .build_notebook import code, md


def build(destination=None):
    here = Path(__file__).parent
    cells = [md('''# 学習済みモデルの評価と同期動画
学習をやり直さず、Driveのモデルを30回評価し、最初の死亡前後の同期動画を自動作成します。
左は元のAtari映像、右はアーケード風の表示です。結果は日時別のフォルダへ保存します。
途中で止まっても各回の評価CSVは残ります。再実行すると新しい評価を始めます。
'''), code("from google.colab import drive\ndrive.mount('/content/drive')"),
             code('!pip install -q gymnasium ale-py numpy scipy pyyaml pillow imageio imageio-ffmpeg matplotlib'),
             code("from pathlib import Path\nPath('hra').mkdir(exist_ok=True)")]
    for name in ('__init__.py', 'config.yaml', 'env.py', 'agent.py', 'run.py', 'arcade.py',
                 'diagnose_collision.py', 'review.py', 'behavior.py', 'audit_suite.py', 'plot.py'):
        cells.append(code(f'%%writefile hra/{name}\n' + (here / name).read_text(encoding='utf-8'), hidden=True))
    cells += [md('## 評価と同期動画を作成'), code('''
from hra.review import review
RUN = 'normalized_div_cnt_eq8_ram_cell'
ROOT = f'/content/drive/MyDrive/HRA_runs/{RUN}'
RESULT = review(f'{ROOT}/checkpoint.pkl', 'hra/config.yaml', ROOT,
                episodes=30, expected_episode=11000, video=True, diagnostics=True)
'''), code('''
from IPython.display import Video, display
import json
from pathlib import Path
print(json.loads((Path(RESULT) / 'summary.json').read_text()))
display(Video(str(Path(RESULT) / 'synchronized_video/comparison.mp4'), embed=True, width=960))
''')]
    cells += [md('## 評価条件の比較（各30回、学習は行わない）'), code('''
from hra.audit_suite import suite
AUDIT = suite(f'{ROOT}/checkpoint.pkl', 'hra/config.yaml', ROOT, episodes=30)
print(AUDIT)
''')]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                     'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}
    path = Path(destination) if destination else here.parent / 'colab_hra_review.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    build()
