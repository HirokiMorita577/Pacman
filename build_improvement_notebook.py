import json
import base64
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ['pacman_env.py', 'dqn_agent.py', 'pso_optimizer.py', 'reward_study.py',
         'record_plays.py', 'improvement_study.py', 'tests/test_reward_study.py', 'README.md']


def cell(kind, source):
    data = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        data.update(execution_count=None, outputs=[])
    return data


def build():
    cells = [cell('markdown', '# パックマン改善実験 v2\n旧実験は保持し、設定とコードを識別して別の保存先で実行する。\nCPUハイメモリ・4並列。途中保存から同じコマンドで再開可能。'),
             cell('code', "from google.colab import drive\ndrive.mount('/content/drive')\n"),
             cell('code', "%pip install -q gymnasium pytest\n"),
             cell('markdown', '## 改修済みコード')]
    for name in FILES:
        encoded = base64.b64encode((ROOT/name).read_bytes()).decode('ascii')
        cells.append(cell('code', f"from pathlib import Path\nimport base64\np = Path('/content/codex_v2/{name}')\np.parent.mkdir(parents=True, exist_ok=True)\np.write_bytes(base64.b64decode({encoded!r}))\n"))
    cells += [cell('code', '%cd /content/codex_v2\n!python -m pytest tests/test_reward_study.py -q\n'),
              cell('markdown', '## 実験\nまずテスト合格を確認する。旧実験0〜7を再実行しない。未使用seedの評価は条件選択後に行う。'),
              cell('code', "import subprocess\nfrom pathlib import Path\nOUT = '/content/drive/MyDrive/DQN_reward_study_v2_20261005'\nPath(OUT).mkdir(parents=True, exist_ok=True)\nsubprocess.run(['python','-m','pytest','tests/test_reward_study.py','-q'], check=True)\nsubprocess.run(['python','-u','improvement_study.py','--out',OUT,'--workers','4','--steps','200000'], check=True)\n"),
              cell('code', "from IPython.display import Markdown, display\nfor name in ['foundation.csv','validation.csv','selection.json','heldout.csv','COMPLETE.json']:\n    p=Path(OUT)/name\n    if p.exists():\n        print(name)\n        print(p.read_text(encoding='utf-8-sig'))\n")]
    notebook = dict(cells=cells, metadata={'colab': {'provenance': [], 'machine_shape': 'hm'},
                                          'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
                    nbformat=4, nbformat_minor=0)
    (ROOT/'codex_improvements.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    with zipfile.ZipFile(ROOT/'codex_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in FILES + ['codex_improvements.ipynb']:
            bundle.write(ROOT/name, name)
    print('Generated codex_improvements.ipynb and codex_bundle.zip')


if __name__ == '__main__':
    build()
