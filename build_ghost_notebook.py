"""Package the four-ghost source plus a resumable Colab notebook."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ['pacman_env.py', 'dqn_agent.py', 'reward_study.py', 'pso_optimizer.py',
         'improvement_study.py', 'ghost_study.py', 'tests/test_reward_study.py']


def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=text.splitlines(True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    cells = [cell('markdown', '# ゴースト4体の学習\n残機3・残機入力あり・γ0.95・捕獲−100・クリア報酬0。'
                  '3 seed × 各20万環境ステップ。モデルは新規初期化。学習と評価の両方で4体を使用。\n'
                  'このノートブックは、Driveへ配置済みのハッシュ検証済みソースから再開する。'),
             cell('code', "from google.colab import drive\ndrive.mount('/content/drive')\n"),
             cell('code', (ROOT / 'colab_launch_ghost4.py').read_text(encoding='utf-8')),
             cell('code', "from pathlib import Path\np=Path('/content/drive/MyDrive/DQN_ghost4_20261006')\n"
                  "for name in ['evaluation.csv','COMPLETE.json']:\n    if (p/name).exists():\n        print((p/name).read_text(encoding='utf-8-sig'))\n")]
    notebook = dict(cells=cells, metadata={'colab': {'provenance': [], 'machine_shape': 'hm'},
                    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}}, nbformat=4, nbformat_minor=0)
    notebook_path = ROOT / 'colab_ghost4.ipynb'
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    with zipfile.ZipFile(ROOT / 'ghost4_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, 'source/' + name)
        archive.writestr('source_hashes.json', json.dumps(hashes, indent=1))
        archive.write(ROOT / 'colab_launch_ghost4.py', 'colab_launch_ghost4.py')
        archive.write(ROOT / 'colab_ghost4_recovery.py', 'colab_ghost4_recovery.py')
        archive.write(notebook_path, notebook_path.name)
    print('Created ghost4_bundle.zip and colab_ghost4.ipynb')


if __name__ == '__main__':
    build()
