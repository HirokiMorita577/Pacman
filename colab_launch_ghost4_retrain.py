#@title ゴースト4体：改善版と基準版の本番再学習（再開可能）
from google.colab import drive
drive.mount('/content/drive')
import hashlib
import json
from pathlib import Path
import subprocess
import sys

folder = Path('/content/drive/MyDrive/DQN_ghost4_improvement_20261006')
hashes = json.loads((folder / 'source_hashes.json').read_text())
name = 'colab_ghost4_retrain.py'
source = folder / 'source' / name
assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[name], 'Launcher source mismatch'
script = Path('/content/colab_ghost4_retrain.py')
script.write_bytes(source.read_bytes())
log_path = Path('/content/ghost4_retrain_run.log')
print('改善版3本→基準版3本 / 各20万ステップ / 各30回評価', flush=True)
print('Drive保存先:', folder, flush=True)
with log_path.open('a', buffering=1) as log:
    process = subprocess.Popen([sys.executable, '-u', str(script)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    print('再学習PID:', process.pid, flush=True)
    for line in process.stdout:
        log.write(line)
        print(line, end='', flush=True)
    status = process.wait()
assert status == 0, f'再学習が停止しました。{log_path} を確認してください。exit={status}'
print('本番比較が完了しました。/content/ghost4_improvement_study_final.zip をPCへ保存できます。', flush=True)
