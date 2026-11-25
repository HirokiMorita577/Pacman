#@title ゴースト4体：容量復旧・ローカル保存で途中から再開
from pathlib import Path
import subprocess
import sys

OUT_G4 = Path('/content/drive/MyDrive/DQN_ghost4_20261006')
script = Path('/content/colab_ghost4_recovery.py')
script.write_bytes((OUT_G4 / script.name).read_bytes())
print('ゴースト4体 / 残機3 / 各20万step / seed101,202,303', flush=True)
print('学習の保存先: /content/ghost4_recovered_study', flush=True)
print('Driveの空き容量が不足した場合は同期を保留し、学習を継続します。', flush=True)
with Path('/content/ghost4_recovery_run.log').open('a', buffering=1) as log:
    process = subprocess.Popen([sys.executable, '-u', str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    print('復旧PID:', process.pid, flush=True)
    for line in process.stdout:
        log.write(line)
        print(line, end='', flush=True)
    if process.wait():
        raise RuntimeError('復旧ログを確認してください。')
print('完了: /content/000_ghost4_final_20261006.zip をPCへ保存できます。')
