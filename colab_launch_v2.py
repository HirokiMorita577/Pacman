#@title Codex v2：検証済みコードで改善実験（再開可能）
# このセルだけを実行。旧実験0〜7のセルは再実行しない。
from pathlib import Path
import hashlib, json, shutil, subprocess, sys
OUT_V2 = Path('/content/drive/MyDrive/DQN_reward_study_v2_20261005')
WORK_V2 = Path('/content/codex_v2')
assert (OUT_V2 / 'source_hashes.json').exists(), 'Driveを接続してください'
hashes = json.loads((OUT_V2 / 'source_hashes.json').read_text())
for name, expected in hashes.items():
    source = OUT_V2 / 'source' / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
    dest = WORK_V2 / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gymnasium', 'pytest'], check=True)
subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_reward_study.py', '-q'], cwd=WORK_V2, check=True)
print('24本 / 各20万ステップ / 4並列 / 保存先:', OUT_V2, flush=True)
with (OUT_V2 / 'run.log').open('a', buffering=1) as logfile:
    process = subprocess.Popen([sys.executable, '-u', 'improvement_study.py', '--out', str(OUT_V2), '--workers', '4', '--steps', '200000'], cwd=WORK_V2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    print('PID:', process.pid, flush=True)
    for line in process.stdout:
        logfile.write(line)
        print(line, end='', flush=True)
    status = process.wait()
    if status:
        raise RuntimeError('実験停止。run.logを確認してください。exit=' + str(status))
print('完了:', OUT_V2 / 'COMPLETE.json')
