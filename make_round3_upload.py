"""
第3回比較 (ghost4_round3) を Colab で回すための Drive アップロード用フォルダを作る。

  python make_round3_upload.py --out output/ghost4_round3_upload

出力:
  <out>/DQN_ghost4_round3/source/...            固定するソース (テスト一式を含む。テストの除外はしない)
  <out>/DQN_ghost4_round3/source_hashes.json
  <out>/DQN_ghost4_round3/colab_launch_round3.py 起動ファイル (版とハッシュを launcher.json に記録)
  <out>/DQN_ghost4_round3/launcher.json
  <out>/colab_cell.txt                           Colab のセルに貼る2行 (起動ファイルのハッシュを照合してから実行する)
フォルダごと Drive の MyDrive/ に置く。起動ファイルをセル内で書き換えることはしない。直すときはこのスクリプトで版を上げて作り直す。
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRIVE_DIR = "DQN_ghost4_round3"
LAUNCHER_VERSION = 2
CELL_NAME = "colab_launch_round3.py"   # 起動ファイルの名前 (make_round4_upload が上書きする)
ROUND_LABEL = "第3回比較"
SOURCES = ["pacman_env.py", "dqn_agent.py", "reward_study.py", "pso_optimizer.py", "improvement_study.py",
           "ghost_study.py", "ghost4_improvement.py", "ghost4_round2.py", "ghost4_round3.py",
           "colab_ghost4_recovery.py", "colab_ghost4_retrain.py", "colab_round2.py", "record_plays.py",
           "record_clear_plays.py", "ghost4_diagnose.py", "ghost4_counterfactual.py", "ghost4_failures.py",
           "ghost4_round2_report.py", "ghost4_eval_ablation.py"]

LAUNCHER = '''#@title ゴースト4体：第3回比較 (launcher v{version}、再開可能)
# このセルは学習が終わるまで実行中のままになる。止まったら同じセルを実行し直すと続きから。
from google.colab import drive
drive.mount('/content/drive')
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

folder = Path('/content/drive/MyDrive/{drive_dir}')
work = Path('/content/ghost4_round3_code')
hashes = json.loads((folder / 'source_hashes.json').read_text())
for name, expected in hashes.items():
    src = folder / 'source' / name
    assert hashlib.sha256(src.read_bytes()).hexdigest() == expected, 'Source mismatch: ' + name
    (work / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, work / name)
print('ソース照合 OK:', len(hashes), 'files', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gymnasium', 'pytest'], check=True)
test = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider'], cwd=work,
                      capture_output=True, text=True)
print(test.stdout[-1500:], flush=True)
assert test.returncode == 0, 'テストが不合格のため学習しません'
log_path = Path('/content/ghost4_round3_run.log')
print('第3回比較: 9条件 x 3 seed → 開発用評価 → 選択 → 学習seed 5本で未使用seed 950000〜 の最終評価', flush=True)
with log_path.open('a', buffering=1) as log:
    process = subprocess.Popen([sys.executable, '-u', str(work / 'colab_round2.py'), '--module', 'ghost4_round3',
                                '--drive', str(folder), '--local', '/content/ghost4_round3_study',
                                '--work', str(work), '--workers', '8'],
                               cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    print('学習PID:', process.pid, flush=True)
    for line in process.stdout:
        log.write(line)
        if not line.startswith(('Using device', 'LOCAL_CHECKPOINT')):
            print(line, end='', flush=True)
    status = process.wait()
assert status == 0, f'学習が停止しました。{{log_path}} を確認してください。exit={{status}}'
print('第3回比較が完了しました。結果は Drive の', folder, flush=True)
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out) / DRIVE_DIR
    if out.exists():
        raise SystemExit(f"{out} already exists; use a new --out (do not overwrite an uploaded version)")
    files = SOURCES + sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tests").glob("test_*.py"))
    hashes = {}
    for name in files:
        dst = out / "source" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
        hashes[name] = sha256(dst)
    (out / "source_hashes.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    launcher = out / CELL_NAME
    launcher.write_text(LAUNCHER.format(version=LAUNCHER_VERSION, drive_dir=DRIVE_DIR), encoding="utf-8")
    info = dict(version=LAUNCHER_VERSION, sha256=sha256(launcher), sources=len(hashes))
    (out / "launcher.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    cell = (f"# {ROUND_LABEL} launcher v{LAUNCHER_VERSION} (ハッシュを照合してから実行)\n"
            f"from google.colab import drive; drive.mount('/content/drive'); import hashlib; "
            f"p = '/content/drive/MyDrive/{DRIVE_DIR}/{CELL_NAME}'; b = open(p, 'rb').read(); "
            f"assert hashlib.sha256(b).hexdigest() == '{info['sha256']}', 'launcher mismatch'; exec(b.decode('utf-8'))\n")
    (Path(a.out) / "colab_cell.txt").write_text(cell, encoding="utf-8")
    print(json.dumps(info), "\n" + cell)


if __name__ == "__main__":
    main()
