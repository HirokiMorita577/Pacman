"""Build a self-contained Colab deployment notebook; no network or cloud writes."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import make_round3_upload as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "colab_route_v2_20261020"

DEPLOY = '''#@title 1. 改修コードをデプロイ・テスト・Driveへ保存
import base64, hashlib, importlib.util, io, json, os, platform, shutil, subprocess, sys, zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

EXPECTED_BUNDLE = "__HASH__"
WORK = Path('/content/ghost4_route_v2_code')
FOLDER = Path('/content/drive/MyDrive/DQN_ghost4_route_v2_20261020')
LOCAL = Path('/content/ghost4_route_v2_study')
archive = base64.b64decode("__PAYLOAD__")
assert hashlib.sha256(archive).hexdigest() == EXPECTED_BUNDLE, 'Bundle hash mismatch'
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    frozen = json.loads(z.read('source_hashes.json'))
    assert set(z.namelist()) == set(frozen) | {'source_hashes.json'}
    for name, expected in frozen.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        data = z.read(name)
        assert hashlib.sha256(data).hexdigest() == expected, name
        target = WORK / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
print('SOURCE_VERIFIED:', len(frozen), 'files; bundle:', EXPECTED_BUNDLE, flush=True)
missing = [p for p in ('numpy', 'gymnasium', 'torch', 'pytest') if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
test = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider'],
                      cwd=WORK, capture_output=True, text=True)
test_log = test.stdout + '\\n' + test.stderr
(WORK / 'deployment_tests.log').write_text(test_log, encoding='utf-8')
print(test_log, flush=True)
if test.returncode:
    raise RuntimeError('Deployment tests failed; training has not started')
probe = subprocess.run([sys.executable, '-c',
    "import argparse; import ghost4_route_v2 as r; from pacman_env import PacManEnv; "
    "a=argparse.Namespace(out='/content/ghost4_route_v2_study',steps=200000,seeds=r.SEEDS,variants=list(r.VARIANTS)); "
    "jobs=r.jobs_for(a); assert len(jobs)==10; assert r.FINAL_SEED0==980000; "
    "assert all(j['explore_steps']==80000 for j in jobs); "
    "assert {j['label'] for j in jobs}=={'safe_lookahead','route_v2'}; "
    "dims={j['label']:PacManEnv(num_ghosts=4,lives=3,include_lives=True,observation_version='tactical_safe',**j['env_options']).observation_space.shape[0] for j in jobs}; "
    "assert dims=={'safe_lookahead':1765,'route_v2':1777}; print('PLAN_VERIFIED:',dims,'10 jobs, 5 seeds, final seed 980000')"],
    cwd=WORK, capture_output=True, text=True)
print(probe.stdout, probe.stderr, flush=True)
probe.check_returncode()
from google.colab import drive
drive.mount('/content/drive')
manifest_path = FOLDER / 'source_hashes.json'
if FOLDER.exists() and any(FOLDER.iterdir()) and not manifest_path.exists():
    raise RuntimeError('Destination is nonempty without source manifest; use a new folder')
if manifest_path.exists() and json.loads(manifest_path.read_text()) != frozen:
    raise RuntimeError('Existing deployment source differs; use a new folder')
FOLDER.mkdir(parents=True, exist_ok=True)
for name, expected in frozen.items():
    target = FOLDER / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORK / name, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, name
manifest_path.write_text(json.dumps(frozen, indent=1), encoding='utf-8')
deployment = dict(bundle_sha256=EXPECTED_BUNDLE, source_files=len(frozen), tests_passed=True,
                  python=platform.python_version(), packages={p:version(p) for p in ('torch','numpy','gymnasium','pytest')},
                  deployed_at=datetime.now(timezone.utc).isoformat(), work=str(WORK), drive=str(FOLDER),
                  training_started=False, module='ghost4_route_v2')
(FOLDER / 'deployment.json').write_text(json.dumps(deployment, indent=1), encoding='utf-8')
shutil.copy2(WORK / 'deployment_tests.log', FOLDER / 'deployment_tests.log')
print('DEPLOY_READY: route_v2 / input=1777 / source and tests saved to', FOLDER, flush=True)
print('本学習はまだ開始していません。次のセルで START_TRAINING=True にすると開始します。', flush=True)
'''

TRAIN = '''#@title 2. 本学習（開始する場合はチェック。停止後も同じセルで再開）
START_TRAINING = False #@param {type:"boolean"}
WORKERS = 2 #@param {type:"integer"}
if START_TRAINING:
    import hashlib, json, os, subprocess, sys
    from pathlib import Path
    FOLDER = Path('/content/drive/MyDrive/DQN_ghost4_route_v2_20261020')
    WORK = Path('/content/ghost4_route_v2_code')
    LOCAL = Path('/content/ghost4_route_v2_study')
    meta = json.loads((FOLDER / 'deployment.json').read_text())
    assert meta['tests_passed'] and meta['bundle_sha256'] == '__HASH__', 'Run deployment cell first'
    frozen = json.loads((FOLDER / 'source_hashes.json').read_text())
    for name, expected in frozen.items():
        assert hashlib.sha256((WORK / name).read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256((FOLDER / 'source' / name).read_bytes()).hexdigest() == expected, name
    available_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES') / 1024**3
    if not 1 <= WORKERS <= 10 or available_gb < WORKERS * 2 + 2:
        raise RuntimeError(f'メモリ不足または並列数が範囲外: workers={WORKERS}, available={available_gb:.1f}GiB')
    command = [sys.executable, '-u', str(WORK / 'colab_round2.py'), '--module', 'ghost4_route_v2',
               '--drive', str(FOLDER), '--local', str(LOCAL), '--work', str(WORK),
               '--steps', '200000', '--workers', str(WORKERS), '--seeds', '101', '202', '303', '404', '505']
    print('基準 / route_v2 × 5 seed、各20万step、探索8万step、最終評価seed 980000〜980099', flush=True)
    with Path('/content/ghost4_route_v2_run.log').open('a', buffering=1) as log:
        process = subprocess.Popen(command, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        print('学習PID:', process.pid, flush=True)
        for line in process.stdout:
            log.write(line)
            if not line.startswith(('Using device', 'LOCAL_CHECKPOINT')):
                print(line, end='', flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f'学習停止: exit={status}。/content/ghost4_route_v2_run.log を確認してください')
    print('学習・最終評価終了。Drive_syncedの表示とDrive保存先を確認してください。', flush=True)
else:
    print('本学習は未開始です。開始する場合は START_TRAINING にチェックして、このセルを実行してください。')
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(set(base.SOURCES + ['ghost4_round4.py', 'ghost4_round5.py', 'ghost4_route_v2.py', 'ROUTE_V2.md']
                       + [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'tests').glob('test_*.py')]))
    blobs = {n:(ROOT / n).read_bytes() for n in names}
    hashes = {n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in blobs.items():
            zi = zipfile.ZipInfo(name, (2026,10,20,0,0,0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, data)
        zi = zipfile.ZipInfo('source_hashes.json', (2026,10,20,0,0,0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, json.dumps(hashes, indent=1).encode())
    data = archive.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    deploy = DEPLOY.replace('__HASH__',digest).replace('__PAYLOAD__',base64.b64encode(data).decode())
    train = TRAIN.replace('__HASH__',digest)
    compile(deploy, '<deploy>', 'exec')
    compile(train, '<train>', 'exec')
    cells = [dict(cell_type='markdown', metadata={}, source=[
        '# パックマン：経路特徴 route v2（2026-10-20）\n',
        '1番のセルでコードを展開し、全テスト・学習設定を確認してDriveに保存します。\n',
        '2番は本学習用です。開始する場合だけチェックして実行してください。\n',
        '基準のclassic_lookaheadとroute_v2を同じ報酬・学習量で比較。各5seed、20万step、探索8万step。\n',
        '保存先：マイドライブ/DQN_ghost4_route_v2_20261020/。旧実験とは別フォルダーです。\n',
        '途中停止後は同じノートブックの1番→2番を実行して再開できます。\n'])]
    for i, code in enumerate((deploy, train),1):
        cells.append(dict(cell_type='code',metadata={'id':f'route-v2-{i}'},source=code.splitlines(keepends=True),
                          execution_count=None,outputs=[]))
    notebook = dict(nbformat=4,nbformat_minor=0,metadata={'colab':{'name':'Pacman_route_v2_20261020.ipynb'},
                    'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},cells=cells)
    (OUT / 'Pacman_route_v2_20261020.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    (OUT / 'deployment_cell.py').write_text(deploy,encoding='utf-8')
    (OUT / 'training_cell.py').write_text(train,encoding='utf-8')
    (OUT / 'source_bundle.zip').write_bytes(data)
    (OUT / 'bundle_manifest.json').write_text(json.dumps(dict(bundle_sha256=digest,source_hashes=hashes),indent=1),encoding='utf-8')
    print(json.dumps(dict(notebook=str(OUT / 'Pacman_route_v2_20261020.ipynb'),bundle_sha256=digest,
                         source_files=len(hashes),zip_bytes=len(data)),indent=1))


if __name__ == '__main__':
    main()
