#@title 2. 本学習（開始する場合はチェック。停止後も同じセルで再開）
START_TRAINING = False #@param {type:"boolean"}
WORKERS = 2 #@param {type:"integer"}
if START_TRAINING:
    import hashlib, json, os, subprocess, sys
    from pathlib import Path
    FOLDER = Path('/content/drive/MyDrive/DQN_ghost4_route_v2_20261020')
    WORK = Path('/content/ghost4_route_v2_code')
    LOCAL = Path('/content/ghost4_route_v2_study')
    meta = json.loads((FOLDER / 'deployment.json').read_text())
    assert meta['tests_passed'] and meta['bundle_sha256'] == '1cd7683c7216ea27ba15952fef21af382b6a7eb9f4d9c9fd205926e0d17642c1', 'Run deployment cell first'
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
