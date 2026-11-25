#@title 2. 全比較を開始・再開（保存時のPythonを復元）
START_TRAINING = True #@param {type:"boolean"}
WORKERS = 8 #@param {type:"integer"}
REPORT_EVERY = 25 #@param {type:"integer"}
if START_TRAINING:
    import hashlib, json, os, platform, shutil, site, subprocess, sys
    from pathlib import Path
    import psutil
    WORK = Path('/content/pacman_comparison_gpu_code')
    LOCAL = Path('/content/pacman_comparison_gpu_study')
    FOLDER = Path('/content/drive/MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PACMAN_DEVICE'] = 'cuda'
    os.environ['PACMAN_TORCH_THREADS'] = '1'
    for task in psutil.process_iter(['pid', 'cmdline']):
        try:
            if any(Path(a).name == 'comparison_study.py' for a in task.info['cmdline'] or []):
                raise RuntimeError(f"An existing comparison process is running: {task.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    meta = json.loads((FOLDER/'deployment.json').read_text())
    assert meta['tests_passed'] and meta['bundle_sha256'] == '9e75afec6a09a84cf5199b2a1bc1d612747d9ad7369f7a50907222189759b0fd'
    frozen = json.loads((FOLDER/'source_hashes.json').read_text())
    for name, expected in frozen.items():
        assert hashlib.sha256((WORK/name).read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256((FOLDER/'source'/name).read_bytes()).hexdigest() == expected, name
    saved_plan = FOLDER/'results/factors/plan.json'
    expected_python = json.loads(saved_plan.read_text())['runtime']['python']
    python_exec = sys.executable
    process_env = os.environ.copy()
    if platform.python_version() != expected_python:
        assert platform.python_version().split('.')[:2] == expected_python.split('.')[:2], 'Restore the original minor version and its packages first'
        managed = Path('/content/pacman_comparison_python')
        subprocess.run(['uv', 'python', 'install', expected_python, '--install-dir', str(managed)], check=True)
        installed = list(managed.glob('cpython-'+expected_python+'-*/bin/python'+'.'.join(expected_python.split('.')[:2])))
        assert len(installed) == 1, installed
        python_exec = str(installed[0])
        process_env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(site.getsitepackages() + [process_env.get('PYTHONPATH', '')]))
    preflight = '''
import json, platform
from pathlib import Path
from experiments.config import manifest
import torch
p = json.loads(Path('/content/drive/MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec/results/factors/plan.json').read_text())
m = manifest(p['plan']['conditions'][0]['config'], p['plan']['seeds'][0])
actual = {k:m[k] for k in p['runtime']}
assert actual == p['runtime'], [k for k in actual if actual[k] != p['runtime'][k]]
assert torch.cuda.is_available()
print('RESUME_RUNTIME_VERIFIED:', platform.python_version(), torch.cuda.get_device_name(0), flush=True)
'''
    subprocess.run([python_exec, '-u', '-c', preflight], cwd=WORK, env=process_env, check=True)
    available = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES') / 1024**3
    WORKERS = min(WORKERS, max(1, (os.cpu_count() or 2)-2))
    assert 1 <= WORKERS <= 8 and available >= WORKERS*3+4
    assert REPORT_EVERY >= 1
    command = [python_exec, '-u', 'comparison_study.py', 'all', '--out', str(LOCAL),
               '--backup', str(FOLDER/'results'), '--workers', str(WORKERS),
               '--report-every', str(REPORT_EVERY)]
    print('RESUMING_COMPARISONS: workers', WORKERS, 'report_every', REPORT_EVERY, flush=True)
    print('保存先:', FOLDER/'results', flush=True)
    with (WORK/'full_study.log').open('a', buffering=1) as log:
        process = subprocess.Popen(command, cwd=WORK, env=process_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        print('COMPARISON_PID:', process.pid, flush=True)
        for line in process.stdout:
            log.write(line)
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'exit={code}: inspect {WORK}/full_study.log')
    print('ALL_COMPARISONS_FINISHED: 各COMPLETE.jsonのfailed_trialsも確認してください。', flush=True)
else:
    print('未開始。チェックして実行すると開始・再開します。')
