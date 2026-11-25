"""Recover the v2 study without changing its training source or manifest."""
from pathlib import Path
import gzip
import hashlib
import os
import shutil
import subprocess
import sys

DRIVE = Path('/content/drive/MyDrive/DQN_reward_study_v2_20261005')
LOCAL = Path('/content/recovered_study')
WORK = Path('/content/codex_v2')

def copy_verified(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(dst) + '.syncing')
    shutil.copy2(src, temporary)
    assert temporary.stat().st_size == src.stat().st_size
    os.replace(temporary, dst)

def sync_small(root):
    for src in root.rglob('*'):
        if src.is_file() and src.suffix in {'.json', '.csv'} or (src.is_file() and src.name == 'model.pt'):
            copy_verified(src, DRIVE / src.relative_to(LOCAL))

if __name__ == '__main__':
    import json
    hashes = json.loads((DRIVE / 'source_hashes.json').read_text())
    for name, expected in hashes.items():
        source = DRIVE / 'source' / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
        copy_verified(source, WORK / name)
    LOCAL.mkdir(exist_ok=True)
    backup = Path('/content/recovery_backup')
    if not (LOCAL / 'runs').exists():
        shutil.copytree(backup if backup.exists() else DRIVE, LOCAL, dirs_exist_ok=True)
    for d in (LOCAL / 'runs').iterdir():
        if (d / 'result.json').exists() and not (d / 'model.pt').exists():
            (d / 'result.json').rename(d / 'result.unsaved_model.json')
        # A zero-byte checkpoint cannot be resumed; retain it in the backup.
        cp = d / 'training.pt'
        if cp.exists() and cp.stat().st_size == 0:
            cp.rename(d / 'training.pt.invalid')
        archive = DRIVE / 'runs' / d.name / 'training.pt.gz'
        if not (d / 'result.json').exists() and archive.exists():
            with gzip.open(archive, 'rb') as src, cp.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_reward_study.py', '-q'], cwd=WORK, check=True)
    sys.path.insert(0, str(WORK))
    os.chdir(WORK)
    import reward_study as study
    import improvement_study
    from dqn_agent import DQNAgent
    original_save = DQNAgent.save_training
    def save_compressed(self, path, progress, manifest):
        original_save(self, path, progress, manifest)
        compressed = Path(str(path) + '.gz')
        with open(path, 'rb') as src, gzip.open(compressed, 'wb', compresslevel=1) as dst:
            shutil.copyfileobj(src, dst)
        copy_verified(compressed, DRIVE / compressed.relative_to(LOCAL))
        print('CHECKPOINT_SYNCED', Path(path).parent.name, progress['total_steps'], flush=True)
    DQNAgent.save_training = save_compressed
    original_run = study.run_job
    def run_and_sync(job):
        result = original_run(job)
        sync_small(Path(job['runs_dir']) / result['id'])
        return result
    study.run_job = run_and_sync
    sys.argv = ['improvement_study.py', '--out', str(LOCAL), '--workers', '4', '--steps', '200000']
    print('RECOVERY_START: cached persisted results only; local training, compressed Drive checkpoints', flush=True)
    improvement_study.main()
    sync_small(LOCAL)
    print('RECOVERY_COMPLETE', DRIVE / 'COMPLETE.json', flush=True)
