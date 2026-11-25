"""Resume frozen four-ghost training locally, with bounded Drive checkpoint slots."""
from pathlib import Path
import argparse
import gc
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

DRIVE = Path('/content/drive/MyDrive/DQN_ghost4_20261006')
LOCAL = Path('/content/ghost4_recovered_study')
WORK = Path('/content/codex_ghost4')
BACKUP = Path('/content/ghost4_recovery_backup_20261006')
HEADROOM = 256 * 1024 * 1024
SYNC_STEPS = 30000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def drive_free():
    s = os.statvfs(DRIVE)
    return s.f_bavail * s.f_frsize


def upload(src, dst):
    """Never rename files on Drive; local data remains authoritative on failure."""
    src, dst = Path(src), Path(dst)
    try:
        if drive_free() < HEADROOM + src.stat().st_size:
            print('SYNC_DEFERRED_LOW_QUOTA', str(src.relative_to(LOCAL)), flush=True)
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and digest(src) == digest(dst):
            return True
        with src.open('rb') as incoming, dst.open('wb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        if digest(src) != digest(dst):
            raise OSError('Uploaded checksum mismatch')
        return True
    except (OSError, ValueError) as exc:
        print('SYNC_DEFERRED', str(src), str(exc), flush=True)
        return False


def sync_checkpoint(path, steps):
    path = Path(path)
    run = path.parent.name
    state_file = path.parent / 'recovery_sync.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if steps != 200000 and steps - state.get('steps', 0) < SYNC_STEPS:
        return False
    slot = 'B' if state.get('slot') == 'A' else 'A'
    target = DRIVE / 'runs' / run / ('recovery_checkpoint_' + slot + '.pt.gz')
    if not upload(path, target):
        return False
    record = dict(slot=slot, steps=steps, sha256=digest(path), file=target.name)
    state_file.write_text(json.dumps(record))
    upload(state_file, target.parent / state_file.name)
    print('CHECKPOINT_SYNCED', run, steps, slot, flush=True)
    return True


def sync_outputs(root):
    success = True
    for src in sorted(Path(root).rglob('*')):
        if not src.is_file() or src.name == 'COMPLETE.json':
            continue
        if src.suffix in {'.json', '.csv', '.png', '.md'} or src.name == 'model.pt':
            if src.name not in {'plan.json', 'plan.from_drive.json', 'recovery_sync.json'}:
                success = upload(src, DRIVE / src.relative_to(LOCAL)) and success
    return success


def prepare():
    import torch
    hashes = json.loads((DRIVE / 'source_hashes.json').read_text())
    for name, expected in hashes.items():
        src = DRIVE / 'source' / name
        if digest(src) != expected:
            raise ValueError('Frozen source mismatch: ' + name)
        dst = WORK / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if not LOCAL.exists():
        shutil.copytree(BACKUP if BACKUP.exists() else DRIVE, LOCAL)
        if (LOCAL / 'plan.json').exists():
            (LOCAL / 'plan.json').rename(LOCAL / 'plan.from_drive.json')
    sys.path.insert(0, str(WORK))
    os.chdir(WORK)
    import ghost_study
    import reward_study as study
    args = argparse.Namespace(out=str(LOCAL), ghosts=4, steps=200000,
                              seeds=[101, 202, 303], eval_episodes=30)
    for job in ghost_study.jobs_for(args):
        run = study.job_id(job)
        local_cp = LOCAL / 'runs' / run / 'training.pt.gz'
        candidates = [local_cp, DRIVE / 'runs' / run / 'training.pt.gz']
        candidates += list((DRIVE / 'runs' / run).glob('recovery_checkpoint_*.pt.gz'))
        best = None
        manifest = study.experiment_manifest(job)
        for path in candidates:
            if not path.exists():
                continue
            try:
                # torch.load seeks inside its ZIP container; decompress once
                # locally to avoid repeated full gzip scans during those seeks.
                with tempfile.TemporaryFile() as raw:
                    with gzip.open(path, 'rb') as f:
                        shutil.copyfileobj(f, raw, 8 * 1024 * 1024)
                    raw.seek(0)
                    checkpoint = torch.load(raw, map_location='cpu', weights_only=False)
                if checkpoint['manifest'] != manifest:
                    raise ValueError('Checkpoint manifest mismatch')
                steps = checkpoint['progress']['total_steps']
                if best is None or steps > best[0]:
                    best = (steps, path)
                del checkpoint
                gc.collect()
            except (OSError, EOFError, RuntimeError, ValueError) as exc:
                print('INVALID_CHECKPOINT', path, str(exc), flush=True)
        if best is None:
            raise ValueError('No valid checkpoint for existing run: ' + run)
        if best[1] != local_cp:
            local_cp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best[1], local_cp)
        print('RESUME_VERIFIED', job['seed'], best[0], run, flush=True)
        state = local_cp.parent / 'recovery_sync.json'
        if not state.exists():
            state.write_text(json.dumps(dict(steps=best[0])))
    return ghost_study, study


def save_local(self, path, progress, manifest):
    if not Path(path).resolve().is_relative_to(LOCAL.resolve()):
        raise ValueError('Checkpoint must be saved in local runtime storage')
    original_save(self, path, progress, manifest)
    print('LOCAL_CHECKPOINT', manifest['job']['seed'], progress['total_steps'], flush=True)
    sync_checkpoint(path, progress['total_steps'])


def load_local(self, path, manifest):
    with tempfile.TemporaryDirectory(dir=str(LOCAL)) as tmp:
        raw = Path(tmp) / 'training.pt'
        with gzip.open(path, 'rb') as src, raw.open('wb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        progress = original_load(self, raw, manifest)
    print('RESUMED', manifest['job']['seed'], progress['total_steps'], flush=True)
    return progress


def run_and_sync(job):
    result = original_run(job)
    sync_outputs(LOCAL / 'runs' / result['id'])
    return result


def main():
    import fcntl
    lock = open('/content/ghost4_recovery.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ghost_study, study = prepare()
    from dqn_agent import DQNAgent
    global original_save, original_load, original_run
    original_save = DQNAgent.save_training
    original_load = DQNAgent.load_training
    original_run = study.run_job

    DQNAgent.save_training = save_local
    DQNAgent.load_training = load_local
    study.run_job = run_and_sync
    print('RECOVERY_START local training; Drive checkpoint slots; low quota defers sync', flush=True)
    ghost_study.main(['--out', str(LOCAL), '--ghosts', '4', '--workers', '3', '--steps', '200000'])
    synced = sync_outputs(LOCAL)
    if synced:
        synced = upload(LOCAL / 'COMPLETE.json', DRIVE / 'COMPLETE.json')
    archive = Path('/content/000_ghost4_final_20261006.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for src in sorted(LOCAL.rglob('*')):
            if src.is_file() and not src.name.endswith(('.lock', '.tmp')):
                z.write(src, src.relative_to(LOCAL))
    print('LOCAL_COMPLETE', archive, digest(archive), 'Drive synced:', synced, flush=True)
    if not synced:
        print('DOWNLOAD_REQUIRED: final archive remains in Colab local storage', flush=True)


if __name__ == '__main__':
    main()
