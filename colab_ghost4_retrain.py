"""Run the frozen four-ghost comparison locally with bounded Drive backups."""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

DEFAULT_DRIVE = '/content/drive/MyDrive/DQN_ghost4_improvement_20261006'
DEFAULT_LOCAL = '/content/ghost4_improvement_study'
DEFAULT_WORK = '/content/codex_ghost4_retrain'
SYNC_STEPS = 30000


def sync_checkpoint(path, steps, budget):
    path = Path(path)
    state_file = path.parent / 'recovery_sync.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state and steps < budget and steps - state.get('steps', 0) < SYNC_STEPS:
        return False
    slot = 'B' if state.get('slot') == 'A' else 'A'
    target = storage.DRIVE / 'runs' / path.parent.name / f'recovery_checkpoint_{slot}.pt.gz'
    if not storage.upload(path, target):
        return False
    record = dict(slot=slot, steps=steps, sha256=storage.digest(path), file=target.name)
    state_file.write_text(json.dumps(record))
    if not storage.upload(state_file, target.parent / state_file.name):
        return False
    print('CHECKPOINT_SYNCED', path.parent.name, steps, slot, flush=True)
    return True


def load_checkpoint(path):
    import torch
    # torch.load performs seeks; decompress once instead of repeated gzip scans.
    with tempfile.TemporaryFile() as raw:
        with gzip.open(path, 'rb') as incoming:
            shutil.copyfileobj(incoming, raw, 8 * 1024 * 1024)
        raw.seek(0)
        return torch.load(raw, map_location='cpu', weights_only=False)


def restore_job(job, study):
    manifest = study.experiment_manifest(job)
    run = study.job_id(job)
    dest = storage.LOCAL / 'runs' / run
    remote = storage.DRIVE / 'runs' / run
    dest.mkdir(parents=True, exist_ok=True)
    # Completion is committed after every model/result output has been synced.
    receipt = remote / 'saved_result.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        paths = {n: remote / n for n in record['sha256']}
        if all(p.exists() and storage.digest(p) == record['sha256'][n] for n, p in paths.items()):
            result = json.loads(paths['result.json'].read_text())
            if result.get('manifest') != manifest:
                raise ValueError('Saved result manifest mismatch: ' + run)
            for name in ['result.json', 'model.pt', 'log.csv', 'events.json', 'manifest.json', 'progress.json']:
                if (remote / name).exists():
                    shutil.copy2(remote / name, dest / name)
            print('RESTORED_COMPLETE', run, flush=True)
            return
    candidates = [dest / 'training.pt.gz', *remote.glob('recovery_checkpoint_*.pt.gz')]
    best = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            checkpoint = load_checkpoint(path)
            if checkpoint.get('manifest') != manifest:
                raise ValueError('Checkpoint manifest mismatch')
            steps = checkpoint['progress']['total_steps']
            if best is None or steps > best[0]:
                best = (steps, path)
            del checkpoint
            gc.collect()
        except (OSError, EOFError, RuntimeError, ValueError) as exc:
            print('INVALID_CHECKPOINT', path, str(exc), flush=True)
    if best is not None:
        if best[1] != dest / 'training.pt.gz':
            shutil.copy2(best[1], dest / 'training.pt.gz')
        if (remote / 'recovery_sync.json').exists():
            shutil.copy2(remote / 'recovery_sync.json', dest / 'recovery_sync.json')
        print('RESUME_VERIFIED', run, best[0], flush=True)


def save_local(self, path, progress, manifest):
    if not Path(path).resolve().is_relative_to(storage.LOCAL.resolve()):
        raise ValueError('Checkpoint must be written to local storage')
    original_save(self, path, progress, manifest)
    print('LOCAL_CHECKPOINT', manifest['job']['label'], manifest['job']['seed'],
          progress['total_steps'], flush=True)
    sync_checkpoint(path, progress['total_steps'], manifest['job']['step_budget'])


def load_local(self, path, manifest):
    with tempfile.TemporaryDirectory(dir=str(storage.LOCAL)) as tmp:
        raw = Path(tmp) / 'training.pt'
        with gzip.open(path, 'rb') as incoming, raw.open('wb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        progress = original_load(self, raw, manifest)
    print('RESUMED', manifest['job']['label'], manifest['job']['seed'], progress['total_steps'], flush=True)
    return progress


def run_and_sync(job):
    result = original_run(job)
    root = storage.LOCAL / 'runs' / result['id']
    if storage.sync_outputs(root):
        receipt = root / 'saved_result.json'
        receipt.write_text(json.dumps(dict(sha256={n: storage.digest(root / n)
                                                  for n in ['model.pt', 'result.json']})))
        storage.upload(receipt, storage.DRIVE / 'runs' / result['id'] / receipt.name)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive', default=DEFAULT_DRIVE)
    parser.add_argument('--local', default=DEFAULT_LOCAL)
    parser.add_argument('--work', default=DEFAULT_WORK)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303])
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'observation', 'mask', 'improved'],
                        default=['improved', 'baseline'])
    parser.add_argument('--eval-episodes', type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.steps, args.workers, args.eval_episodes) <= 0:
        parser.error('steps/workers/eval-episodes must be positive')
    if len(args.seeds) != len(set(args.seeds)) or len(args.variants) != len(set(args.variants)):
        parser.error('seeds and variants must be unique')
    drive, local, work = map(lambda p: Path(p).resolve(), [args.drive, args.local, args.work])
    if local.is_relative_to(drive) or work.is_relative_to(drive):
        parser.error('local/work must be outside the Drive backup directory')
    local.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (local / 'retrain.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = json.loads((drive / 'source_hashes.json').read_text())
    for name, expected in hashes.items():
        src = drive / 'source' / name
        if hashlib.sha256(src.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source mismatch: ' + name)
        dst = work / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    sys.path.insert(0, str(work))
    os.chdir(work)
    import colab_ghost4_recovery
    global storage, original_save, original_load, original_run
    storage = colab_ghost4_recovery
    storage.LOCAL, storage.DRIVE = local, drive
    import ghost4_improvement as comparison
    import reward_study as study
    from dqn_agent import DQNAgent
    for job in comparison.jobs_for(argparse.Namespace(**{**vars(args), 'out': str(local)})):
        restore_job(job, study)
    original_save, original_load, original_run = DQNAgent.save_training, DQNAgent.load_training, study.run_job
    DQNAgent.save_training, DQNAgent.load_training, study.run_job = save_local, load_local, run_and_sync
    print(f'RETRAIN_START variants={args.variants} seeds={args.seeds} steps={args.steps} '
          f'workers={args.workers} Drive_free={storage.drive_free()}', flush=True)
    comparison.main(['--out', str(local), '--steps', str(args.steps), '--workers', str(args.workers),
                     '--seeds', *map(str, args.seeds), '--variants', *args.variants,
                     '--eval-episodes', str(args.eval_episodes)])
    # Small files are copied directly; no Drive rename/replace or trash growth.
    synced = storage.sync_outputs(local)
    synced = storage.upload(local / 'plan.json', drive / 'plan.json') and synced
    if synced:
        synced = storage.upload(local / 'COMPLETE.json', drive / 'COMPLETE.json')
    archive = local.parent / (local.name + '_final.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for p in sorted(local.rglob('*')):
            if p.is_file() and not p.name.endswith(('.lock', '.tmp')):
                z.write(p, p.relative_to(local))
    print('LOCAL_COMPLETE', archive, storage.digest(archive), 'Drive_synced:', synced, flush=True)


if __name__ == '__main__':
    main()
