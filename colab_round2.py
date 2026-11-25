"""Colab で第2回比較 (ghost4_round2.py) を学習し、Drive へ容量を抑えてバックアップする。再開可能。

colab_ghost4_retrain.py (2026-10-06) と同じ方式:
・学習はColabのローカル (/content/ghost4_round2_study) で行い、途中のcheckpointはローカルの1ファイルを置き換える。
・Driveへは3万ステップごとに固定A/Bスロットへ同期 (ゴミ箱を増やさない)。空き不足・失敗時はローカルで継続。
・再開時は、設定・ソース・依存版が一致して読み込めるcheckpointのうち最大ステップのものから続ける。
・比較が終わったら、開発用評価で1条件を選んで selection.json を保存し、選んだ条件と control を未使用seedで最終評価する。
Driveのソースはハッシュを照合してから使う。
"""
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

DEFAULT_DRIVE = '/content/drive/MyDrive/DQN_ghost4_round2_20261007'
DEFAULT_LOCAL = '/content/ghost4_round2_study'
DEFAULT_WORK = '/content/ghost4_round2_code'
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
    best = None
    for path in [dest / 'training.pt.gz', *remote.glob('recovery_checkpoint_*.pt.gz')]:
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
    print('LOCAL_CHECKPOINT', manifest['job']['label'], manifest['job']['seed'], progress['total_steps'], flush=True)
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
        receipt.write_text(json.dumps(dict(sha256={n: storage.digest(root / n) for n in ['model.pt', 'result.json']})))
        storage.upload(receipt, storage.DRIVE / 'runs' / result['id'] / receipt.name)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive', default=DEFAULT_DRIVE)
    parser.add_argument('--local', default=DEFAULT_LOCAL)
    parser.add_argument('--work', default=DEFAULT_WORK)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303])
    parser.add_argument('--variants', nargs='+', default=None)
    parser.add_argument('--no-final', action='store_true')
    parser.add_argument('--module', default='ghost4_round2', help='比較の入口 (ghost4_round2 / ghost4_round3)')
    args = parser.parse_args(argv)
    drive, local, work = (Path(p).resolve() for p in (args.drive, args.local, args.work))
    if local.is_relative_to(drive) or work.is_relative_to(drive):
        parser.error('local/work must be outside the Drive backup directory')
    local.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (local / 'round2.lock').open('a')
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
    import importlib
    comparison = importlib.import_module(args.module)
    import reward_study as study
    from dqn_agent import DQNAgent
    variants = args.variants or list(comparison.VARIANTS)
    ns = argparse.Namespace(out=str(local), steps=args.steps, seeds=args.seeds, variants=variants)
    jobs = comparison.jobs_for(ns)
    # 第3回の最終評価では学習seedを増やすので、その学習も途中から再開できるようにする
    extra_seeds = [s for s in getattr(comparison, 'FINAL_TRAIN_SEEDS', []) if s not in args.seeds]
    if extra_seeds and not args.no_final:
        jobs += comparison.jobs_for(argparse.Namespace(out=str(local), steps=args.steps, seeds=extra_seeds,
                                                       variants=list(comparison.VARIANTS)))
    for job in jobs:
        restore_job(job, study)
    original_save, original_load, original_run = DQNAgent.save_training, DQNAgent.load_training, study.run_job
    DQNAgent.save_training, DQNAgent.load_training, study.run_job = save_local, load_local, run_and_sync
    print(f'STUDY_START module={args.module} variants={variants} seeds={args.seeds} steps={args.steps} '
          f'workers={args.workers} Drive_free={storage.drive_free()}', flush=True)
    common = ['--out', str(local), '--steps', str(args.steps), '--seeds', *map(str, args.seeds), '--variants', *variants]
    comparison.main(common + ['--workers', str(args.workers)])
    synced = storage.sync_outputs(local)
    synced = storage.upload(local / 'plan.json', drive / 'plan.json') and synced
    if synced:
        synced = storage.upload(local / 'COMPLETE_dev.json', drive / 'COMPLETE_dev.json')
    if not args.no_final:
        comparison.main(common + ['--final'])
        synced = storage.sync_outputs(local) and synced
    archive = local.parent / (local.name + '_final.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for p in sorted(local.rglob('*')):
            if p.is_file() and not p.name.endswith(('.lock', '.tmp')):
                z.write(p, p.relative_to(local))
    print('LOCAL_COMPLETE', archive, storage.digest(archive), 'Drive_synced:', synced, flush=True)


if __name__ == '__main__':
    main()
