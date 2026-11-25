"""Repeatable post-training evaluation followed by synchronized recording."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import pickle
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .agent import HRAAgent
from .env import MsPacmanEnv
from .run import load_config, play_episode
from .diagnose_collision import diagnose
from .behavior import BehaviorMetrics


def summarize(rows):
    scores = np.array([r['score'] for r in rows], dtype=float)
    means = np.random.default_rng(1729).choice(scores, (10000, len(scores))).mean(axis=1)
    return {'episodes': len(rows), 'mean': float(scores.mean()),
            'median': float(np.median(scores)), 'std': float(scores.std()),
            'min': float(scores.min()), 'max': float(scores.max()),
            'mean_bootstrap_95_ci': np.quantile(means, [.025, .975]).tolist(),
            'level_reached_rates': {str(k): float(np.mean([r['level_reached'] >= k for r in rows]))
                                    for k in range(2, max(r['level_reached'] for r in rows) + 1)}}


def review(checkpoint, config, out_root, episodes=30, seed=20260927, expected_episode=None,
           video=True, video_max_frames=30000, count_mode='episode', noop_max=None,
           max_frames=108000, diagnostics=False):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    checkpoint = Path(checkpoint)
    cfg = load_config(config)
    if count_mode not in ('episode', 'frozen', 'carry'):
        raise ValueError('Invalid count_mode')
    noop_max = cfg['evaluation']['noop_max'] if noop_max is None else noop_max
    if noop_max < 0:
        raise ValueError('noop_max must be nonnegative')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with checkpoint.open('rb') as f:
        ck = pickle.load(f)
    if expected_episode is not None and ck['episode'] != expected_episode:
        raise ValueError(f"Expected episode {expected_episode}, got {ck['episode']}")
    if 'config' in ck:
        for section in ('env', 'gvf', 'heads', 'aggregator', 'exploration', 'executive_memory'):
            if ck['config'][section] != cfg[section]:
                raise ValueError(f'Checkpoint configuration differs: {section}')
    out = Path(out_root) / datetime.now(timezone.utc).strftime('review_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    config_copy = out / 'effective_config.yaml'
    config_copy.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    metadata = {'checkpoint_sha256': digest, 'trained_episodes': ck['episode'], 'seed': seed,
                'config_provenance': 'embedded_verified' if 'config' in ck else 'explicit_runtime_config_legacy_checkpoint_unverified',
                'protocol': {'noop_max': noop_max, 'count_mode': count_mode,
                             'max_frames': max_frames, 'learning': False,
                             'exploration': 'configured heads retained'},
                'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'gymnasium', 'ale-py')},
                'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'REVIEW_DIR={out}', flush=True)
    rng = np.random.default_rng(seed)
    rows = []
    carried = None
    with (out / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = None
        for i in range(episodes):
            env_seed, agent_seed = seed + 10000 + i, seed + i
            env = MsPacmanEnv(cfg, seed=env_seed)
            try:
                env.load_sites(ck['maze_colors'], ck['pellet_sites'])
                agent = HRAAgent(cfg)
                agent.load_state_dict(ck['agent'])
                if carried is not None:
                    agent.load_state_dict(carried)
                agent.rng = np.random.default_rng(agent_seed)
                agent.update_counts = count_mode != 'frozen'
                noops = int(rng.integers(0, noop_max + 1))
                with ((out / f'trace_{i+1:03d}.jsonl').open('w', encoding='utf-8')
                      if diagnostics else nullcontext()) as trace:
                    extra = {'observer': BehaviorMetrics(trace), 'max_frames': max_frames} if diagnostics else {}
                    row = play_episode(env, agent, learn=False, noops=noops, **extra)
                if count_mode == 'carry':
                    carried = agent.state_dict()
                row.update(index=i + 1, env_seed=env_seed, agent_seed=agent_seed, noops=noops)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(f"EVAL {i+1}/{episodes}: score={row['score']:.0f}, level={row['level_reached']}", flush=True)
            finally:
                env.close()
    summary = summarize(rows)
    summary['protocol'] = metadata['protocol']
    if diagnostics:
        summary['behavior_means'] = {k: float(np.mean([r[k] for r in rows])) for k in
            ('deaths', 'reversals_per_100_moves', 'stationary_frames', 'max_no_reward_frames')}
        summary['truncated_episodes'] = sum(bool(r['truncated']) for r in rows)
    summary.update(trained_episodes=ck['episode'], checkpoint_sha256=digest)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SUMMARY ' + json.dumps(summary), flush=True)
    if video:
        diagnose(str(checkpoint), str(config_copy), str(out / 'synchronized_video'),
                 max_frames=video_max_frames, seed=seed, focus_first_loss=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during review')
    (out / 'COMPLETE.txt').write_text('Evaluation and requested recording completed. Checkpoint unchanged.\n', encoding='utf-8')
    print('COMPLETE ' + str(out), flush=True)
    return str(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--expected-episode', type=int)
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--video-max-frames', type=int, default=30000)
    p.add_argument('--count-mode', choices=['episode', 'frozen', 'carry'], default='episode')
    p.add_argument('--noop-max', type=int)
    p.add_argument('--diagnostics', action='store_true')
    a = p.parse_args()
    review(a.checkpoint, a.config, a.out_root, a.episodes, a.seed, a.expected_episode,
           not a.no_video, a.video_max_frames, a.count_mode, a.noop_max, diagnostics=a.diagnostics)
