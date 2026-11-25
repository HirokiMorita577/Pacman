"""Matched evaluation protocols and isolated clean-training configurations."""
import argparse
import csv
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .review import review
from .run import load_config


def suite(checkpoint, config, out_root, episodes=30, seed=730001):
    root = Path(out_root) / datetime.now(timezone.utc).strftime('audit_%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    results = {}
    scores = {}
    # Same seeds across conditions. No policy tuning on this held-out batch.
    for label, counts, noops in [('baseline', 'episode', 30), ('no_noops', 'episode', 0),
                                 ('frozen_counts', 'frozen', 30), ('carry_counts', 'carry', 30)]:
        path = review(checkpoint, config, root / label, episodes=episodes, seed=seed,
                      video=label == 'baseline', count_mode=counts, noop_max=noops, diagnostics=True)
        results[label] = json.loads((Path(path) / 'summary.json').read_text())
        with (Path(path) / 'episodes.csv').open() as f:
            scores[label] = np.array([float(r['score']) for r in csv.DictReader(f)])
        if label != 'baseline':
            delta = scores[label] - scores['baseline']
            boot = np.random.default_rng(4401).choice(delta, (10000, len(delta))).mean(axis=1)
            results[label]['paired_difference_from_baseline'] = {
                'mean': float(delta.mean()), 'bootstrap95': np.quantile(boot, [.025, .975]).tolist()}
        (root / 'comparison.json').write_text(json.dumps(results, indent=2))
    (root / 'COMPLETE.txt').write_text('Four evaluation protocols finished. No model training performed.\n')
    return root


def clean_configs(config, out):
    """Explicit seeds/settings in names; no accidental resume from mixed legacy run."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    base = load_config(config)
    manifest = []
    for method in ('ram_cell', 'pixels_frames'):
        for seed in (101, 202, 303):
            cfg = copy.deepcopy(base)
            cfg['env']['position_source'] = 'ram' if method == 'ram_cell' else 'pixels'
            cfg['env']['step_mode'] = 'cell' if method == 'ram_cell' else 'frames'
            cfg['gvf']['alpha'] = 1.0
            cfg['experiment'].update(seed=seed, episodes=5000)
            identity = f'clean_{method}_alpha1_seed{seed}'
            cfg['experiment']['name'] = identity
            path = root / f'{identity}.yaml'
            text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
            if path.exists():                # 再実行しても同じ内容なら使い回す。違えば上書きせず止める
                if yaml.safe_load(path.read_text(encoding='utf-8')) != cfg:
                    raise ValueError(f'{path} already exists with different settings')
            else:
                path.write_text(text, encoding='utf-8')
            manifest.append({'config': path.as_posix(), 'name': identity, 'seed': seed,
                             'status': 'prepared_not_trained', 'episodes': 5000})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--prepare-clean', action='store_true')
    a = p.parse_args()
    if a.prepare_clean:
        clean_configs(a.config, a.out)
    else:
        if not a.checkpoint:
            p.error('--checkpoint is required for evaluation')
        print(suite(a.checkpoint, a.config, a.out, a.episodes))
