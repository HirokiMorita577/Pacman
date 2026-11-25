"""Controlled v2 experiments: matched steps, lives input, gamma/clear, held-out evaluation."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import reward_study as study


def jobs_for(args):
    jobs = []
    def add(exp, label, lives, include_lives, gamma=0.95, clear=1560):
        rewards = study.with_reward(study.DESIGNED_REWARDS, death=100, clear=clear)
        for seed in args.seeds:
            jobs.append(dict(exp=exp, label=label, rewards=rewards, gamma=gamma,
                             lives=lives, include_lives=include_lives, seed=seed,
                             episodes=args.steps, step_budget=args.steps, eval_episodes=args.eval_episodes,
                             eval_seed0=910000, save_model=True, checkpoint_every=50,
                             runs_dir=str(Path(args.out) / 'runs')))
    add('foundation', 'lives1_observed', 1, True)
    add('foundation', 'lives3_hidden', 3, False)
    add('foundation', 'lives3_observed', 3, True)
    if args.phase == 'all':
        for gamma in (0.95, 0.99, 0.995):
            for clear in (0, 1560):
                if gamma == 0.95 and clear == 1560:
                    continue  # reference is lives3_observed, reuse identical training
                add('gamma_clear', f'gamma{gamma}_clear{clear}', 3, True, gamma, clear)
    return jobs


def summaries(results):
    rows = []
    for label in sorted({r['label'] for r in results}):
        group = [r for r in results if r['label'] == label]
        scores = np.array([r['eval']['score_mean'] for r in group])
        # Resample training seeds, not correlated evaluation episodes.
        rng = np.random.default_rng(42)
        bootstrap = rng.choice(scores, size=(10000, len(scores)), replace=True).mean(axis=1)
        lo, hi = np.quantile(bootstrap, [0.025, 0.975])
        rows.append(dict(label=label, seeds=len(group), score_mean=float(scores.mean()),
                         score_std_across_seeds=float(scores.std()), ci_low=float(lo), ci_high=float(hi),
                         clear_rate=float(np.mean([r['eval']['clear_rate'] for r in group])),
                         steps_mean=float(np.mean([r['eval']['steps_mean'] for r in group])),
                         total_training_steps=group[0]['total_steps']))
    return rows


def write_summary(path, results):
    rows = summaries(results)
    study.atomic_json(path.with_suffix('.json'), rows)
    with path.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--phase', choices=['foundation', 'all'], default='all')
    args = parser.parse_args()
    if min(args.steps, args.workers, args.eval_episodes) <= 0:
        parser.error('steps/workers/eval-episodes must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(args)
    plan = {'jobs': [{**j, 'id': study.job_id(j)} for j in jobs],
            'validation_seed0': 910000, 'test_seed0': 920000,
            'selection': 'highest validation clear rate, then score; locked before test',
            'termination': 'time limit and budget are external truncations; bootstrap',
            'budget': 'equal environment steps, not episodes',
            'bootstrap': 'training seeds; 3 seeds give a coarse uncertainty estimate'}
    study.atomic_json(out / 'plan.json', plan)
    # Run foundation first so evidence is saved before the longer parameter search.
    foundation = study.run_jobs([j for j in jobs if j['exp'] == 'foundation'], args.workers)
    write_summary(out / 'foundation', foundation)
    remaining = [j for j in jobs if j['exp'] != 'foundation']
    results = foundation + (study.run_jobs(remaining, args.workers) if remaining else [])
    rows = write_summary(out / 'validation', results)
    candidates = [r for r in rows if r['label'] == 'lives3_observed' or r['label'].startswith('gamma')]
    selected = max(candidates, key=lambda r: (r['clear_rate'], r['score_mean']))['label']
    selection = {'selected': selected, 'baseline': 'lives3_observed', 'test_seed0': 920000}
    study.atomic_json(out / 'selection.json', selection)
    from dqn_agent import load_agent
    from pacman_env import PacManEnv
    test_results = []
    for result in results:
        if result['label'] not in {selected, 'lives3_observed'}:
            continue
        run_dir = out / 'runs' / result['id']
        env = PacManEnv(lives=result['lives'], include_lives=result['include_lives'])
        agent = load_agent(str(run_dir / 'model.pt'), env.observation_space.shape[0], 4, device='cpu')
        evaluation = study.evaluate(agent, result['rewards'], args.eval_episodes, result['lives'],
                                    seed0=920000, include_lives=result['include_lives'])
        tested = {**result, 'eval': evaluation}
        test_results.append(tested)
        study.atomic_json(run_dir / 'heldout.json', evaluation)
    write_summary(out / 'heldout', test_results)
    study.atomic_json(out / 'COMPLETE.json', {'jobs': len(results), 'selection': selection})
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
