"""Compare four-ghost observation/action improvements at equal training budgets.

Run on local storage (PC or /content), not directly in a mounted Drive folder.
All variants include the per-ghost proximity reward bug fix. Historical results
are a reference only; the freshly trained baseline isolates the new features.
"""
import argparse
import csv
import json
from pathlib import Path

import reward_study as study

VARIANTS = {
    'baseline': ('legacy', False),
    'observation': ('tactical', False),
    'mask': ('legacy', True),
    'improved': ('tactical', True),
}


def jobs_for(args):
    return [dict(
        exp='ghost4_improvement', label=variant, num_ghosts=4, lives=3, include_lives=True,
        observation_version=VARIANTS[variant][0], mask_illegal_actions=VARIANTS[variant][1],
        rewards=study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0), gamma=0.95,
        seed=seed, episodes=args.steps, step_budget=args.steps,
        eval_episodes=args.eval_episodes, eval_seed0=930000,
        checkpoint_every=50, compress_checkpoint=True, save_model=True, show_progress=True,
        runs_dir=str(Path(args.out) / 'runs'))
        for variant in args.variants for seed in args.seeds]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['baseline', 'improved'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--eval-episodes', type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.steps, args.workers, args.eval_episodes) <= 0:
        parser.error('steps, workers and eval-episodes must be positive')
    if len(args.seeds) != len(set(args.seeds)) or len(args.variants) != len(set(args.variants)):
        parser.error('seeds and variants must be unique')
    out = Path(args.out).resolve()
    if out.is_relative_to(Path('/content/drive').resolve()):
        parser.error('Train in /content or PC storage; copy final artifacts to Drive after completion')
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(args)
    plan = dict(jobs=[{**j, 'id': study.job_id(j)} for j in jobs],
                evaluation_seed0=930000, primary_metric='clear_rate',
                secondary_metrics=['pellet_fraction_mean', 'lives_lost_mean', 'blocked_action_rate', 'score_mean'],
                caveat='All variants use corrected per-ghost proximity rewards; historical results are not a matched control')
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Existing plan differs; use a separate output directory')
    study.atomic_json(plan_path, plan)
    results = study.run_jobs(jobs, args.workers)
    study.atomic_json(out / 'results.json', results)
    rows = []
    for variant in args.variants:
        group = [r for r in results if r['label'] == variant]
        rows.append(dict(variant=variant, seeds=len(group), training_steps_per_seed=args.steps,
                         evaluation_episodes=len(group) * args.eval_episodes,
                         **{k: sum(r['eval'][k] for r in group) / len(group)
                            for k in ['clear_rate', *plan['secondary_metrics']]}))
    study.atomic_json(out / 'evaluation.json', rows)
    with (out / 'evaluation.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    study.atomic_json(out / 'COMPLETE.json', dict(jobs=len(results), steps_per_seed=args.steps,
                                                evaluation_seed0=930000))
    print(json.dumps(rows, indent=2), flush=True)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
