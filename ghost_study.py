"""Four-ghost experiment using the reward setting selected by the one-ghost study."""
import argparse
import json
from pathlib import Path

import numpy as np

import reward_study as study
from improvement_study import write_summary


def jobs_for(args):
    return [dict(exp='ghost_count', label=f'ghosts{args.ghosts}_lives3_gamma0.95_clear0',
                 rewards=study.with_reward(study.DESIGNED_REWARDS, death=100, clear=0),
                 gamma=0.95, num_ghosts=args.ghosts, lives=3, include_lives=True,
                 seed=seed, episodes=args.steps, step_budget=args.steps,
                 eval_episodes=args.eval_episodes, eval_seed0=920000, save_model=True,
                 checkpoint_every=50, compress_checkpoint=True, show_progress=True,
                 runs_dir=str(Path(args.out) / 'runs')) for seed in args.seeds]


def report(out, results):
    rows = write_summary(out / 'evaluation', results)
    study.atomic_json(out / 'results.json', results)
    plt = study._plt()
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for result in results:
        log = study.read_log(out / 'runs' / result['id'])
        axes[0].plot(log['total_steps'], study.mavg(log['score']), label=f"seed {result['seed']}")
        axes[1].plot(log['total_steps'], study.mavg(log['cleared']), label=f"seed {result['seed']}")
    axes[0].set_ylabel('Training score (last 100 episodes)')
    axes[1].set_ylabel('Training clear rate (last 100 episodes)')
    axes[1].set_xlabel('Environment steps')
    axes[1].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    first = results[0]
    fig.suptitle(f"{first['num_ghosts']} ghosts | 3 lives | gamma 0.95 | clear reward 0")
    fig.tight_layout()
    fig.savefig(figures / 'learning.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [str(r['seed']) for r in results]
    axes[0].bar(labels, [r['eval']['score_mean'] for r in results])
    axes[1].bar(labels, [r['eval']['clear_rate'] for r in results])
    axes[0].set_ylabel('Mean evaluation score')
    axes[1].set_ylabel('Evaluation clear rate')
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('Training seed')
    fig.suptitle(f"Greedy evaluation: {len(first['eval']['scores'])} episodes per model, seed 920000+")
    fig.tight_layout()
    fig.savefig(figures / 'evaluation.png', dpi=160)
    plt.close(fig)
    row = rows[0]
    (out / 'README.md').write_text(
        f"# ゴースト{first['num_ghosts']}体の実験\n\n"
        f"残機3・残機観測あり・γ0.95・捕獲−100・クリア報酬0。各{first['total_steps']:,}環境ステップ。\n\n"
        f"学習seed: {labels}。各モデルをε=0、評価seed920000以降の{len(first['eval']['scores'])}回で評価。\n\n"
        f"平均得点 {row['score_mean']:.1f} ± {row['score_std_across_seeds']:.1f}（学習seed間の母標準偏差）。"
        f"クリア率 {row['clear_rate']:.1%}。\n\n"
        "1体の実験で選んだ条件を固定し、4体の結果から再選択しない。1体モデルの重みは移植せず新規学習。\n"
        "入力はゴーストごとの座標・距離を含むため、1体1700次元から4体1709次元に増える。\n"
        "複数ゴーストの同時衝突による最後の残機・捕獲罰の重複を修正。\n"
        "ゴースト捕食の得点機会も増えるため、1体との得点差を性能改善と同一視しない。"
        "クリア率・捕獲終了率・打ち切り率も評価する。\n\n"
        "results.jsonは個別結果、evaluation.csv/jsonは集計、figures/は図、runs/はモデル・ログ・圧縮再開データ。\n",
        encoding='utf-8')
    return rows


def main(argv=None):
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ghosts', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--eval-episodes', type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.steps, args.workers, args.eval_episodes) <= 0:
        parser.error('steps, workers and eval-episodes must be positive')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('training seeds must be unique')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(args)
    plan = dict(jobs=[{**j, 'id': study.job_id(j)} for j in jobs],
                selection='Fixed before training from v2 gamma0.95_clear0; no four-ghost search',
                initialization='from scratch; observation dimension depends on ghost count',
                evaluation_seed0=920000, checkpoint_format='gzip, complete state and RNG',
                comparison_caveat='Historical one-ghost reference; input width and capture bug fix are recorded')
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Existing plan differs. Use a separate output directory.')
    study.atomic_json(plan_path, plan)
    print(f"START ghosts={args.ghosts}, lives=3, steps={args.steps}, seeds={args.seeds}, workers={args.workers}", flush=True)
    results = study.run_jobs(jobs, args.workers)
    report(out, results)
    study.atomic_json(out / 'COMPLETE.json', dict(jobs=len(results), num_ghosts=args.ghosts,
                                               steps_per_seed=args.steps, evaluation_seed0=920000))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
