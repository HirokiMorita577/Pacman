"""Build presentation evidence from the 35 verified, completed power-reward runs."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
package = json.loads((DATA / 'power_results.json').read_text(encoding='utf-8'))
runs = package['runs']
weights = [0, 10, 25, 50, 100, 200, 400]
seeds = package['plan']['seeds']
assert len(runs) == 35 and len(seeds) == 5
fonts = {f.name for f in font_manager.fontManager.ttflist}
font = next((x for x in ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP'] if x in fonts), 'DejaVu Sans')
plt.rcParams.update({'font.family': font, 'font.size': 12, 'axes.unicode_minus': False})
colors = ['#0072B2', '#009E73', '#E69F00', '#222222', '#CC79A7', '#D55E00', '#56B4E9']
indices = np.random.default_rng(7301).integers(0, 5, (10000, 5))


def ci(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), *np.quantile(values[indices].mean(axis=1), [.025, .975])


def write_csv(name, rows):
    with (ROOT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def numeric(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [{k: float(v) for k, v in r.items() if v and k != 'ending'} for r in csv.DictReader(f)]


by_weight = {}
seed_rows = []
for weight in weights:
    group = sorted([r for r in runs if r['power_reward'] == weight], key=lambda r: r['train_seed'])
    assert [r['train_seed'] for r in group] == seeds
    for r in group:
        folder = DATA / 'runs' / r['trial']
        raw = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        final = json.loads((folder / 'evaluation_final.json').read_text(encoding='utf-8'))
        assert raw['model_sha256'] == final['model_sha256']
        assert raw['total_steps'] == r['total_steps'] == 200000
        assert len(final['episodes']) == 100 and final['phase'] == 'final'
        assert len({e['evaluation_seed'] for e in final['episodes']}) == 100
        assert abs(np.mean([e['score'] for e in final['episodes']]) - final['score_mean']) < 1e-8
        assert abs(np.mean([e['cleared'] for e in final['episodes']]) - final['clear_rate']) < 1e-8
        r['train'] = numeric(folder / 'train.csv')
        r['curve'] = numeric(folder / 'learning_curve.csv')
        assert r['train'][-1]['total_steps'] == 200000
        assert r['train'][-1]['learning_episode'] == r['learning_episodes']
        seed_rows.append(dict(power_reward=weight, train_seed=r['train_seed'], trial=r['trial'],
                              learning_episodes=r['learning_episodes'], train_steps=200000,
                              score=final['score_mean'], clear_rate=final['clear_rate'],
                              ghosts_eaten=final['ghosts_eaten_mean'], power_eaten=final['power_eaten_mean'],
                              common_return=final['reference_return_mean'], own_return=final['return_mean']))
    by_weight[weight] = group
write_csv('seed_results.csv', seed_rows)

summary = []
paired = []
baseline = {r['train_seed']: r for r in by_weight[50]}
for weight in weights:
    group = by_weight[weight]
    row = dict(power_reward=weight, n_training_seeds=5, evaluation_episodes_per_seed=100,
               train_steps_per_seed=200000, mean_learning_episodes=float(np.mean([r['learning_episodes'] for r in group])))
    for key in ['score_mean', 'clear_rate', 'ghosts_eaten_mean', 'power_eaten_mean', 'reference_return_mean', 'return_mean']:
        m, lo, hi = ci([r['final'][key] for r in group])
        row.update({key: m, key+'_ci_low': float(lo), key+'_ci_high': float(hi)})
    summary.append(row)
    if weight != 50:
        for metric in ['score_mean', 'clear_rate']:
            diffs = [r['final'][metric] - baseline[r['train_seed']]['final'][metric] for r in group]
            m, lo, hi = ci(diffs)
            paired.append(dict(power_reward=weight, baseline_reward=50, metric=metric, n_training_seeds=5,
                               mean_difference=m, ci_low=float(lo), ci_high=float(hi), exploratory=True))
write_csv('final_evaluation_summary.csv', summary)
write_csv('paired_differences_vs_50.csv', paired)


def aligned(group, source, metric, window=1):
    series = []
    for r in group:
        rows = r[source]
        x = np.array([v['learning_episode'] for v in rows])
        y = np.array([v[metric] for v in rows])
        if window > 1:
            total = np.concatenate([[0], np.cumsum(y)])
            y = np.array([(total[i+1]-total[max(0,i+1-window)]) / min(i+1,window) for i in range(len(y))])
        series.append((x,y))
    lo, hi = max(x[0] for x,y in series), min(x[-1] for x,y in series)
    x = np.array(sorted({v for xx,yy in series for v in xx if lo <= v <= hi}))
    values = np.array([np.interp(x,xx,yy) for xx,yy in series])
    mean = values.mean(axis=0)
    # The unit of replication is an independent training seed.
    low, high = np.quantile(values[indices].mean(axis=1), [.025,.975], axis=0)
    return x,mean,low,high


def learning_plot(source, score_key, clear_key, filename, title, note, window):
    fig, axes = plt.subplots(1,2,figsize=(14.4,6.1))
    for weight,color in zip(weights,colors):
        for ax, key, scale in zip(axes,[score_key,clear_key],[1,100]):
            x,m,lo,hi = aligned(by_weight[weight],source,key,window)
            ax.plot(x,m*scale,label=f'報酬 {weight}',color=color,lw=2)
            ax.fill_between(x,lo*scale,hi*scale,color=color,alpha=.10)
    for ax in axes:
        ax.set_xlabel('学習回数（エピソード）')
        ax.grid(alpha=.22)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('ゲームスコア（点）')
    axes[1].set_ylabel('クリア率（%）')
    axes[1].set_ylim(0,102)
    fig.suptitle(title,fontsize=20,y=.98)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=7,frameon=False)
    fig.text(.5,.015,note,ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.83])
    fig.savefig(FIG/filename,dpi=180,bbox_inches='tight')
    plt.close(fig)


learning_plot('train','score','cleared','01_training_score_clear.png',
              'パワーえさ報酬の比較：学習中のスコアとクリア率',
              '7条件×5学習seed、各20万ステップ。50エピソード移動平均、帯は学習seed間の探索的95%区間。探索行動を含む。',50)
learning_plot('curve','score_mean','clear_rate','02_evaluation_score_clear.png',
              'パワーえさ報酬の比較：探索を止めた途中評価',
              '各評価点は10プレイ×5学習seed。横軸は実際の学習エピソード。seed間の共通範囲のみ補間し、外挿しない。',1)

fig,axes=plt.subplots(1,2,figsize=(14.4,6.3))
for row,color in zip(summary,colors):
    for ax,key,scale in zip(axes,['score_mean','clear_rate'],[1,100]):
        m,lo,hi=[row[key+s]*scale for s in ['', '_ci_low','_ci_high']]
        ax.errorbar(row['mean_learning_episodes'],m,yerr=[[m-lo],[hi-m]],fmt='o',color=color,
                    ms=8,capsize=4,label=f"報酬 {row['power_reward']}")
for ax in axes:
    ax.set_xlabel('学習回数（学習終了時の平均エピソード数）')
    ax.grid(alpha=.22)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('最終評価のゲームスコア（点）')
axes[1].set_ylabel('最終評価のクリア率（%）')
axes[1].set_ylim(45,102)
fig.suptitle('パワーえさ報酬の比較：学習に使わない盤面での最終評価',fontsize=19,y=.98)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=7,frameon=False)
fig.text(.5,.015,'各条件5モデル×100プレイ。縦線は学習seed間の探索的95%区間。全条件の学習量は20万ステップ。',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,.83])
fig.savefig(FIG/'03_final_evaluation.png',dpi=180,bbox_inches='tight')
plt.close(fig)

table='\n'.join(f"| {r['power_reward']} | {r['score_mean']:.1f} | {r['clear_rate']*100:.1f}% | {r['ghosts_eaten_mean']:.2f} | {r['power_eaten_mean']:.3f} |" for r in summary)
report=f'''# パワーえさの報酬比較：発表用結果

比較実験は完了。7条件×5学習seed=35モデル、各20万環境ステップ。
最終評価は各モデル100プレイ、合計3,500プレイ。以下は途中評価ではなく、学習・探索に使わない最終評価盤面の集計。

## 比較した条件

パワーえさの学習用報酬だけを0 / 10 / 25 / 50（基準）/ 100 / 200 / 400へ変更した。
ゲーム画面の得点は全条件でパワーえさ1個50点のまま。学習用報酬0でも画面上の得点は加算され、無敵化も有効。

Double DQN、ゴースト4体、残機3、classicモード、戦術的観測、行動マスクあり。
γ=0.95、学習率0.0005、隠れ層512/256/128、batch128、replay100,000、target更新1,000回。
通常えさ報酬10、ゴースト捕食200、クリア報酬0、捕獲ペナルティ−100。他の報酬も全条件で固定。
学習seedは101/202/303/404/505。途中評価は2万環境ステップごとに10プレイ。
最終評価は別の評価seed（1,000,000から）で100プレイずつ、探索行動なし。
学習は回数固定ではなく20万ステップ固定なので、終了時の学習エピソード数は条件・seedによって異なる。

## 最終評価

値は5学習seedの平均。ゴーストとパワーえさは1プレイ当たりの平均捕食数・取得数。

| 学習用パワー報酬 | ゲームスコア | クリア率 | ゴースト捕食数 | パワーえさ取得数 |
|---:|---:|---:|---:|---:|
{table}

## 発表で述べられること

- 報酬を大きくするほど性能が上がる、という単調な関係は見られなかった。
- 平均ゲームスコアが最も高かった観測条件は報酬0（2,584.9点）。平均クリア率が最も高かった観測条件は報酬10（88.4%）。
- 基準の報酬50は2,472.3点・78.4%。報酬400は2,386.8点・70.8%で、過大な報酬による改善は確認できなかった。
- パワーえさ取得数は全条件で約4個。報酬0でも、周辺のえさや無敵化の利益によって取得する行動が学ばれた可能性がある。
- 報酬400は、報酬0よりゴースト捕食数も低い（3.54対4.38）。取得そのものへの報酬を増やしても、取得後の活用まで改善するとは限らないという仮説が考えられる。行動経路の解析では未確認。

5学習seedによる探索的な比較であり、7条件を見た後に最大値を選んでいる。
「報酬0が普遍的な最適値」「報酬400は有意に悪い」とは断定しない。
誤差区間は独立した学習seedを単位とする10,000回のbootstrap 95%区間で、多重比較補正はしていない。
3,500プレイを3,500個の独立した学習実験として扱わない。
報酬値を変えると総報酬の尺度も変わるため、条件ごとの独自報酬の合計だけで優劣を判断しない。

## 前回の発表への追加案

追加1：目的と条件。「パワーえさに高い報酬を与えると、スコアやクリア率は改善するか」を問い、7条件と固定条件を示す。

追加2：学習曲線。02_evaluation_score_clear.pngを貼り、横軸が学習回数であることを説明する。途中評価と最終評価を区別する。

追加3：上の最終評価表と考察。「高報酬ほど良いわけではない。取得後の使い方まで含む報酬設計が必要」と述べる。
発表時間が短ければ、条件・表・考察を2枚へまとめ、学習曲線は補足へ置く。

### 読み上げ例

今回は、パワーえさを食べたときの学習用報酬だけを7通りに変えました。
各条件を5つの乱数seedで20万ステップずつ学習し、別の評価盤面で各100回プレイしました。
平均スコアは報酬0、クリア率は報酬10で最も高く、報酬を400まで増やしても改善しませんでした。
パワーえさ自体はどの条件でもほぼ4個取得しています。このため、取得への報酬の大きさだけでなく、無敵時間をどう活用するかが課題だと考えています。
ただし5つの学習seedによる結果なので、最適値を断定せず、この環境と学習条件で得られた傾向として報告します。

## ファイル

- figures/01_training_score_clear.png：探索を含む学習時のスコアとクリア率（50エピソード移動平均）
- figures/02_evaluation_score_clear.png：探索なしの途中評価のスコアとクリア率
- figures/03_final_evaluation.png：終了時の学習回数に対する最終評価と誤差区間
- final_evaluation_summary.csv：条件ごとの最終評価と区間
- seed_results.csv：35学習seed別の最終評価
- paired_differences_vs_50.csv：同じ学習seedを対応させた基準50との差（探索的）
- data/：Driveから取り出した元データ。学習曲線・最終評価・設定を保持

## 出典

MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec/results/factors/。
2026-10-27に完了済みの35モデルを抽出し、最終評価が保存モデルのSHA-256と一致することを照合。
パワーえさ比較以外の全1,250本やPSO全探索が完了したという意味ではない。
前回の結果は条件が違う場合があるため、今回の表と同一条件の実験として直接つなげない。
'''
(ROOT/'発表用まとめ.md').write_text(report,encoding='utf-8')
(ROOT/'validated_results.json').write_text(json.dumps(dict(summary=summary,paired=paired),ensure_ascii=False,indent=2),encoding='utf-8')
receipt=dict(verified_models=35,final_evaluation_episodes=3500,x_axis='learning episodes',
             source_archive_sha256=hashlib.sha256((ROOT/'source_results.zip').read_bytes()).hexdigest(),
             source_files={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()})
(ROOT/'verification.json').write_text(json.dumps(receipt,indent=1),encoding='utf-8')
print(json.dumps({'validated':35,'final_evaluation_episodes':3500,'font':font,'figures':len(list(FIG.glob('*.png'))),'summary':[(r['power_reward'],r['score_mean'],r['clear_rate']) for r in summary]},ensure_ascii=False))
