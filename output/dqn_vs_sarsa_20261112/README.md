# DQN と通常SARSAの比較（発表用）

Google Driveの本学習結果から、DQNと通常SARSAを同じ図に重ねた資料です。PNG 5枚をスライドに貼れます。折れ線図の横軸はすべて**学習エピソード数**です。

## 共通条件と比較範囲

- 自作パックマン、ゴースト4体、ライフ3。環境・報酬設計・ネットワーク規模は同じ。
- 各方式で学習seed 101、202、303、404、505。合計10試行、すべて20万環境ステップで完了。
- 各最終モデルを共通の未使用評価seed 100個で探索なし評価。各方式500プレイ。
- SARSAの各 `result.json` と `evaluation_final.json` について、学習seed・ステップ数・モデルSHA256・評価seedを照合済み。DQNのデータは既に検証済みの `output/dqn_vs_double_dqn_20261111/` から取得。
- 今回の通常SARSAはニューラルネットを使う一方、経験再生なし、ターゲット網なし、バッチ1です。DQNは経験再生・ターゲット網・バッチ128を使います。**本結果はこの実装と設定全体の比較**です。SARSAの更新式だけによる差、またはSARSA一般の性能差とは断定できません。

## 最終評価

| 方式 | 平均ゲームスコア | クリア率 | クリア数 / 500 |
|---|---:|---:|---:|
| DQN | 2,526.2 | 78.8% | 394 |
| 通常SARSA | 687.4 | 0.4% | 2 |

同じ学習seedで対にしたDQN−SARSAの平均差は、スコア+1,838.8点、クリア率+78.4ポイントです。各方式5学習seedなので、不確実性の評価は探索的です。

## 図

1. `01_training_score.png`：探索あり学習中のスコア（直近100エピソード平均）。
2. `02_training_clear_rate.png`：探索あり学習中のクリア率（同上）。
3. `03_evaluation_score.png`：学習途中の探索なし評価スコア。
4. `04_evaluation_clear_rate.png`：学習途中の探索なし評価クリア率。
5. `05_heldout_summary.png`：最終モデルの未使用評価。主結果に適しています。

線と棒は5学習seedの平均。淡色と黒線はseed単位の探索的95% bootstrap区間です。5 seedから復元抽出する3,125組を列挙し、平均の2.5%と97.5%分位を使用しています。学習途中の評価は各チェックポイントで10回、seedごとの共通範囲内を線形補間しています。

同じ20万環境ステップでもSARSAは短いプレイが多く、学習エピソード数がDQNより多くなりました。折れ線の長さを学習量の多さと読み替えないでください。最終評価は両方式とも20万ステップ後のモデルです。

## 数値と元データ

- `heldout_seed_metrics.csv`：10試行の個別成績。
- `heldout_summary.csv`：方式別平均とbootstrap区間。
- `paired_difference.csv`：対応する学習seedで比較した差。
- `training_curve_100ep.csv`、`evaluation_curve.csv`：図の元データ。
- `source_links.csv`：各試行のGoogle Driveフォルダーと元ファイル。
- `prepare.py`、`make_graphs.py`：表とPNGの再生成スクリプト。

元の学習結果：Google Drive `MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec/results/factors/runs/`。SARSAの検証済みローカル原本は `C:/PRO/Pacman/slide_work/sarsa_update_20261111/source/` にあります。
