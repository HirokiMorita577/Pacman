# 比較実験のColab配置記録（2026-10-20）

ノートブック：https://colab.research.google.com/drive/1Uz0XsNZRS4mSDsCkW1JlM3-Eg5roXD1j

名称：Pacman_comparison_study_20261020.ipynb

旧route_v2用に作成したノートブックを、今回の統合比較に更新。
旧実験のDriveフォルダーと、手元のroute_v2ノートブック・ソースパッケージは保存済み。

新しいDrive保存先：マイドライブ/DQN_Pacman_comparisons_20261020_3c20ffba/

ソース：source/、ハッシュ：source_hashes.json、検証ログ：deployment_tests.log。
比較結果：results/factors/、PSO等：results/search/。
各figures/にPNG、summary.csv等に集計、runs/に学習曲線・モデル・評価を保存。

転送パッケージ：output/colab_comparison_study_20261020/。
48ファイル、bundle SHA-256：3c20ffba064724062181dd97882e20b419168eb75c662d562ca4068836297efa。

ローカル検証：全136テスト合格。短時間の18条件が全完了、失敗0、PNG113枚。
探索の機能確認も基準・ランダム・グリッド・PSOの4方式で完了。
短時間の検証結果は性能を評価するための研究結果ではない。

再接続時は1番のセルで同じ版のコードを展開して検証し、2番を実行。
START_TRAINING=True、WORKERS=2、REPORT_EVERY=1。
全比較の進捗はセル出力、チェックポイント、trial_status.csv等で確認する。
環境ステップの計画上限は約38億。全比較完了と、一時停止・切断を区別する。

Colab側でも136テスト合格（181.79秒）。48ファイルの照合とDrive保存が完了し、DEPLOY_READYを確認。
2番のセルで全比較を開始。学習PID 17462。
PLAN_START conditions=250 trials=1250 environment_steps=252750000 workers=2 を確認。
続く探索・独立再学習を含む総計画は3,800,550,000環境ステップ。
本実験は進行中であり、全条件完了を意味しない。
Colabの検証出力・起動ログ・画面記録は転送パッケージ内の colab_verification.log、
colab_training_start.log、colab_training_started.png に保存。
