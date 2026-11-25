# route v2 のColab配置（2026-10-20）

専用ノートブック：
https://colab.research.google.com/drive/1Uz0XsNZRS4mSDsCkW1JlM3-Eg5roXD1j

## 確認できたこと

- 改修ソース33ファイルを含むセルと、本学習用セルをノートブックへ配置・保存。
- Colab上でソースハッシュ照合に成功。
- ColabのPythonランタイムで全85テスト成功（116.70秒）。
- 基準の入力1765次元、route v2の入力1777次元を確認。
- 基準 / route_v2 × 学習seed 101 / 202 / 303 / 404 / 505、各20万step、探索8万stepの計10本を設定。
- 報酬は第5回power50と同じ。最終評価は未使用の980000〜980099。
- 本学習は未開始。開始する場合は2番のセルでSTART_TRAININGを有効にする。

## 保存先と再開

コード：`/content/ghost4_route_v2_code`。
学習結果：`/content/ghost4_route_v2_study`。
Drive保存先：`マイドライブ/DQN_ghost4_route_v2_20261020/`。
Drive接続後に、ソース・ハッシュ・テストログ・deployment.jsonを保存する。
学習時は既存のcolab_round2.pyを利用し、圧縮チェックポイントをA/Bの2スロットへ同期する。
再接続・ランタイム再作成後は1番のセルから実行する。

Drive再接続後、`Mounted at /content/drive` と `DEPLOY_READY` を確認済み。
ソース・ハッシュ・テストログ・deployment.jsonのDrive保存が完了した。
直近のColab検証は85件成功（112.48秒）。本学習は未開始。

## ローカル成果物

- `build_route_v2_notebook.py`：セルと自己完結したipynbを生成する。
- `ghost4_route_v2.py`：既存の比較・最終評価の実装に合わせた比較設定。
- `output/colab_route_v2_20261020/`：ノートブック、ソースZIP、マニフェスト、Colab検証ログ。

バンドルSHA256：`1cd7683c7216ea27ba15952fef21af382b6a7eb9f4d9c9fd205926e0d17642c1`。
